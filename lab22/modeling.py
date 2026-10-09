"""GPU-side helpers shared by the notebooks: load, LoRA, generate, DPO, plots.

Import `unsloth` before `trl`/`transformers` in the notebook so Unsloth can
patch them; these helpers import lazily for the same reason.
"""

from __future__ import annotations

import gc
from pathlib import Path

from . import config as C


def _ensure_pad_token(tokenizer, name: str | Path):
    """Qwen3 has a dedicated pad token; smaller Qwen2.5 checkpoints reuse EOS."""
    if tokenizer.pad_token is not None and tokenizer.pad_token_id != tokenizer.eos_token_id:
        return tokenizer
    vocab = tokenizer.get_vocab()
    for candidate in ("<|vision_pad|>", "<|endoftext|>", "<pad>", "[PAD]"):
        if candidate in vocab and vocab[candidate] != tokenizer.eos_token_id:
            tokenizer.pad_token = candidate
            return tokenizer
    if tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
        return tokenizer
    raise ValueError(f"{name}: tokenizer needs a pad token distinct from EOS")


def load_model(name: str | Path, max_len: int = C.MAX_LEN, load_in_4bit: bool | None = None):
    load_in_4bit = C.LOAD_IN_4BIT if load_in_4bit is None else load_in_4bit
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(name),
            max_seq_length=max_len,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
    except ImportError:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        tokenizer = AutoTokenizer.from_pretrained(str(name), trust_remote_code=True)
        kwargs: dict = {"trust_remote_code": True}
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
            if load_in_4bit:
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            else:
                kwargs["dtype"] = torch.float16
        model = AutoModelForCausalLM.from_pretrained(str(name), **kwargs)
        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    tokenizer = _ensure_pad_token(tokenizer, name)
    return model, tokenizer


def add_lora(model, r: int = C.LORA_R, alpha: int = C.LORA_ALPHA):
    try:
        from unsloth import FastLanguageModel

        return FastLanguageModel.get_peft_model(
            model,
            r=r,
            lora_alpha=alpha,
            lora_dropout=0.0,
            bias="none",
            target_modules=C.LORA_TARGETS,
            use_gradient_checkpointing="unsloth",
            random_state=C.SEED,
        )
    except ImportError:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        cfg = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=C.LORA_TARGETS,
        )
        return get_peft_model(model, cfg)


def chat_text(tokenizer, messages: list[dict], add_generation_prompt: bool = True) -> str:
    kwargs = dict(
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **C.CHAT_TEMPLATE_KWARGS,
    )
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def generate(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = C.GEN_MAX_NEW_TOKENS,
    batch_size: int | None = None,
) -> list[str]:
    """Greedy, batched generation for single-turn user prompts."""
    import torch

    try:
        from unsloth import FastLanguageModel

        FastLanguageModel.for_inference(model)
    except ImportError:
        model.eval()
        if hasattr(model, "config"):
            model.config.use_cache = True

    if batch_size is None:
        vram = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        batch_size = 1 if vram < 8 * (1024**3) else 8

    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device
    outputs: list[str] = []
    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            texts = [chat_text(tokenizer, [{"role": "user", "content": p}]) for p in batch]
            # The template already contains the special tokens.
            enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
            new_tokens = out[:, enc["input_ids"].shape[1] :]
            outputs.extend(t.strip() for t in tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = old_side
    return outputs


def train_on_responses_only(trainer, instruction_part: str, response_part: str):
    """Mask prompt tokens so SFT loss is only on the assistant reply."""
    try:
        from unsloth.chat_templates import train_on_responses_only as _unsloth_mask

        return _unsloth_mask(trainer, instruction_part=instruction_part, response_part=response_part)
    except ImportError:
        pass

    if "labels" in getattr(trainer.train_dataset, "column_names", []):
        # TRL already tokenized and built labels; remapping on a 4 GB card
        # just burns RAM and can desync the collator.
        return trainer
    tokenizer = trainer.processing_class
    max_len = getattr(trainer.args, "max_length", None) or C.MAX_LEN
    resp_ids = tokenizer.encode(response_part, add_special_tokens=False)

    def find_sublist(haystack: list[int], needle: list[int]) -> int | None:
        n, m = len(haystack), len(needle)
        if m == 0 or m > n:
            return None
        for i in range(n - m, -1, -1):
            if haystack[i : i + m] == needle:
                return i
        return None

    def tokenize_row(row):
        ids = tokenizer(
            row["text"],
            truncation=True,
            max_length=max_len,
            add_special_tokens=False,
        )["input_ids"]
        labels = list(ids)
        idx = find_sublist(ids, resp_ids)
        if idx is not None:
            cut = idx + len(resp_ids)
            labels = [-100] * cut + labels[cut:]
        return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}

    ds = trainer.train_dataset
    trainer.train_dataset = ds.map(tokenize_row, remove_columns=ds.column_names)
    if hasattr(trainer.args, "dataset_text_field"):
        trainer.args.dataset_text_field = None
    return trainer


def save_merged(model, tokenizer, path: Path):
    """Write a 16-bit merged checkpoint (Unsloth or PEFT)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained_merged"):
        model.save_pretrained_merged(str(path), tokenizer, save_method="merged_16bit")
        return
    from peft import PeftModel

    to_save = model.merge_and_unload() if isinstance(model, PeftModel) else model
    to_save.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))


def cleanup() -> None:
    """Release cached GPU memory. `del` the model/trainer names first."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def precision_flags() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {}
    bf16 = torch.cuda.is_bf16_supported()
    return {"bf16": bf16, "fp16": not bf16}


def trainer_optim() -> str:
    import torch

    return "adamw_8bit" if torch.cuda.is_available() else "adamw_torch"


def dpo_config(output_dir: str | Path, loss_type: list[str] | None = None, **overrides):
    """DPOConfig for LoRA-on-merged-SFT training.

    `precompute_ref_log_probs=True` scores every pair under the starting
    model before the first update. The policy starts as merged SFT plus a
    zero-initialised LoRA, so the reference is exactly the SFT model, with
    no dependence on how adapters are toggled during training.
    """
    from trl import DPOConfig

    params = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=C.TIER.dpo_batch,
        per_device_eval_batch_size=C.TIER.dpo_batch,
        gradient_accumulation_steps=C.TIER.dpo_grad_accum,
        num_train_epochs=C.DPO_EPOCHS,
        learning_rate=C.DPO_LR,
        beta=C.DPO_BETA,
        loss_type=loss_type or C.DPO_LOSS,
        max_length=C.MAX_LEN,
        precompute_ref_log_probs=True,
        warmup_steps=0.1,  # transformers v5: a float < 1 is a ratio; warmup_ratio is gone
        lr_scheduler_type="cosine",
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="no",
        optim=trainer_optim(),
        seed=C.SEED,
        report_to="none",
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        **precision_flags(),
    )
    params.update(overrides)
    return DPOConfig(**params)


def reward_history(log_history: list[dict], prefix: str = ""):
    """Training (prefix="") or eval (prefix="eval_") reward rows as a DataFrame."""
    import pandas as pd

    key = f"{prefix}rewards/chosen"
    rows = [r for r in log_history if key in r]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.rename(columns=lambda c: c[len(prefix):] if prefix and c.startswith(prefix) else c)


def diagnose(df, window: int = 3) -> tuple[str, str]:
    """Classify the reward trajectory. Needs rewards/chosen and rewards/rejected.

    Implicit rewards start at 0 because the policy starts equal to the
    reference, so the sign of the final chosen reward is the direction it moved.
    """
    if df.empty or len(df) < 2:
        return "UNKNOWN", "Not enough logged steps to diagnose."
    end = df.tail(window)
    chosen = float(end["rewards/chosen"].mean())
    rejected = float(end["rewards/rejected"].mean())
    margin = chosen - rejected
    if margin <= 0:
        return "FAILURE", (
            f"Margin {margin:+.3f} <= 0: the model does not prefer chosen. Check data "
            "orientation, lr, and that the reference is the SFT model."
        )
    if chosen < 0:
        return "LIKELIHOOD DISPLACEMENT", (
            f"Margin {margin:+.3f} > 0 but chosen reward {chosen:+.3f} < 0: the gap grew "
            "because rejected fell faster. Compare with RPO in NB3b."
        )
    if chosen > 0:
        return "INTENDED", f"Chosen {chosen:+.3f} up, rejected {rejected:+.3f}, margin {margin:+.3f}."
    return "AMBIGUOUS", f"Margin {margin:+.3f} with flat chosen reward; train longer or raise lr."


def plot_rewards(train_df, eval_df, title: str, path: Path | None = None):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for df, style, tag in ((train_df, "-", "train"), (eval_df, "o--", "held-out")):
        if df is None or df.empty:
            continue
        axes[0].plot(df["step"], df["rewards/chosen"], style, color="#2e548a", label=f"chosen ({tag})")
        axes[0].plot(df["step"], df["rewards/rejected"], style, color="#c83538", label=f"rejected ({tag})")
        margin = df["rewards/chosen"] - df["rewards/rejected"]
        axes[1].plot(df["step"], margin, style, color="#1a3355", label=f"margin ({tag})")
    for ax, ylabel in zip(axes, ("implicit reward  β·log(π/π_ref)", "margin (chosen − rejected)")):
        ax.axhline(0, color="#888", linestyle=":", linewidth=0.7)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig
