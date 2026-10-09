# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB1 — SFT-mini: mô hình SFT tiếng Việt (checkpoint) làm điểm xuất phát cho DPO
#
# **Công nghệ:** Unsloth + LoRA r=16 trên Qwen3-4B-Instruct-2507 (4-bit) + 1k VN Alpaca, 1 epoch.
#
# > **Mục tiêu:** tạo mô hình SFT để DPO align tiếp. Notebook lưu hai thứ:
# > - `adapters/sft-mini/`: LoRA adapter (nhẹ, để nộp bài);
# > - `models/sft-merged/`: SFT đã gộp vào trọng số 16-bit. **NB3 huấn luyện DPO trên mô hình này**,
# >   nên reference của DPO chính là mô hình SFT (lab cũ dùng nhầm mô hình gốc làm reference).
# >
# > Mô hình gốc đã là bản instruct, nên SFT ở đây chủ yếu kéo phong cách trả lời về
# > tiếng Việt kiểu Alpaca. Loss chỉ tính trên phần trả lời (`train_on_responses_only`).

# %%
import sys
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "lab22" / "config.py").exists())
sys.path.insert(0, str(ROOT))

try:
    import unsloth  # noqa: F401  (must load before trl/transformers)
except ImportError:
    pass
import torch

from lab22 import config as C
from lab22 import modeling as MD

if not torch.cuda.is_available():
    print("WARNING: CUDA unavailable (driver wedged or no GPU). Training on CPU.")
C.ensure_dirs()
print(C.summary())

# %% [markdown]
# ## 1. Nạp mô hình 4-bit + LoRA

# %%
model, tokenizer = MD.load_model(C.BASE_MODEL)
model = MD.add_lora(model)
print(f"pad={tokenizer.pad_token!r} eos={tokenizer.eos_token!r}")
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# %% [markdown]
# ## 2. VN Alpaca → chat text

# %%
from datasets import load_dataset

ds = load_dataset(C.SFT_DATASET, split=f"train[:{C.SFT_SLICE}]")
print(f"Loaded {len(ds)} rows. Columns: {ds.column_names}")


def to_text(row):
    # The translated rows carry stray leading spaces; strip so the template stays clean.
    instruction, extra = row["instruction"].strip(), (row.get("input") or "").strip()
    prompt = instruction + (f"\n\n{extra}" if extra else "")
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": row["output"].strip()},
    ]
    return {"text": MD.chat_text(tokenizer, messages, add_generation_prompt=False)}


ds = ds.filter(lambda r: bool(r.get("instruction")) and bool(r.get("output")))
ds = ds.map(to_text, remove_columns=ds.column_names)
print(ds[0]["text"][:400])

# %% [markdown]
# ## 3. Huấn luyện (loss chỉ trên phần trả lời)

# %%
from trl import SFTConfig, SFTTrainer

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=ds,
    args=SFTConfig(
        output_dir=str(C.ADAPTERS / "sft-checkpoints"),
        dataset_text_field="text",
        max_length=C.MAX_LEN,
        per_device_train_batch_size=C.TIER.sft_batch,
        gradient_accumulation_steps=C.TIER.sft_grad_accum,
        num_train_epochs=1,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        logging_steps=10,
        save_strategy="steps",
        save_steps=20,
        optim=MD.trainer_optim(),
        seed=C.SEED,
        report_to="none",
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        **MD.precision_flags(),
    ),
)
# Qwen ChatML markers; prompt tokens get label -100.
trainer = MD.train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)
result = trainer.train()
print(f"Final SFT loss: {result.training_loss:.4f}")

# %%
import matplotlib.pyplot as plt
import pandas as pd

logs = pd.DataFrame([r for r in trainer.state.log_history if "loss" in r])
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(logs["step"], logs["loss"], color="#2e548a")
ax.set_xlabel("step")
ax.set_ylabel("SFT loss (response tokens)")
ax.set_title(f"SFT-mini · {C.COMPUTE_TIER} · {C.BASE_MODEL.split('/')[-1]}")
ax.grid(True, alpha=0.3)
C.SCREENSHOTS.mkdir(parents=True, exist_ok=True)
fig.savefig(C.SCREENSHOTS / "02-sft-loss.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Lưu adapter + mô hình SFT đã gộp

# %%
model.save_pretrained(str(C.SFT_ADAPTER))
tokenizer.save_pretrained(str(C.SFT_ADAPTER))
MD.save_merged(model, tokenizer, C.SFT_MERGED)
print(f"Saved adapter → {C.SFT_ADAPTER}\nSaved merged 16-bit → {C.SFT_MERGED}")

# %%
sample = MD.generate(model, tokenizer, ["Giải thích ngắn gọn (3-4 câu) thuật toán quicksort hoạt động thế nào."], 200)
print(sample[0])

# %% [markdown]
# ## 5. Ghi chú vibe-coding
#
# > **Cần bao nhiêu SFT để DPO có ý nghĩa?** Thử `SFT_SLICE=100` rồi chạy lại NB1 → NB3.
# > Margin ở NB3 còn tăng không? Câu trả lời còn mạch lạc không? Ghi giả thuyết trước
# > khi chạy, kết quả vào `submission/REFLECTION.md` §6.
#
# **Tiếp theo:** NB2 — dữ liệu sở thích (preference) tiếng Việt.
