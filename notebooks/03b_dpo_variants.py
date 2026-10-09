# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB3b — So sánh các biến thể: DPO · RPO · DPO chuẩn hoá độ dài · LD-DPO · ORPO (TUỲ CHỌN, +8)
#
# Cùng dữ liệu (`VARIANT_TRAIN` cặp đầu của NB2), cùng số bước, cùng LoRA. Chỉ đổi loss.
#
# | Run | Cấu hình TRL | Ý tưởng |
# |---|---|---|
# | `dpo` | `loss_type=["sigmoid"]` | mức cơ sở (baseline) |
# | `rpo` | `loss_type=["sigmoid","sft"]` | thêm NLL trên chosen, chống likelihood displacement |
# | `dpo_norm` | `loss_type=["sigmoid_norm"]` | log-prob trung bình theo token (gần SimPO nhưng vẫn có reference) |
# | `ld_dpo` | `ld_alpha=0.5` | giảm trọng số phần token vượt quá độ dài chung (LD-DPO) |
# | `orpo` | `trl.experimental.orpo` | không reference, SFT + odds-ratio trong một bước |
#
# ORPO thường xuất phát từ mô hình *chưa* SFT; ở đây nó chạy trên cùng `models/sft-merged`
# để bảng so sánh chỉ khác nhau ở loss. Đặt `ORPO_FROM_BASE=1` để thử ORPO từ base.
#
# **Đọc kết quả:** độ chính xác reward trên held-out và độ dài đầu ra trung bình. Các
# biến thể có thang reward khác nhau, nên **không so trực tiếp giá trị margin** giữa các dòng.

# %%
import os
import sys
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "lab22" / "config.py").exists())
sys.path.insert(0, str(ROOT))

try:
    import unsloth  # noqa: F401
except ImportError:
    pass
import torch

from lab22 import config as C
from lab22 import modeling as MD

assert torch.cuda.is_available()
assert C.SFT_MERGED.exists() and (C.PREF_DIR / "train.parquet").exists(), "Run NB1 + NB2 first"

from datasets import Dataset

train_ds = Dataset.from_parquet(str(C.PREF_DIR / "train.parquet"))
train_ds = train_ds.select(range(min(C.TIER.variant_train, len(train_ds))))
assert len(train_ds) > 0, "Empty preference train split: rerun NB2"
eval_ds = Dataset.from_parquet(str(C.PREF_DIR / "eval.parquet"))
probe_prompts = [r["prompt"][0]["content"] for r in eval_ds.select(range(min(20, len(eval_ds))))]
print(f"variant train={len(train_ds)} eval={len(eval_ds)} probe prompts={len(probe_prompts)}")

RUNS = {
    "dpo": {"loss_type": ["sigmoid"]},
    "rpo": {"loss_type": ["sigmoid", "sft"], "loss_weights": [1.0, 1.0]},
    "dpo_norm": {"loss_type": ["sigmoid_norm"]},
    "ld_dpo": {"loss_type": ["sigmoid"], "ld_alpha": 0.5},
}
SELECTED = [r for r in os.environ.get("VARIANTS", "dpo,rpo,dpo_norm,ld_dpo,orpo").split(",") if r]

# %% [markdown]
# ## 1. Các biến thể dựa trên DPOTrainer

# %%
import json

from trl import DPOTrainer

results = {}
for name in [r for r in SELECTED if r in RUNS]:
    print(f"\n=== {name} ===")
    model, tokenizer = MD.load_model(C.SFT_MERGED)
    model = MD.add_lora(model)
    overrides = dict(RUNS[name])
    loss_type = overrides.pop("loss_type")
    trainer = DPOTrainer(
        model=model,
        args=MD.dpo_config(C.VARIANTS_DIR / f"{name}-ckpt", loss_type=loss_type, eval_strategy="no", **overrides),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    trainer.train()
    ev = trainer.evaluate()
    outputs = MD.generate(trainer.model, tokenizer, probe_prompts, max_new_tokens=256)
    results[name] = {
        "eval_reward_accuracy": ev.get("eval_rewards/accuracies"),
        "eval_chosen_reward": ev.get("eval_rewards/chosen"),
        "eval_rejected_reward": ev.get("eval_rewards/rejected"),
        "mean_output_chars": sum(map(len, outputs)) / len(outputs),
        "diagnosis": MD.diagnose(MD.reward_history(trainer.state.log_history))[0],
    }
    trainer.model.save_pretrained(str(C.VARIANTS_DIR / name))
    print(results[name])
    del trainer, model
    MD.cleanup()

# %% [markdown]
# ## 2. ORPO (không reference)

# %%
if "orpo" in SELECTED:
    from trl.experimental.orpo import ORPOConfig, ORPOTrainer

    start = C.BASE_MODEL if os.environ.get("ORPO_FROM_BASE") == "1" else C.SFT_MERGED
    model, tokenizer = MD.load_model(start)
    model = MD.add_lora(model)
    orpo_args = ORPOConfig(
        output_dir=str(C.VARIANTS_DIR / "orpo-ckpt"),
        per_device_train_batch_size=C.TIER.dpo_batch,
        per_device_eval_batch_size=C.TIER.dpo_batch,
        gradient_accumulation_steps=C.TIER.dpo_grad_accum,
        num_train_epochs=C.DPO_EPOCHS,
        learning_rate=C.DPO_LR,
        beta=0.1,  # λ in the paper
        max_length=C.MAX_LEN,
        warmup_steps=0.1,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="no",
        optim="adamw_8bit",
        seed=C.SEED,
        report_to="none",
        **MD.precision_flags(),
    )
    trainer = ORPOTrainer(
        model=model, args=orpo_args, train_dataset=train_ds, eval_dataset=eval_ds, processing_class=tokenizer
    )
    trainer.train()
    ev = trainer.evaluate()
    outputs = MD.generate(trainer.model, tokenizer, probe_prompts, max_new_tokens=256)
    results["orpo"] = {
        "start": str(start),
        "eval_reward_accuracy": ev.get("eval_rewards/accuracies"),
        "eval_log_odds_ratio": ev.get("eval_log_odds_ratio"),
        "mean_output_chars": sum(map(len, outputs)) / len(outputs),
    }
    trainer.model.save_pretrained(str(C.VARIANTS_DIR / "orpo"))
    print(results["orpo"])
    del trainer, model
    MD.cleanup()

# %% [markdown]
# ## 3. Bảng tổng hợp (sản phẩm nộp `03b-variants.png`)

# %%
import matplotlib.pyplot as plt
import pandas as pd

table = pd.DataFrame(results).T
print(table.to_string())
(C.VARIANTS_DIR / "variants_summary.json").write_text(json.dumps(results, indent=2, default=str))

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
table["eval_reward_accuracy"].astype(float).plot.bar(ax=axes[0], color="#2e548a")
axes[0].set_ylim(0, 1)
axes[0].set_title("held-out reward accuracy")
table["mean_output_chars"].astype(float).plot.bar(ax=axes[1], color="#c83538")
axes[1].set_title("mean output length (chars)")
fig.tight_layout()
fig.savefig(C.SCREENSHOTS / "03b-variants.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Câu hỏi cho REFLECTION
#
# 1. Biến thể nào làm đầu ra dài ra nhiều nhất? Có khớp với tỉ lệ "chosen dài hơn" ở NB2 không?
# 2. RPO có giữ `rewards/chosen` dương trong khi DPO thì không? (so với chẩn đoán NB3)
# 3. Độ chính xác reward cao hơn có nghĩa là mô hình tốt hơn không? Chạy giám khảo ở NB4 trên
#    adapter `adapters/variants/<name>` (đặt `DPO_ADAPTER_OVERRIDE`) để kiểm tra.
