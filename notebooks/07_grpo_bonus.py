# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB7 — GRPO với reward kiểm chứng được (RLVR) (THƯỞNG, +8)
#
# DPO học từ *cặp* sở thích có sẵn (ngoại tuyến). GRPO (DeepSeekMath, 2024; dùng trong
# DeepSeek-R1) sinh **G câu trả lời cho mỗi câu hỏi**, chấm bằng hàm reward, rồi đẩy
# xác suất các câu có reward cao hơn trung bình nhóm. Không cần mô hình reward hay
# mô hình critic: với toán, reward là "đáp số đúng hay sai" (RLVR, *verifiable rewards*).
#
# Notebook này chạy một vòng GRPO rất nhỏ để thấy cơ chế, **không** để đạt điểm cao.
# T4: ~40 phút cho 60 bước với G=4.
#
# **Dữ liệu:** `vuongtsc/vi-gsm8k-agentic` (MIT): 1.465 bài toán tiểu học viết mới bằng
# tiếng Việt từ seed GSM8K, đáp số dạng số đã kiểm tra bằng chạy code. Bộ này chỉ có
# split `train`, nên notebook tự chia cố định thành tập huấn luyện/kiểm tra. Các bài được lọc để mô hình yếu
# giải sai, nên độ chính xác ban đầu của mô hình 4B có thể thấp: nếu cả G câu trả lời của một
# câu hỏi đều sai thì advantage = 0 và câu hỏi đó không đóng góp gradient.
#
# | | DPO (NB3) | GRPO (NB7) |
# |---|---|---|
# | Dữ liệu | cặp chosen/rejected cố định | chỉ cần câu hỏi + cách chấm |
# | Sinh trong lúc huấn luyện | không | có (tự sinh, on-policy) |
# | Reference / KL | bắt buộc (β) | tuỳ chọn; TRL mặc định `beta=0.0` |
# | Chi phí | 2 forward / cặp | G lần sinh / câu hỏi |

# %%
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
from lab22 import math_reward as MR
from lab22 import modeling as MD

assert torch.cuda.is_available()
assert C.SFT_MERGED.exists(), "Run NB1 first"
C.ensure_dirs()

BIG = C.COMPUTE_TIER == "BIGGPU"
N_TRAIN, N_TEST, MAX_STEPS, G = (1265, 200, 200, 8) if BIG else (400, 100, 60, 4)

# %% [markdown]
# ## 1. Dữ liệu + hàm reward

# %%
from datasets import load_dataset

INSTRUCTION = "Giải bài toán sau. Suy luận ngắn gọn, rồi kết thúc bằng dòng 'Đáp số: <số>'.\n\n"


def gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def to_row(r):
    # vi-gsm8k-agentic stores the number in `final_answer`; GSM8K puts it after "####".
    ref = str(r["final_answer"]) if "final_answer" in r else gold(r["answer"])
    return {"prompt": [{"role": "user", "content": INSTRUCTION + r["question"]}], "answer": ref.strip()}


if C.GRPO_DATASET == "openai/gsm8k":
    gsm = load_dataset("openai/gsm8k", "main")
    train_raw, test_raw = gsm["train"].shuffle(seed=C.SEED), gsm["test"]
else:
    # Single split: hold out a fixed test slice before any training row is drawn.
    parts = load_dataset(C.GRPO_DATASET, split="train").shuffle(seed=C.SEED).train_test_split(
        test_size=N_TEST, seed=C.SEED
    )
    train_raw, test_raw = parts["train"], parts["test"]
cols = train_raw.column_names
train_ds = train_raw.select(range(min(N_TRAIN, len(train_raw)))).map(to_row, remove_columns=cols)
test_ds = test_raw.select(range(min(N_TEST, len(test_raw)))).map(to_row, remove_columns=cols)
assert not set(train_ds["prompt"][i][0]["content"] for i in range(len(train_ds))) & set(
    test_ds["prompt"][i][0]["content"] for i in range(len(test_ds))
), "GRPO train/test overlap"
print(f"{C.GRPO_DATASET}: train {len(train_ds)} · test {len(test_ds)}")

# Đọc số theo cả kiểu Việt ("1.440", "2,5") lẫn kiểu Anh ("1,440", "2.5"): xem lab22/math_reward.py.
correctness_reward, format_reward = MR.correctness_reward, MR.format_reward
assert MR.is_correct(MR.extract_answer("... vậy\nĐáp số: 1.250"), "1250")
assert MR.is_correct(MR.extract_answer("Đáp số: 2,5 kg"), "2.5")
assert correctness_reward([[{"content": "Đáp số: 7"}]], ["8"]) == [0.0]

# %% [markdown]
# ## 2. Độ chính xác trước khi huấn luyện

# %%
def accuracy(model, tokenizer) -> float:
    prompts = [r["prompt"][0]["content"] for r in test_ds]
    outs = MD.generate(model, tokenizer, prompts, max_new_tokens=320)
    return sum(MR.is_correct(MR.extract_answer(o), r["answer"]) for o, r in zip(outs, test_ds)) / len(test_ds)


model, tokenizer = MD.load_model(C.SFT_MERGED)
acc_before = accuracy(model, tokenizer)
print(f"test[{len(test_ds)}] accuracy before GRPO: {acc_before:.3f}")
model = MD.add_lora(model)

# %% [markdown]
# ## 3. GRPO
#
# `per_device_train_batch_size × grad_accum` phải chia hết cho `num_generations` (G):
# mỗi batch chứa trọn các nhóm G câu trả lời của cùng một câu hỏi.

# %%
from trl import GRPOConfig, GRPOTrainer

args = GRPOConfig(
    output_dir=str(C.ADAPTERS / "grpo-checkpoints"),
    per_device_train_batch_size=G,
    gradient_accumulation_steps=1,
    num_generations=G,
    max_completion_length=320,
    max_steps=MAX_STEPS,
    learning_rate=5e-6,
    warmup_steps=0.1,
    lr_scheduler_type="cosine",
    temperature=1.0,
    chat_template_kwargs=C.CHAT_TEMPLATE_KWARGS,
    logging_steps=5,
    save_strategy="no",
    optim="adamw_8bit",
    seed=C.SEED,
    report_to="none",
    **MD.precision_flags(),
)
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[correctness_reward, format_reward],
    args=args,
    train_dataset=train_ds,
    processing_class=tokenizer,
)
trainer.train()

# %% [markdown]
# ## 4. Đường cong reward + độ chính xác sau huấn luyện (sản phẩm nộp `08-grpo-reward.png`)

# %%
import json

import matplotlib.pyplot as plt
import pandas as pd

logs = pd.DataFrame([r for r in trainer.state.log_history if "reward" in r])
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(logs["step"], logs["reward"], color="#2e548a", label="mean reward")
if "reward_std" in logs:
    ax.fill_between(logs["step"], logs["reward"] - logs["reward_std"], logs["reward"] + logs["reward_std"], alpha=0.2)
ax.set_xlabel("step")
ax.set_ylabel("reward (max 2.5)")
ax.set_title(f"GRPO · {C.GRPO_DATASET.split('/')[-1]} · G={G}")
ax.grid(True, alpha=0.3)
fig.savefig(C.SCREENSHOTS / "08-grpo-reward.png", dpi=120, bbox_inches="tight")
plt.show()

acc_after = accuracy(trainer.model, tokenizer)
trainer.model.save_pretrained(str(C.GRPO_ADAPTER))
result = {
    "dataset": C.GRPO_DATASET, "n_test": len(test_ds), "steps": MAX_STEPS,
    "num_generations": G, "acc_before": acc_before, "acc_after": acc_after,
}
(C.GRPO_ADAPTER / "grpo_metrics.json").write_text(json.dumps(result, indent=2))
print(result)

# %% [markdown]
# ## 5. Câu hỏi
#
# 1. Reward tăng nhanh nhất ở thành phần nào: định dạng hay tính đúng? Đó có phải "khai thác lỗ hổng reward (reward hacking)" không?
# 2. Với N_TEST=100, chênh lệch độ chính xác bao nhiêu mới vượt nhiễu? (sai số chuẩn ≈ √(p(1−p)/n)).
# 3. So `acc_before` với GSM8K (tiếng Anh) của SFT ở NB6: ngôn ngữ, độ khó, câu hỏi và cách chấm
#    đều khác, nên con số nào đáng tin hơn cho câu hỏi "GRPO có giúp toán không"?
