# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB5 — Gộp SFT+DPO → GGUF Q4_K_M (TUỲ CHỌN, thưởng điểm)
#
# > Phần lõi của lab = NB0–NB4. Bước này biên dịch llama.cpp lúc chạy (~3–5 phút lần đầu).
#
# **Lỗi của lab cũ:** NB5 chỉ nạp adapter **SFT** rồi gộp, nên file GGUF không có DPO.
# Ở đây ta nạp `adapters/dpo` (adapter config trỏ tới `models/sft-merged`), tức là
# SFT + DPO, rồi xuất file. Có một bước kiểm tra để chắc chắn adapter DPO thực sự được nạp.

# %%
import json
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
assert (C.DPO_ADAPTER / "adapter_config.json").exists(), "Run NB3 first"
C.ensure_dirs()

adapter_cfg = json.loads((C.DPO_ADAPTER / "adapter_config.json").read_text())
print(f"DPO adapter base: {adapter_cfg.get('base_model_name_or_path')}")

# %% [markdown]
# ## 1. Nạp SFT + DPO ở 16-bit
#
# Gộp vào trọng số 4-bit làm mất độ chính xác, nên nạp 16-bit (`load_in_4bit=False`).
# T4 16 GB đủ cho 4B ở fp16 (~8 GB).

# %%
model, tokenizer = MD.load_model(C.DPO_ADAPTER, load_in_4bit=False)
n_lora = sum(1 for n, _ in model.named_parameters() if "lora_" in n)
assert n_lora > 0, "DPO LoRA weights not loaded: the GGUF would be SFT-only"
print(f"LoRA tensors loaded: {n_lora}")

SMOKE_PROMPT = "Giải thích ngắn gọn (3 câu) cách thuật toán Bubble sort hoạt động."
SMOKE_TOKENS = 160
hf_answer = MD.generate(model, tokenizer, [SMOKE_PROMPT], max_new_tokens=SMOKE_TOKENS)[0]
# Render the prompt once with the HF template (thinking off) and feed GGUF the same text,
# so the HF vs GGUF comparison differs only by quantization, not by chat formatting.
smoke_text = tokenizer.apply_chat_template(
    [{"role": "user", "content": SMOKE_PROMPT}], tokenize=False, add_generation_prompt=True, **C.CHAT_TEMPLATE_KWARGS
)
stop_tokens = [tokenizer.eos_token] if tokenizer.eos_token else []
print(f"HF (SFT+DPO) answer:\n{hf_answer}")

# %% [markdown]
# ## 2. Xuất GGUF Q4_K_M
#
# `save_pretrained_gguf` gộp LoRA vào trọng số 16-bit rồi gọi llama.cpp để lượng tử hoá.
# Tên file và thư mục con khác nhau giữa các bản Unsloth, nên ta tìm file bằng glob đệ quy.

# %%
model.save_pretrained_gguf(str(C.GGUF_DIR), tokenizer, quantization_method="q4_k_m")
# Optional for the +3 rigor add-on: quantization_method=["q4_k_m", "q5_k_m", "q8_0"]


def find_gguf(pattern: str = "q4_k_m") -> Path:
    hits = [p for p in C.REPO_ROOT.glob("gguf*/**/*.gguf") if pattern in p.name.lower()]
    assert hits, f"No *{pattern}*.gguf under {C.REPO_ROOT}/gguf*"
    return max(hits, key=lambda p: p.stat().st_mtime)


gguf_path = find_gguf()
print(f"{gguf_path.relative_to(C.REPO_ROOT)}  {gguf_path.stat().st_size / 1e9:.2f} GB")
del model
MD.cleanup()

# %% [markdown]
# ## 3. Kiểm tra nhanh (smoke test) bằng llama-cpp-python (sản phẩm nộp `06-gguf-smoke.png`)
#
# So câu trả lời GGUF với câu trả lời HF ở §1. Lượng tử hoá 4-bit làm câu chữ lệch một
# chút, nhưng nội dung và phong cách phải giống nhau. Nếu GGUF trả lời giống hệt
# mô hình SFT (NB4) thì adapter DPO đã bị bỏ sót.

# %%
from llama_cpp import Llama

llm = Llama(model_path=str(gguf_path), n_ctx=C.MAX_LEN, n_gpu_layers=-1, verbose=False)
resp = llm.create_completion(smoke_text, max_tokens=SMOKE_TOKENS, temperature=0.0, stop=stop_tokens)
gguf_answer = resp["choices"][0]["text"].strip()
print(f"PROMPT: {SMOKE_PROMPT}\n\nGGUF Q4_K_M:\n{gguf_answer}\n\nusage: {resp['usage']}")
llm.close()  # release the llama.cpp GPU buffers before the next notebook
del llm

# %%
deploy_meta = {
    "compute_tier": C.COMPUTE_TIER,
    "base_model": C.BASE_MODEL,
    "adapter": str(C.DPO_ADAPTER.relative_to(C.REPO_ROOT)),
    "adapter_base": adapter_cfg.get("base_model_name_or_path"),
    "gguf_path": str(gguf_path.relative_to(C.REPO_ROOT)),
    "gguf_size_mb": round(gguf_path.stat().st_size / 1e6, 1),
    "quantization": "q4_k_m",
    "smoke_prompt": SMOKE_PROMPT,
    "chat_template_kwargs": C.CHAT_TEMPLATE_KWARGS,
    "max_new_tokens": SMOKE_TOKENS,
    "hf_answer": hf_answer,
    "gguf_answer": gguf_answer,
}
(C.EVAL_DIR / "deploy_meta.json").write_text(json.dumps(deploy_meta, ensure_ascii=False, indent=2))
print("Saved data/eval/deploy_meta.json")

# %% [markdown]
# ## 4. Triển khai (tham khảo, chạy ngoài notebook)
#
# Máy chủ llama.cpp (CPU/GPU, file GGUF):
#
# ```bash
# llama-server -m gguf/<file>.Q4_K_M.gguf --port 8080 -c 2048
# ```
#
# vLLM (BigGPU, ≥16 GB): phục vụ mô hình 16-bit có LoRA mà không cần gộp:
#
# ```bash
# vllm serve models/sft-merged --enable-lora --lora-modules dpo=adapters/dpo \
#   --max-model-len 2048 --port 8000
# ```
#
# Gọi API với `"model": "dpo"`. vLLM giữ tiến trình và GPU, nên chạy ở cửa sổ dòng lệnh riêng.
#
# **Tiếp theo:** NB6 (benchmark) hoặc `make verify`.
