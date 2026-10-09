"""Single source of truth for tier settings, paths, and hyperparameters.

Every notebook and script imports from here, so the T4/BigGPU numbers cannot
drift between the Jupytext sources, the CLI scripts, and the Colab bundles.
Override any value with an environment variable of the same name.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from `start` to the directory that contains the `lab22` package."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "lab22" / "config.py").exists():
            return candidate
    return Path(__file__).resolve().parent.parent


def load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from `.env` without overriding the real environment.

    Supports comments, blank lines, `export`, quotes, and trailing ` # comments`.
    Kept tiny on purpose so the lab has no python-dotenv dependency.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        value = value.strip()
        if value[:1] in ("'", '"') and value[-1:] == value[:1] and len(value) >= 2:
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        os.environ.setdefault(key.strip(), value)


REPO_ROOT = find_repo_root()
load_dotenv(REPO_ROOT / ".env")


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class TierConfig:
    name: str
    base_model: str
    max_len: int
    sft_slice: int
    sft_batch: int
    sft_grad_accum: int
    pref_train: int
    pref_eval: int
    dpo_batch: int
    dpo_grad_accum: int
    variant_train: int
    judge_prompts: int


# Qwen3-4B-Instruct-2507 is the non-thinking instruct release: clean ChatML
# template, no <think> block, Apache-2.0. The Base checkpoint ships without a
# chat template, which is why the lab starts from the instruct model.
_TIERS = {
    "T4": TierConfig(
        name="T4",
        base_model="unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
        max_len=768,
        sft_slice=1000,
        sft_batch=1,
        sft_grad_accum=8,
        pref_train=800,
        pref_eval=100,
        dpo_batch=1,
        dpo_grad_accum=8,
        variant_train=300,
        judge_prompts=50,
    ),
    "BIGGPU": TierConfig(
        name="BIGGPU",
        base_model="unsloth/Qwen3-8B-unsloth-bnb-4bit",
        max_len=1024,
        sft_slice=2000,
        sft_batch=2,
        sft_grad_accum=4,
        pref_train=3500,
        pref_eval=200,
        dpo_batch=2,
        dpo_grad_accum=4,
        variant_train=1000,
        judge_prompts=100,
    ),
}

COMPUTE_TIER = _env("COMPUTE_TIER", "T4").upper()
if COMPUTE_TIER not in _TIERS:
    raise ValueError(f"COMPUTE_TIER must be one of {sorted(_TIERS)}, got {COMPUTE_TIER!r}")
TIER = _TIERS[COMPUTE_TIER]

BASE_MODEL = _env("BASE_MODEL", TIER.base_model)
MAX_LEN = int(_env("MAX_LEN", str(TIER.max_len)))
SEED = int(_env("SEED", "42"))
# 1 = bitsandbytes 4-bit (T4/4B). 0 = fp16, needed to merge LoRA on a 4 GB card
# when BASE_MODEL is a 0.5B–0.6B instruct checkpoint.
LOAD_IN_4BIT = _env("LOAD_IN_4BIT", "1").strip().lower() not in ("0", "false", "no")

# Qwen3 hybrid checkpoints (the BigGPU 8B) think by default. The lab compares
# final answers, so thinking is switched off everywhere a chat template is
# rendered. The 2507 instruct template ignores the flag. Smaller Qwen2.5
# templates reject unknown kwargs such as enable_thinking.
_qwen3_thinking = "Qwen3" in BASE_MODEL and "Instruct-2507" not in BASE_MODEL
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False} if _qwen3_thinking else {}

# --- Data -----------------------------------------------------------------
SFT_DATASET = _env("SFT_DATASET", "saillab/alpaca-vietnamese-cleaned")
SFT_SLICE = int(_env("SFT_SLICE", str(TIER.sft_slice)))

# Vietnamese on-policy UltraFeedback (Sailor2). The lab used English
# UltraFeedback while SFT and evaluation were Vietnamese, which made the
# DPO effect hard to read. Set PREF_DATASET to the argilla id to reproduce
# the English variant as a comparison.
PREF_DATASET = _env("PREF_DATASET", "sailor2/sea-ultrafeedback-onpolicy")
PREF_LANGUAGE = _env("PREF_LANGUAGE", "Vietnamese")
PREF_TRAIN = int(_env("PREF_TRAIN", str(TIER.pref_train)))
PREF_EVAL = int(_env("PREF_EVAL", str(TIER.pref_eval)))

# --- DPO ------------------------------------------------------------------
# LoRA DPO needs a learning rate roughly 10x the full-finetune value: the
# original 5e-7 left the rewards almost flat over ~125 steps.
DPO_BETA = float(_env("DPO_BETA", "0.1"))
DPO_LR = float(_env("DPO_LR", "5e-6"))
DPO_EPOCHS = float(_env("DPO_EPOCHS", "1"))
DPO_LOSS = [s.strip() for s in _env("DPO_LOSS", "sigmoid").split(",") if s.strip()]
LORA_R = int(_env("LORA_R", "16"))
LORA_ALPHA = int(_env("LORA_ALPHA", "32"))
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# --- GRPO (NB7) -------------------------------------------------------------
# Native Vietnamese GSM8K-style problems (MIT, 1,465 rows, numeric `final_answer`).
# Set GRPO_DATASET=openai/gsm8k to use the English original instead.
GRPO_DATASET = _env("GRPO_DATASET", "vuongtsc/vi-gsm8k-agentic")

# --- Judge ----------------------------------------------------------------
# Default "rm": a panel of local reward models, no API key. Scores are per answer,
# so there is no A/B position bias. The models come from different families
# (loaded one at a time, so each only has to fit a T4 on its own); a pair is a DPO
# win only if every judge agrees. Comma-separated; one model also works.
# The Llama judge shares no base model with the data generator (Sailor2, from
# Qwen2.5) or the labeller (Skywork-Reward-Gemma-2-27B); both judges are
# Skywork V2, trained on SynPref-40M rather than the labeller's data.
JUDGE_PROVIDER = _env("JUDGE_PROVIDER", "rm").lower()  # rm | openai | anthropic | gemini
_RM_PANEL = "Skywork/Skywork-Reward-V2-Qwen3-4B,Skywork/Skywork-Reward-V2-Llama-3.2-3B"
JUDGE_RM_MODELS = [m.strip() for m in _env("JUDGE_RM_MODELS", _RM_PANEL).split(",") if m.strip()]
# API judges have no default model id on purpose: ids change faster than the lab,
# so the student picks a current one and records it in REFLECTION.
JUDGE_MODEL = _env("JUDGE_MODEL", "")
JUDGE_PROMPTS = int(_env("JUDGE_PROMPTS", str(TIER.judge_prompts)))
GEN_MAX_NEW_TOKENS = int(_env("GEN_MAX_NEW_TOKENS", "384"))


ADAPTERS = REPO_ROOT / "adapters"
MODELS = REPO_ROOT / "models"
DATA = REPO_ROOT / "data"
SCREENSHOTS = REPO_ROOT / "submission" / "screenshots"

SFT_ADAPTER = ADAPTERS / "sft-mini"
# SFT merged into the base weights. DPO trains a fresh LoRA on top of this,
# so the frozen reference is the SFT model, not the raw base model.
SFT_MERGED = MODELS / "sft-merged"
DPO_ADAPTER = ADAPTERS / "dpo"
VARIANTS_DIR = ADAPTERS / "variants"
GRPO_ADAPTER = ADAPTERS / "grpo"
PREF_DIR = DATA / "pref"
EVAL_DIR = DATA / "eval"
GGUF_DIR = REPO_ROOT / "gguf"


def ensure_dirs() -> None:
    for path in (ADAPTERS, MODELS, PREF_DIR, EVAL_DIR, SCREENSHOTS, GGUF_DIR):
        path.mkdir(parents=True, exist_ok=True)


def summary() -> str:
    return (
        f"COMPUTE_TIER={COMPUTE_TIER} base={BASE_MODEL} max_len={MAX_LEN} seed={SEED}\n"
        f"SFT {SFT_DATASET}[:{SFT_SLICE}]  PREF {PREF_DATASET} ({PREF_LANGUAGE}) "
        f"train={PREF_TRAIN} eval={PREF_EVAL}\n"
        f"DPO beta={DPO_BETA} lr={DPO_LR} epochs={DPO_EPOCHS} loss={DPO_LOSS}"
    )
