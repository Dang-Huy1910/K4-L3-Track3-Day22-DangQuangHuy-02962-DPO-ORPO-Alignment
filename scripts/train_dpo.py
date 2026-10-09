#!/usr/bin/env python3
"""Train one DPO adapter from the command line (same recipe as NB3).

Usage:
    python scripts/train_dpo.py
    python scripts/train_dpo.py --beta 0.05 --output-dir adapters/dpo-b0.05
    python scripts/train_dpo.py --loss sigmoid,sft --output-dir adapters/rpo

Used by `make beta-sweep`. Needs NB1 (models/sft-merged) and NB2 (data/pref).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--loss", default=None, help="comma-separated TRL loss_type list")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    try:
        import unsloth  # noqa: F401  (patch trl/transformers first)
    except ImportError:
        pass
    from datasets import Dataset
    from trl import DPOTrainer

    from lab22 import config as C
    from lab22 import data as D
    from lab22 import modeling as MD

    output = Path(args.output_dir) if args.output_dir else C.DPO_ADAPTER
    beta = args.beta if args.beta is not None else C.DPO_BETA
    lr = args.lr if args.lr is not None else C.DPO_LR
    loss = args.loss.split(",") if args.loss else C.DPO_LOSS
    print(C.summary())
    print(f"→ beta={beta} lr={lr} loss={loss} output={output}")

    model, tokenizer = MD.load_model(C.SFT_MERGED)
    model = MD.add_lora(model)
    train_ds = Dataset.from_parquet(str(C.PREF_DIR / "train.parquet"))
    eval_ds = Dataset.from_parquet(str(C.PREF_DIR / "eval.parquet"))
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=MD.dpo_config(output.parent / f"{output.name}-checkpoints", loss_type=loss, beta=beta, learning_rate=lr),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    result = trainer.train()
    ev = trainer.evaluate()

    output.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    D.save_split_fingerprint(C.PREF_DIR, output)  # NB4 refuses a held-out set this adapter saw
    train_hist = MD.reward_history(trainer.state.log_history)
    end = train_hist.iloc[-1] if not train_hist.empty else {}
    metrics = {
        "compute_tier": C.COMPUTE_TIER,
        "base_model": C.BASE_MODEL,
        "reference": "models/sft-merged (precomputed)",
        "beta": beta,
        "lr": lr,
        "loss_type": loss,
        "final_train_loss": float(result.training_loss),
        "end_chosen_reward": float(end["rewards/chosen"]) if len(end) else None,
        "end_rejected_reward": float(end["rewards/rejected"]) if len(end) else None,
        "end_reward_gap": float(end["rewards/margins"]) if len(end) else None,
        "eval_chosen_reward": ev.get("eval_rewards/chosen"),
        "eval_rejected_reward": ev.get("eval_rewards/rejected"),
        "eval_reward_gap": ev.get("eval_rewards/margins"),
        "eval_reward_accuracy": ev.get("eval_rewards/accuracies"),
        "diagnosis": MD.diagnose(train_hist)[0],
    }
    (output / "dpo_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
