#!/usr/bin/env python3
"""Export SFT+DPO as GGUF from the command line (same recipe as NB5).

Usage:
    python scripts/merge_and_gguf.py
    python scripts/merge_and_gguf.py --adapter adapters/variants/rpo --quant q4_k_m --quant q8_0

The adapter's config points at models/sft-merged, so loading the adapter
loads SFT + DPO. The original script merged only the SFT adapter.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None, help="default: adapters/dpo")
    parser.add_argument("--output", default=None, help="default: gguf/")
    parser.add_argument("--quant", action="append", default=None, help="repeatable; default q4_k_m")
    args = parser.parse_args()

    try:
        import unsloth  # noqa: F401
    except ImportError:
        pass

    from lab22 import config as C
    from lab22 import modeling as MD

    adapter = Path(args.adapter) if args.adapter else C.DPO_ADAPTER
    output = Path(args.output) if args.output else C.GGUF_DIR
    quants = args.quant or ["q4_k_m"]
    if not (adapter / "adapter_config.json").exists():
        print(f"{adapter}/adapter_config.json missing. Run NB3 first.")
        return 1

    model, tokenizer = MD.load_model(adapter, load_in_4bit=False)
    if not any("lora_" in n for n, _ in model.named_parameters()):
        print("LoRA weights were not loaded; refusing to export an SFT-only GGUF.")
        return 1
    model.save_pretrained_gguf(str(output), tokenizer, quantization_method=quants if len(quants) > 1 else quants[0])
    for p in sorted(REPO.glob("gguf*/**/*.gguf")):
        print(f"  {p.relative_to(REPO)}  {p.stat().st_size / 1e6:,.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
