from __future__ import annotations

import argparse
import logging
import os

from clpfn.cli.config import load_config
from clpfn.config import defaults


def main() -> None:
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

    parser = argparse.ArgumentParser(description="Train CausalLongPFN on the synthetic TSCM prior.")
    parser.add_argument("--config", default="configs/train/causal_long_pfn.yaml")
    parser.add_argument("--ckpt-input-dir", default=None, help="Optional directory to resume from.")
    parser.add_argument("--output-dir", default=None, help="Directory for checkpoints.")
    args = parser.parse_args()

    defaults.configure_from_file(args.config)
    cfg = load_config(args.config)
    runtime = cfg.get("runtime", {}) or {}

    ckpt_input_dir = args.ckpt_input_dir if args.ckpt_input_dir is not None else runtime.get("ckpt_input_dir", "")
    output_dir = args.output_dir if args.output_dir is not None else runtime.get("output_dir", "outputs/causal_long_pfn_outputs")

    os.environ["CAUSALLONGPFN_CKPT_INPUT_DIR"] = str(ckpt_input_dir or "")
    os.environ["CAUSALLONGPFN_OUTPUT_DIR"] = str(output_dir)

    from clpfn.training.train_pfn import train

    train()


if __name__ == "__main__":
    main()
