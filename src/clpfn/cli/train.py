from __future__ import annotations

import argparse
import logging
import shlex
import sys

from clpfn.cli.config import load_config
from clpfn.config import defaults


def main() -> None:
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

    parser = argparse.ArgumentParser(description="Train CausalLongPFN on the synthetic TSCM prior.")
    parser.add_argument("--config", default="configs/train/causal_long_pfn.yaml")
    parser.add_argument("--ckpt-input-dir", default=None, help="Optional directory to resume from.")
    parser.add_argument("--output-dir", default=None, help="Directory for checkpoints.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    runtime = cfg.get("runtime", {}) or {}

    ckpt_input_dir = args.ckpt_input_dir if args.ckpt_input_dir is not None else runtime.get("ckpt_input_dir", "")
    output_dir = args.output_dir if args.output_dir is not None else runtime.get("output_dir", "outputs/causal_long_pfn_outputs")

    cfg["runtime"] = {**runtime, "ckpt_input_dir": str(ckpt_input_dir or ""), "output_dir": str(output_dir)}
    defaults.apply_training_config(cfg)

    from clpfn.training.train_pfn import train

    train(
        resolved_config=cfg,
        launch_command=shlex.join(sys.argv),
        ckpt_input_dir=str(ckpt_input_dir or ""),
        output_dir=str(output_dir),
    )


if __name__ == "__main__":
    main()
