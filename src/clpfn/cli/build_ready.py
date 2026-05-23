from __future__ import annotations

import argparse
import logging

from clpfn.cli.config import load_config, pick
from clpfn.data.formatters import pfn_ready_builder as rb


def main() -> None:
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

    parser = argparse.ArgumentParser(description="Build CausalLongPFN-ready files from raw benchmark pickles.")
    parser.add_argument("--config", default="configs/eval/pfn.yaml", help="Ready-builder YAML config.")
    parser.add_argument("--output-dir", default=None, help="Optional override for ready_builder.output_dir.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    ready_cfg = cfg.get("ready_builder", cfg)
    domains = pick(data_cfg, "wanted_domains", pick(ready_cfg, "wanted_domains", ("cancer", "hiv", "warfarin", "mimic")))

    result = rb.run_all(
        wanted_domains=tuple(domains),
        raw_inputs=pick(data_cfg, "raw_inputs", pick(ready_cfg, "raw_inputs", {})),
        pfn_max_context=pick(ready_cfg, "pfn_max_context", 500),
        max_test_rows_per_task=pick(ready_cfg, "max_test_rows_per_task", None),
        output_dir=args.output_dir if args.output_dir is not None else pick(ready_cfg, "output_dir", "outputs/pfn_ready/all_domains"),
        seed=pick(ready_cfg, "seed", 2026),
    )

    print("\nReady build complete")
    print("Ready files:", len(result["ready_files"]))
    print("Output dir:", result["output_dir"])
