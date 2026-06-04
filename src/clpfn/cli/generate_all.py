from __future__ import annotations

import argparse
import logging

from clpfn.cli.config import load_config
from clpfn.data.generators.cancer import CancerGeneratorConfig, generate as generate_cancer
from clpfn.data.generators.hiv import HIVGeneratorConfig, generate as generate_hiv
from clpfn.data.generators.mimic import MIMICGeneratorConfig, generate as generate_mimic
from clpfn.data.generators.warfarin import WarfarinGeneratorConfig, generate as generate_warfarin


DOMAINS = ("cancer", "hiv", "warfarin", "mimic")


def main() -> None:
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

    parser = argparse.ArgumentParser(description="Generate benchmark dataset families.")
    parser.add_argument("--config", default="configs/data/all_benchmarks.yaml")
    parser.add_argument("--only", nargs="*", default=None, choices=DOMAINS)
    args = parser.parse_args()

    cfg = load_config(args.config)
    wanted = set(args.only or DOMAINS)

    if "cancer" in wanted:
        generate_cancer(CancerGeneratorConfig.from_dict(cfg.get("cancer", {})))
    if "hiv" in wanted:
        generate_hiv(HIVGeneratorConfig.from_dict(cfg.get("hiv", {})))
    if "warfarin" in wanted:
        generate_warfarin(WarfarinGeneratorConfig.from_dict(cfg.get("warfarin", {})))
    if "mimic" in wanted:
        generate_mimic(MIMICGeneratorConfig.from_dict(cfg.get("mimic", {})))


if __name__ == "__main__":
    main()
