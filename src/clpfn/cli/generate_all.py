from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

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
    parser.add_argument("--overwrite", action="store_true", help="Replace generated files in the selected existing build directories.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    wanted = set(args.only or DOMAINS)

    output_dirs = []
    if "cancer" in wanted:
        cancer_cfg = CancerGeneratorConfig.from_dict(cfg["cancer"])
        cancer_cfg.overwrite = bool(args.overwrite or cancer_cfg.overwrite)
        generate_cancer(cancer_cfg)
        output_dirs.append(Path(cancer_cfg.output_dir))
    if "hiv" in wanted:
        hiv_cfg = HIVGeneratorConfig.from_dict(cfg["hiv"])
        hiv_cfg.overwrite = bool(args.overwrite or hiv_cfg.overwrite)
        generate_hiv(hiv_cfg)
        output_dirs.append(Path(hiv_cfg.output_dir))
    if "warfarin" in wanted:
        warfarin_cfg = WarfarinGeneratorConfig.from_dict(cfg["warfarin"])
        warfarin_cfg.overwrite = bool(args.overwrite or warfarin_cfg.overwrite)
        generate_warfarin(warfarin_cfg)
        output_dirs.append(Path(warfarin_cfg.output_dir))
    if "mimic" in wanted:
        mimic_cfg = MIMICGeneratorConfig.from_dict(cfg["mimic"])
        mimic_cfg.overwrite = bool(args.overwrite or mimic_cfg.overwrite)
        generate_mimic(mimic_cfg)
        output_dirs.append(Path(mimic_cfg.output_dir))
    build_roots = {directory.parent for directory in output_dirs}
    if len(build_roots) != 1:
        raise ValueError("All selected domains must write beneath one benchmark build directory.")
    build_root = build_roots.pop()
    dataset_manifests = sorted(build_root.glob("*/benchmark_dataset_manifest_*.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in dataset_manifests]
    payload = {
        "benchmark_build_id": build_root.name,
        "output_dir": str(build_root),
        "dataset_count": len(records),
        "dataset_manifest_files": [str(path) for path in dataset_manifests],
        "datasets": records,
    }
    with (build_root / "benchmark_build_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
