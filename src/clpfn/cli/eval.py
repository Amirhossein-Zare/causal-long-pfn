from __future__ import annotations

import argparse
import logging
from typing import Any

from clpfn.cli.config import load_config, pick
from clpfn.evaluation.registry import available_methods, default_config_for, normalize_method, run_evaluation


DEFAULT_DOMAINS = ("cancer", "hiv", "warfarin", "mimic")


def _configured_method(args: argparse.Namespace, cfg: dict[str, Any]) -> str:
    method = args.method or pick(cfg, "method", None)
    if method is None:
        raise ValueError("Provide --method or set method in the evaluation config.")
    return normalize_method(method)


def _load_eval_config(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    initial_cfg = load_config(args.config) if args.config else {}
    method = _configured_method(args, initial_cfg)
    cfg = initial_cfg if args.config else load_config(default_config_for(method))
    return method, cfg


def _run_pfn(args: argparse.Namespace, cfg: dict[str, Any]):
    data_cfg = cfg.get("data", {})
    ready_inputs = data_cfg.get("ready_inputs", {})
    eval_cfg = cfg.get("evaluation", cfg.get("evaluator", cfg))

    checkpoint = args.checkpoint if args.checkpoint is not None else pick(eval_cfg, "checkpoint_path", None)
    if checkpoint is None:
        raise ValueError("Provide --checkpoint or evaluation.checkpoint_path in the config.")

    ready_dirs = args.ready_dir if args.ready_dir is not None else pick(
        eval_cfg,
        "ready_dirs",
        pick(ready_inputs, "ready_dirs", None),
    )
    ready_paths = pick(eval_cfg, "ready_paths", pick(ready_inputs, "ready_paths", None))
    domains = pick(data_cfg, "wanted_domains", pick(eval_cfg, "wanted_domains", DEFAULT_DOMAINS))

    return run_evaluation(
        "pfn",
        checkpoint_path=checkpoint,
        ready_dirs=ready_dirs,
        ready_paths=ready_paths,
        batch_size=pick(eval_cfg, "batch_size", 32),
        wanted_domains=tuple(domains),
        output_dir=args.output_dir if args.output_dir is not None else pick(eval_cfg, "output_dir", None),
        support_sigma_calibration=bool(pick(eval_cfg, "support_sigma_calibration", True)),
    )


def _run_baseline(method: str, args: argparse.Namespace, cfg: dict[str, Any]):
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("evaluation", cfg)
    tune_cfg = cfg.get("tuning", {})
    baseline_cfg = cfg.get("baseline")
    if baseline_cfg is None:
        raise ValueError("Baseline evaluation configs must include a baseline section.")
    return run_evaluation(
        method,
        raw_inputs=pick(data_cfg, "raw_inputs", {}),
        wanted_domains=tuple(pick(data_cfg, "wanted_domains", DEFAULT_DOMAINS)),
        baseline_config=baseline_cfg,
        initial_random_search=int(pick(tune_cfg, "initial_random_search", 40)),
        top_k_reuse=int(pick(tune_cfg, "top_k_reuse", 1)),
        output_dir=args.output_dir if args.output_dir is not None else pick(eval_cfg, "output_dir", None),
    )


def _print_result_summary(result: dict[str, Any]) -> None:
    method = result["method_family"]
    prediction_rows = result.get("prediction_rows", [])

    print("\nEvaluation complete")
    print("Method:", method)
    print("Prediction rows:", len(prediction_rows))
    print("Prediction Parquet:", result.get("prediction_rows_parquet"))
    print("Summary CSV:", result.get("domain_task_summary_csv"))
    if result.get("calibration_summary_csv") is not None:
        print("Calibration CSV:", result["calibration_summary_csv"])


def build_parser(*, default_method: str | None = None, default_config: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate CausalLongPFN or a baseline on benchmark datasets.")
    parser.add_argument("--method", default=default_method, choices=available_methods(), help="Method to evaluate.")
    parser.add_argument("--config", default=default_config, help="Evaluation YAML config.")
    parser.add_argument("--checkpoint", default=None, help="PFN checkpoint override.")
    parser.add_argument("--ready-dir", action="append", default=None, help="PFN ready-file directory override.")
    parser.add_argument("--output-dir", default=None, help="Evaluation output directory override.")
    return parser


def main(*, default_method: str | None = None, default_config: str | None = None) -> None:
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

    parser = build_parser(default_method=default_method, default_config=default_config)
    args = parser.parse_args()

    method, cfg = _load_eval_config(args)
    result = _run_pfn(args, cfg) if method == "pfn" else _run_baseline(method, args, cfg)
    _print_result_summary(result)
