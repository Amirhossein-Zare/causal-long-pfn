from __future__ import annotations

import argparse
import logging
from typing import Any

from clpfn.cli.config import load_config, pick
from clpfn.evaluation.registry import (
    available_methods,
    default_config_for,
    is_pfn_method,
    is_ready_baseline_method,
    normalize_method,
    run_evaluation,
)


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


def _ready_inputs(args: argparse.Namespace, data_cfg: dict[str, Any], eval_cfg: dict[str, Any]):
    ready_inputs = data_cfg.get("ready_inputs", {})
    ready_dirs = args.ready_dir if args.ready_dir is not None else pick(
        eval_cfg,
        "ready_dirs",
        pick(ready_inputs, "ready_dirs", None),
    )
    ready_paths = pick(eval_cfg, "ready_paths", pick(ready_inputs, "ready_paths", None))
    return ready_dirs, ready_paths


def _run_pfn(args: argparse.Namespace, cfg: dict[str, Any]):
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("evaluation", cfg.get("evaluator", cfg))

    checkpoint = args.checkpoint if args.checkpoint is not None else pick(eval_cfg, "checkpoint_path", None)
    if checkpoint is None:
        raise ValueError("Provide --checkpoint or evaluation.checkpoint_path in the config.")

    ready_dirs, ready_paths = _ready_inputs(args, data_cfg, eval_cfg)
    domains = pick(data_cfg, "wanted_domains", pick(eval_cfg, "wanted_domains", DEFAULT_DOMAINS))

    return run_evaluation(
        "pfn",
        checkpoint_path=checkpoint,
        ready_dirs=ready_dirs,
        ready_paths=ready_paths,
        batch_size=pick(eval_cfg, "batch_size", 32),
        wanted_domains=tuple(domains),
        output_dir=args.output_dir if args.output_dir is not None else pick(eval_cfg, "output_dir", None),
        report_calibration=bool(pick(eval_cfg, "report_calibration", True)),
        evaluation_id=args.evaluation_id if args.evaluation_id is not None else pick(eval_cfg, "evaluation_id", None),
    )


def _run_ready_baseline(method: str, args: argparse.Namespace, cfg: dict[str, Any]):
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("evaluation", cfg)
    ready_dirs, ready_paths = _ready_inputs(args, data_cfg, eval_cfg)
    domains = pick(data_cfg, "wanted_domains", pick(eval_cfg, "wanted_domains", DEFAULT_DOMAINS))
    return run_evaluation(
        method,
        ready_dirs=ready_dirs,
        ready_paths=ready_paths,
        wanted_domains=tuple(domains),
        baseline_config=cfg.get("baseline", {}),
        output_dir=args.output_dir if args.output_dir is not None else pick(eval_cfg, "output_dir", None),
        evaluation_id=args.evaluation_id if args.evaluation_id is not None else pick(eval_cfg, "evaluation_id", None),
    )


def _run_baseline(method: str, args: argparse.Namespace, cfg: dict[str, Any]):
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("evaluation", cfg)
    tune_cfg = cfg.get("tuning", {})
    fitting_cfg = cfg.get("fitting", {})
    baseline_cfg = cfg.get("baseline")
    if baseline_cfg is None:
        raise ValueError("Baseline evaluation configs must include a baseline section.")
    execution_cfg = cfg.get("execution", {})
    return run_evaluation(
        method,
        raw_inputs=pick(data_cfg, "raw_inputs", {}),
        wanted_domains=tuple(pick(data_cfg, "wanted_domains", DEFAULT_DOMAINS)),
        baseline_config=baseline_cfg,
        initial_random_search=int(pick(tune_cfg, "initial_random_search", 40)),
        output_dir=args.output_dir if args.output_dir is not None else pick(eval_cfg, "output_dir", None),
        fit_root=pick(fitting_cfg, "fit_root", "outputs/fits"),
        evaluation_id=args.evaluation_id if args.evaluation_id is not None else pick(eval_cfg, "evaluation_id", None),
        mode=args.mode if args.mode is not None else pick(execution_cfg, "mode", None),
        tuning_state_dir=args.tuning_state_dir if args.tuning_state_dir is not None else pick(tune_cfg, "state_dir", None),
        selected_hparams_path=args.selected_hparams if args.selected_hparams is not None else pick(tune_cfg, "selected_hparams_path", None),
        final_fit_seed=int(args.fit_seed if args.fit_seed is not None else pick(fitting_cfg, "final_fit_seed", 101)),
        query_seed=int(args.query_seed if args.query_seed is not None else pick(eval_cfg, "query_seed", 4242)),
        hpo_plan_seed=int(pick(tune_cfg, "plan_seed", 1701)),
        hpo_split_seed=int(pick(tune_cfg, "split_seed", 2701)),
        hpo_train_seed=int(pick(tune_cfg, "train_seed", 3701)),
        hpo_validation_seed=int(pick(tune_cfg, "validation_seed", 4701)),
    )


def _print_result_summary(result: dict[str, Any]) -> None:
    method = result["method_family"]
    prediction_rows = result.get("prediction_rows", [])

    print("\nEvaluation complete")
    print("Method:", method)
    print("Mode:", result.get("mode", "evaluate"))
    if result.get("mode") == "tune":
        print("Completed tasks:", f"{result.get('completed_tasks')}/{result.get('expected_tasks')}")
        print("Selected hyperparameters:", result.get("selected_hparams_path"))
        print("Tuning trials:", result.get("tuning_trials_path"))
        return
    print("Prediction rows:", len(prediction_rows))
    print("Prediction Parquet:", result.get("prediction_rows_parquet"))
    print("Summary CSV:", result.get("domain_task_summary_csv"))
    if result.get("calibration_summary_csv") is not None:
        print("Calibration CSV:", result["calibration_summary_csv"])
    if result.get("calibration_rows_parquet") is not None:
        print("Calibration rows Parquet:", result["calibration_rows_parquet"])


def build_parser(*, default_method: str | None = None, default_config: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate CausalLongPFN or a baseline on benchmark datasets.")
    parser.add_argument("--method", default=default_method, choices=available_methods(), help="Method to evaluate.")
    parser.add_argument("--config", default=default_config, help="Evaluation YAML config.")
    parser.add_argument("--checkpoint", default=None, help="PFN checkpoint override.")
    parser.add_argument("--ready-dir", action="append", default=None, help="Ready-file directory override.")
    parser.add_argument("--output-dir", default=None, help="Evaluation output directory override.")
    parser.add_argument("--evaluation-id", default=None, help="Stable evaluation identifier for resumable output partitions.")
    parser.add_argument("--mode", choices=("tune", "evaluate"), default=None, help="Tune or evaluate from a saved manifest.")
    parser.add_argument("--tuning-state-dir", default=None, help="Writable directory for atomic per-trial tuning state.")
    parser.add_argument("--selected-hparams", default=None, help="Selected-hyperparameter parquet manifest for evaluation-only mode.")
    parser.add_argument("--fit-seed", type=int, default=None, help="Root seed for final model fitting.")
    parser.add_argument("--query-seed", type=int, default=None, help="Root seed for fixed query/origin sampling.")
    return parser


def main(*, default_method: str | None = None, default_config: str | None = None) -> None:
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

    parser = build_parser(default_method=default_method, default_config=default_config)
    args = parser.parse_args()

    method, cfg = _load_eval_config(args)
    if is_pfn_method(method):
        result = _run_pfn(args, cfg)
    elif is_ready_baseline_method(method):
        result = _run_ready_baseline(method, args, cfg)
    else:
        result = _run_baseline(method, args, cfg)
    _print_result_summary(result)


if __name__ == "__main__":
    main()
