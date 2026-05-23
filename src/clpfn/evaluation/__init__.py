"""Evaluation package layout.

``core`` contains method-agnostic benchmark contracts: input discovery,
task definitions, row schemas, output paths, and RMSE summaries.

``pfn`` contains CausalLongPFN-specific evaluation code: ready-batch collation,
checkpoint rollout, and GMM calibration.

Baseline model adapters live under :mod:`clpfn.baselines`; the shared baseline
evaluation loop is :mod:`clpfn.baselines.common.runner`.
"""
