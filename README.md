# CausalLongPFN

CausalLongPFN is a research codebase for longitudinal counterfactual outcome
prediction with prior-fitted networks. It supports synthetic pretraining,
benchmark generation, PFN-ready dataset construction, checkpoint evaluation,
calibration summaries, and baseline comparison.

The benchmark protocol conditions on an observed history through `t_obs`,
receives planned future actions, and predicts the requested future outcome.
Cancer, HIV, and warfarin benchmarks include branchable counterfactual labels;
MIMIC-III-style ICU experiments are factual rolling-origin prediction tasks.

## Features

- Synthetic TSCM episode generation for PFN pretraining.
- Benchmark builders for cancer, HIV, warfarin, and MIMIC-III-style data.
- Unified evaluation for CausalLongPFN and baseline methods.
- Baseline adapters for MSM, RMSN, G-Net, CRN, Causal Transformer, and
  G-Transformer.
- Row-level prediction outputs, aggregate metrics, and PFN calibration reports.

## Installation

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## Quick Start

Generate benchmark data:

```bash
clpfn-generate-all --config configs/data/all_benchmarks.yaml
```

Train CausalLongPFN:

```bash
clpfn-train --config configs/train/causal_long_pfn.yaml
```

Build PFN-ready evaluation files:

```bash
clpfn-build-ready --config configs/eval/pfn.yaml
```

Evaluate a checkpoint:

```bash
clpfn-eval --method pfn --config configs/eval/pfn.yaml --checkpoint /path/to/checkpoint.pt
```

Evaluate a baseline:

```bash
clpfn-eval --method gtransformer --config configs/eval/gtransformer.yaml
```

Other baseline methods are `ct`, `crn`, `gnet`, `rmsn`, and `msm`.

## Repository Layout

```text
configs/      YAML configs for data generation, training, and evaluation
src/clpfn/    CausalLongPFN package source
```

## Outputs and Data

Evaluation outputs are written under the configured `evaluation.output_dir`.
Generated benchmark files use the CLPFN schema:

```text
support_data, test_data, test_data_factuals, test_data_seq, scaling_data
```

Raw datasets, MIMIC-III files, checkpoints, generated benchmark pickles,
PFN-ready files, and evaluation artifacts are not part of the repository. The
`.gitignore` excludes the expected local output directories and generated data
formats.

MIMIC-III data are not redistributed. To reproduce MIMIC-style experiments,
obtain access through the official MIMIC data-use process and place the
MIMIC-Extract-style `all_hourly_data.h5` file at the configured `data/raw/...`
path.

## Notes

The baseline adapters are dependency-light implementations adapted to the CLPFN
benchmark protocol. Preserve upstream attribution and license requirements for
external methods or simulator code when redistributing this repository.

CausalLongPFN is a research tool. Counterfactual interpretation on
observational data requires the standard longitudinal assumptions, including
consistency, positivity, and sequential exchangeability given the measured
history.
