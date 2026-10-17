# CausalLongPFN

[![arXiv](https://img.shields.io/badge/arXiv-2606.05797-b31b1b.svg)](https://arxiv.org/abs/2606.05797)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2606.05797-blue.svg)](https://doi.org/10.48550/arXiv.2606.05797)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow.svg)](https://huggingface.co/Amirhossein-Zare/causal-long-pfn)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.20588199-blue.svg)](https://doi.org/10.5281/zenodo.20588199)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://www.python.org/)

**CausalLongPFN** is the reference implementation for *Causal Longitudinal
Prior-Fitted Networks for Counterfactual Outcome Prediction*.

We introduce **Causal Longitudinal Prior-Fitted Networks
(CausalLongPFN)**, a prior-fitted network for **time-series causal inference in
longitudinal treatment-response data** and **zero-shot in-context
counterfactual outcome prediction**. The project studies history-conditional
potential-outcome prediction from longitudinal treatment-response time series:
given support trajectories from a new domain, a query history, and a planned
future treatment sequence, a frozen CausalLongPFN model predicts a one-step
outcome distribution and recursively rolls out its mean under the planned
treatments, without target-domain gradient updates, propensity-model fitting, or
adversarial balancing. The model is pretrained on synthetic temporal structural
causal models (TSCMs) and evaluated on branchable cancer, HIV, and warfarin
counterfactual benchmarks, as well as factual MIMIC-III ICU rolling-origin
prediction.

## Paper

**Causal Longitudinal Prior-Fitted Networks for Counterfactual Outcome Prediction**  
Amirhossein Zare, Amirhessam Zare, Herlock Rahimi, Reza Salarikia, Mohammad Kashkooli

- arXiv: <https://arxiv.org/abs/2606.05797>
- DOI: <https://doi.org/10.48550/arXiv.2606.05797>
- Pretrained weights: <https://huggingface.co/Amirhossein-Zare/causal-long-pfn>

## Citation

```bibtex
@misc{zare2026causallongitudinalpriorfittednetworks,
  title={Causal Longitudinal Prior-Fitted Networks for Counterfactual Outcome Prediction},
  author={Amirhossein Zare and Amirhessam Zare and Herlock Rahimi and Reza Salarikia and Mohammad Kashkooli},
  year={2026},
  eprint={2606.05797},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2606.05797}
}
```

## What is included

This repository contains code for:

- synthetic CausalLongPFN pretraining and matched pretraining ablations from a temporal structural causal model prior;
- benchmark generation for cancer, HIV, warfarin, and MIMIC-III-style ICU
  treatment-response tasks;
- conversion of benchmark files into CausalLongPFN-ready support/query datasets;
- zero-shot in-context evaluation of the frozen CausalLongPFN model;
- baseline evaluation for MSM, RMSN, G-Net, and Causal Transformer, plus
  persistence and linear-autoregressive reference baselines;
- normalized RMSE summaries, row-level prediction outputs, and one-step
  probabilistic calibration diagnostics.


## Repository layout

```text
configs/
  data/                     Benchmark-generation configs
  train/                    CausalLongPFN synthetic-pretraining configs
  train/ablations/          Seven pretraining-ablation configs
  eval/                     CausalLongPFN and baseline evaluation configs

src/clpfn/
  cli/                      Command-line entry points
  config/                   Runtime defaults
  data/
    generators/             Cancer, HIV, warfarin, and MIMIC-III generators
    formatters/             PFN-ready dataset builder
    priors/                 Synthetic TSCM episode generator
  models/                   CausalLongPFN model
  training/                 Losses, optimizer setup, checkpointing, training loop
  evaluation/               PFN and baseline evaluation pipeline
  baselines/                MSM, RMSN, G-Net, and CT adapters
```

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/Amirhossein-Zare/causal-long-pfn.git
cd causal-long-pfn

python -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

The editable install exposes the command-line tools used below:

```text
clpfn-generate-all
clpfn-train
clpfn-build-ready
clpfn-eval
```

CUDA is recommended for CausalLongPFN pretraining and for neural baselines, but
the package can be imported without a GPU.

## Pretrained weights

The seed-42 checkpoint, pretrained for 10,000 optimizer updates, is available
on Hugging Face as inference-only weights:

<https://huggingface.co/Amirhossein-Zare/causal-long-pfn>

```bash
pip install safetensors huggingface_hub
```

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Amirhossein-Zare/causal-long-pfn",
    local_dir="weights",
)
```

`--checkpoint` accepts a training `.pt` file, a released `.safetensors` file, or a
directory holding the downloaded release:

```bash
clpfn-eval --method pfn --config configs/eval/pfn.yaml --checkpoint weights
```

## End-to-end workflow

### 1. Generate benchmark data

Generate all configured benchmark domains:

```bash
clpfn-generate-all --config configs/data/all_benchmarks.yaml
```

To generate only the branchable simulated/semi-mechanistic domains:

```bash
clpfn-generate-all --config configs/data/all_benchmarks.yaml --only cancer hiv warfarin
```

By default, outputs are written under:

```text
outputs/benchmarks/cancer
outputs/benchmarks/hiv
outputs/benchmarks/warfarin
outputs/benchmarks/mimic
```

### 2. Train CausalLongPFN

```bash
clpfn-train --config configs/train/causal_long_pfn.yaml
```

The default training config uses synthetic TSCM episodes generated on the fly
for 10,000 updates with seed `42`. Checkpoints are written to:

```text
outputs/paper_runs/causal_long_pfn/seed_42
```

To resume from an existing checkpoint directory or write to a different output
directory:

```bash
clpfn-train \
  --config configs/train/causal_long_pfn.yaml \
  --ckpt-input-dir outputs/paper_runs/causal_long_pfn/seed_42 \
  --output-dir outputs/paper_runs/causal_long_pfn/seed_42
```

### 3. Build PFN-ready evaluation files

CausalLongPFN evaluation uses compact support/query files built from the raw
benchmark pickles:

```bash
clpfn-build-ready --config configs/eval/pfn.yaml
```

The default output directory is:

```text
outputs/pfn_ready
```

### 4. Evaluate CausalLongPFN

```bash
clpfn-eval \
  --method pfn \
  --config configs/eval/pfn.yaml \
  --checkpoint outputs/paper_runs/causal_long_pfn/seed_42/ckpt_final.pt
```

Evaluation outputs are written under the configured `evaluation.output_dir`, by
default:

```text
outputs/paper_evaluations/causal_long_pfn/seed_42
```

The main files are:

```text
prediction_rows.parquet
  Row-level predictions and errors.

domain_task_normalized_rmse.csv
  Mean and standard deviation of normalized RMSE by domain, task, and method.

sequential_horizon_normalized_rmse.csv
  Normalized RMSE across rollout horizons 1–5.

calibration_summary_domain.csv
  One-step probabilistic calibration summary by domain, when enabled.

calibration_rows.parquet
  Row-level one-step calibration diagnostics, when enabled.
```

### 5. Evaluate baselines

Each baseline has its own config under `configs/eval/`. For the tuned baselines,
tuning and evaluation are separate operations: tuning writes a
selected-hyperparameter manifest that the evaluation run then consumes.

```bash
clpfn-eval --method msm --config configs/eval/msm.yaml --mode tune
clpfn-eval --method msm --config configs/eval/msm.yaml --mode evaluate
```

The same two-step form applies to `rmsn`, `gnet`, and `ct`. MSM uses a fixed
prespecified configuration; its tuning step records the manifest. The two
reference baselines need no tuning and run in a single step:

```bash
clpfn-eval --method persistence           --config configs/eval/persistence.yaml
clpfn-eval --method linear_autoregressive --config configs/eval/linear_autoregressive.yaml
```

RMSN, G-Net, and CT are selected using support-only validation and refit on the
full support set. MSM and Linear AR are fit on the support data; Persistence
requires no fitting. CausalLongPFN remains frozen, using support trajectories
only as in-context input.

## Benchmarks

| Domain | Evaluation type | Notes |
| --- | --- | --- |
| Cancer | Branchable counterfactual | Tumor-growth treatment-response benchmark with alternative future actions. |
| HIV | Branchable counterfactual | Treatment-dynamics benchmark with alternative future regimens. |
| Warfarin | Branchable counterfactual | PK/PD-style treatment-response benchmark with alternative dose sequences. |
| MIMIC-III | Factual rolling-origin prediction | Real ICU trajectories evaluated only under observed future treatments. |

The simulated and semi-mechanistic domains provide counterfactual labels by
replaying patient-specific dynamics under alternative treatment sequences.
MIMIC-III is factual-only; it should not be used as evidence of individual
counterfactual treatment-effect accuracy under unobserved ICU interventions.

## MIMIC-III data

MIMIC-III is not redistributed with this repository. To run the MIMIC-III
benchmark, obtain credentialed access through the official PhysioNet data-use
process and prepare a MIMIC-Extract-style `all_hourly_data.h5` file.

## Configuration

Most behavior is controlled through YAML files:

- `configs/data/all_benchmarks.yaml` sets domains, support sizes, repetitions,
  sequence length, prediction horizon, and output directories.
- `configs/train/causal_long_pfn.yaml` sets the main pretraining configuration;
  matched 2,500-update ablation configs are under `configs/train/ablations/`,
  with the non-ablated reference in `configs/train/canonical_reference.yaml`.
- `configs/eval/pfn.yaml` sets PFN-ready input directories, checkpoint path,
  batch size, calibration reporting, and output directory.
- `configs/eval/*.yaml` set baseline-specific hyperparameter search spaces,
  limits, and output directories.

The default benchmark grid uses support sizes `40, 80, 160, 320, 500`, five
confounding/task-index levels (`gammas: 1, 3, 5, 7, 9`), one repetition per cell,
sequence length `60`, and horizon `5`, giving 25 datasets per domain.

## Reported results

The paper reports results averaged across three pretraining seeds (`42, 43, 44`),
with each model frozen during evaluation. Across cancer, HIV, and warfarin, CausalLongPFN
has the lowest one-step counterfactual macro-average normalized RMSE (`0.067`)
and remains competitive at five steps (`0.288`), where RMSN has the lowest
mean error. On factual MIMIC-III prediction, none of the four fitted causal
baselines differs significantly from CausalLongPFN at either horizon after Holm
correction; CausalLongPFN outperforms both predictive references at one step.

These results support synthetic pretraining as a way to transfer across
longitudinal tasks without target-domain fitting. Matched ablations highlight
the complementary roles of mixed factual/interventional queries and multiple
labeled support anchors.

For exact numbers and experimental details, see the paper.

## Causal scope and limitations

CausalLongPFN is a research tool for time-series causal inference in
longitudinal treatment-response data, causal sequence modeling, and
history-conditional counterfactual outcome prediction. It does not remove the
assumptions required for causal interpretation of observational data.
Counterfactual interpretation still requires, among other conditions,
consistency, positivity, sequential exchangeability given the measured history,
adequate treatment overlap, and well-defined interventions.

Transfer depends on coverage of the synthetic TSCM prior; predictions may be
unreliable for poorly represented mechanisms, treatment policies, or data
patterns. The current implementation uses discrete treatments, fixed time grids,
and deterministic mean rollout, which does not propagate full trajectory
uncertainty. These are implementation choices, not fundamental restrictions of
the framework. Predictive intervals require domain-specific calibration checks.

MIMIC-III results should be interpreted as factual rolling-origin prediction
under observed clinical practice, not as validation of individual counterfactual
treatment effects under unobserved ICU interventions.

CausalLongPFN is intended for research and hypothesis generation, not as a
standalone clinical decision system.
