# CausalLongPFN

[![arXiv](https://img.shields.io/badge/arXiv-2606.05797-b31b1b.svg)](https://arxiv.org/abs/2606.05797)
[![arXiv DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2606.05797-blue.svg)](https://doi.org/10.48550/arXiv.2606.05797)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow.svg)](https://huggingface.co/Amirhossein-Zare/causal-long-pfn)
[![Zenodo DOI](https://zenodo.org/badge/1247364520.svg)](https://doi.org/10.5281/zenodo.20588199)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://www.python.org/)

**CausalLongPFN** is the reference implementation for *Causal Longitudinal
Prior-Fitted Networks for Counterfactual Outcome Prediction*.

We introduce **Causal Longitudinal Prior-Fitted Networks
(CausalLongPFN)**, a prior-fitted network for **time-series causal inference in
longitudinal treatment-response data** and **zero-shot in-context
counterfactual outcome prediction**. The project studies history-conditional
potential-outcome prediction from longitudinal treatment-response time series:
given support trajectories from a new domain, a query history, and a planned
future treatment sequence, a frozen CausalLongPFN model predicts a distribution
over future outcomes without target-domain gradient updates, propensity-model
fitting, or adversarial balancing. The model is pretrained on synthetic temporal
structural causal models (TSCMs) and evaluated on branchable cancer, HIV, and
warfarin counterfactual benchmarks, as well as factual MIMIC-III ICU
rolling-origin prediction.

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

- synthetic CausalLongPFN pretraining from a temporal structural causal model
  prior;
- benchmark generation for cancer, HIV, warfarin, and MIMIC-III-style ICU
  treatment-response tasks;
- conversion of benchmark files into CausalLongPFN-ready support/query datasets;
- zero-shot in-context evaluation of the frozen CausalLongPFN model;
- baseline evaluation for MSM, RMSN, G-Net, CRN, Causal Transformer, and
  G-Transformer;
- normalized RMSE summaries, row-level prediction outputs, and one-step
  probabilistic calibration diagnostics.


## Repository layout

```text
configs/
  data/                     Benchmark-generation configs
  train/                    CausalLongPFN synthetic-pretraining configs
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
  baselines/                MSM, RMSN, G-Net, CRN, CT, and G-Transformer adapters
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

Pretrained inference-only weights are available on Hugging Face:

<https://huggingface.co/Amirhossein-Zare/causal-long-pfn>

```bash
pip install safetensors huggingface_hub
```

```python
from huggingface_hub import hf_hub_download

weights_path = hf_hub_download(
    repo_id="Amirhossein-Zare/causal-long-pfn",
    filename="causal-long-pfn-v1-step10000.safetensors",
)
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
outputs/data/cancer
outputs/data/hiv
outputs/data/warfarin
outputs/data/mimic
```

### 2. Train CausalLongPFN

```bash
clpfn-train --config configs/train/causal_long_pfn.yaml
```

The default training config uses synthetic TSCM episodes generated on the fly.
Checkpoints are written to:

```text
outputs/causal_long_pfn_outputs
```

To resume from an existing checkpoint directory or write to a different output
directory:

```bash
clpfn-train \
  --config configs/train/causal_long_pfn.yaml \
  --ckpt-input-dir outputs/causal_long_pfn_outputs \
  --output-dir outputs/causal_long_pfn_outputs
```

### 3. Build PFN-ready evaluation files

CausalLongPFN evaluation uses compact support/query files built from the raw
benchmark pickles:

```bash
clpfn-build-ready --config configs/eval/pfn.yaml
```

The default output directory is:

```text
outputs/pfn_ready/all_domains
```

### 4. Evaluate CausalLongPFN

```bash
clpfn-eval \
  --method pfn \
  --config configs/eval/pfn.yaml \
  --checkpoint outputs/causal_long_pfn_outputs/ckpt_final.pt
```

Evaluation outputs are written under the configured `evaluation.output_dir`, by
default:

```text
outputs/eval/causal_long_pfn
```

The main files are:

```text
prediction_rows.parquet
  Row-level predictions and errors.

domain_task_normalized_rmse.csv
  Mean and standard deviation of normalized RMSE by domain, task, and method.

calibration_summary_domain.csv
  One-step probabilistic calibration summary by domain, when enabled.

calibration_rows.parquet
  Row-level one-step calibration diagnostics, when enabled.
```

### 5. Evaluate baselines

Each baseline has its own config under `configs/eval/`:

```bash
clpfn-eval --method msm          --config configs/eval/msm.yaml
clpfn-eval --method rmsn         --config configs/eval/rmsn.yaml
clpfn-eval --method gnet         --config configs/eval/gnet.yaml
clpfn-eval --method crn          --config configs/eval/crn.yaml
clpfn-eval --method ct           --config configs/eval/ct.yaml
clpfn-eval --method gtransformer --config configs/eval/gtransformer.yaml
```

Baselines are trained and selected on the target support data according to their
configuration. CausalLongPFN is evaluated frozen, using the support trajectories
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
- `configs/train/causal_long_pfn.yaml` sets model size, optimizer parameters,
  synthetic-prior ranges, checkpoint cadence, and runtime output directories.
- `configs/eval/pfn.yaml` sets PFN-ready input directories, checkpoint path,
  batch size, calibration reporting, and output directory.
- `configs/eval/*.yaml` set baseline-specific hyperparameter search spaces,
  limits, and output directories.

The default benchmark grid uses support sizes `40, 80, 160, 320, 500`, ten
confounding/task-index levels, two repetitions per cell, sequence length `60`,
and horizon `5`.

## Reported results

In the paper, a single frozen CausalLongPFN model is compared with baselines that
are trained and selected separately for each target domain. The reported model
achieves the best domain-balanced one-step normalized RMSE and the third-best
domain-balanced five-step normalized RMSE. It also ranks first on factual
MIMIC-III rolling-origin prediction at both horizons, where evaluation is under
observed future treatment paths rather than unobserved counterfactual
interventions.

For exact numbers and experimental details, see the paper.

## Causal scope and limitations

CausalLongPFN is a research tool for time-series causal inference in
longitudinal treatment-response data, causal sequence modeling, and
history-conditional counterfactual outcome prediction. It does not remove the
assumptions required for causal interpretation of observational data.
Counterfactual interpretation still requires, among other conditions,
consistency, positivity, sequential exchangeability given the measured history,
adequate treatment overlap, and well-defined interventions.

Performance also depends on support from the synthetic TSCM prior.
Predictions may be unreliable when the target domain contains mechanisms,
missingness patterns, treatment policies, outcome dynamics, or intervention
effects that are poorly represented by the pretraining prior. The current
implementation focuses on discrete treatments, fixed time grids, and
deterministic mean rollout.

MIMIC-III results should be interpreted as factual rolling-origin prediction
under observed clinical practice, not as validation of individual counterfactual
treatment effects under unobserved ICU interventions.

This repository is not intended for clinical decision-making.
