# CausalLongPFN

CausalLongPFN is a research codebase for longitudinal counterfactual outcome
prediction with prior-fitted networks. It contains:

- synthetic TSCM pretraining for the CausalLongPFN model;
- benchmark generators for cancer, HIV, warfarin, and MIMIC-III-style ICU data;
- PFN-ready file construction and frozen-checkpoint evaluation;
- dependency-light baseline adapters for MSM, RMSN, G-Net, CRN, Causal
  Transformer, and G-Transformer;
- result aggregation and PFN calibration summaries.

The code is organized around the benchmark protocol used in the paper:
queries observe a history through `t_obs`, receive planned future actions, and
predict the outcome at the requested target time. Cancer, HIV, and warfarin
provide branchable counterfactual labels. MIMIC-III is factual-only and is used
for rolling-origin prediction under observed future actions.

## Installation

```bash
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

The package exposes the following command-line entrypoints:

- `clpfn-generate-all`
- `clpfn-build-ready`
- `clpfn-train`
- `clpfn-eval`

## Repository Layout

```text
configs/
  data/          Dataset-generation configs
  train/         CausalLongPFN pretraining configs
  eval/          PFN and baseline evaluation configs, including baseline hyperparameters
src/clpfn/
  data/
    generators/  Cancer, HIV, warfarin, and MIMIC benchmark builders
    priors/      Synthetic TSCM episode generator for PFN pretraining
    formatters/  PFN-ready benchmark conversion
  models/        CausalLongPFN architecture
  training/      PFN loss, optimizer, checkpointing, and training loop
  evaluation/    Shared benchmark evaluation logic and PFN rollout/calibration
  baselines/     Baseline adapters and G-Transformer-family model ports
  cli/           Command-line entrypoints
```

## Benchmark Generation

Generate all configured benchmark datasets:

```bash
clpfn-generate-all --config configs/data/all_benchmarks.yaml
```

Generate selected domains:

```bash
clpfn-generate-all --config configs/data/all_benchmarks.yaml --only cancer
clpfn-generate-all --config configs/data/all_benchmarks.yaml --only hiv warfarin
```

Generated raw benchmark pickles use the canonical CLPFN benchmark schema:

```text
support_data
test_data
test_data_factuals
test_data_seq
scaling_data
```

Each split stores the shared arrays used by PFN and baseline evaluators:

```text
states, outcomes, actions, sequence_lengths, static_features
```

MIMIC-III data are not redistributed. To reproduce the MIMIC benchmark, place
the credentialed-access MIMIC-Extract HDF5 file under the configured
`data/raw/...` location and run the MIMIC generator. See [DATA.md](DATA.md) for
data and artifact handling details.

## CausalLongPFN Training

Train the PFN from synthetic TSCM episodes:

```bash
clpfn-train --config configs/train/causal_long_pfn.yaml
```

The training code samples synthetic episodes online from the TSCM prior in
`clpfn.data.priors`, optimizes the Gaussian-mixture predictive objective, and
writes checkpoints to the configured output directory.

## PFN Evaluation

Build PFN-ready files from raw benchmark pickles:

```bash
clpfn-build-ready --config configs/eval/pfn.yaml
```

Evaluate a frozen checkpoint:

```bash
clpfn-eval \
  --method pfn \
  --config configs/eval/pfn.yaml \
  --checkpoint /path/to/checkpoint.pt
```

PFN evaluation saves:

- `prediction_rows.parquet`
- `domain_task_normalized_rmse.csv`
- `calibration_summary_domain.csv` when support-sigma calibration is enabled

## Baseline Evaluation

Evaluate a baseline through the same benchmark evaluation entrypoint:

```bash
clpfn-eval --method gtransformer --config configs/eval/gtransformer.yaml
clpfn-eval --method ct --config configs/eval/ct.yaml
clpfn-eval --method crn --config configs/eval/crn.yaml
clpfn-eval --method gnet --config configs/eval/gnet.yaml
clpfn-eval --method rmsn --config configs/eval/rmsn.yaml
clpfn-eval --method msm --config configs/eval/msm.yaml
```

The baseline model code in `clpfn.baselines.models` keeps one file per model
family and shared LSTM/transformer utilities. The per-method
packages own CLPFN-specific support-set construction, tuning, training, and benchmark
rollout prediction. See [THIRD_PARTY.md](THIRD_PARTY.md) for attribution notes.

The adapters intentionally keep the benchmark IO contract separate from the
external training stack:

- no PyTorch Lightning, Ray Tune, MLflow, EMA weights, or two-optimizer domain
  confusion path;
- grouped support-set tuning by `domain + support_size`;
- predictions reported in the CLPFN benchmark normalized raw-pickle convention;
- normalized RMSE computed against `clip((target_raw - eval_out_mean) / eval_out_std, -10, 10)`,
  with reported predictions clipped to `[-20, 20]` for both PFN and baselines;
- PFN-ready tensors keep the PFN context-normalized input convention, then PFN
  predictions are converted to the shared evaluation normalization before scoring;
- explicit rollout adapters using `t_obs` as the last visible index.

## Batch-Oriented Workflow

The command-line entrypoints are designed to run independently, so dataset
generation, checkpointed PFN training, baseline evaluation, and PFN evaluation
can be split across separate compute sessions.

Generate one benchmark domain at a time:

```bash
clpfn-generate-all --config configs/data/all_benchmarks.yaml --only cancer
clpfn-generate-all --config configs/data/all_benchmarks.yaml --only hiv
clpfn-generate-all --config configs/data/all_benchmarks.yaml --only warfarin
clpfn-generate-all --config configs/data/all_benchmarks.yaml --only mimic
```

Continue PFN pretraining from an existing checkpoint directory:

```bash
clpfn-train \
  --config configs/train/causal_long_pfn.yaml \
  --ckpt-input-dir /path/to/previous/checkpoints \
  --output-dir outputs/causal_long_pfn_outputs
```

Each evaluation command writes complete row-level predictions to
`prediction_rows.parquet` under its configured `evaluation.output_dir`. These
files share the same schema for PFN and baselines and can be pooled later for
downstream analysis.

## Config-First Workflows

Evaluation is config-first. Paths, domains, tuning settings, baseline
hyperparameter spaces, PFN batch size, support-sigma calibration, and output
directories should be set in YAML. For example:

```yaml
method: gtransformer

data:
  wanted_domains: [cancer, hiv, warfarin, mimic]
  raw_inputs:
    pickle_dirs:
      - outputs/data/cancer
      - outputs/data/hiv
      - outputs/data/warfarin
      - outputs/data/mimic

evaluation:
  output_dir: outputs/eval/gtransformer

baseline:
  # The checked-in eval configs include the complete method-specific block.
  limits:
    max_val_origins: 256
    projection_horizon: 0
  default_hparams: { ... }
  search_space: { ... }
```

Named directories are searched recursively. Explicit `pickle_paths` are also
accepted. The evaluation code does not perform implicit filesystem scans outside
the configured paths and does not automatically extract zip archives.

## Public Artifacts

Do not commit raw datasets, credentialed MIMIC files, model checkpoints,
generated ready files, or generated result artifacts. Keep large artifacts in
external storage or attach them to releases only when redistribution is
appropriate.

The `.gitignore` is configured for the expected local outputs:

- `outputs/`
- `checkpoints/`
- `data/raw/`
- `data/processed/`
- model checkpoint extensions such as `.pt`, `.ckpt`, and `.pth`
- generated pickle, parquet, zip, and CSV files

## Notes on Scope

CausalLongPFN is a research tool. Counterfactual interpretation on real
observational data still requires the usual longitudinal assumptions:
consistency, positivity, and sequential exchangeability given the measured
history. MIMIC-III results in this repository are factual rolling-origin
prediction results, not validation of individual treatment effects under
unobserved ICU interventions.
