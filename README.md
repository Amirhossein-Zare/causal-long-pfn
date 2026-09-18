# CausalLongPFN

CausalLongPFN predicts counterfactual longitudinal outcomes from support
trajectories, query history, and planned treatments. The frozen model is
pretrained on synthetic temporal structural causal models and returns a
five-component Gaussian-mixture distribution.

## Repository

```text
configs/data/              benchmark generation
configs/train/             PFN training and canonical ablation reference
configs/train/ablations/   seven pretraining ablations
configs/eval/              PFN and baseline evaluation
src/clpfn/data/            generators and PFN-ready formatting
src/clpfn/models/          CausalLongPFN architecture
src/clpfn/training/        training, losses, optimization, and checkpoints
src/clpfn/evaluation/      shared evaluation and PFN evaluation
src/clpfn/baselines/       CT, G-Net, MSM, and RMSN
```

The main model uses four causal history layers, four PFN attention layers,
width 256, eight heads, four support anchors per trajectory, and a
five-component Gaussian-mixture head. The seven ablations are:

Prior-mechanism ablations, which change what the sampled temporal structural
causal model contains:

- `no_motifs`
- `no_latent_heterogeneity`
- `no_confounding`
- `immediate_effects_only`

Supervision-mixture ablations, which change how the query target is
constructed rather than what the sampled model contains. The canonical prior
draws a factual query with probability 0.50 and an interventional
structural-replay query otherwise; these two variants pin that mixture to its
endpoints:

- `factual_pretraining_only` (`OBSERVATIONAL_QUERY_PROB = 1.0`): the model
  never sees a counterfactual target during pretraining.
- `counterfactual_pretraining_only` (`OBSERVATIONAL_QUERY_PROB = 0.0`): every
  query target is an interventional structural replay.

In-context interface ablation:

- `single_anchor_support` (`N_SUPPORT_ANCHORS = 1`): one labeled support anchor
  per support trajectory instead of the canonical four.

Each ablation changes one pretraining mechanism while retaining the model
architecture. The 2,500-step `canonical_reference` configuration is the matched
comparator for all seven ablations.

Benchmark ready files are always built with the canonical four support anchors.
A checkpoint pretrained with fewer anchors reads the leading slice of the built
anchors at evaluation time, so `single_anchor_support` needs no benchmark
rebuild. Anchor index 0 is the latest valid anchor in both the ready builder and
the synthetic generator, so the leading slice is the matching interface. A
checkpoint requesting more anchors than were built is rejected.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Pretrained weights

The seed-42 main checkpoint is released as safetensors on Hugging Face:
<https://huggingface.co/Amirhossein-Zare/causal-long-pfn>

`--checkpoint` accepts a training `.pt` file, a released `.safetensors` file, or a
directory containing one.

## Run

Generate benchmark data:

```bash
clpfn-generate-all --config configs/data/all_benchmarks.yaml
```

Train the main model or an ablation:

```bash
clpfn-train --config configs/train/causal_long_pfn.yaml
clpfn-train --config configs/train/canonical_reference.yaml
clpfn-train --config configs/train/ablations/no_motifs.yaml
```

Build PFN-ready inputs and evaluate a checkpoint:

```bash
clpfn-build-ready --config configs/eval/pfn.yaml
clpfn-eval --method pfn --config configs/eval/pfn.yaml --checkpoint PATH_TO_CHECKPOINT
```

Baseline tuning and evaluation are separate operations. Tuning writes atomic
trial and dataset-selection state plus consolidated trial and
selected-hyperparameter Parquet files. Evaluation requires that manifest.

```bash
clpfn-eval --method rmsn --config configs/eval/rmsn.yaml --mode tune
clpfn-eval --method rmsn --config configs/eval/rmsn.yaml --mode evaluate
```
