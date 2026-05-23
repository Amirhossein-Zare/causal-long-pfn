# Data and Generated Artifacts

This repository contains code for constructing the benchmark files used by
CausalLongPFN. It does not include raw clinical data, generated benchmark
pickles, PFN-ready files, checkpoints, or generated result artifacts.

## Benchmark Domains

- **Cancer**: simulated tumor-growth trajectories with branchable
  counterfactual labels.
- **HIV**: Adams/WhyNot-style ODE trajectories with branchable counterfactual
  labels.
- **Warfarin**: semi-mechanistic PK/PD trajectories with branchable
  counterfactual labels.
- **MIMIC-III**: factual ICU rolling-origin prediction under observed future
  actions. MIMIC-III rows are not counterfactual treatment-effect labels.

## MIMIC-III

MIMIC-III is a credentialed-access clinical database and is not redistributed
with this repository. To reproduce the MIMIC benchmark, obtain access through
the official MIMIC data-use process and place the MIMIC-Extract-style
`all_hourly_data.h5` file under the configured `data/raw/...` location.

The MIMIC generator searches for `all_hourly_data.h5` under
`MIMICGeneratorConfig.input_root` and prefers paths containing the configured
`merged_dataset_slug`.

## Local Artifacts

The following files are generated locally and should not be committed:

- raw benchmark pickles under `outputs/data/...`;
- PFN-ready files under configured ready directories;
- model checkpoints and training outputs;
- evaluation CSV summaries;
- row-level prediction Parquet files named `prediction_rows.parquet`;
- credentialed clinical data under `data/raw/...`.

The `.gitignore` file excludes these artifact classes by default.
