# Third-Party Code and Method References

This repository includes dependency-light baseline implementations adapted to
the CLPFN benchmark protocol.

## G-Transformer-Family Baselines

The baseline model modules in `src/clpfn/baselines/models` keep one file per
model family with shared recurrent and transformer utilities. The CLPFN adapters
replace the original training stack with explicit PyTorch loops and the unified
benchmark raw-pickle input contract.

The adapted baseline families are:

- MSM
- RMSN
- G-Net
- CRN
- Causal Transformer
- G-Transformer

When publishing or redistributing this repository, keep upstream attribution and
license requirements for the upstream implementations and any simulator code
that informed these ports.

## Data Sources

MIMIC-III and MIMIC-Extract-style assets are not redistributed. Users must obtain
their own access through the official data-use process. See `DATA.md`.
