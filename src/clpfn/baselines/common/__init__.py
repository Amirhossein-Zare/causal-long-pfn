"""Shared baseline infrastructure.

Module ownership:

- ``api``: adapter dataclasses, method-config loading, and small adapter helpers.
- ``runner``: shared evaluation loop for all baseline adapters.
- ``tuning``: grouped support-set hyperparameter selection.
- ``training``: explicit PyTorch training loop and support-validation RMSE.

Model implementations live in ``clpfn.baselines.models``.
"""
