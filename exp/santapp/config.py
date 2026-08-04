"""Clustering hyperparameter presets used by the SANTA++ experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MiniBatchClusterConfig:
    """Hyperparameters from ``mla_santapp.ipynb``.

    This is the default solver configuration used by :mod:`santapp`.
    """

    batch_size: int = 4096
    n_init: int = 1
    max_iter: int = 100
    tol: float = 0.0
    max_no_improvement: int | None = 10
    init_size: int | None = None
    reassignment_ratio: float = 0.01
    random_state: int = 0


@dataclass(frozen=True)
class LloydClusterConfig:
    """Hyperparameters retained from ``exp/mla_santapp.py`` for comparison."""

    iterations: int = 20
    seed: int = 0


default_cluster_config = MiniBatchClusterConfig()
"""Default notebook-compatible MiniBatchKMeans configuration."""

my_cluster_config = LloydClusterConfig()
"""Legacy full-batch Lloyd KMeans configuration; not used by default."""
