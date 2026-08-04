"""Clustering primitives for SANTA++ attention."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import torch

from .minibatch_kmeans import SklearnLikeTorchMiniBatchKMeans

from .config import MiniBatchClusterConfig, default_cluster_config


@dataclass
class ClusterSummary:
    """Compact decode metadata for one batch item and attention head."""

    members: torch.Tensor
    starts: torch.Tensor
    lengths_long: torch.Tensor
    lengths_float: torch.Tensor
    key_centroids: torch.Tensor


@torch.inference_mode()
def fit_default_clusters(
    features: torch.Tensor,
    n_clusters: int,
    *,
    cluster_config: MiniBatchClusterConfig = default_cluster_config,
) -> torch.Tensor:
    """Fit notebook-compatible MiniBatchKMeans to ``features=[N,F]``.

    This small solver boundary lets attention experiments change K or QK
    features independently from the clustering algorithm.
    """
    if features.ndim != 2:
        raise ValueError("Expected features=[N,F]")
    return SklearnLikeTorchMiniBatchKMeans(
        n_clusters=n_clusters, **asdict(cluster_config)
    ).fit_predict(features)


def build_clustering_features(
    old_keys: torch.Tensor,
    prefill_queries: torch.Tensor,
    *,
    feature_mode: str,
    scaling: float,
    probe_count: int,
    probe_start_fraction: float,
    seed: int,
) -> torch.Tensor:
    r"""Build the matrix passed to KMeans for one attention head.

    Inputs have shapes ``old_keys=[N,D]`` and
    ``prefill_queries=[T,D]``. In ``fingerprint`` mode the feature matrix is

    .. math:: F_{i,p}=\alpha k_i^Tq_p,

    standardized independently along every probe-query coordinate. In
    ``key`` mode the FP32 key matrix is returned directly. The KMeans solver
    is therefore independent from the choice of K or QK features.
    """
    if old_keys.ndim != 2 or prefill_queries.ndim != 2:
        raise ValueError("Expected old_keys=[N,D] and prefill_queries=[T,D]")
    old_keys = old_keys.float().contiguous()
    if feature_mode == "key":
        return old_keys
    if feature_mode != "fingerprint":
        raise ValueError(f"Unknown cluster feature mode: {feature_mode!r}")

    query_count = prefill_queries.shape[0]
    probe_start = int(probe_start_fraction * query_count)
    candidates = np.arange(probe_start, query_count)
    if candidates.size < probe_count:
        raise ValueError(
            f"Need {probe_count} probe queries, only {candidates.size} available"
        )
    probe_ids = np.sort(
        np.random.RandomState(seed).choice(candidates, probe_count, replace=False)
    )
    probe_ids = torch.as_tensor(
        probe_ids, dtype=torch.long, device=old_keys.device
    )
    features = old_keys @ prefill_queries[probe_ids].float().T * scaling
    means = features.mean(dim=0, keepdim=True)
    stds = features.std(dim=0, keepdim=True, unbiased=False)
    return ((features - means) / (stds + 1e-6)).contiguous()


def build_cluster_summary(
    old_keys: torch.Tensor,
    labels: torch.Tensor,
    n_clusters: int,
) -> ClusterSummary:
    """Convert labels into decode metadata.

    Inputs have shapes ``old_keys=[N,D]`` and ``labels=[N]``. Empty clusters
    are omitted, while ``members`` stores token indices grouped by label.
    """
    if old_keys.ndim != 2 or labels.ndim != 1:
        raise ValueError("Expected old_keys=[N,D] and labels=[N]")
    labels = labels.long()
    counts = torch.bincount(labels, minlength=n_clusters)
    active = counts > 0
    lengths_long = counts[active]
    lengths_float = lengths_long.float()
    members = torch.argsort(labels)
    starts = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=labels.device),
            torch.cumsum(counts[:-1], dim=0),
        )
    )[active]
    sums = torch.zeros(
        n_clusters,
        old_keys.shape[-1],
        dtype=torch.float32,
        device=old_keys.device,
    )
    sums.index_add_(0, labels, old_keys.float())
    return ClusterSummary(
        members=members,
        starts=starts,
        lengths_long=lengths_long,
        lengths_float=lengths_float,
        key_centroids=sums[active] / lengths_float[:, None],
    )


@torch.inference_mode()
def cluster_old_prefix(
    key_states: torch.Tensor,
    prefill_queries: torch.Tensor,
    old_end: int,
    *,
    feature_mode: str = "fingerprint",
    group_size: int = 16,
    scaling: float,
    probe_count: int = 64,
    probe_start_fraction: float = 0.75,
    seed: int | None = None,
    cluster_config: MiniBatchClusterConfig = default_cluster_config,
    kmeans_kwargs: dict[str, Any] | None = None,
) -> dict[tuple[int, int], ClusterSummary]:
    """Cluster an old KV prefix independently for every batch item and head.

    Inputs have shapes ``key_states=[B,H,K,D]`` and
    ``prefill_queries=[B,H,T,D]``. The notebook-compatible GPU
    MiniBatchKMeans implementation is always used. ``feature_mode`` only
    controls whether that solver receives raw K vectors or QK fingerprints.
    """
    if key_states.ndim != 4 or prefill_queries.ndim != 4:
        raise ValueError("Expected key_states and prefill_queries shaped [B,H,T,D]")
    if old_end <= 0:
        raise ValueError("old_end must be positive")
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    config = cluster_config if seed is None else replace(cluster_config, random_state=seed)
    options = asdict(config)
    options.update(kmeans_kwargs or {})
    feature_seed = config.random_state
    n_clusters = min(old_end, max(2, old_end // group_size))
    summaries: dict[tuple[int, int], ClusterSummary] = {}
    for batch_id in range(key_states.shape[0]):
        for head_id in range(key_states.shape[1]):
            old_keys = key_states[batch_id, head_id, :old_end]
            features = build_clustering_features(
                old_keys,
                prefill_queries[batch_id, head_id],
                feature_mode=feature_mode,
                scaling=scaling,
                probe_count=probe_count,
                probe_start_fraction=probe_start_fraction,
                seed=feature_seed,
            )
            labels = SklearnLikeTorchMiniBatchKMeans(
                n_clusters=n_clusters, **options
            ).fit_predict(features)
            summaries[(batch_id, head_id)] = build_cluster_summary(
                old_keys, labels, n_clusters
            )
    return summaries
