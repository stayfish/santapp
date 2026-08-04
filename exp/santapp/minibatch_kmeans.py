"""CUDA MiniBatchKMeans with scikit-learn-like algorithmic semantics.

Adapted from the supplied SANTA++ notebook. It intentionally mirrors the
notebook's MiniBatchKMeans choices rather than replacing them with full-batch
Lloyd k-means.
"""

from __future__ import annotations

import numpy as np
import torch


# ---- GPU MiniBatchKMeans with scikit-learn-like semantics ----
# This is intentionally not ordinary Lloyd KMeans. It mirrors the algorithm
# used by:
#   MiniBatchKMeans(n_clusters=..., batch_size=4096, n_init=1,
#                   random_state=SEED)
#
# Fidelity choices:
#   * NumPy RandomState controls all stochastic choices, as in scikit-learn.
#   * greedy k-means++ (2 + floor(log(K)) local trials), not vanilla k-means++.
#   * sampling with replacement for each mini-batch.
#   * cumulative-count online means.
#   * scikit-learn's low-count center reassignment and EWA early stopping.
#   * float32 data/centers throughout.
#
# Exact bitwise identity is not guaranteed because GPU and CPU reductions can
# round differently, but on ordinary float32 tests this follows the same path
# extremely closely and can produce identical partitions.

class SklearnLikeTorchMiniBatchKMeans:
    def __init__(
        self,
        n_clusters,
        *,
        batch_size=4096,
        n_init=1,
        max_iter=100,
        tol=0.0,
        max_no_improvement=10,
        init_size=None,
        reassignment_ratio=0.01,
        random_state=0,
        verbose=False,
    ):
        self.n_clusters = int(n_clusters)
        self.batch_size = int(batch_size)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.max_no_improvement = max_no_improvement
        self.init_size = init_size
        self.reassignment_ratio = float(reassignment_ratio)
        self.random_state = random_state
        self.verbose = bool(verbose)

    @staticmethod
    def _squared_distances(x, centers):
        """Squared Euclidean distances, [N,D] x [K,D] -> [N,K]."""
        out = (
            x.square().sum(dim=1, keepdim=True)
            + centers.square().sum(dim=1).unsqueeze(0)
        )
        out.addmm_(x, centers.T, beta=1.0, alpha=-2.0)
        return out.clamp_min_(0.0)

    def _assign(self, x, centers):
        dist2 = self._squared_distances(x, centers)
        min_dist2, labels = dist2.min(dim=1)
        return labels, min_dist2.sum()

    def _greedy_kmeans_plus_plus(self, x, rng):
        """GPU version of scikit-learn's greedy k-means++ initializer."""
        n_samples, n_features = x.shape
        n_local_trials = 2 + int(np.log(self.n_clusters))

        # scikit-learn receives float32 unit sample weights for float32 X.
        p = np.ones(n_samples, dtype=np.float32)
        p /= p.sum(dtype=np.float32)
        first_id = int(rng.choice(n_samples, p=p))

        centers = torch.empty(
            self.n_clusters,
            n_features,
            dtype=x.dtype,
            device=x.device,
        )
        centers[0] = x[first_id]

        closest_dist2 = self._squared_distances(
            x, centers[:1]
        ).squeeze(1)
        current_potential = closest_dist2.sum()

        for center_idx in range(1, self.n_clusters):
            # Generate the same kind and number of NumPy random variates as
            # scikit-learn, while doing search/distances on the GPU.
            uniforms = torch.as_tensor(
                rng.uniform(size=n_local_trials),
                dtype=torch.float64,
                device=x.device,
            )
            candidate_values = (
                uniforms * current_potential.to(torch.float64)
            )
            cumulative = torch.cumsum(closest_dist2, dim=0)
            candidate_ids = torch.searchsorted(
                cumulative, candidate_values
            ).clamp_max_(n_samples - 1)

            candidate_points = x[candidate_ids]
            candidate_dist2 = self._squared_distances(
                candidate_points, x
            )
            candidate_dist2 = torch.minimum(
                candidate_dist2, closest_dist2.unsqueeze(0)
            )
            candidate_potentials = candidate_dist2.sum(dim=1)
            best = int(candidate_potentials.argmin().item())

            centers[center_idx] = candidate_points[best]
            closest_dist2 = candidate_dist2[best]
            current_potential = candidate_potentials[best]

        return centers

    @torch.inference_mode()
    def fit_predict(self, x):
        if x.ndim != 2:
            raise ValueError(f"Expected [samples, features], got {tuple(x.shape)}")
        if x.dtype not in (torch.float32, torch.float64):
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        n_samples = x.shape[0]
        if n_samples < self.n_clusters:
            raise ValueError(
                f"n_samples={n_samples} must be >= n_clusters={self.n_clusters}"
            )

        batch_size = min(self.batch_size, n_samples)
        init_size = self.init_size
        if init_size is None:
            init_size = 3 * batch_size
            if init_size < self.n_clusters:
                init_size = 3 * self.n_clusters
        elif init_size < self.n_clusters:
            init_size = 3 * self.n_clusters
        init_size = min(int(init_size), n_samples)

        rng = np.random.RandomState(self.random_state)

        # scikit-learn draws this validation sample before initialization.
        validation_ids_np = rng.randint(0, n_samples, init_size)
        validation_ids = torch.as_tensor(
            validation_ids_np, dtype=torch.long, device=x.device
        )
        x_valid = x[validation_ids]

        best_centers = None
        best_validation_inertia = None

        for init_idx in range(self.n_init):
            if init_size < n_samples:
                init_ids_np = rng.randint(0, n_samples, init_size)
                init_ids = torch.as_tensor(
                    init_ids_np, dtype=torch.long, device=x.device
                )
                x_init = x[init_ids]
            else:
                x_init = x

            centers = self._greedy_kmeans_plus_plus(x_init, rng)

            # With n_init=1, validation cannot change the selected centers.
            # Skip its expensive assignment while preserving identical output.
            if self.n_init == 1:
                best_centers = centers
            else:
                _, validation_inertia = self._assign(x_valid, centers)
                validation_inertia = float(validation_inertia.item())
                if (
                    best_validation_inertia is None
                    or validation_inertia < best_validation_inertia
                ):
                    best_validation_inertia = validation_inertia
                    best_centers = centers.clone()

        centers = best_centers
        counts = torch.zeros(
            self.n_clusters, dtype=x.dtype, device=x.device
        )

        # scikit-learn scales tol by the mean per-feature variance.
        scaled_tol = (
            float(x.var(dim=0, unbiased=False).mean().item()) * self.tol
            if self.tol > 0.0
            else 0.0
        )

        n_steps = (self.max_iter * n_samples) // batch_size
        ewa_inertia = None
        ewa_inertia_min = None
        no_improvement = 0
        n_since_last_reassign = 0

        for step in range(n_steps):
            n_since_last_reassign += batch_size
            random_reassign = bool(
                (counts == 0).any().item()
                or n_since_last_reassign >= 10 * self.n_clusters
            )
            if random_reassign:
                n_since_last_reassign = 0

            batch_ids_np = rng.randint(0, n_samples, batch_size)
            batch_ids = torch.as_tensor(
                batch_ids_np, dtype=torch.long, device=x.device
            )
            x_batch = x[batch_ids]

            labels, batch_inertia = self._assign(x_batch, centers)

            batch_counts = torch.bincount(
                labels, minlength=self.n_clusters
            ).to(x.dtype)
            batch_sums = torch.zeros_like(centers)
            batch_sums.index_add_(0, labels, x_batch)

            new_counts = counts + batch_counts
            centers_new = centers.clone()
            active = batch_counts > 0
            centers_new[active] = (
                centers[active] * counts[active, None]
                + batch_sums[active]
            ) / new_counts[active, None]
            counts = new_counts

            if random_reassign and self.reassignment_ratio > 0.0:
                to_reassign = (
                    counts
                    < self.reassignment_ratio * counts.max()
                )

                # Same cap as scikit-learn: at most half a mini-batch.
                if int(to_reassign.sum().item()) > 0.5 * x_batch.shape[0]:
                    keep_ids = torch.argsort(counts)[
                        int(0.5 * x_batch.shape[0]):
                    ]
                    to_reassign[keep_ids] = False

                n_reassign = int(to_reassign.sum().item())
                if n_reassign:
                    new_center_rows_np = rng.choice(
                        x_batch.shape[0],
                        replace=False,
                        size=n_reassign,
                    )
                    new_center_rows = torch.as_tensor(
                        new_center_rows_np,
                        dtype=torch.long,
                        device=x.device,
                    )
                    centers_new[to_reassign] = x_batch[new_center_rows]
                    counts[to_reassign] = counts[~to_reassign].min()

            if scaled_tol > 0.0:
                centers_squared_diff = float(
                    (centers_new - centers).square().sum().item()
                )
            else:
                centers_squared_diff = 0.0

            centers = centers_new

            # scikit-learn ignores convergence on the first update.
            if step == 0:
                continue

            mean_batch_inertia = (
                float(batch_inertia.item()) / batch_size
            )
            if ewa_inertia is None:
                ewa_inertia = mean_batch_inertia
            else:
                alpha = min(
                    batch_size * 2.0 / (n_samples + 1), 1.0
                )
                ewa_inertia = (
                    ewa_inertia * (1.0 - alpha)
                    + mean_batch_inertia * alpha
                )

            if (
                scaled_tol > 0.0
                and centers_squared_diff <= scaled_tol
            ):
                break

            if (
                ewa_inertia_min is None
                or ewa_inertia < ewa_inertia_min
            ):
                ewa_inertia_min = ewa_inertia
                no_improvement = 0
            else:
                no_improvement += 1

            if (
                self.max_no_improvement is not None
                and no_improvement >= self.max_no_improvement
            ):
                break

        self.cluster_centers_ = centers
        self.counts_ = counts
        self.n_steps_ = step + 1
        self.n_iter_ = int(
            np.ceil(self.n_steps_ * batch_size / n_samples)
        )

        labels, inertia = self._assign(x, centers)
        self.labels_ = labels
        self.inertia_ = float(inertia.item())

        if self.verbose:
            print(
                f"GPU MiniBatchKMeans: {self.n_steps_} mini-batches, "
                f"{self.n_iter_} effective epochs"
            )

        return labels


def gpu_minibatch_labels(x, n_clusters, seed):
    """Exact parameter match to the original notebook's sklearn call."""
    return SklearnLikeTorchMiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=4096,
        n_init=1,
        max_iter=100,
        tol=0.0,
        max_no_improvement=10,
        init_size=None,
        reassignment_ratio=0.01,
        random_state=seed,
    ).fit_predict(x)
