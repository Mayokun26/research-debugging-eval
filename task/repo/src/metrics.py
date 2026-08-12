from __future__ import annotations

import numpy as np


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(y_true, dtype=int)
    values = np.asarray(scores, dtype=float)
    positive = labels == 1
    n_positive = int(positive.sum())
    n_negative = int((~positive).sum())
    if n_positive == 0 or n_negative == 0:
        raise ValueError("both outcome classes are required")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


def paired_bootstrap_ci(
    y_true: np.ndarray,
    gbm_scores: np.ndarray,
    lr_scores: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    labels = np.asarray(y_true, dtype=int)
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, labels.size, labels.size)
        sampled = labels[indices]
        if np.unique(sampled).size != 2:
            continue
        differences.append(auroc(sampled, gbm_scores[indices]) - auroc(sampled, lr_scores[indices]))
    low, high = np.quantile(np.asarray(differences), [0.025, 0.975])
    return float(low), float(high)
