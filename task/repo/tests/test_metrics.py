from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from src.metrics import auroc


def test_auroc_agrees_with_reference() -> None:
    labels = np.array([0, 1, 0, 1, 1, 0])
    scores = np.array([0.1, 0.8, 0.4, 0.7, 0.4, 0.2])
    assert abs(auroc(labels, scores) - roc_auc_score(labels, scores)) < 1e-12
