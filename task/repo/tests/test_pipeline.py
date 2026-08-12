from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from src.pipeline import build_folds


def test_fold_assignments_cover_patients_once() -> None:
    cohort = pd.DataFrame(
        {
            "patient_id": np.arange(40),
            "site": ["A", "B"] * 20,
            "readmitted_30d": [0, 0, 1, 1] * 10,
        }
    )
    config = {"evaluation": {"folds": 5, "fold_seed": 17}}
    assignments = build_folds(cohort, config)
    assert set(assignments) == {str(value) for value in range(40)}
    assert set(assignments.values()) == set(range(5))


def test_imputer_uses_training_values() -> None:
    training = np.array([[1.0], [3.0], [5.0]])
    transformed = SimpleImputer(strategy="median").fit(training).transform([[np.nan]])
    assert transformed[0, 0] == 3.0
