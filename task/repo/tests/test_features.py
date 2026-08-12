from __future__ import annotations

import pandas as pd
from src.features import build_features


def test_feature_values_follow_measurement_time() -> None:
    admissions = pd.DataFrame(
        {"patient_id": [1], "admit_time": pd.to_datetime(["2025-01-01 00:00:00"])}
    )
    events = pd.DataFrame(
        {
            "patient_id": [1, 1],
            "lab": ["sodium", "sodium"],
            "charttime": pd.to_datetime(["2025-01-01 10:00:00", "2025-01-01 02:00:00"]),
            "value": [142.0, 138.0],
        }
    )
    config = {"study": {"observation_hours": 48, "labs": ["sodium"]}}
    result = build_features(events, admissions, config)
    assert result.loc[0, "sodium_first"] == 138.0
    assert result.loc[0, "sodium_last"] == 142.0
    assert result.loc[0, "sodium_count"] == 2


def test_feature_window_excludes_later_measurements() -> None:
    admissions = pd.DataFrame(
        {"patient_id": [1], "admit_time": pd.to_datetime(["2025-01-01 00:00:00"])}
    )
    events = pd.DataFrame(
        {
            "patient_id": [1, 1],
            "lab": ["sodium", "sodium"],
            "charttime": pd.to_datetime(["2025-01-01 01:00:00", "2025-01-04 00:00:00"]),
            "value": [139.0, 150.0],
        }
    )
    config = {"study": {"observation_hours": 48, "labs": ["sodium"]}}
    result = build_features(events, admissions, config)
    assert result.loc[0, "sodium_count"] == 1
