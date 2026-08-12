from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

TASK_ROOT = Path(__file__).resolve().parents[1]
REPO = TASK_ROOT / "repo"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data import attach_readmission_label, load_records, read_config
from src.features import _artifact_key, build_features


def feature_evidence(repo: Path = REPO) -> tuple[dict[str, Any], pd.DataFrame]:
    config = read_config(repo / "config.yaml")
    events, admissions, patients = load_records(repo)
    admissions = attach_readmission_label(admissions, int(config["study"]["readmission_days"]))
    committed = pd.read_parquet(repo / "data" / "derived" / "features.parquet")
    expected = build_features(events, admissions, config)
    committed = committed.sort_values("patient_id").reset_index(drop=True)
    expected = expected.sort_values("patient_id").reset_index(drop=True)
    changed = [
        column
        for column in committed.columns
        if column != "patient_id" and not committed[column].equals(expected[column])
    ]
    row_changed = (committed[changed] != expected[changed]).any(axis=1)
    patient_ids = committed.loc[row_changed, "patient_id"].astype(int).tolist()
    patient_sites = patients.set_index("patient_id").loc[patient_ids, "site"]
    aggregate_columns = [
        column
        for column in committed.columns
        if column != "patient_id" and not column.endswith(("_first", "_last"))
    ]
    metadata = pq.read_metadata(repo / "data" / "derived" / "features.parquet").metadata or {}
    current_key = _artifact_key(build_features, config)

    positional = events.groupby(["patient_id", "lab"], sort=True, observed=True)["value"].last()
    matches = True
    for patient_id in patient_ids:
        for lab in config["study"]["labs"]:
            observed = committed.loc[
                committed["patient_id"] == patient_id, f"{lab}_last"
            ].iloc[0]
            if observed != positional.loc[(patient_id, lab)]:
                matches = False
                break

    evidence = {
        "affected_columns": sorted(changed),
        "affected_patient_ids": patient_ids,
        "affected_site": "B" if set(patient_sites) == {"B"} else "mixed",
        "aggregate_columns_identical": all(
            committed[column].equals(expected[column]) for column in aggregate_columns
        ),
        "artifact_key_matches_current": metadata.get(b"featurizer_sha", b"").decode()
        == current_key,
        "file_position_matches_committed": matches,
    }
    return evidence, expected
