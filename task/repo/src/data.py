from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def read_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError("configuration must be a mapping")
    return value


def load_records(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = root / "data" / "raw"
    events = pd.read_csv(raw / "events.csv", parse_dates=["charttime"])
    admissions = pd.read_csv(
        raw / "admissions.csv",
        parse_dates=["admit_time", "discharge_time", "next_admit_time"],
    )
    patients = pd.read_csv(raw / "patients.csv")
    return events, admissions, patients


def attach_readmission_label(
    admissions: pd.DataFrame, readmission_days: int
) -> pd.DataFrame:
    frame = admissions.copy()
    delta = frame["next_admit_time"] - frame["discharge_time"]
    frame["readmitted_30d"] = (
        frame["next_admit_time"].notna()
        & delta.ge(pd.Timedelta(0))
        & delta.le(pd.Timedelta(days=readmission_days))
    ).astype("int8")
    return frame
