from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _prepare_events(
    events: pd.DataFrame, admissions: pd.DataFrame, observation_hours: int
) -> pd.DataFrame:
    prepared = events.merge(admissions[["patient_id", "admit_time"]], on="patient_id")
    elapsed = (prepared["charttime"] - prepared["admit_time"]).dt.total_seconds() / 3600
    prepared = prepared.loc[elapsed.between(0, observation_hours)].copy()
    prepared["elapsed_hours"] = elapsed.loc[prepared.index]
    return prepared.sort_values(
        ["patient_id", "lab", "charttime"], kind="mergesort"
    ).reset_index(drop=True)


def _slope(group: pd.DataFrame) -> float:
    ordered = group.sort_values("elapsed_hours", kind="mergesort")
    x = ordered["elapsed_hours"].to_numpy(dtype=float)
    y = ordered["value"].to_numpy(dtype=float)
    denominator = float(np.sum((x - x.mean()) ** 2))
    return 0.0 if denominator == 0 else float(np.sum((x - x.mean()) * (y - y.mean())) / denominator)


def build_features(
    events: pd.DataFrame, admissions: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    prepared = _prepare_events(events, admissions, int(config["study"]["observation_hours"]))
    grouped = prepared.groupby(["patient_id", "lab"], sort=True, observed=True)
    values = grouped["value"].agg(["first", "last"])
    stable = prepared.sort_values(
        ["patient_id", "lab", "charttime"], kind="mergesort"
    ).reset_index(drop=True)
    stable_grouped = stable.groupby(["patient_id", "lab"], sort=True, observed=True)
    aggregates = stable_grouped["value"].agg(["mean", "min", "max", "count"])
    aggregates["slope"] = stable_grouped.apply(_slope, include_groups=False)
    values = values.join(aggregates)
    wide = values.unstack("lab")
    wide.columns = [f"{lab}_{metric}" for metric, lab in wide.columns]
    labs = list(config["study"]["labs"])
    metrics = ["first", "last", "mean", "min", "max", "count", "slope"]
    columns = [f"{lab}_{metric}" for lab in labs for metric in metrics]
    return wide.reindex(columns=columns).reset_index()


def _artifact_key(function: object, config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    material = f"{inspect.getsource(function)}\n{canonical}".encode()
    return hashlib.sha256(material).hexdigest()


def write_features(path: Path, frame: pd.DataFrame, key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[b"featurizer_sha"] = key.encode()
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, path, compression="zstd", version="2.6")


def load_or_build_features(
    path: Path,
    events: pd.DataFrame,
    admissions: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    key = _artifact_key(build_features, config)
    if path.is_file():
        metadata = pq.read_metadata(path).metadata or {}
        if metadata.get(b"featurizer_sha", b"").decode() == key:
            return pd.read_parquet(path)
    frame = build_features(events, admissions, config)
    write_features(path, frame, key)
    return frame
