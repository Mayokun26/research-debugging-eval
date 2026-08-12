from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_folds(cohort: pd.DataFrame, config: dict[str, Any]) -> dict[str, int]:
    settings = config["evaluation"]
    strata = cohort["site"].astype(str) + "_" + cohort["readmitted_30d"].astype(str)
    splitter = StratifiedKFold(
        n_splits=int(settings["folds"]),
        shuffle=True,
        random_state=int(settings["fold_seed"]),
    )
    assignments: dict[str, int] = {}
    for fold, (_, test_index) in enumerate(splitter.split(cohort, strata)):
        for index in test_index:
            assignments[str(int(cohort.iloc[index]["patient_id"]))] = fold
    return assignments


def _artifact_key(function: object, config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    material = f"{inspect.getsource(function)}\n{canonical}".encode()
    return hashlib.sha256(material).hexdigest()


def load_or_build_folds(
    path: Path, cohort: pd.DataFrame, config: dict[str, Any]
) -> dict[str, int]:
    key = _artifact_key(build_folds, config)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("fold_sha") == key:
            return {str(k): int(v) for k, v in payload["assignments"].items()}
    assignments = build_folds(cohort, config)
    payload = {"fold_sha": key, "assignments": assignments}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return assignments


def _lr(c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(C=c_value, max_iter=1000, random_state=seed),
            ),
        ]
    )


def _gbm(leaves: int, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.06,
                    max_iter=140,
                    max_leaf_nodes=leaves,
                    l2_regularization=1.0,
                    random_state=seed,
                ),
            ),
        ]
    )


def _select(
    x: pd.DataFrame,
    y: np.ndarray,
    candidates: list[float | int],
    factory: Callable[[float | int, int], Pipeline],
    config: dict[str, Any],
) -> float | int:
    settings = config["evaluation"]
    splitter = StratifiedKFold(
        n_splits=int(config["models"]["inner_folds"]),
        shuffle=True,
        random_state=int(settings["model_seed"]),
    )
    scored: list[tuple[float, float | int]] = []
    for candidate in candidates:
        fold_scores: list[float] = []
        for train_index, valid_index in splitter.split(x, y):
            model = factory(candidate, int(settings["model_seed"]))
            model.fit(x.iloc[train_index], y[train_index])
            probabilities = model.predict_proba(x.iloc[valid_index])[:, 1]
            fold_scores.append(float(roc_auc_score(y[valid_index], probabilities)))
        scored.append((float(np.mean(fold_scores)), candidate))
    return max(scored, key=lambda item: (item[0], -float(item[1])))[1]


def fit_evaluate(
    cohort: pd.DataFrame,
    assignments: dict[str, int],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[Pipeline], list[Pipeline]]:
    labels = cohort["readmitted_30d"].to_numpy(dtype=int)
    last_columns = [column for column in cohort.columns if column.endswith("_last")]
    excluded = {"patient_id", "site", "readmitted_30d"}
    all_columns = [column for column in cohort.columns if column not in excluded]
    outputs: list[pd.DataFrame] = []
    lr_models: list[Pipeline] = []
    gbm_models: list[Pipeline] = []
    for fold in range(int(config["evaluation"]["folds"])):
        test_mask = cohort["patient_id"].astype(str).map(assignments).to_numpy() == fold
        train_index = np.flatnonzero(~test_mask)
        test_index = np.flatnonzero(test_mask)
        seed = int(config["evaluation"]["model_seed"]) + fold
        lr_c = _select(
            cohort.iloc[train_index][last_columns],
            labels[train_index],
            list(config["models"]["logistic_c"]),
            _lr,
            config,
        )
        leaves = _select(
            cohort.iloc[train_index][all_columns],
            labels[train_index],
            list(config["models"]["gbm_leaf_nodes"]),
            _gbm,
            config,
        )
        lr_model = _lr(float(lr_c), seed)
        gbm_model = _gbm(int(leaves), seed)
        lr_model.fit(cohort.iloc[train_index][last_columns], labels[train_index])
        gbm_model.fit(cohort.iloc[train_index][all_columns], labels[train_index])
        outputs.append(
            pd.DataFrame(
                {
                    "patient_id": cohort.iloc[test_index]["patient_id"].to_numpy(dtype=int),
                    "fold": fold,
                    "site": cohort.iloc[test_index]["site"].to_numpy(),
                    "y_true": labels[test_index],
                    "lr_probability": lr_model.predict_proba(cohort.iloc[test_index][last_columns])[:, 1],
                    "gbm_probability": gbm_model.predict_proba(cohort.iloc[test_index][all_columns])[:, 1],
                }
            )
        )
        lr_models.append(lr_model)
        gbm_models.append(gbm_model)
    predictions = pd.concat(outputs, ignore_index=True).sort_values("patient_id").reset_index(drop=True)
    return predictions, lr_models, gbm_models


def save_models(path: Path, lr_models: list[Pipeline], gbm_models: list[Pipeline]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr_models, path / "lr_final.pkl", compress=3)
    joblib.dump(gbm_models, path / "gbm_final.pkl", compress=3)


def prediction_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
