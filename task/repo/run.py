from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from src.data import attach_readmission_label, load_records, read_config
from src.features import load_or_build_features
from src.metrics import auroc, paired_bootstrap_ci
from src.pipeline import fit_evaluate, load_or_build_folds, save_models


def main() -> None:
    root = Path(__file__).resolve().parent
    config = read_config(root / "config.yaml")
    events, admissions, patients = load_records(root)
    admissions = attach_readmission_label(admissions, int(config["study"]["readmission_days"]))
    feature_path = root / "data" / "derived" / "features.parquet"
    features = load_or_build_features(feature_path, events, admissions, config)
    cohort = (
        features.merge(patients[["patient_id", "site"]], on="patient_id", validate="one_to_one")
        .merge(admissions[["patient_id", "readmitted_30d"]], on="patient_id", validate="one_to_one")
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    assignments = load_or_build_folds(root / "data" / "derived" / "folds.json", cohort, config)
    predictions, lr_models, gbm_models = fit_evaluate(cohort, assignments, config)
    prediction_root = root / "results" / "preds"
    prediction_root.mkdir(parents=True, exist_ok=True)
    for fold, frame in predictions.groupby("fold", sort=True):
        frame.reset_index(drop=True).to_parquet(
            prediction_root / f"fold_{int(fold)}.parquet",
            index=False,
            compression="zstd",
        )
    save_models(root / "models", lr_models, gbm_models)
    y_true = predictions["y_true"].to_numpy(dtype=int)
    lr_scores = predictions["lr_probability"].to_numpy(dtype=float)
    gbm_scores = predictions["gbm_probability"].to_numpy(dtype=float)
    lr_auc = auroc(y_true, lr_scores)
    gbm_auc = auroc(y_true, gbm_scores)
    ci_low, ci_high = paired_bootstrap_ci(
        y_true,
        gbm_scores,
        lr_scores,
        int(config["evaluation"]["bootstrap_samples"]),
        int(config["evaluation"]["bootstrap_seed"]),
    )
    precision = int(config["evaluation"]["reporting_precision"])
    metrics = {
        "bootstrap_samples": int(config["evaluation"]["bootstrap_samples"]),
        "cohort_patients": len(cohort),
        "delta_auroc_gbm_minus_lr": round(gbm_auc - lr_auc, precision),
        "delta_ci_95": [round(ci_low, precision), round(ci_high, precision)],
        "folds": int(config["evaluation"]["folds"]),
        "gbm_auroc": round(gbm_auc, precision),
        "lr_auroc": round(lr_auc, precision),
    }
    results_path = root / "results" / "metrics.json"
    results_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
