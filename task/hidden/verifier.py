from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

HIDDEN = Path(__file__).resolve().parent
ANSWER = json.loads((HIDDEN / "answer.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HIDDEN))

from evidence import feature_evidence

MECHANISMS = {
    "M0": "The analysis is sound as shipped and the reported conclusion stands.",
    "M1": "The derived table was produced from an earlier raw-record snapshot while its source and configuration fingerprints remained current.",
    "M2": "The derived table was produced with a different observation-window configuration while its source fingerprint remained current.",
    "M3": "The derived table retained positional boundary values produced before a helper-level chronological preparation change, while the direct function and configuration fingerprint remained current.",
    "M4": "An accidental digest collision caused an unrelated derived table to be accepted as current.",
    "M5": "A preprocessing statistic is computed across the training and evaluation boundary before the split is applied.",
    "M6": "Records describing the same underlying patient appear on both sides of an evaluation split under distinct identifiers.",
    "M7": "Per-fold results are pooled or weighted in a way that changes the direction of the model comparison.",
    "M8": "The configuration actually executed differs from the configuration recorded in the study outputs in a result-changing way.",
    "M9": "The baseline model is fit or tuned under conditions that structurally handicap it relative to the comparison model.",
    "M10": "The scoring or metric implementation is defective in a way that changes the model comparison.",
    "M11": "The committed result files were not produced by the code and data currently in the repository.",
}


@dataclass
class Verdict:
    reasons: list[str] = field(default_factory=list)
    categories: set[str] = field(default_factory=set)

    def fail(self, category: str, reason: str) -> None:
        self.categories.add(category)
        self.reasons.append(f"{category}: {reason}")

    def result(self) -> dict[str, Any]:
        return {
            "status": "PASS" if not self.reasons else "FAIL",
            "categories": sorted(self.categories),
            "reasons": self.reasons,
        }


def read_object(path: Path, limit: int = 128_000) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"missing or oversized file: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path.name}")
    return value


def run_repaired(repo: Path) -> dict[str, Any]:
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "LOKY_MAX_CPU_COUNT": "1",
        "PYTHONWARNINGS": "ignore",
    }
    completed = subprocess.run(
        [sys.executable, "run.py"],
        cwd=repo,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise ValueError(f"analysis exited {completed.returncode}: {completed.stderr[-300:]}")
    return read_object(repo / "results" / "metrics.json")


def safe_repo(submission: Path) -> Path:
    resolved_submission = submission.resolve(strict=True)
    repaired = (submission / "repaired_repo").resolve(strict=True)
    if not repaired.is_relative_to(resolved_submission):
        raise ValueError("repaired_repo resolves outside submission")
    if not (repaired / "run.py").is_file():
        raise ValueError("repaired_repo/run.py is missing")
    return repaired


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def verify(submission: Path) -> dict[str, Any]:
    verdict = Verdict()
    try:
        diagnosis = read_object(submission / "diagnosis.json")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        verdict.fail("diagnosis", str(error))
        return verdict.result()

    if diagnosis.get("flaw_class") != ANSWER["flaw_class"]:
        verdict.fail("diagnosis", "flaw_class does not match")
    if diagnosis.get("file") != ANSWER["file"]:
        verdict.fail("diagnosis", "file does not match")
    mechanism = diagnosis.get("mechanism_code")
    if mechanism not in MECHANISMS:
        verdict.fail("diagnosis", "mechanism_code is outside the closed enumeration")
    elif mechanism != ANSWER["mechanism_code"]:
        verdict.fail("diagnosis", "mechanism_code does not match")

    try:
        observed_evidence, _ = feature_evidence()
        submitted_evidence = diagnosis.get("evidence_fields")
        if not isinstance(submitted_evidence, dict):
            raise TypeError("evidence_fields must be an object")
        def normalized(value: Any) -> Any:
            if isinstance(value, list):
                return sorted(str(item) for item in value)
            return value

        for field_name in ANSWER["evidence_fields"]:
            if normalized(submitted_evidence.get(field_name)) != normalized(
                observed_evidence[field_name]
            ):
                verdict.fail("evidence", f"{field_name} does not match measured evidence")
        boundary_only = all(
            column.endswith(("_first", "_last"))
            for column in observed_evidence["affected_columns"]
        )
        if not boundary_only:
            verdict.fail("evidence", "affected columns are not limited to boundary summaries")
        patients = pd.read_csv(Path(__file__).resolve().parents[1] / "repo/data/raw/patients.csv")
        sites = patients.set_index("patient_id").loc[
            observed_evidence["affected_patient_ids"], "site"
        ]
        if not set(sites).issubset({"B"}):
            verdict.fail("evidence", "affected patients are not a subset of site B")
        if not observed_evidence["aggregate_columns_identical"]:
            verdict.fail("evidence", "aggregate columns are not identical")
    except (OSError, ValueError, TypeError, KeyError) as error:
        verdict.fail("evidence", str(error))

    if verdict.reasons:
        return verdict.result()

    try:
        reported = read_object(submission / "corrected_results.json")
        repaired = safe_repo(submission)
        observed = run_repaired(repaired)
        repaired_evidence, expected = feature_evidence(repaired)
        actual_features = pd.read_parquet(repaired / "data/derived/features.parquet")
        actual_features = actual_features.sort_values("patient_id").reset_index(drop=True)
        expected = expected.sort_values("patient_id").reset_index(drop=True)
        if not actual_features.equals(expected):
            verdict.fail("repair", "derived features do not agree with raw timestamps")
        if repaired_evidence["affected_columns"]:
            verdict.fail("repair", "repaired repository retains differing feature values")
        for field_name in (
            "lr_auroc",
            "gbm_auroc",
            "delta_auroc_gbm_minus_lr",
            "delta_ci_95",
        ):
            if reported.get(field_name) != observed.get(field_name):
                verdict.fail("rerun", f"reported {field_name} does not match rerun")
        delta = observed.get("delta_auroc_gbm_minus_lr")
        interval = observed.get("delta_ci_95")
        if not finite(delta) or not -0.005 <= delta <= 0.01:
            verdict.fail("conclusion", "corrected AUROC difference is outside the registered band")
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or not all(finite(value) for value in interval)
            or not interval[0] <= 0 <= interval[1]
        ):
            verdict.fail("conclusion", "corrected paired interval does not span zero")
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
        subprocess.TimeoutExpired,
    ) as error:
        verdict.fail("rerun", str(error))
    return verdict.result()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    args = parser.parse_args()
    result = verify(args.submission)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
