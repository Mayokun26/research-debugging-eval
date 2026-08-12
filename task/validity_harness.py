from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tokenize
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

TASK_ROOT = Path(__file__).resolve().parent
REPO = TASK_ROOT / "repo"
HIDDEN = TASK_ROOT / "hidden"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HIDDEN))

from evidence import feature_evidence
from oracle_solution.prepare import prepare as prepare_oracle
from src.data import attach_readmission_label, load_records, read_config
from src.features import _artifact_key as feature_key
from src.features import build_features
from src.metrics import auroc, paired_bootstrap_ci
from src.pipeline import _artifact_key as fold_key
from src.pipeline import build_folds, fit_evaluate

ENVIRONMENT = {
    "PATH": os.environ.get("PATH", ""),
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "LOKY_MAX_CPU_COUNT": "1",
    "PYTHONWARNINGS": "ignore",
}
RUN_CACHE: dict[tuple[str, int], pd.DataFrame] = {}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run(arguments: list[str], cwd: Path, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=cwd,
        env=ENVIRONMENT,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def copy_repo(destination: Path) -> Path:
    copied = destination / "repo"
    shutil.copytree(
        REPO,
        copied,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".ruff_cache"),
    )
    return copied


def repo_run(repo: Path) -> dict[str, Any]:
    completed = run([sys.executable, "run.py"], repo)
    require(completed.returncode == 0, f"repo run failed: {completed.stderr[-500:]}")
    return json.loads((repo / "results" / "metrics.json").read_text(encoding="utf-8"))


def verifier(submission: Path) -> dict[str, Any]:
    completed = run([sys.executable, str(HIDDEN / "verifier.py"), str(submission)], TASK_ROOT)
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise AssertionError(f"invalid verifier output: {completed.stderr[-400:]}") from error


def load_worlds() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, int]]:
    config = read_config(REPO / "config.yaml")
    events, admissions, patients = load_records(REPO)
    admissions = attach_readmission_label(admissions, int(config["study"]["readmission_days"]))
    committed = pd.read_parquet(REPO / "data" / "derived" / "features.parquet")
    corrected = build_features(events, admissions, config)

    def cohort(frame: pd.DataFrame) -> pd.DataFrame:
        return (
            frame.merge(patients[["patient_id", "site"]], on="patient_id")
            .merge(admissions[["patient_id", "readmitted_30d"]], on="patient_id")
            .sort_values("patient_id")
            .reset_index(drop=True)
        )

    payload = json.loads((REPO / "data" / "derived" / "folds.json").read_text(encoding="utf-8"))
    assignments = {str(key): int(value) for key, value in payload["assignments"].items()}
    return cohort(committed), cohort(corrected), config, assignments


def predictions(world: str, seed: int) -> pd.DataFrame:
    key = (world, seed)
    if key not in RUN_CACHE:
        committed, corrected, config, assignments = load_worlds()
        varied = json.loads(json.dumps(config))
        varied["evaluation"]["model_seed"] = seed
        frame = committed if world == "committed" else corrected
        RUN_CACHE[key] = fit_evaluate(frame, assignments, varied)[0]
    return RUN_CACHE[key]


def delta_and_ci(frame: pd.DataFrame, seed: int) -> tuple[float, tuple[float, float]]:
    y = frame["y_true"].to_numpy(dtype=int)
    lr = frame["lr_probability"].to_numpy(dtype=float)
    gbm = frame["gbm_probability"].to_numpy(dtype=float)
    delta = auroc(y, gbm) - auroc(y, lr)
    interval = paired_bootstrap_ci(y, gbm, lr, 1000, seed + 991)
    return delta, interval


def check_oracle() -> str:
    with tempfile.TemporaryDirectory(prefix="task002-oracle-") as temporary:
        submission = prepare_oracle(Path(temporary))
        result = verifier(submission)
        require(result["status"] == "PASS", f"oracle rejected: {result}")
    return "PASS oracle_solution: accepted by verifier"


def check_no_op() -> str:
    with tempfile.TemporaryDirectory(prefix="task002-noop-") as temporary:
        root = Path(temporary)
        submission = root / "submission"
        submission.mkdir()
        copy = copy_repo(submission)
        diagnosis = {
            "flaw_class": "no_flaw",
            "file": "",
            "mechanism_code": "M1",
            "evidence_fields": {"headline_validated": True},
        }
        (submission / "diagnosis.json").write_text(json.dumps(diagnosis), encoding="utf-8")
        (submission / "corrected_results.json").write_bytes(
            (copy / "results" / "metrics.json").read_bytes()
        )
        result = verifier(submission)
        require(result["status"] == "FAIL", "unchanged submission passed")
        require("diagnosis" in result["categories"], f"unexpected no-op verdict: {result}")
    return "PASS no_op: unchanged repository and validated headline rejected"


def check_wrong_diagnoses() -> list[str]:
    cases = {
        "preprocessing_leakage": {
            "flaw_class": "preprocessing_leakage",
            "file": "src/pipeline.py",
            "mechanism_code": "M2",
            "evidence_fields": {"pooled_median": 139.1, "fold_median": 138.7},
        },
        "contamination": {
            "flaw_class": "contamination",
            "file": "data/derived/folds.json",
            "mechanism_code": "M1",
            "evidence_fields": {"overlapping_patient_ids": [100007, 100009]},
        },
        "grader_defect": {
            "flaw_class": "grader_defect",
            "file": "src/metrics.py",
            "mechanism_code": "M4",
            "evidence_fields": {"reference_auroc": 0.78, "reported_auroc": 0.74},
        },
        "invalid_control": {
            "flaw_class": "invalid_control",
            "file": "src/pipeline.py",
            "mechanism_code": "M2",
            "evidence_fields": {"selected_c": 0.1, "wider_grid_c": 100.0},
        },
        "site_shift": {
            "flaw_class": "invalid_control",
            "file": "data/raw/patients.csv",
            "mechanism_code": "M1",
            "evidence_fields": {"site_b_prevalence": 0.2, "site_b_auc": 0.61},
        },
        "config_inconsistency": {
            "flaw_class": "config_inconsistency",
            "file": "config.yaml",
            "mechanism_code": "M2",
            "evidence_fields": {"claimed_hours": 48, "artifact_hours": 24},
        },
    }
    summaries: list[str] = []
    for name, diagnosis in cases.items():
        with tempfile.TemporaryDirectory(prefix=f"task002-{name}-") as temporary:
            submission = Path(temporary)
            (submission / "diagnosis.json").write_text(
                json.dumps(diagnosis), encoding="utf-8"
            )
            result = verifier(submission)
            require(result["status"] == "FAIL", f"wrong diagnosis passed: {name}")
        summaries.append(f"PASS mutation/{name}: rejected")
    return summaries


def check_feature_evidence() -> str:
    evidence, _ = feature_evidence()
    require(evidence["affected_columns"], "no differing boundary columns")
    require(
        all(column.endswith(("_first", "_last")) for column in evidence["affected_columns"]),
        f"unexpected changed columns: {evidence['affected_columns']}",
    )
    require(evidence["affected_site"] == "B", "affected patients are not confined to site B")
    require(evidence["aggregate_columns_identical"], "aggregate columns changed")
    require(evidence["artifact_key_matches_current"], "feature key does not match")
    require(evidence["file_position_matches_committed"], "file-position fingerprint failed")
    require(len(evidence["affected_patient_ids"]) == 180, "affected patient count changed")
    return "PASS feature_forensics: boundary-only, 180 site-B patients, aggregates identical"


def check_determinism_and_refit() -> str:
    reference_metrics = (REPO / "results" / "metrics.json").read_bytes()
    expected_predicates = {
        path.name: path.read_bytes() for path in sorted((REPO / "results" / "preds").glob("*.parquet"))
    }
    expected_hashes = {name: hashlib.sha256(value).hexdigest() for name, value in expected_predicates.items()}
    for _ in range(2):
        repo_run(REPO)
        require(
            (REPO / "results" / "metrics.json").read_bytes() == reference_metrics,
            "metrics changed on naive rerun",
        )
        observed = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((REPO / "results" / "preds").glob("*.parquet"))
        }
        require(observed == expected_hashes, "prediction behavior hashes changed")
    direct = predictions("committed", int(read_config(REPO / "config.yaml")["evaluation"]["model_seed"]))
    for fold, frame in direct.groupby("fold", sort=True):
        bundled = pd.read_parquet(REPO / "results" / "preds" / f"fold_{int(fold)}.parquet")
        require(frame.reset_index(drop=True).equals(bundled), f"fold {fold} refit differs")
    return "PASS determinism_refit: two byte-exact reruns and fold-local refit agree"


def check_folds() -> str:
    reference = (REPO / "data" / "derived" / "folds.json").read_bytes()
    with tempfile.TemporaryDirectory(prefix="task002-folds-") as temporary:
        copied = copy_repo(Path(temporary))
        (copied / "data" / "derived" / "folds.json").unlink()
        repo_run(copied)
        require((copied / "data" / "derived" / "folds.json").read_bytes() == reference, "fold bytes differ")
    payload = json.loads(reference)
    require(len(payload["assignments"]) == 2500, "fold coverage changed")
    return "PASS folds_artifact: guarded artifact regenerates byte-identically"


def check_dedup() -> str:
    with tempfile.TemporaryDirectory(prefix="task002-dedup-") as temporary:
        copied = copy_repo(Path(temporary))
        path = copied / "data" / "raw" / "events.csv"
        events = pd.read_csv(path)
        deduplicated = events.drop_duplicates()
        require(len(deduplicated) == len(events), "generator unexpectedly contains duplicates")
        deduplicated.to_csv(path, index=False, lineterminator="\n")
        observed = repo_run(copied)
        expected = json.loads((REPO / "results" / "metrics.json").read_text(encoding="utf-8"))
        require(observed == expected, "dedup probe changed conclusion")
    return "PASS decoy/dedup_rerun: no duplicates and metrics unchanged"


def check_metric_reference() -> str:
    frames = [pd.read_parquet(path) for path in sorted((REPO / "results" / "preds").glob("*.parquet"))]
    frame = pd.concat(frames, ignore_index=True)
    y = frame["y_true"].to_numpy(dtype=int)
    for column in ("lr_probability", "gbm_probability"):
        require(abs(auroc(y, frame[column].to_numpy()) - roc_auc_score(y, frame[column])) < 1e-12, "AUROC disagreement")
    return "PASS decoy/metric_reference: custom AUROC agrees to 1e-12"


def check_seed_sweep() -> str:
    seeds = [1618, 2027, 4099, 5099, 6199]
    stale: list[float] = []
    corrected: list[float] = []
    corrected_intervals: list[tuple[float, float]] = []
    for seed in seeds:
        stale_delta, _ = delta_and_ci(predictions("committed", seed), seed)
        corrected_delta, interval = delta_and_ci(predictions("corrected", seed), seed)
        stale.append(stale_delta)
        corrected.append(corrected_delta)
        corrected_intervals.append(interval)
    require(min(stale) > 0.025, f"stale conclusion unstable: {stale}")
    require(max(corrected) <= 0.01 and min(corrected) >= -0.01, f"corrected sweep out of band: {corrected}")
    require(all(low <= 0 <= high for low, high in corrected_intervals), f"corrected interval excludes zero: {corrected_intervals}")
    return "PASS decoy/multi_seed: stale lead persists; corrected intervals all span zero"


def check_unused_config() -> str:
    committed, _, config, assignments = load_worlds()
    varied = json.loads(json.dumps(config))
    varied["reporting"]["cohort_label"] = "secondary display label"
    events, admissions, _ = load_records(REPO)
    admissions = attach_readmission_label(admissions, int(config["study"]["readmission_days"]))
    require(build_features(events, admissions, config).equals(build_features(events, admissions, varied)), "unused key changed features")
    require(build_folds(committed, config) == build_folds(committed, varied), "unused key changed folds")
    base = predictions("committed", int(config["evaluation"]["model_seed"]))
    varied_predictions = fit_evaluate(committed, assignments, varied)[0]
    require(base.equals(varied_predictions), "unused key changed predictions")
    require(feature_key(build_features, config) != feature_key(build_features, varied), "config absent from feature key")
    require(fold_key(build_folds, config) != fold_key(build_folds, varied), "config absent from fold key")
    return "PASS decoy/unused_config: outcomes invariant while both keys include canonical config"


def check_older_snapshot_dry_run() -> str:
    config = read_config(REPO / "config.yaml")
    events, admissions, _ = load_records(REPO)
    admissions = attach_readmission_label(admissions, int(config["study"]["readmission_days"]))
    older = events.loc[events["patient_id"] < events["patient_id"].max()].copy()
    frame = build_features(older, admissions, config)
    with tempfile.TemporaryDirectory(prefix="task002-older-") as temporary:
        path = Path(temporary) / "features.parquet"
        table = pa.Table.from_pandas(frame, preserve_index=False)
        metadata = dict(table.schema.metadata or {})
        metadata[b"featurizer_sha"] = feature_key(build_features, config).encode()
        pq.write_table(table.replace_schema_metadata(metadata), path, compression="zstd", version="2.6")
        observed = (pq.read_metadata(path).metadata or {})[b"featurizer_sha"].decode()
        require(observed == feature_key(build_features, config), "older snapshot is not hash-consistent")
        committed = pd.read_parquet(REPO / "data" / "derived" / "features.parquet")
        require(len(frame) != len(committed), "older snapshot unexpectedly matches committed rows")
    return "PASS verifier_dry_run: older-raw-snapshot distractor is hash-consistent"


def check_spot_rate() -> str:
    evidence, _ = feature_evidence()
    pass_rate = 1.0 - len(evidence["affected_patient_ids"]) / 2500
    require(pass_rate >= 0.85, f"strict spot-check expectation too low: {pass_rate}")
    return f"PASS casual_spot_rate: expected strict-check pass rate {pass_rate:.3f}"


def check_canaries() -> str:
    forbidden_everywhere = (
        "STALE_CACHE_KEY_BLIND_TO_HELPER",
        "mechanism_code",
        "affected_patient_ids",
        "aggregate_columns_identical",
        "attack_verdict",
        "hardening",
    )
    forbidden_narrative = ("cache", "stale", "rebuild", "regenerate", "ordering", "freshness")
    for path in REPO.rglob("*"):
        if not path.is_file() or any(part in {"__pycache__", ".pytest_cache", ".ruff_cache"} for part in path.parts):
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        lowered = content.lower()
        for token in forbidden_everywhere:
            require(token.lower() not in lowered, f"private token {token!r} in {path.relative_to(REPO)}")
        require("--rebuild" not in lowered, f"unsupported flag in {path.relative_to(REPO)}")
        if path.name == "README.md":
            for token in (*forbidden_narrative, "sort", "order"):
                require(token not in lowered, f"README narrative token {token!r}")
        if path.suffix == ".py":
            tokens = tokenize.generate_tokens(io.StringIO(content).readline)
            for token_info in tokens:
                if token_info.type == tokenize.COMMENT:
                    comment = token_info.string.lower()
                    for token in forbidden_narrative:
                        require(token not in comment, f"comment narrative token in {path.relative_to(REPO)}")
            for name in [line for line in content.splitlines() if line.startswith("def test_")]:
                for token in forbidden_narrative:
                    require(token not in name.lower(), f"test-name narrative token in {path.relative_to(REPO)}")
    completed = run([sys.executable, "run.py"], REPO)
    require(completed.returncode == 0, "canary run failed")
    output = completed.stdout.lower()
    for token in forbidden_narrative:
        require(token not in output, f"runtime narrative token {token!r}")
    return "PASS canary_scan: agent-facing tree and runtime contain no narrative tells"


def check_tests_and_lint() -> list[str]:
    pytest = run([sys.executable, "-m", "pytest", "-q"], REPO)
    require(pytest.returncode == 0, f"pytest failed: {pytest.stdout[-500:]} {pytest.stderr[-500:]}")
    ruff_available = importlib.util.find_spec("ruff") is not None
    require(ruff_available, "ruff is unavailable")
    ruff = run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--config",
            "repo/pyproject.toml",
            "repo",
            "hidden",
            "validity_harness.py",
        ],
        TASK_ROOT,
    )
    require(ruff.returncode == 0, f"ruff failed: {ruff.stdout[-1000:]}")
    return ["PASS pytest: bland agent tests are green", "PASS ruff: task code is clean"]


def main() -> None:
    checks = [check_oracle(), check_no_op()]
    checks.extend(check_wrong_diagnoses())
    checks.append(check_feature_evidence())
    checks.append(check_determinism_and_refit())
    checks.append(check_folds())
    checks.append(check_dedup())
    checks.append(check_metric_reference())
    checks.append(check_seed_sweep())
    checks.append(check_unused_config())
    checks.append(check_older_snapshot_dry_run())
    checks.append(check_spot_rate())
    checks.append(check_canaries())
    checks.extend(check_tests_and_lint())
    for check in checks:
        print(check)


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"FAIL gate: {error}")
        raise SystemExit(1) from error
