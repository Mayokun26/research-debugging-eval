from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HIDDEN = Path(__file__).resolve().parents[1]
TASK_ROOT = HIDDEN.parent
REPO = TASK_ROOT / "repo"
sys.path.insert(0, str(HIDDEN))

from evidence import feature_evidence


def prepare(destination: Path) -> Path:
    submission = destination / "submission"
    repaired = submission / "repaired_repo"
    shutil.copytree(
        REPO,
        repaired,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".ruff_cache"),
    )
    (repaired / "data" / "derived" / "features.parquet").unlink()
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "LOKY_MAX_CPU_COUNT": "1",
            "PYTHONWARNINGS": "ignore",
        }
    )
    completed = subprocess.run(
        [sys.executable, "run.py"],
        cwd=repaired,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr[-500:])
    evidence, _ = feature_evidence()
    diagnosis = {
        "flaw_class": "provenance",
        "file": "data/derived/features.parquet",
        "mechanism_code": "M3",
        "evidence_fields": evidence,
    }
    (submission / "diagnosis.json").write_text(
        json.dumps(diagnosis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metrics = json.loads((repaired / "results" / "metrics.json").read_text(encoding="utf-8"))
    (submission / "corrected_results.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return submission


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: prepare.py DESTINATION")
    print(prepare(Path(sys.argv[1])))
