"""Minimal Inspect harness for the Silent Ledger worked example."""

from __future__ import annotations

import io
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import TaskState, basic_agent
from inspect_ai.tool import bash, text_editor
from inspect_ai.util import sandbox

RELEASE_ROOT = Path(__file__).resolve().parent
TASK_ROOT = RELEASE_ROOT / "task"
DOCKERFILE = RELEASE_ROOT / "Dockerfile"
CACHE_DIRECTORIES = {".pytest_cache", ".ruff_cache", "__pycache__"}

TASK_ROOTS = {
    "task-002-silent-ledger": TASK_ROOT,
}

HOST_DEPENDENCIES = tuple(
    line
    for line in (RELEASE_ROOT / "canonical-env.txt")
    .read_text(encoding="utf-8")
    .splitlines()
    if line.strip() and not line.startswith("#")
)

PREAMBLE = """\
You are working inside an isolated sandbox. The study repository files are in the
current working directory. Complete the task below and build your final submission
at exactly /submission inside the sandbox: place diagnosis.json, repaired_repo/,
and any evidence files directly at /submission, not inside a nested subdirectory.
Only files under /submission are graded.

"""


def sandbox_files(task_root: Path) -> dict[str, str]:
    """Map the agent-visible repository into the sandbox working directory."""
    repo_root = (task_root / "repo").resolve()
    files: dict[str, str] = {}
    for source in sorted(repo_root.rglob("*")):
        if CACHE_DIRECTORIES.intersection(source.relative_to(repo_root).parts):
            continue
        if source.is_file():
            relative = source.relative_to(repo_root).as_posix()
            files[relative] = str(source)
    return files


def uv_python_command(script: Path, *arguments: str) -> list[str]:
    """Run a task script with its recorded direct dependencies via uv."""
    command = ["uv", "run", "--quiet", "--no-project"]
    for dependency in HOST_DEPENDENCIES:
        command.extend(("--with", dependency))
    return [*command, "python", str(script), *arguments]


def run_hidden_verifier(
    task_root: Path, submission_dir: Path, *, timeout: int = 300
) -> dict[str, Any]:
    """Run a frozen verifier on the host and return its JSON result."""
    verifier = task_root / "hidden" / "verifier.py"
    completed = subprocess.run(
        uv_python_command(verifier, str(submission_dir)),
        cwd=task_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        return {
            "status": "FAIL",
            "categories": ["verifier_error"],
            "reasons": [
                f"Verifier exited with status {completed.returncode}: {detail[-2000:]}"
            ],
        }
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        return {
            "status": "FAIL",
            "categories": ["verifier_error"],
            "reasons": [f"Verifier returned invalid JSON: {error}"],
        }
    if not isinstance(result, dict):
        return {
            "status": "FAIL",
            "categories": ["verifier_error"],
            "reasons": ["Verifier JSON result was not an object"],
        }
    return result


def _normalize_submission_root(root: Path) -> Path:
    """Tolerate a submission wrapped in exactly one named subdirectory."""
    if (root / "diagnosis.json").is_file():
        return root
    entries = [p for p in root.iterdir() if not p.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir() and (entries[0] / "diagnosis.json").is_file():
        return entries[0]
    return root


def _extract_submission(archive: bytes, destination: Path) -> None:
    """Safely extract regular submission files from an untrusted sandbox archive."""
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        for member in bundle.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe archive path: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsupported archive member: {member.name}")
        bundle.extractall(destination, filter="data")


async def export_submission(destination: Path) -> Path:
    """Copy /submission from the active Inspect sandbox to a host directory."""
    archive_path = "/tmp/inspect-submission.tar.gz"
    candidate = "/submission"
    probe = await sandbox().exec(
        ["sh", "-c", f"ls -A {candidate} 2>/dev/null | head -1"], timeout=30
    )
    if not probe.success or not probe.stdout.strip():
        raise FileNotFoundError("no non-empty submission directory found")
    result = await sandbox().exec(
        ["tar", "-C", candidate, "-czf", archive_path, "."], timeout=120
    )
    if not result.success:
        detail = result.stderr.strip()
        raise FileNotFoundError(detail or "no non-empty submission directory found")
    archive = await sandbox().read_file(archive_path, text=False)
    if not isinstance(archive, bytes):
        raise TypeError("Sandbox archive was not returned as bytes")
    _extract_submission(archive, destination)
    return destination


def _explanation(result: dict[str, Any]) -> str:
    reasons = result.get("reasons", [])
    if isinstance(reasons, list):
        reason_text = "; ".join(str(reason) for reason in reasons)
    else:
        reason_text = str(reasons)
    return reason_text or f"Verifier status: {result.get('status', 'unknown')}"


@scorer(metrics=[accuracy()])
def hidden_verifier_scorer(task_slug: str) -> Scorer:
    """Score /submission using the corresponding host-only frozen verifier."""
    task_root = TASK_ROOTS[task_slug]

    async def score(state: TaskState, target: Target) -> Score:
        del state, target
        try:
            with tempfile.TemporaryDirectory(prefix=f"inspect-{task_slug}-") as temp:
                submission = await export_submission(Path(temp) / "submission")
                submission = _normalize_submission_root(submission)
                result = run_hidden_verifier(task_root, submission)
        # Sandbox transport and adversarial archive failures must score zero rather
        # than abort the evaluation, regardless of their provider-specific type.
        except Exception as error:  # noqa: BLE001
            return Score(value=0.0, explanation=f"Could not grade submission: {error}")
        return Score(
            value=1.0 if result.get("status") == "PASS" else 0.0,
            explanation=_explanation(result),
            metadata={"verifier": result},
        )

    return score


def _research_task(
    task_slug: str,
    *,
    message_limit: int,
    token_limit: int,
    time_limit: int,
) -> Task:
    task_root = TASK_ROOTS[task_slug]
    instruction = (task_root / "instruction.md").read_text(encoding="utf-8")
    sample = Sample(
        id=task_slug,
        input=PREAMBLE + instruction,
        files=sandbox_files(task_root),
    )
    return Task(
        dataset=[sample],
        solver=basic_agent(tools=[bash(), text_editor()]),
        scorer=hidden_verifier_scorer(task_slug),
        sandbox=("docker", str(DOCKERFILE)),
        message_limit=message_limit,
        token_limit=token_limit,
        time_limit=time_limit,
    )


@task
def task_002(
    message_limit: int = 100,
    token_limit: int = 200_000,
    time_limit: int = 3_600,
) -> Task:
    """Silent Ledger research-debugging task."""
    return _research_task(
        "task-002-silent-ledger",
        message_limit=message_limit,
        token_limit=token_limit,
        time_limit=time_limit,
    )
