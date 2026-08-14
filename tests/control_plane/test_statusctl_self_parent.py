"""
ER-01: an Integration Task may not be its own parent.

Semantics A keeps the reviewed Work Task and the act of integrating it
as two separate, individually auditable identities. A task whose
parent_task_id equals its own task_id collapses them, leaving the
parent pointer uninformative and the audit trail ambiguous.

The invariant is enforced at three independent points, each covered
here:

    start-integration   refuses to create such a task
    migrate-v3          refuses to migrate into such a task
    the validator       refuses to load such a persisted state
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


WORK_TASK_ID = "work-task-001"


def _start_ready_work(
    run_statusctl,
) -> None:
    assert run_statusctl(
        "start",
        WORK_TASK_ID,
        "Work Task",
    ).returncode == 0

    assert run_statusctl(
        "test",
        "targeted",
        "PASS",
        "pytest targeted",
        "Targeted tests passed.",
    ).returncode == 0

    assert run_statusctl(
        "test",
        "full",
        "PASS",
        "pytest full",
        "Full tests passed.",
    ).returncode == 0

    assert run_statusctl(
        "phase",
        "FINAL_VERIFY",
    ).returncode == 0

    assert run_statusctl(
        "ready"
    ).returncode == 0


def _write_v2_running(
    runtime_dir: Path,
    *,
    task_id: str,
    phase: str,
) -> bytes:
    runtime_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = {
        "schema_version": "2.0",
        "project": "CryptoLab-DataHub",
        "state": "RUNNING",
        "task_id": task_id,
        "task_title": "Legacy Work Task",
        "phase": phase,
        "branch": "test-branch",
        "started_at": "2026-08-13T00:00:00Z",
        "updated_at": "2026-08-13T00:01:00Z",
        "completed_at": None,
        "tests": {
            "targeted": None,
            "full": None,
        },
        "human_gate": None,
        "last_gate_decision": None,
        "report": ".claude/runtime/latest_report.md",
    }

    path = (
        runtime_dir
        / "status.json"
    )

    path.write_text(
        json.dumps(
            data,
            indent=2,
        )
        + "\n"
    )

    return path.read_bytes()


# ================================================================
# 1. start-integration
# ================================================================


def test_start_integration_rejects_self_parent(
    run_statusctl,
    runtime_dir: Path,
    read_status,
    read_audit,
):
    _start_ready_work(
        run_statusctl
    )

    status_before = (
        runtime_dir
        / "status.json"
    ).read_bytes()

    audit_before = (
        runtime_dir
        / "audit.jsonl"
    ).read_bytes()

    result = run_statusctl(
        "start-integration",
        WORK_TASK_ID,
        "Integrate Work Task",
        "--action",
        "none",
    )

    assert result.returncode != 0

    assert (
        runtime_dir
        / "status.json"
    ).read_bytes() == status_before

    assert (
        runtime_dir
        / "audit.jsonl"
    ).read_bytes() == audit_before

    assert not [
        event
        for event in read_audit()
        if event["event"]
        == "INTEGRATION_TASK_STARTED"
    ]

    status = read_status()

    assert (
        status["state"]
        == "READY_FOR_HUMAN_REVIEW"
    )

    assert (
        status["task_kind"]
        == "WORK"
    )

    assert (
        status["task_id"]
        == WORK_TASK_ID
    )


def test_start_integration_rejects_whitespace_padded_self_parent(
    run_statusctl,
    runtime_dir: Path,
    read_audit,
):
    """
    A padded copy of the parent ID is the same task, not a new one.
    """

    _start_ready_work(
        run_statusctl
    )

    status_before = (
        runtime_dir
        / "status.json"
    ).read_bytes()

    result = run_statusctl(
        "start-integration",
        f"  {WORK_TASK_ID}  ",
        "Integrate Work Task",
        "--action",
        "none",
    )

    assert result.returncode != 0

    assert (
        runtime_dir
        / "status.json"
    ).read_bytes() == status_before

    assert not [
        event
        for event in read_audit()
        if event["event"]
        == "INTEGRATION_TASK_STARTED"
    ]


def test_start_integration_still_accepts_a_distinct_task_id(
    run_statusctl,
    read_status,
):
    """
    The guard must not block the legitimate case.
    """

    _start_ready_work(
        run_statusctl
    )

    result = run_statusctl(
        "start-integration",
        "work-task-integration-001",
        "Integrate Work Task",
        "--action",
        "none",
    )

    assert result.returncode == 0, (
        result.stderr
        or result.stdout
    )

    status = read_status()

    assert (
        status["task_kind"]
        == "INTEGRATION"
    )

    assert (
        status["parent_task_id"]
        == WORK_TASK_ID
    )

    assert (
        status["task_id"]
        != status["parent_task_id"]
    )


# ================================================================
# 2. migrate-v3
# ================================================================


def test_migrate_integration_rejects_self_parent(
    run_statusctl,
    runtime_dir: Path,
    read_audit,
):
    before = _write_v2_running(
        runtime_dir,
        task_id="legacy-work-001",
        phase="ANALYZE",
    )

    result = run_statusctl(
        "migrate-v3",
        "INTEGRATION",
        "legacy-work-001",
    )

    assert result.returncode != 0

    output = (
        result.stdout
        + result.stderr
    ).lower()

    assert "parent" in output

    assert (
        runtime_dir
        / "status.json"
    ).read_bytes() == before

    assert not [
        event
        for event in read_audit()
        if event["event"]
        == "RUNTIME_SCHEMA_MIGRATED"
    ]


def test_migrate_integration_accepts_a_distinct_parent(
    run_statusctl,
    runtime_dir: Path,
    read_status,
):
    _write_v2_running(
        runtime_dir,
        task_id="legacy-work-001",
        phase="ANALYZE",
    )

    result = run_statusctl(
        "migrate-v3",
        "INTEGRATION",
        "parent-work-001",
        "--action",
        "commit",
    )

    assert result.returncode == 0, (
        result.stderr
        or result.stdout
    )

    status = read_status()

    assert (
        status["parent_task_id"]
        == "parent-work-001"
    )

    assert (
        status["task_id"]
        != status["parent_task_id"]
    )


# ================================================================
# 3. canonical validator / persisted state
# ================================================================


def test_validator_rejects_self_parent_state(
    run_statusctl,
    read_status,
    load_statusctl_module,
):
    _start_ready_work(
        run_statusctl
    )

    assert run_statusctl(
        "start-integration",
        "work-task-integration-001",
        "Integrate Work Task",
        "--action",
        "none",
    ).returncode == 0

    module = (
        load_statusctl_module()
    )

    data = read_status()

    module.validate_runtime_state(
        data
    )

    data["parent_task_id"] = data[
        "task_id"
    ]

    with pytest.raises(
        SystemExit,
    ) as exc_info:
        module.validate_runtime_state(
            data
        )

    assert (
        "Runtime state validation failed:"
        in str(exc_info.value)
    )


def test_persisted_self_parent_state_fails_closed(
    run_statusctl,
    runtime_dir: Path,
):
    """
    A hand-edited self-parent runtime must not be loadable.
    """

    _start_ready_work(
        run_statusctl
    )

    assert run_statusctl(
        "start-integration",
        "work-task-integration-001",
        "Integrate Work Task",
        "--action",
        "none",
    ).returncode == 0

    status_path = (
        runtime_dir
        / "status.json"
    )

    data = json.loads(
        status_path.read_text()
    )

    data["parent_task_id"] = data[
        "task_id"
    ]

    status_path.write_text(
        json.dumps(
            data,
            indent=2,
        )
        + "\n"
    )

    result = run_statusctl(
        "show"
    )

    assert result.returncode != 0

    assert (
        "Runtime state validation failed:"
        in (
            result.stdout
            + result.stderr
        )
    )
