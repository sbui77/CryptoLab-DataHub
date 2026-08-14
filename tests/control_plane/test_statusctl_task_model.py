from __future__ import annotations

import json
from pathlib import Path


def _start_work(
    run_statusctl,
    task_id: str = "work-task-001",
) -> None:
    result = run_statusctl(
        "start",
        task_id,
        "Work Task",
    )

    assert result.returncode == 0, (
        result.stderr
        or result.stdout
    )


def _record_ready_tests(
    run_statusctl,
) -> None:
    targeted = run_statusctl(
        "test",
        "targeted",
        "PASS",
        "pytest targeted",
        "Targeted tests passed.",
    )

    assert targeted.returncode == 0

    full = run_statusctl(
        "test",
        "full",
        "PASS",
        "pytest full",
        "Full tests passed.",
    )

    assert full.returncode == 0


def _ready_work(
    run_statusctl,
) -> None:
    _record_ready_tests(
        run_statusctl
    )

    phase = run_statusctl(
        "phase",
        "FINAL_VERIFY",
    )

    assert phase.returncode == 0

    ready = run_statusctl(
        "ready"
    )

    assert ready.returncode == 0, (
        ready.stderr
        or ready.stdout
    )


def _v2_running_state() -> dict:
    return {
        "schema_version": "2.0",
        "project": "CryptoLab-DataHub",
        "state": "RUNNING",
        "task_id": "legacy-work-001",
        "task_title": "Legacy Work Task",
        "phase": "IMPLEMENT",
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


def _write_v2_runtime(
    runtime_dir: Path,
    *,
    phase: str = "IMPLEMENT",
) -> None:
    runtime_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = _v2_running_state()
    data["phase"] = phase

    (
        runtime_dir
        / "status.json"
    ).write_text(
        json.dumps(
            data,
            indent=2,
        )
        + "\n"
    )


def test_fresh_runtime_uses_schema_v3(
    run_statusctl,
):
    result = run_statusctl(
        "show"
    )

    assert result.returncode == 0

    data = json.loads(
        result.stdout
    )

    assert (
        data["schema_version"]
        == "3.1"
    )

    assert data["task_kind"] is None
    assert data["parent_task_id"] is None


def test_start_creates_work_task(
    run_statusctl,
    read_status,
):
    _start_work(
        run_statusctl
    )

    status = read_status()

    assert (
        status["schema_version"]
        == "3.1"
    )

    assert (
        status["task_kind"]
        == "WORK"
    )

    assert (
        status["parent_task_id"]
        is None
    )


def test_work_task_accepts_external_review_phases(
    run_statusctl,
    read_status,
):
    _start_work(
        run_statusctl
    )

    external = run_statusctl(
        "phase",
        "EXTERNAL_REVIEW",
    )

    assert external.returncode == 0, (
        external.stderr
        or external.stdout
    )

    assert (
        read_status()["phase"]
        == "EXTERNAL_REVIEW"
    )

    correction = run_statusctl(
        "phase",
        "CORRECT_EXTERNAL_FINDINGS",
    )

    assert correction.returncode == 0, (
        correction.stderr
        or correction.stdout
    )

    assert (
        read_status()["phase"]
        == "CORRECT_EXTERNAL_FINDINGS"
    )


def test_ready_requires_final_verify_phase(
    run_statusctl,
    read_status,
):
    _start_work(
        run_statusctl
    )

    _record_ready_tests(
        run_statusctl
    )

    result = run_statusctl(
        "ready"
    )

    assert result.returncode != 0

    status = read_status()

    assert status["state"] == "RUNNING"
    assert status["phase"] == "ANALYZE"


def test_work_task_cannot_open_git_integration_gate(
    run_statusctl,
    read_status,
):
    _start_work(
        run_statusctl
    )

    result = run_statusctl(
        "gate",
        "open",
        "GIT_INTEGRATION",
        "Approve commit?",
        "Use an Integration Task.",
    )

    assert result.returncode != 0

    status = read_status()

    assert status["state"] == "RUNNING"
    assert status["human_gate"] is None


def test_start_integration_requires_ready_work(
    run_statusctl,
    read_status,
):
    _start_work(
        run_statusctl
    )

    result = run_statusctl(
        "start-integration",
        "work-task-integration-001",
        "Integrate Work Task",
        "--action",
        "none",
    )

    assert result.returncode != 0

    status = read_status()

    assert status["task_id"] == "work-task-001"
    assert status["task_kind"] == "WORK"


def test_start_integration_captures_parent_work_task(
    run_statusctl,
    read_status,
):
    _start_work(
        run_statusctl
    )

    _ready_work(
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

    assert status["state"] == "RUNNING"

    assert (
        status["task_kind"]
        == "INTEGRATION"
    )

    assert (
        status["parent_task_id"]
        == "work-task-001"
    )

    assert (
        status["task_id"]
        == "work-task-integration-001"
    )


def test_integration_task_can_open_git_gate(
    run_statusctl,
    read_status,
):
    _start_work(
        run_statusctl
    )

    _ready_work(
        run_statusctl
    )

    started = run_statusctl(
        "start-integration",
        "work-task-integration-001",
        "Integrate Work Task",
        "--action",
        "commit",
    )

    assert started.returncode == 0

    # Integration actions are gated in INTEGRATE, which is where the
    # task stands at its boundary.
    assert run_statusctl(
        "phase",
        "INTEGRATE",
    ).returncode == 0

    opened = run_statusctl(
        "gate",
        "open",
        "GIT_INTEGRATION",
        "--action",
        "commit",
        "Approve commit?",
        "Approve integration action.",
    )

    assert opened.returncode == 0, (
        opened.stderr
        or opened.stdout
    )

    status = read_status()

    assert (
        status["state"]
        == "BLOCKED_HUMAN_DECISION"
    )

    assert (
        status["human_gate"]["type"]
        == "GIT_INTEGRATION"
    )


def test_integration_task_rejects_work_only_phase(
    run_statusctl,
    read_status,
):
    _start_work(
        run_statusctl
    )

    _ready_work(
        run_statusctl
    )

    started = run_statusctl(
        "start-integration",
        "work-task-integration-001",
        "Integrate Work Task",
        "--action",
        "none",
    )

    assert started.returncode == 0

    result = run_statusctl(
        "phase",
        "IMPLEMENT",
    )

    assert result.returncode != 0

    assert (
        read_status()["phase"]
        == "ANALYZE"
    )


def test_normal_load_refuses_unmigrated_v2_runtime(
    run_statusctl,
    runtime_dir: Path,
):
    _write_v2_runtime(
        runtime_dir
    )

    result = run_statusctl(
        "show"
    )

    assert result.returncode != 0

    output = (
        result.stdout
        + result.stderr
    )

    assert "migrate-v3" in output


def test_migrate_v2_work_to_v3(
    run_statusctl,
    runtime_dir: Path,
    read_status,
    read_audit,
):
    _write_v2_runtime(
        runtime_dir
    )

    result = run_statusctl(
        "migrate-v3",
        "WORK",
    )

    assert result.returncode == 0, (
        result.stderr
        or result.stdout
    )

    status = read_status()

    assert (
        status["schema_version"]
        == "3.1"
    )

    assert (
        status["task_kind"]
        == "WORK"
    )

    assert (
        status["parent_task_id"]
        is None
    )

    assert (
        status["task_id"]
        == "legacy-work-001"
    )

    events = read_audit()

    assert any(
        event["event"]
        == "RUNTIME_SCHEMA_MIGRATED"
        for event in events
    )


def test_migrate_integration_requires_parent(
    run_statusctl,
    runtime_dir: Path,
):
    _write_v2_runtime(
        runtime_dir,
        phase="ANALYZE",
    )

    before = (
        runtime_dir
        / "status.json"
    ).read_bytes()

    result = run_statusctl(
        "migrate-v3",
        "INTEGRATION",
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


def test_migrate_v2_integration_with_parent(
    run_statusctl,
    runtime_dir: Path,
    read_status,
):
    _write_v2_runtime(
        runtime_dir,
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
        status["schema_version"]
        == "3.1"
    )

    assert (
        status["task_kind"]
        == "INTEGRATION"
    )

    assert (
        status["parent_task_id"]
        == "parent-work-001"
    )


def test_migrate_integration_rejects_work_only_phase(
    run_statusctl,
    runtime_dir: Path,
):
    _write_v2_runtime(
        runtime_dir,
        phase="IMPLEMENT",
    )

    before = (
        runtime_dir
        / "status.json"
    ).read_bytes()

    result = run_statusctl(
        "migrate-v3",
        "INTEGRATION",
        "parent-work-001",
    )

    assert result.returncode != 0

    output = (
        result.stdout
        + result.stderr
    ).lower()

    assert "integration" in output
    assert "phase" in output

    assert (
        runtime_dir
        / "status.json"
    ).read_bytes() == before
