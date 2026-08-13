from __future__ import annotations

from pathlib import Path


def _start_task(
    run_statusctl,
    task_id: str = "lifecycle-test-001",
    title: str = "Lifecycle regression test",
) -> None:
    result = run_statusctl(
        "start",
        task_id,
        title,
    )

    assert result.returncode == 0, (
        result.stderr
        or result.stdout
    )


def _record_pass_tests(
    run_statusctl,
) -> None:
    targeted = run_statusctl(
        "test",
        "targeted",
        "PASS",
        "pytest targeted",
        "Targeted tests passed.",
    )

    assert targeted.returncode == 0, (
        targeted.stderr
        or targeted.stdout
    )

    full = run_statusctl(
        "test",
        "full",
        "PASS",
        "pytest full",
        "Full tests passed.",
    )

    assert full.returncode == 0, (
        full.stderr
        or full.stdout
    )


def _mark_final_verify(
    run_statusctl,
) -> None:
    result = run_statusctl(
        "phase",
        "FINAL_VERIFY",
    )

    assert result.returncode == 0, (
        result.stderr
        or result.stdout
    )


def _make_ready(
    run_statusctl,
) -> None:
    _record_pass_tests(
        run_statusctl
    )

    _mark_final_verify(
        run_statusctl
    )

    result = run_statusctl(
        "ready"
    )

    assert result.returncode == 0, (
        result.stderr
        or result.stdout
    )


def test_show_on_fresh_repo_is_read_only_idle(
    run_statusctl,
    runtime_dir: Path,
):
    assert (
        runtime_dir.exists()
        is False
    )

    result = run_statusctl(
        "show"
    )

    assert result.returncode == 0, (
        result.stderr
        or result.stdout
    )

    assert (
        '"state": "IDLE"'
        in result.stdout
    )

    assert (
        runtime_dir.exists()
        is False
    )


def test_start_creates_running_runtime(
    run_statusctl,
    runtime_dir: Path,
    read_status,
    read_audit,
):
    _start_task(
        run_statusctl
    )

    status = read_status()

    assert status["state"] == "RUNNING"

    assert (
        status["task_id"]
        == "lifecycle-test-001"
    )

    assert status["phase"] == "ANALYZE"

    assert status["completed_at"] is None
    assert status["human_gate"] is None

    assert (
        runtime_dir
        / "status.json"
    ).is_file()

    assert (
        runtime_dir
        / "audit.jsonl"
    ).is_file()

    assert (
        runtime_dir
        / "latest_report.md"
    ).is_file()

    events = read_audit()

    assert (
        events[-1]["event"]
        == "TASK_STARTED"
    )


def test_second_start_while_running_rejected_without_mutation(
    run_statusctl,
    runtime_dir: Path,
):
    _start_task(
        run_statusctl
    )

    status_path = (
        runtime_dir
        / "status.json"
    )

    audit_path = (
        runtime_dir
        / "audit.jsonl"
    )

    status_before = (
        status_path.read_bytes()
    )

    audit_before = (
        audit_path.read_bytes()
    )

    result = run_statusctl(
        "start",
        "second-task",
        "Second task must not start",
    )

    assert result.returncode != 0

    assert (
        status_path.read_bytes()
        == status_before
    )

    assert (
        audit_path.read_bytes()
        == audit_before
    )


def test_ready_requires_satisfied_test_requirements(
    run_statusctl,
    runtime_dir: Path,
    read_status,
):
    _start_task(
        run_statusctl
    )

    _mark_final_verify(
        run_statusctl
    )

    status_path = (
        runtime_dir
        / "status.json"
    )

    audit_path = (
        runtime_dir
        / "audit.jsonl"
    )

    status_before = (
        status_path.read_bytes()
    )

    audit_before = (
        audit_path.read_bytes()
    )

    result = run_statusctl(
        "ready"
    )

    assert result.returncode != 0

    assert (
        status_path.read_bytes()
        == status_before
    )

    assert (
        audit_path.read_bytes()
        == audit_before
    )

    status = read_status()

    assert status["state"] == "RUNNING"
    assert status["completed_at"] is None


def test_not_required_tests_can_satisfy_ready(
    run_statusctl,
    read_status,
    read_audit,
):
    _start_task(
        run_statusctl
    )

    targeted = run_statusctl(
        "test-not-required",
        "targeted",
        (
            "Targeted product test "
            "does not apply."
        ),
    )

    assert targeted.returncode == 0

    full = run_statusctl(
        "test-not-required",
        "full",
        (
            "Full product test "
            "does not apply."
        ),
    )

    assert full.returncode == 0

    _mark_final_verify(
        run_statusctl
    )

    ready = run_statusctl(
        "ready"
    )

    assert ready.returncode == 0, (
        ready.stderr
        or ready.stdout
    )

    status = read_status()

    assert (
        status["state"]
        == "READY_FOR_HUMAN_REVIEW"
    )

    assert (
        status["phase"]
        == "FINAL_VERIFY"
    )

    assert status["completed_at"] is not None

    assert (
        status["tests"]["targeted"]["result"]
        == "NOT_REQUIRED"
    )

    assert (
        status["tests"]["full"]["result"]
        == "NOT_REQUIRED"
    )

    events = read_audit()

    assert any(
        event["event"]
        == "TASK_READY_FOR_HUMAN_REVIEW"
        for event in events
    )


def test_ready_work_task_cannot_open_integration_gate(
    run_statusctl,
    runtime_dir: Path,
    read_status,
):
    _start_task(
        run_statusctl
    )

    _make_ready(
        run_statusctl
    )

    status = read_status()

    assert (
        status["state"]
        == "READY_FOR_HUMAN_REVIEW"
    )

    status_path = (
        runtime_dir
        / "status.json"
    )

    audit_path = (
        runtime_dir
        / "audit.jsonl"
    )

    status_before = (
        status_path.read_bytes()
    )

    audit_before = (
        audit_path.read_bytes()
    )

    result = run_statusctl(
        "gate",
        "open",
        "GIT_INTEGRATION",
        "Approve commit?",
        "Use a separate Integration Task.",
    )

    assert result.returncode != 0

    message = (
        result.stderr
        + result.stdout
    )

    assert (
        "Cannot open gate while "
        "state=READY_FOR_HUMAN_REVIEW"
        in message
    )

    assert (
        status_path.read_bytes()
        == status_before
    )

    assert (
        audit_path.read_bytes()
        == audit_before
    )


def test_fail_cannot_bypass_active_human_gate(
    run_statusctl,
    runtime_dir: Path,
    read_status,
):
    _start_task(
        run_statusctl
    )

    opened = run_statusctl(
        "gate",
        "open",
        "OTHER",
        "Protected decision?",
        "Wait for the human.",
    )

    assert opened.returncode == 0

    status = read_status()

    assert (
        status["state"]
        == "BLOCKED_HUMAN_DECISION"
    )

    status_path = (
        runtime_dir
        / "status.json"
    )

    audit_path = (
        runtime_dir
        / "audit.jsonl"
    )

    status_before = (
        status_path.read_bytes()
    )

    audit_before = (
        audit_path.read_bytes()
    )

    result = run_statusctl(
        "fail",
        "Attempt to bypass Human Gate.",
    )

    assert result.returncode != 0

    assert (
        status_path.read_bytes()
        == status_before
    )

    assert (
        audit_path.read_bytes()
        == audit_before
    )

    status = read_status()

    assert (
        status["state"]
        == "BLOCKED_HUMAN_DECISION"
    )

    assert status["human_gate"] is not None


def test_ready_work_task_can_start_separate_integration_task(
    run_statusctl,
    read_status,
    read_audit,
):
    _start_task(
        run_statusctl,
        task_id="work-task-001",
        title="Validated Work Task",
    )

    _make_ready(
        run_statusctl
    )

    work_status = read_status()

    assert (
        work_status["state"]
        == "READY_FOR_HUMAN_REVIEW"
    )

    assert (
        work_status["task_id"]
        == "work-task-001"
    )

    integration = run_statusctl(
        "start",
        "work-task-integration-001",
        "Integrate validated Work Task",
    )

    assert integration.returncode == 0, (
        integration.stderr
        or integration.stdout
    )

    status = read_status()

    assert status["state"] == "RUNNING"

    assert (
        status["task_id"]
        == "work-task-integration-001"
    )

    assert status["phase"] == "ANALYZE"

    assert status["completed_at"] is None

    assert status["human_gate"] is None

    assert (
        status["tests"]["targeted"]
        is None
    )

    assert (
        status["tests"]["full"]
        is None
    )

    events = read_audit()

    starts = [
        event
        for event in events
        if event["event"]
        == "TASK_STARTED"
    ]

    assert len(starts) == 2

    assert (
        starts[-1]["task_id"]
        == "work-task-integration-001"
    )


def test_failed_task_can_start_new_task(
    run_statusctl,
    read_status,
):
    _start_task(
        run_statusctl,
        task_id="failed-task-001",
        title="Task that will fail",
    )

    failed = run_statusctl(
        "fail",
        "Intentional lifecycle failure.",
    )

    assert failed.returncode == 0, (
        failed.stderr
        or failed.stdout
    )

    status = read_status()

    assert status["state"] == "FAILED"
    assert status["completed_at"] is not None

    restarted = run_statusctl(
        "start",
        "recovery-task-001",
        "Recovery task",
    )

    assert restarted.returncode == 0, (
        restarted.stderr
        or restarted.stdout
    )

    status = read_status()

    assert status["state"] == "RUNNING"

    assert (
        status["task_id"]
        == "recovery-task-001"
    )

    assert status["phase"] == "ANALYZE"

    assert status["completed_at"] is None
