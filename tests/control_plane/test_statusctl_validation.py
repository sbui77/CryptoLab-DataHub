from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest


def _start_runtime(
    run_statusctl,
) -> None:
    result = run_statusctl(
        "start",
        "validation-test-001",
        "Runtime validation regression test",
    )

    assert result.returncode == 0, (
        result.stderr
        or result.stdout
    )


def _assert_validation_rejected(
    module,
    data: dict,
) -> None:
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


def _ready_candidate(
    base: dict,
) -> dict:
    data = deepcopy(base)

    data["state"] = (
        "READY_FOR_HUMAN_REVIEW"
    )

    data["phase"] = (
        "FINAL_VERIFY"
    )

    data["completed_at"] = (
        "2026-08-13T00:00:00Z"
    )

    data["human_gate"] = None

    data["tests"]["targeted"] = {
        "command": "pytest targeted",
        "result": "PASS",
        "summary": "targeted pass",
    }

    data["tests"]["full"] = {
        "command": "pytest full",
        "result": "PASS",
        "summary": "full pass",
    }

    return data


def _valid_gate() -> dict:
    return {
        "gate_id": (
            "HG-20260813-999"
        ),
        "type": "OTHER",
        "question": (
            "Test Human Gate?"
        ),
        "recommendation": (
            "Reject test action."
        ),
        "alternatives": [],
        "risks": [],
        "opened_at": (
            "2026-08-13T00:00:00Z"
        ),
    }


def test_current_running_state_validates(
    run_statusctl,
    read_status,
    load_statusctl_module,
):
    _start_runtime(
        run_statusctl
    )

    module = (
        load_statusctl_module()
    )

    module.validate_runtime_state(
        read_status()
    )


def test_ready_without_completed_at_rejected(
    run_statusctl,
    read_status,
    load_statusctl_module,
):
    _start_runtime(
        run_statusctl
    )

    module = (
        load_statusctl_module()
    )

    data = _ready_candidate(
        read_status()
    )

    data["completed_at"] = None

    _assert_validation_rejected(
        module,
        data,
    )


def test_running_with_human_gate_rejected(
    run_statusctl,
    read_status,
    load_statusctl_module,
):
    _start_runtime(
        run_statusctl
    )

    module = (
        load_statusctl_module()
    )

    data = deepcopy(
        read_status()
    )

    data["human_gate"] = (
        _valid_gate()
    )

    _assert_validation_rejected(
        module,
        data,
    )


def test_not_required_with_command_rejected(
    run_statusctl,
    read_status,
    load_statusctl_module,
):
    _start_runtime(
        run_statusctl
    )

    module = (
        load_statusctl_module()
    )

    data = deepcopy(
        read_status()
    )

    data["tests"]["targeted"] = {
        "command": "pytest something",
        "result": "NOT_REQUIRED",
        "summary": (
            "Invalid combination."
        ),
    }

    _assert_validation_rejected(
        module,
        data,
    )


def test_pass_with_null_command_rejected(
    run_statusctl,
    read_status,
    load_statusctl_module,
):
    _start_runtime(
        run_statusctl
    )

    module = (
        load_statusctl_module()
    )

    data = deepcopy(
        read_status()
    )

    data["tests"]["targeted"] = {
        "command": None,
        "result": "PASS",
        "summary": (
            "Invalid combination."
        ),
    }

    _assert_validation_rejected(
        module,
        data,
    )


def test_invalid_timestamp_rejected(
    run_statusctl,
    read_status,
    load_statusctl_module,
):
    _start_runtime(
        run_statusctl
    )

    module = (
        load_statusctl_module()
    )

    data = deepcopy(
        read_status()
    )

    data["started_at"] = (
        "not-a-timestamp"
    )

    _assert_validation_rejected(
        module,
        data,
    )


def test_extra_top_level_field_rejected(
    run_statusctl,
    read_status,
    load_statusctl_module,
):
    _start_runtime(
        run_statusctl
    )

    module = (
        load_statusctl_module()
    )

    data = deepcopy(
        read_status()
    )

    data["unexpected_field"] = True

    _assert_validation_rejected(
        module,
        data,
    )


def test_invalid_gate_id_rejected(
    run_statusctl,
    read_status,
    load_statusctl_module,
):
    _start_runtime(
        run_statusctl
    )

    module = (
        load_statusctl_module()
    )

    data = deepcopy(
        read_status()
    )

    gate = _valid_gate()

    gate["gate_id"] = (
        "BAD-GATE"
    )

    data["state"] = (
        "BLOCKED_HUMAN_DECISION"
    )

    data["human_gate"] = gate

    _assert_validation_rejected(
        module,
        data,
    )


def test_blocked_without_gate_rejected(
    run_statusctl,
    read_status,
    load_statusctl_module,
):
    _start_runtime(
        run_statusctl
    )

    module = (
        load_statusctl_module()
    )

    data = deepcopy(
        read_status()
    )

    data["state"] = (
        "BLOCKED_HUMAN_DECISION"
    )

    data["human_gate"] = None

    _assert_validation_rejected(
        module,
        data,
    )


def test_ready_with_failed_test_rejected(
    run_statusctl,
    read_status,
    load_statusctl_module,
):
    _start_runtime(
        run_statusctl
    )

    module = (
        load_statusctl_module()
    )

    data = _ready_candidate(
        read_status()
    )

    data["tests"]["targeted"] = {
        "command": "pytest targeted",
        "result": "FAIL",
        "summary": "failure",
    }

    _assert_validation_rejected(
        module,
        data,
    )


def test_not_required_ready_state_is_valid(
    run_statusctl,
    read_status,
    load_statusctl_module,
):
    _start_runtime(
        run_statusctl
    )

    module = (
        load_statusctl_module()
    )

    data = _ready_candidate(
        read_status()
    )

    data["tests"]["targeted"] = {
        "command": None,
        "result": "NOT_REQUIRED",
        "summary": (
            "Targeted test does not apply."
        ),
    }

    data["tests"]["full"] = {
        "command": None,
        "result": "NOT_REQUIRED",
        "summary": (
            "Full test does not apply."
        ),
    }

    module.validate_runtime_state(
        data
    )


def test_corrupted_load_fails_closed(
    run_statusctl,
    runtime_dir: Path,
):
    _start_runtime(
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

    report_path = (
        runtime_dir
        / "latest_report.md"
    )

    audit_before = (
        audit_path.read_bytes()
    )

    report_before = (
        report_path.read_bytes()
    )

    data = json.loads(
        status_path.read_text()
    )

    data["state"] = (
        "READY_FOR_HUMAN_REVIEW"
    )

    data["phase"] = (
        "FINAL_VERIFY"
    )

    data["completed_at"] = None

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
        in result.stderr
    )

    assert (
        audit_path.read_bytes()
        == audit_before
    )

    assert (
        report_path.read_bytes()
        == report_before
    )


def test_invalid_save_fails_closed(
    run_statusctl,
    runtime_dir: Path,
    load_statusctl_module,
):
    _start_runtime(
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

    report_path = (
        runtime_dir
        / "latest_report.md"
    )

    status_before = (
        status_path.read_bytes()
    )

    audit_before = (
        audit_path.read_bytes()
    )

    report_before = (
        report_path.read_bytes()
    )

    module = (
        load_statusctl_module()
    )

    data = json.loads(
        status_path.read_text()
    )

    data["state"] = (
        "READY_FOR_HUMAN_REVIEW"
    )

    data["phase"] = (
        "FINAL_VERIFY"
    )

    data["completed_at"] = None

    with pytest.raises(
        SystemExit,
    ) as exc_info:
        module.save(
            data
        )

    assert (
        "Runtime state validation failed:"
        in str(exc_info.value)
    )

    assert (
        status_path.read_bytes()
        == status_before
    )

    assert (
        audit_path.read_bytes()
        == audit_before
    )

    assert (
        report_path.read_bytes()
        == report_before
    )
