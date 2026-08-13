from __future__ import annotations

import re
from pathlib import Path


GATE_ID_RE = re.compile(
    r"^HG-[0-9]{8}-[0-9]{3,}$"
)


def _start_task(
    run_statusctl,
) -> None:
    result = run_statusctl(
        "start",
        "gate-identity-test-001",
        "Human Gate identity regression test",
    )

    assert result.returncode == 0, (
        result.stderr
        or result.stdout
    )


def _open_gate(
    run_statusctl,
    read_status,
    question: str = "Test gate?",
) -> str:
    result = run_statusctl(
        "gate",
        "open",
        "OTHER",
        question,
        "Use exact active gate identity.",
        "--alternative",
        "Do not perform the test action.",
        "--risk",
        "A stale identity must never resolve another gate.",
    )

    assert result.returncode == 0, (
        result.stderr
        or result.stdout
    )

    status = read_status()

    assert (
        status["state"]
        == "BLOCKED_HUMAN_DECISION"
    )

    gate = status["human_gate"]

    assert gate is not None

    gate_id = gate["gate_id"]

    assert GATE_ID_RE.fullmatch(
        gate_id
    )

    return gate_id


def test_gate_open_assigns_identity(
    run_statusctl,
    read_status,
    read_audit,
):
    _start_task(
        run_statusctl
    )

    gate_id = _open_gate(
        run_statusctl,
        read_status,
    )

    status = read_status()

    assert (
        status["human_gate"]["gate_id"]
        == gate_id
    )

    events = read_audit()

    opened = [
        event
        for event in events
        if (
            event["event"]
            == "HUMAN_GATE_OPENED"
        )
    ]

    assert len(opened) == 1

    assert (
        opened[0]["gate_id"]
        == gate_id
    )


def test_resolution_without_active_gate_rejected(
    run_statusctl,
    read_status,
    read_audit,
):
    _start_task(
        run_statusctl
    )

    status_before = read_status()
    audit_before = read_audit()

    result = run_statusctl(
        "gate",
        "reject",
        "HG-19000101-999",
        "No gate is active.",
    )

    assert result.returncode != 0

    assert (
        read_status()
        == status_before
    )

    assert (
        read_audit()
        == audit_before
    )


def test_wrong_gate_id_rejected_without_mutation(
    run_statusctl,
    runtime_dir: Path,
    read_status,
):
    _start_task(
        run_statusctl
    )

    active_gate = _open_gate(
        run_statusctl,
        read_status,
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
        "reject",
        "HG-19000101-999",
        "Intentional stale gate identity.",
    )

    assert result.returncode != 0

    message = (
        result.stderr
        + result.stdout
    )

    assert (
        "Human Gate ID mismatch"
        in message
    )

    assert active_gate in message

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

    assert (
        status["human_gate"]["gate_id"]
        == active_gate
    )


def test_exact_gate_id_approve_succeeds(
    run_statusctl,
    read_status,
    read_audit,
):
    _start_task(
        run_statusctl
    )

    gate_id = _open_gate(
        run_statusctl,
        read_status,
    )

    result = run_statusctl(
        "gate",
        "approve",
        gate_id,
        "Human approved test action.",
    )

    assert result.returncode == 0, (
        result.stderr
        or result.stdout
    )

    status = read_status()

    assert status["state"] == "RUNNING"
    assert status["human_gate"] is None

    decision = (
        status["last_gate_decision"]
    )

    assert (
        decision["gate_id"]
        == gate_id
    )

    assert (
        decision["decision"]
        == "APPROVED"
    )

    events = read_audit()

    assert any(
        event["event"]
        == "HUMAN_GATE_APPROVED"
        and event["gate_id"]
        == gate_id
        for event in events
    )


def test_exact_gate_id_alternative_succeeds(
    run_statusctl,
    read_status,
):
    _start_task(
        run_statusctl
    )

    gate_id = _open_gate(
        run_statusctl,
        read_status,
    )

    result = run_statusctl(
        "gate",
        "alternative",
        gate_id,
        "Human selected test alternative.",
    )

    assert result.returncode == 0, (
        result.stderr
        or result.stdout
    )

    status = read_status()

    assert status["state"] == "RUNNING"
    assert status["human_gate"] is None

    decision = (
        status["last_gate_decision"]
    )

    assert (
        decision["gate_id"]
        == gate_id
    )

    assert (
        decision["decision"]
        == "ALTERNATIVE"
    )


def test_exact_gate_id_reject_succeeds(
    run_statusctl,
    read_status,
):
    _start_task(
        run_statusctl
    )

    gate_id = _open_gate(
        run_statusctl,
        read_status,
    )

    result = run_statusctl(
        "gate",
        "reject",
        gate_id,
        "Human rejected test action.",
    )

    assert result.returncode == 0, (
        result.stderr
        or result.stdout
    )

    status = read_status()

    assert status["state"] == "RUNNING"
    assert status["human_gate"] is None

    decision = (
        status["last_gate_decision"]
    )

    assert (
        decision["gate_id"]
        == gate_id
    )

    assert (
        decision["decision"]
        == "REJECTED"
    )


def test_old_gate_id_cannot_resolve_new_gate(
    run_statusctl,
    runtime_dir: Path,
    read_status,
):
    _start_task(
        run_statusctl
    )

    first_gate = _open_gate(
        run_statusctl,
        read_status,
        question="First gate?",
    )

    result = run_statusctl(
        "gate",
        "reject",
        first_gate,
        "Resolve first gate.",
    )

    assert result.returncode == 0

    second_gate = _open_gate(
        run_statusctl,
        read_status,
        question="Second gate?",
    )

    assert second_gate != first_gate

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

    stale_result = run_statusctl(
        "gate",
        "approve",
        first_gate,
        "Stale approval attempt.",
    )

    assert stale_result.returncode != 0

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

    assert (
        status["human_gate"]["gate_id"]
        == second_gate
    )


def test_wrong_resolution_verb_ids_all_fail_closed(
    run_statusctl,
    runtime_dir: Path,
    read_status,
):
    _start_task(
        run_statusctl
    )

    active_gate = _open_gate(
        run_statusctl,
        read_status,
    )

    status_path = (
        runtime_dir
        / "status.json"
    )

    audit_path = (
        runtime_dir
        / "audit.jsonl"
    )

    for verb, record in (
        (
            "approve",
            "Wrong approval ID.",
        ),
        (
            "alternative",
            "Wrong alternative ID.",
        ),
        (
            "reject",
            "Wrong rejection ID.",
        ),
    ):
        status_before = (
            status_path.read_bytes()
        )

        audit_before = (
            audit_path.read_bytes()
        )

        result = run_statusctl(
            "gate",
            verb,
            "HG-19000101-999",
            record,
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
            status["human_gate"]["gate_id"]
            == active_gate
        )
