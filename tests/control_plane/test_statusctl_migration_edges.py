"""
Migration edge cases found by adversarial review of Hardening 4.2.

The schema-3.0 task model attaches a kind to a task. An IDLE runtime
has no task, and an IDLE state is unwritable by construction because
save() stamps updated_at while IDLE requires updated_at=null. Migration
must therefore refuse an IDLE legacy runtime explicitly, rather than
attempting a transform that can only fail deep inside validation with a
message that appears to blame the operator's file.
"""

from __future__ import annotations

import json
from pathlib import Path


def _write_idle_v2(
    runtime_dir: Path,
) -> bytes:
    runtime_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = {
        "schema_version": "2.0",
        "project": "CryptoLab-DataHub",
        "state": "IDLE",
        "task_id": None,
        "task_title": None,
        "phase": None,
        "branch": None,
        "started_at": None,
        "updated_at": None,
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


def test_migrate_refuses_idle_legacy_runtime(
    run_statusctl,
    runtime_dir: Path,
    read_audit,
):
    before = _write_idle_v2(
        runtime_dir
    )

    result = run_statusctl(
        "migrate-v3",
        "WORK",
    )

    assert result.returncode != 0

    output = (
        result.stdout
        + result.stderr
    ).lower()

    assert "idle" in output

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


def test_migrate_refuses_idle_legacy_runtime_for_integration(
    run_statusctl,
    runtime_dir: Path,
    read_audit,
):
    """
    The refusal cannot depend on the requested kind: an IDLE runtime
    has no task under either one.
    """

    before = _write_idle_v2(
        runtime_dir
    )

    result = run_statusctl(
        "migrate-v3",
        "INTEGRATION",
        "parent-work-001",
    )

    assert result.returncode != 0

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
