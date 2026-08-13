from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _start_task(
    run_statusctl,
) -> None:
    result = run_statusctl(
        "start",
        "concurrency-test-001",
        "Concurrency regression test",
    )

    assert result.returncode == 0, (
        result.stderr
        or result.stdout
    )


def _lock_path(
    control_plane_repo: Path,
) -> Path:
    key = hashlib.sha256(
        str(
            control_plane_repo.resolve()
        ).encode("utf-8")
    ).hexdigest()[:16]

    return (
        Path(
            tempfile.gettempdir()
        )
        / f"cryptolab-statusctl-{key}.lock"
    )


def test_competing_gate_opens_are_serialized(
    control_plane_repo: Path,
    statusctl_path: Path,
    run_statusctl,
    read_status,
    read_audit,
):
    _start_task(
        run_statusctl
    )

    command_a = [
        str(statusctl_path),
        "gate",
        "open",
        "OTHER",
        "Concurrent gate A?",
        "Serialize concurrent gate requests.",
        "--risk",
        (
            "Only one concurrent request "
            "may transition RUNNING to BLOCKED."
        ),
    ]

    command_b = [
        str(statusctl_path),
        "gate",
        "open",
        "OTHER",
        "Concurrent gate B?",
        "Serialize concurrent gate requests.",
        "--risk",
        (
            "Only one concurrent request "
            "may transition RUNNING to BLOCKED."
        ),
    ]

    process_a = subprocess.Popen(
        command_a,
        cwd=control_plane_repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    process_b = subprocess.Popen(
        command_b,
        cwd=control_plane_repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout_a, stderr_a = (
        process_a.communicate(
            timeout=15,
        )
    )

    stdout_b, stderr_b = (
        process_b.communicate(
            timeout=15,
        )
    )

    returncodes = sorted(
        [
            process_a.returncode,
            process_b.returncode,
        ]
    )

    assert returncodes == [
        0,
        1,
    ]

    failed_output = ""

    if process_a.returncode != 0:
        failed_output = (
            stdout_a
            + stderr_a
        )

    if process_b.returncode != 0:
        failed_output = (
            stdout_b
            + stderr_b
        )

    assert (
        "Cannot open gate while "
        "state=BLOCKED_HUMAN_DECISION"
        in failed_output
    )

    status = read_status()

    assert (
        status["state"]
        == "BLOCKED_HUMAN_DECISION"
    )

    gate = status["human_gate"]

    assert gate is not None

    assert gate["question"] in {
        "Concurrent gate A?",
        "Concurrent gate B?",
    }

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
        == gate["gate_id"]
    )


def test_statusctl_waits_for_existing_os_lock(
    control_plane_repo: Path,
    statusctl_path: Path,
):
    lock_path = _lock_path(
        control_plane_repo
    )

    ready_path = (
        control_plane_repo
        / "lock-holder-ready"
    )

    holder_code = """
import fcntl
import sys
import time
from pathlib import Path

lock_path = Path(sys.argv[1])
ready_path = Path(sys.argv[2])

with lock_path.open("a+") as handle:
    fcntl.flock(
        handle.fileno(),
        fcntl.LOCK_EX,
    )

    ready_path.write_text(
        "locked\\n"
    )

    time.sleep(1.2)
"""

    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            holder_code,
            str(lock_path),
            str(ready_path),
        ],
        cwd=control_plane_repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    deadline = (
        time.monotonic()
        + 5.0
    )

    while (
        not ready_path.exists()
        and time.monotonic()
        < deadline
    ):
        time.sleep(
            0.01
        )

    assert ready_path.exists(), (
        "Lock-holder process did not "
        "acquire the test lock."
    )

    started = (
        time.monotonic()
    )

    result = subprocess.run(
        [
            str(statusctl_path),
            "show",
        ],
        cwd=control_plane_repo,
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )

    elapsed = (
        time.monotonic()
        - started
    )

    holder_stdout, holder_stderr = (
        holder.communicate(
            timeout=5,
        )
    )

    assert holder.returncode == 0, (
        holder_stderr
        or holder_stdout
    )

    assert result.returncode == 0, (
        result.stderr
        or result.stdout
    )

    assert elapsed >= 0.8

    assert (
        '"state": "IDLE"'
        in result.stdout
    )
