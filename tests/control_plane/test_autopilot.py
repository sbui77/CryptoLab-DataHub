"""
Hardening 4.2.1 — unattended autopilot contract.

Autopilot is the deterministic controller; Claude is the worker. These
tests pin the properties that make unattended operation safe:

    it never waits on an interactive approval;
    it never waits on stdin;
    it bounds every subprocess with a timeout;
    it surfaces worker failure instead of assuming success;
    it derives its next action from durable runtime state;
    it stops cleanly on a Human Gate;
    it refuses forbidden operations rather than working around them;
    it never resolves a gate and never migrates the runtime;
    it keeps .claude/runtime/latest_report.md canonical;
    it refuses to run twice against the same task.

External review added five more, which the later sections pin:

    ER-01 the real default worker runs under an explicit permission
          mode with pre-authorized capabilities, never a bypass;
    ER-02 the worker session cannot mutate lifecycle or evidence
          state, and gate requests come back structurally;
    ER-03 FINAL_VERIFY freshly records both verifications, fails
          closed without them, and gates `ready` behind them;
    ER-04 FINAL_VERIFY judges worktree integrity by an untracked-entry
          baseline rather than by `git diff --check` alone;
    SHOULD_FIX-A shell-wrapper escapes in injected commands fail
          closed instead of re-opening argv-only execution.

The worker is injected, so no test needs credentials, a network, or a
real Claude session.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import stat
import subprocess
import time
import uuid
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Callable

import pytest


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

AUTOPILOT_SOURCE = (
    PROJECT_ROOT
    / ".claude"
    / "bin"
    / "autopilot"
)


EXIT_OK = 0
EXIT_WAITING_FOR_HUMAN = 10
EXIT_BLOCKED_PERMISSION = 11
EXIT_BLOCKED_TECHNICAL = 12
EXIT_CONCURRENT = 13
EXIT_WAITING_EXTERNAL = 14


# ================================================================
# FIXTURES
# ================================================================


@pytest.fixture
def autopilot_repo(
    control_plane_repo: Path,
) -> Path:
    """
    The standard isolated control-plane repository, with autopilot
    installed alongside statusctl.
    """

    destination = (
        control_plane_repo
        / ".claude"
        / "bin"
        / "autopilot"
    )

    shutil.copy2(
        AUTOPILOT_SOURCE,
        destination,
    )

    destination.chmod(
        0o755
    )

    return control_plane_repo


@pytest.fixture
def make_worker(
    autopilot_repo: Path,
) -> Callable[..., list[str]]:
    """
    Write a stub worker and return the argv that invokes it.

    Injection keeps the worker deterministic and offline.
    """

    counter = {
        "n": 0,
    }

    def make(
        body: str,
    ) -> list[str]:
        counter["n"] += 1

        path = (
            autopilot_repo
            / f"stub_worker_{counter['n']}.py"
        )

        path.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys, os, time\n"
            + body
        )

        path.chmod(
            path.stat().st_mode
            | stat.S_IEXEC
        )

        return [
            "python3",
            str(path),
        ]

    return make


WORKER_PASS = (
    "print(json.dumps({'outcome': 'PASS', "
    "'summary': 'stub worker completed'}))\n"
)

WORKER_GATE = (
    "print(json.dumps({'outcome': 'HUMAN_GATE_REQUIRED', "
    "'summary': 'needs a decision', "
    "'gate_type': 'ARCHITECTURE_DECISION', "
    "'question': 'Approve?', "
    "'recommendation': 'Approve it.'}))\n"
)

WORKER_MALFORMED = (
    "print('this is not json at all')\n"
)

WORKER_NONZERO = (
    "sys.stderr.write('worker exploded\\n')\n"
    "sys.exit(3)\n"
)

WORKER_READS_STDIN = (
    "data = sys.stdin.read()\n"
    "print(json.dumps({'outcome': 'PASS', "
    "'summary': 'stdin len %d' % len(data)}))\n"
)

WORKER_SLEEPS = (
    "time.sleep(30)\n"
    "print(json.dumps({'outcome': 'PASS', 'summary': 'late'}))\n"
)

WORKER_RECORDS_ARGV = (
    "open(os.environ['WORKER_ARGV_LOG'], 'a').write("
    "json.dumps(sys.argv) + '\\n')\n"
    "print(json.dumps({'outcome': 'PASS', 'summary': 'recorded'}))\n"
)


# ER-02 backstop stubs.
#
# These deliberately bypass the read-only statusctl the worker session
# is given, by invoking the real controller through an absolute path
# the test hands them out of band. Nothing a real worker is configured
# with can reach the controller this way; the stubs exist to prove that
# autopilot still refuses to build on state it did not observe.
REAL_STATUSCTL_ENV = "REAL_STATUSCTL"

WORKER_OPENS_GATE_ITSELF = (
    "import subprocess\n"
    "subprocess.run([os.environ['REAL_STATUSCTL'], 'gate', 'open',\n"
    "  'SEMANTIC_DECISION', 'Which semantics?', 'Use A.'],\n"
    "  check=True, capture_output=True)\n"
    "print(json.dumps({'outcome': 'PASS', "
    "'summary': 'opened a gate and finished'}))\n"
)

# A worker that moves the phase is violating its brief: the
# orchestrator owns transitions.
WORKER_CHANGES_PHASE = (
    "import subprocess\n"
    "subprocess.run([os.environ['REAL_STATUSCTL'], 'phase',\n"
    "  'ADVERSARIAL_REVIEW'], check=True, capture_output=True)\n"
    "print(json.dumps({'outcome': 'PASS', "
    "'summary': 'moved the phase'}))\n"
)

WORKER_RECORDS_FAILING_TEST = (
    "import subprocess\n"
    "subprocess.run([os.environ['REAL_STATUSCTL'], 'test', "
    "'targeted',\n"
    "  'FAIL', 'pytest -q', 'two failed'],\n"
    "  check=True, capture_output=True)\n"
    "print(json.dumps({'outcome': 'PASS', "
    "'summary': 'claims success despite a recorded failure'}))\n"
)


def worker_attempting(argv_expression: str) -> str:
    """
    A worker stub that tries one command and records what happened.

    The attempt is never fatal to the stub: it always reports PASS, so
    a test can distinguish "the command was refused" from "the worker
    crashed", and can assert on autopilot's own behaviour afterwards.
    """

    return (
        "import subprocess\n"
        f"argv = {argv_expression}\n"
        "attempt = subprocess.run(\n"
        "  argv, capture_output=True, text=True)\n"
        "open(os.environ['ATTEMPT_LOG'], 'w').write(json.dumps({\n"
        "  'rc': attempt.returncode,\n"
        "  'stdout': attempt.stdout,\n"
        "  'stderr': attempt.stderr}))\n"
        "print(json.dumps({'outcome': 'PASS', "
        "'summary': 'attempted a controller command'}))\n"
    )


WORKER_RECORDS_ENVIRONMENT = (
    "open(os.environ['ENV_LOG'], 'w').write(json.dumps(sorted(\n"
    "  key for key in os.environ if key.startswith('AUTOPILOT_'))))\n"
    "print(json.dumps({'outcome': 'PASS', 'summary': 'env'}))\n"
)


def hermetic_environment() -> dict[str, str]:
    """
    The ambient environment with every AUTOPILOT_ variable removed.

    Autopilot is meant to be run unattended, which means this suite is
    itself executed inside an autopilot worker session, whose
    environment carries the orchestrator's configuration. Inheriting it
    would let the ambient run configure the autopilot under test —
    AUTOPILOT_WORKER_SESSION most sharply, since it makes every
    subprocess here refuse to start at all.

    A test that wants one of these variables sets it explicitly.
    """

    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("AUTOPILOT_")
    }


@pytest.fixture
def run_autopilot(
    autopilot_repo: Path,
) -> Callable[..., subprocess.CompletedProcess[str]]:
    def run(
        *args: str,
        worker: list[str] | None = None,
        timeout_seconds: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float = 60.0,
    ) -> subprocess.CompletedProcess[str]:
        environment = hermetic_environment()

        if worker is not None:
            environment[
                "AUTOPILOT_WORKER"
            ] = json.dumps(
                worker
            )

        if timeout_seconds is not None:
            environment[
                "AUTOPILOT_TIMEOUT"
            ] = timeout_seconds

        if env:
            environment.update(
                env
            )

        return subprocess.run(
            [
                str(
                    autopilot_repo
                    / ".claude"
                    / "bin"
                    / "autopilot"
                ),
                *args,
            ],
            cwd=autopilot_repo,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
            env=environment,
        )

    return run


@pytest.fixture
def load_autopilot_module(
    autopilot_repo: Path,
):
    def load():
        name = (
            "autopilot_test_"
            + uuid.uuid4().hex
        )

        loader = SourceFileLoader(
            name,
            str(
                autopilot_repo
                / ".claude"
                / "bin"
                / "autopilot"
            ),
        )

        spec = importlib.util.spec_from_loader(
            name,
            loader,
        )

        module = importlib.util.module_from_spec(
            spec
        )

        loader.exec_module(
            module
        )

        return module

    return load


def result_payload(
    completed: subprocess.CompletedProcess[str],
) -> dict:
    """
    Extract the machine-readable summary line.
    """

    for line in reversed(
        completed.stdout.splitlines()
    ):
        if line.startswith(
            "AUTOPILOT_RESULT "
        ):
            return json.loads(
                line.split(
                    " ",
                    1,
                )[1]
            )

    raise AssertionError(
        "no AUTOPILOT_RESULT line in stdout:\n"
        f"{completed.stdout}\n{completed.stderr}"
    )


def run_statusctl(
    autopilot_repo: Path,
    *args: str,
) -> subprocess.CompletedProcess[str]:
    """
    Drive the isolated controller directly, as an operator would.
    """

    result = subprocess.run(
        [
            str(
                autopilot_repo
                / ".claude"
                / "bin"
                / "statusctl"
            ),
            *args,
        ],
        cwd=autopilot_repo,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, (
        result.stderr
        or result.stdout
    )

    return result


def start_work_task(
    autopilot_repo: Path,
    task_id: str = "autopilot-work-001",
) -> None:
    run_statusctl(
        autopilot_repo,
        "start",
        task_id,
        "Autopilot work task",
    )


WORK_SEQUENCE = [
    "ANALYZE",
    "IMPLEMENT",
    "TARGETED_TEST",
    "FULL_TEST",
    "ADVERSARIAL_REVIEW",
    "CORRECT_FINDINGS",
    "RETEST",
    "EXTERNAL_REVIEW",
    "CORRECT_EXTERNAL_FINDINGS",
    "FINAL_VERIFY",
]


def advance_to(
    autopilot_repo: Path,
    phase: str,
) -> None:
    """
    Walk the controller to a phase the way an operator would.

    FINAL_VERIFY sits behind EXTERNAL_REVIEW, which autopilot refuses
    to drive through, so tests about completion have to arrive here
    without autopilot's help.
    """

    for step in WORK_SEQUENCE[
        1 : WORK_SEQUENCE.index(phase) + 1
    ]:
        run_statusctl(
            autopilot_repo,
            "phase",
            step,
        )


def read_status(
    autopilot_repo: Path,
) -> dict:
    return json.loads(
        (
            autopilot_repo
            / ".claude"
            / "runtime"
            / "status.json"
        ).read_text()
    )


def read_audit(
    autopilot_repo: Path,
) -> list[dict]:
    path = (
        autopilot_repo
        / ".claude"
        / "runtime"
        / "audit.jsonl"
    )

    if not path.exists():
        return []

    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


# ================================================================
# NON-INTERACTIVE EXECUTION
# ================================================================


def test_safe_autonomous_command_runs_without_interactive_input(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
):
    """
    A normal cycle completes with stdin closed and no approval.
    """

    start_work_task(
        autopilot_repo
    )

    result = run_autopilot(
        "run",
        "--max-cycles",
        "1",
        worker=make_worker(
            WORKER_PASS
        ),
    )

    assert result.returncode == EXIT_OK, (
        result.stderr
        or result.stdout
    )

    payload = result_payload(
        result
    )

    assert payload["result"] != "BLOCKED"


def test_worker_stdin_is_closed_and_cannot_block(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
):
    """
    A worker that reads stdin must see EOF immediately rather than
    stalling the unattended run forever.
    """

    start_work_task(
        autopilot_repo
    )

    started = time.monotonic()

    result = run_autopilot(
        "run",
        "--max-cycles",
        "1",
        worker=make_worker(
            WORKER_READS_STDIN
        ),
        timeout=30.0,
    )

    elapsed = time.monotonic() - started

    assert result.returncode == EXIT_OK, (
        result.stderr
        or result.stdout
    )

    assert elapsed < 25.0


def test_worker_timeout_is_enforced(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
):
    start_work_task(
        autopilot_repo
    )

    started = time.monotonic()

    result = run_autopilot(
        "run",
        "--max-cycles",
        "1",
        worker=make_worker(
            WORKER_SLEEPS
        ),
        timeout_seconds="2",
        timeout=40.0,
    )

    elapsed = time.monotonic() - started

    assert (
        result.returncode
        == EXIT_BLOCKED_TECHNICAL
    )

    assert elapsed < 25.0

    payload = result_payload(
        result
    )

    assert "timeout" in json.dumps(
        payload
    ).lower()


def test_worker_nonzero_exit_is_surfaced(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
    read_status_factory=None,
):
    start_work_task(
        autopilot_repo
    )

    result = run_autopilot(
        "run",
        "--max-cycles",
        "1",
        worker=make_worker(
            WORKER_NONZERO
        ),
    )

    assert (
        result.returncode
        == EXIT_BLOCKED_TECHNICAL
    )

    payload = result_payload(
        result
    )

    assert payload["result"] == "BLOCKED"

    assert (
        read_status(
            autopilot_repo
        )["phase"]
        == "ANALYZE"
    )


def test_malformed_worker_response_fails_closed(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
):
    """
    Unparseable worker output must never be read as success.
    """

    start_work_task(
        autopilot_repo
    )

    result = run_autopilot(
        "run",
        "--max-cycles",
        "1",
        worker=make_worker(
            WORKER_MALFORMED
        ),
    )

    assert (
        result.returncode
        == EXIT_BLOCKED_TECHNICAL
    )

    payload = result_payload(
        result
    )

    assert payload["result"] == "BLOCKED"

    assert (
        read_status(
            autopilot_repo
        )["phase"]
        == "ANALYZE"
    )


# ================================================================
# LIFECYCLE
# ================================================================


def test_lifecycle_advances_without_generic_approval(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
):
    """
    Ordinary autonomous work advances the phase using only
    pre-authorized capabilities.
    """

    start_work_task(
        autopilot_repo
    )

    result = run_autopilot(
        "run",
        "--max-cycles",
        "2",
        worker=make_worker(
            WORKER_PASS
        ),
    )

    assert result.returncode in (
        EXIT_OK,
        EXIT_WAITING_EXTERNAL,
    ), (
        result.stderr
        or result.stdout
    )

    assert (
        read_status(
            autopilot_repo
        )["phase"]
        != "ANALYZE"
    )


def test_human_gate_causes_clean_waiting_exit(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
):
    start_work_task(
        autopilot_repo
    )

    result = run_autopilot(
        "run",
        "--max-cycles",
        "3",
        worker=make_worker(
            WORKER_GATE
        ),
    )

    assert (
        result.returncode
        == EXIT_WAITING_FOR_HUMAN
    )

    status = read_status(
        autopilot_repo
    )

    assert (
        status["state"]
        == "BLOCKED_HUMAN_DECISION"
    )

    assert status["human_gate"] is not None

    payload = result_payload(
        result
    )

    assert (
        payload["human_gate"]["gate_id"]
        == status["human_gate"]["gate_id"]
    )


def test_open_gate_stops_immediately_on_next_run(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
):
    """
    A gate opened by anyone stops unattended execution, and the
    worker is not invoked again while it is open.
    """

    start_work_task(
        autopilot_repo
    )

    assert run_autopilot(
        "run",
        "--max-cycles",
        "3",
        worker=make_worker(
            WORKER_GATE
        ),
    ).returncode == EXIT_WAITING_FOR_HUMAN

    log = (
        autopilot_repo
        / "argv.log"
    )

    result = run_autopilot(
        "run",
        "--max-cycles",
        "3",
        worker=make_worker(
            WORKER_RECORDS_ARGV
        ),
        env={
            "WORKER_ARGV_LOG": str(
                log
            ),
        },
    )

    assert (
        result.returncode
        == EXIT_WAITING_FOR_HUMAN
    )

    assert not log.exists()


def test_external_review_stops_without_final_verify(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
):
    start_work_task(
        autopilot_repo
    )

    result = run_autopilot(
        "run",
        "--max-cycles",
        "20",
        worker=make_worker(
            WORKER_PASS
        ),
        timeout=180.0,
    )

    assert (
        result.returncode
        == EXIT_WAITING_EXTERNAL
    )

    status = read_status(
        autopilot_repo
    )

    assert (
        status["phase"]
        == "EXTERNAL_REVIEW"
    )

    assert status["state"] == "RUNNING"


def test_gate_opened_by_the_worker_is_refused_as_a_violation(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
):
    """
    ER-02: autopilot alone opens Human Gates.

    A worker that reaches the real controller out of band and opens a
    gate itself has mutated lifecycle state. Autopilot must refuse the
    cycle rather than treat the mutation as an ordinary stop, while
    still surfacing the gate a human now has to resolve.
    """

    start_work_task(
        autopilot_repo
    )

    result = run_autopilot(
        "run",
        "--max-cycles",
        "3",
        worker=make_worker(
            WORKER_OPENS_GATE_ITSELF
        ),
        env={
            REAL_STATUSCTL_ENV: str(
                autopilot_repo
                / ".claude"
                / "bin"
                / "statusctl"
            ),
        },
    )

    assert (
        result.returncode
        == EXIT_BLOCKED_PERMISSION
    ), (
        result.stdout
        or result.stderr
    )

    status = read_status(
        autopilot_repo
    )

    assert (
        status["state"]
        == "BLOCKED_HUMAN_DECISION"
    )

    # The phase must not have moved on from the gated cycle.
    assert status["phase"] == "ANALYZE"

    payload = result_payload(
        result
    )

    assert payload["result"] == "REFUSED"

    assert (
        payload["human_gate"]["gate_id"]
        == status["human_gate"]["gate_id"]
    )


def test_worker_changing_the_phase_is_refused(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
):
    """
    Phase transitions belong to the orchestrator alone.
    """

    start_work_task(
        autopilot_repo
    )

    result = run_autopilot(
        "run",
        "--max-cycles",
        "1",
        worker=make_worker(
            WORKER_CHANGES_PHASE
        ),
        env={
            REAL_STATUSCTL_ENV: str(
                autopilot_repo
                / ".claude"
                / "bin"
                / "statusctl"
            ),
        },
    )

    assert (
        result.returncode
        == EXIT_BLOCKED_PERMISSION
    )

    payload = result_payload(
        result
    )

    assert payload["result"] == "REFUSED"

    # Autopilot must not compound the violation by advancing from a
    # phase it never observed.
    assert (
        read_status(
            autopilot_repo
        )["phase"]
        == "ADVERSARIAL_REVIEW"
    )


def test_recorded_test_failure_stops_the_run(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
):
    """
    A worker's PASS never outranks a recorded FAIL.
    """

    start_work_task(
        autopilot_repo
    )

    result = run_autopilot(
        "run",
        "--max-cycles",
        "3",
        worker=make_worker(
            WORKER_RECORDS_FAILING_TEST
        ),
        env={
            REAL_STATUSCTL_ENV: str(
                autopilot_repo
                / ".claude"
                / "bin"
                / "statusctl"
            ),
        },
    )

    assert (
        result.returncode
        == EXIT_BLOCKED_PERMISSION
    )

    status = read_status(
        autopilot_repo
    )

    assert (
        status["tests"]["targeted"]["result"]
        == "FAIL"
    )

    assert status["phase"] == "ANALYZE"


def test_preexisting_test_failure_blocks_before_the_worker_runs(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
):
    """
    A FAIL recorded before the run stops it from durable state alone.

    This is the pre-worker half of the same rule: autopilot must not
    spend a worker cycle on a task whose recorded evidence already
    says it cannot advance.
    """

    start_work_task(
        autopilot_repo
    )

    run_statusctl(
        autopilot_repo,
        "test",
        "targeted",
        "FAIL",
        "pytest -q",
        "two failed",
    )

    log = (
        autopilot_repo
        / "argv.log"
    )

    result = run_autopilot(
        "run",
        "--max-cycles",
        "3",
        worker=make_worker(
            WORKER_RECORDS_ARGV
        ),
        env={
            "WORKER_ARGV_LOG": str(
                log
            ),
        },
    )

    assert (
        result.returncode
        == EXIT_BLOCKED_TECHNICAL
    )

    assert not log.exists()

    assert (
        read_status(
            autopilot_repo
        )["phase"]
        == "ANALYZE"
    )


# ================================================================
# RESTART / RESUME
# ================================================================


def test_resume_derives_action_from_durable_runtime(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
):
    """
    Autopilot keeps no conversational memory: a fresh process must
    reach the same conclusion from status.json alone.
    """

    start_work_task(
        autopilot_repo
    )

    assert run_autopilot(
        "run",
        "--max-cycles",
        "1",
        worker=make_worker(
            WORKER_PASS
        ),
    ).returncode == EXIT_OK

    phase_after_first = read_status(
        autopilot_repo
    )["phase"]

    resumed = run_autopilot(
        "resume",
        "--max-cycles",
        "1",
        worker=make_worker(
            WORKER_PASS
        ),
    )

    assert resumed.returncode in (
        EXIT_OK,
        EXIT_WAITING_EXTERNAL,
    )

    payload = result_payload(
        resumed
    )

    assert (
        payload["phase_before"]
        == phase_after_first
    )


def test_repeated_resume_at_terminal_state_is_idempotent(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
):
    start_work_task(
        autopilot_repo
    )

    first = run_autopilot(
        "run",
        "--max-cycles",
        "20",
        worker=make_worker(
            WORKER_PASS
        ),
        timeout=180.0,
    )

    assert (
        first.returncode
        == EXIT_WAITING_EXTERNAL
    )

    status_before = (
        autopilot_repo
        / ".claude"
        / "runtime"
        / "status.json"
    ).read_bytes()

    second = run_autopilot(
        "resume",
        "--max-cycles",
        "5",
        worker=make_worker(
            WORKER_PASS
        ),
    )

    third = run_autopilot(
        "resume",
        "--max-cycles",
        "5",
        worker=make_worker(
            WORKER_PASS
        ),
    )

    assert (
        second.returncode
        == EXIT_WAITING_EXTERNAL
    )

    assert (
        third.returncode
        == second.returncode
    )

    assert (
        autopilot_repo
        / ".claude"
        / "runtime"
        / "status.json"
    ).read_bytes() == status_before


def test_status_is_read_only(
    autopilot_repo: Path,
    run_autopilot,
):
    start_work_task(
        autopilot_repo
    )

    before = (
        autopilot_repo
        / ".claude"
        / "runtime"
        / "status.json"
    ).read_bytes()

    result = run_autopilot(
        "status"
    )

    assert result.returncode == EXIT_OK

    assert (
        autopilot_repo
        / ".claude"
        / "runtime"
        / "status.json"
    ).read_bytes() == before

    payload = result_payload(
        result
    )

    assert (
        payload["task"]
        == "autopilot-work-001"
    )


# ================================================================
# FORBIDDEN OPERATIONS
# ================================================================


def test_forbidden_commands_are_refused_by_the_guard(
    load_autopilot_module,
):
    module = load_autopilot_module()

    forbidden = [
        [
            "./.claude/bin/statusctl",
            "gate",
            "approve",
            "HG-20260813-001",
            "ok",
        ],
        [
            "./.claude/bin/statusctl",
            "gate",
            "alternative",
            "HG-20260813-001",
            "ok",
        ],
        [
            "./.claude/bin/statusctl",
            "gate",
            "reject",
            "HG-20260813-001",
            "no",
        ],
        [
            "./.claude/bin/statusctl",
            "migrate-v3",
            "WORK",
        ],
        [
            "git",
            "commit",
            "-m",
            "x",
        ],
        [
            "git",
            "push",
        ],
        [
            "git",
            "merge",
            "main",
        ],
        [
            "git",
            "push",
            "--force",
        ],
        [
            "sudo",
            "true",
        ],
        [
            "claude",
            "-p",
            "--dangerously-skip-permissions",
            "hi",
        ],
        [
            "claude",
            "-p",
            "--permission-mode",
            "bypassPermissions",
            "hi",
        ],
    ]

    for argv in forbidden:
        with pytest.raises(
            module.ForbiddenCommand
        ):
            module.assert_command_allowed(
                argv
            )


def test_allowed_commands_pass_the_guard(
    load_autopilot_module,
):
    module = load_autopilot_module()

    allowed = [
        [
            "./.claude/bin/statusctl",
            "show",
        ],
        [
            "./.claude/bin/statusctl",
            "gate",
            "open",
            "OTHER",
            "q",
            "r",
        ],
        [
            "git",
            "diff",
            "--check",
        ],
        [
            "git",
            "status",
            "--short",
        ],
        [
            "python3",
            "-m",
            "pytest",
            "-q",
        ],
    ]

    for argv in allowed:
        module.assert_command_allowed(
            argv
        )


def test_worker_argv_never_contains_bypass_flags(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
):
    log = (
        autopilot_repo
        / "argv.log"
    )

    start_work_task(
        autopilot_repo
    )

    result = run_autopilot(
        "run",
        "--max-cycles",
        "1",
        worker=make_worker(
            WORKER_RECORDS_ARGV
        ),
        env={
            "WORKER_ARGV_LOG": str(
                log
            ),
        },
    )

    assert result.returncode == EXIT_OK

    recorded = log.read_text()

    for flag in (
        "--dangerously-skip-permissions",
        "bypassPermissions",
        "--allow-dangerously-skip-permissions",
    ):
        assert flag not in recorded


def test_autopilot_never_resolves_gates_or_migrates(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
):
    start_work_task(
        autopilot_repo
    )

    run_autopilot(
        "run",
        "--max-cycles",
        "20",
        worker=make_worker(
            WORKER_PASS
        ),
        timeout=180.0,
    )

    events = {
        event["event"]
        for event in read_audit(
            autopilot_repo
        )
    }

    assert (
        "HUMAN_GATE_APPROVED"
        not in events
    )

    assert (
        "HUMAN_GATE_ALTERNATIVE_CHOSEN"
        not in events
    )

    assert (
        "HUMAN_GATE_REJECTED"
        not in events
    )

    assert (
        "RUNTIME_SCHEMA_MIGRATED"
        not in events
    )


def test_missing_worker_configuration_fails_closed(
    autopilot_repo: Path,
    run_autopilot,
):
    """
    An unusable worker must block, never silently self-approve.
    """

    start_work_task(
        autopilot_repo
    )

    result = run_autopilot(
        "run",
        "--max-cycles",
        "1",
        worker=[
            str(
                autopilot_repo
                / "does-not-exist"
            ),
        ],
    )

    assert result.returncode in (
        EXIT_BLOCKED_TECHNICAL,
        EXIT_BLOCKED_PERMISSION,
    )

    assert (
        read_status(
            autopilot_repo
        )["phase"]
        == "ANALYZE"
    )


# ================================================================
# CONCURRENCY
# ================================================================


def test_second_autopilot_refuses_to_drive_the_same_task(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
    load_autopilot_module,
):
    module = load_autopilot_module()

    lock_path = module.lock_path(
        autopilot_repo
    )

    start_work_task(
        autopilot_repo
    )

    holder = subprocess.Popen(
        [
            "python3",
            "-c",
            "import fcntl,sys,time\n"
            "h=open(sys.argv[1],'a+')\n"
            "fcntl.flock(h.fileno(), fcntl.LOCK_EX)\n"
            "time.sleep(8)\n",
            str(
                lock_path
            ),
        ],
    )

    try:
        time.sleep(
            0.5
        )

        result = run_autopilot(
            "run",
            "--max-cycles",
            "1",
            worker=make_worker(
                WORKER_PASS
            ),
            timeout=30.0,
        )

        assert (
            result.returncode
            == EXIT_CONCURRENT
        )

    finally:
        holder.kill()
        holder.wait()


# ================================================================
# REPORTING
# ================================================================


def test_canonical_report_is_the_only_report(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
):
    start_work_task(
        autopilot_repo
    )

    run_autopilot(
        "run",
        "--max-cycles",
        "3",
        worker=make_worker(
            WORKER_PASS
        ),
        timeout=120.0,
    )

    runtime = (
        autopilot_repo
        / ".claude"
        / "runtime"
    )

    assert (
        runtime
        / "latest_report.md"
    ).exists()

    assert not (
        autopilot_repo
        / ".claude"
        / "reports"
    ).exists()

    unexpected = [
        path.name
        for path in runtime.iterdir()
        if path.name
        not in {
            "status.json",
            "audit.jsonl",
            "latest_report.md",
        }
    ]

    assert not unexpected, unexpected


# ================================================================
# ER-01 — REAL WORKER PERMISSION POSTURE
# ================================================================


def default_worker_argv(
    module,
    monkeypatch,
) -> list[str]:
    monkeypatch.delenv(
        "AUTOPILOT_WORKER",
        raising=False,
    )

    return module.worker_argv()


def flag_value(
    argv: list[str],
    flag: str,
) -> str | None:
    for index, token in enumerate(
        argv
    ):
        if (
            token == flag
            and index + 1 < len(argv)
        ):
            return argv[index + 1]

    return None


def test_default_worker_enforces_dontask_permission_mode(
    load_autopilot_module,
    monkeypatch,
):
    """
    ER-01: the real worker runs under an explicit permission mode.

    Unattended operation must not depend on a human being present to
    answer a prompt, and must not buy that by disabling permissions.
    """

    module = load_autopilot_module()

    argv = default_worker_argv(
        module,
        monkeypatch,
    )

    assert argv[0] == "claude"

    assert (
        flag_value(
            argv,
            "--permission-mode",
        )
        == "dontAsk"
    )

    # The brief is appended after this argv, and --allowedTools /
    # --disallowedTools are variadic. A trailing `-p` is what stops
    # them from consuming the prompt.
    assert argv[-1] == "-p"


def test_default_worker_preauthorizes_safe_capabilities(
    load_autopilot_module,
    monkeypatch,
):
    """
    ER-01: implementation capabilities are named, not inherited.
    """

    module = load_autopilot_module()

    argv = default_worker_argv(
        module,
        monkeypatch,
    )

    allowed = flag_value(
        argv,
        "--allowedTools",
    )

    assert allowed is not None

    for capability in (
        "Read",
        "Grep",
        "Glob",
        "Edit",
        "Write",
    ):
        assert capability in allowed

    assert (
        "./.claude/bin/statusctl show"
        in allowed
    )

    disallowed = flag_value(
        argv,
        "--disallowedTools",
    )

    assert disallowed is not None

    for state_change in (
        "statusctl phase",
        "statusctl test",
        "statusctl gate",
        "statusctl ready",
        "statusctl fail",
        "statusctl migrate-v3",
    ):
        assert state_change in disallowed


def test_default_worker_carries_no_permission_bypass(
    load_autopilot_module,
    monkeypatch,
):
    """
    ER-01: no bypassPermissions, no dangerous skip, ever.
    """

    module = load_autopilot_module()

    argv = default_worker_argv(
        module,
        monkeypatch,
    )

    printable = " ".join(
        argv
    ).lower()

    assert (
        "bypasspermissions"
        not in printable
    )

    assert (
        "dangerously-skip-permissions"
        not in printable
    )

    module.assert_command_allowed(
        argv
    )

    module.assert_worker_command_allowed(
        argv
    )


def test_claude_worker_without_the_permission_mode_is_refused(
    load_autopilot_module,
):
    """
    ER-01: the posture is enforced, not merely defaulted.

    An injected claude worker that omits the permission mode or the
    capability allowlist would reintroduce interactive prompting into
    an unattended run.
    """

    module = load_autopilot_module()

    for argv in (
        [
            "claude",
            "-p",
        ],
        [
            "claude",
            "--permission-mode",
            "acceptEdits",
            "-p",
        ],
        [
            "claude",
            "--permission-mode",
            "dontAsk",
            "-p",
        ],
    ):
        with pytest.raises(
            module.ForbiddenCommand
        ):
            module.assert_worker_command_allowed(
                argv
            )


# ================================================================
# ER-02 — WORKER SESSION CANNOT MUTATE LIFECYCLE STATE
# ================================================================


def attempt_record(
    log: Path,
) -> dict:
    assert log.exists(), (
        "the worker never recorded its attempt"
    )

    return json.loads(
        log.read_text()
    )


@pytest.fixture
def attempt_log(
    autopilot_repo: Path,
) -> Path:
    return (
        autopilot_repo
        / "attempt.json"
    )


def run_with_attempt(
    run_autopilot,
    make_worker,
    attempt_log: Path,
    argv_expression: str,
    max_cycles: str = "1",
) -> subprocess.CompletedProcess[str]:
    return run_autopilot(
        "run",
        "--max-cycles",
        max_cycles,
        worker=make_worker(
            worker_attempting(
                argv_expression
            )
        ),
        env={
            "ATTEMPT_LOG": str(
                attempt_log
            ),
        },
    )


def test_worker_session_cannot_change_the_phase(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
    attempt_log: Path,
):
    """
    ER-02: a state-changing controller call fails inside the session.
    """

    start_work_task(
        autopilot_repo
    )

    result = run_with_attempt(
        run_autopilot,
        make_worker,
        attempt_log,
        "[os.environ['STATUSCTL'], 'phase', "
        "'ADVERSARIAL_REVIEW']",
    )

    attempt = attempt_record(
        attempt_log
    )

    assert attempt["rc"] != 0

    assert (
        "refused"
        in attempt["stderr"].lower()
    )

    # The cycle itself was ordinary work, so autopilot advances by
    # exactly one step — the step it owns.
    assert result.returncode == EXIT_OK, (
        result.stdout
        or result.stderr
    )

    assert (
        read_status(
            autopilot_repo
        )["phase"]
        == "IMPLEMENT"
    )


def test_worker_session_cannot_record_test_evidence(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
    attempt_log: Path,
):
    """
    ER-02: the worker cannot manufacture the evidence it is judged on.
    """

    start_work_task(
        autopilot_repo
    )

    run_with_attempt(
        run_autopilot,
        make_worker,
        attempt_log,
        "[os.environ['STATUSCTL'], 'test', 'targeted', "
        "'PASS', 'pytest -q', 'all green']",
    )

    assert (
        attempt_record(
            attempt_log
        )["rc"]
        != 0
    )

    assert (
        read_status(
            autopilot_repo
        )["tests"]["targeted"]
        is None
    )


def test_worker_session_cannot_open_a_human_gate(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
    attempt_log: Path,
):
    """
    ER-02: gate requests come back structurally; autopilot opens them.
    """

    start_work_task(
        autopilot_repo
    )

    run_with_attempt(
        run_autopilot,
        make_worker,
        attempt_log,
        "[os.environ['STATUSCTL'], 'gate', 'open', "
        "'SEMANTIC_DECISION', 'Which semantics?', 'Use A.']",
    )

    assert (
        attempt_record(
            attempt_log
        )["rc"]
        != 0
    )

    assert (
        read_status(
            autopilot_repo
        )["human_gate"]
        is None
    )

    assert not any(
        event["event"]
        == "HUMAN_GATE_OPENED"
        for event in read_audit(
            autopilot_repo
        )
    )


def test_worker_session_cannot_resolve_or_migrate(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
    attempt_log: Path,
):
    """
    ER-02: human and operator authority stay outside the session.
    """

    start_work_task(
        autopilot_repo
    )

    for expression in (
        "[os.environ['STATUSCTL'], 'gate', 'approve', "
        "'HG-20260813-001', 'ok']",
        "[os.environ['STATUSCTL'], 'migrate-v3', 'WORK']",
        "[os.environ['STATUSCTL'], 'migrate-v31']",
        "[os.environ['STATUSCTL'], 'integration', 'performed', "
        "'commit']",
        "[os.environ['STATUSCTL'], 'ready']",
        "[os.environ['STATUSCTL'], 'fail', 'giving up']",
    ):
        attempt_log.unlink(
            missing_ok=True
        )

        run_with_attempt(
            run_autopilot,
            make_worker,
            attempt_log,
            expression,
        )

        assert (
            attempt_record(
                attempt_log
            )["rc"]
            != 0
        ), expression

    events = {
        event["event"]
        for event in read_audit(
            autopilot_repo
        )
    }

    assert not (
        events
        & {
            "HUMAN_GATE_APPROVED",
            "RUNTIME_SCHEMA_MIGRATED",
            "TASK_READY_FOR_HUMAN_REVIEW",
            "TASK_FAILED",
        }
    )


def test_worker_session_statusctl_on_path_is_read_only(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
    attempt_log: Path,
):
    """
    ER-02: PATH resolution is closed too, not only the env pointer.
    """

    start_work_task(
        autopilot_repo
    )

    run_with_attempt(
        run_autopilot,
        make_worker,
        attempt_log,
        "['statusctl', 'phase', 'IMPLEMENT']",
    )

    assert (
        attempt_record(
            attempt_log
        )["rc"]
        != 0
    )


def test_worker_session_can_still_read_runtime_state(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
    attempt_log: Path,
):
    """
    ER-02 must restrict writes without blinding the worker.
    """

    start_work_task(
        autopilot_repo
    )

    run_with_attempt(
        run_autopilot,
        make_worker,
        attempt_log,
        "[os.environ['STATUSCTL'], 'show']",
    )

    attempt = attempt_record(
        attempt_log
    )

    assert attempt["rc"] == 0, attempt

    assert (
        json.loads(
            attempt["stdout"]
        )["task_id"]
        == "autopilot-work-001"
    )


def test_worker_session_sees_no_autopilot_configuration(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
):
    """
    ER-02: the worker cannot read or re-enter the orchestrator's own
    configuration, so it cannot relaunch autopilot with a worker of
    its choosing.
    """

    log = (
        autopilot_repo
        / "env.json"
    )

    start_work_task(
        autopilot_repo
    )

    run_autopilot(
        "run",
        "--max-cycles",
        "1",
        worker=make_worker(
            WORKER_RECORDS_ENVIRONMENT
        ),
        env={
            "ENV_LOG": str(
                log
            ),
            "AUTOPILOT_VERIFY_FULL": json.dumps(
                [
                    "true",
                ]
            ),
        },
    )

    assert json.loads(
        log.read_text()
    ) == [
        "AUTOPILOT_WORKER_SESSION",
    ]


def test_autopilot_refuses_to_run_inside_a_worker_session(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
):
    """
    ER-02: no recursion out of the restricted session.
    """

    start_work_task(
        autopilot_repo
    )

    result = run_autopilot(
        "run",
        "--max-cycles",
        "1",
        worker=make_worker(
            WORKER_PASS
        ),
        env={
            "AUTOPILOT_WORKER_SESSION": "1",
        },
    )

    assert (
        result.returncode
        == EXIT_BLOCKED_PERMISSION
    )

    assert (
        read_status(
            autopilot_repo
        )["phase"]
        == "ANALYZE"
    )


def test_suite_is_hermetic_against_an_ambient_autopilot_run(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
    monkeypatch,
):
    """
    ER-02 follow-on: the ambient orchestrator must not configure the
    autopilot under test.

    The refusal above is correct behaviour, and that is exactly why it
    is a hazard here: an unattended run executes this suite inside a
    worker session, so every subprocess would inherit the marker and
    refuse, turning the whole suite red for an environmental reason.
    The timeout is set alongside it because configuration leaks the
    same way — inherited, it would expire the worker mid-cycle.
    """

    monkeypatch.setenv(
        "AUTOPILOT_WORKER_SESSION",
        "1",
    )

    monkeypatch.setenv(
        "AUTOPILOT_TIMEOUT",
        "0.01",
    )

    start_work_task(
        autopilot_repo
    )

    result = run_autopilot(
        "run",
        "--max-cycles",
        "1",
        worker=make_worker(
            WORKER_PASS
        ),
    )

    assert result.returncode == EXIT_OK, (
        result.stdout
        or result.stderr
    )

    assert (
        read_status(
            autopilot_repo
        )["phase"]
        == "IMPLEMENT"
    )


# ================================================================
# ER-03 — FINAL VERIFY IS FRESH, COMPLETE, AND GATES READY
# ================================================================


@pytest.fixture
def make_script(
    autopilot_repo: Path,
) -> Callable[..., list[str]]:
    """
    Write a stub verification command and return its argv.
    """

    counter = {
        "n": 0,
    }

    def make(
        body: str,
    ) -> list[str]:
        counter["n"] += 1

        path = (
            autopilot_repo
            / f"stub_verify_{counter['n']}.py"
        )

        path.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            + body
        )

        path.chmod(
            path.stat().st_mode
            | stat.S_IEXEC
        )

        return [
            "python3",
            str(path),
        ]

    return make


VERIFY_PASS = "print('7 passed in 1.20s')\n"

VERIFY_FAIL = (
    "print('1 failed, 6 passed in 1.20s')\n"
    "sys.exit(1)\n"
)


def verify_env(
    targeted: list[str] | None = None,
    full: list[str] | None = None,
) -> dict[str, str]:
    env = {}

    if targeted is not None:
        env[
            "AUTOPILOT_VERIFY_TARGETED"
        ] = json.dumps(
            targeted
        )

    if full is not None:
        env[
            "AUTOPILOT_VERIFY_FULL"
        ] = json.dumps(
            full
        )

    return env


def seed_stale_results(
    autopilot_repo: Path,
) -> None:
    """
    Record passing results long before FINAL_VERIFY.

    These satisfy statusctl's `ready` preconditions on their own, so
    they are exactly what a stale-evidence completion would ride on.
    """

    for kind in (
        "targeted",
        "full",
    ):
        run_statusctl(
            autopilot_repo,
            "test",
            kind,
            "PASS",
            "pytest -q",
            "stale result recorded much earlier",
        )


def test_final_verify_fails_closed_without_verification_commands(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
):
    """
    ER-03: missing verification configuration must block `ready`.

    Stale passing records are seeded deliberately: without this rule
    they alone would carry the task into READY_FOR_HUMAN_REVIEW.
    """

    start_work_task(
        autopilot_repo
    )

    seed_stale_results(
        autopilot_repo
    )

    advance_to(
        autopilot_repo,
        "FINAL_VERIFY",
    )

    result = run_autopilot(
        "run",
        "--max-cycles",
        "1",
        worker=make_worker(
            WORKER_PASS
        ),
    )

    assert (
        result.returncode
        == EXIT_BLOCKED_TECHNICAL
    ), (
        result.stdout
        or result.stderr
    )

    status = read_status(
        autopilot_repo
    )

    assert status["state"] == "RUNNING"

    assert status["completed_at"] is None


def test_final_verify_fails_closed_when_only_one_is_configured(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
    make_script,
):
    """
    ER-03: both verifications are required, not either.
    """

    start_work_task(
        autopilot_repo
    )

    seed_stale_results(
        autopilot_repo
    )

    advance_to(
        autopilot_repo,
        "FINAL_VERIFY",
    )

    result = run_autopilot(
        "run",
        "--max-cycles",
        "1",
        worker=make_worker(
            WORKER_PASS
        ),
        env=verify_env(
            targeted=make_script(
                VERIFY_PASS
            ),
        ),
    )

    assert (
        result.returncode
        == EXIT_BLOCKED_TECHNICAL
    )

    assert (
        read_status(
            autopilot_repo
        )["state"]
        == "RUNNING"
    )


def test_final_verify_records_both_results_freshly_then_ready(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
    make_script,
):
    """
    ER-03: both results are re-established at FINAL_VERIFY.

    The seeded stale records must be overwritten by results the
    orchestrator produced in this phase, and only then may the task
    become READY_FOR_HUMAN_REVIEW.
    """

    start_work_task(
        autopilot_repo
    )

    seed_stale_results(
        autopilot_repo
    )

    advance_to(
        autopilot_repo,
        "FINAL_VERIFY",
    )

    targeted = make_script(
        VERIFY_PASS
    )

    full = make_script(
        VERIFY_PASS
    )

    result = run_autopilot(
        "run",
        "--max-cycles",
        "1",
        worker=make_worker(
            WORKER_PASS
        ),
        env=verify_env(
            targeted=targeted,
            full=full,
        ),
    )

    assert result.returncode == EXIT_OK, (
        result.stdout
        or result.stderr
    )

    status = read_status(
        autopilot_repo
    )

    assert (
        status["state"]
        == "READY_FOR_HUMAN_REVIEW"
    )

    assert status["completed_at"] is not None

    for kind, argv in (
        (
            "targeted",
            targeted,
        ),
        (
            "full",
            full,
        ),
    ):
        record = status["tests"][kind]

        assert record["result"] == "PASS"

        assert record["command"] == " ".join(
            argv
        )

        assert (
            "FINAL_VERIFY"
            in record["summary"]
        )

        assert (
            "stale"
            not in record["summary"]
        )


def test_final_verify_failure_prevents_ready(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
    make_script,
):
    """
    ER-03: either verification failing stops completion.
    """

    start_work_task(
        autopilot_repo
    )

    seed_stale_results(
        autopilot_repo
    )

    advance_to(
        autopilot_repo,
        "FINAL_VERIFY",
    )

    result = run_autopilot(
        "run",
        "--max-cycles",
        "1",
        worker=make_worker(
            WORKER_PASS
        ),
        env=verify_env(
            targeted=make_script(
                VERIFY_PASS
            ),
            full=make_script(
                VERIFY_FAIL
            ),
        ),
    )

    assert (
        result.returncode
        == EXIT_BLOCKED_TECHNICAL
    )

    status = read_status(
        autopilot_repo
    )

    assert status["state"] == "RUNNING"

    assert (
        status["tests"]["full"]["result"]
        == "FAIL"
    )

    assert status["completed_at"] is None


def test_final_verify_stops_on_worktree_damage_before_ready(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
    make_script,
):
    """
    ER-03: git diff --check runs after verification and gates `ready`.
    """

    start_work_task(
        autopilot_repo
    )

    seed_stale_results(
        autopilot_repo
    )

    advance_to(
        autopilot_repo,
        "FINAL_VERIFY",
    )

    damaged = (
        autopilot_repo
        / "damaged.txt"
    )

    damaged.write_text(
        "clean line\n"
    )

    subprocess.run(
        [
            "git",
            "add",
            "damaged.txt",
        ],
        cwd=autopilot_repo,
        check=True,
        capture_output=True,
    )

    damaged.write_text(
        "clean line\ntrailing whitespace   \n"
    )

    result = run_autopilot(
        "run",
        "--max-cycles",
        "1",
        worker=make_worker(
            WORKER_PASS
        ),
        env=verify_env(
            targeted=make_script(
                VERIFY_PASS
            ),
            full=make_script(
                VERIFY_PASS
            ),
        ),
    )

    assert (
        result.returncode
        == EXIT_BLOCKED_TECHNICAL
    )

    status = read_status(
        autopilot_repo
    )

    assert status["state"] == "RUNNING"

    # Verification still happened and was recorded honestly.
    assert (
        status["tests"]["full"]["result"]
        == "PASS"
    )


def test_integration_final_verify_does_not_retest_product_code(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
):
    """
    ER-03's fresh-verification rule is a Work Task rule.

    An Integration Task never re-tests product code (CLAUDE.md 2A),
    so autopilot runs no verification for it and configures none. The
    exemption is safe because completion still has to satisfy
    statusctl's own fail-closed `ready` checks against records
    autopilot did not write.
    """

    start_work_task(
        autopilot_repo
    )

    seed_stale_results(
        autopilot_repo
    )

    advance_to(
        autopilot_repo,
        "FINAL_VERIFY",
    )

    run_statusctl(
        autopilot_repo,
        "ready",
    )

    run_statusctl(
        autopilot_repo,
        "start-integration",
        "autopilot-integration-001",
        "Integrate the reviewed work",
        "--action",
        "none",
    )

    run_statusctl(
        autopilot_repo,
        "phase",
        "FINAL_VERIFY",
    )

    # The exemption is not a hole: an Integration Task starts with no
    # test records of its own, and statusctl refuses to complete one
    # that has none. Autopilot cannot supply them, because deciding a
    # test category does not apply is a judgement about the task.
    blocked_run = run_autopilot(
        "run",
        "--max-cycles",
        "1",
        worker=make_worker(
            WORKER_PASS
        ),
    )

    assert (
        blocked_run.returncode
        == EXIT_BLOCKED_TECHNICAL
    )

    assert (
        read_status(
            autopilot_repo
        )["state"]
        == "RUNNING"
    )

    for kind in (
        "targeted",
        "full",
    ):
        run_statusctl(
            autopilot_repo,
            "test-not-required",
            kind,
            "Integration Task never re-tests product code.",
        )

    result = run_autopilot(
        "run",
        "--max-cycles",
        "1",
        worker=make_worker(
            WORKER_PASS
        ),
    )

    assert result.returncode == EXIT_OK, (
        result.stdout
        or result.stderr
    )

    status = read_status(
        autopilot_repo
    )

    assert (
        status["state"]
        == "READY_FOR_HUMAN_REVIEW"
    )

    assert status["task_kind"] == "INTEGRATION"

    assert (
        status["parent_task_id"]
        == "autopilot-work-001"
    )


def test_final_verify_refreshes_the_canonical_report(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
    make_script,
):
    """
    ER-03: the report a human reads describes the completed task.
    """

    start_work_task(
        autopilot_repo
    )

    seed_stale_results(
        autopilot_repo
    )

    advance_to(
        autopilot_repo,
        "FINAL_VERIFY",
    )

    report = (
        autopilot_repo
        / ".claude"
        / "runtime"
        / "latest_report.md"
    )

    report.unlink(
        missing_ok=True
    )

    result = run_autopilot(
        "run",
        "--max-cycles",
        "1",
        worker=make_worker(
            WORKER_PASS
        ),
        env=verify_env(
            targeted=make_script(
                VERIFY_PASS
            ),
            full=make_script(
                VERIFY_PASS
            ),
        ),
    )

    assert result.returncode == EXIT_OK

    assert report.exists()

    assert (
        "READY_FOR_HUMAN_REVIEW"
        in report.read_text()
    )


# ================================================================
# ER-04 — WORKTREE INTEGRITY BEYOND git diff --check
# ================================================================
#
# `git diff --check` inspects tracked content, so a completion check
# that stops there is blind to whatever a run leaves beside the
# repository. These pin the untracked-entry baseline: taken once at
# process start, compared again before `ready`, never refreshed in
# between, and exempting only the non-regular entries a sandbox masks
# onto /dev/null.


WORKER_CONTAMINATES = (
    "for spec in json.loads(os.environ['CONTAMINATION']):\n"
    "    kind, path = spec['kind'], spec['path']\n"
    "    if kind == 'file':\n"
    "        open(path, 'w').write(spec.get('content', ''))\n"
    "    elif kind == 'dir':\n"
    "        os.makedirs(path, exist_ok=True)\n"
    "    elif kind == 'tree':\n"
    "        os.makedirs(path, exist_ok=True)\n"
    "        open(os.path.join(path, 'note.txt'), 'w').write('x\\n')\n"
    "    elif kind == 'mask':\n"
    "        if not os.path.lexists(path):\n"
    "            os.symlink('/dev/null', path)\n"
    "print(json.dumps({'outcome': 'PASS', "
    "'summary': 'created worktree entries'}))\n"
)


def contaminate(
    *specs: tuple[str, str],
) -> dict[str, str]:
    """
    Ask the worker stub to create entries beside the repository.

    Each spec is (kind, path), where kind is file, dir, tree or mask.
    Creation is idempotent so a stub can run in more than one cycle.
    """

    return {
        "CONTAMINATION": json.dumps(
            [
                {
                    "kind": kind,
                    "path": path,
                }
                for kind, path in specs
            ]
        ),
    }


def at_final_verify(
    autopilot_repo: Path,
) -> None:
    """
    Put a Work Task one cycle away from completion.
    """

    start_work_task(
        autopilot_repo
    )

    seed_stale_results(
        autopilot_repo
    )

    advance_to(
        autopilot_repo,
        "FINAL_VERIFY",
    )


def run_to_completion(
    run_autopilot,
    make_worker,
    make_script,
    worker_body: str = WORKER_PASS,
    env: dict[str, str] | None = None,
    max_cycles: str = "1",
) -> subprocess.CompletedProcess[str]:
    """
    Drive the final cycle with both verifications configured.

    Verification is made to pass deliberately: with ER-03 satisfied,
    the only thing left that can stop completion is the worktree.
    """

    environment = verify_env(
        targeted=make_script(
            VERIFY_PASS
        ),
        full=make_script(
            VERIFY_PASS
        ),
    )

    environment.update(
        env
        or {}
    )

    return run_autopilot(
        "run",
        "--max-cycles",
        max_cycles,
        worker=make_worker(
            worker_body
        ),
        env=environment,
    )


def untracked_entries(
    autopilot_repo: Path,
) -> list[str]:
    """
    What git itself reports, independently of autopilot.
    """

    completed = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--directory",
            "--exclude-standard",
            "-z",
        ],
        cwd=autopilot_repo,
        text=True,
        capture_output=True,
        check=True,
    )

    return [
        entry
        for entry in completed.stdout.split(
            "\0"
        )
        if entry
    ]


def git_diff_check(
    autopilot_repo: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "git",
            "diff",
            "--check",
        ],
        cwd=autopilot_repo,
        text=True,
        capture_output=True,
        check=False,
    )


def assert_completion_blocked(
    autopilot_repo: Path,
    result: subprocess.CompletedProcess[str],
) -> dict:
    assert (
        result.returncode
        == EXIT_BLOCKED_TECHNICAL
    ), (
        result.stdout
        or result.stderr
    )

    status = read_status(
        autopilot_repo
    )

    assert status["state"] == "RUNNING"

    assert status["completed_at"] is None

    return result_payload(
        result
    )


def test_empty_untracked_files_appearing_block_completion(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
    make_script,
):
    """
    ER-04: an empty file is still an artifact.

    Emptiness is the tempting exemption and the wrong one: these are
    the names a sandbox also masks, and a completion check that waved
    them through by size could be fed a real file of length zero.
    """

    at_final_verify(
        autopilot_repo
    )

    result = run_to_completion(
        run_autopilot,
        make_worker,
        make_script,
        worker_body=WORKER_CONTAMINATES,
        env=contaminate(
            (
                "file",
                ".bashrc",
            ),
            (
                "file",
                ".gitconfig",
            ),
        ),
    )

    payload = assert_completion_blocked(
        autopilot_repo,
        result,
    )

    assert set(
        payload["contamination"]
    ) == {
        ".bashrc",
        ".gitconfig",
    }


def test_empty_untracked_directories_appearing_block_completion(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
    make_script,
):
    """
    ER-04: an empty directory is the case `git status` cannot see.

    git reports no untracked *file* here at all, which is precisely
    why the check has to be built on the entry inventory rather than
    on a status or diff summary.
    """

    at_final_verify(
        autopilot_repo
    )

    result = run_to_completion(
        run_autopilot,
        make_worker,
        make_script,
        worker_body=WORKER_CONTAMINATES,
        env=contaminate(
            (
                "dir",
                ".idea",
            ),
            (
                "dir",
                ".vscode",
            ),
        ),
    )

    payload = assert_completion_blocked(
        autopilot_repo,
        result,
    )

    assert set(
        payload["contamination"]
    ) == {
        ".idea/",
        ".vscode/",
    }

    assert (
        ".idea"
        not in subprocess.run(
            [
                "git",
                "status",
                "--short",
            ],
            cwd=autopilot_repo,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
    )


def test_non_empty_untracked_directory_blocks_completion(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
    make_script,
):
    """
    ER-04: a directory carrying real content blocks too.
    """

    at_final_verify(
        autopilot_repo
    )

    result = run_to_completion(
        run_autopilot,
        make_worker,
        make_script,
        worker_body=WORKER_CONTAMINATES,
        env=contaminate(
            (
                "tree",
                "scratch",
            ),
        ),
    )

    payload = assert_completion_blocked(
        autopilot_repo,
        result,
    )

    assert payload["contamination"] == [
        "scratch/",
    ]


def test_contamination_blocks_even_when_git_diff_check_is_clean(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
    make_script,
):
    """
    ER-04, stated directly: the old check passing proves nothing.

    The worktree damage `git diff --check` looks for is absent here,
    so a completion gated on it alone would let the run finish with
    a stray file sitting in the tree.
    """

    at_final_verify(
        autopilot_repo
    )

    result = run_to_completion(
        run_autopilot,
        make_worker,
        make_script,
        worker_body=WORKER_CONTAMINATES,
        env=contaminate(
            (
                "file",
                "leftover.txt",
            ),
        ),
    )

    assert (
        git_diff_check(
            autopilot_repo
        ).returncode
        == 0
    )

    payload = assert_completion_blocked(
        autopilot_repo,
        result,
    )

    assert payload["contamination"] == [
        "leftover.txt",
    ]


def test_sandbox_mask_entries_are_not_contamination(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
    make_script,
):
    """
    ER-04: only non-regular entries are exempt.

    A denied path masked onto /dev/null is the sandbox's doing, not
    the work's. git still lists it — asserted here, so the test
    proves the exemption rather than git's silence.
    """

    at_final_verify(
        autopilot_repo
    )

    result = run_to_completion(
        run_autopilot,
        make_worker,
        make_script,
        worker_body=WORKER_CONTAMINATES,
        env=contaminate(
            (
                "mask",
                ".bashrc",
            ),
            (
                "mask",
                ".gitconfig",
            ),
            (
                "mask",
                ".vscode",
            ),
        ),
    )

    listed = untracked_entries(
        autopilot_repo
    )

    for name in (
        ".bashrc",
        ".gitconfig",
        ".vscode",
    ):
        assert name in listed, listed

    assert result.returncode == EXIT_OK, (
        result.stdout
        or result.stderr
    )

    assert (
        read_status(
            autopilot_repo
        )["state"]
        == "READY_FOR_HUMAN_REVIEW"
    )


def test_baseline_untracked_artifacts_still_complete(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
    make_script,
):
    """
    ER-04 must not forbid working untracked material.

    A task legitimately begins with scratch files and directories in
    the tree. What is judged is what appeared during the run, so
    everything the baseline already carried stays allowed — including
    a file whose contents change while the run proceeds.
    """

    at_final_verify(
        autopilot_repo
    )

    (
        autopilot_repo
        / "notes.md"
    ).write_text(
        "working notes\n"
    )

    (
        autopilot_repo
        / "scratch"
    ).mkdir()

    (
        autopilot_repo
        / "scratch"
        / "draft.txt"
    ).write_text(
        "draft\n"
    )

    result = run_to_completion(
        run_autopilot,
        make_worker,
        make_script,
        worker_body=WORKER_CONTAMINATES,
        env=contaminate(
            (
                "file",
                "notes.md",
            ),
            (
                "tree",
                "scratch",
            ),
        ),
    )

    assert result.returncode == EXIT_OK, (
        result.stdout
        or result.stderr
    )

    assert (
        read_status(
            autopilot_repo
        )["state"]
        == "READY_FOR_HUMAN_REVIEW"
    )


def test_clean_baseline_run_completes_unchanged(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
    make_script,
):
    """
    ER-04: the check is a gate, not a wall.

    A run that leaves the untracked set exactly as it found it
    completes, and the inventory is unchanged afterwards.
    """

    at_final_verify(
        autopilot_repo
    )

    # The stub worker and verification scripts are themselves
    # untracked, so the baseline is taken once they exist — the same
    # moment autopilot takes its own.
    worker = make_worker(
        WORKER_PASS
    )

    environment = verify_env(
        targeted=make_script(
            VERIFY_PASS
        ),
        full=make_script(
            VERIFY_PASS
        ),
    )

    before = untracked_entries(
        autopilot_repo
    )

    result = run_autopilot(
        "run",
        "--max-cycles",
        "1",
        worker=worker,
        env=environment,
    )

    assert result.returncode == EXIT_OK, (
        result.stdout
        or result.stderr
    )

    assert (
        read_status(
            autopilot_repo
        )["state"]
        == "READY_FOR_HUMAN_REVIEW"
    )

    assert (
        untracked_entries(
            autopilot_repo
        )
        == before
    )


def test_baseline_is_not_expanded_after_worker_execution(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
    make_script,
):
    """
    ER-04: the baseline is taken once, at process start.

    Contamination is created in the first cycle and the second cycle
    is the one that completes. A baseline re-read per cycle would
    have absorbed the artifact by then and reported a clean tree —
    which is the failure mode this whole check exists to prevent.
    """

    start_work_task(
        autopilot_repo
    )

    seed_stale_results(
        autopilot_repo
    )

    advance_to(
        autopilot_repo,
        "CORRECT_EXTERNAL_FINDINGS",
    )

    result = run_to_completion(
        run_autopilot,
        make_worker,
        make_script,
        worker_body=WORKER_CONTAMINATES,
        env=contaminate(
            (
                "file",
                "early.txt",
            ),
        ),
        max_cycles="2",
    )

    payload = assert_completion_blocked(
        autopilot_repo,
        result,
    )

    assert payload["contamination"] == [
        "early.txt",
    ]

    assert (
        read_status(
            autopilot_repo
        )["phase"]
        == "FINAL_VERIFY"
    )


def test_material_entry_classification(
    autopilot_repo: Path,
    load_autopilot_module,
):
    """
    ER-04: what counts as a real artifact, entry by entry.

    Symlinks are resolved rather than skipped, so a mask is exempt
    while a link to real content is not. An entry that cannot be
    inspected at all counts as material: git listed it, so something
    is there, and an unclassifiable entry must not be assumed
    harmless.
    """

    module = load_autopilot_module()

    regular = (
        autopilot_repo
        / "regular.txt"
    )

    regular.write_text(
        ""
    )

    directory = (
        autopilot_repo
        / "directory"
    )

    directory.mkdir()

    mask = (
        autopilot_repo
        / "mask"
    )

    mask.symlink_to(
        "/dev/null"
    )

    link = (
        autopilot_repo
        / "link"
    )

    link.symlink_to(
        regular
    )

    broken = (
        autopilot_repo
        / "broken"
    )

    broken.symlink_to(
        autopilot_repo
        / "nowhere"
    )

    for path, material in (
        (
            regular,
            True,
        ),
        (
            directory,
            True,
        ),
        (
            link,
            True,
        ),
        (
            broken,
            True,
        ),
        (
            mask,
            False,
        ),
        (
            Path(
                "/dev/null"
            ),
            False,
        ),
    ):
        assert (
            module.is_material_entry(
                path
            )
            is material
        ), path


# ================================================================
# INTEGRATION SCOPE — AUTOPILOT STOPS AT AN UNAUTHORIZED ACTION
# ================================================================


def test_integration_sequence_includes_the_integrate_phase(
    load_autopilot_module,
):
    """
    The declared git actions are gated and performed in INTEGRATE, so
    a commit is not recorded as having happened during ANALYZE.
    """

    module = load_autopilot_module()

    assert module.INTEGRATION_SEQUENCE == [
        "ANALYZE",
        "INTEGRATE",
        "FINAL_VERIFY",
    ]

    # A Work Task never integrates, so the phase is not part of its
    # lifecycle.
    assert (
        "INTEGRATE"
        not in module.WORK_SEQUENCE
    )


def test_autopilot_stops_on_an_unresolved_integration_action(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
):
    """
    An Integration Task carrying an unauthorized push must stop at
    the human decision, not complete around it.

    Autopilot opens gates and never resolves them, so the run ends as
    WAITING_FOR_HUMAN with the outstanding action named.
    """

    start_work_task(
        autopilot_repo
    )

    seed_stale_results(
        autopilot_repo
    )

    advance_to(
        autopilot_repo,
        "FINAL_VERIFY",
    )

    run_statusctl(
        autopilot_repo,
        "ready",
    )

    run_statusctl(
        autopilot_repo,
        "start-integration",
        "autopilot-integration-002",
        "Integrate the reviewed work",
        "--action",
        "push",
    )

    for kind in (
        "targeted",
        "full",
    ):
        run_statusctl(
            autopilot_repo,
            "test-not-required",
            kind,
            "Integration Task never re-tests product code.",
        )

    run_statusctl(
        autopilot_repo,
        "phase",
        "FINAL_VERIFY",
    )

    result = run_autopilot(
        "run",
        "--max-cycles",
        "1",
        worker=make_worker(
            WORKER_PASS
        ),
    )

    assert (
        result.returncode
        == EXIT_WAITING_FOR_HUMAN
    ), (
        result.stdout
        or result.stderr
    )

    payload = result_payload(
        result
    )

    assert payload["unresolved_integration_actions"] == [
        "push",
    ]

    status = read_status(
        autopilot_repo
    )

    assert status["state"] == "RUNNING"

    assert status["completed_at"] is None


# ================================================================
# AR-01 — AUTOPILOT OPENS ACTION-BOUND INTEGRATION GATES
# ================================================================
#
# A GIT_INTEGRATION gate must name exactly one declared action. The
# worker cannot open gates — statusctl is read-only inside its
# session — so it asks, and autopilot opens. If autopilot cannot pass
# the action through, the orchestrated integration path stops at the
# one boundary INTEGRATE exists to reach.
#
# Opening is autonomous. Resolving never is: these pin that autopilot
# stops at the gate rather than deciding it.


def integration_task_at_integrate(
    autopilot_repo: Path,
    *actions: str,
) -> None:
    """
    An Integration Task standing at its integration boundary.
    """

    start_work_task(
        autopilot_repo
    )

    seed_stale_results(
        autopilot_repo
    )

    advance_to(
        autopilot_repo,
        "FINAL_VERIFY",
    )

    run_statusctl(
        autopilot_repo,
        "ready",
    )

    args = [
        "start-integration",
        "autopilot-integration-003",
        "Integrate the reviewed work",
    ]

    for action in actions or (
        "none",
    ):
        args.extend(
            [
                "--action",
                action,
            ]
        )

    run_statusctl(
        autopilot_repo,
        *args,
    )

    for kind in (
        "targeted",
        "full",
    ):
        run_statusctl(
            autopilot_repo,
            "test-not-required",
            kind,
            "Integration Task never re-tests product code.",
        )

    run_statusctl(
        autopilot_repo,
        "phase",
        "INTEGRATE",
    )


def gate_worker(
    body: str,
) -> str:
    return (
        "print(json.dumps({'outcome': 'HUMAN_GATE_REQUIRED', "
        "'summary': 'the action needs authorization', "
        + body
        + "'question': 'Approve the action?', "
        "'recommendation': 'Approve it.'}))\n"
    )


WORKER_GATE_INTEGRATION = gate_worker(
    "'gate_type': 'GIT_INTEGRATION', 'action': 'commit', "
)

WORKER_GATE_INTEGRATION_NO_ACTION = gate_worker(
    "'gate_type': 'GIT_INTEGRATION', "
)

WORKER_GATE_INTEGRATION_UNKNOWN_ACTION = gate_worker(
    "'gate_type': 'GIT_INTEGRATION', 'action': 'rebase', "
)

WORKER_GATE_ACTION_ON_OTHER_TYPE = gate_worker(
    "'gate_type': 'SEMANTIC_DECISION', 'action': 'push', "
)


def assert_gate_was_not_opened(
    autopilot_repo: Path,
    result: subprocess.CompletedProcess[str],
) -> None:
    assert (
        result.returncode
        == EXIT_BLOCKED_TECHNICAL
    ), (
        result.stdout
        or result.stderr
    )

    status = read_status(
        autopilot_repo
    )

    assert status["state"] == "RUNNING"

    assert status["human_gate"] is None

    assert status["phase"] == "INTEGRATE"

    assert not any(
        event["event"] == "HUMAN_GATE_OPENED"
        for event in read_audit(
            autopilot_repo
        )
    )


def test_autopilot_opens_an_action_bound_integration_gate(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
):
    """
    AR-01: the worker's action reaches the gate.
    """

    integration_task_at_integrate(
        autopilot_repo,
        "commit",
    )

    result = run_autopilot(
        "run",
        "--max-cycles",
        "1",
        worker=make_worker(
            WORKER_GATE_INTEGRATION
        ),
    )

    assert (
        result.returncode
        == EXIT_WAITING_FOR_HUMAN
    ), (
        result.stdout
        or result.stderr
    )

    status = read_status(
        autopilot_repo
    )

    assert (
        status["state"]
        == "BLOCKED_HUMAN_DECISION"
    )

    gate = status["human_gate"]

    assert gate["type"] == "GIT_INTEGRATION"

    assert gate["action"] == "commit"

    payload = result_payload(
        result
    )

    assert (
        payload["human_gate"]["gate_id"]
        == gate["gate_id"]
    )

    # Opening is autonomous; deciding is not. The action must still be
    # awaiting a human.
    assert (
        status["integration_actions"][0]["status"]
        == "DECLARED"
    )

    events = {
        event["event"]
        for event in read_audit(
            autopilot_repo
        )
    }

    assert "HUMAN_GATE_OPENED" in events

    assert not (
        events
        & {
            "HUMAN_GATE_APPROVED",
            "HUMAN_GATE_ALTERNATIVE_CHOSEN",
            "HUMAN_GATE_REJECTED",
        }
    )


def test_autopilot_refuses_an_integration_gate_with_no_action(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
):
    """
    AR-01: an unnamed integration gate fails closed inside autopilot.

    The controller would refuse it anyway. Refusing here too means the
    orchestrator reports which requirement the worker missed, instead
    of surfacing a controller error for a request it should never have
    sent.
    """

    integration_task_at_integrate(
        autopilot_repo,
        "commit",
    )

    result = run_autopilot(
        "run",
        "--max-cycles",
        "1",
        worker=make_worker(
            WORKER_GATE_INTEGRATION_NO_ACTION
        ),
    )

    assert_gate_was_not_opened(
        autopilot_repo,
        result,
    )

    assert "action" in json.dumps(
        result_payload(
            result
        )
    ).lower()


def test_autopilot_refuses_an_unknown_integration_action(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
):
    """
    AR-01: the worker cannot invent an action.
    """

    integration_task_at_integrate(
        autopilot_repo,
        "commit",
    )

    result = run_autopilot(
        "run",
        "--max-cycles",
        "1",
        worker=make_worker(
            WORKER_GATE_INTEGRATION_UNKNOWN_ACTION
        ),
    )

    assert_gate_was_not_opened(
        autopilot_repo,
        result,
    )


def test_autopilot_refuses_an_action_on_a_non_integration_gate(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
):
    """
    AR-01: an action binding belongs to an integration decision only.

    Silently dropping it would discard what the worker asked for; the
    mismatch is reported instead.
    """

    integration_task_at_integrate(
        autopilot_repo,
        "commit",
    )

    result = run_autopilot(
        "run",
        "--max-cycles",
        "1",
        worker=make_worker(
            WORKER_GATE_ACTION_ON_OTHER_TYPE
        ),
    )

    assert_gate_was_not_opened(
        autopilot_repo,
        result,
    )


def test_worker_brief_states_the_action_contract(
    load_autopilot_module,
):
    """
    AR-01: a worker can only supply what it is told to supply.
    """

    module = load_autopilot_module()

    brief = json.loads(
        module.build_brief(
            {
                "task_id": "int-001",
                "task_title": "Integrate",
                "task_kind": "INTEGRATION",
                "parent_task_id": "work-001",
                "phase": "INTEGRATE",
            }
        )
    )

    schema = brief["response_schema"]

    assert "action" in schema

    assert (
        "GIT_INTEGRATION"
        in schema["action"]
    )


# ================================================================
# ER-08 — THE RUN STOPS WHERE THE DECISION CAN STILL BE REQUESTED
# ================================================================
#
# Stopping fail-closed is not enough on its own. A GIT_INTEGRATION
# gate may only be opened in INTEGRATE, so an orchestrator that
# advances past that phase while an action is still unresolved leaves
# the human holding a decision they cannot action: the run's own
# message names a gate that the state it left behind refuses to open.
#
# So the scope is checked before the transition out of INTEGRATE, not
# only at completion.


def test_autopilot_stops_in_integrate_while_an_action_is_unresolved(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
):
    """
    ER-08: the phase must not move past the unresolved decision.
    """

    integration_task_at_integrate(
        autopilot_repo,
        "commit",
    )

    result = run_autopilot(
        "run",
        "--max-cycles",
        "3",
        worker=make_worker(
            WORKER_PASS
        ),
        timeout=120.0,
    )

    assert (
        result.returncode
        == EXIT_WAITING_FOR_HUMAN
    ), (
        result.stdout
        or result.stderr
    )

    payload = result_payload(
        result
    )

    assert payload["unresolved_integration_actions"] == [
        "commit",
    ]

    status = read_status(
        autopilot_repo
    )

    assert status["state"] == "RUNNING"

    # The whole point: still standing where the gate is legal.
    assert status["phase"] == "INTEGRATE"

    assert status["completed_at"] is None

    assert (
        status["integration_actions"][0]["status"]
        == "DECLARED"
    )


def test_the_awaited_gate_can_be_opened_where_autopilot_stopped(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
):
    """
    ER-08: the state autopilot leaves must accept the decision it asks
    for. This is the assertion the previous behaviour failed.
    """

    integration_task_at_integrate(
        autopilot_repo,
        "commit",
    )

    assert run_autopilot(
        "run",
        "--max-cycles",
        "3",
        worker=make_worker(
            WORKER_PASS
        ),
        timeout=120.0,
    ).returncode == EXIT_WAITING_FOR_HUMAN

    # run_statusctl asserts a zero exit, so this call failing is the
    # failure being guarded against.
    run_statusctl(
        autopilot_repo,
        "gate",
        "open",
        "GIT_INTEGRATION",
        "--action",
        "commit",
        "Approve the commit?",
        "Approve it.",
    )

    gate = read_status(
        autopilot_repo
    )["human_gate"]

    assert gate["type"] == "GIT_INTEGRATION"

    assert gate["action"] == "commit"


def test_autopilot_advances_out_of_integrate_once_scope_is_resolved(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
):
    """
    ER-08 must gate the transition, not forbid it.

    A declined action is resolved, so the task proceeds to completion
    exactly as before.
    """

    integration_task_at_integrate(
        autopilot_repo,
        "commit",
    )

    run_statusctl(
        autopilot_repo,
        "gate",
        "open",
        "GIT_INTEGRATION",
        "--action",
        "commit",
        "Approve the commit?",
        "Approve it.",
    )

    gate_id = read_status(
        autopilot_repo
    )["human_gate"]["gate_id"]

    run_statusctl(
        autopilot_repo,
        "gate",
        "reject",
        gate_id,
        "The commit belongs to a later integration.",
    )

    result = run_autopilot(
        "run",
        "--max-cycles",
        "3",
        worker=make_worker(
            WORKER_PASS
        ),
        timeout=120.0,
    )

    assert result.returncode == EXIT_OK, (
        result.stdout
        or result.stderr
    )

    status = read_status(
        autopilot_repo
    )

    assert (
        status["state"]
        == "READY_FOR_HUMAN_REVIEW"
    )

    assert (
        status["integration_actions"][0]["status"]
        == "DECLINED"
    )


# ================================================================
# AR-02 — THE REDUNDANT DENYLIST COVERS SCHEMA 3.1
# ================================================================


REQUIRED_WORKER_DENIES = {
    "Bash(./.claude/bin/statusctl start *)",
    "Bash(./.claude/bin/statusctl start-integration *)",
    "Bash(./.claude/bin/statusctl phase *)",
    "Bash(./.claude/bin/statusctl test *)",
    "Bash(./.claude/bin/statusctl test-not-required *)",
    "Bash(./.claude/bin/statusctl gate *)",
    "Bash(./.claude/bin/statusctl ready)",
    "Bash(./.claude/bin/statusctl fail *)",
    # Deliberately the broad form. A worker session has no business
    # with any integration subcommand, so the rule is not narrowed to
    # the one that exists today.
    "Bash(./.claude/bin/statusctl integration *)",
    "Bash(./.claude/bin/statusctl migrate-v3 *)",
    # Both spellings: the pattern above requires a following space, so
    # it never matches migrate-v31, and the bare form carries no
    # argument to match.
    "Bash(./.claude/bin/statusctl migrate-v31)",
    "Bash(./.claude/bin/statusctl migrate-v31 *)",
    "Bash(statusctl *)",
}


def test_worker_denylist_covers_schema_31_state_changes(
    load_autopilot_module,
):
    """
    AR-02: every state-changing subcommand is refused by every layer.

    The read-only shim is the control that actually holds; this list
    is the redundant one the design deliberately keeps. A command that
    can write evidence or migrate the runtime must appear in both.

    Membership is asserted exactly, rather than by searching a joined
    string. A substring test passes on a rule that merely contains the
    text — a malformed pattern, a wrong path prefix, or a rule that
    happens to mention the command in passing would all satisfy it,
    which is precisely the kind of near-miss a permission list must
    not be checked with.
    """

    module = load_autopilot_module()

    denied = module.WORKER_DISALLOWED_TOOLS

    # A set comparison would silently absorb a duplicated entry, so
    # the tuple is checked for duplicates before it becomes one.
    assert len(set(denied)) == len(denied), denied

    missing = sorted(
        REQUIRED_WORKER_DENIES
        - set(denied)
    )

    assert not missing, missing


# ================================================================
# SHOULD_FIX-A — SHELL WRAPPER ESCAPES FAIL CLOSED
# ================================================================


SHELL_ESCAPES = [
    [
        "bash",
        "-c",
        "echo hi",
    ],
    [
        "/bin/sh",
        "-c",
        "echo hi",
    ],
    [
        "zsh",
        "-c",
        "echo hi",
    ],
    [
        "env",
        "PYTHONPATH=src",
        "python3",
        "-m",
        "pytest",
    ],
    [
        "xargs",
        "-I",
        "{}",
        "sh",
    ],
    [
        "timeout",
        "5",
        "bash",
    ],
    [
        "nohup",
        "bash",
    ],
    [
        "python3",
        "-c",
        "import os; os.system('sh')",
    ],
    [
        "node",
        "-e",
        "require('child_process')",
    ],
    [
        "perl",
        "-e",
        "system('sh')",
    ],
]


def test_shell_wrapper_commands_are_refused_by_the_guard(
    load_autopilot_module,
):
    """
    SHOULD_FIX-A: argv-only execution is worth nothing if the first
    argv element is a shell.
    """

    module = load_autopilot_module()

    for argv in SHELL_ESCAPES:
        with pytest.raises(
            module.ForbiddenCommand
        ):
            module.assert_command_allowed(
                argv
            )


def test_ordinary_interpreter_invocations_still_pass(
    load_autopilot_module,
):
    """
    The closure must not swallow legitimate argv-only commands.
    """

    module = load_autopilot_module()

    for argv in (
        [
            "python3",
            "/tmp/stub_worker.py",
        ],
        [
            ".venv/bin/python",
            "-m",
            "pytest",
            "-q",
        ],
        [
            "claude",
            "--permission-mode",
            "dontAsk",
            "--allowedTools",
            "Read",
            "-p",
        ],
    ):
        module.assert_command_allowed(
            argv
        )


def test_shell_wrapper_worker_fails_closed(
    autopilot_repo: Path,
    run_autopilot,
):
    """
    SHOULD_FIX-A: a wrapped worker is refused, not executed.
    """

    start_work_task(
        autopilot_repo
    )

    result = run_autopilot(
        "run",
        "--max-cycles",
        "1",
        worker=[
            "bash",
            "-c",
            "echo '{\"outcome\": \"PASS\", \"summary\": \"x\"}'",
        ],
    )

    assert (
        result.returncode
        == EXIT_BLOCKED_PERMISSION
    )

    assert (
        result_payload(
            result
        )["result"]
        == "REFUSED"
    )

    assert (
        read_status(
            autopilot_repo
        )["phase"]
        == "ANALYZE"
    )


def test_shell_wrapper_verification_fails_closed(
    autopilot_repo: Path,
    run_autopilot,
    make_worker,
    make_script,
):
    """
    SHOULD_FIX-A: the same closure applies to verification commands,
    which are the ones that produce recorded evidence.
    """

    start_work_task(
        autopilot_repo
    )

    seed_stale_results(
        autopilot_repo
    )

    advance_to(
        autopilot_repo,
        "FINAL_VERIFY",
    )

    result = run_autopilot(
        "run",
        "--max-cycles",
        "1",
        worker=make_worker(
            WORKER_PASS
        ),
        env=verify_env(
            targeted=[
                "bash",
                "-c",
                "exit 0",
            ],
            full=make_script(
                VERIFY_PASS
            ),
        ),
    )

    assert (
        result.returncode
        == EXIT_BLOCKED_PERMISSION
    )

    status = read_status(
        autopilot_repo
    )

    assert status["state"] == "RUNNING"

    assert status["completed_at"] is None
