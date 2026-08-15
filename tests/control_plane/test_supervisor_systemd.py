"""
Hardening 4.4 — persistent supervisor contract.

The supervisor is bounded and resumable by design, so making it
persistent is a scheduling problem, not a supervision problem. A timer
fires it; a guard decides whether firing it is useful. Neither is
allowed to become a new way to reach the orchestrator, and neither is
allowed to turn a human decision boundary into a loop.

These tests pin what the persistence layer is permitted to be:

    G-01  the guard proceeds only for a RUNNING task on the schema it
          understands, and skips everything else, including every
          state that means a human is owed a decision;
    G-02  it fails closed — absent, unreadable, malformed, superseded
          or FAIL-bearing runtime state all skip;
    G-03  its exit codes stay inside systemd's ExecCondition skip
          band, so a refusal is never reported as a unit failure;
    G-04  it writes nothing and starts no subprocess, which is what
          keeps it from being a second path to the supervisor;
    S-01  the service cannot run the supervisor without the guard,
          cannot restart into an open gate, and cannot be enabled in
          its own right;
    S-02  the credential is referenced from outside the repository and
          never embedded;
    T-01  the timer is the installable unit and leaves a real gap
          between batches.

The units and the guard are read as delivered. Nothing here installs a
unit, runs systemctl, or invokes the supervisor.
"""

from __future__ import annotations

import ast
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

SYSTEMD_DIR = PROJECT_ROOT / "deploy" / "systemd"

GUARD = SYSTEMD_DIR / "cryptolab-supervisor-guard"

SERVICE = SYSTEMD_DIR / "cryptolab-supervisor.service"

TIMER = SYSTEMD_DIR / "cryptolab-supervisor.timer"

README = SYSTEMD_DIR / "cryptolab-supervisor.README.md"


EXIT_PROCEED = 0

EXIT_SKIP = 1

# systemd reads an ExecCondition exit of 1..254 as "skip this unit,
# and it is not a failure". 255 or a signal is a genuine failure. The
# guard must never leave that band, or a blocked task becomes a failed
# unit on every tick.
SKIP_BAND = range(1, 255)


# ================================================================
# RUNTIME STATE FIXTURES
# ================================================================


def runtime_status(**overrides) -> dict:
    """
    A schema-3.1 status document, valid unless a test breaks it.

    Only the fields the guard actually reads are meaningful here; the
    rest are present so the fixture stays recognisable as the real
    thing.
    """

    status = {
        "schema_version": "3.1",
        "project": "CryptoLab-DataHub",
        "state": "RUNNING",
        "task_id": "some-task-001",
        "task_title": "Some task",
        "task_kind": "WORK",
        "parent_task_id": None,
        "phase": "IMPLEMENT",
        "branch": "claude/system-autonomy-setup",
        "started_at": "2026-08-15T01:30:33.439294Z",
        "updated_at": "2026-08-15T01:34:22.779533Z",
        "completed_at": None,
        "tests": {
            "targeted": None,
            "full": None,
        },
        "human_gate": None,
        "last_gate_decision": None,
        "integration_actions": None,
        "report": ".claude/runtime/latest_report.md",
    }

    status.update(overrides)

    return status


@pytest.fixture
def worktree(tmp_path: Path) -> Path:
    """
    A synthetic worktree carrying only a runtime directory.

    The guard is pointed at this rather than at the real project, so
    no test result depends on what the real task happens to be doing
    while the suite runs.
    """

    runtime = tmp_path / ".claude" / "runtime"

    runtime.mkdir(parents=True)

    return tmp_path


@pytest.fixture
def write_status(worktree: Path):
    def write(status) -> Path:
        path = (
            worktree
            / ".claude"
            / "runtime"
            / "status.json"
        )

        if isinstance(status, str):
            path.write_text(status, encoding="utf-8")

        else:
            path.write_text(
                json.dumps(status, indent=2),
                encoding="utf-8",
            )

        return path

    return write


@pytest.fixture
def run_guard(worktree: Path):
    """
    Invoke the delivered guard against the synthetic worktree.

    Run out of process, because the exit code is the entire interface
    systemd consumes and an in-process call would not exercise it.
    """

    def run(root: Path | None = None, extra=()):
        return subprocess.run(
            [
                sys.executable,
                str(GUARD),
                str(worktree if root is None else root),
                *extra,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

    return run


# ================================================================
# UNIT FILE PARSING
# ================================================================
#
# configparser is not usable here: systemd keys are case-sensitive,
# may repeat within a section, and `%h` would be read as broken
# interpolation. So the format is parsed directly, which is a dozen
# lines and removes every one of those disagreements.


def parse_unit(path: Path) -> dict[str, list[tuple[str, str]]]:
    sections: dict[str, list[tuple[str, str]]] = {}

    current: list[tuple[str, str]] | None = None

    pending: tuple[str, str] | None = None

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()

        if pending is not None:
            # Continuation of the previous directive.
            key, value = pending

            if line.endswith("\\"):
                pending = (key, value + " " + line[:-1].strip())

            else:
                assert current is not None

                current.append((key, (value + " " + line).strip()))

                pending = None

            continue

        if not line or line.startswith(("#", ";")):
            continue

        if line.startswith("[") and line.endswith("]"):
            current = sections.setdefault(line[1:-1], [])

            continue

        if "=" not in line:
            raise AssertionError(
                f"{path.name}: unparseable line {raw!r}"
            )

        key, _, value = line.partition("=")

        key = key.strip()

        value = value.strip()

        if value.endswith("\\"):
            pending = (key, value[:-1].strip())

            continue

        assert current is not None, (
            f"{path.name}: directive {key!r} outside any section"
        )

        current.append((key, value))

    assert pending is None, (
        f"{path.name}: file ends inside a line continuation"
    )

    return sections


def directives(
    sections: dict[str, list[tuple[str, str]]],
    section: str,
    key: str,
) -> list[str]:
    return [
        value
        for name, value in sections.get(section, [])
        if name == key
    ]


def only(values: list[str]) -> str:
    assert len(values) == 1, (
        f"expected exactly one value, got {values!r}"
    )

    return values[0]


def expand(value: str, home: str = "/HOME") -> str:
    """
    Resolve the specifiers these units actually use.

    `%h` is the only one, and expanding it against a synthetic home
    keeps path comparisons independent of whoever runs the suite.
    """

    return value.replace("%%", "\x00").replace(
        "%h", home
    ).replace("\x00", "%")


@pytest.fixture(scope="module")
def service() -> dict[str, list[tuple[str, str]]]:
    return parse_unit(SERVICE)


@pytest.fixture(scope="module")
def timer() -> dict[str, list[tuple[str, str]]]:
    return parse_unit(TIMER)


# ================================================================
# DELIVERY
# ================================================================


def test_the_persistence_layer_is_delivered_whole():
    """
    A unit without its guard, or a guard without its unit, is a
    half-installed control surface. They ship together or not at all.
    """

    for path in (GUARD, SERVICE, TIMER, README):
        assert path.is_file(), f"missing {path}"


def test_guard_is_executable_with_a_python_shebang():
    """
    systemd's ExecCondition runs the file itself, not an interpreter
    named beside it. Losing the executable bit turns a skip into a
    failed unit on every tick.
    """

    mode = GUARD.stat().st_mode

    assert mode & stat.S_IXUSR, "guard is not executable by its owner"

    assert mode & stat.S_IXGRP

    assert mode & stat.S_IXOTH

    first = GUARD.read_text(encoding="utf-8").splitlines()[0]

    assert first == "#!/usr/bin/env python3", first


# ================================================================
# G-01  THE GUARD PROCEEDS ONLY WHERE WORK CAN CONTINUE
# ================================================================


def test_guard_proceeds_for_a_running_task(run_guard, write_status):
    write_status(runtime_status())

    result = run_guard()

    assert result.returncode == EXIT_PROCEED, result.stdout

    assert "proceed" in result.stdout

    assert "RUNNING" in result.stdout


@pytest.mark.parametrize(
    "state",
    [
        "IDLE",
        "BLOCKED_HUMAN_DECISION",
        "READY_FOR_HUMAN_REVIEW",
        "FAILED",
        # Not a state the lifecycle defines. A state added later must
        # start out refused; that is the safe direction for a guard
        # that is read by a scheduler and not by a person.
        "SOMETHING_NEW",
    ],
)
def test_guard_skips_every_state_but_running(
    run_guard,
    write_status,
    state,
):
    write_status(runtime_status(state=state))

    result = run_guard()

    assert result.returncode == EXIT_SKIP, result.stdout

    assert "skip" in result.stdout

    assert state in result.stdout


def test_guard_names_the_open_gate_it_is_waiting_on(
    run_guard,
    write_status,
):
    """
    The gate ID is what a human needs in order to resolve it, so the
    journal line carries it rather than making them go and look.
    """

    write_status(
        runtime_status(
            state="BLOCKED_HUMAN_DECISION",
            human_gate={
                "gate_id": "HG-20260812-009",
                "gate_type": "OS_CHANGE",
            },
        )
    )

    result = run_guard()

    assert result.returncode == EXIT_SKIP

    assert "HG-20260812-009" in result.stdout


# ================================================================
# G-02  EVERY UNCERTAINTY IS A SKIP
# ================================================================


def test_guard_skips_when_there_is_no_runtime_state(run_guard):
    result = run_guard()

    assert result.returncode == EXIT_SKIP, result.stdout

    assert "no runtime state" in result.stdout


def test_guard_skips_when_the_worktree_does_not_exist(
    run_guard,
    tmp_path,
):
    result = run_guard(root=tmp_path / "nowhere")

    assert result.returncode == EXIT_SKIP, result.stdout


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "{",
        "not json at all",
        '["a", "list"]',
        '"a string"',
        "null",
    ],
)
def test_guard_skips_on_malformed_runtime_state(
    run_guard,
    write_status,
    payload,
):
    write_status(payload)

    result = run_guard()

    assert result.returncode == EXIT_SKIP, result.stdout


@pytest.mark.parametrize(
    "version",
    ["2.0", "3.0", "3.2", None, 3.1],
)
def test_guard_skips_a_schema_it_does_not_read(
    run_guard,
    write_status,
    version,
):
    """
    A superseded runtime means a migration is owed, and migration is
    an operator step. The guard declines rather than reading fields
    whose meaning it cannot vouch for.
    """

    write_status(runtime_status(schema_version=version))

    result = run_guard()

    assert result.returncode == EXIT_SKIP, result.stdout

    assert "schema" in result.stdout


@pytest.mark.parametrize("kind", ["targeted", "full"])
def test_guard_skips_a_recorded_test_failure(
    run_guard,
    write_status,
    kind,
):
    """
    The supervisor refuses to advance past a recorded FAIL, so every
    activation from here would stop on the same evidence. Skipping
    keeps the timer quiet until a person looks.
    """

    tests = {"targeted": None, "full": None}

    tests[kind] = {
        "command": "PYTHONPATH=src .venv/bin/python -m pytest -q",
        "result": "FAIL",
        "summary": "two failures",
    }

    write_status(runtime_status(tests=tests))

    result = run_guard()

    assert result.returncode == EXIT_SKIP, result.stdout

    assert kind in result.stdout


@pytest.mark.parametrize("result_value", ["PASS", "NOT_REQUIRED"])
def test_guard_proceeds_past_a_satisfied_test_requirement(
    run_guard,
    write_status,
    result_value,
):
    """
    Only FAIL blocks. A satisfied requirement — passed or justifiably
    exempt — is a task that may continue.
    """

    write_status(
        runtime_status(
            tests={
                "targeted": {
                    "command": None
                    if result_value == "NOT_REQUIRED"
                    else "pytest -q",
                    "result": result_value,
                    "summary": "a reason",
                },
                "full": None,
            }
        )
    )

    result = run_guard()

    assert result.returncode == EXIT_PROCEED, result.stdout


def test_guard_skips_when_given_more_arguments_than_it_takes(
    run_guard,
    write_status,
):
    """
    A misconfigured unit must not fail; it must decline. The guard's
    whole vocabulary is proceed or skip.
    """

    write_status(runtime_status())

    result = run_guard(extra=("unexpected",))

    assert result.returncode == EXIT_SKIP, result.stdout


# ================================================================
# G-03  REFUSALS STAY INSIDE THE SKIP BAND
# ================================================================


@pytest.mark.parametrize(
    "payload",
    [
        None,
        "{",
        '["a", "list"]',
    ],
)
def test_guard_refusals_never_leave_the_systemd_skip_band(
    run_guard,
    write_status,
    payload,
):
    """
    255 or a signal would make systemd fail the unit. Every refusal,
    including the ones that come from a broken control plane, must
    land in 1..254 instead.
    """

    if payload is not None:
        write_status(payload)

    result = run_guard()

    assert result.returncode in SKIP_BAND, result.returncode


# ================================================================
# G-04  THE GUARD IS NOT A SECOND PATH TO THE SUPERVISOR
# ================================================================


def test_guard_starts_no_subprocess():
    """
    The guard adds no new way to reach `.claude/bin/supervisor`.

    That is the property that lets it live outside the control plane's
    permission boundary: systemd composes ExecCondition and ExecStart,
    and systemd is configured by an operator. A guard that invoked the
    orchestrator itself would hand back exactly the authority the deny
    rules on `.claude/bin/supervisor` withhold.

    Proved from the imports rather than by scanning for suspicious
    words, because the import list is the capability list: without
    subprocess, os, socket or a dynamic import there is nothing to
    start a process with, and no blocklist has to be kept complete.
    """

    tree = ast.parse(
        GUARD.read_text(encoding="utf-8"),
        filename=str(GUARD),
    )

    imported = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(
                alias.name.split(".")[0]
                for alias in node.names
            )

        elif isinstance(node, ast.ImportFrom):
            # A relative import has no module name to record; the
            # guard has no package to reach into either way.
            imported.add((node.module or "").split(".")[0])

    assert imported <= {
        "__future__",
        "json",
        "sys",
        "pathlib",
    }, imported

    # The other way to acquire a capability is to build it at runtime.
    dynamic = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id
        in {
            "eval",
            "exec",
            "compile",
            "__import__",
            "getattr",
        }
    }

    assert not dynamic, dynamic


def test_guard_writes_nothing(run_guard, write_status, worktree):
    """
    Read-only in the literal sense: not one byte of the worktree
    differs afterwards, and no file appears that was not there before.
    """

    write_status(runtime_status())

    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in worktree.rglob("*")
        if path.is_file()
    }

    assert before, "fixture wrote nothing to compare against"

    result = run_guard()

    assert result.returncode == EXIT_PROCEED, result.stdout

    after = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in worktree.rglob("*")
        if path.is_file()
    }

    assert after == before


def test_guard_defaults_to_the_worktree_it_ships_in():
    """
    With no argument the guard resolves the repository above it, so a
    unit that names only the executable still reads the right runtime
    state.
    """

    result = subprocess.run(
        [sys.executable, str(GUARD)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=os.sep,
    )

    # It reports on the real project, whose state the suite must not
    # depend on. What is pinned is that it resolved *a* decision from
    # the delivered location rather than crashing.
    assert result.returncode in (EXIT_PROCEED, EXIT_SKIP)

    assert result.stdout.startswith("SUPERVISOR_GUARD ")


# ================================================================
# S-01  THE SERVICE
# ================================================================


def test_service_gates_activation_on_the_guard(service):
    """
    Without ExecCondition the timer would re-activate the supervisor
    against an open Human Gate every interval.
    """

    condition = only(
        directives(service, "Service", "ExecCondition")
    )

    assert condition.endswith(GUARD.name), condition

    assert Path(expand(condition)).is_absolute(), condition


def test_service_runs_the_supervisor_bounded_and_nothing_else(
    service,
):
    start = only(directives(service, "Service", "ExecStart"))

    assert ".claude/bin/supervisor" in start, start

    assert " run" in start, start

    # The bound is the reason a batch ends and control returns to the
    # guard. An unbounded ExecStart would make the timer decorative.
    assert "--max-cycles" in start, start

    assert Path(expand(start).split()[0]).is_absolute(), start


def test_service_is_oneshot(service):
    assert only(directives(service, "Service", "Type")) == "oneshot"


def test_service_never_restarts(service):
    """
    Restart=always would re-enter immediately on exit 10 — pinning the
    orchestrator against the very boundary that exists to hold it. The
    timer, gated by the guard, is the only way a batch begins again.
    """

    assert only(directives(service, "Service", "Restart")) == "no"

    assert not directives(service, "Service", "RestartSec")


def test_service_goes_inactive_so_the_timer_can_re_arm(service):
    """
    OnUnitInactiveSec= is measured from when the service last stopped,
    so a service that never stops is a schedule that never fires
    again. RemainAfterExit=yes would do exactly that: one batch, then
    silence, with the timer still listed as armed and nothing
    obviously wrong.
    """

    assert not directives(
        service, "Service", "RemainAfterExit"
    ), service["Service"]


def test_service_declares_no_dependency_user_scope_cannot_load(
    service,
):
    """
    The system units beside this one order after
    network-online.target, and copying that line here is the obvious
    mistake to make. A --user unit cannot depend on a system unit and
    network-online.target exists in no user-scope search path, so the
    dependency would resolve to a unit that does not load: the
    `Wants=` job would fail on every activation and the `After=` would
    order against nothing. A warning per tick in place of a guarantee
    is worse than no line at all.
    """

    for key in (
        "After",
        "Before",
        "Wants",
        "Requires",
        "Requisite",
        "BindsTo",
        "PartOf",
    ):
        for value in directives(service, "Unit", key):
            assert "network-online.target" not in value, (key, value)


def test_service_cannot_be_enabled_on_its_own(service):
    """
    A timer-activated unit with an [Install] section invites
    `systemctl --user enable cryptolab-supervisor.service`, which
    would start a batch at login outside the schedule.
    """

    assert "Install" not in service, service.get("Install")


def test_service_records_decisions_as_success_and_faults_as_failure(
    service,
):
    """
    10 is WAITING_FOR_HUMAN and 13 is CONCURRENT: the run stopping
    correctly at a decision boundary, and a worktree already busy.
    Neither is a fault. 11, 12 and 14 are, and must stay visible as a
    failed unit.
    """

    declared = only(
        directives(service, "Service", "SuccessExitStatus")
    ).split()

    assert set(declared) == {"10", "13"}, declared


def test_service_bounds_a_wedged_run(service):
    timeout = only(
        directives(service, "Service", "TimeoutStartSec")
    )

    assert timeout not in ("0", "infinity"), timeout


def test_service_paths_all_name_one_worktree(service):
    """
    WorkingDirectory, ExecCondition and ExecStart must agree. A unit
    that guarded one worktree and drove another would be worse than no
    guard at all.
    """

    working = expand(
        only(directives(service, "Service", "WorkingDirectory"))
    )

    condition = expand(
        only(directives(service, "Service", "ExecCondition"))
    )

    start = expand(
        only(directives(service, "Service", "ExecStart"))
    ).split()[0]

    assert condition == f"{working}/deploy/systemd/{GUARD.name}"

    assert start == f"{working}/.claude/bin/supervisor"


def test_service_puts_the_claude_cli_on_path(service):
    """
    A --user unit inherits almost no PATH, and the supervisor invokes
    `claude` by name.
    """

    path = only(
        [
            value
            for value in directives(
                service, "Service", "Environment"
            )
            if value.startswith("PATH=")
        ]
    )

    assert "/.local/bin" in path, path


# ================================================================
# S-02  THE CREDENTIAL IS REFERENCED, NEVER HELD
# ================================================================


def test_credential_is_read_from_outside_the_worktree(service):
    """
    The reviewer's key must not be reachable from the repository, from
    a cycle artifact, or from runtime state. The unit names a file and
    that file lives elsewhere.
    """

    declared = only(
        directives(service, "Service", "EnvironmentFile")
    )

    # The leading `-` makes it optional: a missing credential must not
    # stop the unit from starting. It does not become an approval
    # either — the supervisor fails EXTERNAL_REVIEW closed.
    assert declared.startswith("-"), declared

    credential = Path(expand(declared[1:]))

    working = Path(
        expand(
            only(
                directives(service, "Service", "WorkingDirectory")
            )
        )
    )

    assert credential.is_absolute(), credential

    assert not credential.is_relative_to(working), credential


@pytest.mark.parametrize(
    "path",
    [SERVICE, TIMER, GUARD, README],
    ids=lambda path: path.name,
)
def test_no_delivered_file_embeds_a_credential(path):
    text = path.read_text(encoding="utf-8")

    # Naming the variable is unavoidable — the units and the operator
    # instructions both have to. Giving it a value is not: the only
    # right-hand sides permitted are nothing at all and the ellipsis
    # the documentation uses to stand in for the real thing.
    for match in re.finditer(
        r"OPENAI_API_KEY\s*=\s*(?P<value>\S*)",
        text,
    ):
        assert match.group("value") in ("", "..."), match.group(0)

    assert not re.search(r"\bsk-[A-Za-z0-9_-]{16,}", text)


# ================================================================
# T-01  THE TIMER
# ================================================================


def test_timer_triggers_the_service(timer):
    assert only(directives(timer, "Timer", "Unit")) == SERVICE.name


def test_timer_leaves_a_real_gap_between_batches(timer):
    """
    OnUnitInactiveSec is measured from when the service last STOPPED.
    A batch runs for minutes and its duration varies with the work, so
    only a gap measured after completion stays a gap. OnUnitActiveSec
    would fire against a still-running unit and, once it finally
    landed, would leave no interval at all.
    """

    assert directives(timer, "Timer", "OnUnitInactiveSec")

    assert not directives(timer, "Timer", "OnUnitActiveSec")


def test_timer_does_not_replay_missed_cycles(timer):
    """
    A missed cycle has no value to catch up: the next one reads the
    same durable state and does the same work. Persistent=true would
    only mean a burst of runs at login.
    """

    assert only(directives(timer, "Timer", "Persistent")) == "false"


def test_timer_is_the_unit_an_operator_enables(timer):
    assert (
        only(directives(timer, "Install", "WantedBy"))
        == "timers.target"
    )
