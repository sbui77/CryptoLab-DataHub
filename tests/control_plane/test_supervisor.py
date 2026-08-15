"""
Hardening 4.3 — autonomous supervisor contract.

The supervisor is autopilot plus the two things an unattended run still
had to stop for: an independent review it could not perform, and an
ordinary technical choice it had no way to make. Removing those stops
is only safe if nothing else moved, so these tests pin both halves.

What it may now do:

    S-01  perform EXTERNAL_REVIEW through the reviewer bridge, and
          carry the verdict into the phase that corrects it;
    S-02  settle a technical choice inside approved scope as an
          AUTO_DECISION instead of waiting for a human;
    S-03  record NOT_REQUIRED for an Integration Task's empty test
          slots, on proven parent evidence.

What it still may not do:

    S-04  produce WAITING_FOR_HUMAN from anything but a canonical
          Human Gate opened through statusctl;
    S-05  treat an unclear worker answer as success — one bounded
          re-ask, then fail closed;
    S-06  inherit a test exemption from anything less than a parent
          WORK task with targeted PASS and full PASS;
    S-07  write anything into `.claude/runtime`, which belongs to
          statusctl alone;
    S-08  approve an external review that did not happen, whether the
          reviewer is missing, failing, or answering with prose;
    S-09  resolve a gate, or run at all inside a worker session.

The worker and the reviewer are both injected, so no test needs
credentials, a network, or a real Claude session.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Callable

import pytest


EXIT_OK = 0
EXIT_WAITING_FOR_HUMAN = 10
EXIT_BLOCKED_PERMISSION = 11
EXIT_BLOCKED_TECHNICAL = 12
EXIT_WAITING_EXTERNAL = 14


# ================================================================
# HARNESS
# ================================================================


def hermetic_environment() -> dict[str, str]:
    """
    The ambient environment with every orchestrator variable removed.

    This suite is itself run inside an unattended session, whose
    environment carries the outer run's configuration. Inheriting it
    would let the ambient run configure the supervisor under test —
    AUTOPILOT_WORKER_SESSION most sharply, which makes every
    subprocess here refuse to start, and OPENAI_API_KEY most quietly,
    which would make "no reviewer is configured" untestable.
    """

    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(
            (
                "AUTOPILOT_",
                "SUPERVISOR_",
                "OPENAI_",
            )
        )
    }


def git(
    repo: Path,
    *args: str,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [
            "git",
            *args,
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, (
        result.stderr
        or result.stdout
    )

    return result


@pytest.fixture
def committed_repo(
    supervisor_repo: Path,
) -> Path:
    """
    The control-plane repository with one commit behind it.

    `git diff HEAD` is how the reviewer payload is assembled, and it
    has nothing to compare against in a repository with no commits.
    The commit is made before any task starts, so runtime state stays
    untracked and the diff stays about the work.
    """

    (supervisor_repo / "README.md").write_text(
        "control plane under test\n"
    )

    git(supervisor_repo, "add", "README.md")

    git(
        supervisor_repo,
        "-c",
        "user.email=tests@example.invalid",
        "-c",
        "user.name=Control Plane Tests",
        "commit",
        "-q",
        "-m",
        "baseline",
    )

    return supervisor_repo


def run_statusctl(
    repo: Path,
    *args: str,
) -> subprocess.CompletedProcess[str]:
    """
    Drive the isolated controller directly, as an operator would.
    """

    result = subprocess.run(
        [
            str(
                repo
                / ".claude"
                / "bin"
                / "statusctl"
            ),
            *args,
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, (
        result.stderr
        or result.stdout
    )

    return result


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


def start_work_task(
    repo: Path,
    task_id: str = "supervisor-work-001",
) -> None:
    run_statusctl(
        repo,
        "start",
        task_id,
        "Supervisor work task",
    )


def advance_to(
    repo: Path,
    phase: str,
) -> None:
    for step in WORK_SEQUENCE[
        1 : WORK_SEQUENCE.index(phase) + 1
    ]:
        run_statusctl(
            repo,
            "phase",
            step,
        )


def read_status(
    repo: Path,
) -> dict:
    return json.loads(
        (
            repo
            / ".claude"
            / "runtime"
            / "status.json"
        ).read_text()
    )


def supervisor_result(
    completed: subprocess.CompletedProcess[str],
) -> dict:
    for line in reversed(
        completed.stdout.splitlines()
    ):
        if line.startswith("SUPERVISOR_RESULT "):
            return json.loads(
                line.split(" ", 1)[1]
            )

    raise AssertionError(
        "no SUPERVISOR_RESULT line in stdout:\n"
        f"{completed.stdout}\n{completed.stderr}"
    )


# A stub worker driven by a script of replies, one per invocation.
#
# The last reply repeats, so a test that cares only about the first
# answer does not have to describe every later one. Each invocation
# appends the brief it was given, which is how the tests below observe
# what the supervisor told the worker.
WORKER_SCRIPTED = (
    "state = os.environ['SCRIPT_STATE']\n"
    "count = int(open(state).read()) "
    "if os.path.exists(state) else 0\n"
    "open(state, 'w').write(str(count + 1))\n"
    "open(os.environ['BRIEF_LOG'], 'a').write(\n"
    "    json.dumps(sys.argv[-1]) + '\\n')\n"
    "script = json.loads(os.environ['SCRIPT_REPLIES'])\n"
    "reply = script[min(count, len(script) - 1)]\n"
    "if reply == 'CRASH':\n"
    "    sys.stderr.write('worker exploded\\n')\n"
    "    sys.exit(3)\n"
    "if reply == 'PROSE':\n"
    "    print('I have some thoughts about this phase.')\n"
    "    sys.exit(0)\n"
    "print(json.dumps(reply))\n"
)


PASS_REPLY = {
    "outcome": "PASS",
    "summary": "stub worker completed the phase",
}

GATE_REPLY = {
    "outcome": "HUMAN_GATE_REQUIRED",
    "summary": "a decision is required",
    "gate_type": "ARCHITECTURE_DECISION",
    "question": "Approve the design?",
    "recommendation": "Approve it.",
}

# Not a canonical gate request: it names no gate type, so it is a
# worker that stopped rather than a decision needing authority.
CHOICE_REPLY = {
    "outcome": "BLOCKED",
    "summary": "two acceptable implementations exist",
    "question": "Which cache eviction policy?",
    "recommendation": "Use the LRU policy already used elsewhere.",
    "alternatives": ["Use a TTL policy."],
}

STOPPED_REPLY = {
    "outcome": "BLOCKED",
    "summary": "I am unsure how to proceed.",
}


@pytest.fixture
def scripted_worker(
    supervisor_repo: Path,
    tmp_path: Path,
) -> Callable[..., dict[str, str]]:
    """
    Write the scripted stub and return the environment that drives it.
    """

    path = supervisor_repo / "stub_worker.py"

    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys, os, time\n"
        + WORKER_SCRIPTED
    )

    path.chmod(
        path.stat().st_mode | stat.S_IEXEC
    )

    def configure(
        *replies,
    ) -> dict[str, str]:
        return {
            "AUTOPILOT_WORKER": json.dumps(
                [
                    "python3",
                    str(path),
                ]
            ),
            "SCRIPT_REPLIES": json.dumps(
                list(replies)
            ),
            "SCRIPT_STATE": str(
                tmp_path / "worker-state"
            ),
            "BRIEF_LOG": str(
                tmp_path / "brief-log"
            ),
        }

    return configure


@pytest.fixture
def briefs(
    tmp_path: Path,
) -> Callable[[], list[dict]]:
    def read() -> list[dict]:
        path = tmp_path / "brief-log"

        if not path.exists():
            return []

        return [
            json.loads(
                json.loads(line)
            )
            for line in path.read_text().splitlines()
            if line.strip()
        ]

    return read


@pytest.fixture
def run_supervisor(
    supervisor_repo: Path,
) -> Callable[..., subprocess.CompletedProcess[str]]:
    def run(
        *args: str,
        env: dict[str, str] | None = None,
        timeout: float = 120.0,
    ) -> subprocess.CompletedProcess[str]:
        environment = hermetic_environment()

        if env:
            environment.update(env)

        return subprocess.run(
            [
                str(
                    supervisor_repo
                    / ".claude"
                    / "bin"
                    / "supervisor"
                ),
                *args,
            ],
            cwd=supervisor_repo,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
            env=environment,
        )

    return run


def artifacts(
    repo: Path,
) -> list[dict]:
    """
    Every cycle artifact this run wrote, oldest first.
    """

    directory = (
        repo
        / ".claude"
        / "var"
        / "supervisor"
        / "cycles"
    )

    if not directory.is_dir():
        return []

    records = [
        json.loads(path.read_text())
        for path in directory.glob("*.json")
    ]

    records.sort(
        key=lambda record: record.get("started_at") or "",
    )

    return records


# ================================================================
# EXTERNAL STATES
# ================================================================


def test_status_reports_the_next_action_without_changing_state(
    supervisor_repo: Path,
    run_supervisor,
):
    """
    `status` is an inspection, so it must not be a transition.

    It is also deliberately lock-free, so an operator can look at a
    worktree a run is already driving.
    """

    start_work_task(supervisor_repo)

    before = read_status(supervisor_repo)

    completed = run_supervisor("status")

    assert completed.returncode == EXIT_OK

    payload = supervisor_result(completed)

    assert payload["supervisor_state"] == "RUNNING"
    assert payload["next_action"] == "RUN"
    assert payload["reviewer_configured"] is False

    # The key is reported as absent, never quoted.
    assert payload["reviewer"]["api_key_present"] is False

    assert read_status(supervisor_repo) == before


def test_a_passing_cycle_advances_exactly_one_phase(
    supervisor_repo: Path,
    run_supervisor,
    scripted_worker,
):
    start_work_task(supervisor_repo)

    completed = run_supervisor(
        "run",
        "--max-cycles",
        "1",
        env=scripted_worker(PASS_REPLY),
    )

    assert completed.returncode == EXIT_OK

    payload = supervisor_result(completed)

    assert payload["supervisor_state"] == "RUNNING"
    assert payload["cycles"] == 1

    assert (
        read_status(supervisor_repo)["phase"]
        == "IMPLEMENT"
    )


def test_a_canonical_gate_request_stops_at_waiting_for_human(
    supervisor_repo: Path,
    run_supervisor,
    scripted_worker,
):
    """
    The one path that may produce WAITING_FOR_HUMAN.

    The gate is opened through statusctl, so the durable record and
    the supervisor's own report agree, and only a human can clear it.
    """

    start_work_task(supervisor_repo)

    completed = run_supervisor(
        "run",
        env=scripted_worker(GATE_REPLY),
    )

    assert completed.returncode == EXIT_WAITING_FOR_HUMAN

    payload = supervisor_result(completed)

    assert (
        payload["supervisor_state"]
        == "WAITING_FOR_HUMAN"
    )

    status = read_status(supervisor_repo)

    assert status["state"] == "BLOCKED_HUMAN_DECISION"

    gate = status["human_gate"]

    assert gate["type"] == "ARCHITECTURE_DECISION"
    assert gate["question"] == "Approve the design?"

    # Opened, never resolved.
    assert status["last_gate_decision"] is None


def test_refuses_to_run_inside_a_worker_session(
    supervisor_repo: Path,
    run_supervisor,
    scripted_worker,
):
    """
    A worker reports its outcome; it does not orchestrate.

    Letting a worker session drive a supervisor run would hand back
    every lifecycle capability that session is deliberately denied.
    """

    start_work_task(supervisor_repo)

    environment = scripted_worker(PASS_REPLY)

    environment["AUTOPILOT_WORKER_SESSION"] = "1"

    completed = run_supervisor(
        "run",
        env=environment,
    )

    assert completed.returncode == EXIT_BLOCKED_PERMISSION

    payload = supervisor_result(completed)

    assert payload["result"] == "REFUSED"
    assert payload["supervisor_state"] == "FAILED"

    assert (
        read_status(supervisor_repo)["phase"]
        == "ANALYZE"
    )


# ================================================================
# CYCLE ARTIFACTS
# ================================================================


def test_a_cycle_artifact_is_written_outside_the_runtime(
    supervisor_repo: Path,
    run_supervisor,
    scripted_worker,
):
    """
    `.claude/runtime` holds canonical state and nothing else.

    Cycle artifacts are working data, so they are kept physically
    apart rather than the runtime invariant being widened to admit
    them.
    """

    start_work_task(supervisor_repo)

    run_supervisor(
        "run",
        "--max-cycles",
        "1",
        env=scripted_worker(PASS_REPLY),
    )

    records = artifacts(supervisor_repo)

    assert len(records) == 1

    record = records[0]

    assert record["kind"] == "WORKER"
    assert record["phase"] == "ANALYZE"
    assert record["phase_after"] == "IMPLEMENT"
    assert record["result"] == "PASS"
    assert record["task_id"] == "supervisor-work-001"
    assert record["finished_at"] is not None

    assert len(record["attempts"]) == 1

    attempt = record["attempts"][0]

    assert attempt["outcome"] == "PASS"
    assert attempt["exit_code"] == 0
    assert attempt["session_id"]

    runtime_names = {
        path.name
        for path in (
            supervisor_repo
            / ".claude"
            / "runtime"
        ).iterdir()
    }

    assert runtime_names == {
        "status.json",
        "audit.jsonl",
        "latest_report.md",
    }


def test_an_artifact_is_written_even_when_the_cycle_fails(
    supervisor_repo: Path,
    run_supervisor,
    scripted_worker,
):
    """
    A failed cycle is the one most worth reading afterwards.

    An artifact written only on success would be missing exactly the
    runs an operator needs to diagnose.
    """

    start_work_task(supervisor_repo)

    completed = run_supervisor(
        "run",
        env=scripted_worker("CRASH"),
    )

    assert completed.returncode == EXIT_BLOCKED_TECHNICAL

    assert (
        supervisor_result(completed)["supervisor_state"]
        == "FAILED"
    )

    records = artifacts(supervisor_repo)

    assert len(records) == 1

    record = records[0]

    assert record["result"] == "BLOCKED"

    attempt = record["attempts"][0]

    assert attempt["exit_code"] == 3
    assert "worker exploded" in attempt["stderr_tail"]

    # The phase never moved, so a rerun resumes where this stopped.
    assert (
        read_status(supervisor_repo)["phase"]
        == "ANALYZE"
    )


# ================================================================
# DECISION AUTONOMY POLICY
# ================================================================


def test_a_technical_choice_is_auto_decided_and_the_run_continues(
    supervisor_repo: Path,
    run_supervisor,
    scripted_worker,
    briefs,
):
    """
    A worker that recommends an option has already made the choice.

    Taking that recommendation is the narrowest possible autonomy:
    the supervisor contributes no judgement of its own, it declines to
    treat a decision the worker already made as a question.
    """

    start_work_task(supervisor_repo)

    completed = run_supervisor(
        "run",
        "--max-cycles",
        "1",
        env=scripted_worker(
            CHOICE_REPLY,
            PASS_REPLY,
        ),
    )

    assert completed.returncode == EXIT_OK

    record = artifacts(supervisor_repo)[0]

    assert len(record["auto_decisions"]) == 1

    decision = record["auto_decisions"][0]

    assert (
        decision["choice"]
        == CHOICE_REPLY["recommendation"]
    )

    assert decision["question"] == CHOICE_REPLY["question"]

    assert (
        CHOICE_REPLY["alternatives"][0]
        in decision["options_considered"]
    )

    assert decision["decided_at"]

    # The run continued rather than stopping for a human.
    assert (
        read_status(supervisor_repo)["phase"]
        == "IMPLEMENT"
    )

    assert (
        read_status(supervisor_repo)["state"]
        == "RUNNING"
    )

    # The decision was told to the worker, not merely filed.
    second = briefs()[1]["supervisor"]

    assert (
        CHOICE_REPLY["recommendation"]
        in second["guidance"]
    )


def test_an_unanswerable_stop_is_re_asked_once_then_fails_closed(
    supervisor_repo: Path,
    run_supervisor,
    scripted_worker,
    briefs,
):
    """
    Absent a recommendation there is nothing to select.

    Inventing a choice from alternatives the supervisor cannot
    evaluate would be the opposite of safe, so the worker is asked to
    reformulate exactly once and the run then stops.
    """

    start_work_task(supervisor_repo)

    completed = run_supervisor(
        "run",
        env=scripted_worker(STOPPED_REPLY),
    )

    assert completed.returncode == EXIT_BLOCKED_TECHNICAL

    payload = supervisor_result(completed)

    assert payload["supervisor_state"] == "FAILED"
    assert "twice without a canonical result" in payload["detail"]

    record = artifacts(supervisor_repo)[0]

    # Bounded at two: the re-ask is the only one.
    assert len(record["attempts"]) == 2
    assert record["auto_decisions"] == []

    assert len(briefs()) == 2

    guidance = briefs()[1]["supervisor"]["guidance"]

    assert "only re-ask" in guidance

    # No gate was invented to explain the stop.
    status = read_status(supervisor_repo)

    assert status["state"] == "RUNNING"
    assert status["human_gate"] is None


def test_an_incomplete_gate_request_is_re_asked_not_escalated(
    supervisor_repo: Path,
    run_supervisor,
    scripted_worker,
):
    """
    HUMAN_GATE_REQUIRED without a gate type is a worker that stopped.

    Escalating it would put a question in front of a human that names
    no decision, so it is re-asked instead — and a second answer that
    completes the phase is simply accepted.
    """

    start_work_task(supervisor_repo)

    completed = run_supervisor(
        "run",
        "--max-cycles",
        "1",
        env=scripted_worker(
            {
                "outcome": "HUMAN_GATE_REQUIRED",
                "summary": "something needs deciding",
            },
            PASS_REPLY,
        ),
    )

    assert completed.returncode == EXIT_OK

    status = read_status(supervisor_repo)

    assert status["state"] == "RUNNING"
    assert status["human_gate"] is None
    assert status["phase"] == "IMPLEMENT"


@pytest.mark.parametrize(
    "reply",
    [
        # A GIT_INTEGRATION request naming no action: the gate cannot
        # say which boundary it would authorize.
        {
            "outcome": "HUMAN_GATE_REQUIRED",
            "summary": "the branch needs publishing",
            "gate_type": "GIT_INTEGRATION",
            "question": "Publish the branch?",
            "recommendation": "Push it.",
        },
        # A gate request naming no question: there is nothing for a
        # human to answer.
        {
            "outcome": "HUMAN_GATE_REQUIRED",
            "summary": "the signing key needs rotating",
            "gate_type": "SECURITY_AUTH",
            "recommendation": "Rotate the key.",
        },
        # A gate type this build does not recognise. Still a claim of
        # gate-worthiness, so it may not be settled either.
        {
            "outcome": "HUMAN_GATE_REQUIRED",
            "summary": "the schema changes shape",
            "gate_type": "security_auth",
            "question": "Change it?",
            "recommendation": "Change it.",
        },
    ],
    ids=[
        "missing-action",
        "missing-question",
        "unrecognised-type",
    ],
)
def test_an_incomplete_gate_request_is_never_auto_decided(
    supervisor_repo: Path,
    run_supervisor,
    scripted_worker,
    briefs,
    reply,
):
    """
    A named gate type is a claim that the choice needs authority.

    An incomplete gate request still carries a recommendation, so
    nothing but the gate type distinguishes it from an ordinary
    technical choice. Settling it would use the worker's own
    recommendation to overrule the escalation the worker asked for,
    and an AUTO_DECISION may only settle a choice that triggers no
    gate. So it is re-asked, like any other non-canonical response.
    """

    start_work_task(supervisor_repo)

    completed = run_supervisor(
        "run",
        "--max-cycles",
        "1",
        env=scripted_worker(
            reply,
            PASS_REPLY,
        ),
    )

    assert completed.returncode == EXIT_OK

    record = artifacts(supervisor_repo)[0]

    # The recommendation was not taken as a decision.
    assert record["auto_decisions"] == []

    # It was re-asked instead, and told to reformulate rather than
    # told its own recommendation had been adopted.
    assert len(briefs()) == 2

    guidance = briefs()[1]["supervisor"]["guidance"]

    assert "only re-ask" in guidance
    assert reply["recommendation"] not in guidance

    # No gate was opened from a request that could not carry one.
    status = read_status(supervisor_repo)

    assert status["state"] == "RUNNING"
    assert status["human_gate"] is None
    assert status["phase"] == "IMPLEMENT"


def test_unparseable_output_is_re_asked_once_then_fails_closed(
    supervisor_repo: Path,
    run_supervisor,
    scripted_worker,
    briefs,
):
    start_work_task(supervisor_repo)

    completed = run_supervisor(
        "run",
        env=scripted_worker("PROSE"),
    )

    assert completed.returncode == EXIT_BLOCKED_TECHNICAL

    assert (
        supervisor_result(completed)["supervisor_state"]
        == "FAILED"
    )

    assert len(briefs()) == 2

    guidance = briefs()[1]["supervisor"]["guidance"]

    assert "could not be parsed" in guidance

    record = artifacts(supervisor_repo)[0]

    assert record["attempts"][0]["outcome"] == "UNPARSEABLE"


def test_every_brief_carries_the_decision_autonomy_policy(
    supervisor_repo: Path,
    run_supervisor,
    scripted_worker,
    briefs,
    load_supervisor_module,
):
    start_work_task(supervisor_repo)

    run_supervisor(
        "run",
        "--max-cycles",
        "1",
        env=scripted_worker(PASS_REPLY),
    )

    module = load_supervisor_module()

    brief = briefs()[0]

    assert (
        brief["supervisor"]["decision_autonomy_policy"]
        == module.DECISION_AUTONOMY_POLICY
    )

    # The base brief is autopilot's, so there is one description of
    # the phase contract rather than two that can drift.
    assert brief["phase"] == "ANALYZE"
    assert brief["task_id"] == "supervisor-work-001"

    assert brief["supervisor"]["attempt"] == 1


def test_each_attempt_gets_a_fresh_session(
    supervisor_repo: Path,
    run_supervisor,
    scripted_worker,
):
    """
    Bounded turns come from the supervisor, not from a conversation.

    A session carried forward between attempts would quietly restore
    the unbounded loop the cycle budget exists to prevent.
    """

    start_work_task(supervisor_repo)

    run_supervisor(
        "run",
        env=scripted_worker(STOPPED_REPLY),
    )

    attempts = artifacts(supervisor_repo)[0]["attempts"]

    sessions = [
        attempt["session_id"]
        for attempt in attempts
    ]

    assert len(sessions) == 2
    assert len(set(sessions)) == 2


# ================================================================
# EXTERNAL REVIEW
# ================================================================


REVIEWER_STUB = (
    "payload = json.load(open(sys.argv[-1]))\n"
    "open(os.environ['REVIEW_PAYLOAD_LOG'], 'w').write(\n"
    "    json.dumps(payload))\n"
    "verdict = json.loads(os.environ['REVIEW_VERDICT'])\n"
    "if verdict == 'PROSE':\n"
    "    print('Looks fine to me.')\n"
    "    sys.exit(0)\n"
    "if verdict == 'CRASH':\n"
    "    sys.stderr.write('reviewer exploded\\n')\n"
    "    sys.exit(4)\n"
    "print(json.dumps(verdict))\n"
)


@pytest.fixture
def scripted_reviewer(
    supervisor_repo: Path,
    tmp_path: Path,
) -> Callable[..., dict[str, str]]:
    path = supervisor_repo / "stub_reviewer.py"

    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys, os\n"
        + REVIEWER_STUB
    )

    path.chmod(
        path.stat().st_mode | stat.S_IEXEC
    )

    def configure(verdict) -> dict[str, str]:
        return {
            "SUPERVISOR_REVIEWER": json.dumps(
                [
                    "python3",
                    str(path),
                ]
            ),
            "REVIEW_VERDICT": json.dumps(verdict),
            "REVIEW_PAYLOAD_LOG": str(
                tmp_path / "review-payload"
            ),
        }

    return configure


APPROVE_VERDICT = {
    "verdict": "APPROVE",
    "summary": "The work is sound.",
}


def test_external_review_still_stops_without_a_reviewer(
    committed_repo: Path,
    run_supervisor,
    scripted_worker,
):
    """
    A missing reviewer never becomes an approval.

    EXTERNAL_REVIEW stays exactly the stop autopilot already makes it,
    because the question is whether an independent review can happen
    at all, not whether one is preferred.

    The stop is reported as FAILED rather than as a wait. No gate was
    opened, runtime state is still RUNNING, and there is nothing for a
    human to resolve, so WAITING_FOR_HUMAN would name a decision that
    does not exist — S-04 holds that word for a canonical Human Gate
    alone. EXIT_WAITING_EXTERNAL keeps the reason for the stop
    distinguishable without borrowing the gate's word for it.
    """

    start_work_task(committed_repo)

    advance_to(committed_repo, "EXTERNAL_REVIEW")

    completed = run_supervisor(
        "run",
        env=scripted_worker(PASS_REPLY),
    )

    # The distinguishing exit code survives the correction: the run
    # stopped for want of a reviewer, and 14 still says exactly that.
    assert completed.returncode == EXIT_WAITING_EXTERNAL

    payload = supervisor_result(completed)

    assert payload["result"] == "WAITING_EXTERNAL"

    assert (
        payload["supervisor_state"]
        == "FAILED"
    )

    status = read_status(committed_repo)

    assert status["phase"] == "EXTERNAL_REVIEW"

    # Nothing was fabricated to justify the stop: no Human Gate, and a
    # task still RUNNING.
    assert status["state"] == "RUNNING"
    assert status["human_gate"] is None


def test_an_approving_review_advances_the_phase(
    committed_repo: Path,
    run_supervisor,
    scripted_worker,
    scripted_reviewer,
):
    start_work_task(committed_repo)

    advance_to(committed_repo, "EXTERNAL_REVIEW")

    environment = scripted_worker(PASS_REPLY)

    environment.update(
        scripted_reviewer(APPROVE_VERDICT)
    )

    completed = run_supervisor(
        "run",
        "--max-cycles",
        "1",
        env=environment,
    )

    assert completed.returncode == EXIT_OK

    assert (
        read_status(committed_repo)["phase"]
        == "CORRECT_EXTERNAL_FINDINGS"
    )

    record = artifacts(committed_repo)[0]

    assert record["kind"] == "EXTERNAL_REVIEW"
    assert record["review"]["verdict"] == "APPROVE"

    # A review is not a worker phase; no session was spent on it.
    assert record["attempts"] == []


def test_a_review_requesting_changes_is_carried_into_the_next_phase(
    committed_repo: Path,
    run_supervisor,
    scripted_worker,
    scripted_reviewer,
    briefs,
):
    """
    The findings are the whole point of the phase that follows.

    They are read back from the artifact rather than held in memory,
    so a resumed run corrects against the review that actually
    happened.
    """

    start_work_task(committed_repo)

    advance_to(committed_repo, "EXTERNAL_REVIEW")

    environment = scripted_worker(PASS_REPLY)

    environment.update(
        scripted_reviewer(
            {
                "verdict": "REQUEST_CHANGES",
                "summary": "One defect must be corrected.",
                "findings": [
                    {
                        "classification": "ACCEPT",
                        "detail": "The retry bound is off by one.",
                    },
                ],
            }
        )
    )

    completed = run_supervisor(
        "run",
        "--max-cycles",
        "2",
        env=environment,
    )

    assert completed.returncode == EXIT_OK

    assert (
        read_status(committed_repo)["phase"]
        == "FINAL_VERIFY"
    )

    # The correcting phase was told what to correct.
    review = briefs()[0]["supervisor"]["external_review"]

    assert review["verdict"] == "REQUEST_CHANGES"

    assert (
        review["findings"][0]["detail"]
        == "The retry bound is off by one."
    )


def test_a_review_requiring_a_human_decision_opens_a_gate(
    committed_repo: Path,
    run_supervisor,
    scripted_worker,
    scripted_reviewer,
):
    start_work_task(committed_repo)

    advance_to(committed_repo, "EXTERNAL_REVIEW")

    environment = scripted_worker(PASS_REPLY)

    environment.update(
        scripted_reviewer(
            {
                "verdict": "HUMAN_DECISION_REQUIRED",
                "summary": "A scope decision blocks completion.",
                "gate_type": "SCOPE_EXPANSION",
                "question": "Widen the schema or defer?",
                "recommendation": "Defer to a separate task.",
            }
        )
    )

    completed = run_supervisor(
        "run",
        env=environment,
    )

    assert completed.returncode == EXIT_WAITING_FOR_HUMAN

    status = read_status(committed_repo)

    assert status["state"] == "BLOCKED_HUMAN_DECISION"

    gate = status["human_gate"]

    assert gate["type"] == "SCOPE_EXPANSION"
    assert gate["question"] == "Widen the schema or defer?"

    # The reviewer supplied the substance; it did not resolve it.
    assert status["last_gate_decision"] is None


@pytest.mark.parametrize(
    "verdict, expected",
    [
        ("PROSE", "no verdict"),
        ("CRASH", "exited 4"),
        (
            {
                "verdict": "APPROVE",
                "summary": "",
            },
            "invalid verdict",
        ),
    ],
)
def test_a_review_that_did_not_happen_is_never_an_approval(
    committed_repo: Path,
    run_supervisor,
    scripted_worker,
    scripted_reviewer,
    verdict,
    expected: str,
):
    """
    Prose, a crash and a malformed verdict are the same outcome.

    None of them is a review, so none of them may advance the phase.
    """

    start_work_task(committed_repo)

    advance_to(committed_repo, "EXTERNAL_REVIEW")

    environment = scripted_worker(PASS_REPLY)

    environment.update(
        scripted_reviewer(verdict)
    )

    completed = run_supervisor(
        "run",
        env=environment,
    )

    assert completed.returncode == EXIT_BLOCKED_TECHNICAL

    payload = supervisor_result(completed)

    assert payload["supervisor_state"] == "FAILED"
    assert expected in payload["detail"]

    assert (
        read_status(committed_repo)["phase"]
        == "EXTERNAL_REVIEW"
    )


def test_the_reviewer_sees_the_work_and_nothing_else(
    committed_repo: Path,
    run_supervisor,
    scripted_worker,
    scripted_reviewer,
    tmp_path: Path,
):
    """
    Exactly what the architecture allows the reviewer to receive.

    Worker results, the diff, test evidence, task and phase context,
    the last Human Gate decision and the contract extract. No
    credentials, no environment, no runtime file contents beyond what
    statusctl itself reports.
    """

    start_work_task(committed_repo)

    advance_to(committed_repo, "EXTERNAL_REVIEW")

    (committed_repo / "changed.py").write_text(
        "SECRET_LOOKING = 'not actually a secret'\n"
    )

    environment = scripted_worker(PASS_REPLY)

    environment.update(
        scripted_reviewer(APPROVE_VERDICT)
    )

    environment["OPENAI_API_KEY"] = "sk-must-never-be-sent"

    run_supervisor(
        "run",
        "--max-cycles",
        "1",
        env=environment,
    )

    payload = json.loads(
        (tmp_path / "review-payload").read_text()
    )

    assert set(payload) == {
        "task",
        "tests",
        "integration_actions",
        "last_gate_decision",
        "worker_history",
        "diff",
        "diff_truncated",
        "diff_stat",
        "untracked_entries",
        "contract",
        "contract_extract",
    }

    assert payload["task"]["task_id"] == "supervisor-work-001"
    assert payload["task"]["phase"] == "EXTERNAL_REVIEW"

    assert "changed.py" in payload["untracked_entries"]

    # The contract extract is the relevant sections, not the file.
    assert "## 11." in payload["contract_extract"]
    assert "## 1. Role" not in payload["contract_extract"]

    assert (
        "sk-must-never-be-sent"
        not in json.dumps(payload)
    )


def test_a_resolved_gate_is_carried_into_the_next_review(
    committed_repo: Path,
    run_supervisor,
    scripted_worker,
    scripted_reviewer,
    tmp_path: Path,
):
    """
    A question a human already answered reaches the reviewer answered.

    EXTERNAL_REVIEW resumes on the phase a reviewer-proposed gate
    stopped it at, and every other input is derived from the same work:
    the diff, the tests and the worker history are what they were when
    the reviewer asked. A reviewer given only those has no way to tell
    a first review from a re-review, so it can propose the gate it
    already proposed, and the loop closes on a human answering the same
    question forever.

    The human's decision is the one input that changed, so it is the
    one the payload has to carry.
    """

    start_work_task(committed_repo)

    advance_to(committed_repo, "EXTERNAL_REVIEW")

    run_statusctl(
        committed_repo,
        "gate",
        "open",
        "SEMANTIC_DECISION",
        "Which meaning does WAITING_FOR_HUMAN carry?",
        "Reserve it for a canonical human decision boundary.",
    )

    gate_id = read_status(
        committed_repo
    )["human_gate"]["gate_id"]

    # The human side of the boundary, performed here as an operator
    # would perform it, against this test's own isolated controller.
    run_statusctl(
        committed_repo,
        "gate",
        "approve",
        gate_id,
        "APPROVED: reserve it for a canonical boundary.",
    )

    environment = scripted_worker(PASS_REPLY)

    environment.update(
        scripted_reviewer(APPROVE_VERDICT)
    )

    run_supervisor(
        "run",
        "--max-cycles",
        "1",
        env=environment,
    )

    payload = json.loads(
        (tmp_path / "review-payload").read_text()
    )

    decision = payload["last_gate_decision"]

    assert decision is not None, (
        "The reviewer was handed no record of the decision a human "
        "had already made, so a re-review sees the evidence that "
        "produced the gate and nothing that answers it."
    )

    assert decision["gate_id"] == gate_id
    assert decision["decision"] == "APPROVED"

    assert (
        "canonical boundary"
        in decision["record"]
    )

    # Carried, not acted on: the record is advisory like everything
    # else the reviewer receives, and reaches it as data rather than as
    # a supervisor-side rule about which gates may be raised.
    assert (
        payload["task"]["phase"]
        == "EXTERNAL_REVIEW"
    )


def test_the_reviewer_is_told_to_read_the_resolved_decision(
    load_reviewer_module,
):
    """
    The record is only useful if the reviewer is told what it means.

    A payload field the instructions never mention is a field the
    reviewer may ignore, and ignoring it reopens exactly the loop the
    field exists to close.
    """

    instructions = load_reviewer_module().REVIEW_INSTRUCTIONS

    assert "last_gate_decision" in instructions

    assert "HUMAN_DECISION_REQUIRED" in instructions


# ================================================================
# INTEGRATION TEST DISPOSITION
# ================================================================
#
# NOT_REQUIRED is permitted here, and only here, and only on proof.
# These exercise the evidence rules directly, because constructing a
# real chain of completed tasks would test statusctl rather than the
# rule under examination.


@pytest.fixture
def supervisor_module(
    supervisor_repo: Path,
    load_supervisor_module,
):
    return load_supervisor_module()


@pytest.fixture
def write_audit(
    supervisor_repo: Path,
) -> Callable[[list[dict]], None]:
    def write(events: list[dict]) -> None:
        path = (
            supervisor_repo
            / ".claude"
            / "runtime"
            / "audit.jsonl"
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            "".join(
                json.dumps(event) + "\n"
                for event in events
            )
        )

    return write


def ready_event(
    task_id: str = "parent-work-001",
    task_kind: str = "WORK",
    targeted: str = "PASS",
    full: str = "PASS",
) -> dict:
    return {
        "timestamp": "2026-08-14T09:00:00.000000Z",
        "event": "TASK_READY_FOR_HUMAN_REVIEW",
        "task_id": task_id,
        "task_kind": task_kind,
        "targeted_result": targeted,
        "full_result": full,
    }


def test_proven_parent_evidence_is_accepted(
    supervisor_module,
    write_audit,
):
    write_audit([ready_event()])

    event = supervisor_module.assert_parent_evidence(
        "parent-work-001"
    )

    assert event["task_id"] == "parent-work-001"


def test_absent_parent_evidence_fails_closed(
    supervisor_module,
    write_audit,
):
    write_audit([])

    with pytest.raises(
        supervisor_module.Stop,
        match="no evidence",
    ):
        supervisor_module.assert_parent_evidence(
            "parent-work-001"
        )


def test_an_exemption_is_not_inherited_from_an_exemption(
    supervisor_module,
    write_audit,
):
    """
    The rule that stops NOT_REQUIRED propagating down a chain.

    A parent whose own results were exemptions proves nothing was
    ever tested, so it cannot justify a further exemption.
    """

    write_audit(
        [
            ready_event(
                targeted="NOT_REQUIRED",
                full="NOT_REQUIRED",
            )
        ]
    )

    with pytest.raises(
        supervisor_module.Stop,
        match="only on proven PASS and PASS",
    ):
        supervisor_module.assert_parent_evidence(
            "parent-work-001"
        )


def test_an_integration_parent_is_refused(
    supervisor_module,
    write_audit,
):
    write_audit(
        [
            ready_event(
                task_kind="INTEGRATION",
            )
        ]
    )

    with pytest.raises(
        supervisor_module.Stop,
        match="not WORK",
    ):
        supervisor_module.assert_parent_evidence(
            "parent-work-001"
        )


def test_an_event_without_a_task_kind_is_refused(
    supervisor_module,
    write_audit,
):
    """
    A missing kind is refused rather than assumed to be WORK.

    Events recorded before the task model existed carry no kind, and
    defaulting one would let a task whose kind was never recorded
    supply evidence about a property nobody wrote down.
    """

    event = ready_event()

    event.pop("task_kind")

    write_audit([event])

    with pytest.raises(
        supervisor_module.Stop,
        match="records no task_kind",
    ):
        supervisor_module.assert_parent_evidence(
            "parent-work-001"
        )


def test_the_most_recent_completion_decides(
    supervisor_module,
    write_audit,
):
    """
    A parent that completed again on weaker evidence is judged on it.

    Reading the older, stronger record would answer a question about
    the parent's current state with a fact about its past one.
    """

    write_audit(
        [
            ready_event(),
            ready_event(
                targeted="NOT_REQUIRED",
            ),
        ]
    )

    with pytest.raises(
        supervisor_module.Stop,
        match="only on proven PASS and PASS",
    ):
        supervisor_module.assert_parent_evidence(
            "parent-work-001"
        )


def test_a_work_task_never_takes_the_disposition_path(
    supervisor_module,
    write_audit,
    monkeypatch,
):
    """
    The narrowing that keeps this from being a general exemption.

    A WORK task is the case where recording NOT_REQUIRED would let an
    orchestrator exempt itself from testing, so it returns before any
    evidence is even considered.
    """

    write_audit([ready_event()])

    calls: list[tuple] = []

    monkeypatch.setattr(
        supervisor_module.autopilot,
        "statusctl_checked",
        lambda *args: calls.append(args),
    )

    disposition = (
        supervisor_module.apply_integration_test_disposition(
            {
                "task_kind": "WORK",
                "parent_task_id": None,
                "tests": {
                    "targeted": None,
                    "full": None,
                },
            }
        )
    )

    assert disposition is None
    assert calls == []


def test_an_existing_test_result_is_never_overwritten(
    supervisor_module,
    write_audit,
    monkeypatch,
):
    """
    Only empty slots are filled.

    A real PASS or a real FAIL recorded earlier is evidence someone
    produced, and an exemption written over it would erase it.
    """

    write_audit([ready_event()])

    calls: list[tuple] = []

    monkeypatch.setattr(
        supervisor_module.autopilot,
        "statusctl_checked",
        lambda *args: calls.append(args),
    )

    disposition = (
        supervisor_module.apply_integration_test_disposition(
            {
                "task_kind": "INTEGRATION",
                "parent_task_id": "parent-work-001",
                "tests": {
                    "targeted": {
                        "result": "PASS",
                        "command": "pytest -q",
                    },
                    "full": None,
                },
            }
        )
    )

    assert disposition["kinds"] == ["full"]

    assert [call[1] for call in calls] == ["full"]

    assert calls[0][0] == "test-not-required"

    # The reason quotes the proof rather than asserting the rule.
    assert "parent-work-001" in calls[0][2]
    assert "2026-08-14T09:00:00.000000Z" in calls[0][2]


def test_a_full_exemption_records_both_kinds(
    supervisor_module,
    write_audit,
    monkeypatch,
):
    write_audit([ready_event()])

    calls: list[tuple] = []

    monkeypatch.setattr(
        supervisor_module.autopilot,
        "statusctl_checked",
        lambda *args: calls.append(args),
    )

    disposition = (
        supervisor_module.apply_integration_test_disposition(
            {
                "task_kind": "INTEGRATION",
                "parent_task_id": "parent-work-001",
                "tests": {
                    "targeted": None,
                    "full": None,
                },
            }
        )
    )

    assert disposition["kinds"] == ["targeted", "full"]

    assert [call[1] for call in calls] == [
        "targeted",
        "full",
    ]


# ================================================================
# PREFLIGHT AND GATE TRANSLATION
# ================================================================


def test_an_injected_worker_is_not_preflighted(
    supervisor_module,
):
    """
    Only the real Claude CLI is held to Claude's flag vocabulary.

    An injected worker is a test stub or an operator's own program,
    and preflighting it would fail for a reason unrelated to the
    posture under test.
    """

    result = supervisor_module.preflight_worker_cli(
        [
            "python3",
            "/tmp/stub.py",
        ]
    )

    assert result["checked"] is False
    assert "not the Claude CLI" in result["reason"]


def test_a_cli_without_structured_output_is_refused(
    supervisor_module,
    tmp_path: Path,
):
    """
    The supervisor never falls back to parsing prose.

    The published flag set could not be re-verified when this was
    designed, so it is verified at startup — and an unrecognised flag
    stops the run explicitly rather than degrading silently.
    """

    fake = tmp_path / "claude"

    fake.write_text(
        "#!/usr/bin/env python3\n"
        "print('usage: claude [--permission-mode MODE]')\n"
    )

    fake.chmod(
        fake.stat().st_mode | stat.S_IEXEC
    )

    with pytest.raises(
        supervisor_module.Stop,
        match="does not advertise",
    ):
        supervisor_module.preflight_worker_cli(
            [str(fake)]
        )


def test_a_cli_advertising_the_flags_passes_preflight(
    supervisor_module,
    tmp_path: Path,
):
    fake = tmp_path / "claude"

    fake.write_text(
        "#!/usr/bin/env python3\n"
        "print('usage: claude --output-format json "
        "--json-schema SCHEMA')\n"
    )

    fake.chmod(
        fake.stat().st_mode | stat.S_IEXEC
    )

    result = supervisor_module.preflight_worker_cli(
        [str(fake)]
    )

    assert result["checked"] is True


def test_a_reviewer_may_not_request_an_integration_gate(
    supervisor_module,
):
    """
    The second place that holds the reviewer's own exclusion.

    An integration boundary belongs to a declared Integration Task
    scope and to the human who approves it, so a verdict that reached
    here naming one is refused rather than translated.
    """

    with pytest.raises(
        supervisor_module.Stop,
        match="may not request an integration authorization",
    ):
        supervisor_module.gate_request_from_verdict(
            {
                "gate_type": "GIT_INTEGRATION",
                "question": "Push it?",
                "recommendation": "Push.",
            }
        )


def test_a_translated_gate_request_names_no_action(
    supervisor_module,
):
    request = supervisor_module.gate_request_from_verdict(
        {
            "gate_type": "SEMANTIC_DECISION",
            "question": "Which semantics?",
            "recommendation": "Use A.",
            "alternatives": ["Use B."],
            "risks": ["B is harder to reverse."],
        }
    )

    assert request["outcome"] == "HUMAN_GATE_REQUIRED"
    assert request["gate_type"] == "SEMANTIC_DECISION"
    assert request["action"] is None
    assert request["alternatives"] == ["Use B."]


def test_external_states_cover_every_orchestrator_result(
    supervisor_module,
):
    """
    Four external states, and no result that falls outside them.

    An unmapped result defaults to FAILED, which is the safe reading:
    a run whose outcome the supervisor does not recognise has not
    demonstrably succeeded.
    """

    assert set(
        supervisor_module.SUPERVISOR_STATE_BY_RESULT.values()
    ) == {
        supervisor_module.STATE_RUNNING,
        supervisor_module.STATE_WAITING_FOR_HUMAN,
        supervisor_module.STATE_READY,
        supervisor_module.STATE_FAILED,
    }

    assert (
        supervisor_module.SUPERVISOR_STATE_BY_RESULT.get(
            "something-new"
        )
        is None
    )


# ================================================================
# C1-FIX — ONE SCHEMA SEMANTICS, BOTH ORCHESTRATORS
# ================================================================
#
# The supervisor preflights the exact command its cycles will run. If
# it built that command differently from autopilot, the preflight
# would be checking a program nobody invokes — and the difference that
# mattered here was precisely the --json-schema value, which the CLI
# parses as a document rather than opening as a file.


def test_the_preflight_command_carries_the_schema_document(
    supervisor_module,
    supervisor_repo: Path,
    monkeypatch,
):
    """
    C1-FIX: the preflighted argv is a runnable argv.

    Built through autopilot's own worker seam, so there is one
    definition of the worker command and the preflight cannot drift
    away from the cycles it is meant to vouch for.
    """

    captured: dict = {}

    def capture(argv):
        captured["argv"] = list(argv)

        raise supervisor_module.blocked(
            "stopping after preflight"
        )

    monkeypatch.setattr(
        supervisor_module,
        "preflight_worker_cli",
        capture,
    )

    monkeypatch.chdir(supervisor_repo)

    class Args:
        max_cycles = 1

    supervisor_module.cmd_run(Args())

    argv = captured["argv"]

    assert argv[0] == "claude"

    index = argv.index("--json-schema")

    value = argv[index + 1]

    assert json.loads(value) == (
        supervisor_module.autopilot.worker_response_schema()
    )

    assert not value.startswith("/")

    assert (
        supervisor_module.autopilot.WORKER_RESPONSE_SCHEMA_FILENAME
        not in value
    )


def test_both_orchestrators_send_the_same_schema_value(
    supervisor_module,
    monkeypatch,
):
    """
    C1-FIX: identical semantics, from one implementation.

    The supervisor imports autopilot rather than copying it, so this
    pins that the import is still the whole of the relationship for
    the response schema.
    """

    autopilot_module = supervisor_module.autopilot

    monkeypatch.delenv(
        "AUTOPILOT_WORKER",
        raising=False,
    )

    with autopilot_module.worker_schema_file() as schema:
        argv = autopilot_module.worker_argv(schema)

    value = argv[
        argv.index("--json-schema") + 1
    ]

    assert json.loads(value) == (
        autopilot_module.worker_response_schema()
    )

    assert str(schema) not in argv
