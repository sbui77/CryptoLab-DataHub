"""
Integration scope: which git actions an Integration Task may perform.

The lifecycle gap these pin: a Human Gate carried its scope only as
free text, so the controller could not tell an authorized commit from
an unauthorized push. `ready` had no integration precondition, and
`gate open` requires state=RUNNING — so an Integration Task could
complete with an intended push never authorized, and once complete
there was no lawful way left to authorize it.

Schema 3.1 records the intended scope instead of inferring it:

    an Integration Task declares its actions at start;
    a GIT_INTEGRATION gate names exactly one declared action;
    approval authorizes that action and nothing else;
    rejection declines it, which is the explicit not-in-scope record;
    performing it requires observed git evidence;
    completion requires every declared action to be resolved.

The gap is closed at the front of the lifecycle. Nothing here permits
a gate after READY_FOR_HUMAN_REVIEW — that is asserted as a
regression, because reopening a completed task would be the unsafe
way to solve the same problem.

Some tests drive real git inside the isolated fixture repository —
commits, a side branch, a merge, and a push to a local bare remote.
That is test plumbing for the evidence checks, not an integration
action on this project.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable

import pytest


CURRENT_SCHEMA = "3.1"

PREVIOUS_SCHEMA = "3.0"


# ================================================================
# HELPERS
# ================================================================


def git(
    repo: Path,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "user.name=Control Plane Test",
            *args,
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=check,
    )


def commit_something(
    repo: Path,
    name: str = "file.txt",
) -> str:
    """
    Create a commit in the fixture repository and return its sha.
    """

    (repo / name).write_text(
        f"{name}\n"
    )

    git(
        repo,
        "add",
        name,
    )

    git(
        repo,
        "commit",
        "-q",
        "-m",
        f"add {name}",
    )

    return git(
        repo,
        "rev-parse",
        "HEAD",
    ).stdout.strip()


def merge_a_branch(
    repo: Path,
    name: str = "feature",
) -> str:
    """
    Merge a side branch back with a real merge commit.

    `--no-ff` is deliberate: a fast-forward merge leaves no merge
    commit, and therefore no evidence that a merge happened at all.
    """

    current = git(
        repo,
        "branch",
        "--show-current",
    ).stdout.strip()

    git(
        repo,
        "checkout",
        "-q",
        "-b",
        name,
    )

    commit_something(
        repo,
        f"{name}.txt",
    )

    git(
        repo,
        "checkout",
        "-q",
        current,
    )

    git(
        repo,
        "merge",
        "-q",
        "--no-ff",
        "-m",
        f"merge {name}",
        name,
    )

    return git(
        repo,
        "rev-parse",
        "HEAD",
    ).stdout.strip()


def publish(
    repo: Path,
    remote: Path,
) -> None:
    """
    Give the fixture repository an upstream and publish HEAD to it.
    """

    subprocess.run(
        [
            "git",
            "init",
            "-q",
            "--bare",
            str(remote),
        ],
        check=True,
        capture_output=True,
    )

    git(
        repo,
        "remote",
        "add",
        "origin",
        str(remote),
    )

    branch = git(
        repo,
        "branch",
        "--show-current",
    ).stdout.strip()

    git(
        repo,
        "push",
        "-q",
        "-u",
        "origin",
        branch,
    )


def complete_work_task(
    run_statusctl,
    task_id: str = "work-001",
) -> None:
    """
    Drive a Work Task to READY_FOR_HUMAN_REVIEW.
    """

    assert run_statusctl(
        "start",
        task_id,
        "Work task",
    ).returncode == 0

    for kind in (
        "targeted",
        "full",
    ):
        assert run_statusctl(
            "test",
            kind,
            "PASS",
            "pytest -q",
            "all green",
        ).returncode == 0

    assert run_statusctl(
        "phase",
        "FINAL_VERIFY",
    ).returncode == 0

    assert run_statusctl(
        "ready",
    ).returncode == 0


def start_integration(
    run_statusctl,
    *actions: str,
    task_id: str = "integration-001",
) -> subprocess.CompletedProcess[str]:
    args = [
        "start-integration",
        task_id,
        "Integrate the reviewed work",
    ]

    for action in actions:
        args.extend(
            [
                "--action",
                action,
            ]
        )

    return run_statusctl(*args)


@pytest.fixture
def integration_task(
    run_statusctl,
) -> Callable[..., None]:
    """
    A started Integration Task declaring the given actions.

    Declaring nothing is spelled `none`, because an absent scope is
    refused. The task is left in INTEGRATE, which is the only phase
    where integration actions may be gated or performed; a test about
    the phase rule asks for a different one explicitly.
    """

    def start(
        *actions: str,
        phase: str = "INTEGRATE",
    ) -> None:
        complete_work_task(
            run_statusctl
        )

        assert start_integration(
            run_statusctl,
            *(
                actions
                or (
                    "none",
                )
            ),
        ).returncode == 0

        if phase != "ANALYZE":
            assert run_statusctl(
                "phase",
                phase,
            ).returncode == 0

    return start


def actions_by_name(
    status: dict,
) -> dict[str, dict]:
    return {
        entry["action"]: entry
        for entry in (
            status["integration_actions"]
            or []
        )
    }


def open_gate(
    run_statusctl,
    action: str | None = None,
    gate_type: str = "GIT_INTEGRATION",
    question: str = "Perform the integration action?",
    recommendation: str = "Approve it.",
) -> subprocess.CompletedProcess[str]:
    args = [
        "gate",
        "open",
        gate_type,
        question,
        recommendation,
    ]

    if action is not None:
        args.extend(
            [
                "--action",
                action,
            ]
        )

    return run_statusctl(*args)


def active_gate_id(
    read_status,
) -> str:
    gate = read_status()["human_gate"]

    assert gate is not None, "no active gate"

    return gate["gate_id"]


def approve(
    run_statusctl,
    read_status,
    record: str = "Approved.",
) -> subprocess.CompletedProcess[str]:
    return run_statusctl(
        "gate",
        "approve",
        active_gate_id(
            read_status
        ),
        record,
    )


def authorize(
    run_statusctl,
    read_status,
    action: str,
) -> None:
    """
    Open and approve the gate for one declared action.
    """

    assert open_gate(
        run_statusctl,
        action,
    ).returncode == 0

    assert approve(
        run_statusctl,
        read_status,
    ).returncode == 0


def events(
    read_audit,
) -> list[str]:
    return [
        event["event"]
        for event in read_audit()
    ]


# ================================================================
# DECLARED SCOPE
# ================================================================


def test_declared_actions_are_recorded_at_start(
    run_statusctl,
    read_status,
    read_audit,
):
    """
    Intent is recorded, not inferred later from a gate's prose.
    """

    complete_work_task(
        run_statusctl
    )

    assert start_integration(
        run_statusctl,
        "commit",
        "push",
    ).returncode == 0

    status = read_status()

    assert [
        entry["action"]
        for entry in status["integration_actions"]
    ] == [
        "commit",
        "push",
    ]

    for entry in status["integration_actions"]:
        assert entry["status"] == "DECLARED"

        assert entry["gate_id"] is None

        assert entry["decided_at"] is None

        assert entry["record"] is None

        assert entry["authorized_head"] is None

        assert entry["evidence"] is None

    started = [
        event
        for event in read_audit()
        if event["event"] == "INTEGRATION_TASK_STARTED"
    ][-1]

    assert started["declared_actions"] == [
        "commit",
        "push",
    ]


def test_declared_actions_are_normalized_and_deduplicated(
    run_statusctl,
    read_status,
):
    """
    Scope is a set of boundaries, recorded in a canonical order.
    """

    complete_work_task(
        run_statusctl
    )

    assert start_integration(
        run_statusctl,
        "push",
        "commit",
        "push",
    ).returncode == 0

    assert [
        entry["action"]
        for entry in read_status()["integration_actions"]
    ] == [
        "commit",
        "push",
    ]


def test_unknown_declared_action_is_refused(
    run_statusctl,
    read_status,
):
    complete_work_task(
        run_statusctl
    )

    result = start_integration(
        run_statusctl,
        "rebase",
    )

    assert result.returncode != 0

    # The Work Task must be left exactly as it was.
    assert (
        read_status()["state"]
        == "READY_FOR_HUMAN_REVIEW"
    )


def test_work_task_carries_no_integration_actions(
    run_statusctl,
    read_status,
):
    """
    Semantics A: the field belongs to the integration identity alone.
    """

    assert run_statusctl(
        "start",
        "work-001",
        "Work task",
    ).returncode == 0

    assert (
        read_status()["integration_actions"]
        is None
    )


def test_idle_runtime_carries_no_integration_actions(
    run_statusctl,
):
    status = json.loads(
        run_statusctl(
            "show",
        ).stdout
    )

    assert status["state"] == "IDLE"

    assert status["integration_actions"] is None

    assert (
        status["schema_version"]
        == CURRENT_SCHEMA
    )


# ================================================================
# GATE BINDING
# ================================================================


def test_git_integration_gate_requires_a_declared_action(
    integration_task,
    run_statusctl,
    read_status,
    read_audit,
):
    """
    An unnamed integration gate is exactly the ambiguity being fixed.
    """

    integration_task(
        "commit",
        "push",
    )

    result = open_gate(
        run_statusctl
    )

    assert result.returncode != 0

    status = read_status()

    assert status["state"] == "RUNNING"

    assert status["human_gate"] is None

    assert (
        "HUMAN_GATE_OPENED"
        not in events(
            read_audit
        )
    )


def test_gate_for_an_undeclared_action_is_refused(
    integration_task,
    run_statusctl,
    read_status,
    read_audit,
):
    """
    Scope cannot be widened by opening a gate for something the task
    never declared.
    """

    integration_task(
        "commit",
    )

    result = open_gate(
        run_statusctl,
        "push",
    )

    assert result.returncode != 0

    status = read_status()

    assert status["state"] == "RUNNING"

    assert status["human_gate"] is None

    assert (
        actions_by_name(
            status
        )["commit"]["status"]
        == "DECLARED"
    )

    assert (
        "HUMAN_GATE_OPENED"
        not in events(
            read_audit
        )
    )


def test_action_is_refused_for_a_non_integration_gate_type(
    integration_task,
    run_statusctl,
    read_status,
):
    integration_task(
        "commit",
    )

    result = open_gate(
        run_statusctl,
        "commit",
        gate_type="SEMANTIC_DECISION",
    )

    assert result.returncode != 0

    assert (
        read_status()["state"]
        == "RUNNING"
    )


def test_git_integration_gate_is_still_work_task_forbidden(
    run_statusctl,
    read_status,
):
    """
    The pre-existing Semantics A rule survives the new binding.
    """

    assert run_statusctl(
        "start",
        "work-001",
        "Work task",
    ).returncode == 0

    result = open_gate(
        run_statusctl,
        "commit",
    )

    assert result.returncode != 0

    assert (
        read_status()["human_gate"]
        is None
    )


def test_open_gate_records_the_action_it_authorizes(
    integration_task,
    run_statusctl,
    read_status,
):
    integration_task(
        "commit",
        "push",
    )

    assert open_gate(
        run_statusctl,
        "push",
    ).returncode == 0

    gate = read_status()["human_gate"]

    assert gate["type"] == "GIT_INTEGRATION"

    assert gate["action"] == "push"


def test_second_gate_for_an_authorized_action_is_refused(
    integration_task,
    run_statusctl,
    read_status,
):
    """
    Authorization is not re-requested; it is used or it is not.
    """

    integration_task(
        "commit",
    )

    authorize(
        run_statusctl,
        read_status,
        "commit",
    )

    result = open_gate(
        run_statusctl,
        "commit",
    )

    assert result.returncode != 0

    assert (
        read_status()["human_gate"]
        is None
    )


# ================================================================
# AUTHORIZATION DOES NOT SPREAD
# ================================================================


def test_commit_approval_does_not_authorize_push(
    integration_task,
    run_statusctl,
    read_status,
):
    """
    The defect in one sentence: approving a commit must say nothing
    about a push.
    """

    integration_task(
        "commit",
        "push",
    )

    authorize(
        run_statusctl,
        read_status,
        "commit",
    )

    actions = actions_by_name(
        read_status()
    )

    assert (
        actions["commit"]["status"]
        == "AUTHORIZED"
    )

    assert (
        actions["commit"]["gate_id"]
        is not None
    )

    assert (
        actions["push"]["status"]
        == "DECLARED"
    )

    assert actions["push"]["gate_id"] is None


def test_rejection_declines_the_action(
    integration_task,
    run_statusctl,
    read_status,
):
    """
    A rejected push is the explicit human-made not-in-scope record.
    """

    integration_task(
        "commit",
        "push",
    )

    assert open_gate(
        run_statusctl,
        "push",
    ).returncode == 0

    assert run_statusctl(
        "gate",
        "reject",
        active_gate_id(
            read_status
        ),
        "Not part of this integration.",
    ).returncode == 0

    entry = actions_by_name(
        read_status()
    )["push"]

    assert entry["status"] == "DECLINED"

    assert (
        entry["record"]
        == "Not part of this integration."
    )

    assert entry["decided_at"] is not None


def test_alternative_leaves_the_action_unresolved(
    integration_task,
    run_statusctl,
    read_status,
):
    """
    An alternative says "do something else", which is not an
    authorization. It must fail closed and stay re-openable.
    """

    integration_task(
        "push",
    )

    assert open_gate(
        run_statusctl,
        "push",
    ).returncode == 0

    assert run_statusctl(
        "gate",
        "alternative",
        active_gate_id(
            read_status
        ),
        "Push to a different remote instead.",
    ).returncode == 0

    assert (
        actions_by_name(
            read_status()
        )["push"]["status"]
        == "DECLARED"
    )

    # The decision returned the task to RUNNING, so the boundary can
    # be put to a human again.
    assert open_gate(
        run_statusctl,
        "push",
    ).returncode == 0


# ================================================================
# PERFORMING AN ACTION REQUIRES EVIDENCE
# ================================================================


def test_performed_requires_authorization_first(
    integration_task,
    run_statusctl,
    read_status,
):
    integration_task(
        "commit",
    )

    result = run_statusctl(
        "integration",
        "performed",
        "commit",
    )

    assert result.returncode != 0

    assert (
        actions_by_name(
            read_status()
        )["commit"]["status"]
        == "DECLARED"
    )


def test_performed_commit_requires_head_to_move(
    control_plane_repo: Path,
    integration_task,
    run_statusctl,
    read_status,
):
    """
    A claim is not evidence: the controller checks the repository.
    """

    integration_task(
        "commit",
    )

    authorize(
        run_statusctl,
        read_status,
        "commit",
    )

    refused = run_statusctl(
        "integration",
        "performed",
        "commit",
    )

    assert refused.returncode != 0

    assert (
        actions_by_name(
            read_status()
        )["commit"]["status"]
        == "AUTHORIZED"
    )

    sha = commit_something(
        control_plane_repo
    )

    assert run_statusctl(
        "integration",
        "performed",
        "commit",
    ).returncode == 0

    entry = actions_by_name(
        read_status()
    )["commit"]

    assert entry["status"] == "PERFORMED"

    assert sha[:7] in entry["evidence"]


def test_performed_push_requires_an_upstream(
    control_plane_repo: Path,
    integration_task,
    run_statusctl,
    read_status,
):
    """
    Without an upstream there is nothing to check, so the controller
    refuses rather than assuming the push happened.
    """

    commit_something(
        control_plane_repo,
        "seed.txt",
    )

    integration_task(
        "push",
    )

    authorize(
        run_statusctl,
        read_status,
        "push",
    )

    assert run_statusctl(
        "integration",
        "performed",
        "push",
    ).returncode != 0

    assert (
        actions_by_name(
            read_status()
        )["push"]["status"]
        == "AUTHORIZED"
    )


def test_push_records_the_authorized_commit_and_upstream(
    control_plane_repo: Path,
    tmp_path: Path,
    integration_task,
    run_statusctl,
    read_status,
):
    """
    The evidence names what was published and where.
    """

    authorized = commit_something(
        control_plane_repo,
        "seed.txt",
    )

    publish(
        control_plane_repo,
        tmp_path / "remote.git",
    )

    integration_task(
        "push",
    )

    authorize(
        run_statusctl,
        read_status,
        "push",
    )

    assert run_statusctl(
        "integration",
        "performed",
        "push",
    ).returncode == 0

    entry = actions_by_name(
        read_status()
    )["push"]

    assert entry["status"] == "PERFORMED"

    assert entry["authorized_head"] == authorized

    # Full SHAs, not abbreviations, and the ref that carries them.
    assert authorized in entry["evidence"]

    assert "origin/" in entry["evidence"]


def test_pushing_a_later_commit_does_not_satisfy_the_authorization(
    control_plane_repo: Path,
    tmp_path: Path,
    integration_task,
    run_statusctl,
    read_status,
):
    """
    A push gate authorizes publishing one exact commit.

    Approving the publication of A and then publishing B is not the
    authorized action, even though the branch is legitimately
    up to date afterwards: "HEAD is contained in upstream" would be
    satisfied by any later commit anybody pushed.
    """

    authorized = commit_something(
        control_plane_repo,
        "seed.txt",
    )

    publish(
        control_plane_repo,
        tmp_path / "remote.git",
    )

    integration_task(
        "push",
    )

    authorize(
        run_statusctl,
        read_status,
        "push",
    )

    later = commit_something(
        control_plane_repo,
        "later.txt",
    )

    git(
        control_plane_repo,
        "push",
        "-q",
        "origin",
        "HEAD",
    )

    assert later != authorized

    result = run_statusctl(
        "integration",
        "performed",
        "push",
    )

    assert result.returncode != 0

    entry = actions_by_name(
        read_status()
    )["push"]

    assert entry["status"] == "AUTHORIZED"

    assert entry["evidence"] is None


# ================================================================
# ER-04 — COMMIT AND MERGE EVIDENCE ARE NOT INTERCHANGEABLE
# ================================================================


def test_a_merge_does_not_satisfy_the_commit_action(
    control_plane_repo: Path,
    integration_task,
    run_statusctl,
    read_status,
):
    """
    Both actions move HEAD, so "HEAD moved" cannot tell them apart.

    A commit action is satisfied by an ordinary commit only.
    """

    commit_something(
        control_plane_repo,
        "seed.txt",
    )

    integration_task(
        "commit",
    )

    authorize(
        run_statusctl,
        read_status,
        "commit",
    )

    merge_a_branch(
        control_plane_repo
    )

    result = run_statusctl(
        "integration",
        "performed",
        "commit",
    )

    assert result.returncode != 0

    assert (
        actions_by_name(
            read_status()
        )["commit"]["status"]
        == "AUTHORIZED"
    )


def test_an_ordinary_commit_does_not_satisfy_the_merge_action(
    control_plane_repo: Path,
    integration_task,
    run_statusctl,
    read_status,
):
    """
    The mirror image: a merge action needs a merge.
    """

    commit_something(
        control_plane_repo,
        "seed.txt",
    )

    integration_task(
        "merge",
    )

    authorize(
        run_statusctl,
        read_status,
        "merge",
    )

    commit_something(
        control_plane_repo,
        "ordinary.txt",
    )

    result = run_statusctl(
        "integration",
        "performed",
        "merge",
    )

    assert result.returncode != 0

    assert (
        actions_by_name(
            read_status()
        )["merge"]["status"]
        == "AUTHORIZED"
    )


def test_commit_evidence_requires_exactly_one_transition(
    control_plane_repo: Path,
    integration_task,
    run_statusctl,
    read_status,
):
    """
    ER-07: descending from the authorized state is not enough.

    Authorizing a commit at A and then making B and C leaves C a
    single-parent commit descending from A, so ancestry alone would
    accept it. What the human authorized was one commit, and the
    record must mean that: the direct parent of HEAD has to be the
    authorized commit itself.
    """

    authorized = commit_something(
        control_plane_repo,
        "seed.txt",
    )

    integration_task(
        "commit",
    )

    authorize(
        run_statusctl,
        read_status,
        "commit",
    )

    commit_something(
        control_plane_repo,
        "b.txt",
    )

    third = commit_something(
        control_plane_repo,
        "c.txt",
    )

    # The weaker rule would have passed: A really is an ancestor of C.
    assert git(
        control_plane_repo,
        "merge-base",
        "--is-ancestor",
        authorized,
        third,
        check=False,
    ).returncode == 0

    result = run_statusctl(
        "integration",
        "performed",
        "commit",
    )

    assert result.returncode != 0

    entry = actions_by_name(
        read_status()
    )["commit"]

    assert entry["status"] == "AUTHORIZED"

    assert entry["evidence"] is None


def test_commit_evidence_accepts_one_commit_from_the_authorized_state(
    control_plane_repo: Path,
    integration_task,
    run_statusctl,
    read_status,
):
    """
    ER-07's positive half: the ordinary case still works.
    """

    authorized = commit_something(
        control_plane_repo,
        "seed.txt",
    )

    integration_task(
        "commit",
    )

    authorize(
        run_statusctl,
        read_status,
        "commit",
    )

    made = commit_something(
        control_plane_repo,
        "b.txt",
    )

    assert run_statusctl(
        "integration",
        "performed",
        "commit",
    ).returncode == 0

    entry = actions_by_name(
        read_status()
    )["commit"]

    assert entry["status"] == "PERFORMED"

    assert entry["authorized_head"] == authorized

    assert made in entry["evidence"]

    assert authorized in entry["evidence"]


def test_merge_evidence_requires_the_authorized_commit_as_a_parent(
    control_plane_repo: Path,
    integration_task,
    run_statusctl,
    read_status,
):
    """
    ER-07 for merges: an intervening commit breaks the binding.

    Authorizing a merge at A and then committing B before merging
    produces M whose first parent is B. A is still an ancestor of M,
    so containment would accept it — but the merge that happened is
    not the merge that was authorized.
    """

    authorized = commit_something(
        control_plane_repo,
        "seed.txt",
    )

    integration_task(
        "merge",
    )

    authorize(
        run_statusctl,
        read_status,
        "merge",
    )

    commit_something(
        control_plane_repo,
        "intervening.txt",
    )

    merged = merge_a_branch(
        control_plane_repo
    )

    assert git(
        control_plane_repo,
        "merge-base",
        "--is-ancestor",
        authorized,
        merged,
        check=False,
    ).returncode == 0

    result = run_statusctl(
        "integration",
        "performed",
        "merge",
    )

    assert result.returncode != 0

    entry = actions_by_name(
        read_status()
    )["merge"]

    assert entry["status"] == "AUTHORIZED"

    assert entry["evidence"] is None


def test_merge_action_accepts_a_merge_commit(
    control_plane_repo: Path,
    integration_task,
    run_statusctl,
    read_status,
):
    commit_something(
        control_plane_repo,
        "seed.txt",
    )

    integration_task(
        "merge",
    )

    authorize(
        run_statusctl,
        read_status,
        "merge",
    )

    merged = merge_a_branch(
        control_plane_repo
    )

    assert run_statusctl(
        "integration",
        "performed",
        "merge",
    ).returncode == 0

    entry = actions_by_name(
        read_status()
    )["merge"]

    assert entry["status"] == "PERFORMED"

    assert merged in entry["evidence"]


# ================================================================
# ER-01 — AN EMPTY SCOPE MUST BE SAID, NOT OMITTED
# ================================================================


def test_start_integration_without_a_declared_scope_is_refused(
    run_statusctl,
    read_status,
):
    """
    Silence is not a declaration.

    Omitting the scope is the accident this field exists to prevent,
    so it fails closed and the parent Work Task is left untouched.
    """

    complete_work_task(
        run_statusctl
    )

    result = run_statusctl(
        "start-integration",
        "integration-001",
        "Integrate the reviewed work",
    )

    assert result.returncode != 0

    status = read_status()

    assert (
        status["state"]
        == "READY_FOR_HUMAN_REVIEW"
    )

    assert status["task_id"] == "work-001"


def test_explicit_none_is_the_only_empty_scope(
    run_statusctl,
    read_status,
):
    complete_work_task(
        run_statusctl
    )

    assert start_integration(
        run_statusctl,
        "none",
    ).returncode == 0

    status = read_status()

    assert status["integration_actions"] == []

    assert status["task_kind"] == "INTEGRATION"


def test_none_cannot_be_combined_with_a_real_action(
    run_statusctl,
    read_status,
):
    complete_work_task(
        run_statusctl
    )

    assert start_integration(
        run_statusctl,
        "push",
        "none",
    ).returncode != 0

    assert (
        read_status()["state"]
        == "READY_FOR_HUMAN_REVIEW"
    )


# ================================================================
# ER-02 — INTEGRATION ACTIONS BELONG TO THE INTEGRATE PHASE
# ================================================================


@pytest.mark.parametrize(
    "phase",
    [
        "ANALYZE",
        "FINAL_VERIFY",
    ],
)
def test_git_gate_is_refused_outside_the_integrate_phase(
    phase: str,
    integration_task,
    run_statusctl,
    read_status,
    read_audit,
):
    """
    A commit gated during ANALYZE is a commit recorded as analysis.

    The phase is what says the task is at its integration boundary,
    so the gate is legal only there.
    """

    integration_task(
        "commit",
        phase=phase,
    )

    result = open_gate(
        run_statusctl,
        "commit",
    )

    assert result.returncode != 0

    status = read_status()

    assert status["state"] == "RUNNING"

    assert status["human_gate"] is None

    assert status["phase"] == phase

    assert (
        actions_by_name(
            status
        )["commit"]["status"]
        == "DECLARED"
    )

    assert (
        "HUMAN_GATE_OPENED"
        not in events(
            read_audit
        )
    )


def test_performed_is_refused_outside_the_integrate_phase(
    control_plane_repo: Path,
    integration_task,
    run_statusctl,
    read_status,
):
    """
    The record of an action belongs where the action does.
    """

    integration_task(
        "commit",
    )

    authorize(
        run_statusctl,
        read_status,
        "commit",
    )

    commit_something(
        control_plane_repo
    )

    assert run_statusctl(
        "phase",
        "FINAL_VERIFY",
    ).returncode == 0

    result = run_statusctl(
        "integration",
        "performed",
        "commit",
    )

    assert result.returncode != 0

    status = read_status()

    assert status["phase"] == "FINAL_VERIFY"

    assert (
        actions_by_name(
            status
        )["commit"]["status"]
        == "AUTHORIZED"
    )


def test_the_integrate_phase_permits_both_operations(
    control_plane_repo: Path,
    integration_task,
    run_statusctl,
    read_status,
):
    """
    The rule is a placement rule, not a prohibition.
    """

    integration_task(
        "commit",
    )

    assert open_gate(
        run_statusctl,
        "commit",
    ).returncode == 0

    assert approve(
        run_statusctl,
        read_status,
    ).returncode == 0

    commit_something(
        control_plane_repo
    )

    assert run_statusctl(
        "integration",
        "performed",
        "commit",
    ).returncode == 0

    assert (
        actions_by_name(
            read_status()
        )["commit"]["status"]
        == "PERFORMED"
    )


def test_performed_is_refused_for_an_undeclared_action(
    integration_task,
    run_statusctl,
):
    integration_task(
        "commit",
    )

    assert run_statusctl(
        "integration",
        "performed",
        "push",
    ).returncode != 0


def test_performed_is_refused_for_a_work_task(
    run_statusctl,
    read_status,
):
    assert run_statusctl(
        "start",
        "work-001",
        "Work task",
    ).returncode == 0

    assert run_statusctl(
        "integration",
        "performed",
        "commit",
    ).returncode != 0

    assert (
        read_status()["integration_actions"]
        is None
    )


# ================================================================
# COMPLETION
# ================================================================


def reach_final_verify(
    run_statusctl,
) -> None:
    for kind in (
        "targeted",
        "full",
    ):
        assert run_statusctl(
            "test-not-required",
            kind,
            "An Integration Task never re-tests product code.",
        ).returncode == 0

    assert run_statusctl(
        "phase",
        "FINAL_VERIFY",
    ).returncode == 0


def test_ready_is_refused_while_an_action_is_unresolved(
    control_plane_repo: Path,
    integration_task,
    run_statusctl,
    read_status,
    read_audit,
):
    """
    The whole point: a declared push that nobody authorized stops
    completion instead of being silently dropped.
    """

    integration_task(
        "commit",
        "push",
    )

    authorize(
        run_statusctl,
        read_status,
        "commit",
    )

    commit_something(
        control_plane_repo
    )

    assert run_statusctl(
        "integration",
        "performed",
        "commit",
    ).returncode == 0

    reach_final_verify(
        run_statusctl
    )

    result = run_statusctl(
        "ready",
    )

    assert result.returncode != 0

    assert "push" in (
        result.stderr
        + result.stdout
    )

    status = read_status()

    assert status["state"] == "RUNNING"

    assert status["completed_at"] is None

    # The parent Work Task completed legitimately, so the absence
    # being asserted is specifically the Integration Task's.
    assert not [
        event
        for event in read_audit()
        if event["event"] == "TASK_READY_FOR_HUMAN_REVIEW"
        and event.get("task_id") == "integration-001"
    ]


def test_ready_succeeds_when_every_action_is_resolved(
    control_plane_repo: Path,
    integration_task,
    run_statusctl,
    read_status,
    read_audit,
):
    """
    Performed or declined — both are resolutions, and both are
    recorded.
    """

    integration_task(
        "commit",
        "push",
    )

    authorize(
        run_statusctl,
        read_status,
        "commit",
    )

    commit_something(
        control_plane_repo
    )

    assert run_statusctl(
        "integration",
        "performed",
        "commit",
    ).returncode == 0

    assert open_gate(
        run_statusctl,
        "push",
    ).returncode == 0

    assert run_statusctl(
        "gate",
        "reject",
        active_gate_id(
            read_status
        ),
        "Push belongs to a later integration.",
    ).returncode == 0

    reach_final_verify(
        run_statusctl
    )

    assert run_statusctl(
        "ready",
    ).returncode == 0

    status = read_status()

    assert (
        status["state"]
        == "READY_FOR_HUMAN_REVIEW"
    )

    completion = [
        event
        for event in read_audit()
        if event["event"] == "TASK_READY_FOR_HUMAN_REVIEW"
    ][-1]

    assert completion["integration_actions"] == {
        "commit": "PERFORMED",
        "push": "DECLINED",
    }


def test_integration_task_declaring_nothing_completes(
    integration_task,
    run_statusctl,
    read_status,
):
    """
    Declaring no action is itself the recorded not-in-scope case: the
    audit shows the task claimed no git boundary at all.
    """

    integration_task()

    reach_final_verify(
        run_statusctl
    )

    assert run_statusctl(
        "ready",
    ).returncode == 0

    assert (
        read_status()["state"]
        == "READY_FOR_HUMAN_REVIEW"
    )


def test_ready_still_requires_final_verify(
    integration_task,
    run_statusctl,
):
    """
    The new rule is added to the completion contract, not swapped in
    for the existing ones.
    """

    integration_task()

    for kind in (
        "targeted",
        "full",
    ):
        assert run_statusctl(
            "test-not-required",
            kind,
            "An Integration Task never re-tests product code.",
        ).returncode == 0

    assert run_statusctl(
        "phase",
        "INTEGRATE",
    ).returncode == 0

    assert run_statusctl(
        "ready",
    ).returncode != 0


def test_gate_open_is_still_refused_after_ready(
    integration_task,
    run_statusctl,
    read_status,
):
    """
    Regression against the unsafe fix: the gap is closed at the front
    of the lifecycle, so nothing may reopen a completed task.
    """

    integration_task()

    reach_final_verify(
        run_statusctl
    )

    assert run_statusctl(
        "ready",
    ).returncode == 0

    result = open_gate(
        run_statusctl,
        "push",
    )

    assert result.returncode != 0

    assert (
        read_status()["state"]
        == "READY_FOR_HUMAN_REVIEW"
    )


def test_unresolved_action_in_a_ready_document_fails_validation(
    control_plane_repo: Path,
    integration_task,
    run_statusctl,
    read_status,
):
    """
    The rule lives in the validator, not only in the command, so a
    hand-written completion cannot smuggle an unresolved action past
    the load path.
    """

    integration_task(
        "push",
    )

    reach_final_verify(
        run_statusctl
    )

    assert run_statusctl(
        "ready",
    ).returncode != 0

    status_path = (
        control_plane_repo
        / ".claude"
        / "runtime"
        / "status.json"
    )

    forged = json.loads(
        status_path.read_text()
    )

    forged["state"] = "READY_FOR_HUMAN_REVIEW"

    forged["completed_at"] = forged["updated_at"]

    status_path.write_text(
        json.dumps(
            forged,
            indent=2,
        )
    )

    result = run_statusctl(
        "show",
    )

    assert result.returncode != 0

    assert "push" in (
        result.stderr
        + result.stdout
    )


# ================================================================
# PHASE MODEL
# ================================================================


def test_integrate_phase_is_legal_for_an_integration_task(
    integration_task,
    run_statusctl,
    read_status,
):
    integration_task(
        "commit",
    )

    assert run_statusctl(
        "phase",
        "INTEGRATE",
    ).returncode == 0

    assert (
        read_status()["phase"]
        == "INTEGRATE"
    )


def test_integrate_phase_is_illegal_for_a_work_task(
    run_statusctl,
    read_status,
):
    assert run_statusctl(
        "start",
        "work-001",
        "Work task",
    ).returncode == 0

    assert run_statusctl(
        "phase",
        "INTEGRATE",
    ).returncode != 0

    assert (
        read_status()["phase"]
        == "ANALYZE"
    )


# ================================================================
# MIGRATION 3.0 → 3.1
# ================================================================


def v30_document(
    *,
    task_kind: str = "WORK",
    parent_task_id: str | None = None,
    phase: str = "ANALYZE",
    state: str = "RUNNING",
) -> dict:
    """
    A valid schema-3.0 runtime document, as an operator would have.
    """

    return {
        "schema_version": PREVIOUS_SCHEMA,
        "project": "CryptoLab-DataHub",
        "state": state,
        "task_id": "legacy-001",
        "task_title": "A task recorded before schema 3.1",
        "task_kind": task_kind,
        "parent_task_id": parent_task_id,
        "phase": phase,
        "branch": "claude/system-autonomy-setup",
        "started_at": "2026-08-13T07:07:07.459899Z",
        "updated_at": "2026-08-13T07:07:07.459899Z",
        "completed_at": None,
        "tests": {
            "targeted": None,
            "full": None,
        },
        "human_gate": None,
        "last_gate_decision": None,
        "report": ".claude/runtime/latest_report.md",
    }


@pytest.fixture
def write_status(
    control_plane_repo: Path,
) -> Callable[[dict], Path]:
    def write(
        document: dict,
    ) -> Path:
        path = (
            control_plane_repo
            / ".claude"
            / "runtime"
            / "status.json"
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            json.dumps(
                document,
                indent=2,
            )
        )

        return path

    return write


def test_v30_runtime_is_refused_with_migration_instructions(
    run_statusctl,
    write_status,
):
    """
    A superseded runtime stops the controller instead of being
    silently upgraded by an unrelated command.
    """

    path = write_status(
        v30_document()
    )

    before = path.read_bytes()

    result = run_statusctl(
        "show",
    )

    assert result.returncode != 0

    assert "migrate-v31" in (
        result.stderr
        + result.stdout
    )

    assert path.read_bytes() == before


def test_normal_commands_never_auto_upgrade_a_v30_runtime(
    run_statusctl,
    write_status,
):
    path = write_status(
        v30_document()
    )

    before = path.read_bytes()

    for args in (
        (
            "phase",
            "IMPLEMENT",
        ),
        (
            "test",
            "targeted",
            "PASS",
            "pytest -q",
            "green",
        ),
        (
            "ready",
        ),
        (
            "report",
        ),
    ):
        assert run_statusctl(
            *args
        ).returncode != 0, args

        assert path.read_bytes() == before, args


def test_migrate_v31_upgrades_a_work_runtime(
    run_statusctl,
    write_status,
    read_status,
    read_audit,
):
    write_status(
        v30_document()
    )

    assert run_statusctl(
        "migrate-v31",
    ).returncode == 0

    status = read_status()

    assert (
        status["schema_version"]
        == CURRENT_SCHEMA
    )

    assert status["integration_actions"] is None

    # Every legacy runtime field survives the transform.
    assert status["task_id"] == "legacy-001"

    assert status["task_kind"] == "WORK"

    assert status["phase"] == "ANALYZE"

    assert status["started_at"] == (
        "2026-08-13T07:07:07.459899Z"
    )

    migrated = [
        event
        for event in read_audit()
        if event["event"] == "RUNTIME_SCHEMA_MIGRATED"
    ][-1]

    assert migrated["from_schema"] == PREVIOUS_SCHEMA

    assert migrated["to_schema"] == CURRENT_SCHEMA


def test_migrate_v31_declares_actions_for_an_integration_runtime(
    run_statusctl,
    write_status,
    read_status,
):
    """
    An Integration Task in flight keeps its safety property across the
    migration: the operator states the scope, rather than the
    migration inventing an empty one.
    """

    write_status(
        v30_document(
            task_kind="INTEGRATION",
            parent_task_id="work-001",
        )
    )

    assert run_statusctl(
        "migrate-v31",
        "--action",
        "push",
    ).returncode == 0

    status = read_status()

    assert [
        entry["action"]
        for entry in status["integration_actions"]
    ] == [
        "push",
    ]

    assert (
        status["integration_actions"][0]["status"]
        == "DECLARED"
    )


def test_migrate_v31_refuses_actions_for_a_work_runtime(
    run_statusctl,
    write_status,
):
    path = write_status(
        v30_document()
    )

    before = path.read_bytes()

    assert run_statusctl(
        "migrate-v31",
        "--action",
        "push",
    ).returncode != 0

    assert path.read_bytes() == before


def test_migrate_v31_refuses_a_runtime_that_is_not_v30(
    run_statusctl,
    write_status,
    read_audit,
):
    """
    A rejected migration changes nothing and records no success.
    """

    path = write_status(
        {
            **v30_document(),
            "schema_version": "2.0",
        }
    )

    before = path.read_bytes()

    assert run_statusctl(
        "migrate-v31",
    ).returncode != 0

    assert path.read_bytes() == before

    assert (
        "RUNTIME_SCHEMA_MIGRATED"
        not in events(
            read_audit
        )
    )


def test_migrate_v31_refuses_an_already_current_runtime(
    run_statusctl,
):
    assert run_statusctl(
        "start",
        "work-001",
        "Work task",
    ).returncode == 0

    assert run_statusctl(
        "migrate-v31",
    ).returncode != 0


def test_migrate_v31_refuses_an_invalid_v30_document(
    run_statusctl,
    write_status,
):
    """
    Migration validates its input rather than transforming whatever
    happens to carry the right version string.
    """

    path = write_status(
        {
            **v30_document(),
            "task_kind": "INTEGRATION",
            "parent_task_id": None,
        }
    )

    before = path.read_bytes()

    assert run_statusctl(
        "migrate-v31",
    ).returncode != 0

    assert path.read_bytes() == before
