"""
Hardening 4.3 — one-shot bootstrap installer contract.

A Claude session cannot write the control plane that constrains it.
That is the point of the sandbox, so the protected files this slice
changes were staged instead, and a human installs them with one
command after the delivery gate is approved.

An installer with write access to the control plane is exactly the
thing that must not be trusted loosely, so these tests pin what it is
allowed to be:

    I-01  its scope is compiled into the program, and a manifest
          cannot widen it, shrink it, or belong to other work;
    I-02  every staged hash is verified before anything is mutated,
          and every destination is verified again afterwards;
    I-03  nothing is installed by wildcard, and no path may leave the
          tree it was declared in;
    I-04  it proves the incoming settings remove no existing
          protection and the incoming contract drops no section;
    I-05  it never writes runtime state and never lands on a
          protected branch;
    I-06  a failure part-way through restores what it replaced.

Every test runs against a synthetic repository and a synthetic staging
area, except the first two, which read the real staged delivery and
assert only that it is internally consistent.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import uuid
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Callable

import pytest


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

STAGING_DIR = (
    PROJECT_ROOT
    / "var"
    / "hardening-4.3"
)

INSTALLER = STAGING_DIR / "install"

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_FAILED_ROLLED_BACK = 2


pytestmark = pytest.mark.skipif(
    not INSTALLER.is_file(),
    reason=(
        "The staging area is removed once the delivery has landed; "
        "there is then no installer to exercise."
    ),
)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


@pytest.fixture
def installer():
    """
    The installer, loaded as a module for direct unit assertions.
    """

    name = "install_test_" + uuid.uuid4().hex

    loader = SourceFileLoader(
        name,
        str(INSTALLER),
    )

    spec = importlib.util.spec_from_loader(
        name,
        loader,
    )

    module = importlib.util.module_from_spec(spec)

    # Loading the installer must not litter the staging area it
    # validates. The installer refuses a payload holding an undeclared
    # file, so bytecode written here would be the suite breaking the
    # delivery rather than checking it.
    previous = sys.dont_write_bytecode

    sys.dont_write_bytecode = True

    try:
        loader.exec_module(module)

    finally:
        sys.dont_write_bytecode = previous

    return module


# ================================================================
# THE REAL STAGED DELIVERY
# ================================================================


def test_the_staged_manifest_matches_the_staged_payload(
    installer,
):
    """
    The staging area has not drifted from the hashes it declares.

    An installer that refused at the moment a human ran it would be a
    delivery discovered broken at the worst time, so the same check
    runs here, whenever the suite runs.
    """

    manifest = json.loads(
        (STAGING_DIR / "manifest.json").read_text()
    )

    for entry in manifest["files"]:
        source = STAGING_DIR / entry["source"]

        assert source.is_file(), (
            f"staged source missing: {entry['source']}"
        )

        assert sha256_of(source) == entry["sha256"], (
            f"{entry['destination']} was changed after the manifest "
            "was written; regenerate the manifest."
        )


def test_the_staged_payload_holds_nothing_the_manifest_omits(
    installer,
):
    """
    The real payload holds the five declared files and nothing else.

    The installer refuses an undeclared file in the payload, which is
    the right answer to an undeclared delivery but a poor way to find
    out that something merely wrote a scratch file there. Running a
    test that imports a staged program is enough to do it — so the same
    check runs against the real staging area here, where the cost of
    failing is a red test rather than a refused bootstrap.
    """

    manifest = json.loads(
        (STAGING_DIR / "manifest.json").read_text()
    )

    declared = {
        (STAGING_DIR / entry["source"]).resolve()
        for entry in manifest["files"]
    }

    payload = STAGING_DIR / "payload"

    stray = sorted(
        str(path.relative_to(payload))
        for path in payload.rglob("*")
        if path.is_file()
        and path.resolve() not in declared
    )

    assert not stray, (
        "The staged payload holds files the manifest does not "
        f"declare: {', '.join(stray)}. The installer would refuse "
        "this delivery."
    )


def test_the_staged_manifest_declares_exactly_the_authorized_scope(
    installer,
):
    """
    The manifest and the compiled scope agree, and both are the gate's.

    The scope lives in the program so a substituted manifest cannot
    widen it; this asserts the staged manifest has not quietly
    diverged from it either.
    """

    manifest = json.loads(
        (STAGING_DIR / "manifest.json").read_text()
    )

    assert (
        manifest["authorized_destinations"]
        == list(installer.AUTHORIZED_DESTINATIONS)
    )

    assert {
        entry["destination"]
        for entry in manifest["files"]
    } == set(installer.AUTHORIZED_DESTINATIONS)

    for field, expected in installer.IDENTITY.items():
        assert manifest[field] == expected

    # The write scope HG-20260814-007 authorized, and nothing else.
    assert set(installer.AUTHORIZED_DESTINATIONS) == {
        ".claude/bin/supervisor",
        ".claude/bin/openai_reviewer",
        ".claude/bin/autopilot",
        ".claude/settings.json",
        "CLAUDE.md",
    }


# ================================================================
# SYNTHETIC DELIVERY
# ================================================================


EXISTING_SETTINGS = {
    "permissions": {
        "disableBypassPermissionsMode": "disable",
        "allow": ["Bash(git status *)"],
        "ask": ["Bash(git commit *)"],
        "deny": ["Bash(sudo *)"],
    },
    "sandbox": {
        "enabled": True,
        "failIfUnavailable": True,
        "allowUnsandboxedCommands": False,
        "filesystem": {
            "denyWrite": ["./.claude"],
            "denyRead": ["./.env"],
        },
    },
}

# Additive only: one new deny rule, nothing removed.
INCOMING_SETTINGS = json.loads(
    json.dumps(EXISTING_SETTINGS)
)

INCOMING_SETTINGS["permissions"]["deny"].append(
    "Bash(./.claude/bin/supervisor *)"
)

EXISTING_CONTRACT = (
    "# Contract\n\n"
    "## 1. Role\n\nText.\n\n"
    "## 22. Policy protection\n\nText.\n"
)

INCOMING_CONTRACT = (
    EXISTING_CONTRACT
    + "\n## 23A. Unattended orchestration\n\nText.\n"
)


@pytest.fixture
def root(
    tmp_path: Path,
) -> Path:
    """
    A synthetic repository holding what a REPLACE entry needs.

    The two CREATE destinations are deliberately absent: the installer
    refuses to create over something that already exists, and that is
    a property worth being able to observe.
    """

    root = tmp_path / "repo"

    (root / ".claude" / "bin").mkdir(
        parents=True,
    )

    subprocess.run(
        [
            "git",
            "init",
            "-q",
            "-b",
            "claude/delivery",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    (root / ".claude" / "bin" / "autopilot").write_text(
        "#!/usr/bin/env python3\n# old autopilot\n"
    )

    (root / ".claude" / "settings.json").write_text(
        json.dumps(EXISTING_SETTINGS, indent=2) + "\n"
    )

    (root / "CLAUDE.md").write_text(
        EXISTING_CONTRACT
    )

    # One commit, so the branch the installer refuses to land on is a
    # question `git rev-parse` can actually answer.
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=tests@example.invalid",
            "-c",
            "user.name=Control Plane Tests",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "baseline",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    return root


PAYLOAD_BY_DESTINATION = {
    ".claude/bin/supervisor": (
        "payload/claude/bin/supervisor",
        "#!/usr/bin/env python3\n# supervisor\n",
        "0755",
        "CREATE",
    ),
    ".claude/bin/openai_reviewer": (
        "payload/claude/bin/openai_reviewer",
        "#!/usr/bin/env python3\n# reviewer\n",
        "0755",
        "CREATE",
    ),
    ".claude/bin/autopilot": (
        "payload/claude/bin/autopilot",
        "#!/usr/bin/env python3\n# new autopilot\n",
        "0755",
        "REPLACE",
    ),
    ".claude/settings.json": (
        "payload/claude/settings.json",
        json.dumps(INCOMING_SETTINGS, indent=2) + "\n",
        "0644",
        "REPLACE",
    ),
    "CLAUDE.md": (
        "payload/contract.md",
        INCOMING_CONTRACT,
        "0644",
        "REPLACE",
    ),
}


@pytest.fixture
def staging(
    tmp_path: Path,
    installer,
) -> Callable[..., Path]:
    """
    Build a synthetic staging area, optionally deformed.

    Every refusal test is one small deformation of a delivery that
    would otherwise install cleanly, so the thing under test is the
    deformation rather than the setup.
    """

    def build(
        *,
        contents: dict[str, str] | None = None,
        hashes: dict[str, str] | None = None,
        destinations: dict[str, str] | None = None,
        drop: str | None = None,
        extra_payload: str | None = None,
        manifest_overrides: dict | None = None,
    ) -> Path:
        staging = tmp_path / "staging"

        files = []

        for destination, (
            source,
            body,
            mode,
            disposition,
        ) in PAYLOAD_BY_DESTINATION.items():
            if drop == destination:
                continue

            body = (contents or {}).get(
                destination,
                body,
            )

            path = staging / source

            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            path.write_text(body)

            files.append(
                {
                    "source": source,
                    "destination": (
                        destinations or {}
                    ).get(
                        destination,
                        destination,
                    ),
                    "sha256": (hashes or {}).get(
                        destination,
                        sha256_of(path),
                    ),
                    "mode": mode,
                    "disposition": disposition,
                }
            )

        if extra_payload is not None:
            stray = staging / "payload" / extra_payload

            stray.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            stray.write_text("undeclared\n")

        manifest = {
            "manifest_version": installer.MANIFEST_VERSION,
            **installer.IDENTITY,
            "authorized_destinations": list(
                installer.AUTHORIZED_DESTINATIONS
            ),
            "files": files,
        }

        manifest.update(manifest_overrides or {})

        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )

        return staging

    return build


@pytest.fixture
def install(
    root: Path,
) -> Callable[..., subprocess.CompletedProcess[str]]:
    def run(
        staging: Path,
        *args: str,
        repository: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(INSTALLER),
                "--root",
                str(repository or root),
                "--staging",
                str(staging),
                *args,
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=120.0,
        )

    return run


def snapshot(
    root: Path,
) -> dict[str, tuple[str, str]]:
    return {
        str(path.relative_to(root)): (
            sha256_of(path),
            oct(path.stat().st_mode & 0o777),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and ".git/" not in str(path.relative_to(root))
    }


# ================================================================
# THE HAPPY PATH
# ================================================================


def test_a_dry_run_verifies_everything_and_changes_nothing(
    root: Path,
    staging,
    install,
):
    before = snapshot(root)

    completed = install(
        staging(),
        "--dry-run",
    )

    assert completed.returncode == EXIT_OK, completed.stdout

    assert "DRY RUN" in completed.stdout

    assert snapshot(root) == before

    # The CREATE destinations were not created by a verification.
    assert not (
        root / ".claude" / "bin" / "supervisor"
    ).exists()


def test_installing_lands_every_file_with_its_declared_mode(
    root: Path,
    staging,
    install,
):
    area = staging()

    completed = install(area)

    assert completed.returncode == EXIT_OK, (
        completed.stdout + completed.stderr
    )

    manifest = json.loads(
        (area / "manifest.json").read_text()
    )

    for entry in manifest["files"]:
        target = root / entry["destination"]

        assert target.is_file()

        assert sha256_of(target) == entry["sha256"]

        assert (
            oct(target.stat().st_mode & 0o777)
            == oct(int(entry["mode"], 8))
        )

    # Backups of everything that was replaced, and a record naming
    # the delivery they belong to.
    backups = sorted(
        (area / "backup").iterdir()
    )

    assert len(backups) == 1

    record = json.loads(
        (backups[0] / "backup.json").read_text()
    )

    assert record["task_id"] == "unattended-supervisor-reviewer-001"

    replaced = {
        entry["destination"]
        for entry in record["records"]
        if entry["existed"]
    }

    assert replaced == {
        ".claude/bin/autopilot",
        ".claude/settings.json",
        "CLAUDE.md",
    }


def test_the_installer_is_one_shot(
    root: Path,
    staging,
    install,
):
    """
    A second run is refused rather than being a silent no-op.

    The CREATE destinations exist afterwards, and that either means
    the delivery already ran or the files are not what the manifest
    describes. Both are worth stopping for.
    """

    area = staging()

    assert install(area).returncode == EXIT_OK

    before = snapshot(root)

    completed = install(area)

    assert completed.returncode == EXIT_REFUSED

    assert "already" in completed.stdout

    assert snapshot(root) == before


def test_runtime_state_is_never_touched(
    root: Path,
    staging,
    install,
):
    """
    statusctl is the only writer of runtime state.

    The installer has write access to `.claude`, so this is asserted
    rather than assumed.
    """

    runtime = root / ".claude" / "runtime"

    runtime.mkdir(parents=True)

    (runtime / "status.json").write_text('{"state": "RUNNING"}\n')

    before = snapshot(runtime)

    assert install(staging()).returncode == EXIT_OK

    assert snapshot(runtime) == before


def test_a_runtime_destination_is_refused_outright(
    installer,
    root: Path,
):
    """
    The defence that does not depend on the scope list.

    A destination under `.claude/runtime` is already outside the
    authorized set, so this can only be reached by a program change —
    which is exactly when a second, independent refusal is worth
    having.
    """

    with pytest.raises(
        installer.Refused,
        match="never writes runtime state",
    ):
        installer.assert_destinations_installable(
            [
                {
                    "destination": (
                        ".claude/runtime/status.json"
                    ),
                    "target": (
                        root
                        / ".claude"
                        / "runtime"
                        / "status.json"
                    ),
                    "disposition": "REPLACE",
                },
            ]
        )


# ================================================================
# SCOPE IS COMPILED IN
# ================================================================


def test_a_manifest_may_not_widen_the_scope(
    root: Path,
    staging,
    install,
):
    before = snapshot(root)

    completed = install(
        staging(
            destinations={
                "CLAUDE.md": ".claude/bin/statusctl",
            },
        )
    )

    assert completed.returncode == EXIT_REFUSED

    assert (
        "not in the authorized write scope"
        in completed.stdout
    )

    assert snapshot(root) == before


def test_a_partial_delivery_is_refused(
    root: Path,
    staging,
    install,
):
    """
    A missing destination is a delivery someone forgot to finish.

    Installing four of five files would leave a control plane that
    matches neither the old state nor the new one.
    """

    completed = install(
        staging(
            drop=".claude/settings.json",
        )
    )

    assert completed.returncode == EXIT_REFUSED

    assert "missing authorized destinations" in completed.stdout


def test_a_manifest_belonging_to_other_work_is_refused(
    root: Path,
    staging,
    install,
):
    completed = install(
        staging(
            manifest_overrides={
                "scope_gate_id": "HG-20260101-001",
            },
        )
    )

    assert completed.returncode == EXIT_REFUSED

    assert "belonging to other work" in completed.stdout


def test_an_unknown_manifest_version_is_refused(
    root: Path,
    staging,
    install,
):
    completed = install(
        staging(
            manifest_overrides={
                "manifest_version": "2",
            },
        )
    )

    assert completed.returncode == EXIT_REFUSED

    assert "understands only" in completed.stdout


def test_a_substituted_authorized_list_is_refused(
    root: Path,
    staging,
    install,
):
    """
    The scope in the data may not disagree with the scope in the code.

    A manifest that declared a different authorized set would be
    describing a delivery this program is not the installer for.
    """

    completed = install(
        staging(
            manifest_overrides={
                "authorized_destinations": [
                    ".claude/bin/supervisor",
                ],
            },
        )
    )

    assert completed.returncode == EXIT_REFUSED

    assert (
        "does not match the scope compiled into this "
        "installer" in completed.stdout
    )


def test_nothing_is_installed_by_wildcard(
    root: Path,
    staging,
    install,
):
    """
    An undeclared file in the payload stops the delivery.

    It is either something someone forgot to declare or something
    that should not be there, and both are worth catching while
    nothing has been written.
    """

    before = snapshot(root)

    completed = install(
        staging(
            extra_payload="claude/bin/extra_tool",
        )
    )

    assert completed.returncode == EXIT_REFUSED

    assert "does not declare" in completed.stdout

    assert snapshot(root) == before


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "../outside",
    ],
)
def test_paths_may_not_leave_their_tree(
    installer,
    path: str,
):
    with pytest.raises(
        installer.Refused,
    ):
        installer.safe_relative(
            path,
            "destination",
        )


# ================================================================
# HASHES DECIDE
# ================================================================


def test_a_staged_file_changed_after_the_manifest_is_refused(
    root: Path,
    staging,
    install,
):
    """
    The manifest is the authorization; the bytes must match it.

    A staged file edited after the hash was recorded is no longer the
    file a human approved.
    """

    before = snapshot(root)

    completed = install(
        staging(
            hashes={
                ".claude/bin/supervisor": "0" * 64,
            },
        )
    )

    assert completed.returncode == EXIT_REFUSED

    assert "does not match the manifest" in completed.stdout

    assert snapshot(root) == before


def test_a_malformed_hash_is_refused(
    root: Path,
    staging,
    install,
):
    completed = install(
        staging(
            hashes={
                "CLAUDE.md": "not-a-hash",
            },
        )
    )

    assert completed.returncode == EXIT_REFUSED

    assert "no valid sha256" in completed.stdout


# ================================================================
# NON-WEAKENING PROOFS
# ================================================================


def test_settings_that_drop_a_rule_are_refused(
    root: Path,
    staging,
    install,
):
    """
    The delivery may add a permission; it may not remove one.

    This is checked against the file on disk rather than against what
    a diff was believed to say.
    """

    weakened = json.loads(
        json.dumps(INCOMING_SETTINGS)
    )

    weakened["permissions"]["deny"].remove(
        "Bash(sudo *)"
    )

    completed = install(
        staging(
            contents={
                ".claude/settings.json": json.dumps(
                    weakened,
                    indent=2,
                )
                + "\n",
            },
        )
    )

    assert completed.returncode == EXIT_REFUSED

    assert "may not remove one" in completed.stdout


@pytest.mark.parametrize(
    "flag, value",
    [
        ("enabled", False),
        ("failIfUnavailable", False),
        ("allowUnsandboxedCommands", True),
    ],
)
def test_settings_that_weaken_the_sandbox_are_refused(
    root: Path,
    staging,
    install,
    flag: str,
    value: bool,
):
    weakened = json.loads(
        json.dumps(INCOMING_SETTINGS)
    )

    weakened["sandbox"][flag] = value

    completed = install(
        staging(
            contents={
                ".claude/settings.json": json.dumps(
                    weakened,
                    indent=2,
                )
                + "\n",
            },
        )
    )

    assert completed.returncode == EXIT_REFUSED

    assert (
        "Sandbox protection may not be weakened"
        in completed.stdout
    )


def test_settings_that_re_enable_permission_bypass_are_refused(
    root: Path,
    staging,
    install,
):
    weakened = json.loads(
        json.dumps(INCOMING_SETTINGS)
    )

    weakened["permissions"][
        "disableBypassPermissionsMode"
    ] = "allow"

    completed = install(
        staging(
            contents={
                ".claude/settings.json": json.dumps(
                    weakened,
                    indent=2,
                )
                + "\n",
            },
        )
    )

    assert completed.returncode == EXIT_REFUSED

    assert "permission-bypass" in completed.stdout


def test_settings_that_drop_a_sandbox_deny_path_are_refused(
    root: Path,
    staging,
    install,
):
    weakened = json.loads(
        json.dumps(INCOMING_SETTINGS)
    )

    weakened["sandbox"]["filesystem"]["denyWrite"] = []

    completed = install(
        staging(
            contents={
                ".claude/settings.json": json.dumps(
                    weakened,
                    indent=2,
                )
                + "\n",
            },
        )
    )

    assert completed.returncode == EXIT_REFUSED

    assert "drop sandbox denyWrite" in completed.stdout


def test_a_contract_that_drops_a_section_is_refused(
    root: Path,
    staging,
    install,
):
    """
    This delivery documents the supervisor; it does not remove policy.
    """

    completed = install(
        staging(
            contents={
                "CLAUDE.md": (
                    "# Contract\n\n"
                    "## 1. Role\n\nText.\n\n"
                    "## 23A. Unattended orchestration\n\nText.\n"
                ),
            },
        )
    )

    assert completed.returncode == EXIT_REFUSED

    assert "drops sections that exist today" in completed.stdout

    assert "## 22. Policy protection" in completed.stdout


# ================================================================
# BRANCH SAFETY
# ================================================================


@pytest.mark.parametrize(
    "branch",
    [
        "main",
        "develop",
    ],
)
def test_a_protected_branch_is_refused(
    root: Path,
    staging,
    install,
    branch: str,
):
    """
    Control-plane work belongs on the Claude worktree branch.

    The installer runs no mutating Git command, so this is not about
    Git safety in itself: it is about not landing a control-plane
    change where the contract says it may not land.
    """

    subprocess.run(
        [
            "git",
            "checkout",
            "-q",
            "-b",
            branch,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    before = snapshot(root)

    completed = install(staging())

    assert completed.returncode == EXIT_REFUSED

    assert f"Refusing to install onto {branch}" in completed.stdout

    assert snapshot(root) == before


# ================================================================
# ROLLBACK
# ================================================================


def test_a_failure_part_way_through_restores_what_it_replaced(
    root: Path,
    staging,
    install,
):
    """
    The control plane is never left half-delivered.

    The last destination is made unwritable after every precondition
    has already passed, which is the only way to reach the failure
    path the backups exist for.
    """

    area = staging()

    before = snapshot(root)

    original_mode = root.stat().st_mode

    # CLAUDE.md installs last and lives at the repository root, so
    # making the root unwritable fails that one write and no earlier
    # one.
    os.chmod(root, 0o555)

    try:
        completed = install(area)

    finally:
        os.chmod(root, original_mode)

    assert completed.returncode == EXIT_FAILED_ROLLED_BACK, (
        completed.stdout
    )

    assert "Restoring backups" in completed.stdout
    assert "Rolled back" in completed.stdout
    assert "ROLLBACK INCOMPLETE" not in completed.stdout

    # Replaced files are back, and created files are gone.
    assert snapshot(root) == before

    assert not (
        root / ".claude" / "bin" / "supervisor"
    ).exists()

    assert not (
        root / ".claude" / "bin" / "openai_reviewer"
    ).exists()


def test_a_replace_of_a_missing_file_is_refused(
    root: Path,
    staging,
    install,
):
    (root / "CLAUDE.md").unlink()

    completed = install(staging())

    assert completed.returncode == EXIT_REFUSED

    assert "declared REPLACE but does not exist" in completed.stdout


def test_a_symlinked_destination_is_refused(
    root: Path,
    staging,
    install,
):
    """
    Replacing a symlink writes through it.

    That is a way to reach a file the authorized scope never named,
    so a symlink anywhere in a destination path stops the delivery.

    The link points inside the repository on purpose: a link out of
    the tree is already refused for resolving outside the root, and
    that would test the containment check rather than this one.
    """

    linked = root / "real_contract.md"

    linked.write_text(EXISTING_CONTRACT)

    contract = root / "CLAUDE.md"

    contract.unlink()

    contract.symlink_to(linked)

    completed = install(staging())

    assert completed.returncode == EXIT_REFUSED

    assert "passes through a symlink" in completed.stdout

    # The file the symlink pointed at was not written.
    assert linked.read_text() == EXISTING_CONTRACT


def test_a_symlinked_staged_source_is_refused(
    root: Path,
    staging,
    install,
):
    """
    The same rule on the way in.

    A staged source that is a link reads bytes from somewhere the
    staging area does not contain, which is the same substitution as
    a linked destination, pointed the other way.
    """

    area = staging()

    source = area / "payload" / "claude" / "bin" / "supervisor"

    real = area / "payload" / "claude" / "bin" / "real_supervisor"

    source.rename(real)

    source.symlink_to(real)

    before = snapshot(root)

    completed = install(area)

    assert completed.returncode == EXIT_REFUSED

    assert "passes through a symlink" in completed.stdout

    assert snapshot(root) == before
