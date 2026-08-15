from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Callable, Generator

import pytest


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

STATUSCTL_SOURCE = (
    PROJECT_ROOT
    / ".claude"
    / "bin"
    / "statusctl"
)


# ================================================================
# CONTROL-PLANE SOURCE RESOLUTION
# ================================================================
#
# A Claude session cannot write `.claude/`, so a control-plane change
# is staged under var/ and installed later by a human running
# var/hardening-4.3/install. That leaves two coherent worlds — before
# the bootstrap and after it — and a test that guessed between them
# would exercise a control plane that is half one and half the other.
#
# So the choice is made once, per file, from evidence: if a pending
# manifest declares a hash the installed file does not have, the
# delivery has not landed and the staged copy is the version under
# test. Once it has landed, or once the staging area is removed, the
# installed file is the version under test. Either way exactly one
# version is exercised, and no assertion is relaxed to accommodate the
# other.

STAGING_DIR = (
    PROJECT_ROOT
    / "var"
    / "hardening-4.3"
)

STAGING_MANIFEST = STAGING_DIR / "manifest.json"


def _pending_delivery() -> dict[str, Path]:
    """
    Staged replacements for destinations that are not installed yet.

    Keyed by repository-relative destination. A destination whose
    installed bytes already match the manifest is absent from the
    mapping: it has been delivered, and the installed file is
    canonical.
    """

    if not STAGING_MANIFEST.is_file():
        return {}

    try:
        manifest = json.loads(
            STAGING_MANIFEST.read_text()
        )

    except (OSError, json.JSONDecodeError):
        return {}

    pending = {}

    for entry in manifest.get("files") or []:
        destination = str(
            entry.get("destination") or ""
        )

        source = STAGING_DIR / str(
            entry.get("source") or ""
        )

        expected = str(
            entry.get("sha256") or ""
        )

        if not destination or not source.is_file():
            continue

        installed = PROJECT_ROOT / destination

        if installed.is_file():
            found = hashlib.sha256(
                installed.read_bytes()
            ).hexdigest()

            if found == expected:
                continue

        pending[destination] = source

    return pending


PENDING_DELIVERY = _pending_delivery()


def control_plane_source(
    destination: str,
) -> Path:
    """
    The file a test must exercise for one control-plane destination.
    """

    return PENDING_DELIVERY.get(
        destination,
        PROJECT_ROOT / destination,
    )


AUTOPILOT_SOURCE = control_plane_source(
    ".claude/bin/autopilot"
)

SUPERVISOR_SOURCE = control_plane_source(
    ".claude/bin/supervisor"
)

REVIEWER_SOURCE = control_plane_source(
    ".claude/bin/openai_reviewer"
)

CONTRACT_SOURCE = control_plane_source(
    "CLAUDE.md"
)


def statusctl_lock_path(
    root: Path,
) -> Path:
    key = hashlib.sha256(
        str(
            root.resolve()
        ).encode("utf-8")
    ).hexdigest()[:16]

    return (
        Path(
            tempfile.gettempdir()
        )
        / f"cryptolab-statusctl-{key}.lock"
    )


@pytest.fixture
def control_plane_repo(
    tmp_path: Path,
) -> Generator[
    Path,
    None,
    None,
]:
    """
    Create an isolated temporary repository containing
    the current statusctl implementation.

    Tests must exercise this copy rather than the real
    project runtime.

    The statusctl lock lives outside the temporary
    repository, so the fixture explicitly removes only
    the lock belonging to this repository during
    teardown.
    """

    root = (
        tmp_path
        / "control-plane-repo"
    )

    bin_dir = (
        root
        / ".claude"
        / "bin"
    )

    bin_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = (
        bin_dir
        / "statusctl"
    )

    shutil.copy2(
        STATUSCTL_SOURCE,
        destination,
    )

    destination.chmod(
        0o755
    )

    subprocess.run(
        [
            "git",
            "init",
            "-q",
        ],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )

    lock_path = statusctl_lock_path(
        root
    )

    try:
        yield root

    finally:
        lock_path.unlink(
            missing_ok=True
        )


@pytest.fixture
def statusctl_path(
    control_plane_repo: Path,
) -> Path:
    return (
        control_plane_repo
        / ".claude"
        / "bin"
        / "statusctl"
    )


@pytest.fixture
def run_statusctl(
    control_plane_repo: Path,
    statusctl_path: Path,
) -> Callable[..., subprocess.CompletedProcess[str]]:
    """
    Return a CLI runner bound to the isolated repository.

    check=False is intentional. Many regression tests
    assert that invalid transitions fail with a non-zero
    exit code.
    """

    def run(
        *args: str,
        timeout: float = 15.0,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(statusctl_path),
                *args,
            ],
            cwd=control_plane_repo,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )

    return run


@pytest.fixture
def runtime_dir(
    control_plane_repo: Path,
) -> Path:
    return (
        control_plane_repo
        / ".claude"
        / "runtime"
    )


@pytest.fixture
def read_status(
    runtime_dir: Path,
) -> Callable[[], dict]:
    def read() -> dict:
        return json.loads(
            (
                runtime_dir
                / "status.json"
            ).read_text()
        )

    return read


@pytest.fixture
def read_audit(
    runtime_dir: Path,
) -> Callable[[], list[dict]]:
    def read() -> list[dict]:
        path = (
            runtime_dir
            / "audit.jsonl"
        )

        if not path.exists():
            return []

        return [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]

    return read


@pytest.fixture
def load_statusctl_module(
    statusctl_path: Path,
):
    """
    Import the isolated statusctl copy without invoking
    its CLI main(), allowing direct validator/save tests.

    A unique module name prevents module-cache leakage
    between pytest cases.
    """

    def load():
        module_name = (
            "statusctl_test_"
            + uuid.uuid4().hex
        )

        loader = SourceFileLoader(
            module_name,
            str(statusctl_path),
        )

        spec = (
            importlib.util.spec_from_loader(
                module_name,
                loader,
            )
        )

        if spec is None:
            raise RuntimeError(
                "Could not create statusctl module spec."
            )

        module = (
            importlib.util.module_from_spec(
                spec
            )
        )

        loader.exec_module(
            module
        )

        return module

    return load


# ================================================================
# ORCHESTRATOR CONTROL PLANE
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
def supervisor_repo(
    control_plane_repo: Path,
) -> Path:
    """
    An isolated repository holding the whole control plane.

    The supervisor resolves autopilot and the reviewer from its own
    directory and reads the contract from the repository root, so all
    four are installed here. Nothing in this fixture touches the real
    project.
    """

    bin_dir = (
        control_plane_repo
        / ".claude"
        / "bin"
    )

    for name, source in (
        ("autopilot", AUTOPILOT_SOURCE),
        ("supervisor", SUPERVISOR_SOURCE),
        ("openai_reviewer", REVIEWER_SOURCE),
    ):
        destination = bin_dir / name

        shutil.copy2(
            source,
            destination,
        )

        destination.chmod(0o755)

    shutil.copy2(
        CONTRACT_SOURCE,
        control_plane_repo / "CLAUDE.md",
    )

    return control_plane_repo


def load_module_from(
    path: Path,
    prefix: str,
):
    """
    Import an extensionless control-plane program by location.

    A unique module name per load keeps one test's module state out of
    the next one's.
    """

    name = (
        f"{prefix}_test_"
        + uuid.uuid4().hex
    )

    loader = SourceFileLoader(
        name,
        str(path),
    )

    spec = importlib.util.spec_from_loader(
        name,
        loader,
    )

    if spec is None:
        raise RuntimeError(
            f"Could not create a module spec for {path}."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    # No bytecode written beside the source. Until the delivery lands,
    # some of these programs are loaded straight out of the staging
    # payload, and the installer refuses a payload holding any file its
    # manifest does not declare. A __pycache__ left here would make the
    # suite break the delivery it exists to protect, at the moment a
    # human ran the bootstrap.
    previous = sys.dont_write_bytecode

    sys.dont_write_bytecode = True

    try:
        loader.exec_module(
            module
        )

    finally:
        sys.dont_write_bytecode = previous

    return module


@pytest.fixture
def load_supervisor_module(
    supervisor_repo: Path,
):
    def load():
        return load_module_from(
            supervisor_repo
            / ".claude"
            / "bin"
            / "supervisor",
            "supervisor",
        )

    return load


@pytest.fixture
def load_reviewer_module():
    """
    The reviewer bridge, loaded from the version under test.

    It imports nothing from the control plane it lives in, so no
    isolated repository is needed to exercise it.
    """

    def load():
        return load_module_from(
            REVIEWER_SOURCE,
            "openai_reviewer",
        )

    return load


@pytest.fixture
def load_installed_module():
    """
    A control-plane program as installed, whatever is staged.

    Every other loader resolves the version under test, which is the
    right answer for a test of behaviour and the wrong one for a test
    of deployment. `.claude/bin/supervisor` loads its siblings by
    location, so the installed file is what a live run executes — and
    a suite that only ever exercised the source copy is exactly how a
    corrected schema and a green suite coexisted with a live call
    still sending the old one.
    """

    def load(
        destination: str,
        prefix: str,
    ):
        path = PROJECT_ROOT / destination

        if not path.is_file():
            raise AssertionError(
                f"{destination} is not installed."
            )

        return load_module_from(
            path,
            prefix,
        )

    return load


def staged_manifest() -> dict:
    """
    The staged delivery description, or an empty one.
    """

    if not STAGING_MANIFEST.is_file():
        return {}

    try:
        return json.loads(
            STAGING_MANIFEST.read_text()
        )

    except (OSError, json.JSONDecodeError):
        return {}


def installed_hash(
    destination: str,
) -> str | None:
    path = PROJECT_ROOT / destination

    if not path.is_file():
        return None

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
