from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
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
