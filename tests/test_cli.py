"""Exercise the installed package, not a source-path override."""

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest


def test_module_help_explains_development_scope(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "hermes_agent_sync", "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage: hermes-agent-sync" in result.stdout
    assert "Synchronization is not implemented yet." in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize("arguments", [[], ["--help"]])
def test_console_entry_point_does_not_touch_profile(arguments: list[str], tmp_path: Path) -> None:
    suffix = ".exe" if os.name == "nt" else ""
    executable = Path(sysconfig.get_path("scripts")) / f"hermes-agent-sync{suffix}"
    assert executable.is_file(), "Install the project before running tests"
    profile = tmp_path / "profile"
    profile.mkdir()
    sentinel = profile / "SOUL.md"
    sentinel.write_bytes(b"Keep this exact wording.\n")
    before = sorted(p.relative_to(profile) for p in profile.rglob("*"))
    result = subprocess.run(
        [executable, *arguments],
        cwd=tmp_path,
        env={**os.environ, "HERMES_HOME": str(profile)},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Synchronization is not implemented yet." in result.stdout
    assert result.stderr == ""
    assert sentinel.read_bytes() == b"Keep this exact wording.\n"
    assert sorted(p.relative_to(profile) for p in profile.rglob("*")) == before


def test_unknown_sync_command_fails_without_claiming_success(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "hermes_agent_sync", "sync"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "unrecognized arguments: sync" in result.stderr
