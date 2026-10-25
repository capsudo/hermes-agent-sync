"""Contract checks for disposable test homes and real file-only Git transport."""

import os
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest


def test_environment_uses_only_disposable_paths(isolated_home: Path) -> None:
    assert Path.home() == isolated_home
    assert Path(os.environ["HERMES_HOME"]).is_relative_to(isolated_home)
    assert Path(os.environ["XDG_CONFIG_HOME"]).is_relative_to(isolated_home)
    assert Path(os.environ["GIT_CONFIG_GLOBAL"]).is_relative_to(isolated_home)
    assert os.environ["GIT_CONFIG_NOSYSTEM"] == "1"
    assert "SSH_AUTH_SOCK" not in os.environ


def test_profiles_do_not_share_selected_files(fake_profiles: dict[str, Path]) -> None:
    default, work = fake_profiles["default"], fake_profiles["work"]
    original = (work / "memories/USER.md").read_bytes()
    (default / "memories/USER.md").write_bytes(b"Changed only in default.\n")

    assert (work / "memories/USER.md").read_bytes() == original
    assert work == default / "profiles/work"
    assert (default / "skills/example/references/guide.md").is_file()
    assert (work / "skills/example/references/guide.md").is_file()


def test_git_sandbox_has_two_clones_and_local_remote(
    git_repositories: dict[str, Path], git_run: Callable[..., str]
) -> None:
    left, right, remote = (git_repositories[key] for key in ("left", "right", "remote"))
    assert git_run(remote, "rev-parse", "--is-bare-repository") == "true"
    assert git_run(left, "branch", "--show-current") == "master"
    assert git_run(right, "branch", "--show-current") == "master"
    assert git_run(left, "rev-parse", "HEAD") == git_run(right, "rev-parse", "HEAD")
    assert Path(git_run(left, "remote", "get-url", "origin")) == remote
    assert Path(git_run(right, "remote", "get-url", "origin")) == remote


def test_git_sandbox_transfers_real_files(
    git_repositories: dict[str, Path], git_run: Callable[..., str]
) -> None:
    left, right = git_repositories["left"], git_repositories["right"]
    payload = b"\x00\xffbinary support file\n"
    (left / "asset.bin").write_bytes(payload)
    script = left / "helper.sh"
    script.write_bytes(b"#!/bin/sh\nexit 0\n")
    script.chmod(0o755)
    git_run(left, "add", "--", "asset.bin", "helper.sh")
    git_run(left, "update-index", "--chmod=+x", "--", "helper.sh")
    git_run(left, "commit", "-m", "Add binary and executable fixtures")
    git_run(left, "push", "origin", "master")
    git_run(right, "pull", "--ff-only")

    assert (right / "asset.bin").read_bytes() == payload
    assert git_run(right, "ls-files", "--stage", "--", "helper.sh").startswith("100755 ")
    assert git_run(right, "status", "--porcelain") == ""


def test_git_sandbox_can_reproduce_rejected_push(
    git_repositories: dict[str, Path], git_run: Callable[..., str]
) -> None:
    left, right, remote = (git_repositories[key] for key in ("left", "right", "remote"))
    (left / "SOUL.md").write_text("Left update.\n", encoding="utf-8")
    git_run(left, "add", "--", "SOUL.md")
    git_run(left, "commit", "-m", "Left update")
    git_run(left, "push", "origin", "master")
    published = git_run(remote, "rev-parse", "refs/heads/master")
    (right / "SOUL.md").write_text("Right update.\n", encoding="utf-8")
    git_run(right, "add", "--", "SOUL.md")
    git_run(right, "commit", "-m", "Right update")

    with pytest.raises(subprocess.CalledProcessError) as error:
        git_run(right, "push", "origin", "master")

    assert "rejected" in error.value.stderr.lower()
    assert git_run(remote, "rev-parse", "refs/heads/master") == published
    assert git_run(right, "rev-parse", "HEAD") != published


@pytest.mark.parametrize("protocol", ["https", "ssh"])
def test_git_sandbox_blocks_network_transport(
    protocol: str, tmp_path: Path, git_run: Callable[..., str]
) -> None:
    with pytest.raises(subprocess.CalledProcessError) as error:
        git_run(tmp_path, "ls-remote", f"{protocol}://example.invalid/never-contact.git")

    assert f"transport '{protocol}' not allowed" in error.value.stderr
