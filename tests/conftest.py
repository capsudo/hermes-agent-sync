"""Disposable profile homes; no live Hermes installation or Git credentials needed."""

import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect home/profile/config paths for every test and its child processes."""
    home = tmp_path / "home"
    home.mkdir()
    profile = home / ".hermes"
    profile.mkdir()
    config = home / ".config"
    config.mkdir()
    for name, value in {
        "HOME": home,
        "USERPROFILE": home,
        "HERMES_HOME": profile,
        "XDG_CONFIG_HOME": config,
        "GIT_CONFIG_GLOBAL": home / ".gitconfig",
    }.items():
        monkeypatch.setenv(name, str(value))
    for name in tuple(os.environ):
        if name.startswith("GIT_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(home / ".gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_ATTR_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")
    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    return home


@pytest.fixture
def fake_profiles(isolated_home: Path) -> dict[str, Path]:
    """Model a default home and nested named profile on the same host."""
    default = isolated_home / ".hermes"
    profiles = {"default": default, "work": default / "profiles/work"}
    for name, home in profiles.items():
        files = {
            "SOUL.md": f"Instructions for {name}.\n",
            "memories/USER.md": f"Preferences for {name}.\n",
            "memories/MEMORY.md": f"Machine-local notes for {name}.\n",
            "skills/example/SKILL.md": "---\nname: example\n---\n# Example skill\n",
            "skills/example/references/guide.md": "Supporting documentation.\n",
        }
        for relative, text in files.items():
            path = home / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    return profiles


@pytest.fixture
def git_run(isolated_home: Path) -> Callable[..., str]:
    """Run real Git with empty config/hooks, test identity, and file-only transport."""
    executable = shutil.which("git")
    assert executable is not None, "Git must be installed to run this suite"
    hooks = isolated_home / "empty-hooks"
    hooks.mkdir()
    template = isolated_home / "empty-template"
    template.mkdir()

    def run(directory: Path, *arguments: str) -> str:
        result = subprocess.run(
            [
                executable,
                "-c",
                "user.name=Profile Sync Tests",
                "-c",
                "user.email=profile-sync@example.invalid",
                "-c",
                "credential.helper=",
                "-c",
                f"core.hooksPath={hooks}",
                "-c",
                f"init.templateDir={template}",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "tag.gpgsign=false",
                *arguments,
            ],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.stdout.strip()

    return run


@pytest.fixture
def git_repositories(tmp_path: Path, git_run: Callable[..., str]) -> dict[str, Path]:
    """Create an empty commit on master and two independent clones of a bare repo."""
    remote = tmp_path / "remote.git"
    left, right = tmp_path / "left", tmp_path / "right"
    git_run(tmp_path, "init", "--bare", "--initial-branch=master", str(remote))
    git_run(tmp_path, "clone", "--no-local", str(remote), str(left))
    git_run(left, "commit", "--allow-empty", "-m", "Test baseline")
    git_run(left, "push", "-u", "origin", "master")
    git_run(tmp_path, "clone", "--no-local", str(remote), str(right))
    return {"remote": remote, "left": left, "right": right}
