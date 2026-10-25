"""Real Git transport plus actual profile snapshots; no live data or publication."""

from hermes_agent_sync.comparison import prepare_comparison, render_preview
from hermes_agent_sync.git_store import GitStore
from hermes_agent_sync.selection import SelectionRules

RULES = "version: 1\nskip-symlinks: true\ninclude: [SOUL.md, skills/**]\n"


def publish(left, git_run, data):
    (left / "sync.yaml").write_text(RULES, encoding="utf-8")
    (left / "SOUL.md").write_bytes(data)
    git_run(left, "add", "--", "SOUL.md", "sync.yaml")
    git_run(left, "commit", "-m", "Artificial shared profile")
    git_run(left, "push", "origin", "master")
    return git_run(left, "rev-parse", "HEAD")


def test_real_fetch_preview_preserves_profile_and_baseline(tmp_path, git_repositories, git_run):
    left, remote = git_repositories["left"], git_repositories["remote"]
    first = publish(left, git_run, b"original\n")
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "SOUL.md").write_bytes(b"local addition\n")
    rules = SelectionRules(("SOUL.md", "skills/**"))
    store = GitStore(tmp_path / "private-cache", str(remote), "master", allow_file=True)

    result = prepare_comparison(store, profile, approved_rules=rules)
    assert result.baseline_commit is None
    assert result.fetched_commit == first
    assert result.comparison.changes[0].classification == "first-contact"
    assert store.baseline() is None
    assert (profile / "SOUL.md").read_bytes() == b"local addition\n"

    # Simulate a previously successful sync ONLY via fixture setup; comparison never does this.
    store.acknowledge(first, ("SOUL.md",), expected_commit=None)
    second = publish(left, git_run, b"remote addition\n")
    result = prepare_comparison(store, profile, approved_rules=rules)
    assert result.baseline_commit == first
    assert result.fetched_commit == second
    assert result.comparison.changes[0].classification == "both-changed"
    assert "original" in render_preview(result.comparison)
    assert store.baseline().commit == first
    assert git_run(remote, "rev-parse", "refs/heads/master") == second
    assert (profile / "SOUL.md").read_bytes() == b"local addition\n"


def test_real_missing_on_first_contact_vs_known_deletion(tmp_path, git_repositories, git_run):
    commit = publish(git_repositories["left"], git_run, b"remote\n")
    profile = tmp_path / "profile"
    profile.mkdir()
    rules = SelectionRules(("SOUL.md", "skills/**"))
    store = GitStore(tmp_path / "cache", str(git_repositories["remote"]), "master", allow_file=True)
    first = prepare_comparison(store, profile, approved_rules=rules)
    (change,) = first.comparison.changes
    assert change.local_change == "absent"
    assert not change.deletion_needs_confirmation

    store.acknowledge(commit, ("SOUL.md",), expected_commit=None)
    second = prepare_comparison(store, profile, approved_rules=rules)
    (change,) = second.comparison.changes
    assert change.local_change == "deleted"
    assert change.deletion_needs_confirmation
    assert store.baseline().commit == commit


def test_real_symlinked_skill_is_skipped_not_deleted(tmp_path, git_repositories, git_run):
    left = git_repositories["left"]
    (left / "skills/example").mkdir(parents=True)
    (left / "skills/example/SKILL.md").write_bytes(b"remote skill")
    git_run(left, "add", "--", "skills")
    commit = publish(left, git_run, b"same")
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "SOUL.md").write_bytes(b"same")
    (profile / "skills").symlink_to(tmp_path / "outside", target_is_directory=True)
    rules = SelectionRules(("SOUL.md", "skills/**"))
    store = GitStore(tmp_path / "cache", str(git_repositories["remote"]), "master", allow_file=True)
    store.fetch()
    store.acknowledge(commit, ("SOUL.md", "skills/example/SKILL.md"), expected_commit=None)
    result = prepare_comparison(store, profile, approved_rules=rules)
    assert result.comparison.skipped_paths == ("skills/example/SKILL.md",)
    assert not result.comparison.has_differences
    assert (profile / "skills").is_symlink()
