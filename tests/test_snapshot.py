"""Byte-preserving, no-follow snapshot checks against disposable profiles."""

import os
from pathlib import Path

import pytest


def test_open_root_rejects_symlinked_ancestors(tmp_path: Path) -> None:
    from hermes_agent_sync.snapshot import SnapshotError, _open_root

    actual = tmp_path / "actual"
    (actual / "profile").mkdir(parents=True)
    linked = tmp_path / "linked"
    linked.symlink_to(actual, target_is_directory=True)
    with pytest.raises(SnapshotError):
        with _open_root(linked / "profile"):
            pytest.fail("Followed a symlink")


def test_read_file_preserves_binary_bytes_and_git_executable_mode(tmp_path: Path) -> None:
    from hermes_agent_sync.snapshot import _open_root, _read_file

    path = tmp_path / "asset"
    path.write_bytes(b"\x00\xff\r\n")
    path.chmod(0o751)
    with _open_root(tmp_path) as descriptor:
        entry = _read_file(descriptor, "asset", "asset", path.stat(), 1024)
    assert entry.path == "asset"
    assert entry.data == b"\x00\xff\r\n"
    assert entry.mode == 0o100755


def test_read_file_rejects_replacement_with_link(tmp_path: Path) -> None:
    from hermes_agent_sync.snapshot import SnapshotError, _open_root, _read_file

    path = tmp_path / "asset"
    path.write_bytes(b"original")
    original = path.stat()
    target = tmp_path / "outside"
    target.write_bytes(b"must not read")
    path.unlink()
    path.symlink_to(target)
    with _open_root(tmp_path) as descriptor, pytest.raises(SnapshotError):
        _read_file(descriptor, "asset", "asset", original, 1024)


def test_read_file_enforces_size_limit(tmp_path: Path) -> None:
    from hermes_agent_sync.snapshot import SnapshotError, _open_root, _read_file

    path = tmp_path / "asset"
    path.write_bytes(b"too large")
    with _open_root(tmp_path) as descriptor, pytest.raises(SnapshotError, match="limit"):
        _read_file(descriptor, "asset", "asset", path.stat(), 2)


def test_open_root_yields_readable_directory_descriptor(tmp_path: Path) -> None:
    from hermes_agent_sync.snapshot import _open_root

    (tmp_path / "SOUL.md").write_bytes(b"instructions")
    with _open_root(tmp_path) as descriptor:
        assert "SOUL.md" in os.listdir(descriptor)
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_collect_complete_skill_bytes_modes_and_absence(fake_profiles: dict[str, Path]) -> None:
    from hermes_agent_sync.selection import DEFAULT_RULES_TEXT, parse_rules
    from hermes_agent_sync.snapshot import collect_snapshot

    root = fake_profiles["default"]
    asset = root / "skills/example/assets/picture.bin"
    asset.parent.mkdir()
    asset.write_bytes(b"\x00\xff\r\n")
    asset.chmod(0o755)
    rules = parse_rules(DEFAULT_RULES_TEXT)
    result = collect_snapshot(root, rules, approved_rules=rules)
    files = {item.path: item for item in result.files}

    assert files["skills/example/assets/picture.bin"].data == b"\x00\xff\r\n"
    assert files["skills/example/assets/picture.bin"].mode == 0o100755
    assert "skills/example/references/guide.md" in files
    assert "memories/MEMORY.md" not in files
    assert not any(name.startswith("profiles/") for name in files)
    assert result.status("skills/new/SKILL.md") == "missing"
    assert result.status("SOUL.md") == "present"
    assert result.status("memories/MEMORY.md") == "skipped"


def test_collect_skips_links_without_inventing_deletions(fake_profiles: dict[str, Path]) -> None:
    from hermes_agent_sync.selection import DEFAULT_RULES_TEXT, parse_rules
    from hermes_agent_sync.snapshot import collect_snapshot

    root, other = fake_profiles["default"], fake_profiles["work"]
    (root / "skills/linked").symlink_to(other / "skills", target_is_directory=True)
    (root / "SOUL.md").unlink()
    (root / "SOUL.md").symlink_to(other / "SOUL.md")
    rules = parse_rules(DEFAULT_RULES_TEXT)
    result = collect_snapshot(root, rules, approved_rules=rules)

    assert result.status("SOUL.md") == "skipped"
    assert result.status("skills/linked/example/SKILL.md") == "skipped"
    assert result.status("skills/gone/SKILL.md") == "missing"
    assert {(item.path, item.reason) for item in result.skipped} >= {
        ("SOUL.md", "symlink"),
        ("skills/linked", "symlink"),
    }


def test_collect_rejects_selected_special_files(tmp_path: Path) -> None:
    from hermes_agent_sync.selection import SelectionRules
    from hermes_agent_sync.snapshot import SnapshotError, collect_snapshot

    os.mkfifo(tmp_path / "SOUL.md")
    rules = SelectionRules(("SOUL.md",))
    with pytest.raises(SnapshotError, match="regular"):
        collect_snapshot(tmp_path, rules, approved_rules=rules)


def test_collect_rejects_hardlinks(tmp_path: Path) -> None:
    from hermes_agent_sync.selection import SelectionRules
    from hermes_agent_sync.snapshot import SnapshotError, collect_snapshot

    target = tmp_path / "outside"
    target.write_bytes(b"not a standalone profile file")
    os.link(target, tmp_path / "SOUL.md")
    rules = SelectionRules(("SOUL.md",))
    with pytest.raises(SnapshotError, match="Hard-linked"):
        collect_snapshot(tmp_path, rules, approved_rules=rules)


def test_collect_detects_changes_between_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hermes_agent_sync import snapshot
    from hermes_agent_sync.selection import SelectionRules

    path = tmp_path / "SOUL.md"
    path.write_bytes(b"before")
    rules = SelectionRules(("SOUL.md",))
    original = snapshot._scan
    count = 0

    def mutate(*arguments: object) -> object:
        nonlocal count
        result = original(*arguments)
        count += 1
        if count == 1:
            path.write_bytes(b"new content")
        return result

    monkeypatch.setattr(snapshot, "_scan", mutate)
    with pytest.raises(snapshot.SnapshotError, match="changed"):
        snapshot.collect_snapshot(tmp_path, rules, approved_rules=rules)


def test_collect_preserves_empty_files_and_does_not_write(tmp_path: Path) -> None:
    from hermes_agent_sync.selection import SelectionRules
    from hermes_agent_sync.snapshot import collect_snapshot

    (tmp_path / "SOUL.md").write_bytes(b"")
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    rules = SelectionRules(("SOUL.md",))
    result = collect_snapshot(tmp_path, rules, approved_rules=rules)
    assert result.files[0].data == b""
    assert result.files[0].mode == 0o100644
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before


def test_collect_blocks_safety_exclusions_even_with_broad_rules(
    fake_profiles: dict[str, Path],
) -> None:
    from hermes_agent_sync.selection import SelectionRules
    from hermes_agent_sync.snapshot import collect_snapshot

    root = fake_profiles["default"]
    for relative in [".env", "skills/example/.env", "skills/example/private.key"]:
        (root / relative).write_bytes(b"artificial secret fixture")
    rules = SelectionRules(("**",))
    result = collect_snapshot(root, rules, approved_rules=rules)
    assert result.status(".env") == "skipped"
    assert result.status("skills/example/private.key") == "skipped"
    assert result.status("profiles/work/SOUL.md") == "skipped"
    assert result.status("memories/MEMORY.md") == "present"
    assert not any(b"artificial secret" in item.data for item in result.files)


@pytest.mark.parametrize("kind", ["file", "total", "entries", "depth"])
def test_collect_enforces_limits(kind: str, tmp_path: Path) -> None:
    from hermes_agent_sync.selection import SelectionRules
    from hermes_agent_sync.snapshot import SnapshotError, collect_snapshot

    path = tmp_path / "skills/a/b/data"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"large payload")
    rules = SelectionRules(("skills/**",))
    limits = {
        "file": {"max_file_bytes": 2},
        "total": {"max_total_bytes": 2},
        "entries": {"max_entries": 1},
        "depth": {"max_depth": 1},
    }
    with pytest.raises(SnapshotError, match="limit"):
        collect_snapshot(tmp_path, rules, approved_rules=rules, **limits[kind])


def test_collect_blocks_unreadable_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hermes_agent_sync.selection import SelectionRules
    from hermes_agent_sync.snapshot import SnapshotError, collect_snapshot

    (tmp_path / "SOUL.md").write_bytes(b"private")
    original = os.open

    def deny(path: object, *arguments: object, **options: object) -> int:
        if path == "SOUL.md":
            raise PermissionError("artificial denied read")
        return original(path, *arguments, **options)

    monkeypatch.setattr(os, "open", deny)
    # Capability discovery must recognize the wrapper as supporting dir_fd too.
    monkeypatch.setattr(os, "supports_dir_fd", {*os.supports_dir_fd, deny})
    rules = SelectionRules(("SOUL.md",))
    with pytest.raises(SnapshotError, match="open"):
        collect_snapshot(tmp_path, rules, approved_rules=rules)


def test_load_rules_without_following_links(tmp_path: Path) -> None:
    from hermes_agent_sync.snapshot import SnapshotError, load_rules

    path = tmp_path / "sync.yaml"
    path.write_text("version: 1\nskip-symlinks: true\ninclude: [SOUL.md]\n")
    assert load_rules(path).include == ("SOUL.md",)
    path.rename(tmp_path / "actual.yaml")
    path.symlink_to(tmp_path / "actual.yaml")
    with pytest.raises(SnapshotError):
        load_rules(path)
    with pytest.raises(SnapshotError):
        load_rules(tmp_path / "missing.yaml")


def test_read_detects_mutation_during_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hermes_agent_sync.snapshot import SnapshotError, _open_root, _read_file

    path = tmp_path / "asset"
    path.write_bytes(b"before")
    original = os.read
    changed = False

    def mutate(descriptor: int, count: int) -> bytes:
        nonlocal changed
        data = original(descriptor, count)
        if not changed:
            path.write_bytes(b"changed")
            changed = True
        return data

    monkeypatch.setattr(os, "read", mutate)
    with _open_root(tmp_path) as descriptor, pytest.raises(SnapshotError, match="changed"):
        _read_file(descriptor, "asset", "asset", path.stat(), 1024)


def test_root_replacement_is_detected(tmp_path: Path) -> None:
    from hermes_agent_sync.snapshot import SnapshotError, _open_root

    root = tmp_path / "profile"
    root.mkdir()
    with pytest.raises(SnapshotError, match="changed"), _open_root(root):
        root.rename(tmp_path / "moved")
        root.mkdir()


def test_file_directory_collisions_are_not_missing(tmp_path: Path) -> None:
    from hermes_agent_sync.selection import SelectionRules
    from hermes_agent_sync.snapshot import collect_snapshot

    root = tmp_path / "profile"
    root.mkdir()
    (root / "SOUL.md").mkdir()
    (root / "skills").write_bytes(b"not a directory")
    rules = SelectionRules(("SOUL.md", "skills/**"))
    result = collect_snapshot(root, rules, approved_rules=rules)
    assert result.status("SOUL.md") == "skipped"
    assert result.status("skills/example/SKILL.md") == "skipped"


def test_snapshots_are_immutable_copies(tmp_path: Path) -> None:
    from dataclasses import FrozenInstanceError

    from hermes_agent_sync.selection import SelectionRules
    from hermes_agent_sync.snapshot import collect_snapshot

    root = tmp_path / "profile"
    root.mkdir()
    (root / "SOUL.md").write_bytes(b"original")
    rules = SelectionRules(("SOUL.md",))
    result = collect_snapshot(root, rules, approved_rules=rules)
    (root / "SOUL.md").write_bytes(b"later edit")
    assert result.files[0].data == b"original"
    with pytest.raises(FrozenInstanceError):
        result.files[0].data = b"modified"


def test_collect_rejects_unapproved_rules(fake_profiles: dict[str, Path]) -> None:
    from hermes_agent_sync.selection import SelectionRules
    from hermes_agent_sync.snapshot import SnapshotError, collect_snapshot

    with pytest.raises(SnapshotError, match="approv"):
        collect_snapshot(
            fake_profiles["default"],
            SelectionRules(("**",)),
            approved_rules=SelectionRules(("SOUL.md",)),
        )
