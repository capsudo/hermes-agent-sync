"""Read-only three-way classifications; previews never authorize writes."""

import pytest

from hermes_agent_sync.selection import SelectionRules
from hermes_agent_sync.snapshot import FileSnapshot, Snapshot


def tree(data: bytes | None, mode: int = 0o100644) -> Snapshot:
    rules = SelectionRules(("SOUL.md",))
    entries = () if data is None else (FileSnapshot("SOUL.md", data, mode),)
    return Snapshot(rules, entries, (), ("",))


def test_first_contact_absence_is_not_deletion() -> None:
    from hermes_agent_sync.comparison import compare_snapshots

    result = compare_snapshots(None, tree(b"local"), tree(None))
    (change,) = result.changes
    assert change.path == "SOUL.md"
    assert change.classification == "first-contact"
    assert change.local_change == "present"
    assert change.remote_change == "absent"
    assert not change.deletion_needs_confirmation
    assert result.has_differences
    assert result.review_required


@pytest.mark.parametrize(
    ("a", "b", "c", "kind", "left", "right", "deletion"),
    [
        (b"old", b"new", b"old", "local-only", "modified", "unchanged", False),
        (b"old", b"old", b"new", "remote-only", "unchanged", "modified", False),
        (b"old", b"left", b"right", "both-changed", "modified", "modified", False),
        (b"old", None, b"old", "local-only", "deleted", "unchanged", True),
        (b"old", b"old", None, "remote-only", "unchanged", "deleted", True),
        (b"old", None, b"new", "both-changed", "deleted", "modified", True),
        (b"old", None, None, "identical", "deleted", "deleted", False),
        (None, b"new", None, "local-only", "added", "unchanged", False),
        (None, None, b"new", "remote-only", "unchanged", "added", False),
    ],
)
def test_known_baseline_classifies_both_sides(a, b, c, kind, left, right, deletion) -> None:
    from hermes_agent_sync.comparison import compare_snapshots

    result = compare_snapshots(tree(a), tree(b), tree(c), observed_paths=("SOUL.md",))
    (change,) = result.changes
    assert change.classification == kind
    assert change.local_change == left
    assert change.remote_change == right
    assert change.deletion_needs_confirmation is deletion
    assert result.has_differences is (b != c)


def test_newly_selected_path_does_not_infer_deletion() -> None:
    from hermes_agent_sync.comparison import compare_snapshots

    (change,) = compare_snapshots(tree(b"old"), tree(None), tree(b"new")).changes
    assert change.classification == "first-contact"
    assert change.local_change == "absent"
    assert not change.deletion_needs_confirmation
    assert change.baseline is None


def test_mode_only_change_is_a_real_change() -> None:
    from hermes_agent_sync.comparison import compare_snapshots

    (change,) = compare_snapshots(
        tree(b"same"), tree(b"same", 0o100755), tree(b"same"), observed_paths=("SOUL.md",)
    ).changes
    assert change.classification == "local-only"
    assert change.local_change == "modified"


def test_skipped_link_is_not_a_remote_deletion() -> None:
    from hermes_agent_sync.comparison import compare_snapshots
    from hermes_agent_sync.snapshot import SkippedPath

    skipped = Snapshot(tree(None).rules, (), (SkippedPath("SOUL.md", "symlink"),), ("",))
    result = compare_snapshots(tree(b"old"), skipped, tree(b"remote"), observed_paths=("SOUL.md",))
    assert result.changes == ()
    assert result.skipped_paths == ("SOUL.md",)
    assert not result.has_differences


@pytest.mark.parametrize("invalid", ["duplicates", "mode", "bytes", "unsafe", "observed"])
def test_malformed_inputs_are_rejected(invalid: str) -> None:
    from hermes_agent_sync.comparison import ComparisonError, compare_snapshots

    local = tree(b"text")
    observed = ("SOUL.md",)
    if invalid == "duplicates":
        local = Snapshot(local.rules, local.files * 2, (), ("",))
    elif invalid == "mode":
        local = Snapshot(local.rules, (FileSnapshot("SOUL.md", b"text", 0o120000),), (), ("",))
    elif invalid == "bytes":
        local = Snapshot(local.rules, (FileSnapshot("SOUL.md", "text", 0o100644),), (), ("",))
    elif invalid == "unsafe":
        local = Snapshot(local.rules, (FileSnapshot("../escape", b"text", 0o100644),), (), ("",))
    else:
        observed = ("../escape",)
    with pytest.raises(ComparisonError):
        compare_snapshots(None, local, tree(None), observed_paths=observed)


def test_text_preview_shows_both_directions_without_merging() -> None:
    from hermes_agent_sync.comparison import compare_snapshots, render_preview

    result = compare_snapshots(
        tree(b"old\n"), tree(b"local\n"), tree(b"remote\n"), observed_paths=("SOUL.md",)
    )
    text = render_preview(result)
    assert "both-changed" in text
    assert "baseline -> local" in text and "baseline -> remote" in text
    assert "-old" in text and "+local" in text and "+remote" in text
    assert "<<<<<<<" not in text
    assert "No files were changed" in text


def test_binary_preview_does_not_decode_or_print_payload() -> None:
    from hermes_agent_sync.comparison import compare_snapshots, render_preview

    result = compare_snapshots(None, tree(b"\x00\xffPRIVATE"), tree(None))
    text = render_preview(result)
    assert "binary" in text
    assert "PRIVATE" not in text
    assert "8 bytes" not in text  # Byte length must reflect actual payload, not a guess.
    assert "9 bytes" in text


def test_preview_escapes_terminal_controls_and_bounds_output() -> None:
    from hermes_agent_sync.comparison import compare_snapshots, render_preview

    text = render_preview(
        compare_snapshots(None, tree(b"\x1b[31m" + b"x" * 10000), tree(None)), max_chars=200
    )
    assert "\x1b" not in text
    assert len(text) <= 200
    assert "truncated" in text


def test_rules_mismatch_blocks_comparison() -> None:
    from hermes_agent_sync.comparison import ComparisonError, compare_snapshots

    remote = Snapshot(SelectionRules(("skills/**",)), (), (), ("",))
    with pytest.raises(ComparisonError, match="rules"):
        compare_snapshots(None, tree(None), remote)


def test_previously_skipped_path_returns_as_first_contact() -> None:
    from hermes_agent_sync.comparison import compare_snapshots
    from hermes_agent_sync.snapshot import SkippedPath

    previous = Snapshot(tree(None).rules, (), (SkippedPath("SOUL.md", "symlink"),), ("",))
    (change,) = compare_snapshots(
        previous, tree(None), tree(b"remote"), observed_paths=("SOUL.md",)
    ).changes
    assert change.classification == "first-contact"
    assert not change.deletion_needs_confirmation


def test_exclusions_block_previously_observed_paths() -> None:
    from hermes_agent_sync.comparison import compare_snapshots

    excluded = Snapshot(SelectionRules(("SOUL.md",), ("SOUL.md",)), (), (), ("",))
    result = compare_snapshots(tree(b"old"), excluded, excluded, observed_paths=("SOUL.md",))
    assert not result.has_differences
    assert result.skipped_paths == ("SOUL.md",)


def test_preview_missing_final_newline_remains_different() -> None:
    from hermes_agent_sync.comparison import compare_snapshots

    result = compare_snapshots(None, tree(b"text"), tree(b"text\n"))
    assert result.has_differences


def test_shared_control_file_is_not_profile_content() -> None:
    from hermes_agent_sync.comparison import compare_snapshots

    rules = SelectionRules(("**",))
    left = Snapshot(rules, (FileSnapshot("sync.yaml", b"local", 0o100644),), (), ("",))
    right = Snapshot(rules, (FileSnapshot("sync.yaml", b"remote", 0o100644),), (), ("",))
    result = compare_snapshots(None, left, right)
    assert not result.has_differences
    assert result.skipped_paths == ("sync.yaml",)


def test_prepare_comparison_fetches_without_acknowledging(tmp_path) -> None:
    from types import SimpleNamespace

    from hermes_agent_sync.comparison import prepare_comparison

    rules = SelectionRules(("SOUL.md",))
    (tmp_path / "SOUL.md").write_bytes(b"local")
    calls = []

    class Store:
        def fetch(self):
            calls.append("fetch")
            return "a" * 40

        def read_rules(self, commit):
            assert commit == "a" * 40
            return rules

        def baseline(self):
            return SimpleNamespace(commit="b" * 40, observed_paths=("SOUL.md",))

        def snapshot(self, commit, selection, *, approved_rules):
            assert selection == approved_rules == rules
            return tree(b"remote" if commit == "a" * 40 else b"old")

        def acknowledge(self, *arguments, **options):
            pytest.fail("Comparison must never acknowledge successful sync")

    result = prepare_comparison(Store(), tmp_path, approved_rules=rules)
    assert result.fetched_commit == "a" * 40
    assert result.baseline_commit == "b" * 40
    assert result.comparison.changes[0].classification == "both-changed"
    assert calls == ["fetch"]
    assert (tmp_path / "SOUL.md").read_bytes() == b"local"


def test_failed_fetch_produces_no_profile_write(tmp_path) -> None:
    from hermes_agent_sync.comparison import prepare_comparison

    class Store:
        def fetch(self):
            raise RuntimeError("unavailable test remote")

    (tmp_path / "SOUL.md").write_bytes(b"keep")
    with pytest.raises(RuntimeError, match="unavailable"):
        prepare_comparison(Store(), tmp_path, approved_rules=SelectionRules(("SOUL.md",)))
    assert (tmp_path / "SOUL.md").read_bytes() == b"keep"


def test_unapproved_fetched_scope_blocks_before_profile_collection(tmp_path) -> None:
    from hermes_agent_sync.comparison import ComparisonError, prepare_comparison

    class Store:
        def fetch(self):
            return "a" * 40

        def read_rules(self, commit):
            return SelectionRules(("**",))

    with pytest.raises(ComparisonError, match="approval"):
        prepare_comparison(Store(), tmp_path, approved_rules=SelectionRules(("SOUL.md",)))


def test_comparison_uses_bytes_not_timestamps() -> None:
    from hermes_agent_sync.comparison import compare_snapshots

    result = compare_snapshots(
        tree(b"old\r\n"), tree(b"old\n"), tree(b"old\r\n"), observed_paths=("SOUL.md",)
    )
    assert result.has_differences
    assert result.changes[0].classification == "local-only"


def test_independent_profiles_do_not_share_baseline_eligibility() -> None:
    from hermes_agent_sync.comparison import compare_snapshots

    tracked = compare_snapshots(tree(b"a"), tree(None), tree(b"a"), observed_paths=("SOUL.md",))
    new = compare_snapshots(tree(b"a"), tree(None), tree(b"a"), observed_paths=())
    assert tracked.changes[0].local_change == "deleted"
    assert new.changes[0].local_change == "absent"


def test_prepare_rejects_baseline_change_during_collection(tmp_path) -> None:
    from types import SimpleNamespace

    from hermes_agent_sync.comparison import ComparisonError, prepare_comparison

    rules = SelectionRules(("SOUL.md",))
    (tmp_path / "SOUL.md").write_bytes(b"local")

    class Store:
        reads = 0

        def fetch(self):
            return "a" * 40

        def read_rules(self, commit):
            return rules

        def baseline(self):
            self.reads += 1
            return SimpleNamespace(
                commit=("b" if self.reads == 1 else "c") * 40,
                observed_paths=("SOUL.md",),
            )

        def snapshot(self, *arguments, **options):
            return tree(b"remote")

    with pytest.raises(ComparisonError, match="changed"):
        prepare_comparison(Store(), tmp_path, approved_rules=rules)
    assert (tmp_path / "SOUL.md").read_bytes() == b"local"


def test_scope_expansion_does_not_reuse_ineligible_baseline() -> None:
    from hermes_agent_sync.comparison import compare_snapshots

    previous = Snapshot(SelectionRules(("skills/**",)), (), (), ("",))
    result = compare_snapshots(previous, tree(None), tree(b"remote"), observed_paths=("SOUL.md",))
    (change,) = result.changes
    assert change.classification == "first-contact"
    assert change.baseline is None
    assert change.local_change == "absent"
    assert not change.deletion_needs_confirmation


def test_identical_inputs_do_not_need_review() -> None:
    from hermes_agent_sync.comparison import compare_snapshots

    result = compare_snapshots(None, tree(b"same"), tree(b"same"))
    assert not result.has_differences
    assert not result.review_required
    assert result.changes[0].classification == "identical"
