"""Pure deterministic candidates preserve original comparison context."""

from dataclasses import FrozenInstanceError, replace

import pytest

from hermes_agent_sync.comparison import Comparison, compare_snapshots
from hermes_agent_sync.selection import SelectionRules
from hermes_agent_sync.snapshot import FileSnapshot, Snapshot


def tree(data, mode=0o100644):
    files = () if data is None else (FileSnapshot("SOUL.md", data, mode),)
    return Snapshot(SelectionRules(("**",)), files, (), ("",))


def compare(a, b, c, *, known=True, modes=(0o100644,) * 3):
    return compare_snapshots(
        tree(a, modes[0]) if known else None,
        tree(b, modes[1]),
        tree(c, modes[2]),
        observed_paths=("SOUL.md",) if known else (),
    )


def plan(comparison):
    from hermes_agent_sync.merging import build_merge_plan

    return build_merge_plan(comparison)


@pytest.mark.parametrize(
    "a,b,c,known",
    [
        (None, None, None, False),
        (b"old", b"same", b"same", True),
        (b"old", None, None, True),
        (None, b"\x00\xff", b"\x00\xff", False),
    ],
)
def test_equal_sides_have_no_candidates_and_retain_context(a, b, c, known):
    comparison = compare(a, b, c, known=known)
    result = plan(comparison)
    assert result.comparison is comparison
    assert result.candidates == ()
    with pytest.raises(FrozenInstanceError):
        result.candidates = ()


@pytest.mark.parametrize(
    "a,b,c,known,modes,expected",
    [
        (None, b"local\r\n", None, False, (0o100644, 0o100755, 0o100644), b"local\r\n"),
        (None, None, b"remote", False, (0o100644,) * 3, b"remote"),
        (None, b"", None, True, (0o100644,) * 3, b""),
        (None, None, b"new", True, (0o100644,) * 3, b"new"),
        (b"a\nb\n", b"a\n", b"a\nb\n", True, (0o100644,) * 3, b"a\n"),
        (b"a\nb\n", b"a\nb\n", b"b\n", True, (0o100644,) * 3, b"b\n"),
        (b"text", b"text", b"text", True, (0o100644, 0o100755, 0o100644), b"text"),
    ],
)
def test_one_sided_changes_are_exact_candidates(a, b, c, known, modes, expected):
    comparison = compare(a, b, c, known=known, modes=modes)
    (candidate,) = plan(comparison).candidates
    assert candidate.status == "ready"
    assert candidate.result.data == expected
    changed = comparison.changes[0].local
    if comparison.changes[0].classification == "remote-only" or b is None:
        changed = comparison.changes[0].remote
    assert candidate.result is changed
    assert candidate.change is comparison.changes[0]
    assert candidate.reason
    with pytest.raises(FrozenInstanceError):
        candidate.result = None


@pytest.mark.parametrize(
    "b,c", [(None, b"old"), (b"old", None), (None, b"edited"), (b"edited", None)]
)
def test_whole_file_deletion_always_requires_confirmation(b, c):
    comparison = compare(b"old", b, c)
    (candidate,) = plan(comparison).candidates
    assert candidate.status == "deletion"
    assert candidate.result is None
    assert candidate.change is comparison.changes[0]
    assert candidate.change.deletion_needs_confirmation is True
    assert "confirmation" in candidate.reason


@pytest.mark.parametrize(
    "a,b,c,known,modes",
    [
        (None, b"\x00payload", None, False, (0o100644,) * 3),
        (None, None, b"\xff", False, (0o100644,) * 3),
        (b"text", b"\xc0\x80", b"text", True, (0o100644,) * 3),
        (b"\xff", b"fixed", b"other", True, (0o100644,) * 3),
        (b"\x00", b"\x00", b"\x00", True, (0o100644, 0o100755, 0o100644)),
    ],
)
def test_changed_binary_is_deferred_without_corrupting_sources(a, b, c, known, modes):
    comparison = compare(a, b, c, known=known, modes=modes)
    before = comparison.changes
    (candidate,) = plan(comparison).candidates
    assert candidate.status == "binary"
    assert candidate.result is None
    assert comparison.changes == before
    assert "binary" in candidate.reason.lower()


@pytest.mark.parametrize("b,c", [(b"local", b"remote"), (b"", b"remote")])
def test_first_contact_different_copies_is_ambiguous(b, c):
    comparison = compare(None, b, c, known=False)
    (candidate,) = plan(comparison).candidates
    assert candidate.status == "ambiguous"
    assert candidate.result is None
    assert candidate.change is comparison.changes[0]
    assert "base" in candidate.reason.lower()


@pytest.mark.parametrize(
    "a,b,c,expected",
    [
        (
            b"alpha\nbeta\ngamma\n",
            b"ALPHA\nbeta\ngamma\n",
            b"alpha\nbeta\nGAMMA\n",
            b"ALPHA\nbeta\nGAMMA\n",
        ),
        (
            b"alpha\nbeta\ngamma\n",
            b"new\nalpha\nbeta\ngamma\n",
            b"alpha\nbeta\ngamma\nlast\n",
            b"new\nalpha\nbeta\ngamma\nlast\n",
        ),
        (b"alpha\nbeta\ngamma\n", b"beta\ngamma\n", b"alpha\nbeta\n", b"beta\n"),
        (
            b"alpha\nbeta\ngamma\n",
            b"new\nalpha\nbeta\ngamma\n",
            b"alpha\nbeta\n",
            b"new\nalpha\nbeta\n",
        ),
    ],
)
def test_disjoint_line_edits_merge_exactly(a, b, c, expected):
    for left, right in ((b, c), (c, b)):
        comparison = compare(a, left, right)
        (candidate,) = plan(comparison).candidates
        assert candidate.status == "ready"
        assert candidate.result == FileSnapshot("SOUL.md", expected, 0o100644)
        assert candidate.result is not comparison.changes[0].local
        assert candidate.result is not comparison.changes[0].remote


@pytest.mark.parametrize(
    "a,b,c",
    [
        (b"one\ntwo\n", b"ONE\ntwo\n", b"different\ntwo\n"),
        (b"one\ntwo\n", b"two\n", b"ONE\ntwo\n"),
        (b"one\n", b"one\nlocal\n", b"one\nremote\n"),
        (b"", b"local\n", b"remote\n"),
        (b"a\nb\nc\n", b"a\nnew\nb\nc\n", b"a\nreplacement\n"),
    ],
)
def test_overlapping_edits_are_ambiguous(a, b, c):
    for left, right in ((b, c), (c, b)):
        (candidate,) = plan(compare(a, left, right)).candidates
        assert candidate.status == "ambiguous"
        assert candidate.result is None
        assert candidate.reason


@pytest.mark.parametrize(
    "a,b,c,expected",
    [
        (b"a\nb\nc\n", b"A\nb\nc\n", b"A\nb\nC\n", b"A\nb\nC\n"),
        (b"a\nb\nc\n", b"b\nc\n", b"b\nC\n", b"b\nC\n"),
        (b"a\nb\nc\n", b"x\na\nb\nc\n", b"x\na\nb\nC\n", b"x\na\nb\nC\n"),
    ],
)
def test_identical_shared_edits_are_applied_once(a, b, c, expected):
    for left, right in ((b, c), (c, b)):
        (candidate,) = plan(compare(a, left, right)).candidates
        assert candidate.status == "ready"
        assert candidate.result.data == expected


@pytest.mark.parametrize(
    "a,b,c,expected",
    [
        (b"a\r\nb\r\nc", b"A\r\nb\r\nc", b"a\r\nb\r\nC", b"A\r\nb\r\nC"),
        (b"a\nb\nc", b"A\nb\nc", b"a\nb\nC", b"A\nb\nC"),
        (b"a\rb\nc\n", b"A\rb\nc\n", b"a\rb\nC\n", b"A\rb\nC\n"),
    ],
)
def test_combined_results_preserve_line_ending_bytes(a, b, c, expected):
    (candidate,) = plan(compare(a, b, c)).candidates
    assert candidate.status == "ready"
    assert candidate.result.data == expected


@pytest.mark.parametrize(
    "modes",
    [
        (0o100644, 0o100755, 0o100644),
        (0o100755, 0o100644, 0o100755),
    ],
)
def test_mode_only_change_combines_with_other_side_content(modes):
    for swapped in (False, True):
        b, c = (b"new", b"old") if swapped else (b"old", b"new")
        use_modes = (modes[0], modes[2], modes[1]) if swapped else modes
        comparison = compare(b"old", b, c, modes=use_modes)
        (candidate,) = plan(comparison).candidates
        assert candidate.status == "ready"
        assert candidate.result == FileSnapshot("SOUL.md", b"new", modes[1])


@pytest.mark.parametrize(
    "which,marker",
    [
        (0, b"<<<<<<< baseline\n"),
        (1, b"<<<<<<< local\n"),
        (2, b"=======\n"),
        (1, b">>>>>>> remote\r\n"),
        (2, b"||||||| base\n"),
        (1, b"    <<<<<<< local\n"),
    ],
)
def test_conflict_marker_lines_are_never_ready(which, marker):
    data = [b"old\n", b"local\n", b"old\n"]
    data[which] = marker
    comparison = compare(*data)
    (candidate,) = plan(comparison).candidates
    assert candidate.status == "ambiguous"
    assert candidate.result is None
    assert "marker" in candidate.reason.lower()


@pytest.mark.parametrize("oversized", [b"x" * 65_537, b"x\n" * 4097], ids=["bytes", "lines"])
@pytest.mark.parametrize("which", [0, 1, 2])
def test_oversized_text_is_deferred_before_merge_analysis(oversized, which, monkeypatch):
    import hermes_agent_sync.merging as merging

    def forbidden(*args, **kwargs):
        pytest.fail("oversized input reached the line diff")

    monkeypatch.setattr(merging.difflib, "SequenceMatcher", forbidden)
    data = [b"old\n", b"local\n", b"remote\n"]
    data[which] = oversized
    (candidate,) = plan(compare(*data)).candidates
    assert candidate.status == "ambiguous"
    assert candidate.result is None
    assert "limit" in candidate.reason.lower()


def test_repeated_text_is_deferred_before_quadratic_diff(monkeypatch):
    import hermes_agent_sync.merging as merging

    def forbidden(*args, **kwargs):
        pytest.fail("repeated input reached the line diff")

    monkeypatch.setattr(merging.difflib, "SequenceMatcher", forbidden)
    a = b"same\n" * 1000
    (candidate,) = plan(compare(a, b"local\n" + a, a + b"remote\n")).candidates
    assert candidate.status == "ambiguous"
    assert candidate.result is None
    assert "repeated" in candidate.reason.lower()


@pytest.mark.parametrize("b,c", [(b"local", b"remote"), (b"", b"remote")])
def test_different_additions_with_known_absent_base_are_ambiguous(b, c):
    (candidate,) = plan(compare(None, b, c)).candidates
    assert candidate.status == "ambiguous"
    assert candidate.result is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("classification", "invented"),
        ("classification", "remote-only"),
        ("classification", "identical"),
        ("classification", "first-contact"),
        ("local_change", "unchanged"),
        ("local_change", "present"),
        ("remote_change", "deleted"),
        ("remote_change", "invented"),
        ("deletion_needs_confirmation", True),
        ("deletion_needs_confirmation", 0),
    ],
)
def test_inconsistent_change_metadata_is_rejected(field, value):
    from hermes_agent_sync.merging import MergeError

    comparison = compare(b"old", b"new", b"old")
    forged = replace(comparison.changes[0], **{field: value})
    with pytest.raises(MergeError):
        plan(Comparison((forged,)))


@pytest.mark.parametrize("known", [True, False])
def test_forged_first_contact_baseline_is_rejected(known):
    from hermes_agent_sync.merging import MergeError

    change = compare(None, b"left", b"right", known=known).changes[0]
    forged = replace(
        change,
        baseline=FileSnapshot("SOUL.md", b"base", 0o100644),
        classification="first-contact",
        local_change="present",
        remote_change="present",
    )
    with pytest.raises(MergeError):
        plan(Comparison((forged,)))


@pytest.mark.parametrize("a,b,c", [(b"old", None, b"old"), (b"old", b"new", None)])
def test_suppressed_deletion_confirmation_is_rejected(a, b, c):
    from hermes_agent_sync.merging import MergeError

    change = compare(a, b, c).changes[0]
    with pytest.raises(MergeError):
        plan(Comparison((replace(change, deletion_needs_confirmation=False),)))


@pytest.mark.parametrize(
    "path",
    [
        "../SOUL.md",
        "/SOUL.md",
        "a//b",
        "a/./b",
        "C:SOUL.md",
        "a\\b",
        "x\nmd",
        "sync.yaml",
        "sync.yaml/child",
        ".git/config",
        "config.yaml",
        "skills/.env/file",
    ],
)
def test_unsafe_or_control_change_paths_are_rejected(path):
    from hermes_agent_sync.merging import MergeError

    change = compare(None, b"new", None, known=False).changes[0]
    entry = replace(change.local, path=path)
    with pytest.raises(MergeError):
        plan(Comparison((replace(change, path=path, local=entry),)))


@pytest.mark.parametrize(
    "field,value",
    [
        ("path", "other.md"),
        ("data", "text"),
        ("data", bytearray(b"bytes")),
        ("mode", 0o120000),
        ("mode", "100644"),
        ("mode", True),
        ("mode", float(0o100644)),
    ],
)
@pytest.mark.parametrize("side", ["baseline", "local", "remote"])
def test_invalid_snapshot_values_are_rejected(side, field, value):
    from hermes_agent_sync.merging import MergeError

    change = compare(b"old", b"new", b"old").changes[0]
    entry = replace(getattr(change, side), **{field: value})
    with pytest.raises(MergeError):
        plan(Comparison((replace(change, **{side: entry}),)))


@pytest.mark.parametrize(
    "value",
    [
        None,
        object(),
        (),
        Comparison(None),
        Comparison([]),
        Comparison((object(),)),
        Comparison((None,)),
        Comparison((), []),
        Comparison((), (None,)),
    ],
)
def test_malformed_comparison_structure_raises_merge_error(value):
    from hermes_agent_sync.merging import MergeError

    with pytest.raises(MergeError):
        plan(value)


@pytest.mark.parametrize("kind", ["duplicate", "skipped", "skipped-ancestor", "skipped-descendant"])
def test_duplicate_or_skipped_change_paths_are_rejected(kind):
    from hermes_agent_sync.merging import MergeError

    change = compare(b"old", b"new", b"old").changes[0]
    if kind == "duplicate":
        comparison = Comparison((change, change))
    else:
        if kind == "skipped-ancestor":
            change = replace(
                change,
                path="skills/example.md",
                baseline=None,
                local=FileSnapshot("skills/example.md", b"new", 0o100644),
                remote=None,
                classification="first-contact",
                local_change="present",
                remote_change="absent",
            )
            skipped = "skills"
        else:
            skipped = "SOUL.md/child" if kind == "skipped-descendant" else "SOUL.md"
        comparison = Comparison((change,), (skipped,))
    with pytest.raises(MergeError):
        plan(comparison)


@pytest.mark.parametrize(
    "field,value",
    [
        ("local_change", []),
        ("remote_change", {}),
        ("classification", None),
        ("baseline", object()),
        ("local", object()),
        ("remote", object()),
        ("path", None),
    ],
)
def test_malformed_change_fields_raise_merge_error(field, value):
    from hermes_agent_sync.merging import MergeError

    change = compare(b"old", b"new", b"old").changes[0]
    with pytest.raises(MergeError):
        plan(Comparison((replace(change, **{field: value}),)))


@pytest.mark.parametrize("b,c", [(b"repeat\n" * 3, b"new\n"), (b"new\n", b"repeat\n" * 3)])
def test_mode_content_independence_does_not_require_line_alignment(b, c):
    modes = (
        (0o100644, 0o100755, 0o100644)
        if b.startswith(b"repeat")
        else (0o100644, 0o100644, 0o100755)
    )
    (candidate,) = plan(compare(b"repeat\n" * 3, b, c, modes=modes)).candidates
    assert candidate.status == "ready"
    assert candidate.result == FileSnapshot("SOUL.md", b"new\n", 0o100755)


def test_plan_keeps_unchanged_context_skips_and_sorts_candidates():
    from hermes_agent_sync.snapshot import SkippedPath

    rules = SelectionRules(("**",))
    context = FileSnapshot("context.md", b"unchanged\n", 0o100644)
    old = FileSnapshot("z.md", b"old\n", 0o100644)
    a = Snapshot(rules, (context, old), (), ("",))
    b = Snapshot(
        rules,
        (context, replace(old, data=b"local\n"), FileSnapshot("a.md", b"new\n", 0o100644)),
        (SkippedPath("skills", "symlink"),),
        ("",),
    )
    c = Snapshot(
        rules, (context, old, FileSnapshot("skills/x.md", b"hidden", 0o100644)), (), ("", "skills")
    )
    comparison = compare_snapshots(a, b, c, observed_paths=("context.md", "z.md"))
    assert comparison.skipped_paths == ("skills/x.md",)
    # Order is not a merge decision; retain the exact original context object.
    comparison = replace(comparison, changes=tuple(reversed(comparison.changes)))
    result = plan(comparison)
    assert result.comparison is comparison
    assert [item.change.path for item in result.candidates] == ["a.md", "z.md"]
    assert any(change.path == "context.md" for change in result.comparison.changes)


@pytest.mark.parametrize(
    "a,b,c,status",
    [
        (b"old", b"", b"old", "ready"),
        (b"", None, b"", "deletion"),
        (b"old", None, b"\x00", "binary"),
        (None, b"", b"other", "ambiguous"),
    ],
)
def test_empty_absent_and_binary_deletion_are_distinct(a, b, c, status):
    comparison = compare(a, b, c)
    (candidate,) = plan(comparison).candidates
    assert candidate.status == status
    assert candidate.change is comparison.changes[0]
    assert (candidate.result is None) == (status != "ready")


@pytest.mark.parametrize("payload", [b"x" * 65_537, b"x\n" * 4097], ids=["bytes", "lines"])
def test_one_sided_oversized_text_is_explicitly_deferred(payload):
    (candidate,) = plan(compare(b"old", payload, b"old")).candidates
    assert candidate.status == "ambiguous"
    assert candidate.result is None
    assert "limit" in candidate.reason.lower()


def test_forged_ancestor_file_changes_are_rejected():
    from hermes_agent_sync.merging import MergeError

    original = compare(None, b"new", None, known=False).changes[0]
    changes = tuple(
        replace(original, path=path, local=replace(original.local, path=path))
        for path in ("skills", "skills/x.md")
    )
    with pytest.raises(MergeError):
        plan(Comparison(changes))


@pytest.mark.parametrize("paths", [("same.md", "same.md"), ("../bad",), ("x\x00md",)])
def test_malformed_skipped_paths_are_rejected(paths):
    from hermes_agent_sync.merging import MergeError

    with pytest.raises(MergeError):
        plan(Comparison((), paths))


def test_file_ancestry_is_rejected_even_with_intervening_sorted_sibling():
    from hermes_agent_sync.merging import MergeError

    original = compare(None, b"new", None, known=False).changes[0]
    changes = tuple(
        replace(original, path=path, local=replace(original.local, path=path))
        for path in ("skills", "skills-other.md", "skills/x.md")
    )
    with pytest.raises(MergeError):
        plan(Comparison(changes))


@pytest.mark.parametrize(
    "local,remote,expected",
    [
        (b"A\nb\nc\n", b"a\nB\nc\n", b"A\nB\nc\n"),
        (b"b\nc\n", b"a\nB\nc\n", b"B\nc\n"),
    ],
)
def test_adjacent_distinct_line_edits_are_compatible(local, remote, expected):
    (candidate,) = plan(compare(b"a\nb\nc\n", local, remote)).candidates
    assert candidate.status == "ready"
    assert candidate.result.data == expected


@pytest.mark.parametrize(
    "a,b,c,expected",
    [
        (b"a\nb\nc\n", b"a\nb\nC\n", b"a\nb\nC\n", None),
        (b"a\nb\nc\n", b"A\nb\nc\n", b"a\nb\nC\n", b"A\nb\nC\n"),
    ],
)
def test_same_mode_change_and_disjoint_content_merge(a, b, c, expected):
    comparison = compare(a, b, c, modes=(0o100644, 0o100755, 0o100755))
    result = plan(comparison)
    if expected is None:
        assert result.candidates == ()
    else:
        (candidate,) = result.candidates
        assert candidate.status == "ready"
        assert candidate.result == FileSnapshot("SOUL.md", expected, 0o100755)


def test_input_limits_allow_exact_boundary_without_running_diff():
    for payload in (b"x" * 65_536, b"x\n" * 4096):
        comparison = compare(b"old", payload, b"old")
        (candidate,) = plan(comparison).candidates
        assert candidate.status == "ready"
        assert candidate.result is comparison.changes[0].local


@pytest.mark.parametrize(
    "a,b,c",
    [
        (b"a\nb\nc\nd\n", b"b\na\nc\nd\n", b"a\nc\nd\nb\n"),
        (b"a\nb\nc\n", b"b\na\nc\n", b"a\nc\n"),
    ],
)
def test_simultaneous_move_interactions_preserve_provenance(a, b, c):
    for left, right in ((b, c), (c, b)):
        (candidate,) = plan(compare(a, left, right)).candidates
        assert candidate.status == "ambiguous"
        assert candidate.result is None


@pytest.mark.parametrize("ending", [b"\n", b"\r\n"], ids=["lf", "crlf"])
@pytest.mark.parametrize("other", [(b"a", b"c", b"b", b""), (b"a", b"b", b"")])
@pytest.mark.parametrize("swapped", [False, True])
def test_unterminated_newline_move_interactions_are_ambiguous(ending, other, swapped):
    a = ending.join((b"a", b"b", b"c"))
    b = ending.join((b"c", b"a", b"b", b""))
    c = ending.join(other)
    left, right = (c, b) if swapped else (b, c)
    comparison = compare(a, left, right)
    (candidate,) = plan(comparison).candidates
    assert candidate.status == "ambiguous"
    assert candidate.result is None
    assert candidate.change is comparison.changes[0]
    assert candidate.change.baseline.data == a
    assert candidate.change.local.data == left
    assert candidate.change.remote.data == right


@pytest.mark.parametrize("ending", [b"\n", b"\r\n"], ids=["lf", "crlf"])
@pytest.mark.parametrize("other", [(b"a", b"c", b"b", b""), (b"a", b"b", b"")])
@pytest.mark.parametrize("swapped", [False, True])
def test_unterminated_newline_move_resolution_is_rejected(ending, other, swapped):
    from hermes_agent_sync.merging import MergeError

    a = ending.join((b"a", b"b", b"c"))
    b = ending.join((b"c", b"a", b"b", b""))
    c = ending.join(other)
    left, right = (c, b) if swapped else (b, c)
    change = compare(a, left, right).changes[0]
    unsafe = ending.join((b"c", b"a", b"c", b"b", b"")) if len(other) == 4 else b
    for output in (unsafe, left, right, b"arbitrary"):
        with pytest.raises(MergeError, match="line moves"):
            validate(change, output)


@pytest.mark.parametrize("removed,inserted", [(b"\n", b"\r\n"), (b"\r\n", b"\n")])
@pytest.mark.parametrize("delete", [False, True], ids=["other-move", "delete"])
@pytest.mark.parametrize("swapped", [False, True])
def test_empty_newline_move_interactions_remain_uncertain(removed, inserted, delete, swapped):
    from hermes_agent_sync.merging import MergeError

    a = b"a\nb\n" + removed
    b = inserted + b"a\nb\n"
    c = b"a\nb\n" if delete else b"a\n" + removed + b"b\n"
    left, right = (c, b) if swapped else (b, c)
    comparison = compare(a, left, right)
    (candidate,) = plan(comparison).candidates
    assert candidate.status == "ambiguous"
    assert candidate.result is None
    for output in (left, right, b"arbitrary"):
        with pytest.raises(MergeError, match="line moves"):
            validate(comparison.changes[0], output)


@pytest.mark.parametrize("ending", [b"\n", b"\r\n"], ids=["lf", "crlf"])
@pytest.mark.parametrize("trailing", [b" ", b"\t", b" \t", b"\v", b"\f", b"\r"])
@pytest.mark.parametrize("content", [b"c", b""], ids=["text", "empty"])
def test_newline_move_identity_preserves_trailing_content(ending, trailing, content):
    a = ending.join((b"a", b"b", content + trailing))
    b = ending.join((content, b"a", b"b", b""))
    c = ending.join((b"a", b"B", content + trailing))
    expected = ending.join((content, b"a", b"B", b""))
    for left, right in ((b, c), (c, b)):
        comparison = compare(a, left, right)
        (candidate,) = plan(comparison).candidates
        assert candidate.status == "ready"
        assert candidate.result.data == expected
        validate(comparison.changes[0], expected)


@pytest.mark.parametrize(
    "a,moved",
    [
        (b"a\nb\nc\n", b"b\na\nc\n"),
        (b"a\nb\nc", b"c\na\nb\n"),
        (b"a\r\nb\r\nc", b"c\r\na\r\nb\r\n"),
        (b"a\nb\n\n", b"\r\na\nb\n"),
    ],
)
def test_one_sided_exact_move_remains_ready(a, moved):
    for left, right in ((a, moved), (moved, a)):
        comparison = compare(a, left, right)
        (candidate,) = plan(comparison).candidates
        assert candidate.status == "ready"
        assert candidate.result.data == moved
        validate(comparison.changes[0], moved)


def validate(change, data, mode=0o100644):
    from hermes_agent_sync import merging

    merging.validate_resolution(change, FileSnapshot(change.path, data, mode))


def test_resolution_protects_clear_hunks_and_unchanged_context():
    from hermes_agent_sync.merging import MergeError

    a = b"header\nconflict\ncontext\nedit\nfooter\n"
    b = b"header\nlocal\ncontext\nEDIT\nfooter\n"
    c = b"header\nremote\ncontext\nedit\nfooter\n"
    for left, right in ((b, c), (c, b)):
        change = compare(a, left, right).changes[0]
        with pytest.raises(MergeError):
            validate(change, c)
        with pytest.raises(MergeError):
            validate(change, b"rewritten\nresolved\ncontext\nEDIT\nfooter\n")
        with pytest.raises(MergeError):
            validate(change, b"header\nresolved\nEDIT\nfooter\n")
        validate(change, b"header\nresolved\ncontext\nEDIT\nfooter\n")


@pytest.mark.parametrize("swapped", [False, True])
def test_content_conflict_cannot_reopen_clear_executable_removal(swapped):
    from hermes_agent_sync.merging import MergeError

    b, c = b"local\ncontext\n", b"remote\ncontext\n"
    modes = (0o100755, 0o100644, 0o100755)
    if swapped:
        b, c = c, b
        modes = (modes[0], modes[2], modes[1])
    change = compare(b"base\ncontext\n", b, c, modes=modes).changes[0]
    with pytest.raises(MergeError):
        validate(change, b"resolved\ncontext\n", 0o100755)
    validate(change, b"resolved\ncontext\n", 0o100644)


@pytest.mark.parametrize("known", [False, True])
@pytest.mark.parametrize("mode", [0o100644, 0o100755])
def test_differing_copies_without_a_accept_content_but_preserve_agreed_mode(known, mode):
    from hermes_agent_sync.merging import MergeError

    change = compare(None, b"local", b"remote", known=known, modes=(mode,) * 3).changes[0]
    validate(change, b"new full resolution", mode)
    with pytest.raises(MergeError):
        validate(change, b"new full resolution", 0o100755 if mode == 0o100644 else 0o100644)


def test_no_baseline_disagreeing_modes_require_host_deferral():
    from hermes_agent_sync.merging import MergeError

    change = compare(
        None, b"local", b"remote", known=False, modes=(0o100644, 0o100644, 0o100755)
    ).changes[0]
    for mode in (0o100644, 0o100755):
        with pytest.raises(MergeError):
            validate(change, b"resolution", mode)


@pytest.mark.parametrize(
    "data",
    [b"\x00", b"\xff", b"x" * 65_537, b"x\n" * 4097],
    ids=["nul", "invalid-utf8", "bytes", "lines"],
)
def test_resolution_rejects_unsafe_or_unbounded_output(data):
    from hermes_agent_sync.merging import MergeError

    change = compare(None, b"local", b"remote", known=False).changes[0]
    with pytest.raises(MergeError):
        validate(change, data)


@pytest.mark.parametrize("prefix", ["", " \t", "\u00a0"])
@pytest.mark.parametrize("marker", ["<<<<<<<X", "========", ">>>>>>>remote", "|||||||base"])
def test_resolution_forbids_any_seven_character_marker_after_whitespace(prefix, marker):
    from hermes_agent_sync.merging import MergeError

    change = compare(None, b"local", b"remote", known=False).changes[0]
    with pytest.raises(MergeError):
        validate(change, ("safe\n" + prefix + marker + "\n").encode())


@pytest.mark.parametrize("prefix", ["\u00a0", "\u2003"])
def test_candidate_unicode_whitespace_marker_is_ambiguous(prefix):
    (candidate,) = plan(compare(b"old", (prefix + "<<<<<<<X").encode(), b"old")).candidates
    assert candidate.status == "ambiguous"


@pytest.mark.parametrize(
    "data",
    [b"same\n" * 3, b"x" * 65_537, b"x\n" * 4097, b"\xff"],
    ids=["repeated", "bytes", "lines", "binary"],
)
def test_resolution_uncertain_input_rejects_arbitrary_content_before_diff(data, monkeypatch):
    from hermes_agent_sync import merging

    change = compare(data, b"local\n" + data, data + b"remote\n").changes[0]

    def forbidden(*args, **kwargs):
        pytest.fail("uncertain input reached diff")

    monkeypatch.setattr(merging.difflib, "SequenceMatcher", forbidden)
    with pytest.raises(merging.MergeError):
        validate(change, b"arbitrary replacement")


@pytest.mark.parametrize("known", [False, True])
def test_one_sided_addition_resolution_cannot_erase_clear_content(known):
    from hermes_agent_sync.merging import MergeError

    change = compare(None, b"clear addition", None, known=known).changes[0]
    validate(change, b"clear addition")
    with pytest.raises(MergeError):
        validate(change, b"replacement")


def test_resolution_cannot_join_unchanged_suffix_onto_conflict_line():
    from hermes_agent_sync.merging import MergeError

    change = compare(b"base\ncontext\n", b"local\ncontext\n", b"remote\ncontext\n").changes[0]
    with pytest.raises(MergeError):
        validate(change, b"resolvedcontext\n")


@pytest.mark.parametrize("known", [False, True])
def test_equal_copies_without_a_cannot_be_rewritten_by_resolution(known):
    from hermes_agent_sync.merging import MergeError

    change = compare(None, b"same", b"same", known=known).changes[0]
    validate(change, b"same")
    with pytest.raises(MergeError):
        validate(change, b"arbitrary")


@pytest.mark.parametrize(
    "a,b,c",
    [
        (b"a\nb\nc\nd\n", b"b\na\nc\nd\n", b"a\nc\nd\nb\n"),
        (b"a\nb\nc\n", b"b\na\nc\n", b"a\nc\n"),
    ],
)
def test_move_uncertainty_cannot_be_bypassed_by_resolution(a, b, c):
    from hermes_agent_sync.merging import MergeError

    for left, right in ((b, c), (c, b)):
        change = compare(a, left, right).changes[0]
        for output in (left, right, b"arbitrary"):
            with pytest.raises(MergeError):
                validate(change, output)


def test_multiple_conflicts_preserve_ordered_fixed_regions_and_clear_insertions_deletions():
    from hermes_agent_sync.merging import MergeError

    a = b"header\none\ncontext\ntwo\nmiddle\nremove\nfooter\n"
    b = b"header\nLOCAL1\ncontext\nLOCAL2\nmiddle\nremove\nfooter\nextra\n"
    c = b"header\nREMOTE1\ncontext\nREMOTE2\nmiddle\nfooter\n"
    expected = b"header\nresolved1\ncontext\nresolved2\nmiddle\nfooter\nextra\n"
    for left, right in ((b, c), (c, b)):
        change = compare(a, left, right).changes[0]
        validate(change, expected)
        for bad in (
            expected.replace(b"extra\n", b""),
            expected.replace(b"context\n", b""),
            expected.replace(b"footer\n", b"remove\nfooter\n"),
            expected.replace(b"resolved1\ncontext\n", b"context\nresolved1\ncontext\n"),
        ):
            with pytest.raises(MergeError):
                validate(change, bad)


@pytest.mark.parametrize("ending", [b"\n", b"\r\n", b"\r"])
def test_resolution_preserves_exact_context_line_endings_and_no_final_newline(ending):
    from hermes_agent_sync.merging import MergeError

    change = compare(
        ending.join((b"base", b"context", b"footer")),
        ending.join((b"local", b"context", b"footer")),
        ending.join((b"remote", b"context", b"footer")),
    ).changes[0]
    expected = ending.join((b"resolved", b"context", b"footer"))
    validate(change, expected)
    with pytest.raises(MergeError):
        validate(change, expected + b"\n")


def test_conflict_marker_input_can_be_resolved_but_never_emitted():
    from hermes_agent_sync.merging import MergeError

    change = compare(
        b"header\nbase\nfooter\n", b"header\n<<<<<<<X\nfooter\n", b"header\nremote\nfooter\n"
    ).changes[0]
    assert plan(Comparison((change,))).candidates[0].status == "ambiguous"
    validate(change, b"header\nresolved\nfooter\n")
    with pytest.raises(MergeError):
        validate(change, change.local.data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("path", "other.md"),
        ("data", "text"),
        ("data", bytearray(b"text")),
        ("mode", True),
        ("mode", "100644"),
        ("mode", 0o120000),
    ],
)
def test_resolution_invalid_snapshot_is_sanitized_merge_error(field, value):
    from hermes_agent_sync.merging import MergeError, validate_resolution

    change = compare(None, b"local", b"remote", known=False).changes[0]
    result = replace(FileSnapshot(change.path, b"secret-payload", 0o100644), **{field: value})
    with pytest.raises(MergeError) as error:
        validate_resolution(change, result)
    assert "secret-payload" not in str(error.value)
    assert "local" not in str(error.value)
    assert "remote" not in str(error.value)


@pytest.mark.parametrize("value", [None, object(), b"secret"])
def test_resolution_malformed_inputs_raise_only_merge_error(value):
    from hermes_agent_sync.merging import MergeError, validate_resolution

    change = compare(None, b"local", b"remote", known=False).changes[0]
    with pytest.raises(MergeError):
        validate_resolution(value, FileSnapshot(change.path, b"okay", 0o100644))
    with pytest.raises(MergeError):
        validate_resolution(change, value)


@pytest.mark.parametrize("payload", [b"x" * 65_536, b"x\n" * 4096], ids=["bytes", "lines"])
def test_resolution_bounds_allow_exact_boundary(payload):
    change = compare(None, b"local", b"remote", known=False).changes[0]
    validate(change, payload)


def test_resolution_keeps_shared_hunks_once_and_adjacent_distinct_edits():
    from hermes_agent_sync.merging import MergeError

    a = b"one\nanchor\ntwo\nthree\ncontext\nfour\nfive\n"
    b = b"ONE\nanchor\nTWO\nthree\ncontext\nlocal\nfive\n"
    c = b"ONE\nanchor\ntwo\nTHREE\ncontext\nremote\nfive\n"
    expected = b"ONE\nanchor\nTWO\nTHREE\ncontext\nresolved\nfive\n"
    for left, right in ((b, c), (c, b)):
        change = compare(a, left, right).changes[0]
        validate(change, expected)
        for bad in (
            expected.replace(b"ONE\n", b"one\n"),
            expected.replace(b"TWO\n", b"two\n"),
        ):
            with pytest.raises(MergeError):
                validate(change, bad)


@pytest.mark.parametrize("path", ["../secret", "sync.yaml", ".git/config", "config.yaml"])
def test_resolution_rejects_unsafe_change_scope_with_sanitized_error(path):
    from hermes_agent_sync.merging import MergeError, validate_resolution

    change = compare(None, b"left", b"right", known=False).changes[0]
    change = replace(
        change,
        path=path,
        local=replace(change.local, path=path),
        remote=replace(change.remote, path=path),
    )
    with pytest.raises(MergeError) as error:
        validate_resolution(change, FileSnapshot(path, b"secret-content", 0o100644))
    assert path not in str(error.value)
    assert "secret-content" not in str(error.value)


def test_resolution_ready_deterministic_content_must_remain_exact():
    from hermes_agent_sync.merging import MergeError

    change = compare(b"a\nb\nc\n", b"A\nb\nc\n", b"a\nb\nC\n").changes[0]
    validate(change, b"A\nb\nC\n")
    with pytest.raises(MergeError):
        validate(change, b"rewritten\nb\nC\n")


def test_resolution_repeated_exact_content_and_one_sided_move_need_no_alignment(monkeypatch):
    from hermes_agent_sync import merging

    for a, b, c in (
        (b"same\n" * 3, b"same\n" * 3, b"changed\n"),
        (b"a\nb\nc\n", b"b\na\nc\n", b"a\nb\nc\n"),
    ):
        change = compare(a, b, c).changes[0]

        def forbidden(*args, **kwargs):
            pytest.fail("exact content reached diff")

        monkeypatch.setattr(merging.difflib, "SequenceMatcher", forbidden)
        expected = c if b == a else b
        validate(change, expected)


@pytest.mark.parametrize("kind", ["bytes", "lines"])
def test_combined_candidate_cannot_exceed_resolution_analysis_bounds(kind):
    if kind == "bytes":
        a = b"anchor\n"
        b, c = b"L" * 40_000 + b"\n" + a, a + b"R" * 40_000 + b"\n"
    else:
        a = b"anchor\n"
        b = b"".join(f"local-{index}\n".encode() for index in range(2500)) + a
        c = a + b"".join(f"remote-{index}\n".encode() for index in range(2500))
    (candidate,) = plan(compare(a, b, c)).candidates
    assert candidate.status == "ambiguous"
    assert candidate.result is None
    assert "limit" in candidate.reason.lower()
