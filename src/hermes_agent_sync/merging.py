"""Pure deterministic merge candidates; no candidate grants write approval."""

import difflib
from dataclasses import dataclass

from hermes_agent_sync.comparison import ChangePreview, Comparison
from hermes_agent_sync.selection import InvalidRules, should_skip
from hermes_agent_sync.snapshot import FileSnapshot


class MergeError(ValueError):
    """Malformed or contradictory comparison input."""


@dataclass(frozen=True)
class MergeCandidate:
    """A proposal only: unresolved ``result=None`` never authorizes deletion."""

    change: ChangePreview
    result: FileSnapshot | None
    status: str
    reason: str


@dataclass(frozen=True)
class MergePlan:
    comparison: Comparison
    candidates: tuple[MergeCandidate, ...]


def build_merge_plan(comparison: Comparison) -> MergePlan:
    """Validate all context, then propose only unequal local/remote entries.

    No I/O or approval occurs here. Unsafe, contradictory inputs fail before any
    candidate is built; uncertainty retains the original bytes in ``change``.
    """
    if (
        not isinstance(comparison, Comparison)
        or not isinstance(comparison.changes, tuple)
        or not isinstance(comparison.skipped_paths, tuple)
    ):
        raise MergeError("Expected an immutable Comparison")
    try:
        skipped = set()
        for path in comparison.skipped_paths:
            should_skip(path)
            if path in skipped:
                raise MergeError("Duplicate skipped path")
            skipped.add(path)
    except InvalidRules:
        raise MergeError("Invalid skipped path") from None
    seen = set()
    for change in comparison.changes:
        if not isinstance(change, ChangePreview):
            raise MergeError("Invalid change entry")
        _validate_change(change)
        if change.path in seen or any(
            change.path == path
            or change.path.startswith(path + "/")
            or path.startswith(change.path + "/")
            for path in comparison.skipped_paths
        ):
            raise MergeError("Duplicate or skipped comparison path")
        seen.add(change.path)
    for path in seen:
        parts = path.split("/")
        if any("/".join(parts[:index]) in seen for index in range(1, len(parts))):
            raise MergeError("Comparison file paths overlap by ancestry")
    return MergePlan(
        comparison,
        tuple(
            _candidate(change)
            for change in sorted(comparison.changes, key=lambda item: item.path)
            if change.local != change.remote
        ),
    )


def _candidate(change: ChangePreview) -> MergeCandidate:
    for entry in (change.baseline, change.local, change.remote):
        if entry is None:
            continue
        try:
            if b"\x00" in entry.data:
                raise UnicodeError
            entry.data.decode("utf-8")
        except UnicodeError:
            return MergeCandidate(
                change, None, "binary", "Binary content requires later capability."
            )
    if change.deletion_needs_confirmation:
        return MergeCandidate(
            change, None, "deletion", "Whole-file deletion requires confirmation."
        )
    if any(
        len(entry.data) > 65_536 or len(entry.data.splitlines()) > 4096
        for entry in (change.baseline, change.local, change.remote)
        if entry is not None
    ):
        return MergeCandidate(change, None, "ambiguous", "Text analysis input limit exceeded.")
    if any(
        _has_markers(entry.data)
        for entry in (change.baseline, change.local, change.remote)
        if entry is not None
    ):
        return MergeCandidate(change, None, "ambiguous", "Conflict marker lines require review.")
    if change.baseline is None and change.local is not None and change.remote is not None:
        return MergeCandidate(change, None, "ambiguous", "No shared baseline for differing copies.")
    if change.classification == "both-changed":
        a, b, c = change.baseline, change.local, change.remote
        if b.data == c.data or c.data == a.data:
            data = b.data
        elif b.data == a.data:
            data = c.data
        else:
            data = None
        # Repeated lines admit multiple alignments and costly diff searches.
        # Only align unique bounded chunks; exact content decisions need no diff.
        if data is None and any(
            len(lines) != len(set(lines))
            for entry in (change.baseline, change.local, change.remote)
            if entry is not None
            for lines in (entry.data.splitlines(keepends=True),)
        ):
            return MergeCandidate(
                change, None, "ambiguous", "Repeated lines need contextual review."
            )
        if data is None:
            data = _merge_lines(a.data, b.data, c.data)
        if data is None:
            return MergeCandidate(
                change, None, "ambiguous", "Overlapping text edits require review."
            )
        if len(data) > 65_536 or len(data.splitlines()) > 4096:
            return MergeCandidate(
                change, None, "ambiguous", "Combined text analysis limit exceeded."
            )
        mode = (
            b.mode if b.mode == c.mode or c.mode == a.mode else c.mode if b.mode == a.mode else None
        )
        if mode is None:
            return MergeCandidate(change, None, "ambiguous", "Conflicting executable modes.")
        result = FileSnapshot(change.path, data, mode)
        return MergeCandidate(
            change, result, "ready", "Combined compatible content and mode edits."
        )
    if change.classification == "remote-only" or change.local is None:
        result = change.remote
    else:
        result = change.local
    return MergeCandidate(change, result, "ready", "One-sided change.")


def _merge_lines(base: bytes, local: bytes, remote: bytes) -> bytes | None:
    """Apply byte-preserving line edits, deduplicating only exact shared edits.

    Callers bound input size and reject repeated chunks before difflib. An
    insertion touching a replaced span is conservative ambiguity, including
    different appends at the same position; adjacent replacements are safe.
    """
    before = base.splitlines(keepends=True)
    # Retain side provenance until move detection: deduplicating a shared
    # deletion first can erase a move/delete conflict or duplicate moved text.
    sides = [_line_edits(before, data) for data in (local, remote)]
    if any(_has_move(before, edits) for edits in sides):
        return None
    edits = set(sides[0]) | set(sides[1])
    chunks = []
    cursor = 0
    previous = None
    for start, end, replacement in sorted(edits):
        if previous is not None:
            left, right = previous
            if start < right or (start == right and (start == end or left == right)):
                return None
        previous = (start, end)
        chunks.extend(before[cursor:start])
        chunks.extend(replacement)
        cursor = end
    chunks.extend(before[cursor:])
    return b"".join(chunks)


def _line_edits(before: list[bytes], data: bytes) -> list[tuple[int, int, tuple[bytes, ...]]]:
    after = data.splitlines(keepends=True)
    return [
        (start, end, tuple(after[left:right]))
        for tag, start, end, left, right in difflib.SequenceMatcher(
            None, before, after, autojunk=False
        ).get_opcodes()
        if tag != "equal"
    ]


def _has_move(before: list[bytes], edits: list[tuple[int, int, tuple[bytes, ...]]]) -> bool:
    # Compare logical lines only here: moving an unterminated final line can
    # add a terminator. All alignment, edits and output retain the exact bytes.
    def identity(line: bytes) -> bytes:
        if line.endswith(b"\r\n"):
            return line[:-2]
        return line[:-1] if line.endswith(b"\n") else line

    removed = {identity(line) for start, end, _ in edits for line in before[start:end]}
    inserted = {identity(line) for _, _, replacement in edits for line in replacement}
    return bool(removed & inserted)


def _has_markers(data: bytes) -> bool:
    return any(
        line.lstrip().startswith(("<<<<<<<", "=======", ">>>>>>>", "|||||||"))
        for line in data.decode("utf-8").splitlines()
    )


def validate_resolution(change: ChangePreview, result: FileSnapshot) -> None:
    """Reject a proposal unless every determinable byte outside conflicts survives.

    This validates a proposal, never approves a write. Known-baseline conflicts
    permit replacement only inside connected overlapping line-edit regions.
    Fixed intervening regions must match exactly, in order; uncertain alignment
    is rejected rather than delegated to a model. Repeated-line simultaneous
    edits, moves, binary input, deletion and analysis-limit overflow must defer.
    Exact one-sided/shared content needs no alignment. Without A, differing
    copies may replace full content only when their executable modes agree.
    Input/output text is bounded to 65,536 bytes and 4,096 byte lines; output
    cannot contain marker lines. Errors contain no paths or file data.
    """
    if not isinstance(change, ChangePreview):
        raise MergeError("Invalid resolution change")
    _validate_change(change)
    if (
        not isinstance(result, FileSnapshot)
        or result.path != change.path
        or type(result.data) is not bytes
        or type(result.mode) is not int
        or result.mode not in (0o100644, 0o100755)
    ):
        raise MergeError("Invalid resolution snapshot")
    if len(result.data) > 65_536 or len(result.data.splitlines()) > 4096:
        raise MergeError("Resolution analysis limit exceeded")
    try:
        if b"\x00" in result.data:
            raise UnicodeError
        result.data.decode("utf-8")
    except UnicodeError:
        raise MergeError("Resolution must be UTF-8 text without NUL") from None
    if _has_markers(result.data):
        raise MergeError("Resolution contains conflict marker lines")
    a, b, c = change.baseline, change.local, change.remote
    for entry in (a, b, c):
        if entry is None:
            continue
        if len(entry.data) > 65_536 or len(entry.data.splitlines()) > 4096:
            raise MergeError("Resolution input analysis limit exceeded")
        try:
            if b"\x00" in entry.data:
                raise UnicodeError
            entry.data.decode("utf-8")
        except UnicodeError:
            raise MergeError("Binary input cannot be resolved as text") from None
    if b is None or c is None:
        if a is not None:
            raise MergeError("Deletion cannot be resolved as text")
        present = b if b is not None else c
        mode = present.mode if present is not None else None
    elif a is None:
        mode = b.mode if b.mode == c.mode else None
    else:
        mode = (
            b.mode if b.mode == c.mode or c.mode == a.mode else c.mode if b.mode == a.mode else None
        )
    if mode is None or result.mode != mode:
        raise MergeError("Resolution changes independently determined executable mode")
    if a is None:
        if b is None or c is None:
            if result.data != present.data:
                raise MergeError("Resolution changes determined addition")
        elif b.data == c.data and result.data != b.data:
            raise MergeError("Resolution changes agreed content")
        return
    exact = b.data if b.data == c.data or c.data == a.data else c.data if b.data == a.data else None
    if exact is not None:
        if result.data != exact:
            raise MergeError("Resolution changes determined content")
        return
    if any(
        len(lines) != len(set(lines))
        for entry in (a, b, c)
        for lines in (entry.data.splitlines(keepends=True),)
    ):
        raise MergeError("Resolution preservation cannot be proven for repeated lines")
    before = a.data.splitlines(keepends=True)
    sides = [_line_edits(before, entry.data) for entry in (b, c)]
    if any(_has_move(before, edits) for edits in sides):
        raise MergeError("Resolution preservation cannot be proven for line moves")
    fixed = _protected_segments(before, sorted(set(sides[0]) | set(sides[1])))
    if len(fixed) == 1:
        valid = result.data == fixed[0]
    else:
        # Byte matches alone could hide a protected line inside conflict text.
        # Require whole byte-line segments at actual output line boundaries.
        boundaries = {0}
        offset = 0
        for line in result.data.splitlines(keepends=True):
            offset += len(line)
            boundaries.add(offset)
        cursor, limit = len(fixed[0]), len(result.data) - len(fixed[-1])
        valid = (
            result.data.startswith(fixed[0])
            and result.data.endswith(fixed[-1])
            and cursor in boundaries
            and limit in boundaries
        )
        for segment in fixed[1:-1]:
            positions = [
                position
                for position in boundaries
                if cursor <= position <= limit - len(segment)
                and position + len(segment) in boundaries
                and result.data.startswith(segment, position)
            ]
            if len(positions) != 1:
                valid = False
                break
            cursor = positions[0] + len(segment)
        valid = valid and cursor <= limit
    if not valid:
        raise MergeError("Resolution does not preserve determined edits and context")


def _protected_segments(
    before: list[bytes], edits: list[tuple[int, int, tuple[bytes, ...]]]
) -> list[bytes]:
    """Exact prefix/inter-conflict/suffix segments; gaps alone are editable."""
    groups: list[list[tuple[int, int, tuple[bytes, ...]]]] = []
    for edit in edits:
        start, end, _ = edit
        if groups:
            previous = groups[-1]
            right = max(item[1] for item in previous)
            touches = start == right and (
                start == end or any(x == y == right for x, y, _ in previous)
            )
            if start < right or touches:
                previous.append(edit)
                continue
        groups.append([edit])
    segments = []
    fixed: list[bytes] = []
    cursor = 0
    for group in groups:
        start = group[0][0]
        end = max(item[1] for item in group)
        fixed.extend(before[cursor:start])
        if len(group) == 1:
            fixed.extend(group[0][2])
        else:
            segments.append(b"".join(fixed))
            fixed = []
        cursor = end
    fixed.extend(before[cursor:])
    segments.append(b"".join(fixed))
    return segments


def _delta(before: FileSnapshot | None, after: FileSnapshot | None) -> str:
    if before == after:
        return "unchanged"
    if before is None:
        return "added"
    return "deleted" if after is None else "modified"


def _validate_change(change: ChangePreview) -> None:
    try:
        if should_skip(change.path) or change.path.split("/")[0].casefold() == "sync.yaml":
            raise MergeError("Unsafe or control comparison path")
    except InvalidRules:
        raise MergeError("Invalid comparison path") from None
    a, b, c = change.baseline, change.local, change.remote
    for entry in (a, b, c):
        if entry is not None and (
            not isinstance(entry, FileSnapshot)
            or entry.path != change.path
            or type(entry.data) is not bytes
            or type(entry.mode) is not int
            or entry.mode not in (0o100644, 0o100755)
        ):
            raise MergeError("Invalid comparison file snapshot")
    if not all(
        isinstance(value, str)
        for value in (change.classification, change.local_change, change.remote_change)
    ):
        raise MergeError("Change classifications must be strings")
    # Step 5 encodes baseline knowledge in these two disjoint delta vocabularies.
    # It cannot be reconstructed from a missing baseline entry alone.
    unknown = {"present", "absent"}
    known = {"unchanged", "added", "deleted", "modified"}
    if change.local_change in unknown and change.remote_change in unknown:
        if a is not None:
            raise MergeError("First contact cannot claim a baseline")
        left = "present" if b is not None else "absent"
        right = "present" if c is not None else "absent"
        classification = "identical" if b == c else "first-contact"
        deletion = False
    elif change.local_change in known and change.remote_change in known:
        left, right = _delta(a, b), _delta(a, c)
        classification = (
            "identical"
            if b == c
            else "remote-only"
            if b == a
            else "local-only"
            if c == a
            else "both-changed"
        )
        deletion = a is not None and (b is None) != (c is None)
    else:
        raise MergeError("Unknown or mixed change metadata")
    if (
        change.local_change != left
        or change.remote_change != right
        or change.classification != classification
        or type(change.deletion_needs_confirmation) is not bool
        or change.deletion_needs_confirmation != deletion
    ):
        raise MergeError("Inconsistent comparison classification or deletion flag")
