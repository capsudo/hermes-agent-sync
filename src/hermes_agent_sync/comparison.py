"""Read-only baseline/local/remote comparison. No merge decision or write approval."""

import difflib
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from hermes_agent_sync.selection import SelectionRules
from hermes_agent_sync.snapshot import collect_snapshot

if TYPE_CHECKING:
    from hermes_agent_sync.git_store import GitStore

from hermes_agent_sync.selection import InvalidRules, should_skip
from hermes_agent_sync.snapshot import FileSnapshot, Snapshot


class ComparisonError(ValueError):
    """Incompatible or malformed comparison inputs."""


@dataclass(frozen=True)
class ChangePreview:
    path: str
    baseline: FileSnapshot | None
    local: FileSnapshot | None
    remote: FileSnapshot | None
    classification: str
    local_change: str
    remote_change: str
    deletion_needs_confirmation: bool = False


@dataclass(frozen=True)
class Comparison:
    changes: tuple[ChangePreview, ...]
    skipped_paths: tuple[str, ...] = ()

    @property
    def has_differences(self) -> bool:
        return any(change.local != change.remote for change in self.changes)

    @property
    def review_required(self) -> bool:
        return self.has_differences


@dataclass(frozen=True)
class PreparedComparison:
    fetched_commit: str
    baseline_commit: str | None
    local: Snapshot
    remote: Snapshot
    baseline: Snapshot | None
    comparison: Comparison


def prepare_comparison(
    store: "GitStore",
    profile_home: Path,
    *,
    approved_rules: SelectionRules,
) -> PreparedComparison:
    """Fetch and compare an explicitly selected profile; never acknowledge/apply.

    Caller must serialize per-profile sync work and obtain local setup/scope
    approval first. Model review and final local/remote revalidation are later
    phases. This result alone authorizes no file writes or data publication.
    """
    fetched = store.fetch()
    rules = store.read_rules(fetched)
    if rules != approved_rules:
        raise ComparisonError("Fetched selection rules require local approval")
    previous = store.baseline()
    remote = store.snapshot(fetched, rules, approved_rules=approved_rules)
    local = collect_snapshot(profile_home, rules, approved_rules=approved_rules)
    baseline = (
        None
        if previous is None
        else store.snapshot(previous.commit, rules, approved_rules=approved_rules)
    )
    comparison = compare_snapshots(
        baseline,
        local,
        remote,
        observed_paths=() if previous is None else previous.observed_paths,
    )
    if store.baseline() != previous:
        raise ComparisonError("Previous sync changed during comparison; retry")
    return PreparedComparison(
        fetched,
        None if previous is None else previous.commit,
        local,
        remote,
        baseline,
        comparison,
    )


def _safe_text(text: str) -> str:
    return "".join(
        char
        if char in "\n\t" or not unicodedata.category(char).startswith("C")
        else f"\\u{ord(char):04x}"
        for char in text
    )


def _describe(entry: FileSnapshot | None) -> str:
    if entry is None:
        return "absent"
    try:
        if b"\x00" in entry.data:
            raise UnicodeError
        entry.data.decode("utf-8")
        kind = "text"
    except UnicodeError:
        kind = "binary"
    return f"{len(entry.data)} bytes, {entry.mode:06o}, {kind}"


def render_preview(comparison: Comparison, *, max_chars: int = 20_000) -> str:
    """Bounded plain-text preview; escape controls, summarize binary without decoding.

    This output is display-only. Future review/apply must use the exact structured
    bytes, not this possibly truncated presentation or a guessed merged result.
    """
    if type(max_chars) is not int or max_chars < 100:
        raise ComparisonError("Preview limit must be an integer of at least 100")
    pieces: list[str] = []
    remaining = max_chars
    truncated = False

    def add(text: str) -> None:
        nonlocal remaining, truncated
        text = _safe_text(text)
        if len(text) > remaining:
            truncated = True
        pieces.append(text[:remaining])
        remaining -= min(len(text), remaining)

    add("Comparison only. No files were changed. No merge or deletion is approved.\n")
    for change in comparison.changes:
        if change.classification == "identical":
            continue
        add(f"\n{change.path}: {change.classification}\n")
        for label, entry in (
            ("baseline", change.baseline),
            ("local", change.local),
            ("remote", change.remote),
        ):
            add(f"  {label}: {_describe(entry)}\n")
        if change.deletion_needs_confirmation:
            add("  Whole-file deletion requires confirmation.\n")
        pairs = (
            ((("local", change.local), ("remote", change.remote)),)
            if change.classification == "first-contact"
            else (
                (("baseline", change.baseline), ("local", change.local)),
                (("baseline", change.baseline), ("remote", change.remote)),
            )
        )
        for (left_label, left), (right_label, right) in pairs:
            if left == right:
                continue
            add(f"  {left_label} -> {right_label}\n")
            if any(entry is not None and "binary" in _describe(entry) for entry in (left, right)):
                add("  Binary content omitted.\n")
                continue
            # Bound diff work too, not only final output size.
            if any(entry is not None and len(entry.data) > 65_536 for entry in (left, right)):
                add("  Text content omitted: exceeds preview input limit.\n")
                continue
            before = [] if left is None else left.data.decode("utf-8").splitlines(keepends=True)
            after = [] if right is None else right.data.decode("utf-8").splitlines(keepends=True)
            for line in difflib.unified_diff(
                before, after, fromfile=left_label, tofile=right_label
            ):
                add(line if line.endswith("\n") else line + "\n")
                if remaining == 0:
                    break
        if remaining == 0:
            truncated = True
            break
    if comparison.skipped_paths:
        add("\nSkipped paths (not deletions):\n")
        for path in comparison.skipped_paths:
            add(f"  {path}\n")
            if remaining == 0:
                break
    result = "".join(pieces)
    if truncated:
        suffix = "\n[preview truncated]\n"
        result = result[: max_chars - len(suffix)] + suffix
    return result


def _validate(snapshot: Snapshot) -> None:
    seen: set[str] = set()
    try:
        for item in snapshot.files:
            if (
                item.path in seen
                or not isinstance(item.data, bytes)
                or item.mode not in (0o100644, 0o100755)
                or should_skip(item.path)
                or not snapshot.rules.matches(item.path)
                or snapshot.status(item.path) != "present"
            ):
                raise ComparisonError("Invalid snapshot file entry")
            seen.add(item.path)
        for item in snapshot.skipped:
            should_skip(item.path)
        for directory in snapshot.directories:
            if directory:
                should_skip(directory)
    except InvalidRules:
        raise ComparisonError("Invalid snapshot path") from None


def _change(before: FileSnapshot | None, after: FileSnapshot | None) -> str:
    if before == after:
        return "unchanged"
    if before is None:
        return "added"
    if after is None:
        return "deleted"
    return "modified"


def compare_snapshots(
    baseline: Snapshot | None,
    local: Snapshot,
    remote: Snapshot,
    *,
    observed_paths: tuple[str, ...] = (),
) -> Comparison:
    """Classify files; absence on first contact never proves a deletion."""
    for snapshot in (baseline, local, remote):
        if snapshot is not None:
            _validate(snapshot)
    if not isinstance(observed_paths, tuple) or len(set(observed_paths)) != len(observed_paths):
        raise ComparisonError("Observed paths must be a unique tuple")
    try:
        for path in observed_paths:
            if should_skip(path):
                raise ComparisonError("Unsafe observed path")
    except InvalidRules:
        raise ComparisonError("Invalid observed path") from None
    if local.rules != remote.rules:
        raise ComparisonError("Local and remote selection rules differ")
    base = {} if baseline is None else {item.path: item for item in baseline.files}
    left = {item.path: item for item in local.files}
    right = {item.path: item for item in remote.files}
    changes: list[ChangePreview] = []
    skipped: list[str] = []
    for path in sorted(base.keys() | left.keys() | right.keys() | set(observed_paths)):
        if (
            path == "sync.yaml"
            or local.status(path) == "skipped"
            or remote.status(path) == "skipped"
        ):
            skipped.append(path)
            continue
        known = (
            baseline is not None and path in observed_paths and baseline.status(path) != "skipped"
        )
        a = base.get(path) if known else None
        b, c = left.get(path), right.get(path)
        local_change = _change(a, b) if known else ("present" if b is not None else "absent")
        remote_change = _change(a, c) if known else ("present" if c is not None else "absent")
        if b == c:
            classification = "identical"
        elif not known:
            classification = "first-contact"
        elif b == a:
            classification = "remote-only"
        elif c == a:
            classification = "local-only"
        else:
            classification = "both-changed"
        deletion = known and a is not None and (b is None) != (c is None)
        changes.append(
            ChangePreview(
                path,
                a,
                b,
                c,
                classification,
                local_change,
                remote_change,
                deletion,
            )
        )
    return Comparison(tuple(changes), tuple(skipped))
