"""Read-only snapshots. No live profile writes, Git operations, or model calls.

Descriptor-relative traversal requires POSIX no-follow operations. Unsupported
platforms fail closed rather than silently falling back to path-based copying.
"""

import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from hermes_agent_sync.selection import SelectionRules, parse_rules, should_skip


@dataclass(frozen=True)
class FileSnapshot:
    """Exact bytes and Git's executable mode; never text-normalized or executed."""

    path: str
    data: bytes
    mode: int


class SnapshotError(RuntimeError):
    """Selected data could not be collected safely; do not interpret as deletion."""


@dataclass(frozen=True)
class SkippedPath:
    """A path (and descendants) deliberately not observed, never a deletion."""

    path: str
    reason: str


@dataclass(frozen=True)
class Snapshot:
    """Immutable in-memory copy of eligible bytes; missing is not deletion consent."""

    rules: SelectionRules
    files: tuple[FileSnapshot, ...]
    skipped: tuple[SkippedPath, ...]
    directories: tuple[str, ...]

    def status(self, path: str) -> str:
        """Distinguish observed absence from blocked/excluded/unselected paths."""
        if not self.rules.matches(path) or should_skip(path):
            return "skipped"
        if any(path == item.path or path.startswith(item.path + "/") for item in self.skipped):
            return "skipped"
        if path in self.directories:
            return "skipped"
        for item in self.files:
            if item.path == path:
                return "present"
            if path.startswith(item.path + "/"):
                return "skipped"
        return "missing"


def load_rules(path: Path) -> SelectionRules:
    """Read a bounded sync.yaml without following file or ancestor symlinks."""
    path = Path(path)
    try:
        with _open_root(path.parent) as parent:
            info = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                raise SnapshotError("Selection rules must be a regular non-symlink file")
            entry = _read_file(parent, path.name, path.name, info, 64 * 1024)
            return parse_rules(entry.data)
    except OSError as error:
        raise SnapshotError("Cannot safely read selection rules") from error


def collect_snapshot(
    profile_home: Path,
    rules: SelectionRules,
    *,
    approved_rules: SelectionRules,
    max_file_bytes: int = 16 * 1024 * 1024,
    max_total_bytes: int = 64 * 1024 * 1024,
    max_entries: int = 50_000,
    max_depth: int = 64,
) -> Snapshot:
    """Collect twice and reject unstable trees; caller supplies locally approved rules.

    No files are written. POSIX descriptors pin directories and reject links.
    These checks are not a global filesystem lock: later apply must revalidate.
    """
    if not isinstance(rules, SelectionRules) or rules != approved_rules:
        raise SnapshotError("Selection rules require local approval before collection")
    if any(
        type(value) is not int or value <= 0
        for value in (max_file_bytes, max_total_bytes, max_entries, max_depth)
    ):
        raise SnapshotError("Snapshot limits must be positive integers")
    try:
        with _open_root(profile_home) as descriptor:
            first, markers = _scan(
                descriptor, rules, max_file_bytes, max_total_bytes, max_entries, max_depth
            )
            second, new_markers = _scan(
                descriptor, rules, max_file_bytes, max_total_bytes, max_entries, max_depth
            )
            if first != second or markers != new_markers:
                raise SnapshotError("Profile changed during snapshot collection; retry")
            return first
    except OSError as error:
        raise SnapshotError("Cannot safely collect selected profile files") from error


def _scan(
    root: int,
    rules: SelectionRules,
    file_limit: int,
    total_limit: int,
    entry_limit: int,
    depth_limit: int,
) -> tuple[Snapshot, dict[str, tuple[int, ...]]]:
    files: list[FileSnapshot] = []
    skipped: list[SkippedPath] = []
    directories: list[str] = []
    markers: dict[str, tuple[int, ...]] = {}
    total = 0
    visited = 0

    def walk(descriptor: int, prefix: str, depth: int) -> None:
        nonlocal total, visited
        if depth > depth_limit:
            raise SnapshotError("Snapshot depth limit exceeded")
        before = os.fstat(descriptor)
        markers[prefix] = _fingerprint(before)
        directories.append(prefix)
        for name in sorted(os.listdir(descriptor)):
            visited += 1
            if visited > entry_limit:
                raise SnapshotError("Snapshot entry limit exceeded")
            relative = f"{prefix}/{name}" if prefix else name
            reason = should_skip(relative)
            if reason:
                skipped.append(SkippedPath(relative, reason))
                continue
            if not rules.matches(relative) and not rules.could_include(relative):
                skipped.append(SkippedPath(relative, "not-selected"))
                continue
            info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            markers[relative] = _fingerprint(info)
            if stat.S_ISLNK(info.st_mode):
                skipped.append(SkippedPath(relative, "symlink"))
            elif stat.S_ISDIR(info.st_mode):
                child = os.open(
                    name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
                )
                try:
                    if _fingerprint(os.fstat(child)) != _fingerprint(info):
                        raise SnapshotError("Selected directory changed before reading")
                    walk(child, relative, depth + 1)
                    current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                    if _fingerprint(current) != _fingerprint(os.fstat(child)):
                        raise SnapshotError("Selected directory changed during collection")
                finally:
                    os.close(child)
            elif not stat.S_ISREG(info.st_mode):
                raise SnapshotError("Selected path is not a regular file or directory")
            elif rules.matches(relative):
                entry = _read_file(
                    descriptor, name, relative, info, min(file_limit, total_limit - total)
                )
                files.append(entry)
                total += len(entry.data)
            else:
                skipped.append(SkippedPath(relative, "not-selected"))
        if _fingerprint(os.fstat(descriptor)) != _fingerprint(before):
            raise SnapshotError("Directory changed during snapshot collection")

    walk(root, "", 0)
    return Snapshot(rules, tuple(files), tuple(skipped), tuple(directories)), markers


def _fingerprint(info: os.stat_result) -> tuple[int, ...]:
    # Timestamps detect mutation during a read, not synchronization history.
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _read_file(
    parent: int, name: str, relative: str, expected: os.stat_result, limit: int
) -> FileSnapshot:
    """Read a pinned regular file, detecting replacement and concurrent mutation."""
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    except OSError as error:
        raise SnapshotError("Cannot safely open selected file") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or _fingerprint(before) != _fingerprint(expected):
            raise SnapshotError("Selected file changed before reading")
        if before.st_nlink != 1:
            raise SnapshotError("Hard-linked selected files are not supported")
        if before.st_size > limit:
            raise SnapshotError("Snapshot size limit exceeded")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(65536, limit - size + 1))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > limit:
                raise SnapshotError("Snapshot size limit exceeded")
        after = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if _fingerprint(before) != _fingerprint(after) or _fingerprint(after) != _fingerprint(
            current
        ):
            raise SnapshotError("Selected file changed while reading")
        return FileSnapshot(
            relative, b"".join(chunks), 0o100755 if before.st_mode & 0o111 else 0o100644
        )
    except OSError as error:
        raise SnapshotError("Cannot safely read selected file") from error
    finally:
        os.close(descriptor)


@contextmanager
def _open_root(path: Path) -> Iterator[int]:
    """Pin every ancestor without resolving/following symlinks, then close all fds."""
    if not all(hasattr(os, name) for name in ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK")):
        raise SnapshotError("Safe descriptor traversal is unavailable on this platform")
    if os.open not in os.supports_dir_fd or os.stat not in os.supports_dir_fd:
        raise SnapshotError("Descriptor-relative filesystem access is unavailable")
    path = Path(path)
    if ".." in path.parts:
        raise SnapshotError("Parent traversal in profile root is not allowed")
    absolute = path.absolute()
    descriptors: list[int] = []
    bindings: list[tuple[int, str, int]] = []
    try:
        descriptor = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptors.append(descriptor)
        for name in absolute.parts[1:]:
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            descriptors.append(child)
            bindings.append((descriptor, name, child))
            descriptor = child
        yield descriptor
        for parent, name, child in bindings:
            current = os.stat(name, dir_fd=parent, follow_symlinks=False)
            pinned = os.fstat(child)
            if (current.st_dev, current.st_ino, current.st_mode) != (
                pinned.st_dev,
                pinned.st_ino,
                pinned.st_mode,
            ):
                raise SnapshotError("Profile directory changed during collection")
    except OSError as error:
        raise SnapshotError("Cannot safely access profile directory") from error
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
