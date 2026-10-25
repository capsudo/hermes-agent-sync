"""Pure selection rules for canonical, profile-relative POSIX paths.

Globs are root-anchored and case-sensitive. Only a complete ``**`` segment may
cross directories. Safety checks are deliberately case-insensitive and apply to
ancestors too; the documented deny patterns below cannot be overridden by YAML.
MEMORY.md is excluded only by the configurable packaged template.

Documents are limited to 64 KiB and 16 YAML nesting levels; include/exclude
combined are limited to 256 patterns, each at most 1024 UTF-8 bytes, and 64 KiB
of pattern bytes. Rule equality preserves order and duplicates, normalizing
only list inputs into immutable tuples. No filesystem traversal is done here:
callers must enforce no-follow descriptor-based reads, including ancestors.
"""

import re
import unicodedata
from dataclasses import dataclass
from fnmatch import fnmatchcase
from importlib.resources import files

import yaml

DEFAULT_RULES_TEXT = (
    files("hermes_agent_sync").joinpath("default-sync.yaml").read_text(encoding="utf-8")
)

MAX_RULES_BYTES = 65_536
MAX_PATTERNS = 256
MAX_PATTERN_BYTES = 1_024


class InvalidRules(ValueError):
    """A rules document, pattern, or relative path is invalid."""


def _segments(path: str, *, root: bool = False) -> tuple[str, ...]:
    """Reject rather than normalize attacker-controlled paths."""
    if not isinstance(path, str):
        raise InvalidRules("path must be a string")
    if root and path == "":
        return ()
    if (
        not path
        or "\\" in path
        or re.match(r"^[A-Za-z]:", path)
        or any(unicodedata.category(char) in {"Cc", "Cs"} for char in path)
    ):
        raise InvalidRules("path must be canonical relative POSIX")
    parts = tuple(path.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        raise InvalidRules("path must be canonical relative POSIX")
    return parts


def _validate_pattern(pattern: str) -> None:
    segments = _segments(pattern)
    if len(pattern.encode("utf-8")) > MAX_PATTERN_BYTES:
        raise InvalidRules("pattern is too long")
    for segment in segments:
        if "**" in segment and segment != "**":
            raise InvalidRules("recursive glob must occupy a whole segment")
        index = 0
        while index < len(segment):
            if segment[index] != "[":
                index += 1
                continue
            start = index + 1
            if start < len(segment) and segment[start] == "!":
                start += 1
            end = start + (start < len(segment) and segment[start] == "]")
            end = segment.find("]", end)
            if end < 0 or end == start:
                raise InvalidRules("invalid glob character class")
            body = segment[start:end]
            for offset in range(1, len(body) - 1):
                if body[offset] == "-" and body[offset - 1] > body[offset + 1]:
                    raise InvalidRules("invalid glob character range")
            index = end + 1


@dataclass(frozen=True)
class SelectionRules:
    """Immutable, ordered scope; callers must separately enforce filesystem safety."""

    include: tuple[str, ...]
    exclude: tuple[str, ...] = ()
    version: int = 1
    skip_symlinks: bool = True

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise InvalidRules("version must be 1")
        if self.skip_symlinks is not True:
            raise InvalidRules("skip-symlinks must be true")
        for name in ("include", "exclude"):
            patterns = getattr(self, name)
            if not isinstance(patterns, (list, tuple)):
                raise InvalidRules("patterns must be a list or tuple of strings")
            if len(patterns) > MAX_PATTERNS:
                raise InvalidRules("too many patterns")
            if any(not isinstance(pattern, str) for pattern in patterns):
                raise InvalidRules("patterns must be strings")
            for pattern in patterns:
                _validate_pattern(pattern)
            object.__setattr__(self, name, tuple(patterns))
        patterns = self.include + self.exclude
        if len(patterns) > MAX_PATTERNS:
            raise InvalidRules("too many patterns")
        if sum(len(pattern.encode("utf-8")) for pattern in patterns) > MAX_RULES_BYTES:
            raise InvalidRules("patterns are too large")

    def could_include(self, directory: str) -> bool:
        """Retain potential descendants; only whole-subtree excludes prune.

        An empty directory denotes the root. File-only exclusions and safety
        patterns are conservatively ignored here; walkers also call should_skip.
        """
        parts = _segments(directory, root=True)
        if any(pattern.endswith("/**") and _match(pattern, parts) for pattern in self.exclude):
            return False
        for pattern in self.include:
            segments = tuple(pattern.split("/"))
            if any(index < len(segments) for index in _states_after(segments, parts)):
                return True
        return False

    def matches(self, path: str) -> bool:
        """Match a canonical path; exclusion wins over any inclusion."""
        parts = _segments(path)
        if should_skip(path) is not None:
            return False
        return any(_match(pattern, parts) for pattern in self.include) and not any(
            _match(pattern, parts) for pattern in self.exclude
        )


def _match(pattern: str, parts: tuple[str, ...]) -> bool:
    segments = tuple(pattern.split("/"))
    return len(segments) in _states_after(segments, parts)


def _states_after(segments: tuple[str, ...], parts: tuple[str, ...]) -> set[int]:
    """A small glob automaton avoids exponential recursive '**' backtracking."""
    states = {0}
    for part in (*parts, None):
        # A '**' can consume no path segments, including adjacent '**' patterns.
        for index in range(len(segments)):
            if index in states and segments[index] == "**":
                states.add(index + 1)
        if part is None:
            return states
        states = {
            index if segments[index] == "**" else index + 1
            for index in states
            if index < len(segments)
            and (segments[index] == "**" or fnmatchcase(part, segments[index]))
        }
    return states


# Root-only runtime directories must never enroll sibling profiles or host state.
_ROOT_RUNTIME = frozenset(
    {
        "profiles",
        "plugins",
        "plugin-data",
        "sessions",
        "logs",
        "cache",
        "caches",
        "cron",
        "scripts",
        "hooks",
        "backups",
    }
)
# Root configuration and known credential stores (conservative, case-insensitive).
_ROOT_PRIVATE = (
    "config.yaml",
    "state*.db*",
    "projects.db*",
    "auth.json",
    ".env*",
    "credentials*",
    ".netrc",
    "_netrc",
    ".git-credentials",
    "auth.yaml",
    "auth.yml",
    "oauth*.json",
    "token*",
    "api_key*",
    "api-key*",
    "secrets*",
)

# Any component, at any depth: VCS/private keys/virtualenvs, caches, credentials,
# key containers, lockfiles, bytecode and databases plus their SQLite sidecars.
# Applying file patterns to ancestors also prevents walking a disguised directory.
_ANY_PRIVATE = (
    ".git",
    ".hg",
    ".svn",
    ".ssh",
    ".gnupg",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".cache",
    ".env",
    ".env.*",
    "auth.json",
    "credentials.json",
    "tokens.json",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.lock",
    "*.pyc",
    "*.db",
    "*.db-wal",
    "*.db-shm",
)


def should_skip(path: str) -> str | None:
    """Return ``"safety"`` for forbidden paths, otherwise None.

    This checks names, not file contents or actual symlink state. Walkers must
    still refuse symlinks and unsupported filesystem objects independently.
    """
    parts = tuple(part.casefold() for part in _segments(path))
    if parts[0] in _ROOT_RUNTIME or any(
        fnmatchcase(parts[0], pattern) for pattern in _ROOT_PRIVATE
    ):
        return "safety"
    if any(fnmatchcase(part, pattern) for part in parts for pattern in _ANY_PRIVATE):
        return "safety"
    return None


class _RulesLoader(yaml.SafeLoader):
    """SafeLoader with duplicate keys forbidden rather than silently overwritten."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                if key in mapping:
                    raise InvalidRules("duplicate rules key")
                mapping[key] = self.construct_object(value_node, deep=deep)
            except TypeError:
                raise InvalidRules("invalid rules key") from None
        return mapping


def parse_rules(source: str | bytes) -> SelectionRules:
    """Parse a rules document without reading any user-controlled file path."""
    if not isinstance(source, (str, bytes)):
        raise InvalidRules("rules must be YAML text or bytes")
    try:
        encoded = source.encode("utf-8") if isinstance(source, str) else source
    except UnicodeError:
        raise InvalidRules("invalid YAML rules document") from None
    if len(encoded) > MAX_RULES_BYTES:
        raise InvalidRules("rules document is too large")
    try:
        depth = 0
        for event in yaml.parse(source, Loader=yaml.SafeLoader):
            if isinstance(event, (yaml.MappingStartEvent, yaml.SequenceStartEvent)):
                depth += 1
                if depth > 16:
                    raise InvalidRules("YAML nesting is too deep")
            elif isinstance(event, (yaml.MappingEndEvent, yaml.SequenceEndEvent)):
                depth -= 1
            if isinstance(event, yaml.AliasEvent) or getattr(event, "anchor", None) is not None:
                raise InvalidRules("YAML anchors and aliases are not allowed")
        document = yaml.load(source, Loader=_RulesLoader)
    except InvalidRules:
        raise
    except (
        yaml.YAMLError,
        ValueError,
        TypeError,
        OverflowError,
        RecursionError,
        KeyError,
        AttributeError,
        IndexError,
    ):
        # SafeLoader also raises built-ins for malformed bool/timestamp/numeric scalars.
        raise InvalidRules("invalid YAML rules document") from None
    required = {"version", "skip-symlinks", "include"}
    if (
        not isinstance(document, dict)
        or any(not isinstance(key, str) for key in document)
        or not required <= document.keys()
        or document.keys() - (required | {"exclude"})
    ):
        raise InvalidRules("invalid rules schema")
    return SelectionRules(
        include=document["include"],
        exclude=document.get("exclude", []),
        version=document["version"],
        skip_symlinks=document["skip-symlinks"],
    )
