"""Private bare Git cache: immutable reads and explicit successful-sync baselines.

No checkout, worktree, data commits, hooks, filters, or submodule operations.
The caller owns the cache location and must explicitly acknowledge success.
"""

import json
import os
import re
import selectors
import signal
import stat
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from hermes_agent_sync.selection import InvalidRules, SelectionRules, parse_rules, should_skip
from hermes_agent_sync.snapshot import FileSnapshot, SkippedPath, Snapshot


class GitError(RuntimeError):
    """Git data or transport cannot be trusted; never interpret failure as absence."""


_TIMEOUT = 30
_MAX_OUTPUT = 16 * 1024 * 1024


@dataclass(frozen=True)
class Baseline:
    """Last explicitly successful data revision and its observed eligible paths."""

    commit: str
    observed_paths: tuple[str, ...]


_FETCHED = "refs/profile-sync/fetched"
_BASE = "refs/profile-sync/base"
_OBSERVED = "refs/profile-sync/observed"
_MAX_BLOB = 16 * 1024 * 1024
_MAX_TOTAL = 64 * 1024 * 1024
_MAX_ENTRIES = 50_000
_MAX_TREE = 16 * 1024 * 1024


def _environment() -> dict[str, str]:
    env = dict(os.environ)
    allowed = {
        "GIT_ALLOW_PROTOCOL",
        "GIT_TERMINAL_PROMPT",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_ATTR_NOSYSTEM",
        "GIT_SSH",
        "GIT_SSH_COMMAND",
        "GIT_SSH_VARIANT",
        "GIT_ASKPASS",
        "GIT_TRACE",
        "GIT_TRACE_SETUP",
    }
    if any(key.startswith("GIT_") and key not in allowed for key in env):
        raise GitError("Inherited Git repository/configuration overrides are not allowed")
    if "GIT_CONFIG_GLOBAL" in env and env["GIT_CONFIG_GLOBAL"] not in {
        str(Path.home() / ".gitconfig"),
        os.devnull,
    }:
        raise GitError("Inherited Git configuration overrides are not allowed")
    for key in tuple(env):
        if key.startswith("GIT_") and key not in {
            "GIT_ALLOW_PROTOCOL",
            "GIT_SSH",
            "GIT_SSH_COMMAND",
            "GIT_SSH_VARIANT",
            "GIT_ASKPASS",
        }:
            del env[key]
    env.update(
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_SYSTEM=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_ATTR_NOSYSTEM="1",
        GIT_TERMINAL_PROMPT="0",
        GIT_NO_REPLACE_OBJECTS="1",
        LC_ALL="C",
    )
    # Import only credential-helper configuration, not global rewrites/includes
    # or repository overrides. Helpers remain external; no credentials are copied.
    override = os.environ.get("GIT_CONFIG_GLOBAL")
    if override is not None:
        sources = [Path(override)]
    else:
        # Git reads XDG first, then ~/.gitconfig; --global selects only one
        # file on some Git versions, so read those exact sources explicitly.
        xdg = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
        sources = [xdg / "git/config", Path.home() / ".gitconfig"]
    auth = b""
    for source in sources:
        if source.exists():
            auth += _command(
                ["config", "--file=" + str(source), "--no-includes", "--null", "--list"],
                env,
                limit=65536 - len(auth),
            )
    entries = []
    for record in auth.split(b"\0")[:-1]:
        try:
            key, value = record.decode("utf-8").split("\n", 1)
        except (ValueError, UnicodeError):
            raise GitError("Invalid authentication configuration") from None
        if key.startswith("credential.") and key.rsplit(".", 1)[-1] in {"helper", "usehttppath"}:
            entries.append((key, value))
    env["GIT_CONFIG_COUNT"] = str(len(entries))
    for index, (key, value) in enumerate(entries):
        env[f"GIT_CONFIG_KEY_{index}"] = key
        env[f"GIT_CONFIG_VALUE_{index}"] = value
    return env


def _command(
    args: list[str], env: dict[str, str], *, limit: int = _MAX_OUTPUT, data: bytes | None = None
) -> bytes:
    """Bound wall time and both streams; never expose subprocess diagnostics/URLs."""
    command = [
        "git",
        "--no-replace-objects",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "tag.gpgsign=false",
        *args,
    ]
    try:
        process = subprocess.Popen(
            command,
            env=env,
            stdin=subprocess.PIPE if data is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        selector = selectors.DefaultSelector()
        succeeded = False
        try:
            pending = memoryview(data or b"")
            if data is not None:
                if pending:
                    selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
                else:
                    process.stdin.close()
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            result = bytearray()
            sizes = {"stdout": 0, "stderr": 0}
            deadline = time.monotonic() + _TIMEOUT
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise GitError("Git operation timed out")
                for key, _ in selector.select(min(remaining, 0.2)):
                    if key.data == "stdin":
                        try:
                            pending = pending[os.write(key.fd, pending[:65536]) :]
                        except BrokenPipeError:
                            pending = pending[len(pending) :]
                        if not pending:
                            selector.unregister(key.fileobj)
                            process.stdin.close()
                        continue
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    sizes[key.data] += len(chunk)
                    if sizes[key.data] > (limit if key.data == "stdout" else 65536):
                        raise GitError("Git output size limit exceeded")
                    if key.data == "stdout":
                        result.extend(chunk)
            try:
                code = process.wait(timeout=max(0.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                raise GitError("Git operation timed out") from None
            if code:
                raise GitError("Git operation failed (diagnostics withheld)")
            succeeded = True
            return bytes(result)
        finally:
            selector.close()
            if not succeeded:
                # The Git parent may already have exited while helpers hold pipes.
                # start_new_session makes its PID a group ID distinct from ours.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass  # Do not turn bounded operation cleanup into an unbounded wait.
    except OSError:
        raise GitError("Git operation could not be started") from None


def _check_path(path: Path) -> None:
    if ".." in path.parts:
        raise GitError("Cache path cannot contain parent traversal")
    for part in reversed((path, *path.parents)):
        try:
            mode = part.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise GitError("Cache path and ancestors must be non-symlink directories")


def _destination(url: str, allow_file: bool) -> None:
    from urllib.parse import urlsplit

    if not isinstance(url, str) or any(ord(char) < 33 or ord(char) == 127 for char in url):
        raise GitError("Invalid repository destination")
    try:
        parsed = urlsplit(url)
        if parsed.scheme == "https":
            valid = bool(parsed.hostname) and parsed.username is None and parsed.password is None
            valid = valid and "?" not in url and "#" not in url and not parsed.netloc.endswith(":")
            _ = parsed.port
        elif parsed.scheme == "ssh":
            valid = parsed.username == "git" and parsed.password is None and bool(parsed.hostname)
            valid = valid and "?" not in url and "#" not in url and bool(parsed.path)
            valid = valid and not parsed.hostname.startswith("-")
            _ = parsed.port
        elif re.fullmatch(r"git@[A-Za-z0-9][A-Za-z0-9.-]*:[A-Za-z0-9_./~-]+", url):
            valid = True
        else:
            valid = allow_file and Path(url).is_absolute() and not parsed.scheme
    except ValueError:
        valid = False
    if not valid:
        raise GitError("Repository destination must use approved HTTPS or Git SSH transport")


class GitStore:
    """A caller-scoped private bare cache, bound permanently to one destination."""

    def __init__(self, path: Path, repo_url: str, branch: str, *, allow_file: bool = False):
        _destination(repo_url, allow_file)
        self._env = _environment()
        if not isinstance(branch, str) or not branch or branch.startswith("-"):
            raise GitError("Invalid branch")
        _command(["check-ref-format", "refs/heads/" + branch], self._env)
        if branch.startswith("refs/") or branch == "HEAD":
            raise GitError("Branch must name a branch within heads")
        self.path = Path(path).absolute()
        self.branch, self.repo_url = branch, repo_url
        _check_path(self.path)
        if self.path.exists():
            self._validate_config()
            if self._git("rev-parse", "--is-bare-repository").strip() != b"true":
                raise GitError("Cache must be a private bare repository")
            if (
                self._git("config", "--local", "--get", "profile-sync.url").decode().strip()
                != repo_url
            ):
                raise GitError("Cache destination changed")
            if (
                self._git("config", "--local", "--get", "profile-sync.branch").decode().strip()
                != branch
            ):
                raise GitError("Cache destination changed")
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.mkdir(mode=0o700)
            self._git("init", "--bare", "--template=", str(self.path))
            self._git("config", "--local", "profile-sync.url", repo_url)
            self._git("config", "--local", "profile-sync.branch", branch)

    def _git(self, *args: str, limit: int = _MAX_OUTPUT, data: bytes | None = None) -> bytes:
        _check_path(self.path)
        if (self.path / "config").exists():
            self._validate_config()
        return _command(["--git-dir=" + str(self.path), *args], self._env, limit=limit, data=data)

    def _validate_config(self) -> None:
        """Only our own inert bare config may influence object/transport commands."""
        config = self.path / "config"
        try:
            info = config.lstat()
        except OSError:
            raise GitError("Cache configuration is missing") from None
        if not stat.S_ISREG(info.st_mode) or info.st_size > 65536:
            raise GitError("Unsafe cache configuration")
        raw = _command(
            ["config", "--file=" + str(config), "--no-includes", "--null", "--list"],
            self._env,
            limit=65536,
        )
        allowed = {
            "core.repositoryformatversion",
            "core.filemode",
            "core.bare",
            "core.logallrefupdates",
            "extensions.objectformat",
            "profile-sync.url",
            "profile-sync.branch",
        }
        seen = set()
        for item in raw.split(b"\0")[:-1]:
            try:
                key, value = item.decode("utf-8").split("\n", 1)
            except (ValueError, UnicodeError):
                raise GitError("Invalid cache configuration") from None
            if key not in allowed or key in seen:
                raise GitError("Unexpected cache configuration")
            seen.add(key)
            if (key == "profile-sync.url" and value != self.repo_url) or (
                key == "profile-sync.branch" and value != self.branch
            ):
                raise GitError("Cache destination changed")
            if (
                (key == "core.bare" and value != "true")
                or (key == "core.repositoryformatversion" and value not in {"0", "1"})
                or (key == "extensions.objectformat" and value not in {"sha1", "sha256"})
            ):
                raise GitError("Unsupported cache configuration")

    def fetch(self) -> str:
        """Fetch one heads branch without creating or advancing a successful baseline."""
        baseline = self.baseline()
        # A forced *local fetch ref* detects remote rewrites; never a push or base update.
        self._git(
            "-c",
            "fetch.fsckObjects=true",
            "fetch",
            "--no-tags",
            "--no-write-fetch-head",
            "--no-recurse-submodules",
            "--",
            self.repo_url,
            "+refs/heads/" + self.branch + ":" + _FETCHED,
        )
        commit = self._commit(self._git("rev-parse", "--verify", _FETCHED).decode().strip())
        if baseline:
            try:
                self._git("merge-base", "--is-ancestor", baseline.commit, commit)
            except GitError:
                raise GitError(
                    "Remote no longer descends from baseline; rebootstrap required"
                ) from None
        return commit

    def _commit(self, commit: str) -> str:
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
            raise GitError("Expected a full lowercase commit object ID, not a ref")
        resolved = self._git("rev-parse", "--verify", commit + "^{commit}").decode().strip()
        if resolved != commit:
            raise GitError("Object ID does not name a commit")
        return commit

    def _blob(self, oid: str, limit: int) -> bytes:
        if self._git("cat-file", "-t", oid).strip() != b"blob":
            raise GitError("Expected a regular Git blob")
        try:
            size = int(self._git("cat-file", "-s", oid))
        except ValueError:
            raise GitError("Invalid Git object size") from None
        if size < 0 or size > limit:
            raise GitError("Git blob size limit exceeded")
        data = self._git("cat-file", "blob", oid, limit=limit)
        if len(data) != size:
            raise GitError("Git blob size mismatch")
        actual = self._git("hash-object", "--stdin", data=data).decode().strip()
        if actual != oid:
            raise GitError("Git blob integrity check failed")
        return data

    def _entries(self, commit: str, *, root: bool = False) -> list[tuple[str, str, str, str]]:
        self._commit(commit)
        args = ["ls-tree", "-z", "--full-tree", "--abbrev=64"]
        if not root:
            args += ["-r", "-t"]
        raw = self._git(*args, commit, limit=_MAX_TREE)
        records = raw.split(b"\0")
        if records[-1] or len(records) - 1 > _MAX_ENTRIES:
            raise GitError("Invalid or oversized Git tree")
        entries = []
        seen = set()
        for record in records[:-1]:
            try:
                header, name = record.split(b"\t", 1)
                mode, kind, oid = header.decode("ascii").split(" ")
                path = name.decode("utf-8", errors="strict")
                should_skip(path)  # canonical path validation, including excluded names
                if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", oid):
                    raise ValueError
            except (ValueError, UnicodeError, InvalidRules):
                raise GitError("Invalid or unsafe Git tree entry") from None
            if path in seen:
                raise GitError("Duplicate Git tree path")
            seen.add(path)
            entries.append((mode, kind, oid, path))
        return entries

    def read_rules(self, commit: str) -> SelectionRules:
        """Read the bounded regular root control blob, never a filesystem checkout."""
        for mode, kind, oid, path in self._entries(commit, root=True):
            if path == "sync.yaml":
                if mode not in {"100644", "100755"} or kind != "blob":
                    raise GitError("sync.yaml must be a regular Git blob")
                try:
                    return parse_rules(self._blob(oid, 65536))
                except InvalidRules:
                    raise GitError("Invalid committed selection rules") from None
        raise GitError("Committed sync.yaml is missing")

    def snapshot(
        self, commit: str, rules: SelectionRules, *, approved_rules: SelectionRules
    ) -> Snapshot:
        """Collect only approved eligible blobs; skipped subtrees are never absence."""
        if not isinstance(rules, SelectionRules) or rules != approved_rules:
            raise GitError("Selection rules require local approval")
        files, skipped, directories = [], [], [""]
        total = 0
        blocked: set[str] = set()
        for mode, kind, oid, path in self._entries(commit):
            parts = path.split("/")
            if any("/".join(parts[:index]) in blocked for index in range(1, len(parts))):
                continue
            reason = "control" if path == "sync.yaml" else should_skip(path)
            if not reason and not rules.matches(path) and not rules.could_include(path):
                reason = "not-selected"
            if reason:
                skipped.append(SkippedPath(path, reason))
                blocked.add(path)
            elif mode == "040000" and kind == "tree":
                if len(path.split("/")) > 64:
                    raise GitError("Git tree depth limit exceeded")
                directories.append(path)
            elif mode == "120000" and kind == "blob":
                skipped.append(SkippedPath(path, "symlink"))
                blocked.add(path)
            elif mode == "160000" or kind == "commit":
                raise GitError("Selected Git submodules are not supported")
            elif mode not in {"100644", "100755"} or kind != "blob":
                raise GitError("Unsupported selected Git object")
            elif rules.matches(path):
                data = self._blob(oid, min(_MAX_BLOB, _MAX_TOTAL - total))
                files.append(FileSnapshot(path, data, int(mode, 8)))
                total += len(data)
            else:
                skipped.append(SkippedPath(path, "not-selected"))
        return Snapshot(rules, tuple(files), tuple(skipped), tuple(directories))

    def _refs(self) -> dict[str, str]:
        output = self._git(
            "for-each-ref", "--format=%(refname) %(objectname)", _BASE, _OBSERVED, _FETCHED
        )
        refs = {}
        try:
            for line in output.decode("ascii").splitlines():
                ref, oid = line.split(" ")
                if ref not in {_BASE, _OBSERVED, _FETCHED}:
                    raise ValueError
                if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", oid):
                    raise ValueError
                refs[ref] = oid
        except (ValueError, UnicodeError):
            raise GitError("Invalid cache reference state") from None
        for ref in (_BASE, _OBSERVED, _FETCHED):
            target = self.path / ref
            if os.path.lexists(target):
                if ref not in refs:
                    raise GitError("Corrupt cache reference")
                info = target.lstat()
                if not stat.S_ISREG(info.st_mode) or info.st_size > 256:
                    raise GitError("Unsafe cache reference")
                try:
                    raw = target.read_bytes()
                except OSError:
                    raise GitError("Unreadable cache reference") from None
                if raw != (refs[ref] + "\n").encode("ascii"):
                    raise GitError("Symbolic or corrupt cache reference")
        return refs

    def _observed(self, commit: str, paths: tuple[str, ...]) -> None:
        if not isinstance(paths, tuple) or len(paths) > _MAX_ENTRIES:
            raise GitError("Observed paths must be a bounded tuple")
        seen = set()
        for path in paths:
            try:
                forbidden = should_skip(path)
            except InvalidRules:
                raise GitError("Invalid observed path") from None
            if forbidden or path == "sync.yaml" or path in seen:
                raise GitError("Unsafe or duplicate observed path")
            seen.add(path)
        rules = self.read_rules(commit)
        snapshot = self.snapshot(commit, rules, approved_rules=rules)
        if any(snapshot.status(path) == "skipped" for path in paths):
            raise GitError("Observed paths include ineligible or unobserved paths")

    def _baseline(self, refs: dict[str, str]) -> Baseline | None:
        if _BASE not in refs and _OBSERVED not in refs:
            return None
        if _BASE not in refs or _OBSERVED not in refs:
            raise GitError("Incomplete baseline metadata; rebootstrap required")
        commit = self._commit(refs[_BASE])
        try:

            def unique(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError("duplicate metadata key")
                    result[key] = value
                return result

            document = json.loads(
                self._blob(refs[_OBSERVED], _MAX_OUTPUT), object_pairs_hook=unique
            )
            if (
                not isinstance(document, dict)
                or set(document) != {"version", "commit", "observed_paths"}
                or type(document["version"]) is not int
                or document["version"] != 1
                or document["commit"] != commit
                or not isinstance(document["observed_paths"], list)
            ):
                raise ValueError
            paths = tuple(document["observed_paths"])
            self._observed(commit, paths)
        except (ValueError, TypeError, RecursionError, UnicodeError):
            raise GitError("Corrupt or mismatched baseline metadata") from None
        return Baseline(commit, paths)

    def baseline(self) -> Baseline | None:
        """Read both pinned refs; incomplete or corrupt state fails closed."""
        return self._baseline(self._refs())

    def acknowledge(
        self, commit: str, observed_paths: tuple[str, ...], *, expected_commit: str | None
    ) -> None:
        """Atomically acknowledge success; previews must never call this method.

        Eligible missing paths may be recorded; this is not deletion consent.
        Both refs are compared and updated in one Git reference transaction.
        """
        commit = self._commit(commit)
        if expected_commit is not None:
            self._commit(expected_commit)
        refs = self._refs()
        baseline = self._baseline(refs)
        if (baseline.commit if baseline else None) != expected_commit:
            raise GitError("Baseline changed; retry comparison")
        if _FETCHED not in refs:
            raise GitError("Fetch the destination before acknowledging")
        self._git("merge-base", "--is-ancestor", commit, refs[_FETCHED])
        if baseline:
            self._git("merge-base", "--is-ancestor", baseline.commit, commit)
        self._observed(commit, observed_paths)
        metadata = json.dumps(
            {"version": 1, "commit": commit, "observed_paths": observed_paths},
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        if len(metadata) > _MAX_OUTPUT:
            raise GitError("Baseline metadata size limit exceeded")
        oid = self._git("hash-object", "-w", "--stdin", data=metadata).decode().strip()
        zero = "0" * len(commit)
        transaction = (
            f"start\nupdate {_BASE} {commit} {refs.get(_BASE, zero)}\n"
            f"update {_OBSERVED} {oid} {refs.get(_OBSERVED, zero)}\n"
            f"verify {_FETCHED} {refs[_FETCHED]}\nprepare\ncommit\n"
        )
        self._git("update-ref", "--stdin", data=transaction.encode("ascii"))
