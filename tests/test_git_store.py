"""Real Git transport and immutable object storage, using disposable repositories."""

import pytest


@pytest.fixture
def shared(git_repositories, git_run):
    left = git_repositories["left"]
    (left / "sync.yaml").write_text(
        'version: 1\nskip-symlinks: true\ninclude: ["**"]\nexclude: []\n'
    )
    (left / "SOUL.md").write_bytes(b"shared soul\n")
    git_run(left, "add", ".")
    git_run(left, "commit", "-m", "Shared profile")
    git_run(left, "push", "origin", "master")
    return git_repositories


def store_for(tmp_path, shared):
    from hermes_agent_sync.git_store import GitStore

    return GitStore(tmp_path / "cache.git", str(shared["remote"]), "master", allow_file=True)


def test_fetch_pins_exact_remote_commit_without_checkout(tmp_path, shared, git_run):
    store = store_for(tmp_path, shared)
    expected = git_run(shared["left"], "rev-parse", "HEAD")
    assert store.fetch() == expected
    assert git_run(store.path, "rev-parse", "refs/profile-sync/fetched") == expected
    assert git_run(store.path, "rev-parse", "--is-bare-repository") == "true"
    assert not (store.path / "SOUL.md").exists()


def test_committed_rules_and_snapshot_preserve_binary_executable_and_scope(
    tmp_path, shared, git_run
):
    from hermes_agent_sync.selection import SelectionRules

    left = shared["left"]
    (left / "payload").write_bytes(b"\x00\xffbinary\r\n")
    (left / "payload").chmod(0o755)
    (left / "notes").mkdir()
    (left / "notes" / "ok.md").write_text("nested")
    (left / "secrets.env").write_text("must not read")
    git_run(left, "add", ".")
    git_run(left, "commit", "-m", "Binary executable")
    git_run(left, "push", "origin", "master")
    store = store_for(tmp_path, shared)
    commit = store.fetch()
    rules = store.read_rules(commit)
    assert rules == SelectionRules(("**",))
    snapshot = store.snapshot(commit, rules, approved_rules=rules)
    files = {item.path: item for item in snapshot.files}
    assert set(files) == {"SOUL.md", "payload", "notes/ok.md"}
    assert files["payload"].data == b"\x00\xffbinary\r\n"
    assert files["payload"].mode == 0o100755
    assert files["SOUL.md"].mode == 0o100644
    assert snapshot.status("sync.yaml") == "skipped"
    assert snapshot.status("secrets.env") == "skipped"
    assert snapshot.status("absent.md") == "missing"
    assert "notes" in snapshot.directories


def test_acknowledged_baseline_is_pinned_persistent_and_separate_from_fetch(
    tmp_path, shared, git_run
):
    from hermes_agent_sync.git_store import Baseline, GitStore

    store = store_for(tmp_path, shared)
    first = store.fetch()
    assert store.baseline() is None
    store.acknowledge(first, ("SOUL.md", "absent.md"), expected_commit=None)
    assert store.baseline() == Baseline(first, ("SOUL.md", "absent.md"))
    assert git_run(store.path, "rev-parse", "refs/profile-sync/base") == first
    reopened = GitStore(store.path, str(shared["remote"]), "master", allow_file=True)
    assert reopened.baseline() == store.baseline()
    (shared["left"] / "SOUL.md").write_text("new content")
    git_run(shared["left"], "commit", "-am", "Next data")
    git_run(shared["left"], "push", "origin", "master")
    second = reopened.fetch()
    assert second != first
    assert reopened.baseline() == Baseline(first, ("SOUL.md", "absent.md"))
    reopened.acknowledge(second, ("SOUL.md",), expected_commit=first)
    git_run(store.path, "gc", "--prune=now")
    assert store.baseline() == Baseline(second, ("SOUL.md",))


def test_remote_history_rewrite_requires_rebootstrap_and_preserves_baseline(
    tmp_path, shared, git_run
):
    from hermes_agent_sync.git_store import GitError

    store = store_for(tmp_path, shared)
    original = store.fetch()
    store.acknowledge(original, ("SOUL.md",), expected_commit=None)
    left = shared["left"]
    git_run(left, "checkout", "--orphan", "replacement")
    git_run(left, "commit", "-m", "Unrelated shared history")
    git_run(left, "push", "--force", "origin", "HEAD:master")
    with pytest.raises(GitError, match="rebootstrap"):
        store.fetch()
    assert store.baseline().commit == original


@pytest.mark.parametrize(
    "key,value",
    [
        ("url.ext::sh -c forbidden.insteadOf", "/"),
        ("core.sshCommand", "forbidden-command"),
        ("include.path", "/untrusted/config"),
    ],
)
def test_existing_cache_rejects_unexpected_local_configuration(
    tmp_path, shared, git_run, key, value
):
    from hermes_agent_sync.git_store import GitError, GitStore

    store = store_for(tmp_path, shared)
    git_run(store.path, "config", "--local", key, value)
    with pytest.raises(GitError, match="configuration"):
        GitStore(store.path, str(shared["remote"]), "master", allow_file=True)


def test_fetch_rejects_internal_config_destination_changes_after_open(tmp_path, shared, git_run):
    from hermes_agent_sync.git_store import GitError

    store = store_for(tmp_path, shared)
    git_run(store.path, "config", "--local", "profile-sync.url", "/different.git")
    with pytest.raises(GitError, match="destination"):
        store.fetch()


def test_broken_baseline_reference_is_not_first_use(tmp_path, shared):
    from hermes_agent_sync.git_store import GitError

    store = store_for(tmp_path, shared)
    store.fetch()
    target = store.path / "refs/profile-sync/base"
    target.write_text("not-an-object\n")
    with pytest.raises(GitError):
        store.baseline()


def test_global_auth_helper_survives_without_global_url_rewrites(tmp_path, shared, git_run):
    store = store_for(tmp_path, shared)
    git_run(shared["left"], "config", "--global", "credential.helper", "test-helper")
    git_run(shared["left"], "config", "--global", "url.ext::evil.insteadOf", str(shared["remote"]))
    from hermes_agent_sync.git_store import GitStore

    reopened = GitStore(store.path, str(shared["remote"]), "master", allow_file=True)
    assert reopened._git("config", "--get-all", "credential.helper").strip() == b"test-helper"
    assert reopened.fetch()


def test_https_empty_userinfo_is_rejected_without_echoing_destination(tmp_path):
    from hermes_agent_sync.git_store import GitError, GitStore

    with pytest.raises(GitError) as caught:
        GitStore(tmp_path / "cache", "https://@example.invalid/repo", "master")
    assert "example.invalid" not in str(caught.value)


def test_inherited_global_config_redirection_is_rejected(tmp_path, monkeypatch):
    from hermes_agent_sync.git_store import GitError, GitStore

    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "untrusted-config"))
    with pytest.raises(GitError, match="overrides"):
        GitStore(tmp_path / "cache", "https://example.invalid/repo", "master")


def test_duplicate_baseline_metadata_keys_fail_closed(tmp_path, shared, git_run):
    from hermes_agent_sync.git_store import GitError

    store = store_for(tmp_path, shared)
    commit = store.fetch()
    store.acknowledge(commit, ("SOUL.md",), expected_commit=None)
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        '{"version":0,"version":1,"commit":"' + commit + '","observed_paths":["SOUL.md"]}'
    )
    oid = git_run(store.path, "hash-object", "-w", str(metadata))
    git_run(store.path, "update-ref", "refs/profile-sync/observed", oid)
    with pytest.raises(GitError, match="metadata"):
        store.baseline()


@pytest.mark.parametrize(
    "url",
    [
        "http://example.invalid/repo",
        "ext::sh -c evil",
        "file:///tmp/repo",
        "/tmp/repo",
        "https://user:secret@example.invalid/repo",
        "https://example.invalid/repo?token=secret",
        "https://example.invalid/repo#secret",
        "ssh://root@example.invalid/repo",
        "ssh://git:secret@example.invalid/repo",
        "git@-oProxyCommand=evil:repo",
        "--upload-pack=evil",
        "https://example.invalid/\nsecret",
        "",
        "git@example.invalid:",
    ],
)
def test_invalid_destinations_fail_without_leaking_input(tmp_path, url):
    from hermes_agent_sync.git_store import GitError, GitStore

    with pytest.raises(GitError) as caught:
        GitStore(tmp_path / "cache", url, "master")
    assert "secret" not in str(caught.value)
    assert "example.invalid" not in str(caught.value)
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize(
    "branch",
    ["", "-x", "HEAD", "refs/heads/master", "../x", "x:y", "a b", "x..y", "x.lock", "x@{y", "a\nb"],
)
def test_invalid_branches_are_rejected(tmp_path, branch):
    from hermes_agent_sync.git_store import GitError, GitStore

    with pytest.raises(GitError):
        GitStore(tmp_path / "cache", "https://example.invalid/repo", branch)
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize(
    "url",
    [
        "https://example.invalid/private.git",
        "ssh://git@example.invalid/team/repo.git",
        "git@example.invalid:team/repo.git",
    ],
)
def test_valid_production_destinations_construct_without_network(tmp_path, url):
    from hermes_agent_sync.git_store import GitStore

    assert GitStore(tmp_path / "cache", url, "team/main").branch == "team/main"


@pytest.mark.parametrize(
    "value",
    [
        "HEAD",
        "master",
        "refs/heads/master",
        "-x",
        "a" * 39,
        "A" * 40,
        "a" * 41,
        "a" * 40 + "^{commit}",
        "0" * 40,
    ],
)
def test_public_commit_inputs_must_be_existing_full_commit_ids(tmp_path, shared, value):
    from hermes_agent_sync.git_store import GitError
    from hermes_agent_sync.selection import SelectionRules

    store = store_for(tmp_path, shared)
    store.fetch()
    rules = SelectionRules(("**",))
    with pytest.raises(GitError):
        store.read_rules(value)
    with pytest.raises(GitError):
        store.snapshot(value, rules, approved_rules=rules)
    with pytest.raises(GitError):
        store.acknowledge(value, (), expected_commit=None)


@pytest.mark.parametrize(
    "name",
    [
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_CONFIG",
        "GIT_CONFIG_COUNT",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_NAMESPACE",
        "GIT_REPLACE_REF_BASE",
        "GIT_CONFIG_SYSTEM",
    ],
)
def test_inherited_repository_overrides_are_rejected(tmp_path, monkeypatch, name):
    from hermes_agent_sync.git_store import GitError, GitStore

    monkeypatch.setenv(name, "untrusted")
    with pytest.raises(GitError, match="overrides"):
        GitStore(tmp_path / "cache", "https://example.invalid/repo", "master")


def test_cache_ownership_destination_and_symlink_checks(tmp_path, shared, git_run):
    from hermes_agent_sync.git_store import GitError, GitStore

    remote = str(shared["remote"])
    store = store_for(tmp_path, shared)
    for path, url, branch in [
        (shared["left"], remote, "master"),
        (shared["remote"], remote, "master"),
        (store.path, remote, "other"),
        (store.path, str(tmp_path / "other.git"), "master"),
    ]:
        with pytest.raises(GitError):
            GitStore(path, url, branch, allow_file=True)
    linked = tmp_path / "linked"
    linked.symlink_to(store.path, target_is_directory=True)
    with pytest.raises(GitError, match="symlink"):
        GitStore(linked, remote, "master", allow_file=True)
    parent = tmp_path / "linked-parent"
    parent.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(GitError, match="symlink"):
        GitStore(parent / "new.git", remote, "master", allow_file=True)
    assert not (tmp_path / "new.git").exists()


def test_approval_must_match_exactly_even_when_selection_is_equivalent(tmp_path, shared):
    from hermes_agent_sync.git_store import GitError
    from hermes_agent_sync.selection import SelectionRules

    store = store_for(tmp_path, shared)
    commit = store.fetch()
    with pytest.raises(GitError, match="approval"):
        store.snapshot(commit, SelectionRules(("**",)), approved_rules=SelectionRules(("**", "**")))


def test_symlinks_are_skipped_with_descendant_semantics(tmp_path, shared, git_run):
    left = shared["left"]
    (left / "shortcut").symlink_to("/outside/private")
    git_run(left, "add", "shortcut")
    git_run(left, "commit", "-m", "Link")
    git_run(left, "push", "origin", "master")
    store = store_for(tmp_path, shared)
    commit = store.fetch()
    rules = store.read_rules(commit)
    snapshot = store.snapshot(commit, rules, approved_rules=rules)
    assert snapshot.status("shortcut") == "skipped"
    assert snapshot.status("shortcut/nested") == "skipped"
    assert any(item.path == "shortcut" and item.reason == "symlink" for item in snapshot.skipped)


def test_selected_gitlinks_fail_but_excluded_gitlinks_are_skipped(tmp_path, shared, git_run):
    from hermes_agent_sync.git_store import GitError
    from hermes_agent_sync.selection import SelectionRules

    left = shared["left"]
    commit = git_run(left, "rev-parse", "HEAD")
    git_run(left, "update-index", "--add", "--cacheinfo", "160000," + commit + ",linked")
    git_run(left, "commit", "-m", "Gitlink without submodule execution")
    git_run(left, "push", "origin", "master")
    store = store_for(tmp_path, shared)
    commit = store.fetch()
    rules = store.read_rules(commit)
    with pytest.raises(GitError, match="submodules"):
        store.snapshot(commit, rules, approved_rules=rules)
    restricted = SelectionRules(("**",), ("linked/**",))
    snapshot = store.snapshot(commit, restricted, approved_rules=restricted)
    assert snapshot.status("linked/nested") == "skipped"


def test_excluded_and_safety_blobs_are_never_read(tmp_path, shared, git_run, monkeypatch):
    from hermes_agent_sync.selection import SelectionRules

    left = shared["left"]
    (left / "private").mkdir()
    (left / "private" / "hidden").write_bytes(b"excluded blob")
    (left / ".env").write_bytes(b"safety blob")
    git_run(left, "add", ".")
    git_run(left, "commit", "-m", "Excluded values")
    git_run(left, "push", "origin", "master")
    store = store_for(tmp_path, shared)
    commit = store.fetch()
    forbidden = {
        git_run(store.path, "rev-parse", commit + ":" + name) for name in ("private/hidden", ".env")
    }
    original = store._blob

    def guarded(oid, limit):
        assert oid not in forbidden, "Excluded secret must never be read"
        return original(oid, limit)

    monkeypatch.setattr(store, "_blob", guarded)
    rules = SelectionRules(("**",), ("private/**",))
    snapshot = store.snapshot(commit, rules, approved_rules=rules)
    assert [item.path for item in snapshot.files] == ["SOUL.md"]
    assert snapshot.status("private/hidden") == "skipped"
    assert snapshot.status("private/other") == "skipped"
    assert snapshot.status(".env") == "skipped"


@pytest.mark.parametrize("control", [None, b"version: 1\ninclude: []\n", b"[]", b"\xff"])
def test_missing_or_invalid_rules_block_reads(tmp_path, git_repositories, git_run, control):
    from hermes_agent_sync.git_store import GitError

    left = git_repositories["left"]
    if control is not None:
        (left / "sync.yaml").write_bytes(control)
        git_run(left, "add", ".")
        git_run(left, "commit", "-m", "Bad control")
        git_run(left, "push", "origin", "master")
    store = store_for(tmp_path, git_repositories)
    commit = store.fetch()
    with pytest.raises(GitError):
        store.read_rules(commit)


def test_symlink_control_rules_cannot_be_followed(tmp_path, git_repositories, git_run):
    from hermes_agent_sync.git_store import GitError

    left = git_repositories["left"]
    (left / "sync.yaml").symlink_to("/outside/rules")
    git_run(left, "add", ".")
    git_run(left, "commit", "-m", "Symlink rules")
    git_run(left, "push", "origin", "master")
    store = store_for(tmp_path, git_repositories)
    with pytest.raises(GitError, match="regular"):
        store.read_rules(store.fetch())


@pytest.mark.parametrize(
    "paths",
    [
        ("../bad",),
        ("/absolute",),
        ("a//b",),
        ("a\\b",),
        (".env",),
        ("sync.yaml",),
        ("SOUL.md", "SOUL.md"),
        ("SOUL.md/child",),
        ("profiles/work/SOUL.md",),
        (3,),
        ["SOUL.md"],
    ],
)
def test_ack_rejects_invalid_unobserved_or_unsafe_paths(tmp_path, shared, paths):
    from hermes_agent_sync.git_store import GitError

    store = store_for(tmp_path, shared)
    commit = store.fetch()
    with pytest.raises(GitError):
        store.acknowledge(commit, paths, expected_commit=None)
    assert store.baseline() is None


def test_expected_commit_prevents_lost_updates(tmp_path, shared, git_run):
    from hermes_agent_sync.git_store import GitError, GitStore

    first = store_for(tmp_path, shared)
    commit = first.fetch()
    second = GitStore(first.path, str(shared["remote"]), "master", allow_file=True)
    second.acknowledge(commit, ("SOUL.md",), expected_commit=None)
    metadata = git_run(first.path, "rev-parse", "refs/profile-sync/observed")
    with pytest.raises(GitError, match="changed"):
        first.acknowledge(commit, (), expected_commit=None)
    assert git_run(first.path, "rev-parse", "refs/profile-sync/observed") == metadata
    assert first.baseline().observed_paths == ("SOUL.md",)


@pytest.mark.parametrize(
    "mutation", ["missing-metadata", "missing-base", "invalid-json", "mismatch"]
)
def test_corrupt_or_mismatched_baseline_fails_closed(tmp_path, shared, git_run, mutation):
    from hermes_agent_sync.git_store import GitError

    store = store_for(tmp_path, shared)
    commit = store.fetch()
    store.acknowledge(commit, ("SOUL.md",), expected_commit=None)
    if mutation.startswith("missing"):
        ref = "base" if mutation == "missing-base" else "observed"
        git_run(store.path, "update-ref", "-d", "refs/profile-sync/" + ref)
    else:
        metadata = tmp_path / "bad-metadata"
        metadata.write_text(
            "garbage"
            if mutation == "invalid-json"
            else '{"version":1,"commit":"' + "0" * 40 + '","observed_paths":[]}'
        )
        oid = git_run(store.path, "hash-object", "-w", str(metadata))
        git_run(store.path, "update-ref", "refs/profile-sync/observed", oid)
    with pytest.raises(GitError):
        store.baseline()
    with pytest.raises(GitError):
        store.fetch()


def test_independent_profile_caches_and_branches(tmp_path, shared, git_run):
    from hermes_agent_sync.git_store import GitStore

    left = shared["left"]
    master = git_run(left, "rev-parse", "HEAD")
    git_run(left, "checkout", "-b", "work")
    (left / "SOUL.md").write_text("work")
    git_run(left, "commit", "-am", "Work branch")
    git_run(left, "push", "origin", "work")
    personal = store_for(tmp_path, shared)
    work = GitStore(tmp_path / "work.git", str(shared["remote"]), "work", allow_file=True)
    assert personal.fetch() == master
    work_commit = work.fetch()
    assert work_commit != master
    work.acknowledge(work_commit, ("SOUL.md",), expected_commit=None)
    assert personal.baseline() is None
    assert work.baseline().commit == work_commit


def test_git_tree_depth_is_bounded(tmp_path, shared, git_run):
    from hermes_agent_sync.git_store import GitError

    left = shared["left"]
    target = left.joinpath(*(["d"] * 65), "note.md")
    target.parent.mkdir(parents=True)
    target.write_text("too deep")
    git_run(left, "add", ".")
    git_run(left, "commit", "-m", "Excessive tree depth")
    git_run(left, "push", "origin", "master")
    store = store_for(tmp_path, shared)
    commit = store.fetch()
    rules = store.read_rules(commit)
    with pytest.raises(GitError, match="depth"):
        store.snapshot(commit, rules, approved_rules=rules)


@pytest.mark.parametrize(
    "limit,value", [("_MAX_BLOB", 2), ("_MAX_TOTAL", 2), ("_MAX_ENTRIES", 1), ("_MAX_TREE", 4)]
)
def test_snapshot_resource_limits_fail_closed(tmp_path, shared, monkeypatch, limit, value):
    import hermes_agent_sync.git_store as storage

    store = store_for(tmp_path, shared)
    commit = store.fetch()
    rules = store.read_rules(commit)
    monkeypatch.setattr(storage, limit, value)
    with pytest.raises(storage.GitError):
        store.snapshot(commit, rules, approved_rules=rules)


def test_default_blob_limit_is_enforced_with_real_oversized_git_object(tmp_path, shared, git_run):
    from hermes_agent_sync.git_store import GitError

    left = shared["left"]
    with (left / "oversized").open("wb") as stream:
        stream.truncate(16 * 1024 * 1024 + 1)
    git_run(left, "add", "oversized")
    git_run(left, "commit", "-m", "Oversized blob")
    git_run(left, "push", "origin", "master")
    store = store_for(tmp_path, shared)
    commit = store.fetch()
    rules = store.read_rules(commit)
    with pytest.raises(GitError, match="size limit"):
        store.snapshot(commit, rules, approved_rules=rules)


def test_selected_unreadable_blob_is_not_absence(tmp_path, shared, git_run):
    from hermes_agent_sync.git_store import GitError

    store = store_for(tmp_path, shared)
    commit = store.fetch()
    rules = store.read_rules(commit)
    oid = git_run(store.path, "rev-parse", commit + ":SOUL.md")
    # Preserve the commit and trees as loose objects, then remove the pack.
    objects = git_run(store.path, "rev-list", "--objects", commit).splitlines()
    preserved = []
    for line in objects:
        object_id = line.split(" ", 1)[0]
        if object_id != oid:
            kind = store._git("cat-file", "-t", object_id).decode().strip()
            preserved.append((kind, store._git("cat-file", kind, object_id)))
    for pack in (store.path / "objects/pack").iterdir():
        pack.unlink()
    for kind, data in preserved:
        store._git("hash-object", "-w", "-t", kind, "--stdin", data=data)
    corrupt = store.path / "objects" / oid[:2] / oid[2:]
    corrupt.parent.mkdir(exist_ok=True)
    corrupt.write_bytes(b"corrupt selected loose object")
    with pytest.raises(GitError):
        store.snapshot(commit, rules, approved_rules=rules)


def test_unresolvable_baseline_reference_fails_closed(tmp_path, shared):
    from hermes_agent_sync.git_store import GitError

    store = store_for(tmp_path, shared)
    commit = store.fetch()
    store.acknowledge(commit, ("SOUL.md",), expected_commit=None)
    (store.path / "refs/profile-sync/base").write_text("1" * 40 + "\n")
    with pytest.raises(GitError):
        store.baseline()


def test_atomic_transaction_compares_metadata_even_for_same_commit(tmp_path, shared, monkeypatch):
    from hermes_agent_sync.git_store import GitError, GitStore

    store = store_for(tmp_path, shared)
    commit = store.fetch()
    store.acknowledge(commit, ("SOUL.md",), expected_commit=None)
    other = GitStore(store.path, str(shared["remote"]), "master", allow_file=True)
    original = store._git
    raced = False

    def intercept(*args, **kwargs):
        nonlocal raced
        if args[:2] == ("update-ref", "--stdin") and not raced:
            raced = True
            other.acknowledge(commit, ("absent.md",), expected_commit=commit)
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "_git", intercept)
    with pytest.raises(GitError):
        store.acknowledge(commit, (), expected_commit=commit)
    assert raced
    assert store.baseline().observed_paths == ("absent.md",)


def test_ack_rejects_unfetched_or_unrelated_commits(tmp_path, shared, git_run):
    from hermes_agent_sync.git_store import GitError

    store = store_for(tmp_path, shared)
    commit = store.fetch()
    git_run(store.path, "update-ref", "-d", "refs/profile-sync/fetched")
    with pytest.raises(GitError, match="Fetch"):
        store.acknowledge(commit, (), expected_commit=None)
    assert store.baseline() is None


def test_output_timeout_and_git_failures_are_bounded_and_sanitized(tmp_path, shared, monkeypatch):
    import hermes_agent_sync.git_store as storage

    store = store_for(tmp_path, shared)
    with pytest.raises(storage.GitError, match="size limit"):
        store._git("ls-tree", store.fetch(), limit=4)
    with pytest.raises(storage.GitError) as caught:
        store._git("cat-file", "blob", "https://user:secret@example.invalid/not-an-object")
    assert "secret" not in str(caught.value)
    assert "example.invalid" not in str(caught.value)
    monkeypatch.setattr(storage, "_TIMEOUT", 0)
    with pytest.raises(storage.GitError, match="timed out"):
        store._git("rev-parse", "--is-bare-repository")


def test_valid_zlib_with_wrong_object_hash_is_rejected(tmp_path, shared, git_run):
    import zlib

    from hermes_agent_sync.git_store import GitError

    store = store_for(tmp_path, shared)
    commit = store.fetch()
    rules = store.read_rules(commit)
    oid = git_run(store.path, "rev-parse", commit + ":SOUL.md")
    objects = git_run(store.path, "rev-list", "--objects", commit).splitlines()
    preserved = []
    for line in objects:
        object_id = line.split(" ", 1)[0]
        kind = store._git("cat-file", "-t", object_id).decode().strip()
        preserved.append((kind, store._git("cat-file", kind, object_id)))
    for pack in (store.path / "objects/pack").iterdir():
        pack.unlink()
    for kind, data in preserved:
        store._git("hash-object", "-w", "-t", kind, "--stdin", data=data)
    corrupt = store.path / "objects" / oid[:2] / oid[2:]
    corrupt.chmod(0o600)
    payload = b"wrong contents!"
    corrupt.write_bytes(zlib.compress(b"blob " + str(len(payload)).encode() + b"\0" + payload))
    with pytest.raises(GitError, match="integrity"):
        store.snapshot(commit, rules, approved_rules=rules)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.invalid/repo?",
        "https://example.invalid/repo#",
        "ssh://git:@example.invalid/repo",
    ],
)
def test_empty_url_userinfo_and_delimiters_are_rejected(tmp_path, url):
    from hermes_agent_sync.git_store import GitError, GitStore

    with pytest.raises(GitError):
        GitStore(tmp_path / "cache", url, "master")


def test_baseline_symbolic_reference_is_rejected(tmp_path, shared, git_run):
    from hermes_agent_sync.git_store import GitError

    store = store_for(tmp_path, shared)
    commit = store.fetch()
    store.acknowledge(commit, ("SOUL.md",), expected_commit=None)
    git_run(store.path, "symbolic-ref", "refs/profile-sync/base", "refs/profile-sync/fetched")
    with pytest.raises(GitError):
        store.baseline()


def test_corrupt_root_tree_hash_fails_closed(tmp_path, shared, git_run):
    import zlib

    from hermes_agent_sync.git_store import GitError

    store = store_for(tmp_path, shared)
    commit = store.fetch()
    oid = git_run(store.path, "rev-parse", commit + "^{tree}")
    objects = git_run(store.path, "rev-list", "--objects", commit).splitlines()
    preserved = []
    for line in objects:
        object_id = line.split(" ", 1)[0]
        kind = store._git("cat-file", "-t", object_id).decode().strip()
        preserved.append((kind, store._git("cat-file", kind, object_id)))
    for pack in (store.path / "objects/pack").iterdir():
        pack.unlink()
    for kind, data in preserved:
        store._git("hash-object", "-w", "-t", kind, "--stdin", data=data)
    target = store.path / "objects" / oid[:2] / oid[2:]
    target.chmod(0o600)
    target.write_bytes(zlib.compress(b"tree 0\0"))
    with pytest.raises(GitError):
        store.read_rules(commit)


@pytest.mark.parametrize("behavior", ["exec", "timeout", "stderr-limit", "exited"])
def test_failed_operation_kills_transport_descendants(tmp_path, monkeypatch, behavior):
    import json
    import os
    import signal
    import sys
    import time
    from pathlib import Path

    import hermes_agent_sync.git_store as storage

    pids_file = tmp_path / "transport-pids.json"
    helper = tmp_path / "ssh-helper"
    helper.write_text(
        f"#!{sys.executable}\n"
        "import json, os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        f"behavior = {behavior!r}\n"
        "pids = [os.getppid(), os.getpid()]\n"
        "if behavior != 'exec':\n"
        "    child = subprocess.Popen([sys.executable, '-c', "
        "'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(20)'], "
        "stdout=subprocess.DEVNULL)\n"
        "    pids.append(child.pid)\n"
        f"Path({str(pids_file)!r}).write_text(json.dumps(pids))\n"
        "if behavior == 'exec':\n"
        "    os.execlp('sleep', 'sleep', '20')\n"
        "if behavior == 'exited':\n"
        "    # Close the transport stdout so Git exits, but the child retains stderr.\n"
        "    sys.exit(0)\n"
        "if behavior.endswith('-limit'):\n"
        "    os.write(2, b'x' * 65537)\n"
        "time.sleep(20)\n"
    )
    helper.chmod(0o700)
    env = storage._environment()
    env.update(GIT_ALLOW_PROTOCOL="ssh", GIT_SSH=str(helper), GIT_SSH_VARIANT="ssh")
    monkeypatch.setattr(storage, "_TIMEOUT", 1)

    def running(pid):
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except FileNotFoundError:
            return False
        return state not in {"Z", "X"}

    started = time.monotonic()
    pids = []
    try:
        error = "size limit" if behavior.endswith("-limit") else "timed out"
        with pytest.raises(storage.GitError, match=error):
            storage._command(["ls-remote", "git@example.invalid:repo"], env, limit=4)
        assert time.monotonic() - started < 4
        pids = json.loads(pids_file.read_text())
        if behavior == "exited":
            assert not running(pids[0]), "Git should exit while the child retains stderr"
        deadline = time.monotonic() + 1
        while any(running(pid) for pid in pids) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not [pid for pid in pids if running(pid)], "Transport descendants survived failure"
    finally:
        # Also clean the real test processes when reproducing the unfixed defect.
        if pids_file.exists():
            pids = json.loads(pids_file.read_text())
        for pid in reversed(pids):
            if running(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


@pytest.mark.parametrize("present", ["neither", "home", "xdg", "both"])
@pytest.mark.parametrize("xdg_setting", ["custom", "unset", "empty"])
def test_standard_global_auth_sources_preserve_git_order(
    isolated_home, monkeypatch, present, xdg_setting
):
    from hermes_agent_sync.git_store import _command, _environment

    monkeypatch.delenv("GIT_CONFIG_GLOBAL")
    if xdg_setting == "custom":
        xdg = isolated_home / "custom-config"
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    else:
        xdg = isolated_home / ".config"
        if xdg_setting == "unset":
            monkeypatch.delenv("XDG_CONFIG_HOME")
        else:
            monkeypatch.setenv("XDG_CONFIG_HOME", "")
    (xdg / "git").mkdir(parents=True)
    included = isolated_home / "included-config"
    included.write_text("[credential]\nhelper = forbidden-included-helper\n")
    unsafe = (
        f"[include]\npath = {included}\n"
        '[url "ext::forbidden"]\ninsteadOf = https://example.invalid/\n'
        "[credential]\nusername = not-imported\n"
    )
    if present in {"xdg", "both"}:
        (xdg / "git/config").write_text(
            "[credential]\nhelper = xdg-helper\nuseHttpPath = false\n" + unsafe
        )
    if present in {"home", "both"}:
        (isolated_home / ".gitconfig").write_text(
            "[credential]\nhelper = \nhelper = home-helper\nuseHttpPath = true\n"
            '[credential "https://example.invalid"]\nhelper = scoped-helper\n' + unsafe
        )
    env = _environment()
    entries = [
        (env[f"GIT_CONFIG_KEY_{i}"], env[f"GIT_CONFIG_VALUE_{i}"])
        for i in range(int(env["GIT_CONFIG_COUNT"]))
    ]
    expected = []
    if present in {"xdg", "both"}:
        expected += [("credential.helper", "xdg-helper"), ("credential.usehttppath", "false")]
    if present in {"home", "both"}:
        expected += [
            ("credential.helper", ""),
            ("credential.helper", "home-helper"),
            ("credential.usehttppath", "true"),
            ("credential.https://example.invalid.helper", "scoped-helper"),
        ]
    assert entries == expected
    assert env["GIT_CONFIG_GLOBAL"] == "/dev/null"
    if present != "neither":
        expected_value = b"true" if present in {"home", "both"} else b"false"
        actual = _command(["config", "--get", "credential.usehttppath"], env).strip()
        assert actual == expected_value


@pytest.mark.parametrize("override", ["home", "disabled"])
def test_explicit_global_auth_source_override_stays_isolated(isolated_home, monkeypatch, override):
    import os

    from hermes_agent_sync.git_store import _environment

    xdg = isolated_home / ".config/git"
    xdg.mkdir()
    (xdg / "config").write_text("[credential]\nhelper = xdg-helper\n")
    home_config = isolated_home / ".gitconfig"
    home_config.write_text("[credential]\nhelper = home-helper\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(home_config) if override == "home" else os.devnull)
    env = _environment()
    values = [env[f"GIT_CONFIG_VALUE_{i}"] for i in range(int(env["GIT_CONFIG_COUNT"]))]
    assert values == (["home-helper"] if override == "home" else [])
