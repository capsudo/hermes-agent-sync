"""Selection policy is pure, bounded, root-anchored, and fail-closed."""

from dataclasses import FrozenInstanceError

import pytest


def test_rules_are_immutable_normalized_ordered_values() -> None:
    from hermes_agent_sync.selection import SelectionRules

    rules = SelectionRules(include=["SOUL.md", "skills/**"], exclude=["skills/private/**"])
    assert rules.include == ("SOUL.md", "skills/**")
    assert rules.exclude == ("skills/private/**",)
    assert rules == SelectionRules(("SOUL.md", "skills/**"), ("skills/private/**",))
    assert rules != SelectionRules(("skills/**", "SOUL.md"), ("skills/private/**",))
    assert rules.version == 1
    assert rules.skip_symlinks is True
    with pytest.raises(FrozenInstanceError):
        rules.version = 2


@pytest.mark.parametrize(
    "changes",
    [
        {"version": True},
        {"version": 2},
        {"version": "1"},
        {"version": 1.0},
        {"skip_symlinks": False},
        {"skip_symlinks": 1},
        {"skip_symlinks": "true"},
        {"include": "SOUL.md"},
        {"include": None},
        {"include": {"SOUL.md"}},
        {"include": (None,)},
        {"include": (1,)},
        {"include": (True,)},
        {"exclude": None},
        {"exclude": "skills/**"},
        {"exclude": (False,)},
    ],
)
def test_constructor_rejects_wrong_types(changes: dict) -> None:
    from hermes_agent_sync.selection import InvalidRules, SelectionRules

    with pytest.raises(InvalidRules):
        SelectionRules(**{"include": ("SOUL.md",), **changes})


@pytest.mark.parametrize(
    "pattern",
    [
        "",
        "/SOUL.md",
        "//host/share",
        "C:/SOUL.md",
        "c:SOUL.md",
        "C:",
        "skills\\x",
        ".",
        "..",
        "./SOUL.md",
        "skills/../x",
        "skills/./x",
        "skills//x",
        "skills/",
        "skills/a**",
        "skills/**b",
        "skills/***/x",
        "skills/\x00x",
        "skills/\nx",
        "skills/\x7fx",
        "skills/\x85x",
        "skills/[abc",
        "skills/[]",
        "skills/[!]",
        "skills/[z-a]",
        "skills/\ud800",
    ],
)
@pytest.mark.parametrize("field", ["include", "exclude"])
def test_constructor_rejects_noncanonical_patterns(pattern: str, field: str) -> None:
    from hermes_agent_sync.selection import InvalidRules, SelectionRules

    with pytest.raises(InvalidRules):
        SelectionRules(**{"include": (), field: (pattern,)})


@pytest.mark.parametrize(
    ("pattern", "path", "expected"),
    [
        ("SOUL.md", "SOUL.md", True),
        ("SOUL.md", "nested/SOUL.md", False),
        ("skills/*/SKILL.md", "skills/new/SKILL.md", True),
        ("skills/*/SKILL.md", "skills/new/nested/SKILL.md", False),
        ("skills/?/guide[12].md", "skills/a/guide1.md", True),
        ("skills/?/guide[12].md", "skills/aa/guide1.md", False),
        ("skills/?/guide[12].md", "skills/a/guide3.md", False),
        ("skills/[!a-c]/guide.md", "skills/z/guide.md", True),
        ("skills/[!a-c]/guide.md", "skills/b/guide.md", False),
        ("skills/*", "skills/one/references/guide.md", False),
        ("skills/*", "skills/one", True),
        ("skills/*", "Skills/one", False),
        ("skills/[]].md", "skills/].md", True),
        ("skills/[[]literal.md", "skills/[literal.md", True),
    ],
)
def test_segment_globs_are_root_anchored(pattern: str, path: str, expected: bool) -> None:
    from hermes_agent_sync.selection import SelectionRules

    assert SelectionRules((pattern,)).matches(path) is expected


@pytest.mark.parametrize(
    ("pattern", "path", "expected"),
    [
        ("skills/**", "skills", True),
        ("skills/**", "skills/new/SKILL.md", True),
        ("skills/**", "skills/new/references/deep/asset.bin", True),
        ("skills/**", "other/skills/new/SKILL.md", False),
        ("skills/**/SKILL.md", "skills/SKILL.md", True),
        ("skills/**/SKILL.md", "skills/new/nested/SKILL.md", True),
        ("**/guide.md", "guide.md", True),
        ("**/guide.md", "skills/a/references/guide.md", True),
        ("**/a/**/b/**", "a/b", True),
        ("**/a/**/b/**", "x/a/y/z/b/c", True),
        ("skills/**/SKILL.md", "skills/new/references/guide.md", False),
    ],
)
def test_recursive_glob_matches_zero_or_more_complete_segments(
    pattern: str, path: str, expected: bool
) -> None:
    from hermes_agent_sync.selection import SelectionRules

    assert SelectionRules((pattern,)).matches(path) is expected


def test_exclusions_win_over_later_or_more_specific_inclusions() -> None:
    from hermes_agent_sync.selection import SelectionRules

    rules = SelectionRules(("skills/**", "skills/private/SKILL.md"), ("skills/private/**",))
    assert not rules.matches("skills/private")
    assert not rules.matches("skills/private/SKILL.md")
    assert not rules.matches("skills/private/references/guide.md")
    assert rules.matches("skills/public/references/guide.md")


@pytest.mark.parametrize(
    ("include", "directory", "expected"),
    [
        (("SOUL.md", "memories/USER.md", "skills/**"), "", True),
        (("skills/**",), "skills/new/references", True),
        (("skills/*/SKILL.md",), "skills/new", True),
        (("skills/*/SKILL.md",), "skills/new/references", False),
        (("skills/**/SKILL.md",), "skills/new/references", True),
        (("**/target.md",), "arbitrarily/deep/path", True),
        (("skills/**",), "other/skills", False),
        (("SOUL.md",), "memories", False),
        (("skills",), "skills", False),
        ((), "", False),
    ],
)
def test_could_include_keeps_every_potential_descendant(
    include: tuple[str, ...], directory: str, expected: bool
) -> None:
    from hermes_agent_sync.selection import SelectionRules

    assert SelectionRules(include).could_include(directory) is expected


@pytest.mark.parametrize("directory", ["skills/private", "skills/private/deep"])
def test_could_include_prunes_fully_excluded_subtrees(directory: str) -> None:
    from hermes_agent_sync.selection import SelectionRules

    assert not SelectionRules(("skills/**",), ("skills/private/**",)).could_include(directory)


def test_could_include_does_not_prune_for_file_only_exclusion() -> None:
    from hermes_agent_sync.selection import SelectionRules

    assert SelectionRules(("skills/**",), ("skills/private",)).could_include("skills/private")
    assert SelectionRules(("skills/**",), ("skills/*/SKILL.md",)).could_include("skills/new")


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/SOUL.md",
        "//server/share",
        "C:/SOUL.md",
        "c:x",
        "x\\y",
        ".",
        "..",
        "x/./y",
        "x/../y",
        "x//y",
        "x/",
        "x\x00y",
        "x\ny",
        "x\x7fy",
        None,
        4,
    ],
)
@pytest.mark.parametrize("operation", ["matches", "could_include", "should_skip"])
def test_path_apis_reject_noncanonical_paths(path: str, operation: str) -> None:
    from hermes_agent_sync.selection import InvalidRules, SelectionRules, should_skip

    call = (
        should_skip if operation == "should_skip" else getattr(SelectionRules(("**",)), operation)
    )
    if path == "" and operation == "could_include":
        assert call(path)
    else:
        with pytest.raises(InvalidRules):
            call(path)


@pytest.mark.parametrize(
    "name",
    [
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
    ],
)
@pytest.mark.parametrize("suffix", ["", "/nested/data.txt"])
def test_safety_blocks_profile_runtime_subtrees(name: str, suffix: str) -> None:
    from hermes_agent_sync.selection import SelectionRules, should_skip

    path = name + suffix
    assert should_skip(path) == "safety"
    assert not SelectionRules(("**", path)).matches(path)
    assert should_skip("skills/example/" + path) is None


@pytest.mark.parametrize(
    "name",
    [
        "config.yaml",
        "state.db",
        "state-backup.db-wal",
        "projects.db.backup",
        "auth.json",
        ".env",
        ".env.prod",
        ".environment",
        "credentials.yaml",
        "credentials-backup.txt",
        ".netrc",
        "_netrc",
        ".git-credentials",
        "auth.yaml",
        "auth.yml",
        "oauth_tokens.json",
        "token",
        "tokens.txt",
        "api_key",
        "api-key.json",
        "secrets.json",
    ],
)
def test_safety_blocks_root_configuration_and_credentials(name: str) -> None:
    from hermes_agent_sync.selection import SelectionRules, should_skip

    assert should_skip(name) == "safety"
    assert not SelectionRules(("**",)).matches(name)


@pytest.mark.parametrize(
    "name",
    [
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
        ".env.local",
        "auth.json",
        "credentials.json",
        "tokens.json",
        "id_rsa",
        "id_ed25519",
        "id_ecdsa",
        "private.pem",
        "private.key",
        "cert.p12",
        "cert.pfx",
        "package.lock",
        "compiled.pyc",
        "local.db",
        "local.db-wal",
        "local.db-shm",
        "ID_RSA",
        "AUTH.JSON",
        "PRIVATE.PEM",
    ],
)
@pytest.mark.parametrize("prefix", ["", "skills/new/references/"])
@pytest.mark.parametrize("suffix", ["", "/must-not-walk.txt"])
def test_safety_blocks_sensitive_components_at_any_depth(
    name: str, prefix: str, suffix: str
) -> None:
    from hermes_agent_sync.selection import SelectionRules, should_skip

    path = prefix + name + suffix
    assert should_skip(path) == "safety"
    assert not SelectionRules(("**", path)).matches(path)


@pytest.mark.parametrize(
    "path",
    [
        "SOUL.md",
        "memories/USER.md",
        "memories/MEMORY.md",
        "skills/new/SKILL.md",
        "skills/new/scripts/helper.py",
        "skills/new/references/config.yaml",
        "skills/new/references/drawing.svg",
        "skills/new/.environment",
        "skills/new/lock.md",
    ],
)
def test_safety_does_not_prohibit_portable_content_or_memory(path: str) -> None:
    from hermes_agent_sync.selection import SelectionRules, should_skip

    assert should_skip(path) is None
    assert SelectionRules(("**",)).matches(path)


def test_parse_rules_returns_same_normalized_value_as_constructor() -> None:
    from hermes_agent_sync.selection import SelectionRules, parse_rules

    document = "version: 1\nskip-symlinks: true\ninclude: [SOUL.md, 'skills/**']\n"
    expected = SelectionRules(("SOUL.md", "skills/**"))
    assert parse_rules(document) == expected
    assert parse_rules(document.encode("utf-8")) == expected
    assert parse_rules(document + "exclude: ['skills/private/**']\n") == SelectionRules(
        expected.include, ("skills/private/**",)
    )


@pytest.mark.parametrize(
    "document",
    [
        "",
        "null",
        "[]",
        "42",
        "true",
        "include: [SOUL.md]\n",
        "version: 1\ninclude: [SOUL.md]\n",
        "version: 1\nskip-symlinks: true\n",
        "version: 1\nskip-symlinks: true\ninclude: []\nunknown: true\n",
        "version: 1\nskip-symlinks: true\ninclude: {SOUL.md: true}\n",
        "version: 1\nskip-symlinks: true\ninclude: []\nexclude: null\n",
        "version: 1\nskip-symlinks: true\ninclude: []\ntrue: value\n",
        "version: true\nskip-symlinks: true\ninclude: []\n",
        "version: 2\nskip-symlinks: true\ninclude: []\n",
        "version: 1\nskip-symlinks: false\ninclude: []\n",
    ],
)
def test_parse_rules_rejects_invalid_schema(document: str) -> None:
    from hermes_agent_sync.selection import InvalidRules, parse_rules

    with pytest.raises(InvalidRules):
        parse_rules(document)


@pytest.mark.parametrize(
    "document",
    [
        "version: [BROKEN_PRIVATE_VALUE",
        "!!python/object/apply:os.system ['SHOULD_NEVER_EXECUTE']",
        "---\nversion: 1\n---\ninclude: []",
        b"\xffPRIVATE_VALUE",
        "\ud800PRIVATE_VALUE",
        None,
        42,
        ["version: 1"],
    ],
)
def test_parser_errors_are_sanitized_invalid_rules(document: object) -> None:
    from hermes_agent_sync.selection import InvalidRules, parse_rules

    with pytest.raises(InvalidRules) as error:
        parse_rules(document)
    assert "PRIVATE_VALUE" not in str(error.value)
    assert "SHOULD_NEVER_EXECUTE" not in str(error.value)
    assert error.value.__suppress_context__ or error.value.__context__ is None


@pytest.mark.parametrize(
    "extra",
    [
        "version: 1\n",
        "include: [OTHER.md]\n",
        "exclude: []\nexclude: []\n",
        "skip-symlinks: true\n",
        "'include': []\n",
        "? [unhashable]\n: forbidden\n",
    ],
)
def test_parser_rejects_duplicate_or_unhashable_mapping_keys(extra: str) -> None:
    from hermes_agent_sync.selection import InvalidRules, parse_rules

    document = "version: 1\nskip-symlinks: true\ninclude: [SOUL.md]\n" + extra
    with pytest.raises(InvalidRules):
        parse_rules(document)


@pytest.mark.parametrize(
    "document",
    [
        "version: 1\nskip-symlinks: true\ninclude: &paths [SOUL.md]\n",
        "version: 1\nskip-symlinks: true\ninclude: [&one SOUL.md, *one]\n",
        "version: 1\nskip-symlinks: true\ninclude: [*missing]\n",
        "version: 1\nskip-symlinks: true\ninclude: &loop [*loop]\n",
        "&root {version: 1, skip-symlinks: true, include: []}",
        "version: 1\nskip-symlinks: true\ninclude: []\nexclude: &x []\n",
    ],
)
def test_parser_rejects_every_yaml_anchor_or_alias(document: str) -> None:
    from hermes_agent_sync.selection import InvalidRules, parse_rules

    with pytest.raises(InvalidRules):
        parse_rules(document)


@pytest.mark.parametrize("field", ["include", "exclude"])
def test_constructor_bounds_pattern_count_and_length(field: str) -> None:
    from hermes_agent_sync import selection

    with pytest.raises(selection.InvalidRules):
        selection.SelectionRules(**{"include": (), field: ("x" * 1025,)})
    with pytest.raises(selection.InvalidRules):
        selection.SelectionRules(**{"include": (), field: ("é" * 513,)})
    with pytest.raises(selection.InvalidRules):
        selection.SelectionRules(**{"include": (), field: ("x",) * 257})
    assert selection.SelectionRules(**{"include": (), field: ("x" * 1024,)})
    assert selection.SelectionRules(**{"include": (), field: ("x",) * 256})


def test_pattern_limits_apply_across_include_and_exclude() -> None:
    from hermes_agent_sync.selection import InvalidRules, SelectionRules

    with pytest.raises(InvalidRules):
        SelectionRules(("x",) * 128, ("y",) * 129)
    with pytest.raises(InvalidRules):
        SelectionRules(("x" * 1024,) * 64, ("y",))


@pytest.mark.parametrize(
    "source",
    [
        "#" + "x" * 65536,
        b"#" + b"x" * 65536,
        "#" + "é" * 32768,
    ],
)
def test_parser_bounds_input_bytes_before_yaml_parsing(source: str | bytes) -> None:
    from hermes_agent_sync.selection import InvalidRules, parse_rules

    with pytest.raises(InvalidRules, match="too large"):
        parse_rules(source)


def test_parser_accepts_input_at_the_byte_limit() -> None:
    from hermes_agent_sync.selection import SelectionRules, parse_rules

    document = "version: 1\nskip-symlinks: true\ninclude: []\n#"
    document += "x" * (65536 - len(document.encode("utf-8")))
    assert parse_rules(document) == SelectionRules(())


def test_parser_rejects_excessive_nesting_before_constructing_yaml() -> None:
    from hermes_agent_sync.selection import InvalidRules, parse_rules

    with pytest.raises(InvalidRules, match="nesting"):
        parse_rules("[" * 40 + "]" * 40)


def test_packaged_defaults_select_complete_and_new_skills_with_configurable_memory() -> None:
    from importlib.resources import files

    from hermes_agent_sync.selection import DEFAULT_RULES_TEXT, SelectionRules, parse_rules

    assert (
        files("hermes_agent_sync").joinpath("default-sync.yaml").read_text(encoding="utf-8")
        == DEFAULT_RULES_TEXT
    )
    defaults = parse_rules(DEFAULT_RULES_TEXT)
    assert defaults.include == ("SOUL.md", "memories/USER.md", "skills/**")
    assert defaults.exclude == ("memories/MEMORY.md",)
    for path in (
        "SOUL.md",
        "memories/USER.md",
        "skills/not-yet-created/SKILL.md",
        "skills/not-yet-created/references/deep/asset.bin",
        "skills/not-yet-created/scripts/helper.sh",
    ):
        assert defaults.matches(path)
    assert not defaults.matches("memories/MEMORY.md")
    assert SelectionRules(("memories/**",), defaults.exclude).matches("memories/USER.md")
    assert not SelectionRules(("memories/**",), defaults.exclude).matches("memories/MEMORY.md")
    assert SelectionRules(("memories/**",)).matches("memories/MEMORY.md")


def test_package_metadata_declares_the_default_template() -> None:
    import tomllib
    from pathlib import Path

    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert "default-sync.yaml" in project["tool"]["setuptools"]["package-data"]["hermes_agent_sync"]


@pytest.mark.parametrize(
    "document",
    [
        "version: !!int not-an-int\nskip-symlinks: true\ninclude: []",
        "version: !!float not-a-float\nskip-symlinks: true\ninclude: []",
        "version: !!timestamp 2026-99-99\nskip-symlinks: true\ninclude: []",
        "version: 1\nskip-symlinks: true\ninclude: [" + "1" * 5000 + "]",
    ],
)
def test_scalar_conversion_failures_are_sanitized(document: str) -> None:
    from hermes_agent_sync.selection import InvalidRules, parse_rules

    with pytest.raises(InvalidRules) as error:
        parse_rules(document)
    assert "not-an-int" not in str(error.value)
    assert "not-a-float" not in str(error.value)
    assert "2026-99-99" not in str(error.value)


@pytest.mark.parametrize("as_bytes", [False, True], ids=["str", "bytes"])
@pytest.mark.parametrize(
    "scalar",
    [
        "!!bool PRIVATE_VALUE",
        "!!timestamp PRIVATE_VALUE",
        '!!int ""',
        '!!float ""',
    ],
    ids=["invalid-bool", "invalid-timestamp", "empty-int", "empty-float"],
)
def test_scalar_constructor_errors_are_sanitized(scalar: str, as_bytes: bool) -> None:
    import traceback

    from hermes_agent_sync.selection import InvalidRules, parse_rules

    document = f"version: {scalar}\nskip-symlinks: true\ninclude: []"
    source = document.encode("utf-8") if as_bytes else document
    with pytest.raises(InvalidRules) as error:
        parse_rules(source)
    assert str(error.value) == "invalid YAML rules document"
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__
    displayed = "".join(traceback.format_exception(error.value))
    assert "private_value" not in displayed.casefold()
    assert document not in displayed
