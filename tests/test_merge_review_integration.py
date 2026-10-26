"""Exercise real snapshots/Git with explicitly artificial model completions."""

import json

import pytest

from hermes_agent_sync.comparison import compare_snapshots, prepare_comparison
from hermes_agent_sync.git_store import GitStore
from hermes_agent_sync.merging import build_merge_plan
from hermes_agent_sync.review import HostReviewer, ReviewError, review_plan
from hermes_agent_sync.selection import SelectionRules
from hermes_agent_sync.snapshot import FileSnapshot, Snapshot


def tree(files):
    return Snapshot(
        SelectionRules(("SOUL.md", "skills/**")),
        tuple(FileSnapshot(path, value, 0o100644) for path, value in sorted(files.items())),
        (),
        ("",),
    )


class ArtificialHost:
    def __init__(self, action="accept", text=None, question=""):
        self.calls = []
        self.action, self.text, self.question = action, text, question

    def complete(self, **options):
        from types import SimpleNamespace

        self.calls.append(options)
        document = json.loads(options["messages"][1]["content"])
        return SimpleNamespace(
            text=json.dumps(
                {
                    "request_id": document["request_id"],
                    "decisions": [
                        {
                            "path": path,
                            "action": self.action,
                            "text": self.text,
                            "mode": 0o100644 if self.action == "resolve" else None,
                            "question": self.question,
                            "reason": "Artificial fixture, not real inference.",
                        }
                        for path in document["changed_paths"]
                    ],
                }
            )
        )


def test_clean_text_merge_requires_host_review_and_keeps_context():
    a = tree({"SOUL.md": b"one\ntwo\nthree\n", "skills/example/SKILL.md": b"Preserve wording."})
    b = tree({"SOUL.md": b"ONE\ntwo\nthree\n", "skills/example/SKILL.md": b"Preserve wording."})
    c = tree({"SOUL.md": b"one\ntwo\nTHREE\n", "skills/example/SKILL.md": b"Preserve wording."})
    plan = build_merge_plan(
        compare_snapshots(a, b, c, observed_paths=("SOUL.md", "skills/example/SKILL.md"))
    )
    assert plan.candidates[0].status == "ready"
    host = ArtificialHost()
    result = review_plan(plan, HostReviewer(host), binding="fixture-profile/repo/master/A/B/C")
    assert len(host.calls) == 1
    assert "Preserve wording" in host.calls[0]["messages"][1]["content"]
    assert result.status == "ready"
    assert result.result_files[0].data == b"ONE\ntwo\nTHREE\n"
    assert len(result.result_files) == 2


@pytest.mark.parametrize("question", ["", "Use this combined wording?"])
def test_first_contact_differing_copies_need_explicit_model_resolution(question):
    a, b = tree({"SOUL.md": b"Left wording."}), tree({"SOUL.md": b"Right wording."})
    plan = build_merge_plan(compare_snapshots(None, a, b))
    assert plan.candidates[0].status == "ambiguous"
    host = ArtificialHost("resolve", "Combined wording.", question)
    result = review_plan(plan, HostReviewer(host), binding="fixture-profile/first-contact")
    assert result.outcomes[0].result.data == b"Combined wording."
    assert result.status == ("pending" if question else "ready")
    if question:
        with pytest.raises(ReviewError):
            _ = result.result_files


def test_real_git_review_never_applies_publishes_or_advances_baseline(
    tmp_path, git_repositories, git_run
):
    left, remote = git_repositories["left"], git_repositories["remote"]
    (left / "sync.yaml").write_text(
        "version: 1\nskip-symlinks: true\ninclude: [SOUL.md, skills/**]\n"
    )
    (left / "SOUL.md").write_bytes(b"remote instructions\n")
    git_run(left, "add", "--", "sync.yaml", "SOUL.md")
    git_run(left, "commit", "-m", "Artificial remote data")
    git_run(left, "push", "origin", "master")
    tip = git_run(remote, "rev-parse", "refs/heads/master")
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "SOUL.md").write_bytes(b"local instructions\n")
    rules = SelectionRules(("SOUL.md", "skills/**"))
    store = GitStore(tmp_path / "cache", str(remote), "master", allow_file=True)
    prepared = prepare_comparison(store, profile, approved_rules=rules)
    result = review_plan(
        build_merge_plan(prepared.comparison),
        HostReviewer(ArtificialHost("ask", question="Which wording is intended?")),
        binding=f"fixture-profile/repo/master/{prepared.fetched_commit}",
    )
    assert result.status == "pending"
    assert store.baseline() is None
    assert git_run(remote, "rev-parse", "refs/heads/master") == tip
    assert (profile / "SOUL.md").read_bytes() == b"local instructions\n"
    assert sorted(p.name for p in profile.iterdir()) == ["SOUL.md"]


@pytest.mark.parametrize("remote", [b"a\nc\nb\n", b"a\nb\n"])
@pytest.mark.parametrize("swap", [False, True])
def test_model_accept_cannot_make_unterminated_line_move_publishable(remote, swap):
    a = tree({"SOUL.md": b"a\nb\nc"})
    b = tree({"SOUL.md": b"c\na\nb\n"})
    c = tree({"SOUL.md": remote})
    if swap:
        b, c = c, b
    plan = build_merge_plan(compare_snapshots(a, b, c, observed_paths=("SOUL.md",)))
    with pytest.raises(ReviewError):
        review_plan(
            plan, HostReviewer(ArtificialHost()), binding="fixture-profile/moves", max_attempts=1
        )
