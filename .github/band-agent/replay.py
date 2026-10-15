"""Build band-agent's SDK revision: upstream main, the fork's own commits, then each
listed upstream pull request at its current head.

The fork never stores copies of the pull requests. Every run reads their heads from
GitHub, so a pull request updated upstream is picked up on the next run, and one that
merged or closed drops out. A pull request that no longer applies fails the run: the
fix is a rebase of that pull request upstream, never a change in the fork.

Run from a checkout of ``band-agent-patches`` with remotes ``origin`` (the fork) and
``upstream`` (band-ai/band-sdk-python), and ``gh`` authenticated:

    python .github/band-agent/replay.py [--dry-run]

The result is left checked out on ``band-agent-patches-latest``; ``--dry-run`` only
prints what it built. Outputs for the workflow go to ``$GITHUB_OUTPUT`` when set.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

UPSTREAM = "band-ai/band-sdk-python"
STAGED_BRANCH = "band-agent-patches-latest"
PR_LIST = Path(__file__).with_name("prs.txt")
MAIN_REF = "refs/remotes/upstream/main"
STAGED_REF = f"refs/remotes/origin/{STAGED_BRANCH}"


class ReplayError(Exception):
    """The run must stop; the message says what to fix upstream."""


@dataclass(frozen=True)
class Entry:
    number: int
    after: int | None


@dataclass(frozen=True)
class PullRequest:
    number: int
    state: str  # "open", "merged" or "closed"
    head: str
    title: str


def git(*args: str, check: bool = True) -> str:
    done = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    if check and done.returncode != 0:
        raise ReplayError(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout.strip()


def is_ancestor(ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant], check=False
        ).returncode
        == 0
    )


def read_entries(path: Path) -> list[Entry]:
    entries: list[Entry] = []
    seen: set[int] = set()
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].split()
        if not line:
            continue
        if len(line) == 1:
            number, after = int(line[0]), None
        elif len(line) == 3 and line[1] == "after":
            number, after = int(line[0]), int(line[2])
            if after not in seen:
                raise ReplayError(
                    f"#{number} is listed after #{after}, which is not above it"
                )
        else:
            raise ReplayError(f"unreadable line in {path.name}: {raw!r}")
        if number in seen:
            raise ReplayError(f"#{number} is listed twice")
        seen.add(number)
        entries.append(Entry(number, after))
    return entries


def pull_request(number: int) -> PullRequest:
    done = subprocess.run(
        ["gh", "api", f"repos/{UPSTREAM}/pulls/{number}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        raise ReplayError(f"cannot read #{number}: {done.stderr.strip()}")
    data = json.loads(done.stdout)
    state = "merged" if data.get("merged_at") else data["state"]
    return PullRequest(number, state, data["head"]["sha"], data["title"])


def fetch(pr: PullRequest) -> None:
    git(
        "fetch",
        "--quiet",
        "upstream",
        f"+pull/{pr.number}/head:refs/band-agent/pr/{pr.number}",
    )
    if git("rev-parse", f"refs/band-agent/pr/{pr.number}") != pr.head:
        raise ReplayError(f"#{pr.number} moved while this run read it; run again")


def apply(pr: PullRequest, base: str, earlier: list[int]) -> bool:
    """Apply the pull request's own changes (``base``..head) as one commit.

    The change is cherry-picked from a synthetic commit whose parent is ``base`` and
    whose tree is the head's, so git's three-way merge applies it. Returns False when
    the pull request changes nothing over ``base`` or its changes are already on main.
    """
    if git("rev-parse", f"{base}^{{tree}}") == git("rev-parse", f"{pr.head}^{{tree}}"):
        return False
    message = "\n".join(
        [
            pr.title,
            "",
            f"Source-Base: {base}",
            f"Source-Head: {pr.head}",
            f"Upstream-PR: {UPSTREAM}#{pr.number}",
        ]
    )
    squash = git("commit-tree", f"{pr.head}^{{tree}}", "-p", base, "-m", message)
    before = git("rev-parse", "HEAD")
    done = subprocess.run(
        ["git", "cherry-pick", "--empty=drop", squash],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        git("cherry-pick", "--abort", check=False)
        where = "upstream main" + (
            " plus " + ", ".join(f"#{n}" for n in earlier) if earlier else ""
        )
        raise ReplayError(
            f"#{pr.number} ({pr.head[:12]}) no longer applies to {where}; "
            f"rebase #{pr.number} upstream"
        )
    return git("rev-parse", "HEAD") != before


def refresh_staged() -> None:
    """Point STAGED_REF at the fork's staged branch, or remove it when there is none.

    A stale local copy would make an unpublished build look unchanged, so it is dropped
    first; only a confirmed absent branch is a normal state (nothing staged yet).
    """
    git("update-ref", "-d", STAGED_REF)
    probe = subprocess.run(
        ["git", "ls-remote", "--exit-code", "origin", f"refs/heads/{STAGED_BRANCH}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode == 0:
        git("fetch", "--quiet", "origin", f"+refs/heads/{STAGED_BRANCH}:{STAGED_REF}")
    elif probe.returncode != 2:  # ls-remote's "no such ref"
        raise ReplayError(
            f"cannot read origin's {STAGED_BRANCH}: {probe.stderr.strip()}"
        )


def replay(entries: list[Entry]) -> dict[str, str]:
    git("fetch", "--quiet", "upstream", "main")
    refresh_staged()
    target = git("rev-parse", MAIN_REF)
    own = git("rev-list", "--reverse", "HEAD", "--not", target).split()
    if not own:
        raise ReplayError("band-agent-patches carries no fork commit")
    for sha in own:
        if "Upstream-PR:" in git("log", "-1", "--format=%B", sha):
            raise ReplayError(
                f"band-agent-patches carries a pull request copy ({sha[:12]}); "
                "keep only fork commits there and list the pull request in prs.txt"
            )

    git("checkout", "--quiet", "-B", STAGED_BRANCH, target)
    for sha in own:
        git("cherry-pick", sha)

    heads: dict[int, PullRequest] = {}
    applied: list[str] = []
    applied_numbers: list[int] = []
    merged: list[str] = []
    closed: list[str] = []
    notes: list[str] = []
    for entry in entries:
        pr = pull_request(entry.number)
        heads[pr.number] = pr
        if pr.state == "merged":
            merged.append(f"#{pr.number}")
            continue
        if pr.state == "closed":
            closed.append(f"#{pr.number}")
            continue
        fetch(pr)
        prerequisite = heads.get(entry.after) if entry.after is not None else None
        if prerequisite is not None and prerequisite.state == "open":
            base = prerequisite.head
            if not is_ancestor(base, pr.head):
                raise ReplayError(
                    f"#{pr.number} is not built on #{prerequisite.number}'s current head; "
                    f"rebase #{pr.number} onto #{prerequisite.number} upstream"
                )
            # Only the changes over the prerequisite are this pull request's own. Upstream
            # commits merged into it but not into the prerequisite would be counted as its
            # own and could undo later upstream changes, so they must arrive through the
            # prerequisite instead.
            if not is_ancestor(git("merge-base", target, pr.head), base):
                raise ReplayError(
                    f"#{pr.number} contains upstream main commits that "
                    f"#{prerequisite.number} does not; rebase #{prerequisite.number} onto "
                    f"main, then #{pr.number} onto #{prerequisite.number}, upstream"
                )
        else:
            base = git("merge-base", target, pr.head)
            if entry.after is not None:
                notes.append(
                    f"#{pr.number}'s prerequisite #{entry.after} is no longer open; "
                    f"applied #{pr.number}'s changes since main"
                )
        if apply(pr, base, applied_numbers):
            applied.append(f"#{pr.number}@{pr.head[:12]}")
            applied_numbers.append(pr.number)
        else:
            notes.append(f"#{pr.number} adds nothing to upstream main")

    staged = git("rev-parse", "--verify", "--quiet", STAGED_REF, check=False)
    changed = not staged or git("rev-parse", f"{staged}^{{tree}}") != git(
        "rev-parse", "HEAD^{tree}"
    )
    return {
        "target": target,
        "changed": "true" if changed else "false",
        "applied": " ".join(applied),
        "merged": " ".join(merged),
        "closed": " ".join(closed),
        "notes": "; ".join(notes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build band-agent's SDK revision.")
    parser.add_argument("--prs", type=Path, default=PR_LIST)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build locally and print; write no outputs",
    )
    args = parser.parse_args()
    try:
        outputs = replay(read_entries(args.prs))
    except ReplayError as error:
        print(f"::error::{error}")
        return 1
    for key, value in outputs.items():
        print(f"{key}={value}")
    if not args.dry_run and (path := os.environ.get("GITHUB_OUTPUT")):
        with open(path, "a") as out:
            out.writelines(f"{key}={value}\n" for key, value in outputs.items())
    return 0


if __name__ == "__main__":
    sys.exit(main())
