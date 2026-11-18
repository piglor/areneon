---
name: lessons-learned
description: Anti-patterns and mistakes to avoid when working on this project. Read before starting any implementation.
disable-model-invocation: true
---

# Lessons Learned

Internal Areneon skill: a short checklist of durable project lessons. Keep dated execution history outside Git (Redmine). Use the owning specialist skill (`code-review`, `diagnosing-bugs`, `worktree-status`) for full procedures.

## Commits and worktrees

- **Never `git add -A` or `git commit -a`.** Stage only the files you changed, by path.
- **Worktree changes can be lost.** Commit and push from the worktree before removing it.
- **Define completion by the requested end state.** A local green run is not delivery: verify the pushed branch, the PR gates and the tracker state the task requires.
- **Re-check before any external write or cleanup.** Confirm current state and exact targets immediately before the mutation; never delete an active worktree from a stale inventory.
- **Reconcile a shared PR immediately before pushing.** Fetch the branch, rebase onto its current tip, drop patches already present, and never force-push a shared branch.

## Code edits

- **Use the Edit tool for multi-line changes**; inspect the diff straight after. Restore a corrupted file from a known baseline before repairing it.
- **Removing an implementation does not remove its product claim.** Search the README, CLAUDE.md and docs for stale promises.
- **Do not ship placeholder code.** Implement from a spec requirement or do not add the field.

## Coverage

- **Delete, do not exempt.** If new code cannot be covered by a test, remove it; never use coverage exemptions on production code.
- **Run every applicable quality gate after each code change**, review fixes included.

## Redmine

- **Follow the `start-work` skill for any session that references a Redmine ticket.**
- **ADR wiki pages use Markdown and have `ADR` as their parent page** (see `adr-wiki`). Follow the configured ADR transitions: Proposed, Under Review, Accepted or Rejected.
- **Ticket status follows the configured lifecycle.** A commit, an Accepted ADR or a passing CI run does not by itself resolve a ticket. Feature/Bug/Task complete as Resolved (3); Closed (5) is for rejected or superseded work. An Accepted ADR records a decision; it does not start implementation.
- **Preserve the hierarchy.** Every ADR has a standalone ADR issue; its Epic is created on the Wave roadmap only after acceptance; implementation tickets are Epic children. Reviews are journal notes on the ADR issue.

## Code review

- **Request independent review after code changes** with the `code-review` skill, Standards and Spec as separate axes, and review the complete PR diff after rebases against the exact head.

## CI

- **Every test must run in CI.** pytest collects `tests/` and vitest runs every `packages/*` project from the root config; a test outside those roots is dead. Name the CI job that runs your new test.
- **Workflow `name:` is lowercase-with-hyphens** (`ci`, `rules-publish`).

## Areneon domain and privacy

- **Use the project glossary exactly** (`CONTEXT.md`): Preview Policy, Battle Policy, Closed Team Sheet (CTS), player-perspective observation, legal joint choice, `request_seq`, `rqid`.
- **The pinned Showdown engine is the only legality authority.** Validate through the pinned engine (`side.choose`, team validator) and stop on unknown legality.
- **`rqid` exists only on the Showdown server path.** Identify requests by `request_seq` + `request_hash` on both paths.
- **CTS runs are Bo1 with every actor declining team sheets**, proven by a fixture. Open-team-sheet play follows ADR-027.
- **Never commit private teams.** They live in the private R2 bucket (ADR-015) and, on the owner's workstation, only under gitignored `private-data/`. Tokens never enter the repository, a log, a chat or a command line. Public fixtures are sanitized or synthetic, labelled with provenance, and CC0.
- **Actor inputs are player-perspective only.** The omniscient stream, opponent hidden sets or future state never reach a policy input, log, checkpoint or model input; hidden state is only an evaluation upper bound, labelled as such.
- **Heuristics are integration fixtures, not AI.** Never describe a heuristic or untrained policy as competent.
- **Do not copy AGPL code** (Showdown client, Champions Calc); write independently.
- **Deadlines derive from the live VGC Timer.** 60 s / 20 s are upper bounds; never let Showdown auto-choose fire.
