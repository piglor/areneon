## Agent instructions

Before starting any task, read `CONTEXT.md` for domain vocabulary and `README.md` for the project overview. ADRs live on the Redmine wiki (`redmine.piglor.com/projects/areneon/wiki/ADR`).

Before writing code, read the `lessons-learned` skill. It records mistakes from prior sessions so you don't repeat them.

### Worktree isolation

Every task MUST run in its own `git worktree`. Never work directly on `main`. Name the worktree after the Redmine ticket (e.g. `670-showdown-pin`). Commit from the worktree, then remove it when done.

```bash
git worktree add ../areneon-<ticket>-<name> -b <ticket>-<name> main
cd ../areneon-<ticket>-<name>
# ... work ...
git worktree remove ../areneon-<ticket>-<name>
```

### Issue tracker

Redmine: `redmine.piglor.com/projects/areneon` is the single tracker for engineering work, research, ADRs and experiments. GitHub Issues (`piglor/areneon`) are for community contributions only. See `.agents/issue-tracker.md`. Use the `start-work` skill when beginning any Redmine ticket.

### Multi-agent models

Use the model-selection policy in `.agents/model-policy.md` when delegating implementation and independent review work. Announce any fallback.

### Triage labels

Default five-label vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `.agents/triage-labels.md`.

### Domain knowledge

- **CONTEXT.md**: domain glossary (use exact terms)
- **Redmine wiki**: canonical ADRs
- **Notion** (Piglor / Areneon): product goals and Wave specs
- See `.agents/domain.md` for consumer rules

### Privacy rules (hard)

- Never commit a real competitive team. Private teams live only in gitignored `private-data/`; public fixtures are sanitized or synthetic and CC0.
- Policy inputs are player-perspective only. Opponent hidden information must never enter an observation, log, checkpoint or extension message.
- The pinned Showdown engine is the only legality authority.
- Do not copy AGPL code (Showdown client, Champions Calc).

### Quality gates (local)

- Python: `ruff check`, `ruff format --check`, `mypy --strict`, `pytest` with ≥90% line coverage.
- TypeScript: `eslint`, `tsc --noEmit` (strict), `vitest` with ≥90% line coverage.
- Rust (when the first crate lands): `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test`, `cargo llvm-cov` with ≥90% line coverage, `cargo deny`. Rust never reimplements mechanics or legality.
- A change to the Showdown pin, request decoder or legal-choice enumeration must re-run the golden fixtures.

If new code cannot be covered by a test, remove it rather than exempting it.

### Commits and PRs

- Conventional Commits, `Refs #<redmine-id>` in every commit and PR, DCO `Signed-off-by` (`git commit -s`).
- `main` is protected: PRs with required CI only, merged by the maintainer. Never push to `main` directly.
- A Redmine issue closes on evidence (tests, CI run, artifacts), not just because a PR merged.

### Repository layout and spikes

- `main` holds only real code and docs, laid out by purpose (`packages/`, `services/`, `crates/`, `docs/<topic>.md`). Redmine ticket numbers go in branch names, commit trailers (`Refs #N`) and PRs, never in file or directory names.
- A spike (throwaway experiment for an ADR) lives only on its own branch and PR and is never merged to `main`. Its evidence goes on the Redmine ticket (journal note with the results and a link to the branch commit), and the ADR cites that commit.
- Anything a spike proves worth keeping is rebuilt in a properly named package or service with the full quality gates.

### Features and sizing

- **0 → 1** (new package, service, extension or model pipeline): architecture first, ADR required (`prepare-adr`, `adr-wiki` skills).
- **0 → many** (test expansion): keep tests on public interfaces (seams), not internals.
- **MVP ≠ simplest**: cut the thinnest slice *of the final architecture*, on the seams the full vision needs (schema versions, ports, documented extension points).
