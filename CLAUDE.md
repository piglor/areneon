# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

The rules above (worktrees, Redmine, privacy, gates, commits) are binding. This file adds commands and the architecture you need before touching code.

## Commands

Clone with `--recurse-submodules`. Toolchain: Node 24 + pnpm 10, uv (Python 3.12). GNU Make ≥ 3.82.

```bash
make setup          # build pinned Showdown, uv sync, pnpm install, pre-commit install
make check          # every offline CI gate: python + ts + guards
make showdown       # rebuild vendor/pokemon-showdown (needed before any TS test)
make ts             # pnpm lint && pnpm typecheck && pnpm test (vitest, coverage)
make python         # ruff check, ruff format --check, mypy, pytest (coverage)
make guards         # PokePaste team guard + production licence allowlist
```

Single tests. The coverage thresholds apply per file and fail on a subset, so run without coverage:

```bash
pnpm exec vitest run packages/showdown-bridge/test/runner.test.ts        # one file
pnpm exec vitest run packages/showdown-bridge/test/request.test.ts -t "forced switch"
uv run pytest tests/scripts/test_check_licenses.py --no-cov               # add ::test_name for one
```

Re-capture golden fixtures and team verdicts (only after a deliberate Showdown pin bump or engine-facing change; review the diff):

```bash
node packages/showdown-bridge/scripts/capture-golden.ts
```

TypeScript runs directly under Node's type stripping (`.ts` imports, `erasableSyntaxOnly`): no build step, no parameter properties, enums or namespaces.

## Architecture

**Polyglot monorepo** (ADR-002): TypeScript in `packages/*` (pnpm workspace), Python in `scripts/` and `tests/` today (`services/`, `ml/` later), Rust in `crates/` once the first crate lands. `packages/schema` holds the JSON Schema contracts: the rules records today (ADR-007). #671 makes it the source of truth for every contract, with the TS, pydantic and serde types generated from it.

**The reference engine is upstream Showdown, unmodified.** `vendor/pokemon-showdown` is a git submodule pinned to an exact SHA and built with `make showdown`. Areneon never decides legality itself: it asks the engine (`TeamValidator`, `side.choose`) and stops on unknowns. `docs/showdown-champions-gaps.md` lists what the engine does for Champions and what is unverified.

**`packages/showdown-bridge`** is the only code that talks to the engine:
- `engine.ts`: `loadEngine()` loads `dist/sim` via `createRequire` behind narrow local interfaces (`ShowdownSim`). It refuses a missing build or a checkout whose HEAD is not `PINNED_SHOWDOWN_COMMIT`. A test checks that constant against the submodule gitlink, so a pin bump changes both.
- `rules.ts` / `rules-load.ts` (ADR-007): Areneon's own rules are JSON records in `rules/`, never code constants. That covers which formats it runs and the forbidden engine rules (e.g. `forceopenteamsheets`, because CTS needs Bo1 with sheets declined). They are validated by `packages/schema/rules/v1` (Ajv) and fail closed. `jcs()` (RFC 8785) gives `rules_snapshot_hash` (provenance) and `format_rules_hash` (replay identity). `rules.ts` imports only Node built-ins, because CI reruns old commits' `scripts/rules-hash.ts` from bare archives.
- `formats.ts`: `resolveFormat` checks a format against the rules snapshot and the pinned engine. It returns the version manifest `{showdown_commit, format_id, regulation, ruleset_hash, format_rules_hash}` plus the embedded format rules record that every fixture and episode carries.
- `scripts/check-rules.ts` (in `make ts` and CI): validates every record and checks `rules/channels/stable.json`. The channel must name a snapshot built for the current pin, and `seq` rises by 1 per change. A pin bump must also move the channel (CONTRIBUTING).
- `request.ts` / `request-identity.ts`: decode one `|request|` into a phase (`preview`/`move`/`forceSwitch`/`wait`). Request identity is `request_seq` + `request_hash`, a SHA-256 of canonical JSON that excludes `rqid`. `rqid` exists only on the Showdown server path, never on the in-process sim.
- `runner.ts`: `runBattle` runs a seeded in-process BattleStream battle (runtime A1). Teams are validated first. Each `Actor` sees only its own player's requests. Any engine `|error|` aborts; nothing is retried. `|t:|` wall-clock lines are dropped so the same commit, seed and choices reproduce `omniscient_log`. That log holds both sides' hidden information: an evaluation artifact, never actor input. `replayActor` replays recorded choices.
- `teams.ts`: `importPaste` / `exportPaste` wrap the engine's `Teams.import` / `Teams.export`. Showdown silently skips unknown lines (an `SPs:` line zeroes the Stat Points; they belong in `EVs:`), so `importPaste` rejects a paste unless each Pokémon's lines all reappear in that Pokémon's re-export: only the engine's canonical form passes. `validateTeam` returns the pinned `TeamValidator`'s messages verbatim.
- `golden.ts`: the golden battle, team-fixture and verdict builders shared by the tests and the capture script.

**Fixtures** live only under `fixtures/` (CC0, synthetic or sanitized, with provenance). `fixtures/showdown/<format>/` holds the golden request fixtures plus `battle.summary.json` (the log hash and choices). `fixtures/teams/<format>/` holds synthetic PokePaste teams, a `manifest.json` (provenance, expected verdict and reason) and `verdicts.json` (the validator's log). CI re-captures and compares both, so engine behaviour drift fails the build. Pastes keep Showdown's trailing spaces (`.editorconfig`, `.gitattributes`).

**Repo guards** (`scripts/`, Python): `check_no_pokepaste.py` rejects PokePaste-shaped files outside `fixtures/` (exact-path exemptions in `.pokepaste-guard-allow`). `check_licenses.py` fails closed on any production dependency licence outside the allowlist.

**CI** (`.github/workflows/ci.yml`) runs `python`, `typescript` (submodule checkout + `make showdown`), `team-guard`, `licences`, `gitleaks`, `osv` and `dco`. The aggregate `ci` job is the required check on `main`, together with CodeQL.

Planned runtimes (ADR-004/005, not built yet): A2 is a bot client on a local Showdown server for sparring; the Chromium MV3 advisor extension is read-only and sees only the player's own request.
