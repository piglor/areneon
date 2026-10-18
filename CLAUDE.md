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
make ts             # pnpm lint && pnpm typecheck && pnpm test (vitest, coverage) + check-rules.ts
make python         # ruff check, ruff format --check, mypy, pytest (coverage)
make contracts      # contract schemas fit their ADR-008 profile; generated TS types have no drift
make contracts-update  # regenerate packages/schema/generated/ after a schema change (commit the result)
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
node packages/showdown-bridge/scripts/capture-choices.ts   # golden choice corpus; also after an enumerator change
```

TypeScript runs directly under Node's type stripping (`.ts` imports, `erasableSyntaxOnly`): no build step, no parameter properties, enums or namespaces.

## Architecture

**Polyglot monorepo** (ADR-002): TypeScript in `packages/*` (pnpm workspace), Python in `scripts/` and `tests/` today (`services/`, `ml/` later), Rust in `crates/` once the first crate lands. `packages/schema` holds the JSON Schema contracts: the rules records (`rules/`, ADR-007) and the cross-language contracts (`contracts/v<N>/`, ADR-008), from which the TS types in `generated/ts/` are generated and checked in (pydantic and serde follow in #671's later slices). A contract's `$id` (`urn:areneon:schema:<contracts|engine>:v<N>:<name>`) declares its profile, and `src/profile.ts` enforces it as a keyword whitelist: closed objects, bounded integers, local `$ref`s, `kind`-tagged `oneOf`; `engine` schemas, which copy the pinned engine's wire shapes, also allow scalar type unions and get their own Ajv instance (`src/ajv.ts`). `make contracts` fails on a profile violation or on generated types that differ from a fresh run.

**The reference engine is upstream Showdown, unmodified** (ADR-003). `vendor/pokemon-showdown` is a git submodule pinned to an exact SHA and built with `make showdown`. Areneon never decides legality itself: it asks the engine (`TeamValidator`, `side.choose`) and stops on unknowns. `docs/showdown-champions-gaps.md` lists what the engine does for Champions and what is unverified. It is generated from the gap ledger records in `rules/gaps/` by `scripts/gaps-doc.ts`; edit the records, never the doc (a test compares them). Gap edits move `rules_snapshot_hash` but never `format_rules_hash`.

**`packages/showdown-bridge`** is the only code that talks to the engine:
- `engine.ts`: `loadEngine()` loads `dist/sim` via `createRequire` behind narrow local interfaces (`ShowdownSim`). It refuses a missing build or a checkout whose HEAD is not `PINNED_SHOWDOWN_COMMIT`. A test checks that constant against the submodule gitlink, so a pin bump changes both.
- `rules.ts` / `rules-load.ts` (ADR-007): Areneon's own rules are JSON records in `rules/`, never code constants. That covers which formats it runs and the forbidden engine rules (e.g. `forceopenteamsheets`, because CTS needs Bo1 with sheets declined). They are validated by `packages/schema/rules/v1` (Ajv) and fail closed. `jcs()` (RFC 8785) gives `rules_snapshot_hash` (provenance) and `format_rules_hash` (replay identity). `packages/showdown-bridge/scripts/rules-hash.ts` and everything it pulls in (`rules.ts`, `request-identity.ts`, `canonical-json.ts`, `engine.ts`) import only Node built-ins, because CI reruns old commits' copy of it from bare archives. `canonical-json.ts` (`canonicalJson`, `jcs`) imports nothing at all, so the browser-bundled observation builder can use it too (ADR-008 §5); a test enforces that.
- `formats.ts`: `resolveFormat` checks a format against the rules snapshot and the pinned engine. It returns the version manifest `{showdown_commit, format_id, regulation, ruleset_hash, format_rules_hash}` plus the embedded format rules record that every fixture (and, later, every episode) carries.
- `scripts/check-rules.ts` (in `make ts` and CI; policy in `rules-channel.ts` / `rules-history.ts`): validates every record and checks `rules/channels/`, which must hold exactly one file per channel in the channel schema's enum. Each channel must name a snapshot built for the current pin, and `seq` rises by 1 per change. A pin bump must also move the channel (CONTRIBUTING).
- `request.ts` / `request-identity.ts`: decode one `|request|` into a phase (`preview`/`move`/`forceSwitch`/`wait`). Request identity is `request_seq` + `request_hash`, a SHA-256 of canonical JSON that excludes `rqid`. `rqid` exists only on the Showdown server path, never on the in-process sim.
- `runner.ts`: `runBattle` runs a seeded in-process BattleStream battle (runtime A1). Teams are validated first. Each `Actor` sees only its own player's requests. Each request carries its `legal` choice set, judged by a `CloneJudge` on the stream's own battle. The CloneJudge's verdicts on pending candidates go to `hidden_verdicts`, never to an actor. If the actor submits a pending candidate and the engine refuses it as `[Unavailable choice]`, the updated request that follows is a new request (`request_seq` + 1). Any other engine `|error|` aborts; nothing is retried. `|t:|` wall-clock lines are dropped so the same commit, seed and choices reproduce `omniscient_log`. That log holds both sides' hidden information: an evaluation artifact, never actor input. `replayActor` replays recorded choices.
- `teams.ts`: `importPaste` / `exportPaste` wrap the engine's `Teams.import` / `Teams.export`. Showdown silently skips unknown lines (an `SPs:` line zeroes the Stat Points; they belong in `EVs:`), so `importPaste` rejects a paste unless each Pokémon's lines all reappear in that Pokémon's re-export: only the engine's canonical form passes. `validateTeam` returns the pinned `TeamValidator`'s messages verbatim.
- `choices.ts` / `choice-judge.ts` (ADR-006): `enumerateCandidates` turns the player's own request into every engine-distinct joint choice, in a fixed canonical order. It is pure and fails closed on any request field it does not know. Candidates whose legality depends on `maybeTrapped` / `maybeDisabled` / `maybeLocked` are `pending`. `legalChoiceSet` has a `ChoiceJudge` check every non-pending one; it never asks about pending ones, so no hidden-information verdict reaches a policy. `CloneJudge` judges on `Battle.fromJSON(battle.toJSON())`, never on the live battle, and refuses unless the battle is at the request's `request_hash`. `runBattle` builds one per request (A1); the corpus and tests use it too. `Battle.fromJSON` rebuilds `activeRequest` from the Pokémon, so the clone carries the live request over: after an `[Unavailable choice]` the live one was edited in place. `judgeSubmission` (the SubmissionJudge) classifies the engine's `|error|` after a submission. If the engine refuses a non-pending candidate, that is an `EnumeratorDefectError`. On A2 and in the advisor (`judge: null`), the engine judges only at submission.
- `choice-corpus.ts`: the golden choice corpus (ADR-006 rule 7). Each case is a seeded battle log that reaches one situation (`CASE_KINDS`: preview, target types, Mega, forced switches, trapping, Imprison, a fainted active, a locked move), plus every candidate with the engine's verdict. Verdicts on pending candidates are hidden state, kept for evaluation. CI replays each log (it never re-runs the search in `scripts/capture-choices.ts`) and compares the result. `missedActions` is the brute-force oracle: it judges every syntactic choice string and fails on any engine-accepted action the enumerator did not offer.
- `golden.ts`: the golden battle (each side plays `firstLegal`: the first judged legal choice), team-fixture and verdict builders shared by the tests and the capture script.

**Fixtures** live only under `fixtures/` (CC0, synthetic or sanitized, with provenance). `fixtures/showdown/<format>/` holds the golden request fixtures plus `battle.summary.json` (the log hash and choices), and `choices/` the choice corpus. `fixtures/teams/<format>/` holds synthetic PokePaste teams, a `manifest.json` (provenance, expected verdict and reason) and `verdicts.json` (the validator's log). CI re-captures and compares the request fixtures, summary and verdicts, so engine behaviour drift fails the build. Pastes keep Showdown's trailing spaces (`.editorconfig`, `.gitattributes`).

**Repo guards** (`scripts/`, Python): `check_no_pokepaste.py` rejects PokePaste-shaped files outside `fixtures/` (exact-path exemptions in `.pokepaste-guard-allow`). `check_licenses.py` fails closed on any production dependency licence outside the allowlist. `licences-selftest.sh` (`make licences-selftest`, CI only: it needs network) injects an AGPL `file:` dependency into a scratch copy and requires `make licences` there to fail. `.github/CODEOWNERS` puts the gates under code-owner review.

**CI** (`.github/workflows/ci.yml`) runs `python`, `typescript` (submodule checkout + `make showdown`), `team-guard`, `licences`, `gitleaks`, `osv` and `dco`. The aggregate `ci` job is the required check on `main`, together with CodeQL.

Planned runtimes (ADR-004/005, not built yet): A2 is a bot client on a local Showdown server for sparring; the Chromium MV3 advisor extension is read-only and sees only the player's own request.
