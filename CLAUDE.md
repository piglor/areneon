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
make migrate-check  # the migration parity suite (#1466, docs/runbooks/migration.md): every golden, parity and differential check in dependency order, a summary of what drifted; never rewrites a golden (`MIGRATE_ARGS=--recapture` rewrites and lists changed files; `--only ID`, `--list`); run green on the current pin before a pin bump (#1103) or a new backend (#1457)
make contracts      # contract schemas fit their ADR-008 profile; generated TS types and pydantic models have no drift
make contracts-update  # regenerate the TS types and pydantic models after a schema change (commit the result)
make guards         # team guards (PokePaste text, JSON data_class) + production licence allowlist
make db-up db-test  # rules store on a fresh local Postgres 18 (127.0.0.1:55432): roles, migrations, tests
make db-down        # remove that container
scripts/check-evaluation-store.sh --store DIR --owner USER  # owner only: read-only isolation checks of the evaluation store (docs/runbooks/evaluation-store.md)
make db-bootstrap-test  # bootstrap.sql on its own throwaway Postgres: refuses missing/blank passwords, sends none outside its CREATE ROLE
make importer-e2e   # build the importer image, import the recorded releases into a fresh db-up DB twice (commit, then no-op); db-up again before db-test
RULES_DB_PORT=55433 RULES_DB_COMPOSE_PROJECT=areneon-e2e make importer-e2e  # the same beside the default DB (db-up/db-test/db-down honour both too)
```

Single tests. The coverage thresholds apply per file and fail on a subset, so run without coverage:

```bash
pnpm exec vitest run packages/showdown-bridge/test/runner.test.ts        # one file
pnpm exec vitest run packages/showdown-bridge/test/request.test.ts -t "forced switch"
uv run pytest tests/scripts/test_check_licenses.py --no-cov               # add ::test_name for one
```

Reducer sweep (#919; reads a verified dataset, writes only to `--out`; the actor mixes include `uniform-all`, a sampler over every candidate pending ones included, which is what a random-initialised policy is; about 0.15 s per battle after a 30 s pool load; the check takes about 20 ms per request):

```bash
node packages/team-dataset/scripts/reducer-sweep.ts --config RUN_CONFIG.json --mix random-legal|pending-first|rule-vs-random|uniform-all|uniform-vs-random --battles 1100 [--start 0] [--run-key sweep-1] [--only-ability Illusion] [--only-move Transform,Copycat] --out OUT.jsonl
uv run python scripts/reducer_sweep.py OUT.jsonl...   # reducer and encoder at every request; each refusal kind with a count and its shortest example; exit 1 on any refusal (nothing is excluded; the Illusion `|replace|` is reduced, #851) and on an input that checked nothing; the play script exits 2 on an unknown flag, a missing value, an unknown mix or a `--battles` below 1; `--only-ability Illusion` plays only teams that hold the ability and ignores the config's `excluded_abilities` (the way to sweep Illusion battles); `--only-move a,b` plays only teams that hold one of the moves (names or ids; the Transform sweep of #938)
```

Re-capture golden fixtures and team verdicts (only after a deliberate Showdown pin bump or engine-facing change; review the diff):

```bash
node packages/showdown-bridge/scripts/capture-golden.ts
node packages/showdown-bridge/scripts/capture-choices.ts   # golden choice corpus; also after an enumerator change
node packages/showdown-bridge/scripts/capture-choice-channels.ts   # the player channel of each choice-corpus case (`fixtures/showdown/<format>/choice-channels/channels.json`, read by tests/ml/model/test_target.py); after capture-choices.ts or a pin bump
node packages/showdown-bridge/scripts/capture-samples.ts   # shared contract sample corpus; also after a contract change
node packages/showdown-bridge/scripts/export-team-vocabulary.ts   # golden team vocabulary (about 10 s); also after a vocabulary-export change; run before capture-samples.ts, which reads it
node packages/showdown-bridge/scripts/export-role-sets.ts   # golden role-set table (ADR-020); after a pin bump or a role-set export change, once the team vocabulary is current
node packages/showdown-bridge/scripts/export-move-classes.ts   # golden move-class table (the move-use diagnostic); after a pin bump or a class-rule change
node packages/showdown-bridge/scripts/export-generator-prior.ts --commit <40 hex>   # config/opponent-model/<format>/generator-matched.<config_id>.json (ADR-025 §10); after a generator config change or a deliberate change of the proposal's draw rules (the commit of that code)
node packages/showdown-bridge/scripts/export-role-sets-prior.ts   # config/opponent-model/<format>/role-sets.fill-<config_id>.json (ADR-025 §2); after a role-set table, fill-prior or `role-sets-prior.ts` change; review the diff
node packages/showdown-bridge/scripts/capture-random-corpus.ts   # random-play channel corpus (needs a built engine); also after an engine-facing change
node packages/showdown-bridge/scripts/capture-open-regime-corpus.ts   # open-regime channel corpus `fixtures/showdown/gen9championsvgc2026regmcbo3/open-regime/` (ADR-027 slice 1: seeded A1 battles of the open-sheet format, one game each, on the random corpus's teams); after a pin bump, an engine-facing change or a change of the random corpus's first eight battles
node packages/showdown-bridge/scripts/capture-unavailable-corpus.ts   # channels with `|error|[Unavailable choice]` lines (pending-first play on the fixtures-v1 teams); then rewrite the reduced snapshots (tests/ml/reducer/test_unavailable.py)
node packages/showdown-bridge/scripts/capture-search-fidelity-corpus.ts [--extend] [--dataset DIR]...   # the targeted fidelity corpus `fixtures/showdown/<format>/search-fidelity/` (ADR-026 slice T); after a pin bump, an engine-facing change or an edit of its `targets.json`; DIR is a `generate` output whose dev split supplies teams (see its README); `--extend` keeps the committed battles and adds only what a short target still lacks (#1187)
node packages/showdown-bridge/test/fidelity/report-cli.ts --json fixtures/showdown/<format>/search-fidelity/fidelity-report.json > fixtures/showdown/<format>/search-fidelity/fidelity-report.md   # the full fork-oracle fidelity report (ADR-026 slice F; about 5 minutes, test tooling); after a corpus, pin or record change (a test fails on a stale `inputs`), then edit `admitted` in config/search/world-writers.v1.json to the report's admissions
node packages/team-dataset/scripts/differential-report.ts [--part ID --parts DIR | --merge DIR]   # the differential harness's self-differential report `fixtures/showdown/<format>/differential/self-report.{json,md}` (ADR-030 G1: the pinned probe against itself over every query the feature builders issue on the committed corpora; about half an hour of probe work, `--part` per corpus in separate processes, then `--merge`); after a pin bump, a builder, probe, corpus or harness change (a test fails on stale `inputs`)
node packages/team-dataset/scripts/differential-battle.ts --battles 2000 [--start 0] [--chunk 50] [--max-turns 100] [--dataset DIR] [--out FILE]   # the battle-level differential's size check (ADR-030 G2, docs/architecture/differential.md): a fresh corpus of seeded battles on generator-pool teams (dev and training, never validation) across uniform-all, random-legal and pending-first, each recorded as a `BattleTrace` on the unmodified engine and replayed with its draws forced; prints differences, entity and interaction-class counts, wall time and trace bytes per battle; writes nothing but --out; exit 1 on any difference
node packages/showdown-bridge/scripts/capture-illusion-battle.ts   # the Illusion battle (a Zoroark on both sides whose disguises break); then rewrite the reduced snapshots with `ARENEON_UPDATE_GOLDEN=1 uv run pytest tests/ml/reducer/test_illusion_battle.py --no-cov`
node packages/showdown-bridge/scripts/capture-fling-battle.ts   # the Fling of an item already gone (`-enditem` with no name, #1162); then rewrite the reduced snapshots with `ARENEON_UPDATE_GOLDEN=1 uv run pytest tests/ml/reducer/test_fling.py --no-cov`
node packages/showdown-bridge/scripts/capture-baton-pass-battle.ts   # the Baton Pass battles (stages passed on a `[from] Baton Pass` switch, #1099; a passed Substitute, #1102); then rewrite the reduced snapshots with `ARENEON_UPDATE_GOLDEN=1 uv run pytest tests/ml/reducer/test_baton_pass.py tests/ml/reducer/test_baton_pass_volatiles.py --no-cov`
node packages/showdown-bridge/scripts/mechanics-inventory.ts   # the mechanics coverage inventory, its weights report and docs/mechanics-inventory.md (#1465); after a pin bump or an export change; `--dataset DIR [--out FILE]` weights over fixtures-v1 + a generated dataset's dev and training splits, `--run-dir R --generations 100,101` run counts printed (read only, synthetic Episodes, never committed)
node packages/showdown-bridge/scripts/export-volatile-copy.ts   # which volatiles a Baton Pass / Shed Tail hands over (`noCopy` in the engine's condition data, #1102); after a pin bump
node packages/showdown-bridge/scripts/export-engine-data.ts # engine-data table + hash (ADR-010); after a pin bump
node packages/showdown-bridge/scripts/export-engine-data-2.ts # engine-data version 2 table + summary (ADR-010 erratum A4 item 7); after a pin bump or an export change
PYTHONPATH=.:packages/schema/python python -m scripts.derive_usage_prior --inputs config/opponent-model/<format>/smogon-chaos.<month>-<cutoff>.inputs.json --raw-dir .tmp/chaos --out config/opponent-model/<format>/usage-prior.<month>-<cutoff>.json [--report R.json] [--fallback]   # the Smogon usage prior (ADR-025 §2, #1120; `--fallback` #1209: fill the species the source lacks from the generator-matched prior); the raw files are downloaded by hand into the gitignored .tmp/chaos and checked against the input record's size and SHA-256
node packages/opponent-model/scripts/usage-prior-report.ts PRIOR.json   # the move design's largest deviation from a prior's stated marginals (derivation report)
```

Generate a synthetic team dataset (ADR-013; the pilot of 40,000 draws takes about a minute, then about 1 ms per draw; `pilot` records the pilot alone, `generate --pilot FILE` reuses it; output goes to the `--out` directory and is not committed unless it is a deliberate fixture):

```bash
node packages/team-dataset/scripts/generate.ts generate --config config/teams/gen9championsvgc2026regmc/generator.json --labels config/teams/gen9championsvgc2026regmc/labels.json --format gen9championsvgc2026regmc --out OUT_DIR --dataset-id ID --generation-key HEX64 --assignment-key HEX64 --dev 100 --validation 100 --training 1000 [--reference earlier/records.dev.jsonl ...]
node packages/team-dataset/scripts/sealed.ts <generate|reveal-hash|retire|verify> ...   # owner only, inside the isolated store (docs/runbooks/evaluation-store.md section 9)
node packages/team-dataset/scripts/duckdb-writer.ts write --dataset OUT_DIR --labels config/teams/gen9championsvgc2026regmc/labels.json --out SNAP_DIR --shard-key HEX64 --snapshot-id ID [--splits dev,validation] [--require-pinned]   # Parquet shards + snapshot.json (needs the DuckDB CLI: DUCKDB_BIN or PATH)
node packages/team-dataset/scripts/duckdb-writer.ts verify --snapshot SNAP_DIR/snapshot.json --shards SNAP_DIR --dataset OUT_DIR --labels ...   # shards back to the dataset's records, byte for byte
DUCKDB_BIN=... node packages/team-dataset/scripts/measure-shards.ts OUT_DIR SNAP_DIR [--split dev]   # write/read time and memory
node packages/team-dataset/scripts/check-team-labels.ts   # in make ts and CI: label sets and generator configs under config/teams/ against the pinned engine
```

TypeScript runs directly under Node's type stripping (`.ts` imports, `erasableSyntaxOnly`): no build step, no parameter properties, enums or namespaces.

## Architecture

The full subsystem documentation lives in `docs/architecture/` (index: `docs/architecture/README.md`). Read the doc of the area you touch before changing it; this file holds only what every agent must hold in mind.

### Layout

**Polyglot monorepo** (ADR-002): TypeScript in `packages/*` (pnpm workspace), Python in `scripts/`, `tests/`, `ml/areneon_ml` (policy worker, reducer, encoder, networks, learner, loop, progress evaluation, inspector) and the generated-contracts package `packages/schema/python` (uv workspace members), Rust in `crates/` once the first crate lands, `services/` later. `packages/schema` holds the JSON Schema contracts: the rules records (`rules/`, ADR-007) and the cross-language contracts (`contracts/v<N>/`, ADR-008), from which the TS types (`generated/ts/`) and the pydantic models (`python/areneon_contracts/models/`) are generated and checked in. `fixtures/` (CC0, synthetic or sanitized) is the only place for committed data; `config/` holds reviewed, versioned, hashed data (rules never live in code constants).

### Invariants every agent must hold

1. **The pinned engine is the only legality and mechanics authority** (ADR-003). `vendor/pokemon-showdown` is a submodule pinned to an exact SHA, upstream and unmodified. Areneon asks the engine (`TeamValidator`, `side.choose`, the mechanics probe) and fails closed on anything unknown. Game rules, move/item/ability lists and numeric values come from the engine or from reviewed, hashed data records under `rules/` and `config/`, never from code constants. Only `packages/showdown-bridge` talks to the engine. A change to the pin, the request decoder or the legal-choice enumeration re-runs the golden fixtures (see Commands). Do not copy AGPL code (Showdown client, Champions Calc); `@smogon/calc` is an oracle in one test only.
2. **Player perspective only.** A policy, observation, encoder, belief, search or feature input is a pure function of ONE player's own channel (the stream that player received, branded `PlayerChannel`), that player's own team and the version manifest. The omniscient log, hidden verdicts (`CloneJudge` verdicts on pending candidates), the battle seed, the opponent's team and any `|split|` secret half never reach a policy input, an observation, a log line given to a worker, a checkpoint or an Episode. Enforce this by types (branded channel types, no field for the forbidden input), by invariance tests (equal player channel gives equal output whatever the opponent holds unrevealed) with a negative control, and by canary tests. New consumers of battle data follow the same pattern.
3. **Privacy and data classes** (ADR-008, ADR-009, ADR-015). Every record carries a `data_class` (`synthetic` or `private`). Real competitive teams are never committed; they live only in the private bucket and, on the owner's workstation, under gitignored `private-data/`. Private data never reaches a public dataset, checkpoint, Episode, cloud worker or training stream; `loadTrainingSet`, `readEpisode`, `stampTeam`/`teamHash` provenance and the guards (`scripts/check_data_class.py`, `check_binary_files.py`, `check_no_pokepaste.py`, gitleaks) fail closed. A team is `synthetic` only if its hash is in a verified synthetic set, never by a caller's label. Evaluation-sealed data and omniscient game records are evaluation artifacts, never training input. Error messages on private paths state a rule and a position, never a value.
4. **Determinism and reproducibility.** Same commit, config, seeds and inputs give byte-identical outputs: seeded PRNGs only (never `Math.random`), RFC 8785 canonical JSON (`jcs`) for every hashed record, integer arithmetic in contracts and in anything hashed (ppm integers, no floats), no wall-clock value in a committed record. Training and evaluation claims are per machine and software (CPU, fixed thread count, deterministic algorithms); a CUDA learner is not bit-reproducible and nothing is claimed across machines or torch builds. Evaluation runs one query per forward pass, never batched.
5. **Identity and format constants.** Every behaviour that changes what a hashed identity covers is named by a constant (examples: `BELIEF_FORMAT`, `ENCODER_FORMAT`, `MECHANICS_ADAPTER_FORMAT`, `WORLD_BUILDER_FORMAT`, `WORLD_FACTS_FORMAT`, `REPLY_MODEL_FORMAT`, `DAMAGE_VALUE_FORMAT`, `LEAF_FORMAT`, `SPEED_CHECK_FORMAT`, `DAMAGE_CHECK_FORMAT`, `OPPONENT_SHEET_FORMAT`, `BELIEF_FEATURES_FORMAT`, `HYPOTHETICAL_BATTLE_FORMAT`, `PAYOFF_LEAF_FORMAT`, `SOLVER_FORMAT`, `FIDELITY_FORMAT`, `GENERATOR_PRIOR_SCRIPT`, `STATE_SCHEMA_VERSION`). Bump the constant when a value is computed differently for the same input (a change the hash cannot see); a golden test then fails until the fixture is rewritten deliberately and reviewed. A pin bump, probe change, reader change, assumption-record change or new feature set is a new identity, a new hash and a new run: never edit a record a run has used (a new version is a new file); a record no run has used yet may be edited in place and its dependent hashes, vectors and corpus samples are regenerated (`node packages/showdown-bridge/scripts/capture-samples.ts`). Never edit `packages/public-state/src/interpreter.cjs` for a feature: its hash (`readerSha256`) moves every baseline, feature-set and belief identity.
6. **Contracts follow the ADR-008 profile.** A contract's `$id` (`urn:areneon:schema:<contracts|engine>:v<N>:<name>`) declares its profile; `src/profile.ts` enforces a keyword whitelist: closed objects, bounded integers (no `number`), local `$ref`s, `kind`-tagged `oneOf` (`engine` schemas copy the pinned engine's wire shapes and also allow scalar type unions). Additive optional fields are a minor change within the version (a reader older than the field rejects such a record, so generated types, models and samples ship with the schema change; absent fields keep every byte and hash); anything else is a new `schema_version`. Reader rules the profile cannot say (sorted and unique lists, sums, references that resolve, recomputed hashes) live in a TS reader and its Python twin, held equal by the shared sample corpus `fixtures/contracts/` (valid, invalid and `reader-invalid/` samples plus JCS digests in `hashes.json`): Ajv and python-jsonschema must classify it identically and every language must reproduce the digests. `make contracts` fails on a profile violation or generated types that differ from a fresh run; `make contracts-update` regenerates them.
7. **Fail closed, state limits.** Unknown line types, fields, request keys, ids or shapes raise a typed error; nothing is silently skipped or guessed. A limit or approximation is documented where it lives (the doc of its subsystem) and tested. Fixtures are CC0, synthetic or sanitized, with provenance; CI re-captures golden fixtures so engine drift fails the build.
8. **Coverage and gates.** Every new line must be covered by a test (90% per file) or be removed; see `AGENTS.md` for the local gates and the Commands section for single-test invocations. Hooks run on every commit and are never bypassed.

### Package map

One or two sentences per area; each links to its doc under `docs/architecture/`.

- [`contracts.md`](docs/architecture/contracts.md): `packages/schema` (JSON Schema contracts, generated TS types and pydantic models, `fixtures/contracts/` sample corpus) and `fixtures/`.
- [`showdown-bridge.md`](docs/architecture/showdown-bridge.md): `packages/showdown-bridge`: engine loading, rules/formats, request decoding, `runBattle` (A1), `enumerateCandidates`/`CloneJudge`, golden fixtures.
- [`observation-episodes.md`](docs/architecture/observation-episodes.md): `observation.ts`, `episode.ts`, `own-team-stats.ts`, `training-loader.ts`, `team-class.ts`: the privacy-critical path from a channel to training data.
- [`engine-data-vocabulary.md`](docs/architecture/engine-data-vocabulary.md): `engine-data.ts`, `team-vocabulary.ts`, `role-sets.ts`: hashed vocabularies derived from the pinned engine.
- [`policy-worker.md`](docs/architecture/policy-worker.md): `policy-protocol.ts`, `policy-worker.ts`, `policy-actor.ts` and `ml/areneon_ml/worker`.
- [`reducer-encoder.md`](docs/architecture/reducer-encoder.md): `ml/areneon_ml/reducer`, `model`, `checkpoint`: clean-room reducer, encoder (+ options), networks, checkpoints.
- [`mechanics-probe.md`](docs/architecture/mechanics-probe.md): `packages/showdown-bridge/src/mechanics-*.ts`: the engine reads every stat, speed and damage value; Areneon computes none, except the damage kernel for training games on admitted content (ADR-030).
- [`opponent-model.md`](docs/architecture/opponent-model.md): `packages/opponent-model` and the prior derivations: belief over the opponent from the player view, the sampler, evidence, usage priors, sheet block.
- [`opponent-sheet.md`](docs/architecture/opponent-sheet.md): ADR-027: opponent sheet block, builders, open/closed sheet regimes.
- [`damage-aware.md`](docs/architecture/damage-aware.md): `damage-aware@v1` to `v7`: inputs, actor, v2 over belief worlds, v3 engine choice location, v4 to v7 situational and success-chance scoring, identity and the opponent wiring.
- [`search.md`](docs/architecture/search.md): World facts, world builder and writers, fidelity oracle, search worlds, `SearchConfig`, the search actor.
- [`battle-state-tensors.md`](docs/architecture/battle-state-tensors.md): `battle-state-spec.ts`, `config/engine-state/`: the `BattleStateSpec` (ADR-030 step 5 groundwork): battle-state tensor fields with engine sources, visibility classes, projection rules and the encoder/reducer map; no engine yet.
- [`search-distillation.md`](docs/architecture/search-distillation.md): ADR-026 E2: search labellers, label stream, distillation.
- [`population.md`](docs/architecture/population.md): ADR-021: population loop, ladder, Bradley-Terry rating, copy and explore, health, PFSP.
- [`team-preview.md`](docs/architecture/team-preview.md): ADR-029: solver, hypothetical preview battle, payoff table, best-response gain.
- [`features.md`](docs/architecture/features.md): ADR-022: candidate feature builders, wire fields, encoder use and cost gates.
- [`damage-kernel.md`](docs/architecture/damage-kernel.md): `damage-kernel*.ts`, `config/backends/<format>.damage-kernel.{effects.json,admission.jsonl}`, `fixtures/backends/<format>/damage-kernel-targeted.json`, `differential-kernel.ts`: ADR-030 step 3's clean-room integer damage kernel behind the probe interface, training games only, admitted by byte-equality; not wired into any run yet.
- [`training-loop-learner.md`](docs/architecture/training-loop-learner.md): `ml/areneon_ml/learner`, `loop`: PPO learner, cloning, generation loop, run metrics, teacher data.
- [`league.md`](docs/architecture/league.md): `league-seed.ts`, `league-run-record.ts`.
- [`progress-evaluation.md`](docs/architecture/progress-evaluation.md): `ml/areneon_ml/progress`, `report`: progress evaluation, head to head, scripted h2h, diagnostics, dashboard.
- [`evaluation-protocol.md`](docs/architecture/evaluation-protocol.md): `evaluation-register.ts`, `evaluation-plan.ts`, `sealed*.ts`: records, register chain and sealed sets.
- [`team-dataset.md`](docs/architecture/team-dataset.md): `packages/team-dataset`: generator, records, shards, working database.
- [`private-teams.md`](docs/architecture/private-teams.md): `packages/private-team-store`, owner-session guard, private test: the owner-only path for real teams.
- [`inspector-replay.md`](docs/architecture/inspector-replay.md): `packages/public-state`, `replay-view`, `ml/areneon_ml/inspector`: offline inspection pages and replays.
- [`rules-store.md`](docs/architecture/rules-store.md): `packages/rules-store`, `rules-importer`, `db/`.
- [`repo-guards-ci.md`](docs/architecture/repo-guards-ci.md): `scripts/check_*.py`, `.github/workflows/ci.yml`.
- [`a2.md`](docs/architecture/a2.md): `packages/bot-client`, `server-lines.ts`, `session.ts`.
