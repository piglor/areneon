# Migration parity suite

Run `make migrate-check` before any migration starts and again after each step:
the Regulation M-D pin bump ([#1103](https://redmine.piglor.com/issues/1103), whose checklist
this runbook serves) and any new simulation backend (ADR-030, #1457). The rule (owner,
2026-11-20): the checks that prove a migration safe exist and run green on the current pin
before the migration begins. The suite is `scripts/migrate_check.py`; it reuses the
repository's own tests and scripts and reimplements none of them.

## Running it

```bash
make migrate-check                              # compare only; never rewrites a golden
make migrate-check MIGRATE_ARGS="--only engine-tables --only mechanics-probe"
make migrate-check MIGRATE_ARGS=--list          # the checks and their order
make migrate-check MIGRATE_ARGS=--recapture     # rewrite goldens, compare, list changed files
python -m scripts.migrate_check --json out.json # machine-readable results (any python with the repo on PYTHONPATH)
```

It needs the built engine (`make showdown`), the installed toolchain and a `.venv` at the repository root (the TS parity tests spawn it as their Python reference; a worktree symlinks it). It runs at
`nice -n 19` unless `--no-nice` is given. The summary lists, per check, a status and what
drifted:

| status | meaning |
| --- | --- |
| `PASS` | every command of the check passed and no golden changed on disk |
| `DRIFT` | a command failed; the lines name the failing tests, which name the golden |
| `REWRITTEN` | the check passed but a tracked file under `fixtures/`, `config/`, `docs/`, `ml/areneon_ml/reducer/data`, `packages/*/generated` or `packages/*/src` changed while comparing (a defect in the check); detected with `git status` hashes before and after each check |
| `SKIPPED` | a test in the check was skipped (vitest JSON report, pytest `-rs`); a skip is not evidence, so the check is red |
| `ERROR` | the check could not run: interpreter (`--python`, default the running one; `.venv` at the root for the TS parity tests) or built engine missing, timeout (the whole process group is killed), a recapture command failed |
| `ABSENT` | a check listed but not built yet (none at present) |

Compare mode scrubs write-enabling variables (`ARENEON_UPDATE_GOLDEN`, `ARENEON_PRINT_*`, `ARENEON_HISTORY_PARITY_DUMP` and any `ARENEON_UPDATE|RECAPTURE|DUMP|WRITE*`) from the environment, and sets `ARENEON_PYTHON`, `ARENEON_REQUIRE_PYTHON_WORKER=1` and `CI=1` so a missing interpreter fails instead of skipping. Test files are named exactly: vitest's JSON report must list every named file.

The exit code is 1 on any `DRIFT`, `REWRITTEN`, `SKIPPED` or `ERROR`. A diff is a review item: the
compare run never writes a golden. `--recapture` runs each check's recapture commands (the
"Re-capture golden fixtures" list of CLAUDE.md), then compares, and prints every golden whose
bytes changed so each diff can be reviewed with `git diff` before it is committed.

## Order and what each check proves

1. `pin-and-rules`: the pinned commit constant equals the submodule gitlink, rules records and
   channels name this pin.
2. `golden-fixtures`: a fresh capture of the request fixtures, team verdicts, battle summary
   and channels equals the committed ones.
3. `choice-corpus`: the choice corpus replays, the brute-force `missedActions` oracle finds no
   action the enumerator missed, choice channels replay.
4. `engine-tables`: engine-data v1 and v2, team vocabulary, role sets, move classes, volatile
   copy, stall moves, history facts and the mechanics inventory re-export byte for byte; the
   protocol-volatile literal scan.
5. `mechanics-probe`: the speed, damage and KO fixtures, the real-battle cross-checks, both
   config checkers and the parity report.
6. `calc-differential`: the probe against `@smogon/calc`.
7. `reader-parity`: shared reader against the Python reducer, world facts and history facts
   parity, reduced goldens.
8. `battle-state-spec-oracle`: the `BattleStateSpec` (ADR-030) records v1 and v2 against the
   engine, the reducer map and the encoder map; the G4 reference packer and projection of v2
   bit-equal to the encoder at every committed request (the four corpus shards `battle-state-projection-corpus-{0..3}.python.test.ts`; the check's timeout is two hours), with the canary and the invariance test.
9. `encoder-golden`: encoder golden digests and `encoder_version`.
10. `a1-a2-equality`: A1 and A2 channels and observations equal, encoder pairs.
11. `fidelity-corpus`: the targeted fidelity corpus replays and its committed report is fresh.
12. `battle-differential`: every committed battle is recorded on the unmodified engine
    (`BattleTrace`, every draw forced) and replayed on a second engine object with no
    difference at any request, the broken battle backends are caught (ADR-030 G2).
13. `contract-corpus`: Ajv and python-jsonschema classify the corpus identically, generated
    types have no drift, digests reproduce.

The order is dependency order: a later check reads artifacts an earlier one proves (the
contract samples read the golden team vocabulary; role sets read the vocabulary).

## Expected drift on a pin bump

After the bump (`PINNED_SHOWDOWN_COMMIT`, the gitlink and the rules channel move together) run
`make migrate-check` and expect red. Each drift below is legitimate only if its reason holds;
anything else is a regression to investigate, not to recapture.

| golden | why it changes | how to review |
| --- | --- | --- |
| `capture.provenance.json`, `battle.summary.json`, request fixtures, channels | the engine commit is recorded; the log hash moves if engine behaviour or RNG consumption changed | the diff must be the commit id alone, or a battle-log difference you can name from the engine changelog |
| `choices/`, `choice-channels/` | enumerator inputs or engine verdicts changed | every changed case must have a reason in the request shape; a verdict flip on a non-pending candidate is an enumerator defect |
| `engine-data*.json`, `team-vocabulary*.json`, `role-sets*.json`, `move-classes*.json`, `volatile-copy`, `stall-moves`, `history-facts`, `mechanics-inventory*` | derived from the pinned data files | read the added, removed and changed ids against the engine diff; counts in the `.summary.json` files must match |
| encoder digests, `encoder_version` | the engine-data hash is in the version | the version string moves, tensor digests move with it; recapture only with the engine-data change explained |
| mechanics fixtures, calc divergences, parity report | the hand-worked values are fixed by paper; only the engine may move them | a changed hand-worked value is wrong until worked again on paper; triage a new calc divergence by the fixture, never the calc |
| `reduced/`, `random/`, `unavailable/`, `search-fidelity/` and its report | seeded battles replay differently if the engine's RNG use changed | recapture the corpora, then the fidelity report (about 5 minutes), then edit `admitted` in `config/search/world-writers.v1.json` to the report |
| `projection/points.json` | the engine state at each kept step is recorded, so a pin bump or a recorder change moves it | rerun `capture-projection-points.ts` (wired to `--recapture`) only after the corpus gate (`battle-state-projection-corpus-{0..3}.python.test.ts`) is green; the state may change, the decisions must not |
| (nothing: `battle-differential` keeps no golden) | traces are recorded fresh on each run, so a pin bump changes nothing it holds; a difference there is a recorder or backend defect, never a recapture | read the difference's kind, step and seat; the corpora it replays are recaptured by the rows above |
| contract samples, `hashes.json` | the capture reads the golden vocabulary; identity hashes (damage-aware, feature set, belief, search) move with the pin | review that only hash fields and engine-derived values changed |
| `docs/showdown-champions-gaps.md`, `docs/mechanics-parity.md`, `docs/mechanics-inventory.md` | generated from records, fixtures and exports | edit the records, never the docs |

## What `--recapture` rewrites, and the manual steps

Wired (run in this order inside their check, with `ARENEON_UPDATE_GOLDEN=1` set for the reduced
snapshot rewrites): `capture-golden` and `capture-random-corpus` (golden-fixtures);
`capture-choices` and `capture-choice-channels` (choice-corpus); the exports in
`engine-tables` (engine-data v1 and v2, team vocabulary, role sets, move classes, volatile
copy, stall moves, history facts, mechanics inventory); `mechanics-parity` (the generated
report); `capture-unavailable-corpus`, `capture-illusion-battle`, `capture-fling-battle`,
`capture-baton-pass-battle` and the reduced-snapshot rewrites (reader-parity);
`capture-projection-points` (battle-state-spec-oracle);
`capture-encoder-pairs` (a1-a2-equality); `capture-samples` (contract-corpus, after the
vocabulary export). The gap doc and the parity report are generated from records and
fixtures: edit those, never the docs.

Manual, in order, after `--recapture` (they need inputs or minutes that a one-command suite
should not hide): (1) `capture-search-fidelity-corpus.ts` (`--extend`, `--dataset DIR`; the
corpus README names the keys), (2) `test/fidelity/report-cli.ts --json
fixtures/showdown/<format>/search-fidelity/fidelity-report.json` redirected to
`fidelity-report.md` (about 5 minutes), then edit `admitted` in
`config/search/world-writers.v1.json` to the report, (3) `export-generator-prior.ts --commit <hex>`
and `export-role-sets-prior.ts` when the generator or role-set table moved, (4) the usage prior
derivations from the recorded Smogon inputs, (5) `capture-samples.ts` again if a step above
changed a contract input, (6) `make migrate-check` green.

## Reviewing a recapture

1. `make migrate-check MIGRATE_ARGS=--recapture` and read the changed-file list.
2. `git diff --stat`, then `git diff` per golden; confirm each against the table above.
3. Commit goldens in their own commit with the reason, then `make migrate-check` must be green.
4. Never recapture to silence a red check you cannot explain; the compare run exists to make
   that drift visible.

## Proof that drift is caught

`tests/scripts/test_migrate_check.py` doctors a golden and requires the suite to report the
drift and exit 1, requires a stub that ignores drift to fail the same assertion, and requires
a compare run that rewrites a golden to be reported `REWRITTEN`.
