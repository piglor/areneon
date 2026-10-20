# Synthetic team datasets (dev and validation fixtures)

License: CC0 1.0 (see `fixtures/LICENSE`). Every team here is synthetic: drawn
from the pinned engine's own data by a seeded generator, never a real team.

## `fixtures-v1/`

The first public dataset (ADR-013 slice (c)). It holds the **dev** and
**validation** splits as JSON Lines. The training split is generated in the
same run (the pool is assigned once) but is not committed.

| File | What it is |
| --- | --- |
| `teams-dev.jsonl`, `teams-validation.jsonl` | One `TeamRecord` per line, canonical (RFC 8785) form. 120 teams each. |
| `manifest.json` | The `TeamDatasetManifest`: generator commit, config, label set and vocabulary hashes, version manifest, public keys, pilot counts and the quota plan of all three splits. |
| `hashes.json` | The sorted `record_hash` and `canonical_team_hash` lists of each committed split, and the manifest hash. A Parquet shard of the same split must list the same `record_hash`es. |
| `pilot.json` | The recorded pilot (40,000 draws): the per-cell hit counts the plan was made from. |
| `coverage.json` | The coverage report: cells, strata and the teams each split holds. |
| `run.json` | The inputs of the run: dataset id, requested sizes, generator commit and the two **public** keys. |

### Provenance

- Generator: `packages/team-dataset` at commit `b7cda7af2f1744ad4934d38fb379b1d3cfc3ffc2`.
- Config: `config/teams/gen9championsvgc2026regmc/generator.json`, hash `de2f06b2547de57401fb4072b172386f906f2988f73cf2d12a929b27b8c80def`.
- Label set: `config/teams/gen9championsvgc2026regmc/labels.json`, hash `78008091053b6398add3a3e15d67711082dd7627ea81d98b07adae9c3f5f26db`.
- Engine: Pokemon Showdown `7244fe8e28fa8bc4d0809ae30d183c608b62bb2b`, format `gen9championsvgc2026regmc`.
- Public keys (they derive from a label, so anyone can recompute them; they protect nothing, and a public dataset has no secret key):
  - generation key `372cc1a98b1ae8c29ea8fc7a166b84de1d9c4ad81e23e831d9af7713e73a3ce7` = SHA-256 of `areneon/team-fixtures/v1/generation`
  - assignment key `d0b56458ea47add3c8a5f577cd5d55562f8699849af72b59bb6be486410be726` = SHA-256 of `areneon/team-fixtures/v1/assignment`
- Every team passes the pinned engine's `TeamValidator`, round-trips through `importPaste` and `exportPaste`, and carries labels equal to the labeller's output. The reader (`packages/team-dataset/src/dataset-reader.ts`) checks all of it.

### Splits

`assignSplits` orders each stratum by HMAC-SHA256(assignment key, canonical team hash) and deals dev, validation and training by the per-stratum quotas, with at least `n_stratum_min` teams of each stratum in each split (ADR-010 E1). No team is in two splits. Validation is never trained on: the training pool refuses a validation record and any team that repeats one.

### Regenerating

```bash
node packages/team-dataset/scripts/fixtures.ts check          # about 5 s: reuses pilot.json
node packages/team-dataset/scripts/fixtures.ts check --full   # about 45 s: also reruns the pilot
node packages/team-dataset/scripts/fixtures.ts write --full   # remake the files (review the diff)
```

A test makes the same comparison, so a file cannot differ from a fresh run.
