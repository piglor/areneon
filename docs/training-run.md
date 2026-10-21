# Starting a training run on a generated team dataset

The generation loop draws its teams from the interim pool of committed fixture
teams unless the run config names a generated dataset (`team_dataset` in the
`TrainingRunConfig`: a directory and the SHA-256 of its manifest). The pool then
holds the dev and training teams of the dataset, never the validation teams, and
each team is read back through the dataset reader (the pinned engine validates
it, every hash is recomputed). A run without `team_dataset` plays the interim
pool exactly as before.

Notes on how the teams are used:

- A set that names a Mega forme starts a battle as its base forme (the engine's
  validator normalises it); the pool holds that form, and the cumulative team set
  lists its hashes.
- `generation.excluded_abilities` of the run config is applied to the pool.
- The dataset is generated without `--reference`, so it deduplicates against
  nothing; its own validation split is the one the pool refuses duplicates of.

## 1. Generate the local training split

The training split is never committed. Generate a dataset into the gitignored
`.tmp/` (about a minute for the pilot, then about 1 ms per draw). The keys are
public values for a public dataset; pick fresh 64-hex strings for a new dataset.

```bash
node packages/team-dataset/scripts/generate.ts generate \
  --config config/teams/gen9championsvgc2026regmc/generator.json \
  --labels config/teams/gen9championsvgc2026regmc/labels.json \
  --format gen9championsvgc2026regmc --out .tmp/datasets/train-v1 \
  --dataset-id train-v1 --generation-key HEX64 --assignment-key HEX64 \
  --dev 200 --validation 200 --training 20000
```

Verify it and print the manifest hash:

```bash
node packages/team-dataset/scripts/dataset-hash.ts --dir .tmp/datasets/train-v1 --format gen9championsvgc2026regmc
```

## 2. Configure the run

`config/training/public-gpu16.json` is sized for a 16 GB GPU (6 layers of width
256, 256 self-play games and 64 games against each scripted baseline per
generation, 8 games in parallel). Copy it, then change:

- `seeds.run_key`: a fresh name for every run (it names the battle seeds, the
  team draws and each Episode's collection record);
- `team_dataset.manifest_hash`: the hash printed above (`path` is relative to
  the repository root, where the driver runs).

The config is hashed as a whole, so a change is a new run.

## 3. Start the run

The learner runs from the separate GPU environment (`ml/gpu-env/README.md`); the
policy worker stays on CPU torch from the root environment.

```bash
ml/gpu-env/.venv/bin/python -m areneon_ml.loop run --config my-run.json \
  --run-dir .tmp/runs/my-run --engine-data fixtures/showdown/gen9championsvgc2026regmc/engine-data.json \
  --generations 10 --worker-python .venv/bin/python
```

A run with a `team_dataset` starts `packages/team-dataset/scripts/play-generation.ts`
for each generation; one without starts the bridge's own script. The driver
resumes a killed run from the last committed generation.
