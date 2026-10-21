# Inspector fixture run

A two-generation run of the tiny loop (a 16-wide network, three self-play games and one game against
each of two scripted baselines per generation, at most four turns each), kept so the training-data
inspector (`python -m areneon_ml.loop inspect`) has a real run to read. CC0, synthetic: the teams are
the interim pool's committed synthetic fixtures, the Episodes are synthetic and `cc0-synthetic`.

The run was played with the loop itself:

```bash
python -m areneon_ml.loop run --config run-config.json --run-dir RUN \
  --engine-data fixtures/showdown/gen9championsvgc2026regmc/engine-data.json --generations 2
```

and then pruned to what the inspector reads: `run-config.json`, each generation's `team-set.json`,
`plan.json`, `generation.json`, `league-run.json` and `episodes/`, and a `manifest.json` whose file
hashes are those of the files kept (the learner states and checkpoints are binary and are not
committed; the originals' checkpoint ids were dropped from the manifest). `pool.json` is what
`node packages/team-dataset/scripts/print-pool.ts --config run-config.json` prints for this config;
`tests/ml/inspector/test_node_pool.py` and `packages/team-dataset/test/pool-listing.test.ts` compare
it with a fresh run of that script.
