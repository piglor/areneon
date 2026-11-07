# Population inspector fixture run

A two-generation run of the tiny population loop (two members, a 16-wide network, four seats per
member: two ladder pairs, one self-play pair, one past-pool pair and, for the one scripted anchor
pair per baseline, `actor-rule@v1` and `random-legal@v1`; at most 12 turns a game), kept so the
training-data inspector (`python -m areneon_ml.loop inspect`, `replay`) has a real population run
to read. CC0, synthetic: the teams are the interim pool's committed synthetic fixtures, the
Episodes are synthetic and `cc0-synthetic`.

The run has what the inspector shows: a copy (generation 1's ladder decides that `m0` is donor and
`m1` loser; at generation 2 `m1`'s new records name `m0`'s frozen pair as parent) and the frozen
past pool (generation 0's snapshots `m0@g0`, `m1@g0` are met in generation 1, generation 1's in
generation 2). It was played with the loop itself:

```bash
python -m areneon_ml.loop run --config run-config.json --run-dir RUN \
  --engine-data fixtures/showdown/gen9championsvgc2026regmc/engine-data.json --generations 2
```

The config is `fixtures/contracts/v1/training-run-config/valid/tiny.json` with the population
section of `valid/population.json` reduced to two members, shares of 500000 / 250000 / 250000
(ladder, self-play, past pool), a past pool that snapshots every generation, a copy that needs two
pairs and a lower bound above even odds (`ready_generations` 1, `n_min` 2, `margin_ppm` 0,
`alpha_ppm` 150000), baselines `actor-rule@v1` and `random-legal@v1` with two games each and
`max_turns` 12. Its run key `popfix-e` was picked from a handful tried: outcomes of untrained
networks are luck, and this one is a run in which the copy fired.

It was then pruned to what the inspector reads: `run-config.json`, each generation's
`team-set.json`, `past-pool.json`, and, from generation 1, `plan.json`, `generation.json`,
`league-run.json`, `ladder.json`, per member `members/<m>/records/{battle,preview}.json` and
`members/<m>/episodes/`, and a `manifest.json` whose file hashes are those of the files kept (the
learner states, checkpoints, streams and training sets are binary or large and are not committed;
the manifest's other facts, the sealed checkpoint ids and hashes, are the originals'). `pool.json` is
what `node packages/team-dataset/scripts/print-pool.ts --config run-config.json` prints for this
config.
