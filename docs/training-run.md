# Starting a training run on a generated team dataset

The generation loop draws its teams from the interim pool of committed fixture
teams unless the run config names a generated dataset (`team_dataset` in the
`TrainingRunConfig`: a directory and a dataset hash). The pool then
holds the dev and training teams of the dataset, never the validation teams, and
each team is read back through the dataset reader (the pinned engine validates
it, every hash is recomputed). A run without `team_dataset` plays the interim
pool exactly as before.

Notes on how the teams are used:

- A set that names a Mega forme starts a battle as its base forme (the engine's
  validator normalises it); the pool holds that form, and the cumulative team set
  lists its hashes. The stamped hash is over this normalised form, so a
  Mega-forme paste's raw hash differs from its stamped hash.
- Pastes that normalise identically collapse to one pool entry.
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

Verify it and print its dataset hash (SHA-256 of the manifest hash and the sorted record hashes of every split, recomputed from the records, so any swapped, added or removed record changes it):

```bash
node packages/team-dataset/scripts/dataset-hash.ts --dir .tmp/datasets/train-v1 --format gen9championsvgc2026regmc
```

## 2. Configure the run

`config/training/public-gpu16.json` is sized for a 16 GB GPU (6 layers of width
256, 256 self-play games and 64 games against each scripted baseline per
generation, 8 games in parallel). The encoder holds up to 8 moves per Pokémon
(`max_moves`), because Mimic, Sketch and Transform can show a Pokémon more moves
than its set holds; it refuses a Pokémon that shows more than `max_moves`, so a
run fails closed rather than dropping a move. Copy it, then change:

- `seeds.run_key`: a fresh name for every run (it names the battle seeds, the
  team draws and each Episode's collection record);
- `team_dataset.dataset_hash`: the hash printed above (`path` is relative to
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

Self-play is bound by the one policy worker and the one Node process that judges every
request. `--game-processes S` shares a generation's games over `S` Node processes and
`--policy-workers K` gives each its own `K` worker processes (for example `--game-processes 4
--policy-workers 3` on a 28-core machine). Both default to 1 and are flags of the driver, not
fields of the run config: the plan, the seeds, every Episode, the stream and the learner-state
hashes are the same for any `S` and `K` (a game stays on one worker, every worker is
deterministic and no draw depends on which process served a game). Each game process loads the
team pool itself, so memory and start-up CPU grow with `S`. Progress evaluation never
batches or shards.

After each generation commits, the driver writes `<run-dir>/metrics/g<NNNNNN>.json`: wall
clock per phase (play, load, train, write, seal), throughput, the learner's peak
accelerator memory (null on a CPU learner), device and threads. The file is outside the
generation manifest and every hash, so reruns stay byte-identical. A crash between the
commit and the write loses that generation's timings, and a resume does not write them
again; the run dashboard (`python -m areneon_ml.loop report`) says `not recorded` for it
and lists the SHA-256 of each metrics file among its inputs.

## 4. Look at a generation: the progress evaluation

`python -m areneon_ml.loop progress` plays the frozen pair of a committed
generation against the scripted baselines on the **validation** teams of the
run's team dataset, so you can see whether training moves. It is a **progress
view only**. It is not the sealed Training gate (#790): it makes no strength
claim, it never selects a checkpoint, and nothing in the run reads its output.
"Progress view on validation teams. Not the Training gate. Loss is not skill."

```bash
.venv/bin/python -m areneon_ml.loop progress --run-dir .tmp/runs/my-run \
  --engine-data fixtures/showdown/gen9championsvgc2026regmc/engine-data.json \
  --generation 5 --games 100 --record-games 8      # or --all
```

`--all` evaluates the committed generations whose number is a multiple of 5,
generation 0 included (0, 5, 10, ...), which is the default schedule (ADR-017
erratum E2); `--generation G` evaluates any one generation. A committed
generation is verified and skipped, never rewritten.

It reads the run directory and never writes into it. The results go to the
progress root, `<run-dir>.progress/` (or `--progress-dir`), one directory per
generation, committed atomically (a partial directory, `manifest.json` last, one
rename) and never overwritten: running the command again verifies what is there
and skips it.

- `progress.json` is the `ProgressEvaluationRecord`: integers only (wins, ties,
  losses, turns, forfeits by side and cause, one row per game).
- `games/<match>-<k>.json` are `ProgressGameRecord`s, the first `--record-games`
  games of every match: the seed, both teams, both seats' candidates and
  choices, and the engine's **omniscient** log. That log holds both sides'
  hidden information. It is for you to look at on a synthetic run: never a
  policy input, never training data (the training loader refuses it), and it
  stays under the progress root.

**What is played.** Four matches per generation: `random-legal@v1` and
`actor-rule@v1`, each against the pair in sample mode (temperature 1, as in
training) and in argmax mode. With `--previous-matches`, from generation 1 on,
there are two more, against the previous generation's frozen pair:
`previous-sample` (both pairs sample) and `previous-argmax` (both argmax), six
in all. They are off by default because a comparison of adjacent generations is
too noisy to show progress (ADR-017 erratum E2). A `previous-*` row names the
opponent's checkpoint ids and is a **seen, same-lineage comparison**: the two
pairs share a lineage, so it can cycle and it says nothing about a held-out
opponent. Generation 0 has no previous generation and plays four whatever the
flag says; the output says so. The previous pair is served by a second worker.
A committed generation is never rewritten, so the matches cannot be added to
it: evaluate into a new `--progress-dir` (the command reports `previous: none`
for a generation without them).
`--anchors 0,10,20` adds, for each listed generation earlier than the one evaluated,
an `anchor-g<A>-sample` and an `anchor-g<A>-argmax` match against generation A's
frozen pair (`--anchor-games` games each, even, default 200, so 100 pairs), each
served by a worker of its own; the checkpoint ids and policy ids of an anchor are
verified from generation A's own records when the evaluation is made and again when
a committed directory is verified. An `anchor-*` row is a **seen, same-lineage
comparison** too, and a directory committed without the anchors it was asked for
is kept and reported (`anchors: ... missing`).

**Every pair is played from both seats (ADR-017 erratum E2).** Validation pair
`k` is two games, `2k` and `2k+1`: the candidate on p1 holding team `a` against
the opponent on p2 holding `b`, then the opponent on p1 holding `a` against the
candidate on p2 holding `b`. The teams stay on their seats and the policies
swap, so the two games cancel the seat and the team advantage; both games use
the same `progress-seed/1` battle seed (the pair index names it; the sampling
draw still differs, because the seat is in its label). A match of `n` pairs has
`2n` games, so `--games` and `--anchor-games` are even, and **the unit of
analysis is the pair**. The pairs and seeds are the same for every opponent,
mode and generation, so two generations are compared on the same plan. At most
as many games per match as there are validation teams (214 for 214 teams: 107
pairs); more is refused, never wrapped. The run's `team-set.json` must not hold
a validation team, and no progress seed may equal a seed of a committed
training plan; either refuses the command. The seed check runs at evaluation
time against the plans committed then: a later generation's `plan.json` is not
covered by an earlier record, and an evaluation of generation 0 sees no plans
yet. The remaining risk is about 2^-128 per seed (the two derivations hash
different purpose strings). A request above the pairs the validation split
gives is refused before the worker starts, naming the limit.

**Records.** A new evaluation writes a `ProgressEvaluationRecordV2`
(`schema_version` 2, contract `progress-evaluation-record-2` beside the
unchanged version 1). Its games name their pair (`game >> 1`, also an optional
`pair` field of the stored game record, a minor change within v1, whose seed is
then the pair's). The opponent variants are `baseline`, `previous` and `anchor`
(`{kind, generation, policy_id, checkpoints}`); `plan.anchor_games_per_match` is
present exactly when anchor matches are, and `teams.pairs_used` is half the
largest games per match. A version 1 record (one game per pair, the candidate
alternating seats) stays readable as it is and is never rewritten or upgraded.
The pair-level interval and the dashboard and replay pages for version 2 are
separate tickets (#948, #951); until they land the printed summary and the dashboard still print the per-game Wilson
interval, which for a both-seats record is not valid (the two games of a pair
are not independent): read the counts, not that interval.

**Team Preview ablation (#940).** `--preview-ablation` with
`--generation G` (1 or more; not `--all`; no `--anchors` or `--previous-matches`)
plays generation G's **Battle network** with three Team Previews instead of the
usual matches: (i) its own learned Preview, (ii) a uniformly random Preview and
(iii) generation 0's learned Preview, each in argmax against `random-legal@v1`
and `actor-rule@v1`: six matches `preview-<variant>-<baseline>-argmax`
(`learned`, `random`, `g0`), on the same validation pairs and seeds and from both
seats as every other match. The random Preview is a uniform pick among the legal
Preview choices the engine listed in the candidate's own request (no rule is
written in the code), seeded per seat and game from the policy stream label
(`random-preview@v1`, protocol D9); the worker is not asked for the Preview, and
it never sees opponent information. Generation 0's Preview needs a worker of its
own that holds that Preview and G's Battle. Use a progress directory of its own
(`--progress-dir`): it holds one `g<NNNNNN>/` per generation like any other. The
record is a `ProgressEvaluationRecordV2` whose matches carry an optional
`candidate_preview` (`learned`, `random` or `generation` with its checkpoint id);
a candidate's policy id is built from that Preview (`random-preview@v1` for the
random one), and an earlier generation's Preview is checked against that
generation's own records. The command prints, per baseline and variant, the mean
pair score in parts per million and, for the random and the earlier Preview, the
paired difference to the learned one on the same pairs: the mean of `d_k`
(variant minus learned) and how many pairs were better, equal or worse. All
integers, conditional on the fixed validation plan; the interval over pairs is
the pair-level interval's own ticket. With 100 pairs a Preview matters only if
the difference is clear of what the pair-to-pair spread allows: the counts of
better and worse pairs are the first thing to look at. A progress view, never
the Training gate.

**Reading the numbers.**

- The record stores counts. A **score** counts a tie as half a win:
  `(wins + ties / 2) / games`. The command prints it, with its **Wilson 95%
  interval**, in parts per million (`score_ppm`, `wilson_low_ppm`,
  `wilson_high_ppm`); the interval is derived, never stored. For a version 1
  record at 100 games a score of 0.5 has an interval of about 0.40 to 0.60.
- Every interval is **conditional on the fixed validation plan**: it describes
  these validation teams under this pairing and these seeds, not the team
  generator's whole distribution. That is a labelled deviation from protocol
  6.2, which allows Wilson only for fresh team pairs from the generator (a
  protocol erratum candidate). Within a match the games are independent given the
  plan (version 1); across matches, modes and generations they share the plan on
  purpose.
- Looking at several generations is a sequence of looks: their intervals are not
  simultaneous, and a difference between two generations is not a test result.
- Every look at a validation result is logged. The validation teams are used
  again at every look, so the result is a progress view the owner has looked at
  many times: it is never cited as held-out evidence, a strength claim or a gate
  result. The log is the sibling file `<run-dir>.progress.looks.jsonl` (for a
  custom root, `<progress-dir>.looks.jsonl`), never inside the progress root or the
  run. `progress`, `report` and `replay --progress-dir` each append one line
  (run, generations, matches, game counts, command) under an exclusive lock when
  they show a result; `inspect` shows none and writes none. Each line is the JCS
  bytes of the entry with the SHA-256 of the line before (the first carries a
  fixed genesis value), so a dropped, changed or reordered line is refused by
  every reader and by the next append. The dashboard shows the number of looks
  and lists the log's SHA-256 among its inputs.
- A forfeit is a scored loss for the failing side, kept with its cause. A harness
  incident (an unsupported candidate, an enumerator defect, a protocol error)
  fails the whole evaluation: nothing is committed and the command exits
  non-zero.
- The pair trains against these two baselines, so a score against them is a
  progress signal, not a held-out result. A score says how the pair did there,
  not how strong it is.

**Cost.** The evaluation runs one query per forward pass on CPU, with no
batching. A measured run on the shared development machine (a 6-layer, width-256
pair, while a training run used the CPU) took about 0.6 s per game and about 25 s
of fixed start-up per command (verifying the generation, reading the dataset), so
100 games in each of the 4 matches took about 5 minutes. Each anchor adds two matches of `--anchor-games` games (200 games: about 4
minutes at 0.6 s per game, more while two networks answer). With the two previous-generation
matches (600 games, from generation 1, `--previous-matches`) the same machine gives about
6.5 minutes
per generation: 48 games at generation 2 took 29 s (0.6 s per game) and a three-generation
`--all` of 8 games per match took 2.5 minutes in all. The figure depends on the
machine and on how long the games are.
