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

**The damage-aware baseline as a training opponent.** `generation.baseline_games`
may list `{"actor": "damage-aware", "policy_id": "damage-aware@v1", "games": N}`
beside `random-legal` and `actor-rule` (ADR-018). That seat is scripted: it
chooses by the pinned engine's own damage and turn order, read through the
mechanics probe, from its own view only, and it is never queried against the
policy worker. Its Episodes are kept and the training loader refuses them; only
the frozen pair's seats train. A run that adds it is a new run (a new `run_key`):
the list is part of the config hash. `league-run.json` then carries `baselines`:
the baseline's policy id, its identity hash (config, mechanics config, adapter
format, state-reader file and version manifest: games with another hash are
another baseline) and the summed per-game counts (decisions, probe queries and
runs, `unsupported` answers, reader and assumed-set fallbacks, `epsilon_plays`).
A rising `unsupported_queries` or fallback count shows the baseline drifting
towards `actor-rule`. Wall-clock time is not recorded in the run files (they stay
byte-identical). The baseline reads the committed vocabulary of the one
supported format (`gen9championsvgc2026regmc`) and refuses another. The record
and the config must agree: `baselines` lists exactly the config's damage-aware
entries; one identity hash per policy id per generation is enforced when the
record is built (a record holds one hash per entry). The cost measured in
ADR-018 slice 2 on a loaded shared machine (a pessimistic estimate, not a
guarantee) was a median of about 170 ms per baseline decision (95th percentile
about 350 ms, worst about 520 ms) and about 14 minutes for 200 games at 25
decisions each, so size the share of games against it from that figure: a
baseline game costs roughly 5 to 10 times a `random-legal` game in
simulator time. Use `epsilon_ppm` of the baseline's reviewed config record for
diversity only in training; it is part of the identity.

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
team pool itself, so memory and start-up CPU grow with `S`. Byte equality holds while no query overruns its latency budget (an overrun is a recorded
forfeit, as at one worker). Progress evaluation never batches or shards.

After each generation commits, the driver writes `<run-dir>/metrics/g<NNNNNN>.json`: wall
clock per phase (play, load, train, write, seal), throughput, the learner's peak
accelerator memory (null on a CPU learner), device and threads, and optionally the peak
reserved accelerator memory and the peak resident sets (see below). The file is outside the
generation manifest and every hash, so reruns stay byte-identical. A crash between the
commit and the write loses that generation's timings, and a resume does not write them
again; the run dashboard (`python -m areneon_ml.loop report`) says `not recorded` for it
and lists the SHA-256 of each metrics file among its inputs.

### Sizing a run to a machine

Four operational flags of the driver change how much memory and how many cores a generation
uses and nothing it writes: the plan, the seeds, every Episode, the stream, the tensors and the
learner-state hashes are the same for any value (each flag has a test, and a full-size generation
was compared byte for byte with and without them). None is in the run config.

| flag | what it sets | default |
| --- | --- | --- |
| `--game-processes S` | Node game processes that share a generation's games | 1 |
| `--policy-workers K` | policy worker processes per game process | 1 |
| `--node-heap-mb M` | `--max-old-space-size` of each Node game process | Node's own |
| `--load-processes N` | worker processes that check and encode the stream in the load phase | 1 |
| `--load-memory-mb M` | cap on what the load workers may add (about 400 MB each); fewer run if `N` does not fit | none |

What a process holds, measured on the 28-core, 24 GB, 16 GB GPU machine with the 20,000 team
`train-v1` pool, 768 games per generation (`S` = 4, `K` = 2):

- A Node game process peaks near 0.86 GB while it loads the team pool and 1.1 GB over a
  generation. It writes each Episode to the
  generation's `episodes/` directory when its game ends and keeps none, so its memory no longer
  grows with the number of games (the stream is built by reading the files back one at a time).
  `--node-heap-mb 640` makes V8 collect before it grows: 0.59 GB for the largest game process
  (0.57 GB at 448 and 230 s; below that the process only collects more often, so 512 to 768 is
  a safe range) at the same wall time (219 s against 242 s without the change).
- A policy worker is 0.42 GB (CPU torch and the two networks) and does not shrink.
- The whole play tree peaked at 7.76 GB before, 7.56 GB with Episodes on disk and 5.70 GB
  with `--node-heap-mb 640`. A rule of thumb for the play phase:
  `RAM = S x (0.6 + 0.43 x K) GB`, so `S` = 4, `K` = 2 is about 5.8 GB and `S` = 6, `K` = 3 about 11 GB.
- The driver holds the learner and the generation's encoded steps (on the CPU; only the
  current minibatches go to the GPU). Its resident set peaked at 0.78 GB in the load phase of a
  1.2 GB stream (18,959 steps), 0.93 GB with `--load-processes 6`; a CUDA learner adds its
  context, so plan 1.5 to 2 GB per run (measured on the CPU only). Each load worker peaked at 0.29
  GB, most of it shared with the driver by fork, and `--load-memory-mb` limits them (400 MB each).
- The load phase of that stream took 103 s in one process. The contract check no longer
  re-checks the repeated log prefix (91 s), and `--load-processes 6` gives 23 s and 12 gives 16 s,
  to the same tensors (a digest over every tensor of the 2,432 trajectories is equal before and
  after). The workers' results are sent as one byte blob each, not through `multiprocessing`'s
  tensor transport, which sends every tensor through its own file descriptor and made the
  parallel load slower than the serial one.
- On the GPU a 6-layer, 256-wide learner on that stream peaked at 4.5 GB allocated and 7.6 GB
  reserved. The learner process used to keep the reserve between generations (the 15.5 GB of two
  concurrent runs was mostly cache). The driver now calls `torch.cuda.empty_cache()` after each
  train phase: the reserve drops to 1.0 GB (305 MB of it in use: weights and optimiser state)
  until the next phase. `run-metrics` records the peak reserved memory next to the peak
  allocated. From outside, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the learner's
  environment limits fragmentation further.

How many runs fit. On this machine host RAM is the limit, then the 6 GB swap: one run at
`S` = 4, `K` = 2 with `--node-heap-mb 640` needs about 5.7 GB during play (and 1.5 GB in load and
train), so two overlap in 24 GB with room to spare and three only if their play phases are
staggered (play lasts 220 s of a 280 to 400 s generation) or `S` is lowered. Before this change
two runs filled the swap and made workers miss their 30 s latency budget (a recorded
`latency_overrun` forfeit): never let a machine swap during play. The GPU does not limit the
number of runs: the learner is busy 5 to 10 percent of a generation and its reserve is now given
back. On a larger machine scale `S` with the cores (one game process per 4 cores is a start) and
the memory with the rule above; a rented GPU needs no change beyond `device: "cuda"`.

The run's `metrics/g<NNNNNN>.json` has the resident-set peaks of the driver, the largest game
process and the largest policy worker (`peak_rss_bytes`, Linux, optional) and the peak reserved
accelerator memory (`peak_accelerator_reserved_bytes`, optional), so a run reports what it used.
The driver's figure is the process's high-water mark so far, not the generation's.

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
`--damage-aware` adds, for any generation, `damage-aware-sample` and
`damage-aware-argmax`: the pair against the damage-aware scripted baseline
(ADR-018), which chooses by the pinned engine's own damage and turn order from
its own channel and team only, on the same pairs, seeds and both seats as the
other baselines. It is off by default because it is slower than the others
(it asks the mechanics probe at every decision, see **Cost**). It is a **seen
scripted baseline**: the lineage can have trained against it, so its rows are
never held-out evidence and never a gate result. Each match row carries the
baseline identity hash (SHA-256 of the JCS of its config hash, mechanics config
hash, adapter format, state-reader hash and version manifest: games with another
hash are another baseline, and one record holds one) and the integer counts
summed over its games, and each game row and each stored game carry that game's
counts: decisions, scored decisions, probe queries and runs, queries the probe
answered `unsupported` (scored 0, never approximated), and fallbacks to the
`actor-rule` candidate (a flagged line of the state reader, or no legal assumed
opponent set). A baseline drifting towards `actor-rule` therefore shows as
unsupported queries and fallbacks. The counts recompute on read: a match's are
the sum of its rows', and a stored game's are its row's. Every damage-aware
match has a wall-clock budget, `--damage-aware-budget-minutes` (default 30); a
match that passes it fails the whole evaluation and commits nothing. A
committed generation without the matches is kept, never rewritten (evaluate
into a new `--progress-dir` to add them). The command prints the wall-clock
milliseconds of every match and the mean and slowest game of each (`game_ms`); the
records hold no time.
`--anchors 0,10,20` adds, for each listed generation earlier than the one evaluated,
an `anchor-g<A>-sample` and an `anchor-g<A>-argmax` match against generation A's
frozen pair (`--anchor-games` games each, even, default 200, so 100 pairs), each
served by a worker of its own; the checkpoint ids and policy ids of an anchor are
verified from generation A's own records when the evaluation is made and again when
a committed directory is verified. An `anchor-*` row is a **seen, same-lineage
comparison** too, and a directory committed without the anchors it was asked for
is kept and reported (`anchors: ... missing`).

**Head to head against another run (ADR-010 erratum A4 item 6).**
`--generation G --versus-run-dir OTHER --versus-generation H [--run-games 200]`
adds `run-sample` and `run-argmax` matches: this run's generation-G pair against
run OTHER's generation-H frozen pair (sample against sample, argmax against
argmax), every validation pair of this run's plan from both seats on the same seeds,
beside the usual `actor-rule@v1` and `random-legal@v1` matches. The other run is
only read (no lock, nothing written into it); it gets a worker of its own that loads
its checkpoints and is checked against its own encoder version, so the two runs need
not share one. Refused before any game: the same run directory or run key, a
different format or team dataset (so one validation split), a run without a team
dataset, a generation the other run has not committed, a non-synthetic checkpoint
record, a progress seed equal to a seed of either
run's committed plans, a validation team in either run's `team-set.json`, and more
games than disjoint pairs. The opponent variant of the record is
`{kind: "run", run_id, run_config_hash, generation, policy_id, checkpoints}`
(`plan.run_games_per_match` is present exactly when such matches are). The printed
summary carries `head_to_head` (label "cross-run comparison", the other run,
and `lineage`: the C1/C6 checker's verdict on both runs' records, "independent
lineage" only when the ancestries are disjoint, else "related lineage: shared
start"). Runs started from equal initial seeds share their seeded generation-0
pair; that is a relation to label, not a refusal. The comparison is refused only
when the two compared pairs share a checkpoint, or one of them is in the other's
ancestry (a pair against its own past); the dashboard and the
replay pages carry the same label, with the pair-level interval and the pair score
minus 1/2. It is a progress view, never held-out evidence or a gate result, and the
adopt rule is applied by the experiment, not here. Use a progress directory of its own
(a committed generation is never rewritten). The actor-rule matches in the record are
this run's, on the same pairs and seeds; the other run's actor-rule score on these
pairs is not played.

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
then the pair's). The opponent variants are `baseline`, `previous`, `anchor`
(`{kind, generation, policy_id, checkpoints}`) and `run`; `plan.anchor_games_per_match` is
present exactly when anchor matches are, and `teams.pairs_used` is half the
largest games per match. A version 1 record (one game per pair, the candidate
alternating seats) stays readable as it is and is never rewritten or upgraded.
A version 2 record carries no per-game Wilson interval: its games are not
independent, and its printed summary and dashboard show the pair-level interval
below (**The pair-level interval**). The replay pages for version 2 are a separate
ticket (#951).

**The pair-level interval (ADR-017 erratum E2 items 2 and 6; #948).** For a
version 2 record the unit is the pair. A pair's score is the mean of its two game
scores (0, 1/4, 1/2, 3/4, 1; a tie is half, a forfeit a scored loss for the side
that forfeited), kept as an integer in half-games (0 to 4), and each pair counts
once. The interval is a **percentile bootstrap over pairs** (the protocol's 6.1
team-clustered bootstrap specialised to disjoint pairs, not the 6.3 block plan),
in integers only: `B = 4000` resamples of `n` pairs with replacement, each
summarised by its integer sum; the interval is the sorted sums at ranks
`floor(B * 25 / 1000)` = 100 and `B - 1 - 100` (the 2.5% and 97.5% order
statistics, a 95% interval), in parts per million, rounded outward. `B` and the
rule are fixed in `progress/stats.py` and are not stored (the protocol's
`B_boot` of 20,000 belongs to the sealed run; a progress view does not need it).
The resampling stream is SHA-256 over `JCS({run_key, generation, label, purpose:
"progress-bootstrap/1"})` used as `SHA-256(key || counter)` in 32-bit words with
rejection (no modulo bias), where `label` names the statistic (`match:<id>`,
`ablation:<baseline>:<variant>:<score|minus-learned>`); so the same record shows
the same interval on every machine and every look, and two statistics do not
share draws. `summary_of` returns, instead of the Wilson keys, `pairs`,
`interval_low_ppm`, `interval_high_ppm` and an `interval` label ("pair-level
percentile bootstrap, descriptive, conditional on the fixed validation plan"); a
match against an anchor or the previous generation also returns `diff_mean_ppm`,
`diff_low_ppm` and `diff_high_ppm`, the pair score minus 1/2 (the same resamples,
shifted). The Preview ablation rows carry `low_ppm` and `high_ppm`, and the paired
difference to the learned Preview its own interval over `d_k`
(`paired_differences`). Intervals are **descriptive**: one evaluation shows many
(every match, baseline, anchor and variant) with no multiplicity control, none is a
test, a rating or a claim, and all are conditional on the fixed validation plan.
Nothing is stored: the record keeps counts and game rows. Version 1 records keep
the per-game Wilson interval, unchanged. Paired differences of the candidate
against an anchor's own baseline matches (`d_k` = candidate minus anchor on the
same baseline pairs) need the anchor to play the baselines in the same record;
the record does not hold those matches yet, so only `paired_differences` (used for
the ablation) exists, and the head-to-head `diff_*` is what anchors give today.

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
integers, with the pair-level interval of the section above, conditional on the
fixed validation plan. With 100 pairs a Preview matters only if
the difference is clear of what the pair-to-pair spread allows: the counts of
better and worse pairs are the first thing to look at. A progress view, never
the Training gate.

**Reading the numbers.**

- The record stores counts. A **score** counts a tie as half a win:
  `(wins + ties / 2) / games`. The command prints it, with its **Wilson 95%
  interval** (version 1 records;
  a version 2 record has the pair-level interval above), in parts per million (`score_ppm`, `wilson_low_ppm`,
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
machine and on how long the games are. The damage-aware baseline asks the mechanics probe at
every scored decision: measured on the same shared machine (load average 3 to 8)
over 118 games per match (59 validation pairs, a stand-in candidate that plays the
`actor-rule` choice, games of 11 turns on average), a damage-aware game took about 1.5 s
on average and 3.9 s at the slowest (about 115 ms per decision), against about 0.17 s
for an `actor-rule` game. A 200-game damage-aware match is then about 5 minutes at the mean
and about 13 minutes at the slowest game, inside the 30-minute working budget; a pair that
plays longer games costs more in proportion. Both modes together add about twice that. The measurement used a stand-in candidate; a learned pair that plays
longer games costs more. The budget is checked after every game, so a single game that hangs
is bounded by the per-decision latency budget, not by the match budget. The readers check that
the records agree on one identity hash; the evaluation recomputes it from the reviewed records
and the pinned engine when it plays.

## 5. Teacher data for the cloning learner

The cloning learner (ADR-019 track 1, `python -m areneon_ml.learner clone`) imitates
the scripted baseline `damage-aware@v1` (ADR-018). `teacher-run.ts` plays that
teacher and writes the one directory the learner needs: league Episodes of
synthetic teams, a learner-ready `TrainingSet` whose behaviour policy is the
teacher, and the teacher's identity. It is a teacher run of its own (its own
`--run-key`), not a generation of a training run, and it needs no policy worker.

```bash
node packages/team-dataset/scripts/dataset-hash.ts --dir DATASET_DIR --format gen9championsvgc2026regmc   # the --dataset-hash value
node packages/team-dataset/scripts/teacher-run.ts --run-key TEACHER-1 --format gen9championsvgc2026regmc \
  --games 1500 --dataset DATASET_DIR --dataset-hash HEX64 --game-processes 8 --out .tmp/teacher/TEACHER-1
```

- **Who plays.** `--games N` are teacher-against-teacher games: both seats are the
  teacher and both seats' Episodes train (2 Episodes per game). `--random-games` and
  `--actor-rule-games` add games against `random-legal@v1` or `actor-rule@v1` for
  diversity: the teacher takes one seat (it alternates), only its seat loads, and the
  baseline seat's Episode is kept in `episodes/` but not loaded (the loader would refuse it); so "every Episode of the run" loads only in a run without baseline games. The other seat's policy
  id is always in the Episode's collection record (`opponent_policy_id`).
- **Teams.** From the dataset's dev and training records through the loop's own
  `datasetTeamPool` (validation never; the dataset hash must match). Without
  `--dataset` the interim fixture pool is used (tests, smoke runs).
  `--excluded-abilities` defaults to `Illusion`, as the committed run configs do.
- **Seeds.** `league-seed/1` generation 1 under the teacher run's key: the same key,
  pool and commit give the same bytes, whatever `--game-processes` is (each process
  plays the games `g` with `g mod N = i`). The battle seed is in `plan.json` and in each
  Episode's `collection.seed` (the A1 Episode contract requires it); no
  `TrainingExample` carries a seed or a draw.
- **Exploration.** `--epsilon-ppm` (ADR-018 Q5, training use only; default 0) makes the
  teacher play a random candidate with that chance. It is another baseline identity and
  the policy id becomes `damage-aware@v1-eps<ppm>`.
- **Budget.** `--battle-ms` and `--preview-ms` (default 30,000) are the teacher's
  per-decision limit; an overrun, or any failure of the teacher, is a harness incident
  and fails the run (exit 1, nothing is written under `--out`). The teacher's integer
  counts (decisions, scored decisions, probe queries and runs, `unsupported_queries`,
  `reader_fallbacks`, `assumed_set_fallbacks`, `epsilon_plays`, `probe_refusals`), per game and summed, are
  in `teacher.json` with the baseline identity hash and the budget, so a teacher drifting
  towards `actor-rule` is visible. A speed order the probe cannot state (an action that
  queues more than a move, a fractional-priority item) is counted as `unsupported` and
  scores no discount, as ADR-018 says. Any other `MechanicsProbeError` of a query the actor built (a declared state the engine does not take; not a config for another format, which is a defect of the run) never ends a game: that request is answered by the `actor-rule` candidate and counted in `probe_refusals` (a high rate means the teacher plays like `actor-rule`; the same count is in a league run's `baselines` counts, written only when above 0). A public status contradicts an unrevealed assumed ability that would block it (Limber on a paralysed Pokemon), so the next ability of the record's order is assumed instead (found by public-main-004, game 604). The probe queries of that check add to `queries` and `probe_runs`. A probe result of kind `refused` (an invalid set the actor built) is still a defect and fails the run. The probes' results for queries they used to answer are unchanged and `MECHANICS_ADAPTER_FORMAT` is deliberately not bumped (only states they used to refuse are now declared), so the baseline identity hash stays the same.

The directory (built beside `--out` and renamed into place at the end):

| File | Content |
|---|---|
| `episodes/<hash>.json` | every played seat's Episode (league collection, `run_id` the key) |
| `stream.jsonl`, `training-set.json` | the teacher's examples and their `TrainingSet` (`learner_ready`, `synthetic`, `behaviour_policy_id` the teacher) |
| `teacher.json` | (beside the plan; also `teacher_identity_hash` in `run.json`) policy id, baseline identity hash, epsilon, budget, counts |
| `plan.json`, `run.json` | the plan (harness seeds, owner only) and the game rows, hashes and dataset hash |
| `version-manifest.json` | `showdown_commit`, `format_id`, `regulation`, `ruleset_hash`, `format_rules_hash` |

The printed line carries `training_set_hash` (SHA-256 of the TrainingSet's JCS bytes):
that is `CloneRunConfig.teacher.training_set_hash`, with `teacher.behaviour_policy_id`
the printed `teacher_policy_id`. Then:

```bash
.venv/bin/python -m areneon_ml.learner clone --config CLONE.json --stream-dir .tmp/teacher/TEACHER-1 \
  --engine-data fixtures/showdown/gen9championsvgc2026regmc/engine-data.json \
  --version-manifest .tmp/teacher/TEACHER-1/version-manifest.json --out .tmp/clone/CLONE-1
```

**Cost.** Measured on the shared development machine (CPU, one process, the fixtures
dataset, while a validation evaluation ran): about 2.3 s per teacher-against-teacher
game (5 to 12 turns, about 11 decisions per seat, so about 11 examples per Episode)
and about 3 s of start-up. 12 games took 31 s in one process and 11 s with 4 game
processes, byte-identical. So 1,500 games (3,000 Episodes, about 33,000 examples) take
about 1 hour in one process and about 10 minutes with 8; the real dataset's pool load
adds to the start-up. Memory grows with the number of Episodes held (the same as a
generation of that many games).
