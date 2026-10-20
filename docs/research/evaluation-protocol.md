# Held-out evaluation protocol

Research for Redmine #681 (Notion Wave 1 acceptance gate: "held-out evaluation protocol defined without fabricated Elo or win-rate thresholds"). It feeds the Wave 2 go/no-go (#683), ADR-010 (AI policy and training loop, #729) and the Training gate (#790).

**What this note fixes.** How a learned policy is judged offline: against which opponents, on which teams and seeds, in which seats, with which confidence intervals, and under which rules the sealed test stays sealed. It sets no numeric target. Every value that would be a policy choice (interval width, team count, seed count, run budget, error rate) is named as a **parameter** in [§10](#10-parameters-left-to-adr-010) with the rule for choosing it; ADR-010 records the values.

**What the results mean.** Evaluation teams are synthetic (§4.1), so every result generalises to the **team generator's distribution** (after the §4.4 pre-checks), not to the distribution of teams people actually play. A Training-gate candidate and every evaluation opponent train on synthetic teams only; real teams appear only in a separate private lineage after the gate (erratum E6, ADR-010). The gap between the synthetic and the real distribution is not measured by this protocol.

**How to read it.**

- Facts that are not the project's own decision cite a source label `[S#]` (see [Sources](#sources)). Engine facts cite `file:line` in the pinned submodule `vendor/pokemon-showdown` at `7244fe8e28fa8bc4d0809ae30d183c608b62bb2b` [S1].
- **Decision** marks a rule this protocol sets. **Inference** marks a conclusion drawn from cited facts that no source states.
- The [Errata](#errata-adr-010) at the end record corrections decided in ADR-010 (accepted r7); where an erratum and the text it names differ, the erratum governs, and the named text has been corrected in place with a pointer to it.
- Glossary terms (`CONTEXT.md`) are used exactly: Preview Policy, Battle Policy, legal heuristic, sealed test, version manifest, Episode, player-perspective observation.

## 1. Questions answered and questions left open

| # | Question | Answered in |
|---|---|---|
| Q1 | Which opponents is a candidate measured against, and what does each pool tell us? | §3 |
| Q2 | Which teams are used, how is a held-out team set made, split and kept unseen? | §4 |
| Q3 | How are games paired (seeds, seats, team assignment) and what is the unit of resampling? | §5 |
| Q4 | Which interval method, which multiple-comparison and sequential-looking rules, what is reported? | §6 |
| Q5 | Who may run the sealed test, how often, what is logged, what happens when the seal breaks? | §7 |
| Q6 | What must be recorded so a reviewer can rerun an evaluation? | §8 |

**Left to ADR-010 (policy and training):** the policy architecture and training algorithm; the training league's composition and schedule; the parameter values of §10; which checkpoint is put forward as a candidate; and the pass rule of the Training gate built from this protocol's report.

**Left to ADR-009 (data sources):** whether any usage-statistics source may be used, under which licence and attribution; it decides whether the usage-statistics pool of §3 exists at all.

**Out of scope:** legality and mechanics (the pinned engine decides, ADR-003, ADR-006); the observation contract (ADR-008); human sparring (ADR-005 A2), which is never an evaluation input.

## 2. Evaluation setting

| Item | Value | Source |
|---|---|---|
| Format | `[Gen 9 Champions] VGC 2026 Reg M-C`, id `gen9championsvgc2026regmc`, Doubles, bring 6 choose 4 | `rules/formats/gen9championsvgc2026regmc.json`; [regmc-format-matrix.md](regmc-format-matrix.md) R3-R5, T3 |
| Team sheets | Closed Team Sheet: Bo1 with Open Team Sheets declined; `forceopenteamsheets` is a forbidden rule | format rules record; ADR-003 |
| Runtime | **Headless runtime (A1)**: `runBattle` runs a seeded in-process battle on the pinned engine; each actor sees only its own requests and legal choice set; teams are validated by the engine's `TeamValidator` first | `packages/showdown-bridge/src/runner.ts`; ADR-005 |
| Randomness | One PRNG per battle, built from the battle's seed (`sim/battle.ts:224-225`). A seed string starting `sodium,` selects the sodium RNG (`sim/prng.ts:18`, `:36-37`, `:58-63`). Every chance event goes through it (`randomChance`, `sim/battle.ts:350-352`). | [S1] |
| Determinism | Same engine, format, seed, teams and choices reproduce the same `omniscient_log`; `replayActor` replays recorded choices | `runner.ts` (`runBattle`, `replayActor`) |
| Timer | A1 runs no timer ([timer-deadline-budget.md §1.6](timer-deadline-budget.md)). This protocol measures decision latency against a budget (D2); it does not test compliance with the VGC Timer. | — |
| Outcome | `winner` is `p1`, `p2` or `null` (tie). The runner forces a tie when `maxTurns` is reached. | `runner.ts` (`BattleRecord.winner`, `maxTurns`) |

**Decision (D1): score.** A game scores 1 for a win, ½ for a tie, 0 for a loss, from the candidate's side. Ties are also counted and reported on their own, because a tie can come from the runner's turn cap rather than from play.

**Decision (D2): failures are classified in advance.**

| Class | Events | Treatment |
|---|---|---|
| **Policy forfeit** | For the candidate or a learned opponent: an exception or crash in the policy, out-of-memory, a submission that is not in the request's legal choice set, or one decision whose compute time exceeds `latency_budget` (parameter, §10) | The game is scored as a **loss for the side whose policy failed** and recorded as a forfeit with its cause. The candidate is penalised for its own failures. Forfeits are reported per opponent. A latency forfeit depends on hardware timing and cannot be replayed exactly (§8.2). |
| **Harness incident** | `EnumeratorDefectError`, `InvalidTeamError`, `NoCandidateError`, an engine `|error|` that aborts the battle (`BattleAbortedError`), or a failure of a scripted baseline | Not scored. The **run fails** and is reported with the block id; its results are not used. Dropping these games instead would bias the estimate towards the games the code survives. |

A sealed run that fails still uses one unit of the sealed budget (I3): a rerun would be a second look at the same set. Recording a forfeit needs harness support around `runBattle`, which has no forfeit outcome today (O12).

**Deadline compliance.** `latency_budget` is measured on A1 on the evaluation hardware. Compliance with the VGC Timer itself (bank never exhausted, no server auto-choose) can only be observed where the timer runs: ADR-005's A2 runtime with its deadline watchdog and the `deadline_outcome` records ([timer-deadline-budget.md §4](timer-deadline-budget.md)). That evidence is outside this protocol, and any gate that asks for clean deadline metrics must take it from there.

**Known engine gap.** When both sides' last Pokémon faint together, `checkWin` gives the win to `faintData.target.side`, the side of the last entry processed in the faint queue (`sim/battle.ts:2610-2614` [S1]), which may differ from the game's unpublished tiebreak (gap `simultaneous-faint`, [showdown-champions-gaps.md](../showdown-champions-gaps.md)). That order follows the faint queue (speed and move order), not the seat, so the seat swap of §5 is not claimed to balance it. Such games are labelled once a detector exists (O7).

## 3. Opponent pools

The candidate is a pair of frozen checkpoints: one Preview Policy and one Battle Policy, each identified by a content hash. Every opponent is identified the same way: an opponent id, the code commit or checkpoint hash that defines it, and, for stochastic opponents, how its random stream is derived from the block seed (§5.4). Every learned opponent trains only on teams of the training pool (§4.2): never on sealed teams, and on dev teams only under the D6 rule (they then join the training pool); never on validation teams (erratum E2, ADR-010).

| Pool | Members | How built | Versioned and frozen by | What it tells you | Seen or held out |
|---|---|---|---|---|---|
| **B. Scripted baselines** | `actor-rule`: the shared actor rule (first non-pending candidate in canonical order, else first pending) applied to the legal choice set; `random-legal`: uniform over the non-pending candidates, else over the pending ones | Code in `showdown-bridge` (`actor-rule.ts`); `random-legal` is a small module of the same shape (not yet built, O8) | The repository commit; a baseline never changes within a protocol version | A floor and a bug detector. Failing to beat a **legal heuristic** points at a defect, not at a research result. Neither is described as AI. | Seen (anyone may train against them) |
| **F. Frozen checkpoints** | Earlier checkpoints, from the candidate's own lineage and from at least one independent lineage (C6) | Frozen at points fixed in advance (step counts or calendar), each a content-hashed Preview + Battle pair with its provenance record (C7) | Checkpoint hash; immutable once registered | Progress and regression over time; whether improvement transfers to opponents trained elsewhere | Own lineage: **seen**. Independent lineage: **held out** |
| **U. Usage-statistics opponents** (conditional) | Opponents whose team sets and choice preferences follow aggregate usage statistics | Only if ADR-009 admits a source with reuse rights. Built from one dated snapshot of that source; the snapshot hash is recorded. Their teams count as training-pool teams (§4.2) | Source snapshot hash plus builder commit | Performance against a population-like distribution of play, not against the project's own style | Held out unless the same snapshot also feeds training |
| **X. Evaluation exploiters** | Policies trained only to beat one frozen candidate | Trained after the candidate is frozen, from the evaluation budget, against that candidate alone, on training-pool teams | Exploiter hash plus the hash of the candidate it targets | How badly the candidate can be beaten by an opponent trained for it: robustness, not average strength. AlphaStar's exploiters exist for "exposing its flaws, rather than maximising their own win rate" [S14] | Held out for that candidate |

**Why several pools.** A policy trained against other learning agents can overfit to their policies and fail to generalise; Lanctot et al. name this effect and measure it with joint-policy correlation [S13]. Ratings over a single pool can also be distorted by the pool's make-up (§6.6) [S10]. Reporting per pool, with seen and held-out pools apart, shows both effects instead of averaging them away.

### 3.1 Contamination rules between the training league and the evaluation pools

**Decision (D3).**

| Rule | Statement |
|---|---|
| C1 | Each opponent is classified **seen** or **held out** relative to each candidate, computed by script from the provenance records (C7), never by hand. The report states the class next to every number. |
| C2 | The **sealed opponent set**, the opponents whose results count as held-out evidence, contains held-out opponents only: independent-lineage checkpoints, evaluation exploiters, and usage-statistics opponents not used in training. Seen opponents (pool B baselines and own-lineage checkpoints) may still play on sealed teams in a sealed run; their rows are reported as seen and are not held-out evidence. |
| C3 | An evaluation exploiter is never added to the training league of the candidate it targets. It may join the league of a later generation only after the candidate's sealed result is recorded; from then on it is **seen** for every checkpoint whose provenance reaches it. |
| C4 | Games played for evaluation never become training data: no behaviour cloning, replay buffer, reward model or hyperparameter fit may read them. Their Episodes and records carry an evaluation run id and a sealed or open status, are stored apart from training data, and the training data loader refuses them (contract field: O3). |
| C5 | Scripted baselines may appear in training and in evaluation; they are reported as seen. |
| C6 | **Lineage and independence.** Build the provenance graph whose nodes are checkpoints and data sources and whose edges are "initialised from", "trained against" (league membership, exploiter target) and "trained on data generated by". The **ancestry** of a checkpoint is the checkpoint itself plus every node reachable from it along these edges towards its past, as recorded when it was frozen: what it was initialised from, trained against, or learned from, transitively. Two checkpoints are **independent** only if their ancestries share no learned checkpoint and no behaviour-cloning or fine-tuning data source; scripted baselines and the shared training team pool (§4.2) do not count. A checkpoint that later trained against the candidate has the candidate in its ancestry, so it is related; two lineages that share a behaviour-cloning warm start are related too. Closing over descendants as well (in either direction) would link checkpoints through any later common descendant, and independent lineages would become impossible to keep over time; ancestry by time avoids that without missing any influence on either checkpoint. |
| C7 | **Provenance record.** Every checkpoint (candidate, frozen opponent, exploiter) has a content-hashed record: parent checkpoint, league opponents, exploiter target, behaviour-cloning and fine-tuning data source hashes, and the hash of a **content-addressed team-set manifest**: the list of per-team JCS hashes of every team it trained on, with each team stored under its hash where the evaluation job can read it (L1, L2 and L2a need the teams, not just one set hash). "Every team used at any point" (D6) is enforceable only if every team a training job draws is logged into that manifest, including teams generated on the fly for self-play. A checker script computes C1 and C6 from these records; a checkpoint without a record cannot enter an evaluation pool. Neither the record schema nor the checker exists yet (O11). |

## 4. Held-out teams

### 4.1 Where teams come from

**Decision (D4).** Evaluation teams are **synthetic**: made by a seeded team generator or by contributors by hand, never copied or adapted from real competitive teams. Every team must be accepted by the pinned engine's `TeamValidator` (`validateTeam` in `teams.ts`) before it is used, and must round-trip through `importPaste` / `exportPaste`. Teams carry `data_class: "synthetic"` (ADR-008 §5). Real teams stay in the gitignored `private-data/` and are never part of any evaluation split.

A generator draws species, items, abilities, moves and Stat Points from what the engine accepts at the pin (the champions mod data and the format's rules), with constraints for the coverage targets of §4.3. The generator's code is public; its seed for the sealed split is secret (§4.4). The generator and its coverage labeller do not exist yet (O9).

### 4.2 Splits and the training pool

**Decision (D5).** Teams are assigned to splits **before any training run reads them**, by a stratified, seeded assignment over the strata of D20 (§4.3).

| Split | Used for | Committed? | Seeds |
|---|---|---|---|
| **dev** | Debugging, smoke runs, ablations, plots. Free to use. | Yes, as CC0 fixtures under `fixtures/` | Public derivation key |
| **validation** | Model selection: checkpoint choice, early stopping, hyperparameters, league tuning, and the coverage check of §6.1 | Yes, as CC0 fixtures (erratum E2, ADR-010). Its results may be read as often as needed, which is why it is not a test | Public derivation key, distinct from dev |
| **sealed test** | One confirmatory measurement per registered candidate (§7) | **No.** Only commitments (hashes) are committed while the set is sealed; the teams live in the isolated store of I1 | Secret key, committed by hash |

**Decision (D6): the training pool is a union.** The **training pool** is the union of every team used, at any point, by:

- every lineage that contributes a checkpoint to an evaluation pool or to the candidate (C6), including independent lineages (pool F) and evaluation exploiters (pool X);
- every behaviour-cloning and fine-tuning data source;
- usage-statistics opponents (pool U), if they exist;
- every **retired sealed set**, once published (L5), whether or not anyone trained on it yet.

The whole training pool is disjoint from the sealed set and from its L2 neighbourhood. Dev and validation teams must also be disjoint from the sealed set. A training job may use dev teams only if they then join the training pool, so L1 and L2 (and L2a) are applied to them against the sealed set. Training on validation teams is **forbidden** (erratum E2, ADR-010): validation stays out-of-sample for selection, the coverage check of §6.1 and D11. Sealed teams and unretired sealed sets are never trained on.

### 4.3 Size and coverage

**Size logic (no fixed number).** The number of sealed teams `n_teams` and seeds per team pair `n_seeds` are chosen so the projected half-width of the §6.1 interval, at the **adjusted** level `α/(m·B_sealed)` of §6.4 (with `m` pre-registered comparisons), is at most the target half-width `h` (a parameter, §10). Procedure:

1. Run the §5 design on the validation split with the candidate under development.
2. Estimate how much of the score variance lies between teams and how much between seeds within a team pair.
3. Project the half-width for candidate values of `n_teams` and `n_seeds` and choose the cheapest pair that meets `h` and the `n_teams` floor of §10 (set by the coverage check of §6.1).

**Inference.** When the between-team share dominates, adding seeds stops narrowing the interval and only more teams help; when it is small, seeds are the cheap lever. The variance split is measured, not assumed.

**Coverage labels.** Each team carries labels computed from its own data by a public, deterministic labeller (O6, O9): for example a speed-control mode (Trick Room setter, Tailwind setter, neither), a weather or terrain setter, a Mega Stone holder, redirection, Fake Out. The labels are multi-dimensional, so they are not strata by themselves.

**Strata (Decision D20).** The strata are **one partition** of the teams: each team's stratum is the tuple of its label values. A cell with fewer than `n_stratum_min` teams (§10, at least 2, because a one-team cell is never varied by resampling and so understates variance) is merged by a merge rule fixed before assignment: drop the label dimensions one at a time in a pre-registered order, coarsening only the undersized cells, until every cell meets the minimum. The minimum holds **in each split**, the sealed split included: the merge rule runs on the generated pool before assignment and coarsens until every cell, under the stratified per-split quota, gets at least `n_stratum_min` teams in each split, and the evaluation job computes the sealed counts inside the isolated store at assignment and logs them with the L3 commitment (erratum E1, ADR-010). The same partition drives the D5 split assignment and the §6.1 resampling. The labels, their order and `n_stratum_min` depend on O6, which this rule blocks on. The report lists the cells and their counts per split.

**Roster coverage.** The report lists, per split, the species used and their overlap with the training pool. The held-out unit is the **team** (its combination of sets), not the species: the roster is finite and species will recur. A separate, labelled stress subset may hold species absent from training.

### 4.4 Leakage controls

| Control | Rule |
|---|---|
| L1 Disjointness | No team appears in two of {training pool (D6), dev, validation, sealed}. Equality is by the JCS hash of the canonical team (sets in a fixed order). |
| L2 Near duplicates | No sealed team is within distance `d_team` of any team in the training pool (D6, retired sets included), dev or validation, where distance counts differing (species, item, ability, move set, Stat Points, nature) fields across matched sets. `d_team` is a parameter (§10). |
| L2a Re-check | The training pool can grow after the sealed set is committed. Before every sealed run, the evaluation job re-runs L1 and L2 inside the isolated store (I1) against the training pool recorded in the candidate's and opponents' provenance (C7), and logs only pass or fail. A failure retires the set (I8): removing teams after the commitment would be a post-hoc change. A failure becomes likelier as the training pool grows, so the cost of retiring and regenerating sets is budgeted in the evaluation register's planning (O1) and in ADR-010. |
| L3 Commitment | Before the first training run of a generation, the sealed set is committed: `SHA-256(JCS({salt, teams}))` with a random salt, plus a hash of the secret seed key, both in the trusted record of I2. Only the hashes are committed. The salt stops a short list being guessed by hashing candidates. |
| L4 Secret generator seed | The generator seed for the sealed split comes from a cryptographic random source, not from a public derivation, so the public generator cannot regenerate the set. |
| L5 Reveal on retirement | When a sealed set retires (budget used, or I8), its salt, teams, rejected candidates and seed key are published as CC0 synthetic fixtures. Anyone can then check them against the L3 commitment and rerun every sealed result made with them. From that moment the set belongs to the training pool (D6). |
| L6 No team-level sealed output while sealed | Per-team or per-label breakdowns of a sealed run stay with the run artifacts until retirement; only aggregate and per-opponent figures are reported. |

**What the pre-checks do (Inference).**

- L1 and L2 reject generated teams that are close to known teams. The sealed set is therefore drawn from the generator's distribution **conditioned on being far from every known team**, not from the generator's distribution itself. The report gives the rejection rate, and results generalise to that conditional distribution.
- Which teams were rejected is itself information about the training pool and the sealed set. Rejected candidates stay sealed with the set until L5, and generation is never repeated selectively depending on what was rejected.

## 5. Paired design

### 5.1 The block

**Decision (D7).** The unit of play is a **block**: one opponent `O`, one ordered team pair `(T_a, T_b)` with `a ≠ b`, and one seed `s`. A block is four games on A1 with the same seed:

| Game | p1 | p2 |
|---|---|---|
| 1 | candidate with `T_a` | `O` with `T_b` |
| 2 | `O` with `T_b` | candidate with `T_a` |
| 3 | candidate with `T_b` | `O` with `T_a` |
| 4 | `O` with `T_a` | candidate with `T_b` |

- **Seat swap** (games 1-2, 3-4): the same seed and team assignment with the seats exchanged. Across the block each side plays each seat equally often, so a seat effect cannot favour either side.
- **Team swap** (games 1-3, 2-4): each side plays both teams of the pair. This removes the difference in team strength from the candidate-versus-opponent comparison. The idea is borrowed from duplicate matches in poker, where each human "will receive the same cards as the computer receives against" the paired human so that the outcome "is not due to luck" [S15]; the guarantee here is much weaker (next paragraph).

**Limit of seed pairing (Inference).** In duplicate poker the shared cards are fixed before play. In Showdown the battle has one PRNG, and how many numbers each event draws, and in which order, depends on the game state and the choices made ([S1] `sim/battle.ts:224-225`, `:350-352`). Two games with the same seed therefore share their luck only **until their choices diverge or their teams differ**:

- games 1 and 3 (team swap) put different Pokémon on each side from the first turn, so they share almost no luck;
- games 1 and 2 (seat swap) have identical teams, but exchanging the seats can change the order in which draws are consumed, so even they share luck only partially.

Seed pairing is a weak variance-reduction device whose gain must be measured (report the within-block correlation of scores), not assumed. The interval of §6.1 does not need that correlation to be high, because whole blocks are resampled with their teams; its validity is asymptotic in the number of teams and is checked at the planned `n_teams` (§6.1).

### 5.2 Crossing teams

**Decision (D8).** For a team split of `n` teams, the block plan uses every unordered pair `{T_a, T_b}` with `a ≠ b` (the team swap covers both orders), crossed with every opponent in the pool and `n_seeds` seeds per pair. Mirror matches (`a = b`) are a separate, optional table. If the full cross is too large, a uniformly random subset of pairs is fixed in advance from the split's derivation key and recorded in the plan.

### 5.3 What is paired and what is independent

| Element | Paired or independent | Why |
|---|---|---|
| Seats within a block | Paired | Same seed and teams |
| Team assignment within a block | Paired | Team swap |
| Two candidates (or a candidate and a frozen checkpoint) compared with each other | Paired: both play the **same** block plan (same opponents, team pairs, seeds) | Their per-block difference cancels block difficulty (§6.3) |
| Different seeds of one team pair | Independent draws, nested in the pair | Seeds are derived independently (§5.4) |
| Different team pairs | Not independent: pairs that share a team are correlated | Crossed design; handled by resampling teams (§6.1) |
| Different opponents | Reported separately; never pooled into one number without saying so | Pools answer different questions (§3) |

**Unit of resampling: the team.** Resampling individual games or blocks would treat correlated games as independent and give intervals that are too narrow. §6.1 resamples teams and keeps every block (four games, all seeds) of each kept pair intact.

### 5.4 Seeds

**Decision (D9).** Seeds are derived, never typed in: `seed = "sodium," + hex32(HMAC-SHA-256(key_split, canonical(block_id)))`, where `block_id` names the opponent, the ordered team pair and the seed index, and `hex32` keeps the first 32 hex digits (the format the runner already uses, `runner.ts` `BattleOptions.seed`). Each split has its own key (§4.2). A stochastic policy (candidate or opponent) gets its own random stream from the same derivation with the seat, the policy id and the game index in the label (erratum E5, ADR-010), so that its sampling is reproducible and independent of the battle PRNG.

## 6. Statistics

### 6.1 Primary interval: team-clustered bootstrap

**Decision (D10).** The primary estimate for a candidate against one opponent is the mean score over all games of the block plan. Its interval is a percentile bootstrap [S4] that resamples **teams**:

1. Within each cell of the D20 partition, draw as many teams with replacement as the cell holds.
2. For every unordered pair of distinct drawn copies whose teams differ, keep that pair's blocks once, with all seeds and all four games. A pair of two copies of the **same** team has no block (D7 requires `a ≠ b`) and is skipped; mirror matches are not substituted. A resample with duplicated teams therefore holds fewer pairs than the original plan; the coverage check below covers the effect of this.
3. Compute the mean score on that resample. Repeat `B_boot` times (§10) with a recorded bootstrap seed.
4. The interval is the `α'/2` and `1 − α'/2` quantiles, where `α'` is the adjusted level of §6.4. For a directional gate claim the interval is instead the one-sided bound `[L, 1]` (`[L, ∞)` for a paired difference), `L` being the `α'`-quantile of the resampled values (erratum E3, ADR-010).

Why this method, and what is not proven:

- Games in a block, and blocks of one team pair, share teams and seeds, so they are clustered. A cluster bootstrap gives consistent variance estimates under both a transformation model and a random-effects model for clustered data [S5].
- Teams are **crossed** (each team meets many others). For crossed random-effects data, Owen's pigeonhole bootstrap resamples the rows and the columns separately and is mean-consistent; no bootstrap is exact in that setting [S6]. For crossed designs with more factors, a bootstrap that gives each observation the product of independently sampled factor weights is "mildly conservative, meaning biased toward overestimating the variance" under sufficient conditions [S19].
- Here the candidate's teams and the opponent's teams are the **same** set, so one draw serves both factors: a symmetric, dyadic adaptation. Neither [S6] nor [S19] covers this scheme; its properties are **unproven** and are checked empirically below.
- Inference targets new **teams drawn from the generator's distribution** under the §4.4 pre-checks, which is the held-out question; a game-level interval would only describe the specific teams tested. It says nothing about real-world teams.
- Deep RL evaluation makes the same move: in the few-run regime "point estimates are unlikely to provide definitive answers" ([S11] §3), and its stratified bootstrap re-samples "runs with replacement independently for each task" ([S11] §4.1); reported improvements without significance measures are hard to judge [S12].

**Coverage check (Decision D11).** Cluster-based inference can be unreliable with few clusters: standard cluster-robust tests "can over-reject … with few (5-30) clusters" [S18]. A percentile bootstrap on a small number of teams can likewise undercover. Before any sealed run, on the validation split and at the planned `n_teams`, `n_seeds` and `α'`. For gate claims the check is run on the one-sided bound of D10 step 4 (erratum E3, ADR-010), so "the interval" below means that bound there, and its coverage target is `1 − α'` for the one-sided bound:

1. **Null on real games.** Evaluate the candidate against an identical copy of itself (a different policy-sampling stream, D9). By the symmetry of the block, the true mean score is exactly ½, and the paired difference (§6.3) between a checkpoint and itself is exactly 0. Record whether the intervals (for gate claims, the one-sided bounds) contain these values over disjoint resamples of validation teams. The check is informative only for a policy that samples: for a deterministic policy, games 1 and 4 (and 2 and 3) of a block against its own copy are the same battle with the sides exchanged, so the block mean is ½ by construction and step 2 carries the check alone.
2. **Simulation at known strength.** Generate synthetic score tables with the plan's exact structure (teams, pairs, seeds, four games per block) from a random-effects model using the variance components measured in §4.3 and a known true mean, at several true means. Estimate the interval's coverage over `R_cov` replications (§10). This tests the bootstrap procedure under the assumed random-effects model; it does not test whether that model describes the games.
3. **Pass rule.** Coverage at least `1 − α'` minus a Monte Carlo tolerance (`cov_tol`, §10). At the level `α' = α/(m·B_sealed)` a miss is rare, so `R_cov` must be at least `1/α'` (and in practice several times that) for misses to be observed at all. The simulation step then costs `R_cov · B_boot` bootstrap resamples of a synthetic table: arithmetic only, no battles. Step 1 costs battles and is run at whatever replication the validation budget allows.
4. **Fallback.** If the check fails, switch to the BCa interval or a studentised (bootstrap-t) cluster interval, the improved bootstrap intervals of [S17], the latter in the cluster form shown to correct over-rejection with few clusters [S18], and repeat the check. If it still fails, raise `n_teams` until it passes. That value is the `n_teams` floor (§10).

### 6.2 Secondary interval: Wilson, for independent trials only

For a table whose games are **independent**, the score interval of Wilson [S2] is reported. Independence holds only if every game draws a **fresh team pair from the generator** and a fresh seed, so that no two games share a team; a fixed split such as dev does not qualify, because its games share teams. Such a table describes the generator's distribution, like §6.1. Brown, Cai and DasGupta show that the standard Wald interval has erratic coverage and recommend the Wilson or the equal-tailed Jeffreys interval for small `n` and the Agresti-Coull interval for larger `n` [S3]. Agresti-Coull is an adjusted Wald interval; for small independent tables, Wilson is the recommended choice. With ties counted as half a win the score is not binomial, so the Wilson interval on it is **approximate**; the table states this.

**Decision (D12).** A Wilson interval is never computed on the paired, crossed plan of §5, because its games are not independent; doing so would understate the uncertainty.

### 6.3 Paired differences between checkpoints

Two checkpoints `A` and `B` are compared on the **same** block plan. For each block the difference of their mean scores is taken; the estimate is the mean difference, and the interval comes from the §6.1 bootstrap applied to the differences (the same team draw for both). Reported per opponent. Positive means `A` scored higher.

### 6.4 Multiple comparisons, p-values and sequential looking

- **Bootstrap p-value (D13).** A percentile interval does not by itself define a p-value. This protocol defines it by **interval inversion**: for a statistic `θ` and the null value `θ_0` (½ for a mean score, 0 for a paired difference), `p` is the smallest level at which the equal-tailed percentile interval excludes `θ_0`, computed with the usual add-one correction so that `p` is never 0: for a **directional** claim "`θ > θ_0`", `p = (#{θ* ≤ θ_0} + 1) / (B_boot + 1)` (and symmetrically for "`θ < θ_0`"); for a two-sided claim, `p = min(1, 2·min(#{θ* ≤ θ_0} + 1, #{θ* ≥ θ_0} + 1) / (B_boot + 1))`, over the `B_boot` resampled values `θ*`. With the BCa fallback, the BCa interval is inverted instead. For a directional claim the one-sided percentile bound is inverted (the `α'`-quantile of the resampled values), with the add-one formula above and `θ_0` set to the claim's margin null (erratum E3, ADR-010). The resolution of `p` is about `1/B_boot`, so `B_boot` must make the tail at the smallest Holm threshold well populated (§10).
- **Pre-registered family (D14).** Before a sealed run, the plan lists every comparison that may support a claim (candidate versus each opponent, candidate versus each named checkpoint), `m` in all, with the direction of each claim. Because each claim is registered with a direction, Holm's step-down procedure [S7] runs on the **one-sided** D13 p-values; a claim registered without a direction uses the two-sided p-value. Claims across that family control the familywise error rate this way, or, for intervals, with Bonferroni-adjusted levels. The Training gate's family uses Bonferroni: each claim passes when its one-sided lower bound at `α'` exceeds its margin null; Holm p-values are reported beside the bounds but do not change the verdict (erratum E5, ADR-010). Any other comparison in the report is labelled exploratory.
- **Across the sealed budget (D15).** Choosing a new candidate after seeing an earlier sealed result is adaptive use of the set; a budget bounds that reuse but does not correct for it [S9]. Either `B_sealed = 1`, or the level is split across the budget: each run uses `α/B_sealed` as its family level, so the per-comparison Bonferroni level is `α' = α/(m·B_sealed)`. When later candidates are chosen after seeing earlier sealed results, the runs are adaptive [S9] and this split is only a **heuristic** bound on the familywise error across the budget; it also does not remove selection effects from the point estimates of later runs. `B_sealed = 1` is the only clean option, and the report states which was used.
- **No peeking on the sealed test (D16).** The sealed plan is fixed in size in advance and run once to completion. Results are not inspected while it runs, and it is not stopped early on a good or bad result.
- **Validation may be monitored, with limits.** If validation runs are watched as they accumulate and stopped when an interval looks decisive, a fixed-`n` interval loses its coverage. Time-uniform confidence sequences hold "for all stopping times" [S8], but their guarantees rest on the observations forming a (super)martingale structure with respect to a filtration, with conditions on the increments [S8] §2. [S8] does not require independence. **Inference** of this document: games that share teams in the crossed design do not give increments with the required conditional structure for the mean over the team distribution, because games revealed later depend on team effects already seen. Confidence sequences are therefore applied only to **independent units** (the §6.2 setting); on the crossed plan any sequential interval is labelled **heuristic**. Validation results remain non-confirmatory either way.

### 6.5 What is reported

| Item | Content |
|---|---|
| Per opponent | Mean score with the §6.1 interval (method and whether the fallback was used); wins, ties, losses, forfeits by cause; seen or held out (C1) |
| Per pool | The per-opponent rows grouped by pool (§3), never one cross-pool average without the rows |
| Per seat | Mean score as p1 and as p2 (diagnostic) |
| Paired differences | §6.3 for each pre-registered pair of checkpoints, with D13 p-values and the Holm decisions (reported, non-gate; the gate verdict is Bonferroni, erratum E5) |
| Head-to-head matrix | Every pair of policies evaluated in the run, with intervals |
| Pairing gain | Within-block correlation of scores (§5.1) |
| Coverage check | §6.1 check results at the planned design |
| Pre-check | L1/L2 rejection rate at generation; L2a pass or fail |
| Engine and contract health | Harness incidents (D2, must be 0 for a valid run); `[Unavailable choice]` refusals per game; privacy test status (ADR-008 adversarial tests) at the evaluated commit |
| Decision latency | Distribution of per-decision compute time on the evaluation hardware against `latency_budget` |
| Plan and provenance | The §8 manifest |

### 6.6 Elo-style ratings: reported, never targeted

A Bradley-Terry model, in which the probability that `i` beats `j` is `τ_i / (τ_i + τ_j)` [S16], may be fitted to the head-to-head matrix and shown as ratings with bootstrap intervals, with `actor-rule` anchored at 0.

**Decision (D17): fitting rules.**

- **Scale.** With `λ_i = ln τ_i`, the shown rating is `(400 / ln 10)·λ_i`, so a 400-point difference means odds of 10 to 1. This is a display convention only.
- **Ties.** Ties enter as half a win and half a loss in a quasi-likelihood fit, or through Davidson's tie model for Bradley-Terry, which adds a parameter for the probability of a tie [S20]. The report names the choice.
- **Separation.** If a policy wins (or loses) every game against the others it met, its maximum-likelihood rating is infinite. The fit then uses a Jeffreys-prior penalty on the likelihood (Firth's bias-reducing penalty [S21]), which gives finite estimates in a broad class of binomial generalized linear models that includes logistic regression [S22]; a Bradley-Terry fit on wins and losses is a logistic regression. Or it uses a Gaussian prior on `λ` with scale `σ_prior` (§10). The report says which, and flags every separated policy.

**Decision (D18).** Such a rating is a summary, never a goal, a gate criterion or a selection rule, because:

- Elo "is meaningless – it has no predictive power – in cyclic games like rock-paper-scissors" [S10]. Team matchups in a team game can be cyclic.
- "An agent's Elo rating can be inflated by instantiating many copies of an agent it beats" [S10]: the number depends on who else is in the pool, which the evaluator chooses.

The matrix and per-opponent intervals are the result; the rating is a view of them.

## 7. Sealed-test isolation rules

**Decision (D19).**

| # | Rule |
|---|---|
| I1 | **Who, and how it is enforced.** Only the evaluation job runs the sealed test. The sealed key, the sealed teams and the sealed run artifacts live in an **isolated store** outside the repository working tree: a separate OS user or container whose files and secret store training jobs cannot read. Training jobs run under a different user or container without access. The owner holds the key and starts sealed runs. The report is reviewed before the owner accepts it (#790), under the relaxed O14 rule of erratum E7 (ADR-010). The concrete store is O11. |
| I2 | **Pre-registration with a trusted timestamp.** Before a sealed run, its plan is written down and hashed: candidate checkpoint hashes and provenance record hashes; opponent ids and hashes; the comparison family with each claim's direction and the decision rule (D13-D15); the block plan size; `α`, `α'`, `B_boot`, the interval method and fallback; the evaluation harness commit; the report template; the L3 set and key commitments. The plan holds no secret. It is timestamped by a mechanism the owner cannot backdate: merged to `main` through a pull request, or attested with GitHub's artifact attestation as the rules publishing workflow already does (`.github/workflows/rules-publish.yml`). The run must match the registered plan. |
| I3 | **Budget.** Each sealed set allows at most `B_sealed` runs (§10), one registered candidate each, with the level split of D15. Reusing a holdout adaptively "can easily lead to overfitting to the holdout set itself" [S9]. Failed runs count against the budget (D2). When the budget is spent, the set retires (L5) and a new set is made under L3. |
| I4 | **Never for model selection.** A sealed result may not choose between checkpoints, set a hyperparameter, stop training, change the league or the training pool, or tune an opponent. Model selection uses the validation split only. |
| I5 | **Seeds fixed in advance.** Sealed seeds come from the committed key by D9; no seed is chosen, replaced or skipped after the plan is registered. |
| I6 | **Everything is logged.** Every sealed run, including failed, aborted or unreported ones, and every L2a re-check, is appended to the evaluation register (O1): date, operator, registered plan hash, candidate hashes, outcome hash, and the reason if not reported. A run absent from the register did not happen and its result may not be used. |
| I7 | **Artifact storage.** A sealed run's `BattleRecord`s hold the `omniscient_log`, which carries both sides' hidden information, and `hidden_verdicts`, the CloneJudge's verdicts on pending candidates, which are hidden state. Both are **evaluation artifacts, never actor input** (ADR-008 §5; `runner.ts`). They, the per-player channels and any Episodes from the run stay in the isolated store (they contain sealed teams) until retirement, are never read by training (C4), and are never built into an observation for a policy. |
| I8 | **Incident rule.** The seal is broken if any of these happens: the key, a sealed team or a sealed artifact is read by a training job or committed before retirement; a sealed result is used as in I4; a run is not in the register; the budget is exceeded; a sealed run deviates from its plan; an L2a re-check fails. Then: (1) the set is marked **contaminated** in the register and retired at once; (2) no result from it may be used for the Training gate; (3) a new set is made under L3 with a new key; (4) the incident, its cause and the affected candidates are written to the register and to the tracker. Results obtained before the breach are kept but labelled with it. |

**Preventing an early commit.** The repository's `check_data_class.py` guard accepts `synthetic` teams under `fixtures/`, so it cannot tell a sealed team from a fixture. An early commit is prevented by keeping sealed material only in the isolated store (I1), never in the working tree, with `private-data/` gitignored for any local copy, and by review of every change under `fixtures/`. The evaluation job holds the per-team hashes of the sealed set, so it can also scan the repository tree cheaply: canonicalise every team-shaped file and compare its JCS hash with the sealed hashes, before every sealed run and after merges to `main`. A hit is an I8 incident. Scheduling that scan is O13.

## 8. Reproducibility

### 8.1 Recorded for every evaluation run

| Field | Content |
|---|---|
| Version manifest | `{showdown_commit, format_id, regulation, ruleset_hash, format_rules_hash}` (ADR-008), from `resolveFormat` |
| Rules provenance | `rules_snapshot_hash`; the embedded format rules record |
| Code | Areneon repository commit of the evaluation harness; the commit of this protocol |
| Candidate | Preview Policy and Battle Policy checkpoint hashes, `behaviour_policy_id`, provenance record hash (C7), inference settings (sampling temperature or argmax, precision, device) |
| Opponents | Each opponent id with its commit or checkpoint hash and provenance record hash; source snapshot hash for pool U |
| Teams | Split name and the team-set hash (for the sealed split, the L3 commitment) |
| Seeds | The key commitment, the block plan (block ids) and the derivation rule D9 |
| Plan | The pre-registered plan hash and its timestamp record (I2) |
| Results | One row per game: block id, game number, seats, teams, winner, turns, forfeit or incident with cause; the hash of that table |
| Statistics | `α`, `α'`, `B_boot`, the bootstrap seed, the interval method and coverage-check result, the multiple-comparison method |
| Environment | Hardware, OS, Node and Python versions, ML library versions and determinism settings |

### 8.2 What a reviewer needs to rerun it

- **Engine-level replay (exact).** With the battle records (seed, teams, choices) and the pinned engine, `runBattle` with `replayActor` must reproduce each game's `omniscient_log` and winner. This checks the engine side and the scoring with no model at all. Exact replay holds only for this engine side: a forfeit for exceeding `latency_budget` (D2) depends on hardware timing and is reproduced only from the record, not by rerunning the policy.
- **Policy-level rerun.** With the checkpoints, the opponent definitions, the key (published for dev and validation; revealed at retirement for sealed sets, L5) and the plan, rerun the block plan and compare per-game choices and results. Neural inference may not be bit-identical across hardware; the rerun reports the share of decisions that differ and recomputes the statistics on its own games (O5).
- **Statistics.** From the per-game table, the bootstrap seed and `B_boot`, the intervals and p-values must reproduce exactly.

## 9. Pipeline

```text
 splits (dev | validation | sealed)     opponent pools B, F, U, X       candidate
 teams validated by the pinned engine   (ids + hashes + provenance C7)  (Preview + Battle hashes)
          |  sealed: commitment only            |                              |
          +-----------------+-------------------+------------------------------+
                            v
          coverage check on validation (§6.1) -> design: n_teams, n_seeds, method
                            v
          block plan: opponents x team pairs x seeds   <- seeds from split key (D9)
          (sealed: registered plan, trusted timestamp I2; L2a re-check)
                            v
          A1 runBattle, 4 games per block (seat swap x team swap)
             |                                  |
             v                                  v
          BattleRecord: omniscient_log,      per-player channels / Episodes
          hidden_verdicts                    tagged with the evaluation run id
          (evaluation artifact only)         (never training data, C4)
             |
             v
          per-game score table --> stratified team bootstrap --> D13 p-values, Holm
                                         v
          report (§6.5) + run manifest (§8.1) --> evaluation register (I6)
```

## 10. Parameters left to ADR-010

No value is set here. Each is chosen and recorded in ADR-010 by the stated rule.

| Parameter | Meaning | How to choose |
|---|---|---|
| `h` | Target half-width of the §6.1 interval for the primary comparison | From the smallest improvement the research claim needs to resolve |
| `α` | Familywise error rate across one sealed set | Conventional level, fixed once per protocol version |
| `m` | Number of pre-registered comparisons per run (D14) | Fixed by the registered plan; `α' = α/(m·B_sealed)` |
| `B_sealed` | Sealed runs per sealed set | `1`, or a larger value with the level split of D15; weigh the cost of a new set against the narrower per-run level |
| `n_teams` floor | Smallest number of sealed teams allowed | The smallest `n_teams` at which the §6.1 coverage check passes at the planned design and `α'` |
| `n_teams`, `n_seeds` | Sealed teams; seeds per team pair | §4.3 projection at the adjusted level, not below the floor |
| `B_boot` | Bootstrap resamples | Far above `m·B_sealed/α`, so that the expected number of resamples beyond the smallest Holm threshold (`B_boot·α'/2` for the equal-tailed reported intervals; `B_boot·α'` for the one-sided gate tail, erratum E3) is large; check that interval endpoints and p-values change by less than the reporting precision between two bootstrap seeds |
| `R_cov`, `cov_tol` | Coverage-check replications and tolerance | `R_cov` at least `1/α'` and large enough that the Monte Carlo standard error of the estimated coverage is small against `α'`; `cov_tol` a small multiple of that standard error |
| `n_stratum_min` | Smallest stratum cell (D20) | At least 2, in each split (erratum E1); raise it until the coverage check passes with the merged cells |
| `d_team` | Near-duplicate distance for L2 | Smallest distance at which two teams play differently in validation runs |
| `latency_budget` | Per-decision compute limit on A1 (D2) | From the policy budget rules of [timer-deadline-budget.md](timer-deadline-budget.md), measured on the evaluation hardware |
| `σ_prior` | Prior scale for the rating fit under separation (D17) | Weak enough that it moves ratings of non-separated policies by less than their interval width |
| Pool membership | Which frozen checkpoints and lineages, whether pool U exists (ADR-009), how many exploiters per candidate | ADR-010 league design, with C1-C7 applied |
| `maxTurns` | Turn cap that forces a tie | Above the longest validation game observed; the runner default is 300. Held in `EvaluationParameters` and passed to `runBattle` explicitly, never taken from the runner's default (erratum E5, ADR-010) |

## 11. Open questions

| # | Question | Blocks |
|---|---|---|
| O1 | Where the append-only evaluation register lives and who may append to it | The first sealed run (I6) |
| O2 | The parameter values of §10 | ADR-010 acceptance; the Training gate (#790) |
| O3 | The Episode contract cannot mark an evaluation Episode, its run id, or a **sealed** status: `data_class` is only `synthetic` or `private` and `rights` only `cc0-synthetic` or `owner-private`, so a sealed synthetic Episode looks like any CC0 one. C4 and I7 cannot yet be enforced by the data loader. An optional field is a minor version change under ADR-008 §3 | Training on any stored Episodes once evaluation runs exist |
| O4 | Whether pool U exists depends on ADR-009 rights | Pool U only |
| O5 | Whether neural inference is deterministic on the evaluation hardware, and the tolerance for a policy-level rerun | The exact-rerun claim of §8.2 |
| O6 | The coverage labels (§4.3) and their definitions | Stratified splits (D5); stratified resampling (§6.1) |
| O7 | A detector for simultaneous-last-faint games (gap `simultaneous-faint`) | Labelling those games; not the protocol |
| O8 | A `random-legal` baseline actor does not exist yet; it needs an implementation ticket with the full quality gates | Pool B as specified |
| O9 | A synthetic team generator with the O6 coverage labeller does not exist yet; it needs an implementation ticket | The first held-out team sets |
| O10 | Whether a Preview Policy is also evaluated alone (with the Battle Policy fixed to a baseline) or only as part of the pair | Reporting of preview-only progress |
| O11 | The isolated store of I1 (OS user, container or secret store), the provenance record schema and the C1/C6 checker of C7 | The first sealed set; any held-out classification |
| O12 | Harness support for policy forfeits (D2) around `runBattle`, which has no forfeit outcome | Scoring forfeits as losses |
| O13 | Scheduling the evaluation job's repository scan for sealed team hashes (§7) | Nothing; until it is scheduled, review and the isolated store are the only defence |
| O14 | Independence of the reviewer in a small team (I1). Relaxed by erratum E7 (ADR-010 G8, O14) | Acceptance of a sealed report |

## 12. Acceptance checklist

| Criterion (#681) | Where |
|---|---|
| Opponent pools defined | §3, contamination rules §3.1 (C1-C7) |
| Held-out teams defined | §4 (sources D4, splits D5, training pool D6, size and coverage §4.3, leakage L1-L6) |
| Paired seeds defined | §5.1, §5.4 (D7, D9) |
| Both sides defined | §5.1 seat swap and team swap; per-seat reporting §6.5 |
| Confidence-interval method defined | §6.1 (primary, D10; coverage check and fallback, D11), §6.2 (Wilson, independent trials only), §6.3 (paired differences), §6.4 (p-values, multiplicity, sealed budget, sequential looking) |
| No numeric Elo or win-rate targets | No targets anywhere; values are parameters (§10); Elo reported only (D17, D18) |
| Sealed-test isolation rules written down | §7 (I1-I8), §4.4 (L2a-L6) |

## Errata (ADR-010)

Corrections decided by the accepted ADR-010 (AI policy and training loop, #729, revision 7, section 8 and its resolved conflicts C-R1 and C-R4), recorded by #848. Each names the text it corrects, which has been fixed in place with a pointer back here. They change no other part of the protocol.

| # | ADR-010 source | Correction |
|---|---|---|
| E1 | §8 E1 (D20/D5) | `n_stratum_min` holds in **each split**, the sealed split included. The merge rule runs on the generated pool before assignment and coarsens until every cell, under the stratified per-split quota, gets at least `n_stratum_min` teams in each split. The evaluation job computes the sealed counts inside the isolated store at assignment and logs them with the L3 commitment. Corrects D20 (§4.3) and the `n_stratum_min` row of §10. |
| E2 | §8 E2 (D6) | Training on validation teams is **forbidden**: validation stays out-of-sample for selection, the §4.3 projection and D11. Dev teams may be trained on and then join the training pool. Validation teams are committed as CC0 fixtures. Corrects D6 and the validation row of §4.2, and §3. |
| E3 | §8 E3 (D10, D11, D13) | A directional claim inverts the one-sided percentile bound (the `α'`-quantile of the resampled values), with the add-one formula of D13 and `θ_0` set to the claim's margin null. For gate claims the D10 interval is the one-sided bound `[L, 1]` (`[L, ∞)` for a paired difference), not the equal-tailed interval. D11 checks the coverage of that one-sided bound. Corrects D10 (§6.1) and D13 (§6.4). |
| E4 | §8 E4 | The S22 entry of the Sources is a verbatim quotation of the published abstract (the arXiv record of the same paper), and the one in-text mention of its result is worded to match. |
| E5 | §8 E5 (D14, D9, §10) | (a) The Training gate's claim family uses Bonferroni at `α'` with one-sided lower bounds against margin nulls; Holm p-values are reported beside them and do not change the verdict (D14, §6.4). (b) The policy-stream label of D9 carries the game index as well as the seat and the policy id (§5.4). (c) `maxTurns` is held in `EvaluationParameters` and passed to `runBattle` explicitly (§10). |
| E6 | §8 E6, C-R1 | A Training-gate candidate, `L2` and every exploiter train on synthetic teams only. Real teams appear only in a separate private lineage after the gate. Corrects "What the results mean" (top) and the validation row of §4.2. |
| E7 | §8 E7, C-R4 (G8, O14) | O14 (and the independent review of I1) is relaxed to the ADR-010 rule: fresh-context reviewers given only the artifacts (registered plan, report, per-game table, records), the pre-registered gate (I2), the owner's sign-off on #790, and the single-model-family limit stated in every report. Reviewer independence across model families is not met and is not claimed. |

## Sources

All accessed 2026-10-19.

| Label | Title | URL | Publisher / version / date | Claim supported |
|---|---|---|---|---|
| S1 | Pokémon Showdown (pinned submodule) | <https://github.com/smogon/pokemon-showdown/tree/7244fe8e28fa8bc4d0809ae30d183c608b62bb2b> | Smogon / Pokémon Showdown contributors (MIT); commit `7244fe8e28fa8bc4d0809ae30d183c608b62bb2b` | `sim/battle.ts:224-225`: one PRNG per battle, built from `options.seed`; `:350-352`: `randomChance` draws from it; `:2610-2614`: `checkWin` gives a both-sides-empty win to `faintData.target.side`; `sim/prng.ts:18`, `:36-37`, `:58-63`: seed strings, `sodium,` selects the sodium RNG |
| S2 | E. B. Wilson, "Probable Inference, the Law of Succession, and Statistical Inference" | <https://doi.org/10.1080/01621459.1927.10502953> | Journal of the American Statistical Association 22(158), 209-212, 1927 | Origin of the score (Wilson) interval for a binomial proportion |
| S3 | L. D. Brown, T. T. Cai, A. DasGupta, "Interval Estimation for a Binomial Proportion" | <https://doi.org/10.1214/ss/1009213286> | Statistical Science 16(2), 101-133, 2001 | The Wald interval's coverage is erratic; Wilson or equal-tailed Jeffreys recommended for small `n`, Agresti-Coull for larger `n` |
| S4 | B. Efron, "Bootstrap Methods: Another Look at the Jackknife" | <https://doi.org/10.1214/aos/1176344552> | Annals of Statistics 7(1), 1-26, 1979 | The bootstrap estimates the sampling distribution of a statistic by resampling the observed data |
| S5 | C. A. Field, A. H. Welsh, "Bootstrapping clustered data" | <https://doi.org/10.1111/j.1467-9868.2007.00593.x> | Journal of the Royal Statistical Society Series B 69(3), 369-390, 2007 | The cluster bootstrap gives consistent variance estimates under both the transformation and the random-effects model |
| S6 | A. B. Owen, "The pigeonhole bootstrap" | <https://doi.org/10.1214/07-AOAS122> (preprint <https://arxiv.org/abs/0712.1111>) | Annals of Applied Statistics 1(2), 386-411, 2007 | For crossed, unbalanced random-effects data, resampling rows and columns separately is mean-consistent; no bootstrap is exact there (citing McCullagh 2000) |
| S7 | S. Holm, "A Simple Sequentially Rejective Multiple Test Procedure" | <https://www.jstor.org/stable/4615733> | Scandinavian Journal of Statistics 6(2), 65-70, 1979 | Step-down refinement of Bonferroni that controls the familywise error rate |
| S8 | S. R. Howard, A. Ramdas, J. McAuliffe, J. Sekhon, "Time-uniform, nonparametric, nonasymptotic confidence sequences" | <https://arxiv.org/abs/1810.08240> (read as <https://ar5iv.labs.arxiv.org/html/1810.08240>) | Annals of Statistics 49(2), 2021 (arXiv v9, 2022-08) | §1.1: coverage "for all stopping times"; §2: the setup assumes an underlying filtration and a sub-ψ supermartingale condition, needing "neither independence nor a common mean" |
| S9 | C. Dwork, V. Feldman, M. Hardt, T. Pitassi, O. Reingold, A. Roth, "Generalization in Adaptive Data Analysis and Holdout Reuse" | <https://arxiv.org/abs/1506.02629> | arXiv, 2015 (v2, 2015-09) | "Reusing a holdout set adaptively multiple times can easily lead to overfitting to the holdout set itself" |
| S10 | D. Balduzzi, K. Tuyls, J. Pérolat, T. Graepel, "Re-evaluating Evaluation" | <https://arxiv.org/abs/1806.02643> (read as <https://ar5iv.labs.arxiv.org/html/1806.02643>) | NeurIPS 2018 (arXiv v2, 2018-10-30) | §1, §2.1: Elo has no predictive power in cyclic games; an agent's Elo can be inflated by adding copies of an agent it beats (worked example in Appendix A) |
| S11 | R. Agarwal, M. Schwarzer, P. S. Castro, A. Courville, M. G. Bellemare, "Deep Reinforcement Learning at the Edge of the Statistical Precipice" | <https://arxiv.org/abs/2108.13264> (read as <https://ar5iv.labs.arxiv.org/html/2108.13264>) | NeurIPS 2021 | §3 "Case Study: The Atari 100k benchmark": "in the few-run regime, point estimates are unlikely to provide definitive answers"; §4.1 "Stratified Bootstrap Confidence Intervals": "we re-sample runs with replacement independently for each task" |
| S12 | P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, D. Meger, "Deep Reinforcement Learning that Matters" | <https://arxiv.org/abs/1709.06560v3> | AAAI 2018; arXiv v3, 2019-01-30 | "Without significance metrics and tighter standardization of experimental reporting, it is difficult to determine whether improvements over the prior state-of-the-art are meaningful" |
| S13 | M. Lanctot et al., "A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning" | <https://arxiv.org/abs/1711.00832> | NIPS 2017 | Policies learned against other learning agents "can overfit to the other agents' policies during training"; joint-policy correlation measures it |
| S14 | O. Vinyals et al., "Grandmaster level in StarCraft II using multi-agent reinforcement learning", and the publisher's summary "AlphaStar: Grandmaster level in StarCraft II using multi-agent reinforcement learning" | <https://doi.org/10.1038/s41586-019-1724-z>; <https://deepmind.google/discover/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/> | Nature 575, 350-354, 2019; DeepMind, 2019-10-30 | Exploiter agents "focus on helping the main agent grow stronger by exposing its flaws, rather than maximising their own win rate" |
| S15 | "Poker Play Begins in 'Brains Vs. AI: Upping the Ante'" | <https://www.cmu.edu/news/stories/archives/2017/january/poker-play.html> | Carnegie Mellon University News, 2017-01-11 | Duplicate matches: "Player A in each pair will receive the same cards as the computer receives against Player B, and vice versa", to ensure the outcome "is not due to luck" |
| S16 | R. A. Bradley, M. E. Terry, "Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons" | <https://doi.org/10.2307/2334029> | Biometrika 39(3/4), 324-345, 1952 | Paired-comparison model in which the probability that `i` is preferred to `j` is `τ_i / (τ_i + τ_j)` |
| S17 | T. J. DiCiccio, B. Efron, "Bootstrap Confidence Intervals" | <https://doi.org/10.1214/ss/1032280214> | Statistical Science 11(3), 189-228, 1996 | Surveys BCa, bootstrap-t, ABC and calibration intervals, whose goal is "to improve by an order of magnitude upon the accuracy of the standard intervals" |
| S18 | A. C. Cameron, J. B. Gelbach, D. L. Miller, "Bootstrap-Based Improvements for Inference with Clustered Errors" | <https://www.nber.org/papers/t0344> | NBER Technical Working Paper 344, 2007-09; Review of Economics and Statistics 90(3), 2008 | Abstract: standard asymptotic tests "can over-reject … with few (5-30) clusters"; cluster bootstrap-t procedures reduce rejection rates of ten percent to the nominal five percent in their Monte Carlo studies |
| S19 | A. B. Owen, D. Eckles, "Bootstrapping data arrays of arbitrary order" | <https://arxiv.org/abs/1106.2125> | Annals of Applied Statistics 6(3), 895-927, 2012 | For multifactor crossed random-effects data, a bootstrap giving each observation "the product of independently sampled factor weights" is "mildly conservative, meaning biased toward overestimating the variance" under sufficient conditions |
| S20 | R. R. Davidson, "On extending the Bradley-Terry model to accommodate ties in paired comparison experiments" | <https://doi.org/10.1080/01621459.1970.10481082> | Journal of the American Statistical Association 65(329), 317-328, 1970 | Extends Bradley-Terry with a parameter governing the probability of a tie |
| S21 | D. Firth, "Bias reduction of maximum likelihood estimates" | <https://doi.org/10.1093/biomet/80.1.27> | Biometrika 80(1), 27-38, 1993 | Origin of the bias-reducing penalised likelihood, which for logistic regression is a Jeffreys-prior penalty |
| S22 | I. Kosmidis, D. Firth, "Jeffreys-prior penalty, finiteness and shrinkage in binomial-response generalized linear models" | <https://doi.org/10.1093/biomet/asaa052> (preprint <https://arxiv.org/abs/1812.01938>) | Biometrika 108(1), 71-82, 2021 | Abstract, quoted verbatim from the arXiv record (first two sentences): "Penalization of the likelihood by Jeffreys' invariant prior, or by a positive power thereof, is shown to produce finite-valued maximum penalized likelihood estimates in a broad class of binomial generalized linear models. The class of models includes logistic regression, where the Jeffreys-prior penalty is known additionally to reduce the asymptotic bias of the maximum likelihood estimator; and also models with other commonly used link functions such as probit and log-log." |

Notes on sources:

- Full text read: [S8], [S10], [S11] (HTML renderings named above), [S18] (NBER PDF, abstract page), [S15], and the pinned engine [S1].
- Abstract or index level only: [S2], [S3], [S4], [S7], [S16], [S17], [S20], [S21]. Their full texts were not retrievable (scanned or behind publisher access); the claims cited are the abstract-level content of each paper, checked against publisher or index records. For [S20] and [S21] that content comes from later literature and software documentation describing them, which is the weakest support in this list. [S5], [S6], [S9], [S12], [S13], [S19], [S22]: claims taken from the abstracts.
- [S14]: the Nature article was not retrievable without a login; the quoted sentence is from the publisher's own summary page.
- [S15] describes the event design announced before play, not the published paper's analysis.
