# Architecture documentation

Subsystem documentation split out of `CLAUDE.md`. `CLAUDE.md` keeps the commands and the invariants every agent must hold; read the doc of the area you touch. Files are laid out by subsystem, not by ticket.

| Doc | Covers |
| --- | --- |
| [contracts.md](contracts.md) | Contracts and fixtures: The ADR-008 contract profile, the generated TS and pydantic types, the shared sample corpus and the fixtures rules. |
| [showdown-bridge.md](showdown-bridge.md) | Showdown bridge: The only package that talks to the pinned engine: engine loading, rules records, request decoding, the in-process runner, team import, the choice enumerator and judges, the actor rule and the random-legal baseline. |
| [observation-episodes.md](observation-episodes.md) | Observations, Episodes and the training loader: `buildObservation`, the Episode record and reader, the own-team stat check, the training loader and team data classes. |
| [engine-data-vocabulary.md](engine-data-vocabulary.md) | Engine data, team vocabulary and role sets: The engine-data export (v1 and v2), the team vocabulary and the role-set table, all derived from the pinned engine. |
| [policy-worker.md](policy-worker.md) | Policy worker protocol: The policy worker contracts, the TS client and the Python worker, play temperature and the evaluate query. |
| [reducer-encoder.md](reducer-encoder.md) | Reducer, encoder, networks and checkpoints: The log-to-state reducer, recent history, the encoder and its options, the networks, checkpoints and the C1/C6 provenance checker. |
| [mechanics-probe.md](mechanics-probe.md) | Mechanics probe: The ADR-016 mechanics probe (stats, speed order, damage, KO counts), its config and the calc differential. |
| [opponent-model.md](opponent-model.md) | Opponent model: usage priors, belief and sampler: Usage priors and weight tables, the opponent belief and its evidence, the sampler, the generator-matched and role-sets priors, and the opponent sheet block. |
| [opponent-sheet.md](opponent-sheet.md) | Opponent sheet block and sheet regimes: The opponent sheet input (closed and open regimes), its builders, encoder use, sidecar, the team sheet reader and open-sheet evaluation. |
| [damage-aware.md](damage-aware.md) | Damage-aware baselines: The `damage-aware` scripted baselines (v1 to v7: v2 over belief worlds, v3 engine choice location, v4 to v7 situational and success-chance scoring), their inputs, identity and use as opponents. |
| [search.md](search.md) | Decision-time search: World facts, the scenario world builder, the fork oracle, search worlds, the probe reply model and the `rerank-1` search actor. |
| [battle-state-tensors.md](battle-state-tensors.md) | Battle-state tensors: The `BattleStateSpec` (ADR-030 step 5 groundwork): the fields of a batched battle-state tensor with their engine sources, visibility classes, projection rules, seat bookkeeping and the map from the reducer and encoder. |
| [search-distillation.md](search-distillation.md) | Search distillation: Search labellers, the label stream contracts and distillation in the learner. |
| [population.md](population.md) | Population loop and ladder: The ADR-021 population: config, plan, ladder, rating, driver, selection, health and frozen pool. |
| [team-preview.md](team-preview.md) | Team Preview solver and best-response: The Team Preview solver, the hypothetical Team Preview battle, the payoff table and the best-response measurement. |
| [features.md](features.md) | Candidate features: The `probe-damage` feature sets, their transport and encoding, the play-time sidecar, the pool and bench. |
| [differential.md](differential.md) | Differential harness and backend admission: The engine-agnostic differential harness of ADR-030 (backend port, byte comparison against the pinned probe, the `BackendAdmission` ledger, fresh-corpus rule, shadow sampler, self-differential report; level 2: the `BattleTrace` recorder on the unmodified pin, the battle-step backend port, the Showdown oracle backend, request-by-request comparison). |
| [damage-kernel.md](damage-kernel.md) | Damage kernel: The clean-room integer damage kernel behind the probe interface (ADR-030 step 3), its data reflection, effects record, admission through the differential harness and measured speed. |
| [throughput.md](throughput.md) | Throughput harness: Decisions per host-core-hour of an arm (the Showdown path, a native engine, a GPU engine) measured one way: exact host CPU of the whole process tree, the ADR-028 idle-machine guard, the `ThroughputRecord` and `ArmReport` contract and the `throughput-bench.ts` command. |
| [training-loop-learner.md](training-loop-learner.md) | Training loop and learner: TrainingRunConfig, the PPO and cloning learners, warm start, schedules, the generation loop, run metrics, teacher data and the GPU learner. |
| [league.md](league.md) | League seeds: League seed derivation and the LeagueRunRecord. |
| [progress-evaluation.md](progress-evaluation.md) | Progress evaluation and reports: The progress evaluation, head-to-head tools, diagnostics, the look log and the run dashboard. |
| [evaluation-protocol.md](evaluation-protocol.md) | Evaluation records, register and sealed sets: Evaluation records, the append-only register and sealed team generation. |
| [team-dataset.md](team-dataset.md) | Team dataset and generator: The synthetic team generator, mixtures, pilot gates, Parquet shards and the DuckDB working database. |
| [private-teams.md](private-teams.md) | Private teams: The private team store, snapshots, owner-session guard, private test and inventory. |
| [inspector-replay.md](inspector-replay.md) | Inspector and replay viewer: The training-data inspector, replay viewer, shared player-view state reader and progress-game replays. |
| [rules-store.md](rules-store.md) | Rules store and importer: The PostgreSQL rules store, publishing workflow and the importer. |
| [repo-guards-ci.md](repo-guards-ci.md) | Repo guards and CI: The repository guards (data class, binaries, licences, CUDA separation) and the CI jobs. |
| [ninjask.md](ninjask.md) | Vendored native engine: The Areneon fork of Ninjask (`vendor/ninjask`): the pin, `make ninjask`, the pin and isolation tests, the repin procedure. |
| [a2.md](a2.md) | A2 bot client and server-path filter: The A2 bot client, the server-line filter and reconnect handling. |
