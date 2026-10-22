// The shared contract sample corpus (ADR-008 §2): valid and invalid instances
// of every contract, plus reader-invalid ones (ADR-009 §4: schema-valid, but a
// reader must refuse them), most built from the golden battle, which every language
// with contracts must classify the same way, plus their JCS digests. Captured
// by scripts/capture-samples.ts into fixtures/contracts/; CI compares a fresh
// capture with the committed files.

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import type { PrivateTeamSpec } from "@areneon/schema/types/v1/private-team-spec.ts";
import { jcs, type Json } from "./canonical-json.ts";
import { legalChoiceSet } from "./choice-judge.ts";
import { evaluationSamples } from "./evaluation-samples.ts";
import { type EngineData, engineDataHash, exportEngineData } from "./engine-data.ts";
import type { Engine } from "./engine.ts";
import { episodeFromRecord } from "./episode-io.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT, goldenBattle, syntheticFixtureHashes, syntheticTeam } from "./golden.ts";
import { LEAGUE_SEED_DERIVATION } from "./league-seed.ts";
import { mechanicsSamples } from "./mechanics-samples.ts";
import { observe, recordedChannel } from "./player-channel.ts";
import { privateSamples, privateSnapshotSamples } from "./private-samples.ts";
import { progressSamples } from "./progress-samples.ts";
import { contractVersions } from "./policy-protocol.ts";
import { battleClass, stampTeam, teamHash } from "./team-class.ts";
import { canonicalTeamHash, logicalSnapshotHash, recordHash, shardLogicalHash } from "./team-records.ts";
import { exportPaste } from "./teams.ts";
import { readTeamVocabulary, TEAM_VOCABULARY_FILE, teamVocabularyHash } from "./team-vocabulary.ts";
import { evaluationRunManifest, loadTrainingSet } from "./training-loader.ts";
import { runConfigHash, type TrainingRunConfig } from "./training-run-config.ts";
import type { RequestRecord } from "./runner.ts";

export const SAMPLES_DIR = fileURLToPath(new URL("../../../fixtures/contracts", import.meta.url));
export const SAMPLE_HASHES = "hashes.json";

const TIMER = { starting: 420, grace: 90, add_per_turn: 0, max_per_turn: 55, max_first_turn: 90, tick: 5 };

/** A copy of `value` with `change` applied: one mutation per invalid sample. */
function mutated<T>(value: T, change: (copy: Record<string, unknown>) => void): unknown {
  const copy = structuredClone(value) as unknown as Record<string, unknown>;
  change(copy);
  return copy;
}

/**
 * Every sample, keyed by its path under `fixtures/contracts/`:
 * `v1/<contract>/{valid,invalid,reader-invalid}/<case>.json`, plus
 * `hashes.json` mapping each sample's path to the SHA-256 of its JCS bytes.
 * A `reader-invalid` sample passes the schema; the contract's reader refuses it.
 */
export async function contractSamples(engine: Engine): Promise<Record<string, unknown>> {
  const record = await goldenBattle(engine);
  const known = syntheticFixtureHashes(engine);
  const team = stampTeam(syntheticTeam() as PrivateTeamSpec["team"], known);
  // The golden battle has all three phases (golden.test.ts pins GOLDEN_PHASES).
  const first = (phase: RequestRecord["phase"]) => record.requests.p1.find((r) => r.phase === phase) as RequestRecord;
  const preview = first("preview");
  const turn = first("move");
  const force = first("forceSwitch");
  const at = (seq: number) => recordedChannel(record, "p1", seq, battleClass(record.team_hashes, known));
  const context = {
    timer_source: "server" as const,
    timer_rules: TIMER,
    turn_seconds_left: 55,
    bank_seconds_left: 400,
    grace_seconds_left: 0,
    remaining_decisions_estimate: 8,
    safety_margin_ms: 3000,
    submit_within_ms: 20000,
    policy_budget_ms: 19000,
  };
  const legal = legalChoiceSet({ request: preview, showdown_commit: engine.commit, judge: null });
  const previewObservation = await observe({ channel: at(preview.request_seq), team, manifest: record.manifest });
  const battleObservation = await observe({
    channel: at(turn.request_seq),
    team,
    manifest: record.manifest,
    decision_context: context,
  });
  const episode = await episodeFromRecord(record, "p1", {
    team,
    opponent_team: team,
    known_synthetic: known,
    behaviour_policy_id: "first-legal@golden",
  });
  const league = { kind: "league", run_id: "league-run-1", seed: record.seed, opponent_policy_id: "first-legal@golden" } as const;
  const evaluation = { kind: "evaluation", run_id: "eval-run-1", sealed: true, seed: record.seed } as const;
  const classed = { opponent_team_data_class: "synthetic" } as const;
  const collected = (collection: unknown) =>
    mutated(episode, (e) => ((e.provenance as Record<string, unknown>).collection = collection));
  const identity = { request_seq: turn.request_seq, request_hash: turn.request_hash };

  const leagueEpisode = await episodeFromRecord(record, "p1", {
    team,
    opponent_team: team,
    known_synthetic: known,
    behaviour_policy_id: "first-legal@golden",
    collection: league,
  });
  const sealedEpisode = await episodeFromRecord(record, "p1", {
    team,
    opponent_team: team,
    known_synthetic: known,
    behaviour_policy_id: "first-legal@golden",
    collection: evaluation,
  });
  // Hash-only: all of a sealed run that may leave the isolated store (ADR-009 §4).
  const sealedManifest = await evaluationRunManifest({ run_id: "eval-run-1", sealed: true, episodes: [sealedEpisode] });
  const training = await loadTrainingSet({ episodes: [leagueEpisode], evaluation_manifests: [sealedManifest], known_synthetic: known, behaviour_policy_id: "first-legal@golden" });
  const exampleOf = (kind: "preview" | "battle") => training.examples.find((e) => e.observation.kind === kind) as (typeof training.examples)[number];
  // The stored shape before candidates and outcome: still valid (ADR-010 section 5).
  const legacyExample = mutated(exampleOf("battle"), (e) => {
    delete e.candidates;
    delete e.outcome;
  });
  const legacySet = mutated(training.set, (t) => {
    delete t.learner_ready;
    delete t.behaviour_policy_id;
  });
  const candidatesOf = (count: number) => Array.from({ length: count }, (_, i) => ({ choice: `team ${i}`, pending: false }));
  const [hashA, hashB] = ["a", "b"].map((c) => c.repeat(64)) as [string, string];

  // A slim table (full type list, two entries of each other vocabulary): the
  // complete one is fixtures/showdown/<format>/engine-data.json.
  const full = exportEngineData(engine, GOLDEN_FORMAT);
  const species = full.species.filter((x) => x.base_species === x.id).slice(0, 2);
  const moves = full.moves.slice(0, 2);
  const engineData = {
    ...full,
    species,
    moves,
    items: full.items.slice(0, 2),
    abilities: full.abilities.filter((a) => species.some((x) => x.abilities.includes(a.id))),
    move_targets: [...new Set(moves.map((m) => m.target))].sort(),
    move_flags: [...new Set(moves.flatMap((m) => m.flags))].sort(),
  };
  const [speciesA, speciesB] = engineData.species as [(typeof species)[number], (typeof species)[number]];

  // The policy worker's contracts (ADR-010 section 2). A query is the
  // training example's observation and candidates plus a sampling rule; its
  // A2 form adds an rqid and a decision context, which the encoder ignores.
  const policyId = `policy:${"c".repeat(64)}`;
  const queryOf = (kind: "preview" | "battle", sampling: unknown) => {
    const example = exampleOf(kind);
    return {
      schema_version: 1,
      query_id: 1,
      identity: example.observation.identity,
      observation: example.observation,
      candidates: (example.candidates as NonNullable<typeof example.candidates>).map(({ choice, pending }) => ({ choice, pending })),
      sampling,
      policy_id: policyId,
    };
  };
  const sampleBy = { kind: "sample", u: 4503599627370496 };
  const battleQuery = queryOf("battle", sampleBy);
  const previewQuery = queryOf("preview", { kind: "argmax" });
  const withRqid = (line: string): string =>
    line.startsWith("|request|") ? `|request|${JSON.stringify({ ...JSON.parse(line.slice("|request|".length)), rqid: 7 })}` : line;
  const battleQueryA2 = mutated(battleQuery, (q) => {
    q.query_id = 2;
    const observation = q.observation as { identity: Record<string, unknown>; log: string[]; decision_context?: unknown };
    observation.identity = { ...observation.identity, rqid: 7 };
    observation.log = observation.log.map(withRqid);
    observation.decision_context = context;
    q.identity = observation.identity;
    q.sampling = { kind: "argmax" };
  });
  const recommendation = {
    schema_version: 1,
    identity: battleQuery.identity,
    choice: battleQuery.candidates[0]?.choice,
    source_policy_id: policyId,
    elapsed_ms: 3,
    deadline_outcome: "in_budget",
  };
  const replyOf = (reply: unknown) => ({ schema_version: 1, query_id: 1, reply });
  const hello = { schema_version: 1, message: { kind: "hello", contracts: contractVersions() } };
  const checkpoints = [
    { role: "battle", checkpoint_id: hashA },
    { role: "preview", checkpoint_id: hashB },
  ];
  const ready = {
    schema_version: 1,
    message: {
      kind: "ready",
      contracts: contractVersions(),
      encoder_version: "stub-1",
      engine_data_hash: engineDataHash(full),
      checkpoints,
    },
  };
  const readyWith = (change: (m: Record<string, unknown>) => void) => mutated(ready, (r) => change(r.message as Record<string, unknown>));
  const helloWith = (change: (m: Record<string, unknown>) => void) => mutated(hello, (h) => change(h.message as Record<string, unknown>));
  const contractsOf = () => contractVersions() as unknown as Record<string, unknown>[];

  // A CheckpointRecord (ADR-010 section 5): ids are placeholders, the shape is what the corpus pins.
  const hexOf = (c: string): string => c.repeat(64);
  const checkpointRecord = {
    schema_version: 1,
    checkpoint_id: hexOf("1"),
    role: "battle",
    lineage_id: "main-0",
    parent: hexOf("2"),
    league_opponents: [hexOf("3"), hexOf("4")],
    training_set_hashes: [hexOf("5"), hexOf("6")],
    team_set_manifest_hash: hexOf("7"),
    run_config_hash: hexOf("8"),
    learner_state_hash: hexOf("9"),
    encoder_version: `enc1:${hexOf("e")}`,
    data_class: "synthetic",
    manifest: record.manifest,
  };
  const recordWith = (change: (r: Record<string, unknown>) => void) => mutated(checkpointRecord, change);

  // The team system's records (ADR-013 section 4). The team is the golden
  // synthetic one; ids and keys that name no real artefact are placeholders.
  const goldenSets = syntheticTeam();
  const labelSet = {
    schema_version: 1,
    label_set_id: "golden-labels",
    labels: [
      {
        label_id: "speed_control",
        values: [
          { value: "trick_room", when: { kind: "has_move", moves: ["trickroom"] } },
          { value: "tailwind", when: { kind: "has_move", moves: ["tailwind"] } },
        ],
        default: "neither",
      },
      { label_id: "fake_out", values: [{ value: "yes", when: { kind: "has_move", moves: ["fakeout"] } }], default: "no" },
      {
        label_id: "sun_setter",
        values: [{ value: "yes", when: { kind: "all_of", of: [{ kind: "has_ability", abilities: ["drought"] }, { kind: "has_move", moves: ["sunnyday"] }] } }],
        default: "no",
      },
    ],
  };
  const labelSetHash = recordHash(labelSet);
  const teamRecord = {
    schema_version: 1,
    data_class: "synthetic",
    team_hash: teamHash(goldenSets),
    canonical_team_hash: canonicalTeamHash(goldenSets),
    paste: exportPaste(engine, goldenSets),
    origin: { dataset_id: "golden-1", split: "dev", index: 0, draw_counter: 0 },
    label_set_hash: labelSetHash,
    labels: [
      { label_id: "speed_control", value: "neither" },
      { label_id: "fake_out", value: "yes" },
      { label_id: "sun_setter", value: "no" },
    ],
  };
  const recordWithTeam = (change: (r: Record<string, unknown>) => void) => mutated(teamRecord, change);
  const generatorConfig = {
    schema_version: 1,
    config_id: "golden-config",
    iv: 31,
    stat_point_max_per_stat: 32,
    gender_policy: "unset",
    nickname_policy: "none",
    default_item_weight: 1,
    item_weights: [
      { item: full.items[0]?.id as string, weight: 3 },
      { item: full.items[1]?.id as string, weight: 5 },
    ],
    d_max: 4_000_000,
    pilot_draws: 40_000,
    pilot_min_hits: 20,
    n_stratum_min: 2,
    fill_quantile_permille: 990,
    sealed_margin_permille: 200,
    range_draws_log2: 20,
    range_margin_permille: 100,
    shard_rows_log2: 18,
    two_platform_bound: 1_000_000,
    sampled_ranges: 16,
  };
  const configWith = (change: (c: Record<string, unknown>) => void) => mutated(generatorConfig, change);
  // The golden vocabulary is read, not exported: the export takes seconds, and team-vocabulary.test.ts holds the file equal to a fresh one.
  const fullVocabulary = readTeamVocabulary(JSON.parse(readFileSync(join(GOLDEN_DIR, TEAM_VOCABULARY_FILE), "utf8")), full);
  const vocabularyEntries = fullVocabulary.roster.filter((r) =>
    full.species.some((x) => x.id === r.id && x.base_species === r.id),
  ).slice(0, 2).map((r) => ({ ...r, moves: r.moves.slice(0, 2) as [string, ...string[]] }));
  // A slim table of just what the slim vocabulary names, so its reader can close every id (the complete one is fixtures/showdown/<format>/team-vocabulary.json).
  const vocabularyTable: EngineData = {
    ...full,
    species: full.species.filter((x) => vocabularyEntries.some((r) => r.id === x.id)),
    moves: full.moves.filter((m) => vocabularyEntries.some((r) => r.moves.includes(m.id))),
    items: full.items.slice(0, 2),
    abilities: full.abilities.filter((a) => full.species.some((x) => vocabularyEntries.some((r) => r.id === x.id) && x.abilities.includes(a.id))),
  };
  const vocabulary = {
    ...fullVocabulary,
    engine_data_hash: engineDataHash(vocabularyTable),
    natures: fullVocabulary.natures.slice(0, 3),
    items: vocabularyTable.items.map((i) => i.id),
    roster: vocabularyEntries.map((r) => ({ ...r, abilities: [...(vocabularyTable.species.find((x) => x.id === r.id)?.abilities as [string, ...string[]])].sort() as [string, ...string[]] })),
  };
  const vocabularyWith = (change: (v: Record<string, unknown>) => void) => mutated(vocabulary, change);
  const datasetManifest = {
    schema_version: 1,
    data_class: "synthetic",
    dataset_id: "golden-1",
    plan_id: "golden-plan-1",
    generator: {
      generator_commit: "a".repeat(40),
      config_hash: recordHash(generatorConfig),
      label_set_hash: labelSetHash,
      vocabulary_hash: teamVocabularyHash(vocabulary as never),
      manifest: record.manifest,
    },
    key: { kind: "public", generation_key: hexOf("1"), assignment_key: hexOf("2") },
    d_max: 4_000_000,
    pilot_draws: 40_000,
    splits: [
      { split: "dev", teams: 5, cells: [{ stratum: 0, pilot_hits: 21_000, quota: 3 }, { stratum: 7, pilot_hits: 40, quota: 2 }] },
      { split: "validation", teams: 2, cells: [{ stratum: 0, pilot_hits: 21_000, quota: 2 }] },
    ],
    reference_pool_hash: hexOf("3"),
    rejection_log: { log_sha256: hexOf("4"), draws: 52_000, rejections: 300, duplicates: 2, overflows: 51_000 },
  };
  const datasetWith = (change: (d: Record<string, unknown>) => void) => mutated(datasetManifest, change);
  const shardOf = (split: string, rows: number, seed: string, ranges: { stratum: number; first_row: number; rows: number }[]) => ({
    object_key: `pub/v1/shards/${hexOf(seed)}`,
    file_sha256: hexOf(seed),
    bytes: 4096 * rows,
    logical_hash: shardLogicalHash([hexOf(seed), hexOf("0")]),
    rows,
    split,
    cell_ranges: ranges,
  });
  const sealSnapshot = (snapshot: Record<string, unknown>): Record<string, unknown> => ({
    ...snapshot,
    logical_snapshot_hash: logicalSnapshotHash(snapshot as never),
  });
  const snapshotBase = {
    schema_version: 1,
    data_class: "synthetic",
    snapshot_id: "golden-snapshot-1",
    dataset_hashes: [recordHash(datasetManifest)],
    manifest: record.manifest,
    vocabulary_hash: teamVocabularyHash(vocabulary as never),
    logical_snapshot_hash: hexOf("0"),
    shard_key: hexOf("5"),
    shards: [shardOf("dev", 5, "6", [{ stratum: 0, first_row: 0, rows: 3 }, { stratum: 7, first_row: 3, rows: 2 }])],
    cells: [{ split: "dev", stratum: 0, rows: 3 }, { split: "dev", stratum: 7, rows: 2 }],
    writer: { name: "duckdb", version: "1.5.6", binary_sha256: hexOf("7") },
    parquet_schema_version: 1,
  };
  const snapshot = sealSnapshot(snapshotBase);
  const snapshotWith = (change: (s: Record<string, unknown>) => void) => mutated(snapshot, change);
  // The same data after a second dataset: a parent, and a validation shard.
  const extendedSnapshot = sealSnapshot({
    ...snapshotBase,
    parent_snapshot_hash: hexOf("8"),
    shards: [
      ...snapshotBase.shards,
      shardOf("validation", 2, "9", [{ stratum: 0, first_row: 0, rows: 2 }]),
    ],
    cells: [...snapshotBase.cells, { split: "validation", stratum: 0, rows: 2 }],
  });
  // A TrainingRunConfig (ADR-010 section 5): the tiny run the CPU end-to-end tests train, in the corpus so every language reads the same record.
  const runConfig = {
    schema_version: 1,
    lineage_id: "tiny-main",
    format_id: GOLDEN_FORMAT,
    seeds: { run_key: "tiny-run", init_preview: 11, init_battle: 12, learner: 13 },
    model: {
      encoder: { volatile_buckets: 4, condition_buckets: 4, max_moves: 4, max_team: 6, max_picks: 4, stat_scale: 500, turn_scale: 100 },
      d_model: 16,
      n_layers: 1,
      n_heads: 2,
      d_ff: 32,
    },
    generation: {
      games: 4,
      baseline_games: [{ actor: "random-legal", policy_id: "random-legal@tiny", games: 2 }],
      excluded_abilities: ["Illusion"],
      max_turns: 6,
      n_parallel: 2,
      latency_budget_ms: { preview_ms: 30000, battle_ms: 30000 },
      handshake_timeout_ms: 60000,
      max_line_bytes: 4000000,
    },
    ppo: {
      epochs: 2,
      minibatch_size: 16,
      learning_rate_micro: 1000,
      discount_micro: 1000000,
      gae_lambda_micro: 950000,
      clip_micro: 200000,
      value_coef_micro: 500000,
      entropy_coef_micro: 10000,
      max_grad_norm_micro: 500000,
      adam_beta1_micro: 900000,
      adam_beta2_micro: 999000,
      adam_eps_nano: 10,
      normalize_advantages: true,
    },
    learner: { device: "cpu", threads: 1 },
  };
  // A LeagueRunRecord (ADR-009 section 3): one generation of the tiny run, tied to that config by its hash.
  const pairId = `policy:${"a".repeat(64)}`;
  const leagueRun = {
    schema_version: 1,
    run_id: "tiny-run",
    run_config_hash: await runConfigHash(runConfig as unknown as TrainingRunConfig),
    generation: 1,
    seed_derivation: LEAGUE_SEED_DERIVATION,
    plan_hash: "b".repeat(64),
    pair_policy_id: pairId,
    games: 6,
    played: 5,
    forfeits: 1,
    forfeit_causes: [{ cause: "crash", count: 1 }],
    episodes: 10,
    opponents: [
      { opponent_policy_id: pairId, games: 4 },
      { opponent_policy_id: "random-legal@tiny", games: 2 },
    ],
  };
  // A RunMetrics record (ADR-010 erratum A1, item 3): one generation of the tiny run on a CPU learner.
  const runMetrics = {
    schema_version: 1,
    run_config_hash: await runConfigHash(runConfig as unknown as TrainingRunConfig),
    generation: 1,
    phase_ms: { play: 41_250, write: 310, load: 1_900, train: 8_400, seal: 120 },
    counts: { games: 6, decisions: 480, optimiser_steps: 36 },
    per_hour: { games: 523, decisions: 205_714, optimiser_steps: 15_428 },
    peak_accelerator_memory_bytes: null,
    device: "cpu",
    threads: 1,
  };
  const runMetricsWith = (change: (r: Record<string, unknown>) => void) => mutated(runMetrics, change);
  const leagueRunWith = (change: (r: Record<string, unknown>) => void) => mutated(leagueRun, change);
  const runConfigWith = (change: (c: Record<string, unknown>) => void) => mutated(runConfig, change);
  const runSection = (c: Record<string, unknown>, name: string) => c[name] as Record<string, unknown>;

  const valid: Record<string, unknown> = {
    "engine-data/golden": engineData,
    "checkpoint-record/battle": checkpointRecord,
    "checkpoint-record/preview-fresh": recordWith((r) => {
      r.role = "preview";
      delete r.parent;
      r.league_opponents = [];
    }),
    "checkpoint-record/exploiter": recordWith((r) => {
      r.lineage_id = "exploiter-of-main-0";
      r.exploiter_target = hexOf("2");
      r.league_opponents = [hexOf("2")];
    }),
    "team-vocabulary/golden": vocabulary,
    "team-record/golden": teamRecord,
    "team-label-set/golden": labelSet,
    "team-generator-config/golden": generatorConfig,
    "team-dataset-manifest/golden": datasetManifest,
    "team-dataset-manifest/merged": datasetWith((d) => {
      d.merged_cells = [{ stratum: 7, cells: [7, 9] }];
      d.splits = [datasetManifest.splits[0], { split: "validation", teams: 3, cells: [{ stratum: 0, pilot_hits: 21_000, quota: 2 }, { stratum: 7, pilot_hits: 40, quota: 1 }] }];
    }),
    "team-snapshot-manifest/first": snapshot,
    "team-snapshot-manifest/extended": extendedSnapshot,
    "training-run-config/tiny": runConfig,
    "training-run-config/no-baselines": runConfigWith((c) => (runSection(c, "generation").baseline_games = [])),
    "training-run-config/target-kl": runConfigWith((c) => (runSection(c, "ppo").target_kl_micro = 20000)),
    "training-run-config/team-dataset": runConfigWith((c) => (c.team_dataset = { path: ".tmp/datasets/train-v1", dataset_hash: hexOf("5") })),
    "run-metrics/cpu": runMetrics,
    "run-metrics/cuda": runMetricsWith((r) => {
      r.device = "cuda";
      r.peak_accelerator_memory_bytes = 3_221_225_472;
    }),
    "run-metrics/seeded-pair": runMetricsWith((r) => {
      r.generation = 0;
      r.phase_ms = { play: 0, write: 95, load: 0, train: 0, seal: 60 };
      r.counts = { games: 0, decisions: 0, optimiser_steps: 0 };
      r.per_hour = { games: 0, decisions: 0, optimiser_steps: 0 };
    }),
    "league-run-record/tiny": leagueRun,
    "league-run-record/self-play-only": leagueRunWith((r) => {
      r.games = 4;
      r.played = 4;
      r.forfeits = 0;
      delete r.forfeit_causes;
      r.episodes = 8;
      r.opponents = [{ opponent_policy_id: pairId, games: 4 }];
    }),
    "league-run-record/mixed-causes": leagueRunWith((r) => {
      r.games = 8;
      r.played = 5;
      r.forfeits = 3;
      r.forfeit_causes = [{ cause: "crash", count: 1 }, { cause: "latency_overrun", count: 2 }];
      r.opponents = [{ opponent_policy_id: pairId, games: 6 }, { opponent_policy_id: "random-legal@tiny", games: 2 }];
    }),
    "training-example/preview": exampleOf("preview"),
    "training-example/battle": exampleOf("battle"),
    "training-example/legacy-battle": legacyExample,
    "training-set/league": training.set,
    "training-set/legacy": legacySet,
    "evaluation-run-manifest/sealed": sealedManifest,
    "evaluation-run-manifest/open-empty": { schema_version: 1, run_id: "eval-run-2", sealed: false, episode_hashes: [] },
    "request-identity/a1": identity,
    "request-identity/server": { ...identity, rqid: 5 },
    "legal-choice-set/preview": legal,
    "decision-context/battle": context,
    "engine-request/preview": preview.request,
    "engine-request/turn": turn.request,
    "engine-request/force-switch": force.request,
    "player-channel-log/p1": { schema_version: 1, data_class: "synthetic", player: "p1", lines: record.channels.p1 },
    "private-team-spec/synthetic": team,
    "version-manifest/golden": record.manifest,
    "preview-observation/p1": previewObservation,
    "battle-observation/p1": battleObservation,
    "policy-query/battle": battleQuery,
    "policy-query/preview": previewQuery,
    "policy-query/battle-a2": battleQueryA2,
    "policy-reply/recommendation": replyOf({ kind: "recommendation", recommendation }),
    "policy-reply/unsupported-candidate": replyOf({ kind: "error", error: "unsupported_candidate", detail: "a candidate has a shape the worker does not know" }),
    "policy-reply/invalid-query": replyOf({ kind: "error", error: "invalid_query", detail: "the query's identity differs from its observation's identity" }),
    "policy-handshake/hello": hello,
    "policy-handshake/ready": ready,
    "recommendation/in-budget": {
      schema_version: 1,
      identity,
      choice: "move 1 1, move 1 2",
      source_policy_id: "first-legal@golden",
      elapsed_ms: 3,
      deadline_outcome: "in_budget",
    },
    "episode/p1": episode,
    "episode/league": leagueEpisode,
    "episode/evaluation-sealed": sealedEpisode,
  };
  // Schema-valid, refused by readEpisode (the sparring rule and the class rule).
  // No valid sparring sample exists: it must be private, and the data-class
  // guard refuses a committed "data_class": "private".
  const readerInvalid: Record<string, unknown> = {
    // Schema-valid, refused by the CheckpointRecord reader: the arrays are sorted and unique, and no id refers to the record itself.
    "checkpoint-record/unsorted-league-opponents": recordWith((r) => (r.league_opponents = [hexOf("4"), hexOf("3")])),
    // Schema-valid, refused by the TrainingRunConfig reader: the model, encoder and baseline rules the profile cannot say.
    "training-run-config/heads-do-not-divide-width": runConfigWith((c) => (runSection(c, "model").n_heads = 3)),
    "training-run-config/picks-exceed-team": runConfigWith((c) => {
      const encoder = runSection(runSection(c, "model"), "encoder");
      encoder.max_team = 3;
      encoder.max_picks = 4;
    }),
    "training-run-config/duplicate-baseline": runConfigWith((c) => {
      const games = runSection(c, "generation");
      games.baseline_games = [...(games.baseline_games as unknown[]), ...(games.baseline_games as unknown[])];
    }),
    // Schema-valid, refused by the LeagueRunRecord reader: the counts agree, and the opponents are sorted and unique.
    "league-run-record/counts-disagree": leagueRunWith((r) => (r.forfeits = 2)),
    "league-run-record/episodes-not-two-per-game": leagueRunWith((r) => (r.episodes = 9)),
    "league-run-record/unsorted-opponents": leagueRunWith((r) => (r.opponents = [...(leagueRun.opponents as unknown[])].reverse())),
    "league-run-record/duplicate-opponent": leagueRunWith((r) => (r.opponents = [{ opponent_policy_id: pairId, games: 3 }, { opponent_policy_id: pairId, games: 3 }])),
    "league-run-record/causes-do-not-sum": leagueRunWith((r) => (r.forfeit_causes = [{ cause: "crash", count: 2 }])),
    "league-run-record/unsorted-causes": leagueRunWith((r) => {
      r.games = 8;
      r.forfeits = 3;
      r.forfeit_causes = [{ cause: "latency_overrun", count: 2 }, { cause: "crash", count: 1 }];
      r.opponents = [{ opponent_policy_id: pairId, games: 6 }, { opponent_policy_id: "random-legal@tiny", games: 2 }];
    }),
    "league-run-record/duplicate-cause": leagueRunWith((r) => {
      r.forfeits = 2;
      r.games = 7;
      r.forfeit_causes = [{ cause: "crash", count: 1 }, { cause: "crash", count: 1 }];
      r.opponents = [{ opponent_policy_id: pairId, games: 5 }, { opponent_policy_id: "random-legal@tiny", games: 2 }];
    }),
    "league-run-record/opponents-do-not-sum": leagueRunWith((r) => (r.opponents = [{ opponent_policy_id: pairId, games: 4 }, { opponent_policy_id: "random-legal@tiny", games: 1 }])),
    "checkpoint-record/duplicate-training-sets": recordWith((r) => (r.training_set_hashes = [hexOf("5"), hexOf("5")])),
    "checkpoint-record/parent-is-self": recordWith((r) => (r.parent = r.checkpoint_id)),
    "checkpoint-record/opponent-is-self": recordWith((r) => (r.league_opponents = [r.checkpoint_id])),
    "checkpoint-record/exploiter-of-self": recordWith((r) => (r.exploiter_target = r.checkpoint_id)),
    "engine-data/unsorted-species": { ...engineData, species: [speciesB, speciesA] },
    "engine-data/duplicate-species": { ...engineData, species: [speciesA, speciesA] },
    "engine-data/dangling-type": mutated(engineData, (t) => ((t.species as { types: string[] }[])[0] as { types: string[] }).types = ["nosuchtype"]),
    "engine-data/short-damage-row": mutated(engineData, (t) => ((t.types as { damage_taken: number[] }[])[0] as { damage_taken: number[] }).damage_taken.pop()),
    // Schema-valid, refused by the readers: the hashes must be sorted and unique.
    "evaluation-run-manifest/unsorted": { schema_version: 1, run_id: "eval-run-3", sealed: false, episode_hashes: [hashB, hashA] },
    "evaluation-run-manifest/duplicate": { schema_version: 1, run_id: "eval-run-3", sealed: false, episode_hashes: [hashA, hashA] },
    // Schema-valid, refused by readTrainingSet: a learner_ready set must name its policy.
    "training-set/learner-ready-no-policy": mutated(training.set, (t) => delete t.behaviour_policy_id),
    // Schema-valid, refused by the stream check: choice must be among the candidates.
    "training-example/choice-not-in-candidates": { ...exampleOf("battle"), choice: "not a candidate" },
    // Schema-valid, refused by the worker's and the client's readers: the profile cannot compare fields.
    "policy-query/identity-differs": mutated(battleQuery, (q) => (q.identity = { ...battleQuery.identity, request_seq: battleQuery.identity.request_seq + 1 })),
    "policy-query/duplicate-candidate": mutated(battleQuery, (q) => ((q.candidates as unknown[])[1] = (battleQuery.candidates as unknown[])[0])),
    "policy-handshake/unsorted-contracts": helloWith((m) => (m.contracts = contractsOf().reverse())),
    "policy-handshake/duplicate-contracts": helloWith((m) => (m.contracts = [contractsOf()[0], contractsOf()[0]])),
    "policy-handshake/unsorted-checkpoints": readyWith((m) => (m.checkpoints = [...checkpoints].reverse())),
    // Schema-valid, refused by the team readers (ADR-013 section 4): what the profile cannot compute or compare.
    "team-vocabulary/unsorted-roster": vocabularyWith((v) => (v.roster = [...vocabulary.roster].reverse())),
    "team-vocabulary/duplicate-moves": vocabularyWith((v) => ((v.roster as { moves: string[] }[])[0] as { moves: string[] }).moves = [(vocabulary.roster[0] as { moves: string[] }).moves[0] as string, (vocabulary.roster[0] as { moves: string[] }).moves[0] as string]),
    "team-vocabulary/unsorted-natures": vocabularyWith((v) => (v.natures = [...vocabulary.natures].reverse())),
    "team-vocabulary/unsorted-items": vocabularyWith((v) => (v.items = [...vocabulary.items].reverse())),
    "team-vocabulary/picked-over-max": vocabularyWith((v) => (v.rules = { ...vocabulary.rules, picked_team_size: vocabulary.rules.max_team_size + 1 })),
    "team-vocabulary/min-over-max": vocabularyWith((v) => (v.rules = { ...vocabulary.rules, min_team_size: vocabulary.rules.max_team_size + 1 })),
    "team-vocabulary/stale-engine-data-hash": vocabularyWith((v) => (v.engine_data_hash = hexOf("f"))),
    "team-vocabulary/unknown-species": vocabularyWith((v) => ((v.roster as { id: string }[])[0] as { id: string }).id = "aaaaaaaaa"),
    "team-vocabulary/foreign-ability": vocabularyWith((v) => ((v.roster as { abilities: string[] }[])[0] as { abilities: string[] }).abilities = [(vocabularyTable.abilities.find((a) => !(vocabularyTable.species[0] as { abilities: string[] }).abilities.includes(a.id)) as { id: string }).id]),
    "team-record/team-hash-mismatch": recordWithTeam((r) => (r.team_hash = hexOf("a"))),
    "team-record/canonical-hash-mismatch": recordWithTeam((r) => (r.canonical_team_hash = hexOf("a"))),
    "team-record/paste-not-engine-export": recordWithTeam((r) => (r.paste = `${teamRecord.paste}\n\n`)),
    "team-record/paste-unreproduced-line": recordWithTeam((r) => (r.paste = `${teamRecord.paste}\nSPs: 1 HP`)),
    "team-record/duplicate-label-ids": recordWithTeam((r) => (r.labels = [teamRecord.labels[0], teamRecord.labels[0], teamRecord.labels[2]])),
    "team-record/labels-out-of-order": recordWithTeam((r) => (r.labels = [teamRecord.labels[1], teamRecord.labels[0], teamRecord.labels[2]])),
    "team-record/value-not-in-label": recordWithTeam((r) => (r.labels = [{ label_id: "speed_control", value: "sideways" }, teamRecord.labels[1], teamRecord.labels[2]])),
    "team-record/foreign-label-set": recordWithTeam((r) => (r.label_set_hash = hexOf("a"))),
    "team-label-set/duplicate-label-ids": mutated(labelSet, (l) => (l.labels = [labelSet.labels[0], labelSet.labels[0]])),
    "team-label-set/duplicate-values": mutated(labelSet, (l) => ((l.labels as { values: unknown[] }[])[0] as { values: unknown[] }).values = [(labelSet.labels[0] as { values: unknown[] }).values[0], (labelSet.labels[0] as { values: unknown[] }).values[0]]),
    "team-label-set/default-is-a-value": mutated(labelSet, (l) => ((l.labels as { default: string }[])[1] as { default: string }).default = "yes"),
    "team-label-set/unsorted-ids": mutated(labelSet, (l) => ((l.labels as { values: { when: { moves: string[] } }[] }[])[0] as { values: { when: { moves: string[] } }[] }).values = [{ value: "both", when: { kind: "has_move", moves: ["trickroom", "tailwind"] } } as never]),
    "team-label-set/unknown-engine-id": mutated(labelSet, (l) => ((l.labels as { values: { when: { moves: string[] } }[] }[])[0] as { values: { when: { moves: string[] } }[] }).values = [{ value: "ghost", when: { kind: "has_move", moves: ["nosuchmove"] } } as never]),
    "team-generator-config/unsorted-item-weights": configWith((c) => (c.item_weights = [...generatorConfig.item_weights].reverse())),
    "team-generator-config/duplicate-item-weights": configWith((c) => (c.item_weights = [generatorConfig.item_weights[0], generatorConfig.item_weights[0]])),
    "team-generator-config/unknown-item": configWith((c) => (c.item_weights = [{ item: "nosuchitem", weight: 1 }])),
    "team-dataset-manifest/splits-out-of-order": datasetWith((d) => (d.splits = [...datasetManifest.splits].reverse())),
    "team-dataset-manifest/stratum-not-lowest-member": datasetWith((d) => (d.merged_cells = [{ stratum: 7, cells: [8, 9] }])),
    "team-dataset-manifest/split-cell-is-a-member": datasetWith((d) => (d.merged_cells = [{ stratum: 0, cells: [0, 7] }])),
    "team-dataset-manifest/merged-stratum-missing": datasetWith((d) => (d.merged_cells = [{ stratum: 3, cells: [3, 5] }])),
    "team-dataset-manifest/teams-not-quota-sum": datasetWith((d) => (d.splits = [{ ...datasetManifest.splits[0], teams: 6 }, datasetManifest.splits[1]])),
    "team-dataset-manifest/unsorted-cells": datasetWith((d) => (d.splits = [{ ...datasetManifest.splits[0], cells: [...(datasetManifest.splits[0] as { cells: unknown[] }).cells].reverse() }, datasetManifest.splits[1]])),
    "team-snapshot-manifest/unsorted-dataset-hashes": snapshotWith((s) => (s.dataset_hashes = [hexOf("b"), hexOf("a")])),
    "team-snapshot-manifest/wrong-logical-hash": snapshotWith((s) => (s.logical_snapshot_hash = hexOf("a"))),
    "team-snapshot-manifest/object-path-differs": snapshotWith((s) => ((s.shards as { object_key: string }[])[0] as { object_key: string }).object_key = `pub/v1/shards/${hexOf("a")}`),
    "team-snapshot-manifest/ranges-do-not-tile": snapshotWith((s) => (((s.shards as { cell_ranges: { first_row: number }[] }[])[0] as { cell_ranges: { first_row: number }[] }).cell_ranges[1] as { first_row: number }).first_row = 4),
    "team-snapshot-manifest/ranges-out-of-order": snapshotWith((s) => ((s.shards as { cell_ranges: unknown[] }[])[0] as { cell_ranges: unknown[] }).cell_ranges = [{ stratum: 7, first_row: 0, rows: 2 }, { stratum: 0, first_row: 2, rows: 3 }]),
    "team-snapshot-manifest/rows-not-range-sum": snapshotWith((s) => ((s.shards as { rows: number }[])[0] as { rows: number }).rows = 6),
    "team-snapshot-manifest/cells-not-shard-sum": snapshotWith((s) => (s.cells = [{ split: "dev", stratum: 0, rows: 4 }, { split: "dev", stratum: 7, rows: 2 }])),
    "team-snapshot-manifest/cells-missing": snapshotWith((s) => (s.cells = [{ split: "dev", stratum: 0, rows: 3 }])),
    "team-snapshot-manifest/cells-out-of-order": snapshotWith((s) => (s.cells = [{ split: "dev", stratum: 7, rows: 2 }, { split: "dev", stratum: 0, rows: 3 }])),
    "team-snapshot-manifest/shards-out-of-order": mutated(extendedSnapshot, (s) => (s.shards = [...(extendedSnapshot.shards as unknown[])].reverse())),
    "team-snapshot-manifest/duplicate-shard": mutated(snapshot, (s) => (s.shards = [snapshotBase.shards[0], { ...snapshotBase.shards[0], split: "validation" }])),
    "training-set/unsorted-episodes": { ...training.set, episode_hashes: [hashB, hashA] },
    "training-set/duplicate-manifests": { ...training.set, evaluation_manifest_hashes: [hashA, hashA] },
    "episode/sparring-synthetic": collected({ kind: "sparring", session_id: "sparring-session-1" }),
    "episode/class-below-opponent": collected({ ...league, opponent_team_data_class: "private" }),
  };
  const invalid: Record<string, unknown> = {
    "checkpoint-record/no-manifest": recordWith((r) => delete r.manifest),
    "checkpoint-record/short-checkpoint-id": recordWith((r) => (r.checkpoint_id = "abc")),
    "checkpoint-record/uppercase-parent": recordWith((r) => (r.parent = hexOf("A"))),
    "checkpoint-record/unknown-role": recordWith((r) => (r.role = "value")),
    "checkpoint-record/public-class": recordWith((r) => (r.data_class = "public")),
    "checkpoint-record/blank-lineage": recordWith((r) => (r.lineage_id = "")),
    "checkpoint-record/no-learner-state": recordWith((r) => delete r.learner_state_hash),
    "run-metrics/extra-field": runMetricsWith((r) => (r.notes = "x")),
    "run-metrics/fractional-ms": runMetricsWith((r) => (r.phase_ms = { ...runMetrics.phase_ms, play: 1.5 })),
    "run-metrics/negative-memory": runMetricsWith((r) => (r.peak_accelerator_memory_bytes = -1)),
    "run-metrics/missing-phase": runMetricsWith((r) => (r.phase_ms = { play: 1, write: 1, load: 1, train: 1 })),
    "run-metrics/unknown-device": runMetricsWith((r) => (r.device = "tpu")),
    "run-metrics/zero-threads": runMetricsWith((r) => (r.threads = 0)),
    "run-metrics/short-config-hash": runMetricsWith((r) => (r.run_config_hash = "abc")),
    "run-metrics/no-counts": runMetricsWith((r) => delete r.counts),
    "league-run-record/unknown-forfeit-cause": leagueRunWith((r) => (r.forfeit_causes = [{ cause: "timeout", count: 1 }])),
    "league-run-record/empty-forfeit-causes": leagueRunWith((r) => (r.forfeit_causes = [])),
    "league-run-record/zero-cause-count": leagueRunWith((r) => (r.forfeit_causes = [{ cause: "crash", count: 0 }])),
    "league-run-record/extra-field": leagueRunWith((r) => (r.notes = "x")),
    "league-run-record/fractional-games": leagueRunWith((r) => (r.games = 6.5)),
    "league-run-record/zero-generation": leagueRunWith((r) => (r.generation = 0)),
    "league-run-record/next-derivation": leagueRunWith((r) => (r.seed_derivation = "league-seed/2")),
    "league-run-record/short-config-hash": leagueRunWith((r) => (r.run_config_hash = "abc")),
    "league-run-record/bare-pair-id": leagueRunWith((r) => (r.pair_policy_id = "a".repeat(64))),
    "league-run-record/no-plan-hash": leagueRunWith((r) => delete r.plan_hash),
    "training-run-config/extra-field": runConfigWith((c) => (c.notes = "x")),
    "training-run-config/fractional-rate": runConfigWith((c) => (runSection(c, "ppo").learning_rate_micro = 0.5)),
    "training-run-config/zero-target-kl": runConfigWith((c) => (runSection(c, "ppo").target_kl_micro = 0)),
    "training-run-config/zero-games": runConfigWith((c) => (runSection(c, "generation").games = 0)),
    "training-run-config/unknown-device": runConfigWith((c) => (runSection(c, "learner").device = "tpu")),
    "training-run-config/no-ppo": runConfigWith((c) => delete c.ppo),
    "training-run-config/blank-run-name": runConfigWith((c) => (runSection(c, "seeds").run_key = "")),
    "training-run-config/unknown-baseline": runConfigWith((c) => {
      ((runSection(c, "generation").baseline_games as Record<string, unknown>[])[0] as Record<string, unknown>).actor = "heuristic";
    }),
    "training-run-config/dataset-short-hash": runConfigWith((c) => (c.team_dataset = { path: ".tmp/datasets/train-v1", dataset_hash: "abc" })),
    "training-run-config/dataset-no-path": runConfigWith((c) => (c.team_dataset = { dataset_hash: hexOf("5") })),
    "training-run-config/next-version": runConfigWith((c) => (c.schema_version = 2)),
    "checkpoint-record/extra-field": recordWith((r) => (r.verdicts = [])),
    "checkpoint-record/bad-encoder-version": recordWith((r) => (r.encoder_version = "stub-1")),
    "checkpoint-record/next-version": recordWith((r) => (r.schema_version = 2)),
    "engine-data/damage-code-4": mutated(engineData, (t) => ((t.types as { damage_taken: number[] }[])[0] as { damage_taken: number[] }).damage_taken[0] = 4),
    "engine-data/three-types": mutated(engineData, (t) => ((t.species as { types: string[] }[])[0] as { types: string[] }).types = ["bug", "dark", "fire"]),
    "engine-data/unknown-category": mutated(engineData, (t) => ((t.moves as { category: string }[])[0] as { category: string }).category = "Typeless"),
    "engine-data/extra-field": { ...engineData, learnsets: [] },
    "engine-data/next-version": { ...engineData, schema_version: 2 },
    "engine-data/short-commit": { ...engineData, showdown_commit: "7244fe8" },
    "team-vocabulary/no-rules": vocabularyWith((v) => delete v.rules),
    "team-vocabulary/short-commit": vocabularyWith((v) => (v.showdown_commit = "7244fe8")),
    "team-vocabulary/extra-field": vocabularyWith((v) => (v.learnsets = [])),
    "team-vocabulary/next-version": vocabularyWith((v) => (v.schema_version = 2)),
    "team-vocabulary/bad-stat": vocabularyWith((v) => (v.natures = [{ ...vocabulary.natures[0], plus: "hp" }])),
    "team-vocabulary/empty-pool": vocabularyWith((v) => (v.roster = [{ ...vocabulary.roster[0], moves: [] }])),
    "team-vocabulary/zero-item-clause": vocabularyWith((v) => (v.rules = { ...vocabulary.rules, item_clause: 0 })),
    "team-record/no-paste": recordWithTeam((r) => delete r.paste),
    "team-record/uppercase-team-hash": recordWithTeam((r) => (r.team_hash = hexOf("A"))),
    "team-record/extra-field": recordWithTeam((r) => (r.rating = 1500)),
    "team-record/next-version": recordWithTeam((r) => (r.schema_version = 2)),
    "team-record/sealed-split": recordWithTeam((r) => (r.origin = { ...teamRecord.origin, split: "sealed" })),
    "team-record/negative-index": recordWithTeam((r) => (r.origin = { ...teamRecord.origin, index: -1 })),
    "team-record/float-draw-counter": recordWithTeam((r) => (r.origin = { ...teamRecord.origin, draw_counter: 1.5 })),
    "team-record/label-extra-field": recordWithTeam((r) => (r.labels = [{ ...teamRecord.labels[0], weight: 1 }])),
    "team-record/empty-paste": recordWithTeam((r) => (r.paste = "")),
    "team-label-set/no-labels": mutated(labelSet, (l) => (l.labels = [])),
    "team-label-set/unknown-predicate": mutated(labelSet, (l) => ((l.labels as { values: unknown[] }[])[0] as { values: unknown[] }).values = [{ value: "x", when: { kind: "has_type", types: ["fire"] } }]),
    "team-label-set/nested-all-of": mutated(labelSet, (l) => ((l.labels as { values: unknown[] }[])[0] as { values: unknown[] }).values = [{ value: "x", when: { kind: "all_of", of: [{ kind: "all_of", of: [] }] } }]),
    "team-label-set/empty-ids": mutated(labelSet, (l) => ((l.labels as { values: unknown[] }[])[0] as { values: unknown[] }).values = [{ value: "x", when: { kind: "has_move", moves: [] } }]),
    "team-label-set/extra-field": mutated(labelSet, (l) => (l.scores = [])),
    "team-generator-config/float-weight": configWith((c) => (c.default_item_weight = 0.5)),
    "team-generator-config/iv-over-31": configWith((c) => (c.iv = 32)),
    "team-generator-config/pilot-below-40000": configWith((c) => (c.pilot_draws = 39_999)),
    "team-generator-config/quantile-as-fraction": configWith((c) => (c.fill_quantile_permille = 0.99)),
    "team-generator-config/unknown-policy": configWith((c) => (c.gender_policy = "male")),
    "team-generator-config/no-d-max": configWith((c) => delete c.d_max),
    "team-dataset-manifest/no-splits": datasetWith((d) => (d.splits = [])),
    "team-dataset-manifest/sealed-split": datasetWith((d) => (d.splits = [{ ...datasetManifest.splits[0], split: "sealed" }])),
    "team-dataset-manifest/negative-quota": datasetWith((d) => (d.splits = [{ ...datasetManifest.splits[0], cells: [{ stratum: 0, pilot_hits: 1, quota: -1 }] }])),
    "team-dataset-manifest/short-stream-hex": datasetWith((d) => (d.key = { ...datasetManifest.key, generation_key: "abc" })),
    "team-dataset-manifest/merged-single-cell": datasetWith((d) => (d.merged_cells = [{ stratum: 7, cells: [7] }])),
    "team-dataset-manifest/extra-field": datasetWith((d) => (d.notes = "x")),
    "team-dataset-manifest/no-rejection-log": datasetWith((d) => delete d.rejection_log),
    "team-snapshot-manifest/no-shards": snapshotWith((s) => (s.shards = [])),
    "team-snapshot-manifest/bad-object-path": snapshotWith((s) => ((s.shards as { object_key: string }[])[0] as { object_key: string }).object_key = "shards/x"),
    "team-snapshot-manifest/writer-no-version": snapshotWith((s) => (s.writer = { name: "duckdb", binary_sha256: hexOf("7") })),
    "team-snapshot-manifest/empty-range": snapshotWith((s) => (((s.shards as { cell_ranges: { rows: number }[] }[])[0] as { cell_ranges: { rows: number }[] }).cell_ranges[0] as { rows: number }).rows = 0),
    "team-snapshot-manifest/short-dataset-hash": snapshotWith((s) => (s.dataset_hashes = ["abc"])),
    "team-snapshot-manifest/next-version": snapshotWith((s) => (s.schema_version = 2)),
    "team-snapshot-manifest/extra-field": snapshotWith((s) => (s.uploaded_by = "x")),
    "training-example/provenance": { ...exampleOf("battle"), provenance: { seed: record.seed } },
    "training-example/no-episode-hash": mutated(exampleOf("battle"), (e) => delete e.episode_hash),
    "training-example/omniscient-observation": mutated(exampleOf("battle"), (e) => {
      (e.observation as Record<string, unknown>).omniscient_log = ["|split|p2"];
    }),
    "training-example/short-episode-hash": { ...exampleOf("battle"), episode_hash: "abc" },
    "training-example/empty-candidates": { ...exampleOf("battle"), candidates: [] },
    "training-example/too-many-candidates": { ...exampleOf("battle"), candidates: candidatesOf(1025) },
    "training-example/candidate-no-pending": mutated(exampleOf("battle"), (e) => delete (e.candidates as Record<string, unknown>[])[0]?.pending),
    "training-example/candidate-verdict": mutated(exampleOf("battle"), (e) => ((e.candidates as Record<string, unknown>[])[0] as Record<string, unknown>).legal = true),
    "training-example/unknown-outcome": { ...exampleOf("battle"), outcome: "draw" },
    "training-set/learner-ready-false": { ...training.set, learner_ready: false },
    "training-set/empty-policy-id": { ...training.set, behaviour_policy_id: "" },
    "training-set/no-stream-hash": mutated(training.set, (t) => delete t.stream_sha256),
    "training-set/no-examples": { ...training.set, examples: 0 },
    "training-set/public-class": { ...training.set, data_class: "public" },
    "evaluation-run-manifest/no-run-id": { schema_version: 1, sealed: true, episode_hashes: [] },
    "evaluation-run-manifest/uppercase-hash": { ...sealedManifest, episode_hashes: [hashA.toUpperCase()] },
    "evaluation-run-manifest/sealed-flag-string": { ...sealedManifest, sealed: "yes" },
    "policy-query/no-candidates": { ...battleQuery, candidates: [] },
    "policy-query/too-many-candidates": { ...battleQuery, candidates: candidatesOf(1025) },
    "policy-query/seed": { ...battleQuery, seed: record.seed },
    "policy-query/candidate-verdict": mutated(battleQuery, (q) => ((q.candidates as Record<string, unknown>[])[0] as Record<string, unknown>).legal = true),
    "policy-query/unknown-sampling": { ...battleQuery, sampling: { kind: "temperature", t: 2 } },
    "policy-query/u-negative": { ...battleQuery, sampling: { kind: "sample", u: -1 } },
    "policy-query/argmax-with-u": { ...battleQuery, sampling: { kind: "argmax", u: 1 } },
    "policy-query/no-policy-id": mutated(battleQuery, (q) => delete q.policy_id),
    "policy-query/no-query-id": mutated(battleQuery, (q) => delete q.query_id),
    "policy-query/omniscient-observation": mutated(battleQuery, (q) => ((q.observation as Record<string, unknown>).omniscient_log = ["|split|p2"])),
    "policy-reply/unknown-error": replyOf({ kind: "error", error: "timeout", detail: "slow" }),
    "policy-reply/unknown-kind": replyOf({ kind: "retry" }),
    "policy-reply/extra-field": { ...replyOf({ kind: "recommendation", recommendation }), probabilities: [1] },
    "policy-reply/no-query-id": mutated(replyOf({ kind: "recommendation", recommendation }), (r) => delete r.query_id),
    "policy-reply/bad-outcome": replyOf({ kind: "recommendation", recommendation: { ...recommendation, deadline_outcome: "late" } }),
    "policy-handshake/hello-with-encoder": helloWith((m) => (m.encoder_version = "stub-1")),
    "policy-handshake/unknown-contract": helloWith((m) => (m.contracts = [{ ...contractsOf()[0], name: "Episode" }])),
    "policy-handshake/short-schema-hash": helloWith((m) => (m.contracts = [{ ...contractsOf()[0], schema_sha256: "abc" }])),
    "policy-handshake/ready-no-checkpoints": readyWith((m) => delete m.checkpoints),
    "policy-handshake/unknown-kind": helloWith((m) => (m.kind = "bye")),
    "request-identity/rqid-zero": { ...identity, rqid: 0 },
    "legal-choice-set/next-version": mutated(legal, (s) => (s.schema_version = 2)),
    "decision-context/negative-budget": { ...context, policy_budget_ms: -1 },
    "engine-request/z-move-offer": mutated(turn.request, (r) => {
      ((r.active as unknown[])[0] as Record<string, unknown>).canZMove = [];
    }),
    "player-channel-log/public-class": {
      schema_version: 1,
      data_class: "public",
      player: "p1",
      lines: record.channels.p1,
    },
    "private-team-spec/extra-set-field": mutated(team, (s) => {
      ((s.team as unknown[])[0] as Record<string, unknown>).secret = "x";
    }),
    "version-manifest/short-commit": { ...record.manifest, showdown_commit: "7244fe8" },
    "preview-observation/omniscient-log": { ...previewObservation, omniscient_log: ["|split|p2"] },
    "battle-observation/opponent-team": { ...battleObservation, opponent_team: team },
    "recommendation/unknown-outcome": {
      schema_version: 1,
      identity,
      choice: "move 1 1, move 1 2",
      source_policy_id: "first-legal@golden",
      elapsed_ms: 3,
      deadline_outcome: "late",
    },
    "episode/omniscient-log": { ...episode, omniscient_log: [] },
    "episode/collection-no-run-id": collected({ ...classed, kind: "league", seed: record.seed, opponent_policy_id: "first-legal@golden" }),
    "episode/collection-league-sealed": collected({ ...league, ...classed, sealed: false }),
    "episode/collection-evaluation-no-seed": collected({ ...classed, kind: "evaluation", run_id: "eval-run-1", sealed: true }),
    "episode/collection-bad-seed": collected({ ...league, ...classed, seed: "1,2,3,4" }),
    "episode/collection-no-opponent-class": collected(league),
    "episode/collection-bad-opponent-class": collected({ ...league, opponent_team_data_class: "public" }),
    "episode/collection-unknown-kind": collected({ ...classed, kind: "training", run_id: "league-run-1" }),
    "episode/collection-sparring-run-id": collected({ kind: "sparring", session_id: "s", run_id: "r" }),
  };

  const samples: Record<string, unknown> = {};
  const evalSamples = evaluationSamples();
  const progressOnes = await progressSamples(engine);
  const privateOnes = privateSamples(engine);
  const mechanicsOnes = mechanicsSamples(engine);
  const privateManifests = privateSnapshotSamples({ manifest: record.manifest, vocabularyHash: teamVocabularyHash(vocabulary as never), labelSetHash });
  for (const kind of ["valid", "invalid", "readerInvalid"] as const) Object.assign(privateOnes[kind], privateManifests[kind]);
  for (const [kind, cases] of [["valid", { ...valid, ...evalSamples.valid, ...progressOnes.valid, ...mechanicsOnes.valid, ...privateOnes.valid }], ["invalid", { ...invalid, ...evalSamples.invalid, ...progressOnes.invalid, ...mechanicsOnes.invalid, ...privateOnes.invalid }], ["reader-invalid", { ...readerInvalid, ...evalSamples.readerInvalid, ...progressOnes.readerInvalid, ...mechanicsOnes.readerInvalid, ...privateOnes.readerInvalid }]] as const) {
    for (const [key, value] of Object.entries(cases)) {
      const [contract, name] = key.split("/") as [string, string];
      samples[`v1/${contract}/${kind}/${name}.json`] = value;
    }
  }
  const hashes: Record<string, string> = {};
  for (const [path, value] of Object.entries(samples)) {
    hashes[path] = createHash("sha256").update(jcs(value as Json)).digest("hex");
  }
  samples[SAMPLE_HASHES] = hashes;
  return samples;
}
