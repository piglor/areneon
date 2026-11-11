// One generation of the training loop's A1 play (ADR-010 §1): the frozen pair
// plays its match plan, every seat's Episode is written (a league collection
// record, the seed the battle ran with), and only the Episodes of the pair's
// own seats go through the training loader, which refuses any other policy.
// The plan is a pure function of the run config, the generation and the team
// pool, so a generation that was killed is replayed from the same plan.
//
// Teams are drawn from the committed synthetic fixtures (the golden team, the
// legal paste fixtures and the random-play corpus): the ADR-013 team
// generator is not built yet, and this pool is the interim. Every team drawn
// is reported by its hash, which becomes the team-set manifest of the run.

import { closeSync, fsyncSync, mkdirSync, openSync, readFileSync, writeSync } from "node:fs";
import { join } from "node:path";
import type { Engine, PokemonSet } from "./engine.ts";
import { jcs, type Json } from "./canonical-json.ts";
import { type DamageAwareActor, type DamageAwareOptions, damageAwareActor } from "./damage-aware-actor.ts";
import { type BeliefFeatureCounts, type BeliefFeatureSetup, createFeatureMemo, BELIEF_THREATS_PER_FOE, type BeliefThreatsSetup, type FeatureConfigs, type FeatureMemo, FEATURE_SET_3, FEATURE_SET_BELIEF } from "./candidate-features.ts";
import { configuredSetHash } from "./feature-build.ts";
import { BELIEF_EVIDENCE_MODES, type BeliefEvidenceMode, type BeliefFeaturesFactory, type BeliefFeaturesPort, createSheetBuilder, type SheetSetup, sheetStatement } from "./belief-features.ts";
import { type SheetCounts, sumSheetCounts } from "./sheet-counts.ts";
import { checkScenario } from "./mechanics-scenario.ts";
import { loadDamageAwareConfig, loadDamageAwareConfigV2 } from "./damage-aware-config.ts";
import type { BeliefWorldsFactory } from "./damage-aware-worlds.ts";
import { damageAwarePolicyId } from "./damage-aware-identity.ts";
import { type BaselineGameReport, buildLeagueRunRecord, type LeagueRunRecord } from "./league-run-record.ts";
import { loadMechanicsConfig } from "./mechanics-config.ts";
import { loadRulesSnapshot } from "./rules-load.ts";
import { leagueGameStreams, privateTeamSource } from "./league-seed.ts";
import { episodeFromRecord } from "./episode-io.ts";
import { episodeHash } from "./episode.ts";
import { addSince, countDecisions, profiling, startTimer, timedAsync } from "./profile.ts";
import { guardActor, type LatencyBudget, type PlayedGame } from "./evaluation-harness.ts";
import { firstLegal, GOLDEN_FORMAT, goldenTeamVocabulary, syntheticTeam, TEAMS_DIR, teamFixtureManifest } from "./golden.ts";
import { goldenWorldFactsData } from "./world-facts-data.ts";
import { resolveFormat, type VersionManifest } from "./formats.ts";
import { nodeSha256 } from "./player-channel.ts";
import { type PolicyClient } from "./policy-worker.ts";
import type { PolicyQueryInput } from "./policy-protocol.ts";
import { type PolicyGameOptions, policyGame, type PolicySeat } from "./policy-actor.ts";
import { readerSha256 } from "@areneon/public-state";
import { loadRandomCorpus } from "./random-corpus.ts";
import { randomLegalActor } from "./random-legal.ts";
import type { Actor, BattleRecord, ForfeitCause, Player } from "./runner.ts";
import { importPaste } from "./teams.ts";
import { stampTeam, type SyntheticTeamHashes, teamHash } from "./team-class.ts";
import { loadTrainingSet, type StoredEpisode, type TrainingSet, writeStoredTrainingStream, writeTrainingStream } from "./training-loader.ts";
import { type ChunkLimits } from "./chunked-file.ts";
import { readFeatureRows, writeFeatureRows, writePlayRows } from "./feature-recompute.ts";
import type { OpponentSheet } from "@areneon/schema/types/v1/opponent-sheet.ts";
import { readSheetBlocks, writePlaySheets } from "./sheet-recompute.ts";
import type { AnyCandidateFeatures } from "./feature-build.ts";
import type { TrainingRunConfig } from "./training-run-config.ts";
import type { Episode } from "./episode.ts";
import { loadSearchConfig, type SearchConfig, searchConfigHash } from "./search-config.ts";
import type { SearchSetup } from "./search-actor.ts";
import { type LabelContext, type LabelEpisodeRef, LabelError, labelCounts, labelPolicy, type LabelSetup, type LabelTotals, writeLabelFiles, writeLabelStream } from "./search-labeller.ts";

/** The teams a run may draw: synthetic by source, in a fixed order. */
export interface TrainingPool {
  /** Sorted by team hash, so an index names the same team on every machine. */
  teams: readonly PokemonSet[][];
  hashes: readonly string[];
  /** Every team of the pool, proven synthetic by where it was read from. */
  known_synthetic: SyntheticTeamHashes;
  /**
   * A private component (ADR-015 erratum E1 item 3): the LAST `count` entries of `teams` and `hashes` are the owner's, none of
   * them in `known_synthetic`, so `stampTeam` makes each of them `private`. Absent: every team is synthetic and the plan is the old one.
   */
  private?: PrivateComponent;
}

/** The private part of a pool: its size, the share of seats that draw from it and the hold-out flag of each of its teams. */
export interface PrivateComponent {
  count: number;
  /** A seat draws from this component when its `league-team-source/1` draw is below this many permille. */
  share_permille: number;
  /** One flag per private team, in the pool's order. */
  holdout: readonly boolean[];
}

/** What a plan needs of a pool: its size and its private component, if any. */
export interface PoolShape {
  size: number;
  private?: { count: number; share_permille: number };
}

export const poolShape = (pool: TrainingPool): PoolShape => ({ size: pool.teams.length, ...(pool.private === undefined ? {} : { private: { count: pool.private.count, share_permille: pool.private.share_permille } }) });

/**
 * The interim team pool (the team generator is ADR-013, not built): the golden
 * team, every legal paste fixture and every team of the random-play corpus.
 * All three are committed under `fixtures/` as CC0 synthetic data. A team with
 * a set whose ability is in `excludedAbilities` (the run config's list) is left out.
 */
export function trainingTeamPool(engine: Engine, excludedAbilities: readonly string[] = []): TrainingPool {
  const byHash = new Map<string, PokemonSet[]>();
  const add = (team: PokemonSet[]): void => void byHash.set(teamHash(team), team);
  add(syntheticTeam());
  const manifest = teamFixtureManifest();
  for (const [file, fixture] of Object.entries(manifest.fixtures)) {
    if (fixture.expect === "legal") add(importPaste(engine, readFileSync(join(TEAMS_DIR, file), "utf8")));
  }
  for (const battle of loadRandomCorpus()) {
    add(battle.teams.p1);
    add(battle.teams.p2);
  }
  const excluded = new Set(excludedAbilities);
  const hashes = [...byHash.keys()].filter((hash) => !(byHash.get(hash) as PokemonSet[]).some((set) => excluded.has(set.ability))).sort();
  if (hashes.length === 0) throw new GenerationError("the team pool is empty after the excluded abilities");
  return { teams: hashes.map((hash) => byHash.get(hash) as PokemonSet[]), hashes, known_synthetic: new Set(hashes) };
}

/**
 * A generation does not play a pool with a private component yet: its Episodes would be written into the public run directory
 * before the loader refuses them. The guard goes with the ticket that writes private outputs to the private root (#1063).
 */
export function refusePrivatePool(pool: TrainingPool): void {
  if (pool.private !== undefined) throw new GenerationError("a pool with a private component cannot be played by this generation yet");
}

/** The sorted hashes of the synthetic teams a plan draws (the cumulative team set): a private team's hash is never in it. */
export function syntheticDrawnHashes(plan: readonly GamePlan[], pool: TrainingPool): string[] {
  const syntheticCount = pool.teams.length - (pool.private?.count ?? 0);
  const drawn = new Set<string>();
  for (const entry of plan) for (const player of ["p1", "p2"] as const) if (entry.teams[player] < syntheticCount) drawn.add(pool.hashes[entry.teams[player]] as string);
  return [...drawn].sort();
}

/** What a team pool is built from: the part of a run config (or a teacher run) that names the format, the dataset and the abilities left out. */
export interface PoolRequest {
  format_id: string;
  team_dataset?: { path: string; dataset_hash: string };
  generation: { excluded_abilities: readonly string[] };
}

/** The scripted opponent of a baseline game; the policy takes the other seat. */
export interface BaselineOpponent {
  actor: "actor-rule" | "random-legal" | "damage-aware";
  policy_id: string;
}

/** What a damage-aware baseline seat is built from (everything of `DamageAwareOptions` but the engine and the per-game seed). */
export type DamageAwareSetup = Omit<DamageAwareOptions, "engine" | "seed" | "commit" | "data_class">;

/**
 * The reviewed records a damage-aware baseline reads, against the golden team vocabulary (the committed
 * export of the one supported format; another format is refused, never given another vocabulary).
 * `version` "v3" is the v1 scorer with the engine's choice-location convention (#1180); "v4" is v3 plus the two situational
 * actions (#1329), which also read the world facts' engine data (the stall moves) from the pinned checkout.
 */
export function damageAwareSetup(engine: Engine, formatId: string, version: "v1" | "v3" | "v4" = "v1"): DamageAwareSetup {
  if (formatId !== GOLDEN_FORMAT) throw new GenerationError(`the damage-aware baseline has a vocabulary for ${GOLDEN_FORMAT} only, not ${formatId}`);
  const vocabulary = goldenTeamVocabulary();
  const config = loadDamageAwareConfig(version, vocabulary);
  return {
    vocabulary,
    config,
    mechanics: loadMechanicsConfig(formatId),
    snapshot: loadRulesSnapshot(engine.commit),
    ...(config.situational === undefined ? {} : { facts_data: goldenWorldFactsData() }),
  };
}

/** The feature set a run reads and the records its rows are built from (ADR-022 section 3). */
export interface FeatureSetup {
  configs: FeatureConfigs;
  /** What the policy worker is expected to state at the handshake (and the rows must be of). */
  statement: { feature_set: string; feature_set_hash: string };
}

/**
 * The candidate feature set a run config names (`model.encoder.options.candidate_features`), set up from the
 * reviewed records of this checkout, or undefined when the config names none (then nothing about the run changes).
 * Fails closed: the hash this checkout computes (pin, mechanics and assumed-set configs, probe adapter, reader) must be
 * the one the config names, so a run never plays or trains on rows of another set than its checkpoints were made for.
 * `probe-damage-belief/1` (ADR-025 slice 5) also needs the belief port of the composition root (`beliefWorlds`; the bridge does
 * not import the opponent model); its world count and belief come from the committed `damage-aware.v2.json` record.
 */
export function featureSetup(engine: Engine, config: TrainingRunConfig, beliefWorlds?: BeliefWorldsFactory, beliefFeatures?: BeliefFeaturesFactory): FeatureSetup | undefined {
  const named = config.model.encoder.options?.candidate_features;
  if (named === undefined) return undefined;
  if (config.format_id !== GOLDEN_FORMAT) throw new GenerationError(`the candidate features have a vocabulary for ${GOLDEN_FORMAT} only, not ${config.format_id}`);
  const vocabulary = goldenTeamVocabulary();
  const build = (evidence?: BeliefEvidenceMode): FeatureConfigs => ({
    vocabulary,
    config: loadDamageAwareConfig("v1", vocabulary),
    mechanics: loadMechanicsConfig(config.format_id),
    snapshot: loadRulesSnapshot(engine.commit),
    data_class: "synthetic",
    feature_set: named.feature_set,
    ...(named.feature_set === FEATURE_SET_BELIEF ? { belief: beliefSetup(engine, vocabulary, config.format_id, beliefWorlds) } : {}),
    ...(named.feature_set === FEATURE_SET_3 ? { belief_threats: beliefThreatsSetup(engine, vocabulary, config.format_id, beliefFeatures, evidence) } : {}),
  });
  // Only `probe-damage/3` has a belief mode to choose; the named hash selects it (the informative belief first, else the prior alone, #1309).
  const modes = named.feature_set === FEATURE_SET_3 ? BELIEF_EVIDENCE_MODES : [undefined];
  let configs = build();
  let found = configuredSetHash(engine, configs);
  const tried = [found];
  for (const evidence of modes.slice(1)) {
    if (found === named.feature_set_hash) break;
    configs = build(evidence);
    found = configuredSetHash(engine, configs);
    tried.push(found);
  }
  if (found !== named.feature_set_hash) {
    const others = tried.length > 1 ? `; the other mode computes ${tried.slice(1).join(", ")}` : "";
    throw new GenerationError(`the run config names the candidate feature set ${named.feature_set} ${named.feature_set_hash}, which this checkout computes as ${tried[0]}${others}`);
  }
  return { configs, statement: { feature_set: named.feature_set, feature_set_hash: found } };
}

/** The factory of the composition root (`packages/team-dataset`) that joins the search's records and the opponent model for a search config. */
export type SearchSetupFactory = (engine: Engine, formatId: string, config: SearchConfig) => SearchSetup;

/** The name of the label search config (`config/search/rerank-1-label.v1.json`) every run's `generation.search_labels` names. */
export const LABEL_CONFIG_NAME = "rerank-1-label.v1";

/**
 * The labelling a run config asks for (`generation.search_labels`, ADR-026 erratum E2), set up through the composition root. None when
 * the config has no block (nothing below changes). Refused (`GenerationError`) when there is no factory, or when the config's hash is
 * not the one this checkout computes for the committed label config.
 */
export function labelSetup(engine: Engine, config: TrainingRunConfig, pair: { preview: string; battle: string }, factory?: SearchSetupFactory, features?: FeatureConfigs): LabelSetup | undefined {
  const named = config.generation.search_labels;
  if (named === undefined) return undefined;
  const distill = config.ppo.search_distill;
  if (distill === undefined) throw new GenerationError("generation.search_labels comes with ppo.search_distill");
  if (config.format_id !== GOLDEN_FORMAT) throw new GenerationError(`the search labels have a vocabulary for ${GOLDEN_FORMAT} only, not ${config.format_id}`);
  if (factory === undefined) throw new GenerationError("the run config asks for search labels, and this script has no search setup factory");
  const searchConfig = loadSearchConfig(config.format_id, LABEL_CONFIG_NAME);
  const found = searchConfigHash(searchConfig);
  if (found !== named.search_config_hash) throw new GenerationError(`the run config names the label search config ${named.search_config_hash}, which this checkout computes as ${found}`);
  const temperature = config.generation.play?.temperature_ppm;
  return {
    setup: factory(engine, config.format_id, searchConfig),
    search_config_hash: found,
    share_ppm: named.label_share_ppm,
    start_generation: named.start_generation,
    min_spread_ppm: distill.min_spread_ppm,
    pair,
    manifest: resolveFormat(engine, config.format_id).manifest,
    ...(features === undefined ? {} : { features }),
    ...(temperature === undefined ? {} : { play: { temperature_ppm: temperature } }),
  };
}

/** The opponent sheet source a run reads and what builds its blocks (ADR-027 slice 4). */
export interface RunSheetSetup {
  setup: SheetSetup;
  /** What the policy worker is expected to state at the handshake (and the blocks must be of). */
  statement: { feature_set: string; feature_set_hash: string };
}

/**
 * The opponent sheet source a run config names (`model.encoder.options.opponent_sheet`), set up through the composition root's
 * factory (the bridge does not import the opponent model), or undefined when the config names none. Fails closed: the hash the
 * port computes (belief identity, k values, rounding, builder format) must be the one the config names, so a run never plays or
 * trains on blocks of another source than its checkpoints were made for; a config that names a source with no factory is refused.
 */
export function sheetSetup(engine: Engine, config: TrainingRunConfig, factory?: BeliefFeaturesFactory): RunSheetSetup | undefined {
  const named = config.model.encoder.options?.opponent_sheet;
  if (named === undefined) return undefined;
  if (config.format_id !== GOLDEN_FORMAT) throw new GenerationError(`the opponent sheet has a vocabulary for ${GOLDEN_FORMAT} only, not ${config.format_id}`);
  if (factory === undefined) throw new GenerationError("the run config names an opponent sheet source, and this script has no belief features factory");
  const vocabulary = goldenTeamVocabulary();
  const manifest = resolveFormat(engine, config.format_id).manifest;
  // The named hash selects the mode: the informative belief, else the uninformative one (#1309), each with its own source hash.
  let port: BeliefFeaturesPort | undefined;
  const tried: string[] = [];
  for (const evidence of BELIEF_EVIDENCE_MODES) {
    const candidate = factory({ engine, vocabulary, reader_sha256: readerSha256(), manifest, ...(evidence === undefined ? {} : { evidence }) });
    tried.push(candidate.features_hash);
    if (candidate.features_hash === named.feature_set_hash) {
      port = candidate;
      break;
    }
  }
  if (port === undefined) {
    throw new GenerationError(`the run config names the opponent sheet source ${named.feature_set} ${named.feature_set_hash}, which this checkout computes as ${tried[0]} (informative belief); the other mode computes ${tried.slice(1).join(", ")}`);
  }
  const setup: SheetSetup = { engine, vocabulary, port, scenario_id: checkScenario(engine, loadMechanicsConfig(config.format_id)) };
  return { setup, statement: sheetStatement(setup) };
}

/** The belief a `probe-damage-belief/1` run samples its worlds from: the v2 record's prior, belief config and world count, through the composition root's factory. */
function beliefSetup(engine: Engine, vocabulary: ReturnType<typeof goldenTeamVocabulary>, formatId: string, beliefWorlds: BeliefWorldsFactory | undefined): BeliefFeatureSetup {
  if (beliefWorlds === undefined) throw new GenerationError(`the candidate feature set ${FEATURE_SET_BELIEF} needs the belief port of the composition root (packages/team-dataset)`);
  const config = loadDamageAwareConfigV2(vocabulary);
  const manifest = resolveFormat(engine, formatId, loadRulesSnapshot(engine.commit)).manifest;
  return { port: beliefWorlds({ engine, vocabulary, config, reader_sha256: readerSha256(), manifest }), worlds: config.assumed_set.worlds };
}

/**
 * The belief a `probe-damage/3` run reads its move inclusions from: the committed prior and belief config through the composition
 * root's factory (the same port the opponent sheet reads), cut to `BELIEF_THREATS_PER_FOE` damaging moves a foe.
 */
function beliefThreatsSetup(engine: Engine, vocabulary: ReturnType<typeof goldenTeamVocabulary>, formatId: string, factory: BeliefFeaturesFactory | undefined, evidence?: BeliefEvidenceMode): BeliefThreatsSetup {
  if (factory === undefined) throw new GenerationError(`the candidate feature set ${FEATURE_SET_3} needs the belief features factory of the composition root (packages/team-dataset)`);
  const port = factory({ engine, vocabulary, reader_sha256: readerSha256(), manifest: resolveFormat(engine, formatId, loadRulesSnapshot(engine.commit)).manifest, ...(evidence === undefined ? {} : { evidence }) });
  const inclusions = port.moveInclusions;
  if (inclusions === undefined) throw new GenerationError(`the belief features port has no move inclusions, which ${FEATURE_SET_3} reads`);
  return { port: { identity_hash: port.identity_hash, moveInclusions: (input) => inclusions.call(port, input) }, per_foe: BELIEF_THREATS_PER_FOE };
}

/** The policy id of `damage-aware@v2`, the baseline that plays on sampled belief worlds (ADR-025 section 6). */
export const DAMAGE_AWARE_V2_POLICY_ID = damageAwarePolicyId("v2");

/** The policy id of `damage-aware@v3`, the v1 scorer with the engine's choice-location convention (#1180, #1183). */
export const DAMAGE_AWARE_V3_POLICY_ID = damageAwarePolicyId("v3");

/** The policy id of `damage-aware@v4`, v3 plus the switch and Protect values (#1329). */
export const DAMAGE_AWARE_V4_POLICY_ID = damageAwarePolicyId("v4");

/**
 * The records `damage-aware@v2` reads: its config (which names the usage prior and the belief config by hash) and
 * the belief port the composition root builds from them. The bridge does not import the opponent model, so the
 * root supplies `beliefWorlds`; the actor refuses a port that is not the one the record names, or whose prior is
 * for another format.
 */
export function damageAwareSetupV2(engine: Engine, formatId: string, beliefWorlds: BeliefWorldsFactory): DamageAwareSetup {
  if (formatId !== GOLDEN_FORMAT) throw new GenerationError(`the damage-aware baseline has a vocabulary for ${GOLDEN_FORMAT} only, not ${formatId}`);
  const vocabulary = goldenTeamVocabulary();
  const config = loadDamageAwareConfigV2(vocabulary);
  const snapshot = loadRulesSnapshot(engine.commit);
  const manifest = resolveFormat(engine, formatId, snapshot).manifest;
  return {
    vocabulary,
    config,
    mechanics: loadMechanicsConfig(formatId),
    snapshot,
    worlds: beliefWorlds({ engine, vocabulary, config, reader_sha256: readerSha256(), manifest }),
  };
}

export interface GamePlan {
  /** The game's index within the generation, the sampler's `game`. */
  game: number;
  seed: string;
  /** Indices into the pool. */
  teams: Record<Player, number>;
  /** `self`: the pair plays both seats. Otherwise the pair takes `policy_seat` and a baseline the other. */
  opponent: "self" | BaselineOpponent;
  policy_seat: Player | "both";
  /** Seeds a baseline's stream: random-legal's, or damage-aware's epsilon draws (an unsigned 32-bit integer). */
  baseline_seed: number;
}

/**
 * The generation's match plan: `generation.games` self-play games, then each
 * listed baseline's games. Every number comes from `leagueGameStreams` (league-seed/1: SHA-256
 * of the JCS of the run key, the generation and the game index), so the plan
 * is reproducible from the config alone. A baseline game alternates the policy's seat.
 */
export function planGeneration(config: TrainingRunConfig, generation: number, poolSize: number | PoolShape): GamePlan[] {
  const { games, baseline_games } = config.generation;
  const opponents: GamePlan["opponent"][] = [];
  for (let i = 0; i < games; i++) opponents.push("self");
  for (const baseline of baseline_games) {
    for (let i = 0; i < baseline.games; i++) opponents.push({ actor: baseline.actor, policy_id: baseline.policy_id });
  }
  return planGames(config.seeds.run_key, generation, poolSize, opponents);
}

/**
 * The plan of one run of games from its run key, a generation number and the opponent of each
 * game, in game order (`league-seed/1`; a game's streams depend on those three alone). The pair
 * takes seat p1 of an even game and p2 of an odd one when a baseline is the other seat.
 */
export function planGames(run_key: string, generation: number, pool: number | PoolShape, opponents: readonly GamePlan["opponent"][]): GamePlan[] {
  if (!Number.isInteger(generation) || generation < 1) throw new RangeError("a generation is numbered from 1");
  const shape: PoolShape = typeof pool === "number" ? { size: pool } : pool;
  if (!Number.isInteger(shape.size) || shape.size < 1) throw new RangeError("the team pool is empty");
  const component = shape.private;
  if (component !== undefined && (!Number.isInteger(component.count) || component.count < 1 || component.count >= shape.size)) throw new RangeError("the private component is empty or the synthetic one is");
  const synthetic = component === undefined ? shape.size : shape.size - component.count;
  // The index of a seat's team: within its component, by the existing league-seed/1 draw. Without a private component no further draw is made.
  const pick = (draw: number, game: number, seat: Player): number =>
    component !== undefined && privateTeamSource(run_key, generation, game, seat, component.share_permille) ? synthetic + (draw % component.count) : draw % synthetic;
  return opponents.map((opponent, game) => {
    const streams = leagueGameStreams(run_key, generation, game);
    return {
      game,
      seed: streams.seed,
      teams: { p1: pick(streams.team_draws.p1, game, "p1"), p2: pick(streams.team_draws.p2, game, "p2") },
      opponent,
      policy_seat: opponent === "self" ? "both" : game % 2 === 0 ? "p1" : "p2",
      baseline_seed: streams.baseline_seed,
    };
  });
}

export interface GenerationOptions {
  engine: Engine;
  config: TrainingRunConfig;
  generation: number;
  /** The frozen pair's `behaviour_policy_id` (`policy:` + SHA-256). */
  policy_id: string;
  /** A started worker client; `beginGame` (when it has one) runs before each wave, as the client's overrun rule asks. */
  client: PolicyClient & { beginGame?: () => Promise<void> };
  /**
   * Further workers of the same frozen pair (#957). A game stays on one worker, and
   * every worker answers a query the same way (it is deterministic and the sampling
   * draw is the harness's), so the Episodes do not depend on how many workers there
   * are or which serves a game. Waves grow with the workers so each keeps
   * `n_parallel` games in flight.
   */
  peers?: readonly (PolicyClient & { beginGame?: () => Promise<void> })[];
  /** The games this process plays (default all). The rest come from `others`. */
  shard?: Shard;
  /** The entries of the games of the other shards, played by other game processes. */
  others?: Promise<readonly PlayedEntry[]>;
  pool: TrainingPool;
  /** The damage-aware baseline's records; read from `config/` when a plan game needs them and this is absent (a test seam). */
  damage_aware?: DamageAwareSetup;
  /** `damage-aware@v3`'s records, likewise (#1183). */
  damage_aware_v3?: DamageAwareSetup;
  /** `damage-aware@v4`'s records, likewise (#1329). */
  damage_aware_v4?: DamageAwareSetup;
  /** `damage-aware@v2`'s records, likewise (built from `belief_worlds` when a plan game needs them and this is absent). */
  damage_aware_v2?: DamageAwareSetup;
  /**
   * The candidate features the run's pair reads (`featureSetup`): given to the pair's seats, which attach rows to a query
   * only for a worker that stated the set, and to the training loader, which recomputes them for every example.
   */
  features?: FeatureConfigs;
  /**
   * The opponent sheet the run's pair reads (`sheetSetup`): given to the pair's seats, which attach a block to a query only for a
   * worker that stated the source, and to the training loader, which recomputes it for every example.
   */
  sheet?: SheetSetup;
  /**
   * Called with each decision the pair's own seats play: its query and the answered choice (batched-inference divergence, #1195;
   * `inference-divergence.ts`). Observes only; never changes a byte.
   */
  on_answer?: (query: PolicyQueryInput, choice: string) => void;
  /** Called after each game with the counts of each of the pair's seats' sheet builders (blocks built, refusals by code). */
  on_sheet_counts?: (game: number, counts: readonly SheetCounts[]) => void;
  /** The belief port factory of the composition root (`packages/team-dataset`); without it a plan that names `damage-aware@v2` is refused. */
  belief_worlds?: BeliefWorldsFactory | undefined;
  /**
   * Write the training stream to this file as the examples are made (in chunks; the stream
   * is never one string: a generation of 800 games is over Node's string limit, #998), and
   * keep none of it in the result. Without it the stream is built in memory, which is only
   * for streams under the limit (tests). `writeGeneration` expects it at `<dir>/stream.jsonl`.
   */
  stream_path?: string;
  /** Chunk size and the string-size limit (a test injects a small one). */
  stream_limits?: ChunkLimits;
  /**
   * Write each Episode to `<episodes_dir>/<hash>.json` (flushed) as its game finishes and keep none of
   * them, so a generation's memory does not grow with its games (#1069). The training loader then
   * reads them back one at a time in hash order. The files and the stream are the bytes of the
   * in-memory path; `writeGeneration` expects the directory to be `<dir>/episodes` and writes nothing
   * more there. Needs `stream_path`.
   */
  episodes_dir?: string;
  /**
   * With `episodes_dir` and `features`: a scratch directory the game processes leave the recomputed candidate feature
   * rows of their Episodes in once their games are done (`feature-recompute.ts`, #1185), so the rows of the stream are
   * recomputed by every process in parallel instead of serially by the one that writes the stream. The files are
   * never part of the generation, and the stream's bytes do not depend on whether they exist.
   */
  features_dir?: string | undefined;
  /**
   * The search labelling of the run (`generation.search_labels`, ADR-026 erratum E2, #1290): from `start_generation` the pair's own
   * seats' Battle move decisions are labelled by the search after the game, from the stored Episode. Needs `episodes_dir` and
   * `stream_path`. The labels go to `labels_path` (the generation's `labels.jsonl`) with their manifest in the result; nothing
   * about the PPO stream, the Episodes or any other file changes.
   */
  labels?: LabelSetup;
  /** The scratch directory the game processes leave their label files in; required with `labels` (at or after its start generation). */
  labels_dir?: string | undefined;
  /** Where the label stream is written; `writeGeneration` expects `<dir>/labels.jsonl`. */
  labels_path?: string;
  /**
   * Ends this process's own games (#1100): checked before each wave and after each worker
   * restart, so a process whose generation has failed finishes its current wave and plays no further one and starts no
   * further worker. `playGeneration` aborts it when a game process fails.
   */
  signal?: AbortSignal;
  /** The number of waves (and worker restarts) in progress, for the idle watchdog (#1100). */
  activity?: { busy: number };
  /** How many further game processes are still running; `playGeneration`'s idle watchdog reads it. */
  live_children?: () => number;
  /** How long a generation may have no live game process and no game in flight before it fails (default 60 s; a test injects a short one). */
  idle_grace_ms?: number;
}

export interface GameRow {
  game: number;
  seed: string;
  opponent: string;
  outcome: "played" | "forfeit";
  winner: Player | null;
  turns: number;
  /** Only on a forfeited game: the seat that lost, why, at which request of its channel, and the harness's detail (#912). */
  forfeit?: { loser: Player; cause: ForfeitCause; request_seq: number; detail: string };
  /**
   * Only a game of a pair that reads `probe-damage-belief/1` (#1271): what the belief did for the pair's seats in this game
   * (requests, worlds, legality redraws, worlds dropped, requests that fell back to the v1 assumed sets by cause).
   * Absent for every other run, so those keep their bytes.
   */
  feature_belief?: BeliefRowCounts;
  /**
   * Only a game of a pair that reads `opponent_sheet` (ADR-025 erratum E4 point 5): the blocks its seats' builders built in this game
   * and the refusals by code, summed over the seats. Absent for every other run, so those keep their bytes.
   */
  sheet?: SheetCounts;
}

/** `BeliefFeatureCounts` as a game row holds it: integers, the fallback causes sorted. */
export type BeliefRowCounts = BeliefFeatureCounts;

/** The belief counts of the pair's seats summed, with the fallback causes sorted (the memo's insertion order is not part of a row). */
export function beliefRowCounts(memos: readonly FeatureMemo[]): BeliefRowCounts | undefined {
  const all = memos.flatMap((memo) => (memo.belief === undefined ? [] : [memo.belief]));
  if (all.length === 0) return undefined;
  const fallbacks: Record<string, number> = {};
  for (const counts of all) for (const [cause, count] of Object.entries(counts.fallbacks)) fallbacks[cause] = (fallbacks[cause] ?? 0) + count;
  return {
    decisions: all.reduce((sum, c) => sum + c.decisions, 0),
    worlds_sampled: all.reduce((sum, c) => sum + c.worlds_sampled, 0),
    redraws: all.reduce((sum, c) => sum + c.redraws, 0),
    contradicted_worlds: all.reduce((sum, c) => sum + c.contradicted_worlds, 0),
    fallbacks: Object.fromEntries(Object.entries(fallbacks).sort(([a], [b]) => (a < b ? -1 : 1))),
  };
}

/** The forfeits of a generation by cause, sorted by cause; empty when none. */
export function forfeitCauses(games: readonly GameRow[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const game of games) if (game.forfeit !== undefined) counts[game.forfeit.cause] = (counts[game.forfeit.cause] ?? 0) + 1;
  return Object.fromEntries(Object.entries(counts).sort(([a], [b]) => (a < b ? -1 : 1)));
}

export interface GenerationResult {
  generation: number;
  plan: GamePlan[];
  games: GameRow[];
  /** The Episodes of every seat that was played (the league record), as stored. Empty when they were written to `episodes_dir`. */
  episodes: Episode[];
  /** The directory the Episodes were written to as their games finished (`GenerationOptions.episodes_dir`), else null. */
  episodes_dir: string | null;
  /** Episode hashes, in `episodes` order. */
  episode_hashes: string[];
  /** The stream's bytes, only when it was built in memory (no `stream_path`); else null. */
  stream: string | null;
  /** The file the stream was written to (`GenerationOptions.stream_path`), else null. */
  stream_path: string | null;
  set: TrainingSet;
  /** What the generation played, tied to the run config and the seed derivation (`league-run.json`). */
  run_record: LeagueRunRecord;
  /** SHA-256 of the TrainingSet's JCS bytes: what `CheckpointRecord.training_set_hashes` holds. */
  training_set_hash: string;
  manifest: VersionManifest;
  /** Hashes of the teams the plan drew, sorted and unique. */
  team_hashes: string[];
  /** Only a labelled generation (`GenerationOptions.labels`, at or after its start): the label stream's manifest (null: nothing was labelled, no label file) and counts. */
  labels?: { search_config_hash: string; search_identity_hash: string; set: TrainingSet | null; path: string; totals: LabelTotals };
}

/** A game that could not be played: a harness incident, which fails the generation. */
export class GenerationError extends Error {
  override name = "GenerationError";
}

type StampSets = Parameters<typeof stampTeam>[0];

const rival = (player: Player): Player => (player === "p1" ? "p2" : "p1");

/** What a baseline game reads of the options: the engine and the damage-aware records. */
export type BaselineRecordOptions = Pick<GenerationOptions, "engine" | "damage_aware" | "damage_aware_v2" | "damage_aware_v3" | "damage_aware_v4" | "belief_worlds">;

/**
 * The damage-aware records the config's baselines need and `options` lacks, read once from `config/`
 * (v2 through the composition root's belief port). Refused when the plan names `damage-aware@v2` and there is no port.
 */
export function baselineRecords(options: BaselineRecordOptions, config: Pick<TrainingRunConfig, "format_id" | "generation">): Partial<GenerationOptions> {
  const named = config.generation.baseline_games.filter((baseline) => baseline.actor === "damage-aware");
  const needsV1 = named.some((baseline) => baseline.policy_id !== DAMAGE_AWARE_V2_POLICY_ID && baseline.policy_id !== DAMAGE_AWARE_V3_POLICY_ID && baseline.policy_id !== DAMAGE_AWARE_V4_POLICY_ID);
  const needsV3 = named.some((baseline) => baseline.policy_id === DAMAGE_AWARE_V3_POLICY_ID);
  const needsV4 = named.some((baseline) => baseline.policy_id === DAMAGE_AWARE_V4_POLICY_ID);
  const needsV2 = named.some((baseline) => baseline.policy_id === DAMAGE_AWARE_V2_POLICY_ID);
  if (needsV2 && options.damage_aware_v2 === undefined && options.belief_worlds === undefined) throw new GenerationError(`the plan names ${DAMAGE_AWARE_V2_POLICY_ID}: its script needs the belief port of the composition root (packages/team-dataset)`);
  return {
    ...(needsV1 && options.damage_aware === undefined ? { damage_aware: damageAwareSetup(options.engine, config.format_id) } : {}),
    ...(needsV3 && options.damage_aware_v3 === undefined ? { damage_aware_v3: damageAwareSetup(options.engine, config.format_id, "v3") } : {}),
    ...(needsV4 && options.damage_aware_v4 === undefined ? { damage_aware_v4: damageAwareSetup(options.engine, config.format_id, "v4") } : {}),
    ...(needsV2 && options.damage_aware_v2 === undefined ? { damage_aware_v2: damageAwareSetupV2(options.engine, config.format_id, options.belief_worlds as BeliefWorldsFactory) } : {}),
  };
}

export interface BaselineSeat {
  actor: Actor;
  /** The damage-aware baseline's report once the game is over; absent for the other baselines. */
  report?: () => BaselineGameReport;
}

export function baselineActor(options: BaselineRecordOptions, opponent: BaselineOpponent, plan: Pick<GamePlan, "baseline_seed">, budget: LatencyBudget): BaselineSeat {
  // Scripted baseline actors are E_probe in the generation profile (ADR-028 section 1): the wrapper only reads the clock.
  const guarded = (actor: Actor): Actor => {
    const inner = guardActor(actor, { role: "scripted", latency_budget: budget });
    return profiling() ? (request, context) => timedAsync("e_probe", async () => inner(request, context)) : inner;
  };
  if (opponent.actor !== "damage-aware") return { actor: guarded(opponent.actor === "actor-rule" ? firstLegal : randomLegalActor(plan.baseline_seed)) };
  const setup = opponent.policy_id === DAMAGE_AWARE_V2_POLICY_ID ? options.damage_aware_v2 : opponent.policy_id === DAMAGE_AWARE_V3_POLICY_ID ? options.damage_aware_v3 : opponent.policy_id === DAMAGE_AWARE_V4_POLICY_ID ? options.damage_aware_v4 : options.damage_aware;
  if (setup === undefined) throw new GenerationError("a damage-aware baseline game needs the baseline's records");
  // A fresh actor per game: it keeps the reader's state, the game's memo and counts. The seed is drawn from only when epsilon_ppm is above 0.
  const damage: DamageAwareActor = damageAwareActor({
    ...setup,
    engine: options.engine,
    data_class: "synthetic",
    commit: opponent.policy_id.slice(opponent.policy_id.indexOf("@") + 1),
    seed: plan.baseline_seed,
  });
  return {
    actor: guarded(damage),
    report: () => {
      const { policy_id, identity_hash, counts } = damage.report();
      if (policy_id !== opponent.policy_id) throw new GenerationError(`the config names the baseline ${opponent.policy_id}, which plays as ${policy_id}`);
      return { policy_id, identity_hash, counts };
    },
  };
}

interface PlayedOne extends PlayedGame {
  baseline?: BaselineGameReport;
  /** The pair's seats' rows as they were attached to the queries (with `features_dir`), per seat. */
  play_rows?: Partial<Record<Player, ReadonlyMap<number, AnyCandidateFeatures>>>;
  /** The pair's seats' belief counts, when the pair reads `probe-damage-belief/1`. */
  feature_belief?: BeliefRowCounts;
  /** The pair's seats' sheet block counts, when the pair reads `opponent_sheet`. */
  sheet?: SheetCounts;
  /** The pair's seats' sheet blocks as they were attached to the queries (with `features_dir`), per seat. */
  play_sheets?: Partial<Record<Player, ReadonlyMap<number, OpponentSheet>>>;
}

async function playOne(options: GenerationOptions, plan: GamePlan, client: PolicyClient): Promise<PlayedOne> {
  const { engine, config, policy_id, pool } = options;
  const { generation: settings } = config;
  const budget = { preview_ms: settings.latency_budget_ms.preview_ms, battle_ms: settings.latency_budget_ms.battle_ms };
  const play = settings.play?.temperature_ppm === undefined ? undefined : { temperature_ppm: settings.play.temperature_ppm };
  // Each of the pair's seats keeps its own memo and, with a rows directory, the rows it attaches to its queries (the play-time sidecar, #1271).
  const memos: Partial<Record<Player, FeatureMemo>> = {};
  const rows: Partial<Record<Player, Map<number, AnyCandidateFeatures>>> = {};
  const keepRows = options.features !== undefined && options.features_dir !== undefined && options.episodes_dir !== undefined;
  const builders: ReturnType<typeof createSheetBuilder>[] = [];
  const keepSheets = options.sheet !== undefined && options.features_dir !== undefined && options.episodes_dir !== undefined;
  const sheets: Partial<Record<Player, Map<number, OpponentSheet>>> = {};
  const sheetFor = (player: Player): Pick<PolicySeat, "sheet"> => {
    if (options.sheet === undefined) return {};
    const builder = createSheetBuilder(options.sheet);
    builders.push(builder);
    if (!keepSheets) return { sheet: { builder } };
    const kept = new Map<number, OpponentSheet>();
    sheets[player] = kept;
    return { sheet: { builder, record: (seq: number, block: OpponentSheet) => void kept.set(seq, block) } };
  };
  const observed = options.on_answer === undefined ? {} : { on_answer: options.on_answer };
  const seatFor = (player: Player): PolicySeat => {
    if (options.features === undefined) return { client, policy_id, sampling: "sample", ...(play === undefined ? {} : { play }), ...observed, ...sheetFor(player) };
    const memo = createFeatureMemo();
    memos[player] = memo;
    const kept = new Map<number, AnyCandidateFeatures>();
    rows[player] = kept;
    return { client, policy_id, sampling: "sample", ...(play === undefined ? {} : { play }), features: { configs: options.features, memo, ...(keepRows ? { record: (seq: number, found: AnyCandidateFeatures) => void kept.set(seq, found) } : {}) }, ...observed, ...sheetFor(player) };
  };
  const baseline = plan.opponent === "self" ? undefined : baselineActor(options, plan.opponent, plan, budget);
  const seats: PolicyGameOptions["seats"] =
    baseline === undefined ? { p1: seatFor("p1"), p2: seatFor("p2") } : plan.policy_seat === "p1" ? { p1: seatFor("p1"), p2: baseline.actor } : { p1: baseline.actor, p2: seatFor("p2") };
  const played = await policyGame({
    engine,
    formatId: config.format_id,
    seed: plan.seed,
    game: plan.game,
    teams: { p1: pool.teams[plan.teams.p1] as PokemonSet[], p2: pool.teams[plan.teams.p2] as PokemonSet[] },
    known_synthetic: pool.known_synthetic,
    seats,
    latency_budget: budget,
    maxTurns: settings.max_turns,
    // Training self-play: a seat that does not read the legal set is not judged on a clone (ADR-028 F4).
    skip_unread_judge: true,
  });
  const sheetCounts = builders.length === 0 ? undefined : sumSheetCounts(builders.map((b) => b.counts()));
  options.on_sheet_counts?.(plan.game, builders.map((b) => b.counts()));
  const belief = beliefRowCounts(Object.values(memos));
  return {
    ...played,
    ...(baseline?.report === undefined ? {} : { baseline: baseline.report() }),
    ...(keepRows ? { play_rows: rows } : {}),
    ...(keepSheets ? { play_sheets: sheets } : {}),
    ...(belief === undefined ? {} : { feature_belief: belief }),
    ...(sheetCounts === undefined ? {} : { sheet: sheetCounts }),
  };
}

const idOf = (opponent: GamePlan["opponent"], policy_id: string): string => (opponent === "self" ? policy_id : opponent.policy_id);

/** A slice of the plan for one game process: games `g` with `g mod of = index`. */
export interface Shard {
  index: number;
  of: number;
}

/** One played game: its row and the Episodes of its seats (none for a forfeit). */
export interface PlayedEntry {
  game: number;
  row: GameRow;
  episodes: Episode[];
  /** With `episodes_dir`: the Episodes already written to it, in seat order, instead of `episodes`. */
  stored?: StoredRef[];
  /** Only a game against the damage-aware baseline: its identity hash and this game's counts (ADR-018 §5). */
  baseline?: BaselineGameReport;
}

/** An Episode written to disk: the hash its file is named by and the policy that played the seat. */
export interface StoredRef {
  hash: string;
  behaviour_policy_id: string;
}

function checkShard(shard: Shard): void {
  if (!Number.isInteger(shard.of) || shard.of < 1 || !Number.isInteger(shard.index) || shard.index < 0 || shard.index >= shard.of) {
    throw new RangeError("a shard is an index below a positive count");
  }
}

/**
 * One Episode per seat of a played game (a league collection record: `run_id`, the seed the battle ran
 * with and the other seat's policy id), each classified by both teams. `sides` names each seat's policy.
 */
export async function seatEpisodes(record: BattleRecord, teams: Record<Player, number>, pool: TrainingPool, run_id: string, sides: Record<Player, string>): Promise<Episode[]> {
  const episodes: Episode[] = [];
  for (const player of ["p1", "p2"] as const) {
    episodes.push(
      await episodeFromRecord(record, player, {
        team: stampTeam(pool.teams[teams[player]] as StampSets, pool.known_synthetic),
        opponent_team: stampTeam(pool.teams[teams[rival(player)]] as StampSets, pool.known_synthetic),
        known_synthetic: pool.known_synthetic,
        behaviour_policy_id: sides[player],
        collection: { kind: "league", run_id, seed: record.seed, opponent_policy_id: sides[rival(player)] },
      }),
    );
  }
  countDecisions(episodes.reduce((total, episode) => total + episode.decisions.length, 0)); // the profile's decision count: both seats, baseline seats included
  return episodes;
}

/**
 * Play the games of `shard` (all of them without one) against `options.client` and
 * `options.peers` (started, with the frozen pair loaded) in waves, and write an
 * Episode for every played seat. A wave holds `n_parallel` games for each worker.
 * A game's result depends on its plan entry alone, never on its worker, its wave
 * or the other games, so any split of the plan gives the same entries.
 *
 * Refused: a harness incident in any game (nothing is retried).
 */
export async function playEntries(options: GenerationOptions, shard: Shard = { index: 0, of: 1 }): Promise<PlayedEntry[]> {
  checkShard(shard);
  refusePrivatePool(options.pool);
  const { config, generation, policy_id, client } = options;
  const plan = planGeneration(config, generation, poolShape(options.pool));
  // The damage-aware records are read once, and only when a game of this process needs them.
  const withRecords: GenerationOptions = { ...options, ...baselineRecords(options, config) };
  const { pool } = withRecords;
  const workers = [client, ...(options.peers ?? [])];
  const mine = plan.filter((entry) => entry.game % shard.of === shard.index);
  const entries: PlayedEntry[] = [];
  const waveSize = config.generation.n_parallel * workers.length;
  const activity = options.activity ?? { busy: 0 };
  const stopped = (): void => {
    if (options.signal?.aborted === true) throw new GenerationError("the generation was stopped: this process plays no further game");
  };
  for (let start = 0; start < mine.length; start += waveSize) {
    stopped();
    activity.busy++;
    let played: PlayedOne[];
    const wave = mine.slice(start, start + waveSize);
    try {
      await Promise.all(workers.map((worker) => worker.beginGame?.()));
      stopped(); // a restart that began before the stop must not start games
      // A game stays on one worker, chosen by its place in the wave.
      played = await Promise.all(wave.map((entry, place) => playOne(withRecords, entry, workers[place % workers.length] as PolicyClient)));
    } finally {
      activity.busy--;
    }
    stopped();
    for (const [index, game] of played.entries()) {
      const entry = wave[index] as GamePlan;
      const { outcome, record, baseline } = game;
      const reported = baseline === undefined ? {} : { baseline };
      if (outcome.kind === "incident") throw new GenerationError(`game ${entry.game} (${entry.seed}): ${outcome.incident}: ${outcome.message}`);
      const row = { game: entry.game, seed: entry.seed, opponent: idOf(entry.opponent, policy_id), ...(game.feature_belief === undefined ? {} : { feature_belief: game.feature_belief }), ...(game.sheet === undefined ? {} : { sheet: game.sheet }) };
      if (outcome.kind === "forfeit" || record === null) {
        // A forfeit has no Episode, but it is kept the way its generation keeps them: an empty list of references on the spilled path (#1184).
        const kept = options.episodes_dir === undefined ? {} : { stored: [] };
        if (outcome.kind !== "forfeit") {
          entries.push({ game: entry.game, row: { ...row, outcome: "forfeit", winner: null, turns: 0 }, episodes: [], ...kept, ...reported });
        } else {
          const { loser, cause, request_seq, detail } = outcome;
          entries.push({ game: entry.game, row: { ...row, outcome: "forfeit", winner: outcome.winner, turns: outcome.turns, forfeit: { loser, cause, request_seq, detail } }, episodes: [], ...kept, ...reported });
        }
        continue;
      }
      const sides: Record<Player, string> = {
        p1: entry.opponent !== "self" && entry.policy_seat !== "p1" ? entry.opponent.policy_id : policy_id,
        p2: entry.opponent !== "self" && entry.policy_seat !== "p2" ? entry.opponent.policy_id : policy_id,
      };
      const episodes = await seatEpisodes(record, entry.teams, pool, config.seeds.run_key, sides);
      const base = { game: entry.game, row: { ...row, outcome: "played" as const, winner: record.winner, turns: record.turns }, ...reported };
      if (options.episodes_dir === undefined) {
        entries.push({ ...base, episodes });
      } else {
        // Written now and dropped: the next waves do not carry this game's Episodes (#1069).
        const stored: StoredRef[] = [];
        for (const [seat, episode] of episodes.entries()) {
          const hash = await episodeHash(episode, nodeSha256);
          // Not renamed into place: the directory is the generation's partial one, discarded whole on a restart, and the loader checks each file against its name.
          writeDurable(join(options.episodes_dir, `${hash}.json`), `${jcs(episode as unknown as Json)}\n`);
          stored.push({ hash, behaviour_policy_id: episode.provenance.behaviour_policy_id });
          // The sheet sidecar (#1315), the same way: the blocks the seat built at play time; a gap or a failure leaves the stream writer to recompute.
          const playedSheets = game.play_sheets?.[seat === 0 ? "p1" : "p2"];
          if (playedSheets !== undefined && options.sheet !== undefined && options.features_dir !== undefined && episode.provenance.behaviour_policy_id === policy_id) {
            try {
              await writePlaySheets(options.sheet.port.features_hash, episode, hash, playedSheets, options.features_dir);
            } catch (error) {
              process.stderr.write(`sheet sidecar skipped for game ${entry.game}: ${(error as Error).name}\n`);
            }
          }
          // The play-time sidecar: the pair's own seat leaves the rows it attached to its queries (a gap leaves nothing: the post-game pass builds them).
          const played = game.play_rows?.[seat === 0 ? "p1" : "p2"];
          if (played !== undefined && options.features !== undefined && options.features_dir !== undefined && episode.provenance.behaviour_policy_id === policy_id) {
            // A failure here skips the sidecar (the post-game pass builds the rows) and never fails the generation; it is logged without any content.
            try {
              await writePlayRows({ engine: options.engine, configs: options.features }, episode, hash, played, options.features_dir);
            } catch (error) {
              process.stderr.write(`rows sidecar skipped for game ${entry.game}: ${(error as Error).name}\n`);
            }
          }
        }
        entries.push({ ...base, episodes: [], stored });
      }
    }
  }
  if (options.features_dir !== undefined && options.features !== undefined && options.episodes_dir !== undefined) {
    const dir = options.episodes_dir;
    const own = entries.flatMap((entry) => entry.stored ?? []).filter((ref) => ref.behaviour_policy_id === policy_id);
    await writeFeatureRows({ engine: options.engine, configs: options.features, known_synthetic: pool.known_synthetic }, own.map(({ hash }) => ({ hash, path: join(dir, `${hash}.json`) })), options.features_dir, () => options.signal?.aborted === true);
  }
  if (labelsActive(options) && options.labels_dir !== undefined && options.episodes_dir !== undefined) {
    const dir = options.episodes_dir;
    const own = labelRefs(entries, policy_id, dir);
    // All of this process's workers label (Episodes spread over them); the idle watchdog sees the process busy meanwhile.
    activity.busy++;
    try {
      await writeLabelFiles(labelContext(withRecords, client), own, options.labels_dir, () => options.signal?.aborted === true, workers as LabelContext["client"][]);
    } finally {
      activity.busy--;
    }
  }
  return entries;
}

/** Whether this generation is labelled: the run asks for it and the generation is at or after its start. */
const labelsActive = (options: Pick<GenerationOptions, "labels" | "generation">): boolean => options.labels !== undefined && options.generation >= options.labels.start_generation;

/** The pair's own stored Episodes of played entries, with the game that played each. */
function labelRefs(entries: readonly PlayedEntry[], policy_id: string, dir: string): LabelEpisodeRef[] {
  return entries.flatMap((entry) => (entry.stored ?? []).filter((ref) => ref.behaviour_policy_id === policy_id).map(({ hash }) => ({ hash, path: join(dir, `${hash}.json`), game: entry.game })));
}

function labelContext(options: GenerationOptions, client: PolicyClient): LabelContext & { known_synthetic: TrainingPool["known_synthetic"]; behaviour_policy_id: string } {
  return {
    engine: options.engine,
    labels: options.labels as LabelSetup,
    client: client as LabelContext["client"],
    run_key: options.config.seeds.run_key,
    generation: options.generation,
    known_synthetic: options.pool.known_synthetic,
    behaviour_policy_id: options.policy_id,
  };
}

/** How long a generation may stand with no live game process and no game in flight (#1100). */
const IDLE_GRACE_MS = 60_000;

/**
 * The idle watchdog (#1100): while the other game processes' entries are still awaited, a
 * generation with none of them alive and none of this process's games or worker restarts in
 * progress is not going to finish. `tripped` rejects after `graceMs` of that state; it never
 * resolves. Without a `live` probe (one game process) there is nothing to watch.
 */
function watchIdle(live: (() => number) | undefined, activity: { busy: number }, graceMs: number, awaited: Promise<unknown>): { tripped: Promise<never>; stop: () => void } {
  let timer: ReturnType<typeof setInterval> | undefined;
  let settled = false;
  awaited.then(() => (settled = true), () => (settled = true));
  const tripped = new Promise<never>((_, reject) => {
    if (live === undefined) return;
    let since: number | undefined;
    timer = setInterval(() => {
      const idle = !settled && live() === 0 && activity.busy === 0;
      if (!idle) {
        since = undefined;
        return;
      }
      since ??= Date.now();
      if (Date.now() - since >= graceMs) reject(new GenerationError(`the generation is idle: no game process is running and no game is in flight for ${graceMs} ms, yet the other game processes' games have not been reported`));
    }, Math.max(1, Math.min(1000, Math.floor(graceMs / 4))));
    timer.unref();
  });
  tripped.catch(() => undefined);
  return { tripped, stop: () => clearInterval(timer) };
}

/**
 * Play the generation's plan: `options.shard`'s games here, the rest through
 * `options.others` (entries played by other game processes), then write nothing
 * yet and load the pair's own seats through the training loader. The entries are
 * put in plan order, so the result is the one a single process gives.
 *
 * Refused: a harness incident in any game (the generation fails, nothing is
 * retried); a game missing or played twice; no Episode of the pair at all (every
 * game forfeited). A game that a policy forfeited is not an Episode (it is not
 * training data) and is reported in `games`.
 */
export async function playGeneration(options: GenerationOptions): Promise<GenerationResult> {
  const { engine, config, generation, policy_id, pool } = options;
  refusePrivatePool(pool);
  const plan = planGeneration(config, generation, poolShape(pool));
  const stop = new AbortController();
  const activity = options.activity ?? { busy: 0 };
  const here = playEntries({ ...options, signal: stop.signal, activity }, options.shard);
  const there = options.others ?? Promise.resolve([]);
  here.catch(() => stop.abort()); // a failure of either side ends the other's games at once (#1100)
  there.catch(() => stop.abort());
  let entries: PlayedEntry[];
  const idle = watchIdle(options.live_children, activity, options.idle_grace_ms ?? IDLE_GRACE_MS, there);
  try {
    entries = (await Promise.race([Promise.all([here, there]), idle.tripped])).flat().sort((a, b) => a.game - b.game);
  } catch (error) {
    stop.abort();
    throw error;
  } finally {
    idle.stop();
  }
  if (entries.length !== plan.length || entries.some((entry, index) => entry.game !== index)) {
    throw new GenerationError("the played games are not the plan's games, each once");
  }
  const mergeStart = startTimer();
  const rows = entries.map((entry) => entry.row);
  const spilled = options.episodes_dir !== undefined;
  if (entries.some((entry) => (entry.stored !== undefined) !== spilled || (spilled && entry.episodes.length > 0))) {
    throw new GenerationError("the played games are not all kept the way this generation keeps its Episodes");
  }
  const episodes = entries.flatMap((entry) => entry.episodes);
  const refs: StoredRef[] = entries.flatMap((entry) => entry.stored ?? []);
  const limits = options.stream_limits ?? {};
  let stream: string | null = null;
  let set: TrainingSet;
  let episodeHashes: string[];
  let labelled: GenerationResult["labels"];
  if (labelsActive(options) && options.episodes_dir === undefined) throw new GenerationError("labels need episodes_dir");
  if (labelsActive(options) && options.labels_dir === undefined) throw new GenerationError("labels need labels_dir: the scratch directory the game processes leave their label files in");
  if (options.episodes_dir !== undefined) {
    if (options.stream_path === undefined) throw new GenerationError("episodes_dir needs stream_path");
    const dir = options.episodes_dir;
    const ownRefs = refs.filter((ref) => ref.behaviour_policy_id === policy_id);
    if (ownRefs.length === 0) throw new GenerationError("no Episode of the frozen pair: every game was forfeited");
    const stored: StoredEpisode[] = ownRefs.map(({ hash }) => ({ hash, path: join(dir, `${hash}.json`) }));
    set = await writeStoredTrainingStream({ episodes: stored, evaluation_manifests: [], known_synthetic: pool.known_synthetic, behaviour_policy_id: policy_id, ...(options.features === undefined ? {} : { features: { engine, configs: options.features, ...(options.features_dir === undefined ? {} : { precomputed: readFeatureRows(engine, options.features, options.features_dir) }) } }), ...(options.sheet === undefined ? {} : { sheet: { ...options.sheet, ...(options.features_dir === undefined ? {} : { precomputed: readSheetBlocks(options.sheet.port.features_hash, options.features_dir) }) } }) }, { ...limits, path: options.stream_path });
    episodeHashes = refs.map((ref) => ref.hash);
    if (labelsActive(options)) {
      if (options.labels_path === undefined) throw new GenerationError("labels need labels_path");
      const labels = options.labels as LabelSetup;
      try {
        const written = await writeLabelStream({
          context: labelContext(options, options.client),
          clients: [options.client, ...(options.peers ?? [])] as LabelContext["client"][],
          episodes: labelRefs(entries, policy_id, dir),
          dir: options.labels_dir as string,
          path: options.labels_path,
          limits,
          ...(options.features === undefined ? {} : { features: { engine, configs: options.features, ...(options.features_dir === undefined ? {} : { precomputed: readFeatureRows(engine, options.features, options.features_dir) }) } }),
        });
        labelled = { search_config_hash: labels.search_config_hash, search_identity_hash: labelPolicy(labels, options.client).identity_hash, set: written.set, path: options.labels_path, totals: written.totals };
      } catch (error) {
        if (error instanceof LabelError) throw new GenerationError(error.message);
        throw error;
      }
    }
  } else {
    const own = episodes.filter((episode) => episode.provenance.behaviour_policy_id === policy_id);
    if (own.length === 0) throw new GenerationError("no Episode of the frozen pair: every game was forfeited");
    const input = { episodes: own, evaluation_manifests: [], known_synthetic: pool.known_synthetic, behaviour_policy_id: policy_id, ...(options.features === undefined ? {} : { features: { engine, configs: options.features } }), ...(options.sheet === undefined ? {} : { sheet: options.sheet }) };
    if (options.stream_path === undefined) {
      ({ stream, set } = await loadTrainingSet(input, limits));
    } else {
      set = await writeTrainingStream(input, { ...limits, path: options.stream_path });
    }
    episodeHashes = await Promise.all(episodes.map((episode) => episodeHash(episode, nodeSha256)));
  }
  const drawn = syntheticDrawnHashes(plan, pool);
  const result: GenerationResult = {
    generation,
    plan,
    games: rows,
    episodes,
    episodes_dir: options.episodes_dir ?? null,
    episode_hashes: episodeHashes,
    stream,
    stream_path: options.stream_path ?? null,
    set,
    run_record: await buildLeagueRunRecord({ config, generation, pair_policy_id: policy_id, plan, games: rows, baseline_reports: entries.flatMap((entry) => (entry.baseline === undefined ? [] : [entry.baseline])) }),
    training_set_hash: await nodeSha256(new TextEncoder().encode(jcs(set as unknown as Json))),
    manifest: resolveFormat(engine, config.format_id).manifest,
    team_hashes: drawn,
    ...(labelled === undefined ? {} : { labels: labelled }),
  };
  addSince("merge", mergeStart); // the serial tail after the games: stream, hashing, run record
  return result;
}

/**
 * Write a generation's files under `dir` (created if needed), all canonical JSON
 * except the stream, which is the loader's own bytes:
 * `episodes/<episode hash>.json` (every played seat), `stream.jsonl` (already on disk when the
 * generation was played with a `stream_path`, written in chunks),
 * `training-set.json`, `plan.json`, `league-run.json` (the LeagueRunRecord) and `generation.json` (the per-game rows and
 * the hashes the Python driver records). Nothing in them depends on the clock.
 */
export function writeGeneration(dir: string, result: GenerationResult): void {
  const episodes = join(dir, "episodes");
  mkdirSync(episodes, { recursive: true });
  const put = (name: string, value: unknown): void => writeDurable(join(dir, name), `${jcs(value as Json)}\n`);
  if (result.episodes_dir === null) {
    result.episodes.forEach((episode, index) => put(`episodes/${result.episode_hashes[index]}.json`, episode));
  } else if (result.episodes_dir !== episodes) {
    throw new GenerationError(`the Episodes were written to ${result.episodes_dir}, not to ${episodes}`);
  }
  const streamPath = join(dir, "stream.jsonl");
  if (result.stream_path === null) {
    writeDurable(streamPath, result.stream as string);
  } else if (result.stream_path !== streamPath) {
    throw new GenerationError(`the stream was written to ${result.stream_path}, not to ${streamPath}`);
  }
  put("training-set.json", result.set);
  if (result.labels !== undefined) {
    if (result.labels.path !== join(dir, "labels.jsonl")) throw new GenerationError(`the label stream was written to ${result.labels.path}, not to ${join(dir, "labels.jsonl")}`);
    if (result.labels.set !== null) put("label-set.json", result.labels.set);
  }
  put("plan.json", result.plan);
  put("league-run.json", result.run_record);
  // generation.json last: it is the file that says the rest is whole.
  put("generation.json", {
    schema_version: 1,
    generation: result.generation,
    games: result.games,
    forfeits: result.games.filter((game) => game.outcome === "forfeit").length,
    // Present only when a game was forfeited, so a run without forfeits writes the bytes it always did.
    ...(result.games.some((game) => game.outcome === "forfeit") ? { forfeit_causes: forfeitCauses(result.games) } : {}),
    episode_hashes: [...result.episode_hashes].sort(),
    training_set_hash: result.training_set_hash,
    examples: result.set.examples,
    manifest: result.manifest,
    team_hashes: result.team_hashes,
    // Present only in a labelled generation, so every other run writes the bytes it always did.
    ...(result.labels === undefined ? {} : { search_labels: { search_config_hash: result.labels.search_config_hash, search_identity_hash: result.labels.search_identity_hash, ...labelCounts(result.labels.totals) } }),
  });
  syncDir(episodes);
  syncDir(dir);
}

/** Write a file and flush it to disk before returning. */
export function writeDurable(path: string, data: string): void {
  const descriptor = openSync(path, "w");
  try {
    writeSync(descriptor, data);
    fsyncSync(descriptor);
  } finally {
    closeSync(descriptor);
  }
}

/** Flush a directory's entries to disk. */
export function syncDir(path: string): void {
  const descriptor = openSync(path, "r");
  try {
    fsyncSync(descriptor);
  } finally {
    closeSync(descriptor);
  }
}
