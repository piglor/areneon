// The TrainingRunConfig reader (ADR-010 §5): the run's seeds, generation size,
// model sizes and PPO settings are a reviewed JSON record, validated fail-closed
// and named only by the SHA-256 of its JCS bytes. The schema checks the shape;
// the three rules the contract profile cannot say are checked here, and again
// by the Python reader (`areneon_ml.learner.config`).

import { ajvFor } from "@areneon/schema/ajv";
import trainingRunConfigSchema from "@areneon/schema/contracts/v1/training-run-config.schema.json" with { type: "json" };
import type { TrainingRunConfig } from "@areneon/schema/types/v1/training-run-config.ts";
import { type Json, jcs } from "./canonical-json.ts";
import { nodeSha256 } from "./player-channel.ts";

export type { TrainingRunConfig };

export class TrainingRunConfigError extends Error {
  override name = "TrainingRunConfigError";
}

const validConfig = ajvFor("contract").compile(trainingRunConfigSchema);

/**
 * Read a TrainingRunConfig: the schema, then the rules it cannot say: the
 * attention heads divide the token width, a Team Preview order does not name
 * more positions than a side holds, each scripted baseline appears once, and a warm start names different
 * checkpoints (a checkpoint holds the network of one role; the Battle role alone is allowed), a value warm-up needs a warm start, and so does a KL anchor, a shared trunk excludes a two-role warm start (a Battle-only one is allowed), an explicit play temperature of 1,000,000 is refused (the default is omitted), generation.inference needs 2 <= max_batch <= n_parallel and excludes a population and search labels, a mixed-format run's `sheet_regimes` obey their own rules (checkSheetRegimes), and a population obeys its own rules (opponent shares sum to 10^6, `min_roots` at most `members`, `max_past_per_member` at most `max_size`, `member_ppo` sorted and unique, every explore bound contains its starting value, and no two-role warm start: members need distinct pairs).
 */
export function readTrainingRunConfig(value: unknown): TrainingRunConfig {
  if (!validConfig(value)) throw new TrainingRunConfigError(`invalid TrainingRunConfig: ${JSON.stringify(validConfig.errors)}`);
  const config = structuredClone(value) as unknown as TrainingRunConfig;
  const { model, generation } = config;
  if (model.d_model % model.n_heads !== 0) {
    throw new TrainingRunConfigError(`d_model ${model.d_model} must be a multiple of n_heads ${model.n_heads}`);
  }
  if (model.encoder.max_picks > model.encoder.max_team) {
    throw new TrainingRunConfigError(`max_picks ${model.encoder.max_picks} cannot exceed max_team ${model.encoder.max_team}`);
  }
  const actors = generation.baseline_games.map((baseline) => baseline.actor);
  if (new Set(actors).size !== actors.length) throw new TrainingRunConfigError("a scripted baseline may be listed once");
  if (config.warm_start?.preview !== undefined && config.warm_start.preview === config.warm_start.battle) {
    throw new TrainingRunConfigError("warm_start names one checkpoint for both roles: a checkpoint holds one role's network");
  }
  if (config.warm_start?.preview !== undefined && config.model.options?.shared_trunk !== undefined) {
    throw new TrainingRunConfigError("model.options.shared_trunk cannot be combined with a warm_start that names the preview role: two cloned trunks conflict (use a Battle-only warm start)");
  }
  if (config.ppo.value_warmup !== undefined && config.warm_start === undefined) {
    throw new TrainingRunConfigError("ppo.value_warmup needs warm_start: only a cloned start has an untrained value head to warm up");
  }
  if (config.ppo.anchor !== undefined && config.warm_start === undefined) {
    throw new TrainingRunConfigError("ppo.anchor needs warm_start: the anchor is the warm-start pair");
  }
  checkThreatenedAnchor(config);
  if (generation.play?.temperature_ppm === 1_000_000) {
    throw new TrainingRunConfigError("generation.play.temperature_ppm 1000000 is the default: omit generation.play (one rule, one encoding)");
  }
  if (generation.inference !== undefined) {
    if (generation.inference.max_batch > generation.n_parallel) {
      throw new TrainingRunConfigError(`generation.inference.max_batch ${generation.inference.max_batch} cannot exceed generation.n_parallel ${generation.n_parallel}`);
    }
    if (config.population !== undefined) throw new TrainingRunConfigError("generation.inference cannot be combined with a population (the population run plays batch 1)");
    if (generation.search_labels !== undefined) throw new TrainingRunConfigError("generation.inference cannot be combined with generation.search_labels (labels are training targets and stay batch-1 functions of the Episode)");
  }
  checkSearchDistillation(config);
  checkSheetRegimes(config);
  if (config.population !== undefined && config.warm_start?.preview !== undefined) {
    throw new TrainingRunConfigError("population: a warm_start that names the preview role would give every member the same pair (use a Battle-only warm start; each member's Preview is seeded from its own member-init draw)");
  }
  if (config.population !== undefined) checkPopulation(config);
  return config;
}

/**
 * The rules of `sheet_regimes` the profile cannot say (ADR-027 5a, #1153): the open format is another format than the closed one
 * (that its rules record is open and the pair passes the sheet-variant check is asked of the engine where one is loaded, `resolveRegimes`);
 * the belief closed-game source, the opponent sheet block in a mixed run (the open regime's engine-sheet source and the closed regime's
 * revealed-only source are later slices), a population (its records are version 2) and search labels (they play the closed format) are refused.
 */
function checkSheetRegimes(config: TrainingRunConfig): void {
  const sheet = config.sheet_regimes;
  if (sheet === undefined) return;
  const fail = (message: string): never => {
    throw new TrainingRunConfigError(`sheet_regimes: ${message}`);
  };
  if (sheet.open_format === config.format_id) fail("open_format must be another format than format_id (the closed target format)");
  if (sheet.closed_sheet_source === "belief") fail("closed_sheet_source belief is accepted once the belief source for the closed games exists (ADR-027 slice 6)");
  if (config.population !== undefined) fail("cannot be combined with a population (its records are version 2)");
  if (config.generation.search_labels !== undefined) fail("cannot be combined with generation.search_labels (the search plays the closed format)");
  if (config.model.encoder.options?.opponent_sheet !== undefined) fail("cannot be combined with the opponent_sheet encoder option yet (the open regime's engine-sheet source is a later slice)");
}

/** Feature sets whose rows carry `threats`, `speeds` and `foe_priorities`, which the Threatened flag of `ppo.anchor.threatened` reads (ADR-010 erratum A11.2). */
const THREAT_ROW_FEATURE_SETS: readonly string[] = ["probe-damage/2", "probe-outcome/1"];

/**
 * The rules of `ppo.anchor.threatened` the profile cannot say (ADR-010 erratum A11.2, #1366): `ppo.anchor.kl_micro` may be 0 only together with
 * `threatened` (no anchor outside Threatened decisions), and `threatened` needs a candidate feature set whose rows carry the threat,
 * speed and foe-priority arrays.
 */
function checkThreatenedAnchor(config: TrainingRunConfig): void {
  const anchor = config.ppo.anchor;
  if (anchor === undefined) return;
  if (anchor.threatened === undefined) {
    if (anchor.kl_micro < 1) throw new TrainingRunConfigError("ppo.anchor.kl_micro 0 means no anchor outside Threatened decisions: it needs ppo.anchor.threatened");
    return;
  }
  const features = config.model.encoder.options?.candidate_features?.feature_set;
  if (features === undefined || !THREAT_ROW_FEATURE_SETS.includes(features)) {
    throw new TrainingRunConfigError("ppo.anchor.threatened reads the threats, speeds and foe_priorities rows: it needs candidate_features of probe-damage/2 or probe-outcome/1");
  }
}

/** Feature sets whose value head reads no candidate row (ADR-026 E2 decision 7): `/2` and `/3` pool the rows into the tokens. */
const VALUE_FREE_FEATURE_SETS: readonly string[] = ["probe-damage/1", "probe-damage-belief/1"];

/**
 * The rules of `generation.search_labels` and `ppo.search_distill` the profile cannot say (ADR-026 E2 decision 5, #1291):
 * both blocks or neither; a warm start that names the Battle network and a value warm-up, with `start_generation` after the warm-up;
 * no opponent sheet and no candidate feature set whose value reads candidate rows; a team dataset.
 */
function checkSearchDistillation(config: TrainingRunConfig): void {
  const labels = config.generation.search_labels;
  const distill = config.ppo.search_distill;
  if ((labels === undefined) !== (distill === undefined)) {
    throw new TrainingRunConfigError("generation.search_labels and ppo.search_distill come together or not at all");
  }
  if (labels === undefined || distill === undefined) return;
  if (config.population !== undefined) throw new TrainingRunConfigError("search distillation cannot be combined with a population (no per-member labels)");
  if (config.warm_start === undefined) throw new TrainingRunConfigError("search distillation needs warm_start (a Battle network cloned from the teacher)");
  const warmup = config.ppo.value_warmup;
  if (warmup === undefined) throw new TrainingRunConfigError("search distillation needs ppo.value_warmup: a cloned value head is untrained until the warm-up ends");
  if (labels.start_generation < warmup.generations + 1) {
    throw new TrainingRunConfigError("generation.search_labels.start_generation must be at least ppo.value_warmup.generations + 1");
  }
  const options = config.model.encoder.options;
  if (options?.opponent_sheet !== undefined) throw new TrainingRunConfigError("search distillation cannot be combined with the opponent_sheet encoder option");
  const features = options?.candidate_features?.feature_set;
  if (features !== undefined && !VALUE_FREE_FEATURE_SETS.includes(features)) {
    throw new TrainingRunConfigError(`search distillation cannot be combined with candidate feature set ${features}: its value reads candidate rows`);
  }
  if (config.team_dataset === undefined) throw new TrainingRunConfigError("search distillation needs a team_dataset: the belief prior is generator-matched");
}

const START_FIELDS = ["learning_rate_micro", "entropy_coef_micro", "epochs", "clip_micro", "target_kl_micro"] as const;
const SHARE_FIELDS = ["ladder_ppm", "self_play_ppm", "past_pool_ppm"] as const;

/** The population rules the profile cannot say (ADR-021 sections 2, 3, 6): sums, orders and containment. */
function checkPopulation(config: TrainingRunConfig): void {
  const population = config.population;
  if (population === undefined) return;
  const fail = (message: string): never => {
    throw new TrainingRunConfigError(`population: ${message}`);
  };
  const { shares, past_pool, selection, explore } = population;
  if (shares.ladder_ppm + shares.self_play_ppm + shares.past_pool_ppm !== 1_000_000) fail("the opponent shares must sum to 1000000");
  if (selection.min_roots > population.members) fail("selection.min_roots cannot exceed members");
  if (past_pool.max_past_per_member > past_pool.max_size) fail("past_pool.max_past_per_member cannot exceed past_pool.max_size");
  const overrides = population.member_ppo ?? [];
  overrides.forEach((entry, i) => {
    if (entry.member >= population.members) fail("member_ppo names a member at or above members");
    if (i > 0 && entry.member <= (overrides[i - 1]?.member ?? -1)) fail("member_ppo must be sorted by member and unique");
    if (entry.target_kl_micro !== undefined && config.ppo.target_kl_micro === undefined) fail("member_ppo target_kl_micro needs ppo.target_kl_micro");
  });
  const bounds = explore.bounds;
  if (bounds.target_kl_micro !== undefined && config.ppo.target_kl_micro === undefined) fail("explore.bounds.target_kl_micro needs ppo.target_kl_micro");
  const within = (name: string, value: number | undefined, bound: { min: number; max: number } | undefined): void => {
    if (value === undefined || bound === undefined) return;
    if (bound.min > value || value > bound.max) fail(`explore.bounds.${name} must contain its starting value`);
  };
  for (let member = 0; member < population.members; member += 1) {
    const override = overrides.find((entry) => entry.member === member);
    for (const field of START_FIELDS) within(field, override?.[field] ?? config.ppo[field], bounds[field]);
  }
  for (const field of SHARE_FIELDS) within(field, shares[field], bounds[field]);
}

/** The hash `CheckpointRecord.run_config_hash` holds: SHA-256 of the record's JCS bytes. */
export const runConfigHash = (config: TrainingRunConfig): Promise<string> =>
  nodeSha256(new TextEncoder().encode(jcs(config as unknown as Json)));
