// The population generation's match plan (ADR-021 sections 2, 3 and 5, slice 4): a pure function of the run
// config, the generation and the size of the team pool. Every ladder, self-play and anchor game is a team
// PAIR played twice on one battle seed with the policies' seats swapped (the teams stay on their seats, as
// the progress evaluation's pairs do, ADR-017 erratum E2); a pair has one global index per generation, and
// its battle seed, team draws and baseline stream come from `league-seed/1` at that index. The only other
// draw, the ladder opponent, comes from `population-seed/1` (purpose `opponent-draw`). Teams are drawn by
// index into the pool the caller hands over, which is the run's training pool (dev and training splits):
// nothing here can name a validation or sealed team.
//
// `planGeneration` is untouched: a config without `population` plans as it always did.

import { GenerationError } from "./league-generation.ts";
import { drawPast, type PastPoolFile } from "./past-pool.ts";
import { calibrationGameStreams, leagueGameStreams } from "./league-seed.ts";
import { populationDraw } from "./population-seed.ts";
import type { Player } from "./runner.ts";
import type { TrainingRunConfig } from "./training-run-config.ts";

/** Names the plan layout below; a change to what a plan holds or how it is drawn is a new number. */
export const POPULATION_PLAN_FORMAT = "population-plan/1";

/** A population plan cannot be made for this config (fail closed, fixed words). */
export class PopulationPlanError extends GenerationError {
  override name = "PopulationPlanError";
}

/** Fixed words of the refusals, so a caller and a test can name them. */
export const PAST_POOL_MISSING = "population: a non-zero past_pool_ppm needs the committed past pool of the generation before";
export const EXPLOITER_REFUSAL = "population: exploiter seats are refused until they are built";
export const ODD_ANCHOR_REFUSAL = "population: every baseline's games are team pairs played twice, so each count must be even";

/** What takes a seat: a learning member `m<k>`, or a scripted baseline. */
export type PopulationSeat = { member: string } | { snapshot: string; policy_id: string } | { actor: "actor-rule" | "random-legal" | "damage-aware"; policy_id: string };

export type PopulationGameKind = "ladder" | "self" | "past" | "anchor";

export interface PopulationGamePlan {
  /** The game's index within the generation (2 * pair + half): the sampler's `game`. */
  game: number;
  /** The pair's global index: `league-seed/1`'s game index. */
  pair: number;
  /** 0: the initiating member on p1; 1: the policies' seats swapped (a self-play pair swaps its teams' seats instead). */
  half: 0 | 1;
  /** The pair's battle seed, shared by both of its games. */
  seed: string;
  /** Indices into the training pool. Ladder and anchor pairs keep each team on its seat in both games; a self-play pair swaps the two teams' seats. */
  teams: Record<Player, number>;
  baseline_seed: number;
  kind: PopulationGameKind;
  /** The member whose seat this game was scheduled for. */
  member: string;
  seats: Record<Player, PopulationSeat>;
}

export interface SeatCounts {
  ladder: number;
  self: number;
  past: number;
}

/**
 * The split of a member's `seats_per_member` over the three shares: floors, then one seat to each of the
 * largest remainders (ties in the order ladder, self, past). With one member there is nobody to meet on the
 * ladder, so its seats go to self-play (ADR-021 section 3). Exact whenever the shares divide the seats.
 */
export function seatCounts(population: NonNullable<TrainingRunConfig["population"]>): SeatCounts {
  const { seats_per_member: total, shares } = population;
  const parts = [shares.ladder_ppm, shares.self_play_ppm, shares.past_pool_ppm].map((ppm) => total * ppm);
  const floors = parts.map((part) => Math.floor(part / 1_000_000));
  let left = total - floors.reduce((sum, n) => sum + n, 0);
  const order = parts.map((part, index) => ({ index, rest: part % 1_000_000 })).sort((a, b) => b.rest - a.rest || a.index - b.index);
  for (const { index } of order) {
    if (left === 0) break;
    floors[index] = (floors[index] as number) + 1;
    left -= 1;
  }
  const [ladder, self, past] = floors as [number, number, number];
  return population.members === 1 ? { ladder: 0, self: self + ladder, past } : { ladder, self, past };
}

export const memberLabel = (k: number): string => `m${k}`;

/** What a plan with a past-pool share reads (ADR-021 section 3): the committed pool of the generation before and each member's current pair. */
export interface PastPlanInput {
  pool: PastPoolFile;
  /** member label -> the `policy_id` of the pair it plays this generation (a snapshot equal to it is never drawn). */
  current: Readonly<Record<string, string>>;
}

/**
 * The generation's plan, in game order: member 0's ladder pairs, self-play pairs, past-pool pairs and anchor pairs
 * (one block per baseline, in the config's order), then member 1's, and so on. A past pair meets a snapshot drawn by
 * PFSP (`drawPast`); a member with no snapshot to meet gives its past pairs to self-play. Refused: a past-pool share
 * above 0 with no `past` input, exploiter seats, an odd baseline count, or a config without `population`.
 */
export function planPopulation(config: TrainingRunConfig, generation: number, poolSize: number, past?: PastPlanInput): PopulationGamePlan[] {
  const population = config.population;
  if (population === undefined) throw new PopulationPlanError("population: the config has no population section");
  if (!Number.isInteger(generation) || generation < 1) throw new RangeError("a generation is numbered from 1");
  if (!Number.isInteger(poolSize) || poolSize < 1) throw new RangeError("the team pool is empty");
  if (population.shares.past_pool_ppm !== 0 && past === undefined) throw new PopulationPlanError(PAST_POOL_MISSING);
  if (population.exploiters !== 0) throw new PopulationPlanError(EXPLOITER_REFUSAL);
  if (config.generation.baseline_games.some((baseline) => baseline.games % 2 !== 0)) throw new PopulationPlanError(ODD_ANCHOR_REFUSAL);
  const counts = seatCounts(population);
  const run_key = config.seeds.run_key;
  const plan: PopulationGamePlan[] = [];
  let pair = 0;
  const addPair = (kind: PopulationGameKind, member: string, first: PopulationSeat, second: PopulationSeat): void => {
    const streams = leagueGameStreams(run_key, generation, pair);
    const drawn = { p1: streams.team_draws.p1 % poolSize, p2: streams.team_draws.p2 % poolSize };
    // Self-play: both seats are the same policy, so swapping the policies' seats would repeat the game. The pair
    // swaps the two teams' seats instead, and the two teams are kept apart (two equal teams would repeat the game too).
    // This nudge is the one team index not taken directly from `league-seed/1` (`team_draws`): a Python twin must repeat it.
    const selfTeams = { p1: drawn.p1, p2: drawn.p1 === drawn.p2 && poolSize > 1 ? (drawn.p2 + 1) % poolSize : drawn.p2 };
    for (const half of [0, 1] as const) {
      plan.push({
        game: 2 * pair + half,
        pair,
        half,
        seed: streams.seed,
        teams: kind !== "self" ? drawn : half === 0 ? selfTeams : { p1: selfTeams.p2, p2: selfTeams.p1 },
        baseline_seed: streams.baseline_seed,
        kind,
        member,
        seats: half === 0 || kind === "self" ? { p1: first, p2: second } : { p1: second, p2: first },
      });
    }
    pair += 1;
  };
  for (let k = 0; k < population.members; k++) {
    const member = memberLabel(k);
    const others = Array.from({ length: population.members }, (_, other) => other).filter((other) => other !== k);
    for (let i = 0; i < counts.ladder; i++) {
      const draw = populationDraw(run_key, generation, member, "opponent-draw", i);
      // The 64-bit draw reduced modulo the number of other members: the bias is below 2^-60 for at most 15 others, so it is
      // accepted; any Python twin must use the same rule (draw % len(others), others ascending).
      const opponent = others[Number(draw % BigInt(others.length))] as number;
      addPair("ladder", member, { member }, { member: memberLabel(opponent) });
    }
    const snapshots = past === undefined ? [] : drawPast(config, generation, member, counts.past, past.current[member] ?? "", past.pool);
    for (let i = 0; i < counts.self + (counts.past - snapshots.length); i++) addPair("self", member, { member }, { member });
    for (const snapshot of snapshots) addPair("past", member, { member }, { snapshot: snapshot.id, policy_id: snapshot.policy_id });
    for (const baseline of config.generation.baseline_games) {
      for (let i = 0; i < baseline.games / 2; i++) addPair("anchor", member, { member }, { actor: baseline.actor, policy_id: baseline.policy_id });
    }
  }
  return plan;
}

/** The members that hold a seat in a game (snapshots and baselines are not members). */
function membersOf(entry: PopulationGamePlan): string[] {
  return [...new Set([entry.seats.p1, entry.seats.p2].flatMap((seat) => ("member" in seat ? [seat.member] : [])))].sort(compareMembers);
}

const compareMembers = (a: string, b: string): number => {
  if (!/^m\d+$/.test(a) || !/^m\d+$/.test(b)) throw new PopulationPlanError("population: a member label is m<k>");
  return Number(a.slice(1)) - Number(b.slice(1));
};

/**
 * Which game process plays each game (#1355): a pure function of the plan and the number of processes. The games are
 * ordered by the members that hold their seats (self-play, past and anchor games of `m0`, then its ladder games against
 * `m1`, `m2`, ..., then `m1`'s, and so on) and cut into `of` consecutive runs whose sizes differ by at most one, so a
 * process plays the games of few members and starts workers only for those (one game index modulo the count put every
 * member in every process). Only the choice of process depends on this: a game is a function of its plan entry alone.
 * Returns the process index of each plan entry, in plan order.
 */
export function populationShards(plan: readonly PopulationGamePlan[], of: number): number[] {
  if (!Number.isInteger(of) || of < 1) throw new RangeError("a shard count is a positive integer");
  const keyed = plan.map((entry, at) => ({ at, key: membersOf(entry) }));
  keyed.sort((a, b) => {
    for (let i = 0; i < Math.max(a.key.length, b.key.length); i++) {
      if (a.key[i] === undefined) return -1;
      if (b.key[i] === undefined) return 1;
      const order = compareMembers(a.key[i] as string, b.key[i] as string);
      if (order !== 0) return order;
    }
    return a.at - b.at;
  });
  const owner = new Array<number>(plan.length).fill(0);
  keyed.forEach(({ at }, rank) => {
    owner[at] = Math.floor((rank * of) / plan.length);
  });
  return owner;
}

/** Names the calibration plan layout below. */
export const CALIBRATION_PLAN_FORMAT = "population-calibration/1";

/** The anchors of the ladder (ADR-021 section 4): the scripted baselines a run may rate against, by policy id. */
export const ANCHOR_ACTORS: Readonly<Record<string, BaselineActor>> = { "actor-rule@v1": "actor-rule", "damage-aware@v1": "damage-aware", "random-legal@v1": "random-legal" };
/** The anchor whose strength is 10^9 by definition. */
export const REFERENCE_ANCHOR = "actor-rule@v1";
export const NO_REFERENCE_REFUSAL = "population: baseline_games must list actor-rule@v1, the ladder's reference anchor, with at least 2 games";

type BaselineActor = "actor-rule" | "random-legal" | "damage-aware";

export interface AnchorOpponent {
  actor: BaselineActor;
  policy_id: string;
}

/**
 * The anchors of a run: its configured baselines (at least 2 games each, so a team pair is played) whose policy id is
 * one of `ANCHOR_ACTORS` and whose actor kind matches, sorted by policy id. The reference anchor must be among them.
 */
export function calibrationAnchors(config: TrainingRunConfig): AnchorOpponent[] {
  const found = new Map<string, AnchorOpponent>();
  for (const baseline of config.generation.baseline_games) {
    if (baseline.games < 2 || ANCHOR_ACTORS[baseline.policy_id] !== baseline.actor) continue;
    found.set(baseline.policy_id, { actor: baseline.actor, policy_id: baseline.policy_id });
  }
  if (!found.has(REFERENCE_ANCHOR)) throw new PopulationPlanError(NO_REFERENCE_REFUSAL);
  return [...found.values()].sort((a, b) => (a.policy_id < b.policy_id ? -1 : 1));
}

/** One game of the anchor calibration (generation 0): two anchors, a team pair played twice with the anchors' seats swapped. */
export interface CalibrationGamePlan {
  game: number;
  pair: number;
  half: 0 | 1;
  seed: string;
  teams: Record<Player, number>;
  /** Seeds the stream of anchor `a` (`baseline_seed`) and of anchor `b` (`baseline_seed + 1` modulo 2^32), by role and not by seat. */
  baseline_seed: number;
  a: AnchorOpponent;
  b: AnchorOpponent;
  /** Anchor `a` is on p1 in half 0 and on p2 in half 1; the teams stay on their seats. */
  seats: Record<Player, AnchorOpponent>;
}

/**
 * The generation-0 calibration plan: every pair of anchors (`a` before `b` in policy-id order, in the order of the
 * pairs), `rating.calibration_pairs` team pairs each, pair `i` of the plan drawn by `league-seed/1` at generation 0
 * (`calibrationGameStreams`). Teams are indices into the pool the caller passes (the run's training pool).
 */
export function planCalibration(config: TrainingRunConfig, poolSize: number): CalibrationGamePlan[] {
  const population = config.population;
  if (population === undefined) throw new PopulationPlanError("population: the config has no population section");
  if (!Number.isInteger(poolSize) || poolSize < 1) throw new RangeError("the team pool is empty");
  const anchors = calibrationAnchors(config);
  const plan: CalibrationGamePlan[] = [];
  let pair = 0;
  for (let i = 0; i < anchors.length; i++) {
    for (let j = i + 1; j < anchors.length; j++) {
      const a = anchors[i] as AnchorOpponent;
      const b = anchors[j] as AnchorOpponent;
      for (let n = 0; n < population.rating.calibration_pairs; n++) {
        const streams = calibrationGameStreams(config.seeds.run_key, pair);
        const teams = { p1: streams.team_draws.p1 % poolSize, p2: streams.team_draws.p2 % poolSize };
        for (const half of [0, 1] as const) {
          plan.push({ game: 2 * pair + half, pair, half, seed: streams.seed, teams, baseline_seed: streams.baseline_seed, a, b, seats: half === 0 ? { p1: a, p2: b } : { p1: b, p2: a } });
        }
        pair += 1;
      }
    }
  }
  return plan;
}
