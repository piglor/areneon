// The Team Preview payoff table (ADR-029 section 3, findings B1, N3 and the N7 clamp; Figure 1). One decision's table of
// U(s, L): our strategies s against the opponent's ordered lead pairs L, each entry the mean over the worlds of the policy's
// value at our first move request of a hypothetical battle (or after a rollout). A library: no actor, no policy id, no solver
// call; `matrix` is what the integer solver takes.
//
// Everything hypothetical stays behind the brands of `hypothetical-preview.ts`: this module reaches a worker only through
// `evaluateHypothetical`, sees no channel text, and takes the worlds as started hypothetical battles (the caller maps
// `sampleTeam`'s draw to engine sets and starts them). Integers only.
//
// A STRATEGY is (ordered lead pair, unordered back set): the 360 orders of four of six collapse to 180. Its probability under
// a Preview network is the sum over its two orders and it is played as the more probable order (ties in canonical order).

import { jcs } from "./canonical-json.ts";
import { decodeRequest } from "./request.ts";
import { evaluateHypothetical, type HypotheticalCandidate, type HypotheticalObservation, type HypotheticalPreview, hypotheticalRequest } from "./hypothetical-preview.ts";
import type { PolicyClient, EvaluationClient } from "./policy-worker.ts";
import { sha256Hex } from "./rules.ts";
import { outcomeScore } from "./search-leaf.ts";

/** Bumped when an entry is computed differently for the same input (the leaf, the clamp, the back pair rule); part of the solver's identity. */
export const PAYOFF_LEAF_FORMAT = 1;

/** The purpose of the opponent's back-pair draw under the rollout leaf (its own, apart from every other stream). */
export const BACK_DRAW_PURPOSE = "preview-opp-back/1";

const PPM = 1_000_000;

export class PayoffError extends Error {
  override name = "PayoffError";
}

export type OwnRule = "prior-top-k" | "distinct-leads";
export type PayoffLeaf = { kind: "first-request" } | { kind: "rollout"; turns: number };

/** Declared values, no defaults (S0-P freezes them). */
export interface PayoffConfig {
  k_own: number;
  k_opp: number;
  own_rule: OwnRule;
  leaf: PayoffLeaf;
  /**
   * Also report, per column, how many used worlds the opponent's Preview network ALONE would play that ordered lead pair in under argmax
   * (the first highest candidate, as `argmax` of a play mode does): `opp[].argmax_worlds`. Absent keeps the table's shape.
   */
  argmax_columns?: boolean;
}

/** One strategy: its ordered leads, its back set (sorted), the order it is played as and its probability (ppm units of the network's prior). */
export interface Strategy {
  readonly key: string;
  readonly leads: readonly [number, number];
  readonly back: readonly number[];
  /** The `team ...` choice it is played as: the more probable of its orders, the canonically first on a tie. */
  readonly order: string;
  readonly ppm: number;
  /** Position of its first order in the canonical candidate list. */
  readonly rank: number;
}

export interface PayoffWorld {
  /** The world's index in the decision (the stream index of its `sampleTeam`). */
  index: number;
  preview: HypotheticalPreview;
}

export type PayoffClient = EvaluationClient & Pick<PolicyClient, "featureSet">;

export interface PayoffInput {
  config: PayoffConfig;
  worlds: readonly PayoffWorld[];
  client: PayoffClient;
  /** The battle id the worker client keys its one-query-at-a-time rule by. */
  battle: string;
  policy_id: string;
  /** Our REAL Team Preview request (the body of the `|request|` line): every world must give our side exactly this request (fail closed). */
  own_request: string;
  /** A test seam: the rollout's step cap (default `MAX_ROLLOUT_STEPS`). */
  max_rollout_steps?: number;
  /** The harness's per-decision stream (32 bytes); it reaches only the back-pair draw. */
  stream: Uint8Array;
}

export interface PayoffCounts {
  worlds_asked: number;
  /** Worlds whose opponent observation was evaluated. */
  worlds_used: number;
  /** Evaluate queries on Team Preview observations (ours once, the opponent's once per world). */
  preview_queries: number;
  /** Entries attempted: (strategy, lead pair, world). */
  entry_attempts: number;
  /** Entries that produced a value. */
  entries: number;
  /** Evaluate queries on first-request observations (the leaf). */
  value_queries: number;
  /** Evaluate queries that chose a seat's order in a rollout. */
  rollout_choices: number;
  /** Back pairs drawn (rollout leaf only). */
  back_draws: number;
  clamped: number;
  clamped_low: number;
  clamped_high: number;
  /** Entries scored by the result of a game that ended first. */
  game_end_entries: number;
  /** Entries the engine refused. */
  engine_refusals: number;
  /** Entries whose request was not a move request. */
  not_move_request: number;
  /** Worlds dropped for every cell because one of their entries was refused or was not a move request (ADR-029 section 3). */
  worlds_dropped: number;
  /** Rollouts stopped by the step cap and valued at the request they reached. */
  rollout_capped: number;
  /** Table cells no world produced. */
  missing_entries: number;
}

export interface PayoffTable {
  readonly leaf: PayoffLeaf;
  /** Our strategies, the rows. */
  readonly own: readonly Strategy[];
  /** The opponent's ordered lead pairs, the columns, with their mean mass (the network's prior summed by lead pair, floor-averaged over worlds). */
  readonly opp: readonly { readonly leads: readonly [number, number]; readonly mean_ppm: number; readonly argmax_worlds?: number }[];
  /** U(s, L) in ppm, floor mean over the worlds that produced the entry; null where none did. */
  readonly entries: readonly (readonly (number | null)[])[];
  /** How many worlds produced each entry. */
  readonly contributors: readonly (readonly number[])[];
  /** Every entry present: the matrix is the integer solver's input (the opponent's payoff is 10^6 minus it). */
  readonly complete: boolean;
  readonly matrix: readonly (readonly number[])[] | null;
  readonly counts: PayoffCounts;
}

const emptyCounts = (): PayoffCounts => ({
  worlds_asked: 0,
  worlds_used: 0,
  preview_queries: 0,
  entry_attempts: 0,
  entries: 0,
  value_queries: 0,
  rollout_choices: 0,
  back_draws: 0,
  clamped: 0,
  clamped_low: 0,
  clamped_high: 0,
  game_end_entries: 0,
  engine_refusals: 0,
  not_move_request: 0,
  worlds_dropped: 0,
  rollout_capped: 0,
  missing_entries: 0,
});

/**
 * The strategies of a Team Preview request: the candidates' `team ...` orders grouped by (ordered lead pair, back set), in
 * canonical order of first appearance. `prior` is the network's prior per candidate, in the candidates' order.
 */
export function strategiesOf(candidates: readonly Pick<HypotheticalCandidate, "order">[], prior: readonly number[]): Strategy[] {
  if (prior.length !== candidates.length) throw new PayoffError("the prior does not match the candidates");
  const groups = new Map<string, { leads: [number, number]; back: number[]; order: string; best: number; ppm: number; rank: number }>();
  candidates.forEach((candidate, rank) => {
    const match = /^team (\d+)$/.exec(candidate.order);
    if (match === null || (match[1] as string).length < 3) throw new PayoffError("not a Team Preview order of at least three positions");
    const positions = [...(match[1] as string)].map(Number);
    if (new Set(positions).size !== positions.length) throw new PayoffError("a Team Preview order repeats a position");
    const leads: [number, number] = [positions[0] as number, positions[1] as number];
    const back = positions.slice(2).sort((a, b) => a - b);
    const key = `${leads.join("")}|${back.join("")}`;
    const weight = prior[rank] as number;
    const known = groups.get(key);
    if (known === undefined) groups.set(key, { leads, back, order: candidate.order, best: weight, ppm: weight, rank });
    else {
      known.ppm += weight;
      if (weight > known.best) {
        known.best = weight;
        known.order = candidate.order;
      }
    }
  });
  return [...groups.entries()].map(([key, g]) => ({ key, leads: g.leads, back: g.back, order: g.order, ppm: g.ppm, rank: g.rank })).sort((a, b) => a.rank - b.rank);
}

const byMass = (a: { ppm: number; rank: number }, b: { ppm: number; rank: number }): number => b.ppm - a.ppm || a.rank - b.rank;

/** Our strategies to consider: the top `k` by probability (ties canonical), by the declared rule. */
export function selectOwn(strategies: readonly Strategy[], k: number, rule: OwnRule): Strategy[] {
  const ordered = [...strategies].sort(byMass);
  if (rule === "prior-top-k") return ordered.slice(0, k);
  const seen = new Set<string>();
  const chosen: Strategy[] = [];
  for (const strategy of ordered) {
    const lead = strategy.leads.join("");
    if (seen.has(lead)) continue;
    seen.add(lead);
    chosen.push(strategy);
    if (chosen.length === k) break;
  }
  return chosen;
}

const leadKey = (leads: readonly number[]): string => leads.join("");

/** The prior summed by ordered lead pair, keyed in order of first appearance. */
function leadMass(strategies: readonly Strategy[]): Map<string, { leads: readonly [number, number]; ppm: number; rank: number }> {
  const masses = new Map<string, { leads: readonly [number, number]; ppm: number; rank: number }>();
  for (const strategy of strategies) {
    const key = leadKey(strategy.leads);
    const known = masses.get(key);
    if (known === undefined) masses.set(key, { leads: strategy.leads, ppm: strategy.ppm, rank: strategy.rank });
    else known.ppm += strategy.ppm;
  }
  return masses;
}

/** A back pair's draw: an integer in [0, total) from SHA-256 of the JCS of {purpose, stream, world, leads}: its own stream per world and lead pair. */
export function backDraw(stream: Uint8Array, world: number, leads: readonly number[], total: number): number {
  const hash = sha256Hex(jcs({ leads: [...leads], purpose: BACK_DRAW_PURPOSE, stream: Buffer.from(stream).toString("hex"), world }));
  return Number((BigInt(`0x${hash.slice(0, 16)}`) * BigInt(total)) >> 64n);
}

function checkConfig(config: PayoffConfig): void {
  const whole = (n: number): boolean => Number.isInteger(n) && n >= 1;
  if (!whole(config.k_own) || !whole(config.k_opp)) throw new PayoffError("k_own and k_opp must be integers of at least 1");
  if (config.own_rule !== "prior-top-k" && config.own_rule !== "distinct-leads") throw new PayoffError("unknown own rule");
  if (config.leaf.kind === "rollout" && !whole(config.leaf.turns)) throw new PayoffError("a rollout leaf plays at least one turn");
  if (config.leaf.kind !== "rollout" && config.leaf.kind !== "first-request") throw new PayoffError("unknown leaf");
}

/** The first highest non-pending candidate by prior (the argmax of the pair's Battle network), else the first highest of all. */
function argmax(candidates: readonly HypotheticalCandidate[], prior: readonly number[]): string {
  const pool = candidates.map((candidate, i) => ({ candidate, weight: prior[i] as number })).filter((c) => !c.candidate.pending);
  const among = pool.length > 0 ? pool : candidates.map((candidate, i) => ({ candidate, weight: prior[i] as number }));
  let best = among[0];
  if (best === undefined) throw new PayoffError("a request with no candidates");
  for (const item of among) if (item.weight > best.weight) best = item;
  return best.candidate.order;
}

/** Build the payoff table of one Team Preview decision. */
export async function buildPayoffTable(input: PayoffInput): Promise<PayoffTable> {
  const { config, worlds, client, battle, policy_id, stream, own_request } = input;
  const max_steps = input.max_rollout_steps;
  checkConfig(config);
  if (worlds.length === 0) throw new PayoffError("no worlds");
  const counts = emptyCounts();
  counts.worlds_asked = worlds.length;

  const first = (worlds[0] as PayoffWorld).preview;
  // Our side must see the real Team Preview request in every world (fail closed): the caller starts each world with `own_request`, which
  // fixes our sets' genders (the engine would draw a missing one from the world's seed).
  const realHash = decodeRequest(own_request).request_hash;
  for (const { preview } of worlds) {
    if (hypotheticalRequest(preview.own_channel).request_hash !== realHash) throw new PayoffError("a world gives our side another Team Preview request than the real one");
  }
  // Our strategies, from our Preview network on our own Team Preview observation (the same in every world).
  counts.preview_queries += 1;
  const ours = await evaluateHypothetical(client, battle, await first.observation(first.own_channel), policy_id);
  const own = selectOwn(strategiesOf(ours.candidates, ours.evaluation.prior_ppm), config.k_own, config.own_rule);

  // The opponent's ordered lead pairs: our Preview network on each world's hypothetical opponent observation, summed by lead pair.
  const foeStrategies = new Map<number, Strategy[]>();
  const leadSums = new Map<string, { leads: readonly [number, number]; sum: number; rank: number }>();
  const argmaxWorlds = new Map<string, number>();
  for (const { index, preview } of worlds) {
    counts.preview_queries += 1;
    const evaluated = await evaluateHypothetical(client, battle, await preview.observation(preview.foe_channel), policy_id);
    const strategies = strategiesOf(evaluated.candidates, evaluated.evaluation.prior_ppm);
    foeStrategies.set(index, strategies);
    counts.worlds_used += 1;
    if (config.argmax_columns === true) {
      const top = /^team (\d+)$/.exec(argmax(evaluated.candidates, evaluated.evaluation.prior_ppm));
      if (top === null) throw new PayoffError("not a Team Preview order");
      const lead = (top[1] as string).slice(0, 2);
      argmaxWorlds.set(lead, (argmaxWorlds.get(lead) ?? 0) + 1);
    }
    for (const [key, mass] of leadMass(strategies)) {
      const known = leadSums.get(key);
      if (known === undefined) leadSums.set(key, { leads: mass.leads, sum: mass.ppm, rank: mass.rank });
      else known.sum += mass.ppm;
    }
  }
  const opp = [...leadSums.values()]
    .sort((a, b) => b.sum - a.sum || a.rank - b.rank)
    .slice(0, config.k_opp)
    .map(({ leads, sum }) => ({ leads, mean_ppm: Math.floor(sum / counts.worlds_used), ...(config.argmax_columns === true ? { argmax_worlds: argmaxWorlds.get(leadKey(leads)) ?? 0 } : {}) }));

  /** The opponent's order for lead pair L in a world: the most probable back set given L (first-request leaf) or one drawn from the conditional (rollout leaf). */
  const foeOrder = (world: number, leads: readonly number[]): string | null => {
    const lead = leadKey(leads);
    const conditional = (foeStrategies.get(world) as Strategy[]).filter((s) => leadKey(s.leads) === lead);
    if (conditional.length === 0) return null;
    if (config.leaf.kind === "first-request") return [...conditional].sort(byMass)[0]?.order ?? null;
    const total = conditional.reduce((sum, s) => sum + s.ppm, 0);
    counts.back_draws += 1;
    if (total <= 0) return (conditional[0] as Strategy).order;
    const target = backDraw(stream, world, leads, total);
    let cumulative = 0;
    for (const s of conditional) {
      cumulative += s.ppm;
      if (cumulative > target) return s.order;
    }
    return (conditional[conditional.length - 1] as Strategy).order;
  };

  const sums = own.map(() => opp.map(() => 0));
  const contributors = own.map(() => opp.map(() => 0));
  const clamp = (value: number): number => {
    if (value < 0) {
      counts.clamped += 1;
      counts.clamped_low += 1;
      return 0;
    }
    if (value > PPM) {
      counts.clamped += 1;
      counts.clamped_high += 1;
      return PPM;
    }
    return value;
  };
  const choose = async (observation: HypotheticalObservation): Promise<string> => {
    counts.rollout_choices += 1;
    const answer = await evaluateHypothetical(client, battle, observation, policy_id);
    return argmax(answer.candidates, answer.evaluation.prior_ppm);
  };

  for (const { index, preview } of worlds) {
    // Every cell of a world or none (ADR-029 section 3): a refused or non-move entry drops the world, so all cells average the same worlds.
    const values: number[][] = own.map(() => opp.map(() => 0));
    let dropped = false;
    for (let r = 0; r < own.length && !dropped; r += 1) {
      for (let c = 0; c < opp.length && !dropped; c += 1) {
        const strategy = own[r] as Strategy;
        const leads = (opp[c] as { leads: readonly number[] }).leads;
        const order = foeOrder(index, leads);
        if (order === null) {
          dropped = true;
          break;
        }
        counts.entry_attempts += 1;
        const turn = config.leaf.kind === "rollout" ? await preview.rollout(strategy.order, order, config.leaf.turns, choose, max_steps) : preview.play(strategy.order, order);
        if (turn.kind === "refused") {
          counts.engine_refusals += 1;
          dropped = true;
        } else if (turn.kind === "ended") {
          counts.game_end_entries += 1;
          (values[r] as number[])[c] = outcomeScore(turn.outcome);
        } else if (turn.request.phase !== "move" && turn.capped !== true) {
          counts.not_move_request += 1;
          dropped = true;
        } else {
          if (turn.capped === true) counts.rollout_capped += 1;
          counts.value_queries += 1;
          const answer = await evaluateHypothetical(client, battle, await preview.observation(turn.channel), policy_id);
          (values[r] as number[])[c] = clamp(answer.evaluation.value_ppm);
        }
      }
    }
    if (dropped) {
      counts.worlds_dropped += 1;
      continue;
    }
    counts.entries += own.length * opp.length;
    values.forEach((row, r) =>
      row.forEach((value, c) => {
        (sums[r] as number[])[c] = ((sums[r] as number[])[c] as number) + value;
        (contributors[r] as number[])[c] = ((contributors[r] as number[])[c] as number) + 1;
      }),
    );
  }

  const entries = sums.map((row, r) => row.map((sum, c) => ((contributors[r] as number[])[c] === 0 ? null : Math.floor(sum / ((contributors[r] as number[])[c] as number)))));
  counts.missing_entries = entries.flat().filter((entry) => entry === null).length;
  const complete = counts.missing_entries === 0;
  return { leaf: config.leaf, own, opp, entries, contributors, complete, matrix: complete ? (entries as number[][]) : null, counts };
}
