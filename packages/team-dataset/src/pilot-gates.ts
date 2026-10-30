// The pilot gates of a mixture dataset (ADR-020 section 6) and the report that carries them (section 8, E-13.5, E-13.6).
// Everything here is an integer: counts, thousandths, and bits. Nothing writes a game rule; the thresholds are the
// config's `proposal.gates` and the numbers come from the pilot, the plan, the vocabulary and the role-set table.
//
//   P1  mapping: the share of the source's ids that resolve in the vocabulary (component R reads the table; 1000 is required)
//   P2  acceptance: the share of a component's pilot draws the validator accepts
//   P3  concentration: an integer upper bound on the probability of the single most likely team under the component
//   P4  coverage: the plan's strata against the uniform config's plan at the same d_max
//
// P3 is a bound on the generator, not a proof that no real team is ever drawn. It treats a team as the SET of its
// members (the order the generator draws them in is summed over: size! orderings of the members), ignores the Stat Point
// draw (a factor of at most 1, so the bound stays valid and is looser), counts a set's moves as an unordered subset
// (valid for ordered identity too) and bounds the Item Clause by removing, from the item total, the largest weights an
// earlier member could have taken. The bound is the exact rational below, reported as the integer b with bound <= 2^-b.

import type { RoleSetTable, TeamGeneratorConfig, TeamVocabulary } from "@areneon/showdown-bridge";
import type { QuotaPlan, Split } from "./plan.ts";
import type { Pilot } from "./pilot.ts";
import { type UsageSource, usageMembers, type UsageMember } from "./usage-proposal.ts";

export type Gates = NonNullable<NonNullable<TeamGeneratorConfig["proposal"]>["gates"]>;

/** A non-negative rational. */
interface Fraction {
  num: bigint;
  den: bigint;
}

const frac = (num: bigint, den: bigint): Fraction => ({ num, den });
const mul = (a: Fraction, b: Fraction): Fraction => frac(a.num * b.num, a.den * b.den);
const add = (a: Fraction, b: Fraction): Fraction => frac(a.num * b.den + b.num * a.den, a.den * b.den);

function binomial(n: number, k: number): bigint {
  let result = 1n;
  const j = Math.min(k, n - k);
  for (let i = 1; i <= j; i++) result = (result * BigInt(n - j + i)) / BigInt(i);
  return result;
}

/** The largest b >= 0 with `value <= 2^-b`, that is `value * 2^b <= 1` (0 when the value exceeds 1/2... or 1). */
export function bitsOf(value: Fraction): number {
  if (value.num <= 0n) throw new RangeError("a probability bound of zero has no bit count");
  if (value.num > value.den) return 0;
  let b = Math.max(0, value.den.toString(2).length - value.num.toString(2).length - 1);
  while ((value.num << BigInt(b)) > value.den) b--;
  while ((value.num << BigInt(b + 1)) <= value.den) b++;
  return b;
}

export type BoundVocabulary = Pick<TeamVocabulary, "roster" | "items" | "natures"> & { rules: Pick<TeamVocabulary["rules"], "max_team_size" | "item_clause"> };
export type BoundConfig = Pick<TeamGeneratorConfig, "item_weights" | "default_item_weight" | "gender_policy" | "moves_per_set">;

/** The factors every member shares: the nature, the gender and a bound on one item draw. */
function commonFactor(vocabulary: BoundVocabulary, config: BoundConfig): { free: Fraction; itemFree: Fraction } {
  const override = new Map(config.item_weights.map((w) => [w.item, w.weight]));
  const weights = vocabulary.items.map((item) => override.get(item) ?? config.default_item_weight).sort((a, b) => b - a);
  const total = weights.reduce((a, b) => a + b, 0);
  const limit = vocabulary.rules.item_clause;
  // At most this many items can be out of reach when a member draws: each needs `limit` earlier holders.
  const removable = limit === undefined || limit < 1 ? 0 : Math.floor((vocabulary.rules.max_team_size - 1) / limit);
  const removed = weights.slice(0, removable).reduce((a, b) => a + b, 0);
  const max = weights[0] ?? 0;
  const left = total - removed;
  const itemFree = max === 0 || left <= 0 ? frac(1n, 1n) : left < max ? frac(1n, 1n) : frac(BigInt(max), BigInt(left));
  const gender = config.gender_policy === "random" ? 2n : 1n;
  return { free: frac(1n, BigInt(vocabulary.natures.length) * gender), itemFree };
}

/** The sets the generator can draw for one member of the uniform component: 1 / (abilities x moves subsets). */
const setFactor = (abilities: number, pool: number, k: number): Fraction => frac(1n, BigInt(abilities) * binomial(pool, Math.min(k, pool)));

/** size! / (n)_size: the chance of any one ordered choice of `size` distinct entries out of n, summed over the orderings of a fixed set. */
function orderings(n: number, size: number): Fraction {
  let falling = 1n;
  let factorial = 1n;
  for (let i = 0; i < size; i++) {
    falling *= BigInt(n - i);
    factorial *= BigInt(i + 1);
  }
  return frac(factorial, falling);
}

/** The product of the `size` largest entries (by value) of `bounds`. */
function topProduct(bounds: Fraction[], size: number): Fraction {
  const sorted = [...bounds].sort((a, b) => {
    const left = a.num * b.den;
    const right = b.num * a.den;
    return left < right ? 1 : left > right ? -1 : 0;
  });
  return sorted.slice(0, size).reduce((product, f) => mul(product, f), frac(1n, 1n));
}

/** The upper bound on the probability of the most likely team of the uniform component. */
export function uniformBound(vocabulary: BoundVocabulary, config: BoundConfig): Fraction {
  const { free, itemFree } = commonFactor(vocabulary, config);
  const k = config.moves_per_set ?? 4;
  const size = vocabulary.rules.max_team_size;
  const entries = vocabulary.roster.map((e) => mul(mul(setFactor(e.abilities.length, e.moves.length, k), e.required_item === undefined ? itemFree : frac(1n, 1n)), free));
  return mul(orderings(entries.length, size), topProduct(entries, size));
}

/** The upper bound for the role-set component: a member is a table entry, then a role, so an entry's bound averages its roles' bounds. */
export function roleSetsBound(vocabulary: BoundVocabulary, config: BoundConfig, table: Pick<RoleSetTable, "species">): Fraction {
  const { free, itemFree } = commonFactor(vocabulary, config);
  const k = config.moves_per_set ?? 4;
  const size = vocabulary.rules.max_team_size;
  const roster = new Map(vocabulary.roster.map((e) => [e.id, e]));
  const entries = table.species.map((entry) => {
    const roles = entry.sets.reduce((sum, role) => add(sum, setFactor(role.abilities.length, role.moves.length, k)), frac(0n, 1n));
    const required = roster.get(entry.id)?.required_item;
    return mul(mul(mul(roles, frac(1n, BigInt(entry.sets.length))), required === undefined ? itemFree : frac(1n, 1n)), free);
  });
  return mul(orderings(entries.length, size), topProduct(entries, size));
}

/** The mixture's bound: sum of share x component bound. */
export function mixtureBound(parts: { share_permille: number; bound: Fraction }[]): Fraction {
  const total = parts.reduce((sum, p) => add(sum, mul(frac(BigInt(p.share_permille), 1000n), p.bound)), frac(0n, 1n));
  return total;
}

/** P1 for component R: how many of the table's ids (species, then each set's moves and abilities) resolve in the vocabulary. */
export function roleSetsMapping(vocabulary: Pick<TeamVocabulary, "roster">, table: Pick<RoleSetTable, "species">): { mapped: number; total: number } {
  const roster = new Set(vocabulary.roster.map((e) => e.id));
  const moves = new Set(vocabulary.roster.flatMap((e) => e.moves));
  const abilities = new Set(vocabulary.roster.flatMap((e) => e.abilities));
  let mapped = 0;
  let total = 0;
  for (const entry of table.species) {
    total++;
    if (roster.has(entry.id)) mapped++;
    for (const set of entry.sets) {
      for (const move of set.moves) {
        total++;
        if (moves.has(move)) mapped++;
      }
      for (const ability of set.abilities) {
        total++;
        if (abilities.has(ability)) mapped++;
      }
    }
  }
  return { mapped, total };
}

export type UsageComponentConfig = Extract<NonNullable<TeamGeneratorConfig["proposal"]>["components"][number], { kind: "usage" }>;

/** P1 for component U: the aggregate mapped mass and, per list kind, the mapped share in permille (ADR-020 section 6 asks 95% per kind). */
export interface UsageMappingResult {
  mapped: number;
  total: number;
  /** Mapped share per list kind in permille, floored. */
  kinds: Record<(typeof USAGE_LIST_KINDS)[number], number>;
}

export const USAGE_LIST_KINDS = ["species", "abilities", "items", "moves", "spreads", "teammates"] as const;

/**
 * P1 for component U. Each of the six list kinds (species usage, abilities, items, moves, spreads, teammates) has a total of
 * 10^12 units; a kind's unmapped mass counts per species weighted by its usage, and a species the roster lacks
 * (unmapped_species_ppm) is unmapped in all six. `mapped` and `total` are the aggregate over the six kinds; `kinds` is
 * each kind's mapped share in permille, which the gate takes the minimum of.
 */
export function usageMapping(table: UsageSource["table"]): UsageMappingResult {
  const unit = 1_000_000;
  const per = unit * unit;
  const unmapped = { species: table.unmapped_species_ppm * unit, abilities: 0, items: 0, moves: 0, spreads: 0, teammates: 0 };
  for (const kind of USAGE_LIST_KINDS) if (kind !== "species") unmapped[kind] = table.unmapped_species_ppm * unit;
  for (const s of table.species) {
    unmapped.abilities += s.usage_ppm * s.abilities.unmapped_ppm;
    unmapped.items += s.usage_ppm * s.items.unmapped_ppm;
    unmapped.moves += s.usage_ppm * s.moves.unmapped_ppm;
    unmapped.spreads += s.usage_ppm * s.spreads.unmapped_ppm;
    unmapped.teammates += s.usage_ppm * s.teammates.unmapped_ppm;
  }
  const kinds = Object.fromEntries(USAGE_LIST_KINDS.map((k) => [k, Math.floor(((per - unmapped[k]) * 1000) / per)])) as UsageMappingResult["kinds"];
  const total = USAGE_LIST_KINDS.length * per;
  return { mapped: total - USAGE_LIST_KINDS.reduce((sum, k) => sum + unmapped[k], 0), total, kinds };
}

const maxOf = (weights: readonly number[]): number => weights.reduce((a, b) => Math.max(a, b), 0);
const sumOf = (weights: readonly number[]): number => weights.reduce((a, b) => a + b, 0);
const largest = (weights: readonly number[], count: number): number => sumOf([...weights].sort((a, b) => b - a).slice(0, count));
const atMostOne = (f: Fraction): Fraction => (f.num > f.den ? frac(1n, 1n) : f);
const fractionMax = (a: Fraction, b: Fraction): Fraction => (a.num * b.den >= b.num * a.den ? a : b);

/** (1 - floor) of the weighted bound plus floor of the uniform bound: the bound on one draw that may be taken uniformly. */
const mixed = (floor: number, weightedP: Fraction, uniformP: Fraction): Fraction => add(mul(frac(BigInt(1000 - floor), 1000n), weightedP), mul(frac(BigInt(floor), 1000n), uniformP));

/**
 * The upper bound for component U: the number of orderings of a team's members times the product of the largest
 * conditional species probabilities (a teammate step is bounded by the mediant inequality: the best single chosen
 * species' top teammate weight over what that species leaves after removing its largest other-teammate weights), times
 * the product of the six largest member bounds (item under the Item Clause, ability, ordered moves times their
 * orderings, spread, gender), each draw mixed with the uniform bound at the per-slot floor.
 */
export function usageBound(vocabulary: BoundVocabulary, config: BoundConfig, component: UsageComponentConfig, source: UsageSource): Fraction {
  return usageBounds(vocabulary, config, component, source).set;
}

/**
 * Both P3 figures for component U. `set` is the bound above, on the probability of a team as a SET of members (the gate's
 * figure). `sequence` is the ADR-020 section 6 wording: the product of the largest conditional weights along ONE draw route
 * (the first pick, the later picks, each member's set draws as an ordered sequence), with no ordering factors, so it bounds
 * the probability of one particular draw order and is smaller by up to 6! x (4!)^6. Neither counts a canonical team reached
 * by two routes (a Mega entry is rewritten to its base forme holding the stone, up to a factor of 2 for each such member)
 * or the last-resort uniform fallback (an empty list), where the bound mixes the uniform part in as if it were a list.
 */
export function usageBounds(vocabulary: BoundVocabulary, config: BoundConfig, component: UsageComponentConfig, source: UsageSource): { set: Fraction; sequence: Fraction } {
  const { itemFree } = commonFactor(vocabulary, config);
  const floor = component.floor_permille;
  const k = config.moves_per_set ?? 4;
  const size = vocabulary.rules.max_team_size;
  const n = vocabulary.roster.length;
  const members = usageMembers(source.table, component.set_root);
  const natures = BigInt(vocabulary.natures.length);
  const gender = config.gender_policy === "random" ? 2n : 1n;
  const limit = vocabulary.rules.item_clause;
  const removable = limit === undefined || limit < 1 ? 0 : Math.floor((size - 1) / limit);

  // Species steps. The first pick is at most the largest w_species share. A later pick of species j, whatever was chosen,
  // is at most R(j): by the mediant inequality its teammate share is at most the best single chosen species' tw_c[j] over
  // what c leaves of its list after removing its largest (size - 1) other weights; when no chosen species names an
  // unchosen one the pick falls back to w_species, at most w_j over the total less the largest (size - 1) weights. The
  // size - 1 later picks are distinct species, so their product is at most the product of the size - 1 largest R.
  const usageWeights = vocabulary.roster.map((e) => members.get(e.id)?.w_species ?? 0);
  const totalUsage = sumOf(usageWeights);
  const firstP = totalUsage > 0 ? frac(BigInt(maxOf(usageWeights)), BigInt(totalUsage)) : frac(1n, 1n);
  const first = mixed(floor, firstP, frac(1n, BigInt(n)));
  const fallbackRest = totalUsage - largest(usageWeights, size - 1);
  const reach = new Map<string, Fraction>(vocabulary.roster.map((e) => [e.id, fallbackRest > 0 ? atMostOne(frac(BigInt(members.get(e.id)?.w_species ?? 0), BigInt(fallbackRest))) : frac(1n, 1n)]));
  for (const m of members.values()) {
    const weights = [...m.teammates.values()];
    const rest = sumOf(weights) - largest(weights, size - 1);
    for (const [id, w] of m.teammates) reach.set(id, fractionMax(reach.get(id) ?? frac(0n, 1n), rest > 0 ? atMostOne(frac(BigInt(w), BigInt(rest))) : frac(1n, 1n)));
  }
  const laterSteps = [...reach.values()].map((r) => mixed(floor, r, frac(1n, BigInt(n - size + 1))));
  laterSteps.sort((a, b) => (a.num * b.den < b.num * a.den ? 1 : a.num * b.den > b.num * a.den ? -1 : 0));
  const steps = laterSteps.slice(0, size - 1).reduce((product, f) => mul(product, f), first);
  let factorial = 1n;
  for (let i = 2; i <= size; i++) factorial *= BigInt(i);

  // Member factors, one per roster entry.
  const entries = vocabulary.roster.map((e) => {
    const m: UsageMember | undefined = members.get(e.id);
    const itemP = e.required_item !== undefined ? frac(1n, 1n) : m === undefined || m.items.weights.length === 0 ? itemFree : mixed(floor, itemShare(m.items.weights, removable), itemFree);
    const abilityU = frac(1n, BigInt(e.abilities.length));
    const abilityP = m === undefined || m.abilities.weights.length === 0 ? abilityU : mixed(floor, frac(BigInt(maxOf(m.abilities.weights)), BigInt(sumOf(m.abilities.weights))), abilityU);
    const count = Math.min(k, e.moves.length);
    // An unordered subset of `count` moves: count! orderings, each at most the product of the count largest weights
    // over the product of (total - the j largest) for pick j (the picks so far weigh at most the j largest).
    const weights = [...(m?.moves.weights ?? [])].sort((x, y) => y - x);
    // With a floor each pick is a sum of a weighted and a uniform part, so the largest weight bounds every position;
    // with none, the j-th largest weight bounds the j-th position of the product.
    const totalMoves = sumOf(weights);
    let moveP = frac(1n, 1n);
    for (let j = 0; j < count; j++) {
      const uniformP = frac(1n, BigInt(e.moves.length - j));
      const rest = totalMoves - largest(weights, j);
      const w = floor === 0 ? weights[j] : weights[0];
      moveP = mul(moveP, j >= weights.length || w === undefined || rest <= 0 ? mixed(floor, floor === 0 && j >= weights.length ? uniformP : frac(1n, 1n), uniformP) : mixed(floor, atMostOne(frac(BigInt(w), BigInt(rest))), uniformP));
    }
    const moveSequence = moveP;
    for (let j = 2; j <= count; j++) moveP = mul(moveP, frac(BigInt(j), 1n));
    moveP = atMostOne(moveP);
    const spreadU = frac(1n, natures);
    const spreadP = m === undefined || m.spreads.weights.length === 0 ? spreadU : mixed(floor, frac(BigInt(maxOf(m.spreads.weights)), BigInt(sumOf(m.spreads.weights))), spreadU);
    const rest = mul(mul(mul(itemP, abilityP), spreadP), frac(1n, gender));
    return { set: mul(rest, moveP), sequence: mul(rest, moveSequence) };
  });
  return { set: mul(mul(frac(factorial, 1n), steps), topProduct(entries.map((e) => e.set), size)), sequence: mul(steps, topProduct(entries.map((e) => e.sequence), size)) };
}

/** The largest item share when the Item Clause may have taken the `removable` heaviest items out of reach. */
function itemShare(weights: readonly number[], removable: number): Fraction {
  const total = sumOf(weights);
  const rest = total - largest(weights, removable);
  return rest <= 0 ? frac(1n, 1n) : atMostOne(frac(BigInt(maxOf(weights)), BigInt(rest)));
}

/** A gate: the measured value, the config's threshold (null when none) and the verdict (null when there is nothing to compare). */
export interface GateResult {
  value: number;
  threshold: number | null;
  pass: boolean | null;
}

const gate = (value: number, threshold: number | undefined, passes: (v: number, t: number) => boolean): GateResult =>
  threshold === undefined ? { value, threshold: null, pass: null } : { value, threshold, pass: passes(value, threshold) };

export interface ComponentCoverage {
  kind: string;
  share_permille: number;
  pilot: { draws: number; rejected: number; accepted: number };
  p1: (GateResult & { mapped: number; total: number; kinds?: Record<string, number> }) | null;
  p2: GateResult;
  p3: GateResult;
  /** Usage only: the ADR-literal ordered-sequence P3 figure in bits, reported beside the set-level one the gate uses (never gated). */
  p3_sequence_bits?: number;
  /** Accepted pilot draws per base cell. */
  pilot_cell_hits: number[];
  /** Delivered teams of the dataset per base cell, over all splits. */
  delivered_cells: number[];
  delivered: Partial<Record<Split, number>>;
}

export interface ProposalCoverage {
  gates: Gates | null;
  components: ComponentCoverage[];
  /** The mixture's own P3 bound: sum of share x component bound. */
  mixture_p3_bits: number;
  p4: { strata: number; reference_strata: number | null; value: number | null; threshold: number | null; pass: boolean | null };
  /** The gates that failed or could not be evaluated, as `P2:role_sets`, `P4`, ... */
  failed: string[];
  /** Every gate evaluated and met (false without configured gates). */
  publishable: boolean;
}

export interface DeliveredComponent {
  cells: number[];
  splits: Partial<Record<Split, number>>;
}

export interface GateInput {
  config: TeamGeneratorConfig;
  vocabulary: BoundVocabulary & Pick<TeamVocabulary, "roster">;
  roleSets?: { table: Pick<RoleSetTable, "species"> } | undefined;
  /** The usage weight table a `usage` component reads. */
  usage?: UsageSource | undefined;
  pilot: Pilot;
  plan: Pick<QuotaPlan, "strata">;
  /** The number of strata of the uniform config's plan at the same d_max; null when no uniform reference pilot was given. */
  referenceStrata: number | null;
  /** The teams each component delivered, by kind. */
  delivered: ReadonlyMap<string, DeliveredComponent>;
}

export class GateError extends Error {
  override name = "GateError";
}

/** E-13.6: the population the dataset's teams are a sample of. */
export function populationOf(config: TeamGeneratorConfig, note?: string): string {
  const shares = (config.proposal?.components ?? [{ kind: "uniform", share_permille: 1000 }]).map((c) => `${c.kind} ${c.share_permille} per mille`).join(", ");
  return `proposal ${config.proposal === undefined ? "uniform" : "mixture"} ${config.config_id} (${shares}) conditioned on the merged cells, weighted by the published quotas${note === undefined ? "" : `; ${note}`}`;
}

/** The pilot gates and the per-component report of a mixture dataset. */
export function proposalCoverage(input: GateInput): ProposalCoverage {
  const { config, pilot } = input;
  const proposal = config.proposal;
  if (proposal === undefined) throw new GateError("the config has no proposal");
  if (pilot.components === undefined) throw new GateError("the pilot has no per-component counts: record it again with this config");
  const gates = proposal.gates;
  const bounds: { share_permille: number; bound: Fraction }[] = [];
  const components = proposal.components.map((component): ComponentCoverage => {
    const part = pilot.components?.find((p) => p.kind === component.kind);
    if (part === undefined) throw new GateError(`the pilot has no counts for component ${component.kind}`);
    const accepted = part.draws - part.rejected;
    const usageBoundsOf = component.kind === "usage" ? usageBounds(input.vocabulary, config, component, input.usage ?? fail("no usage weight table was given")) : undefined;
    const bound =
      usageBoundsOf !== undefined
        ? usageBoundsOf.set
        : component.kind === "role_sets"
          ? roleSetsBound(input.vocabulary, config, (input.roleSets ?? fail("no role-set table was given")).table)
          : uniformBound(input.vocabulary, config);
    bounds.push({ share_permille: component.share_permille, bound });
    const usageMap = component.kind === "usage" ? usageMapping((input.usage as UsageSource).table) : undefined;
    const mapping = component.kind === "role_sets" ? roleSetsMapping(input.vocabulary, (input.roleSets as { table: Pick<RoleSetTable, "species"> }).table) : usageMap;
    const p1Threshold = component.kind === "usage" ? gates?.p1_min_usage_permille : 1000;
    const delivered = input.delivered.get(component.kind);
    return {
      kind: component.kind,
      share_permille: component.share_permille,
      pilot: { draws: part.draws, rejected: part.rejected, accepted },
      // Usage: the gate takes the minimum over the six list kinds (ADR-020 section 6: per kind); the aggregate is reported beside it.
      p1:
        mapping === undefined
          ? null
          : {
              ...gate(usageMap !== undefined ? Math.min(...Object.values(usageMap.kinds)) : mapping.total === 0 ? 0 : Math.floor((mapping.mapped * 1000) / mapping.total), p1Threshold, (v, t) => v >= t),
              mapped: mapping.mapped,
              total: mapping.total,
              ...(usageMap === undefined ? {} : { kinds: usageMap.kinds }),
            },
      p2: gate(part.draws === 0 ? 0 : Math.floor((accepted * 1000) / part.draws), gates?.p2_min_accept_permille, (v, t) => v >= t),
      p3: gate(bitsOf(bound), component.kind === "usage" ? (gates?.p3_min_bits_usage ?? gates?.p3_min_bits) : gates?.p3_min_bits, (v, t) => v >= t),
      ...(usageBoundsOf === undefined ? {} : { p3_sequence_bits: bitsOf(usageBoundsOf.sequence) }),
      pilot_cell_hits: part.cell_hits,
      delivered_cells: delivered?.cells ?? new Array<number>(part.cell_hits.length).fill(0),
      delivered: delivered?.splits ?? {},
    };
  });
  const strata = input.plan.strata.length;
  const p4 =
    input.referenceStrata === null
      ? { strata, reference_strata: null, value: null, threshold: gates?.p4_min_strata_permille ?? null, pass: null }
      : {
          ...gate(Math.floor((strata * 1000) / Math.max(input.referenceStrata, 1)), gates?.p4_min_strata_permille, (v, t) => v >= t),
          strata,
          reference_strata: input.referenceStrata,
        };
  const failed: string[] = [];
  if (gates === undefined) failed.push("gates:not_configured");
  for (const c of components) {
    if (c.p1 !== null && c.p1.pass !== true) failed.push(`P1:${c.kind}`);
    if (c.p2.pass !== true) failed.push(`P2:${c.kind}`);
    if (c.p3.pass !== true) failed.push(`P3:${c.kind}`);
  }
  if (p4.pass !== true) failed.push("P4");
  return {
    gates: gates ?? null,
    components,
    mixture_p3_bits: bitsOf(mixtureBound(bounds)),
    p4,
    failed,
    publishable: failed.length === 0,
  };
}

function fail(message: string): never {
  throw new GateError(message);
}
