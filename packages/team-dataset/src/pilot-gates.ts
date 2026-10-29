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
  p1: (GateResult & { mapped: number; total: number }) | null;
  p2: GateResult;
  p3: GateResult;
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
    const bound = component.kind === "role_sets" ? roleSetsBound(input.vocabulary, config, (input.roleSets ?? fail("no role-set table was given")).table) : uniformBound(input.vocabulary, config);
    bounds.push({ share_permille: component.share_permille, bound });
    const mapping = component.kind === "role_sets" ? roleSetsMapping(input.vocabulary, (input.roleSets as { table: Pick<RoleSetTable, "species"> }).table) : undefined;
    const delivered = input.delivered.get(component.kind);
    return {
      kind: component.kind,
      share_permille: component.share_permille,
      pilot: { draws: part.draws, rejected: part.rejected, accepted },
      p1: mapping === undefined ? null : { ...gate(mapping.total === 0 ? 0 : Math.floor((mapping.mapped * 1000) / mapping.total), 1000, (v, t) => v >= t), mapped: mapping.mapped, total: mapping.total },
      p2: gate(part.draws === 0 ? 0 : Math.floor((accepted * 1000) / part.draws), gates?.p2_min_accept_permille, (v, t) => v >= t),
      p3: gate(bitsOf(bound), gates?.p3_min_bits, (v, t) => v >= t),
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
