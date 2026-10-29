// Damage evidence of isolated hits (ADR-025 section 7, ticket #1132; finding N9). A hit of the opponent's Pokemon
// (as the attacker or as the defender), with its shown hit-point change, is evidence about the hidden set that
// took or dealt it. The check itself is the ADR-016 damage probe, behind the `DamageCheck` port that the bridge
// implements over the pinned engine (this package stays engine-free): for a candidate set the port says whether
// the probe produces some roll whose shown hit points equal what was observed. No damage formula, tolerance or
// modifier rule is written here; the reviewed rules (how many candidates are checked, which candidates the probe
// does not model) are `BeliefConfig.damage_evidence`.
//
// Likelihood: an indicator. A candidate that passes, or that is not checked (beyond the bound, an "other spread"
// that names no nature or Stat Points, a candidate the config calls unmodelled, a probe answer `unsupported`),
// keeps its weight; a candidate that fails gets weight 0. So the true set always keeps a positive weight unless
// the port is wrong. The belief is a product of per-field marginals given the forme, so the evidence enters it
// as the marginals of the exact posterior over the candidate space (mean field): each ability, item and spread
// is reweighted by the share of the OTHER fields' mass that passes with it, and the forme keeps the passing share
// of its mass. Worlds drawn from the product may pair a failing combination; they are never wrong about a field.
// Bounded cost: per event and forme only the `max_checked` most likely candidates (by prior mass) are asked.
// An event whose checked candidates are all ruled out is dropped and counted (`contradictions`): the evidence
// and the model disagree (a modifier nobody saw), and the belief is never emptied by it.

import { type Dist, parseSpreadKey } from "./fields.ts";
import { type Ratio, ONE, ZERO, add, integerise, isZero, mul, ratio } from "./ratio.ts";

/** One isolated hit that concerns one opponent slot. */
export interface DamageEvent {
  /** An opaque id, unique among the check's events (used in fixed-word notes only). */
  readonly id: string;
  /** The 0-based position in the opponent's preview roster. */
  readonly slot: number;
  /** Which side of the hit the slot's Pokemon was on. */
  readonly role: "attacker" | "defender";
}

/** A candidate set of one slot: the fields the evidence can touch (the rest is not asked). */
export interface DamageCandidate {
  /** The preview species id of the slot. */
  readonly species: string;
  /** The prior forme id (a Mega forme or the species itself). */
  readonly forme: string;
  /** The forme is a Mega forme (a stone holder before the Evolution, or the evolved Pokemon). */
  readonly mega: boolean;
  readonly ability: string;
  readonly item: string | null;
  readonly nature: string;
  readonly stat_points: Readonly<Record<"hp" | "atk" | "def" | "spa" | "spd" | "spe", number>>;
}

/**
 * The port the bridge implements over the damage probe. `check` is true when the candidate can have produced the
 * observed hit points, false when no roll can, and null when the probe cannot say (`unsupported`, a refused set):
 * null adds no constraint.
 */
export interface DamageCheck {
  readonly events: readonly DamageEvent[];
  check(event: DamageEvent, candidate: DamageCandidate): boolean | null;
}

/** The reviewed values the belief reads (`BeliefConfig.damage_evidence`). */
export interface DamageRules {
  readonly max_checked: number;
  readonly unmodelled: { readonly abilities: readonly string[]; readonly items: readonly string[] };
}

/** What the evidence did, counted per belief (integers; events are the port's, candidates are per forme and event). */
export interface DamageCounts {
  /** Events given for the slots the belief reads. */
  events: number;
  /** Events that removed weight from some candidate. */
  applied: number;
  /** Candidates the probe was asked about. */
  checked: number;
  /** Candidates the probe ruled out. */
  rejected: number;
  /** Candidates that were not asked (unmodelled) or that the probe could not answer. */
  unsupported: number;
  /** Events with no candidate the port could be asked about (no named spread, or nothing enumerated). */
  unchecked: number;
  /** Events dropped because they ruled out every candidate. */
  contradictions: number;
}

export const emptyDamageCounts = (): DamageCounts => ({ events: 0, applied: 0, checked: 0, rejected: 0, unsupported: 0, unchecked: 0, contradictions: 0 });

/** The slice of one forme's work the evidence reads and rewrites. */
export interface DamageForme {
  readonly id: string;
  readonly mega: boolean;
  readonly abilities: Dist<string>;
  readonly spreads: Dist<string>;
  /** The probability mass of each item (the item evidence is inside it; null is "holds no item"). */
  readonly items: ReadonlyMap<string | null, Ratio>;
}

interface Candidate {
  readonly ability: string;
  readonly item: string | null;
  readonly spread: string;
  readonly weight: bigint;
}

const keyOf = (item: string | null): string => (item === null ? "" : item);

/** The candidates of a forme, most likely first (integer weights; ties by ids), named spreads only, at most `limit`. */
function top(forme: DamageForme, limit: number): Candidate[] {
  const itemIds = [...forme.items.keys()];
  const itemWeights = integerise(itemIds.map((id) => forme.items.get(id) as Ratio)).weights;
  const out: Candidate[] = [];
  for (const s of forme.spreads.options) {
    if (parseSpreadKey(s.id) === null) continue;
    for (const a of forme.abilities.options) {
      itemIds.forEach((item, k) => {
        const w = itemWeights[k] as bigint;
        if (w > 0n) out.push({ ability: a.id, item, spread: s.id, weight: a.weight * w * s.weight });
      });
    }
  }
  out.sort((x, y) => (x.weight === y.weight ? (x.ability < y.ability ? -1 : x.ability > y.ability ? 1 : keyOf(x.item) < keyOf(y.item) ? -1 : keyOf(x.item) > keyOf(y.item) ? 1 : x.spread < y.spread ? -1 : 1) : x.weight > y.weight ? -1 : 1));
  return out.slice(0, limit);
}

/** 1 - r. */
const complement = (r: Ratio): Ratio => add(ONE, { num: -r.num, den: r.den });

/** Each option's share times the share of the other fields' mass that passes with it, positives only. */
function reweigh<T extends string | null>(options: readonly { readonly id: T; readonly share: Ratio }[], failed: ReadonlyMap<T, Ratio>): { id: T; p: Ratio }[] {
  return options.map((o) => ({ id: o.id, p: mul(o.share, complement(failed.get(o.id) ?? ZERO)) })).filter((o) => !isZero(o.p));
}

function distOf(list: readonly { id: string; p: Ratio }[]): Dist<string> {
  const { weights } = integerise(list.map((o) => o.p));
  return { total: weights.reduce((s, w) => s + w, 0n), options: list.map((o, k) => ({ id: o.id, weight: weights[k] as bigint })) };
}

/** The formes of a slot after the evidence; an empty list is a defect of this module (the argument is in `applyDamage`), never a belief. */
export function nonEmpty<F>(formes: readonly F[]): F[] {
  if (formes.length === 0) throw new RangeError("the damage evidence left a slot with no forme");
  return [...formes];
}

/**
 * Apply the events of one slot to its formes. Returns the rewritten formes (those with every field still possible)
 * and fixed-word notes; `counts` is added to. The input formes are not changed.
 */
export function applyDamage<F extends DamageForme>(formes: readonly F[], species: string, events: readonly DamageEvent[], check: DamageCheck, rules: DamageRules, counts: DamageCounts): { formes: F[]; notes: string[] } {
  let current: F[] = [...formes];
  const notes: string[] = [];
  const unmodelledAbilities = new Set(rules.unmodelled.abilities);
  const unmodelledItems = new Set(rules.unmodelled.items);
  for (const event of events) {
    counts.events += 1;
    let checked = 0;
    let unsupported = 0;
    const failures = new Map<string, Candidate[]>();
    for (const forme of current) {
      const failed: Candidate[] = [];
      for (const c of top(forme, rules.max_checked)) {
        const spread = parseSpreadKey(c.spread);
        if (spread === null) continue;
        if (unmodelledAbilities.has(c.ability) || (c.item !== null && unmodelledItems.has(c.item))) {
          unsupported += 1;
          continue;
        }
        const verdict = check.check(event, { species, forme: forme.id, mega: forme.mega, ability: c.ability, item: c.item, nature: spread.nature, stat_points: spread.stat_points });
        if (verdict === null) unsupported += 1;
        else {
          checked += 1;
          if (!verdict) failed.push(c);
        }
      }
      failures.set(forme.id, failed);
    }
    const rejected = [...failures.values()].reduce((s, f) => s + f.length, 0);
    counts.checked += checked;
    counts.rejected += rejected;
    counts.unsupported += unsupported;
    if (checked + unsupported === 0) {
      counts.unchecked += 1;
      notes.push(`damage_unchecked:${event.role}`);
      continue;
    }
    if (rejected === 0) continue;
    if (rejected === checked && unsupported === 0) {
      counts.contradictions += 1;
      notes.push(`damage_contradiction:${event.role}`);
      continue;
    }

    const next: F[] = [];
    for (const forme of current) {
      const failed = failures.get(forme.id) as Candidate[];
      if (failed.length === 0) {
        next.push(forme);
        continue;
      }
      const itemTotal = [...forme.items.values()].reduce(add, ZERO);
      const pa = new Map(forme.abilities.options.map((o) => [o.id, ratio(o.weight, forme.abilities.total)]));
      const ps = new Map(forme.spreads.options.map((o) => [o.id, ratio(o.weight, forme.spreads.total)]));
      const pn = (item: string | null): Ratio => mul(forme.items.get(item) as Ratio, ratio(itemTotal.den, itemTotal.num));
      const byAbility = new Map<string, Ratio>();
      const bySpread = new Map<string, Ratio>();
      const byItem = new Map<string | null, Ratio>();
      for (const c of failed) {
        const a = pa.get(c.ability) as Ratio;
        const s = ps.get(c.spread) as Ratio;
        const i = pn(c.item);
        byAbility.set(c.ability, add(byAbility.get(c.ability) ?? ZERO, mul(i, s)));
        bySpread.set(c.spread, add(bySpread.get(c.spread) ?? ZERO, mul(a, i)));
        byItem.set(c.item, add(byItem.get(c.item) ?? ZERO, mul(a, s)));
      }
      const abilities = reweigh([...pa].map(([id, share]) => ({ id, share })), byAbility);
      const spreads = reweigh([...ps].map(([id, share]) => ({ id, share })), bySpread);
      const items = new Map<string | null, Ratio>();
      for (const [id, mass] of forme.items) {
        const left = mul(mass, complement(byItem.get(id) ?? ZERO));
        if (!isZero(left)) items.set(id, left);
      }
      if (abilities.length === 0 || spreads.length === 0 || items.size === 0) continue;
      next.push({ ...forme, abilities: distOf(abilities), spreads: distOf(spreads), items });
    }
    // `next` is never empty here: a forme is dropped only when a field loses every option, which needs all of the forme's
    // candidates to have failed (all enumerated, none unsupported), and all formes in that state is the contradiction above.
    counts.applied += 1;
    notes.push(`damage_applied:${event.role}`);
    current = next;
  }
  return { formes: nonEmpty(current), notes };
}
