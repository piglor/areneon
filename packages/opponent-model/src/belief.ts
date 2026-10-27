// `belief`: the opponent's hidden team as an exact distribution (ADR-025 section 3), a pure function
// of the shared reader's opponent state at the request. For each preview slot the model is a forme
// from the prior's forme weights (the stone-holding share for a Mega forme), and given the forme
// an ability, an item, a spread and a move set drawn independently from the prior's lists with the
// residual expanded. The slots couple only through Item Clause (exact subset DP, "no item" exempt)
// and the brought set (uniform over the subsets of the picked size that hold every settled
// Pokemon). Hard constraints from the evidence allowlist (`evidence.ts`) give a world weight 0.
// All arithmetic is exact (BigInt and `Ratio`); there is no floating point. No game rule and no
// species, item or move name is written here: rule values and lists come from the vocabulary.

import type { PublicMon } from "@areneon/public-state";
import type { UsagePrior } from "@areneon/schema/types/v1/usage-prior.ts";
import { BeliefError } from "./error.ts";
import { type SlotEvidence, slotEvidence } from "./evidence.ts";
import { type Dist, abilityDist, itemDist, only, spreadDist, uniformDist, weightOf } from "./fields.ts";
import { solveItemClause } from "./item-clause.ts";
import { moveModel } from "./moves.ts";
import { type Ratio, ONE, ZERO, add, greater, integerise, isZero, mul, ratio } from "./ratio.ts";
import { BELIEF_FORMAT, type BeliefInput, type BeliefVocabulary, type Cell, type FormeBelief, type Marginal, type OpponentBelief, type SlotBelief } from "./types.ts";

type PriorSpecies = UsagePrior["species"][number];
type PriorForme = PriorSpecies["formes"][number];
type RosterEntry = BeliefVocabulary["roster"][number];

/** A forme of a slot that survived the evidence, before the Item Clause. */
interface FormeWork {
  readonly id: string;
  readonly mega: boolean;
  /** w_f / D times the likelihood of the ability and move evidence: everything but the item. */
  readonly weight: Ratio;
  readonly abilities: Dist<string>;
  readonly spreads: Dist<string>;
  readonly moves: NonNullable<ReturnType<typeof moveModel>>["model"];
  /** P(item | forme) including the item evidence, for every item with a positive probability (null: no item). */
  readonly items: ReadonlyMap<string | null, Ratio>;
}

interface Context {
  readonly vocabulary: BeliefVocabulary;
  readonly roster: ReadonlyMap<string, RosterEntry>;
  readonly items: ReadonlySet<string>;
}

const compare = (a: string | null, b: string | null): number => (a === b ? 0 : a === null ? -1 : b === null ? 1 : a < b ? -1 : 1);

function entryOf(ctx: Context, id: string, slot: number): RosterEntry {
  const entry = ctx.roster.get(id);
  if (entry === undefined) throw new BeliefError("forme_not_in_prior", "a forme of the prior is not in the vocabulary roster", slot);
  return entry;
}

/** A forme that has its item fixed by the vocabulary (a Mega forme, a mask): its entry requires one. */
const fixedItem = (entry: RosterEntry): string | null => entry.required_item ?? null;

/** A Mega forme of the preview species: a forme the entry makes by a required item and that is not the species itself. */
const isMega = (species: PriorSpecies, forme: PriorForme, entry: RosterEntry): boolean => entry.required_item !== undefined && forme.id !== species.id;

/**
 * The formes the shown forme and a Mega Evolution leave possible. A Mega Evolution must be explained by a
 * Mega forme of the species with that stone. A forme the prior has no entry for that is not a Mega
 * (a forme change in battle, such as a broken Disguise) adds no forme constraint: the belief stays wider.
 */
function consistentFormes(species: PriorSpecies, ev: SlotEvidence, ctx: Context, slot: number, notes: string[]): PriorForme[] {
  const mega = (f: PriorForme): boolean => isMega(species, f, entryOf(ctx, f.id, slot));
  if (ev.mega) {
    const kept = species.formes.filter((f) => mega(f) && f.id === ev.shown && (ev.megaStone === null || entryOf(ctx, f.id, slot).required_item === ev.megaStone));
    if (kept.length === 0) throw new BeliefError("forme_not_in_prior", "no Mega forme of the species' prior entry explains the Mega Evolution", slot);
    return kept;
  }
  // Before the Evolution a stone holder is shown as the preview species.
  const kept = species.formes.filter((f) => ev.shown === null || f.id === ev.shown || (mega(f) && ev.shown === species.id));
  if (kept.length > 0) return kept;
  notes.push(`forme_unmatched:${ev.shown}`);
  return species.formes;
}

/** The forme-and-item weights of one slot, after the ability, item and move evidence. */
function slotWork(species: PriorSpecies, ev: SlotEvidence, ctx: Context, slot: number, notes: string[]): FormeWork[] {
  const formes = consistentFormes(species, ev, ctx, slot, notes);
  const entries = new Map(formes.map((f) => [f.id, entryOf(ctx, f.id, slot)]));
  const speciesEntry = entryOf(ctx, species.id, slot);
  const stones = new Set(species.formes.flatMap((f) => (isMega(species, f, entryOf(ctx, f.id, slot)) ? [fixedItem(entryOf(ctx, f.id, slot))] : [])).filter((s): s is string => s !== null));
  const base = species.formes.find((f) => f.id === species.id);
  const baseAbilities: Dist<string> = base === undefined ? uniformDist(speciesEntry.abilities) : abilityDist(base.abilities.entries, base.abilities.residual, speciesEntry.abilities);
  const preAbilities = new Set(speciesEntry.abilities);

  // Moves the Pokemon chose: a move no consistent forme can hold (Struggle, a copied move) adds no evidence.
  const poolOf = (id: string): ReadonlySet<string> => new Set((entries.get(id) as RosterEntry).moves);
  const placeable = ev.moves.filter((m) => formes.some((f) => poolOf(f.id).has(m)));
  for (const m of ev.moves) notes.push(placeable.includes(m) ? `move:${m}` : `move_skipped:${m}`);
  if (ev.transformed) notes.push("moves_skipped:transform");

  // The ability list in force per forme, and what the revealed ability says about each.
  const abilityOf = formes.map((f) => {
    const entry = entries.get(f.id) as RosterEntry;
    const mega = isMega(species, f, entry);
    const evolved = mega && (ev.mega || ev.shown === f.id);
    const dist = mega && !evolved ? baseAbilities : abilityDist(f.abilities.entries, f.abilities.residual, entry.abilities);
    const verdict = ev.ability === null ? "free" : evolved && preAbilities.has(ev.ability) ? "ambiguous" : weightOf(dist, ev.ability) > 0n ? "fixed" : "excluded";
    return { f, mega, dist, verdict };
  });
  const usable = ev.ability !== null && abilityOf.some((a) => a.verdict !== "excluded");
  if (ev.ability !== null) {
    const kinds = abilityOf.map((a) => a.verdict);
    notes.push(!usable ? `ability_skipped:${ev.ability}` : kinds.includes("fixed") ? `ability:${ev.ability}` : `ability_ambiguous:${ev.ability}`);
  }
  const itemUsable = ev.item !== null && ctx.items.has(ev.item);
  if (ev.item !== null) notes.push(itemUsable ? `item:${ev.item}` : `item_skipped:${ev.item}`);

  const work: FormeWork[] = [];
  for (const { f, mega, dist, verdict } of abilityOf) {
    const entry = entries.get(f.id) as RosterEntry;
    const pool = poolOf(f.id);
    if (placeable.some((m) => !pool.has(m))) continue;
    if (usable && verdict === "excluded") continue;
    const model = moveModel(f.moves, [...pool], placeable);
    if (model === null) continue;

    // The ability the forme is in the revealed state of: a point mass when the evidence fixes it.
    const fixed = usable && verdict === "fixed" && ev.ability !== null;
    const abilityLikelihood = fixed ? ratio(weightOf(dist, ev.ability as string), dist.total) : ONE;
    const abilities = fixed ? only(ev.ability as string) : dist;

    const stone = fixedItem(entry);
    const full: Dist<string | null> = stone !== null ? only<string | null>(stone) : itemDist(f.items.entries, f.items.no_item, f.items.residual, ctx.vocabulary.items, stones);
    if (full.total === 0n) throw new BeliefError("empty_distribution", "a prior item list has no mass", slot);
    const items = new Map<string | null, Ratio>();
    for (const o of full.options) {
      if (itemUsable && o.id !== ev.item) continue;
      items.set(o.id, ratio(o.weight, full.total));
    }
    if (items.size === 0) continue;

    const spreads = spreadDist(f.spreads.entries, f.spreads.residual);
    const weight = mul(mul(ratio(BigInt(f.weight), BigInt(species.denominator)), abilityLikelihood), model.likelihood);
    work.push({ id: f.id, mega, weight, abilities, spreads, moves: model.model, items });
  }
  if (work.length === 0) throw new BeliefError("evidence_has_no_support", "the evidence leaves no forme of the slot with a positive weight", slot);
  return work;
}

/** Subsets of `n` of the picked size that hold the `settled` Pokemon: C(n - settled, picked - settled). */
function subsetCount(n: number, picked: number, settled: number): bigint {
  let count = 1n;
  const free = n - settled;
  const take = picked - settled;
  for (let k = 0; k < take; k += 1) count = (count * BigInt(free - k)) / BigInt(k + 1);
  return count;
}

/** Marginals as a sorted list of the options with a positive probability. */
function marginalList<T extends string | null>(totals: ReadonlyMap<T, Ratio>): Marginal<T>[] {
  return [...totals].filter(([, p]) => !isZero(p)).map(([id, p]) => ({ id, p })).sort((a, b) => compare(a.id, b.id));
}

const mode = <T extends string | null>(list: readonly Marginal<T>[]): T => list.reduce((best, m) => (greater(m.p, best.p) ? m : best)).id;

export function belief(input: BeliefInput): OpponentBelief {
  const { prior, vocabulary, config, opponent, flags } = input;
  if (flags.length > 0) throw new BeliefError("reader_flagged", "the reader flagged a line of this channel");
  if (prior.format_id !== vocabulary.format_id || prior.engine_data_hash !== vocabulary.engine_data_hash) {
    throw new BeliefError("format_mismatch", "the usage prior is for another format or engine data than the vocabulary");
  }
  const clause = vocabulary.rules.item_clause;
  if (clause !== undefined && clause !== 1) throw new BeliefError("item_clause_unsupported", "the rule table's Item Clause is neither absent nor 1");
  const mons: readonly PublicMon[] = opponent.roster;
  const picked = vocabulary.rules.picked_team_size;
  if (mons.some((m) => m.phantom === true)) throw new BeliefError("reader_flagged", "the reader holds a phantom entry (an Illusion twin)");
  if (mons.length < vocabulary.rules.min_team_size || mons.length > vocabulary.rules.max_team_size || picked < 1 || picked > mons.length) {
    throw new BeliefError("team_size", "the opponent's preview is not a team of the format");
  }
  const ctx: Context = { vocabulary, roster: new Map(vocabulary.roster.map((r) => [r.id, r])), items: new Set(vocabulary.items) };
  const hiding = new Set(config.identity_hiding_abilities);
  const speciesById = new Map(prior.species.map((s) => [s.id, s]));

  const chosen = mons.map((mon, slot) => {
    const species = speciesById.get(mon.base_species) ?? (mon.species === null ? undefined : speciesById.get(mon.species));
    if (species === undefined) throw new BeliefError("species_not_in_prior", "a previewed species has no entry in the usage prior", slot);
    const ids = [species.id, ...species.formes.map((f) => f.id)];
    if (ids.some((id) => ctx.roster.get(id)?.abilities.some((a) => hiding.has(a)))) {
      throw new BeliefError("identity_hiding_species", "a previewed species can hide its identity (the Illusion-aware belief is a later slice)", slot);
    }
    return { species, ev: slotEvidence(mon) };
  });

  const settled = chosen.flatMap((c, i) => (c.ev.settled ? [i] : []));
  if (settled.length > picked) throw new BeliefError("brought_exceeds_picked", "more Pokemon have been in play than the picked team size");

  const notes: string[][] = chosen.map(() => []);
  const works = chosen.map((c, i) => slotWork(c.species, c.ev, ctx, i, notes[i] as string[]));

  // Each slot's cells on one integer scale, then the Item Clause DP over all slots.
  const cells = works.map((work) => {
    const raw = work.flatMap((w) => [...w.items].map(([item, p]) => ({ forme: w.id, item, p: mul(w.weight, p) }))).filter((c) => !isZero(c.p));
    const { weights } = integerise(raw.map((c) => c.p));
    return raw.map((c, k): Cell => ({ forme: c.forme, item: c.item, weight: weights[k] as bigint }));
  });
  const itemIds = [...new Set(cells.flatMap((slot) => slot.flatMap((c) => (c.item === null ? [] : [c.item]))))].sort();
  const weight = (slot: readonly Cell[], item: string | null): bigint => slot.filter((c) => c.item === item).reduce((s, c) => s + c.weight, 0n);
  const solved = solveItemClause({ items: cells.map((slot) => itemIds.map((id) => weight(slot, id))), none: cells.map((slot) => weight(slot, null)) }, clause === 1);
  const Z = solved.normaliser;
  if (Z === 0n) throw new BeliefError("evidence_has_no_support", "the evidence and the Item Clause leave no world with a positive weight");

  const slots: SlotBelief[] = works.map((work, i) => {
    const slotCells = cells[i] as Cell[];
    const joint = (item: string | null): bigint => (item === null ? (solved.none[i] as bigint) : ((solved.items[i] as bigint[])[itemIds.indexOf(item)] as bigint));
    const itemP = new Map<string | null, Ratio>([[null, ratio(joint(null), Z)], ...itemIds.map((id): [string, Ratio] => [id, ratio(joint(id), Z)])]);

    const formes: FormeBelief[] = work.map((w) => {
      let p = ZERO;
      for (const c of slotCells.filter((x) => x.forme === w.id)) p = add(p, ratio(joint(c.item) * c.weight, Z * weight(slotCells, c.item)));
      return { id: w.id, mega: w.mega, p, abilities: w.abilities, spreads: w.spreads, moves: w.moves };
    }).filter((f) => !isZero(f.p));

    const mix = <K extends string>(of: (f: FormeBelief) => ReadonlyMap<K, Ratio>): Map<K, Ratio> => {
      const out = new Map<K, Ratio>();
      for (const f of formes) for (const [id, q] of of(f)) out.set(id, add(out.get(id) ?? ZERO, mul(f.p, q)));
      return out;
    };
    const share = (d: Dist<string>): Map<string, Ratio> => new Map(d.options.map((o) => [o.id, ratio(o.weight, d.total)]));
    const abilities = marginalList(mix((f) => share(f.abilities)));
    const spreads = marginalList(mix((f) => share(f.spreads)));
    const moves = marginalList(mix((f) => new Map(f.moves.inclusion.map((m) => [m.id, m.p]))));
    const items = marginalList(itemP);
    const formeMarginals = formes.map((f) => ({ id: f.id, p: f.p }));
    const { ev, species } = chosen[i] as (typeof chosen)[number];
    const isSettled = ev.settled;
    return {
      index: i,
      species: species.id,
      shown: { name: ev.name, species: ev.shown, level: ev.level, gender: ev.gender },
      settled: isSettled,
      p_brought: isSettled ? ONE : ratio(BigInt(picked - settled.length), BigInt(mons.length - settled.length)),
      formes,
      abilities,
      items,
      moves,
      spreads,
      mode: { forme: mode(formeMarginals), ability: mode(abilities), item: mode(items) },
      cells: slotCells,
      constraints: [`species:${species.id}`, ...(ev.shown === null ? [] : [`forme:${ev.shown}`]), ...(ev.level === null ? [] : [`level:${ev.level}`]), ...(ev.gender === null ? [] : [`gender:${ev.gender}`]), ...(ev.mega ? [`mega:${ev.megaStone ?? "unknown"}`] : []), ...(isSettled ? ["settled"] : []), ...(notes[i] as string[])],
    };
  });

  return {
    format: BELIEF_FORMAT,
    slots,
    itemClause: { enforced: clause === 1, normaliser: Z },
    brought: { picked, settled, subsets: subsetCount(mons.length, picked, settled.length) },
    constraints: [clause === 1 ? "item_clause:1" : "item_clause:absent", `brought:picked=${picked}`],
  };
}
