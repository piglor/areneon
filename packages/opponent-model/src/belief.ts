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
import { enumerate, hasMatching } from "./disguise.ts";
import { type SlotEvidence, slotEvidence, trapEvidence } from "./evidence.ts";
import { type Dist, abilityDist, itemDist, only, spreadDist, uniformDist, weightOf } from "./fields.ts";
import { solveItemClause } from "./item-clause.ts";
import { moveModel } from "./moves.ts";
import { type TrapSlot, trapExclusions } from "./trapping.ts";
import { type Ratio, ONE, ZERO, add, greater, integerise, isZero, lcm, mul, ratio } from "./ratio.ts";
import { BELIEF_FORMAT, type BeliefInput, type BeliefVocabulary, type Cell, type DisguiseAssignment, type FormeBelief, type Marginal, type OpponentBelief, type SlotBelief, type SlotMarginal } from "./types.ts";

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
function slotWork(species: PriorSpecies, ev: SlotEvidence, ctx: Context, slot: number, notes: string[], known: ReadonlySet<string> | null, trap: ReadonlySet<string> | null): FormeWork[] {
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
  // Under a possible disguise a real move that no consistent forme can hold belongs to another member (Struggle, in no pool, still adds nothing).
  if (known !== null && ev.moves.some((m) => !placeable.includes(m) && known.has(m))) {
    throw new BeliefError("evidence_has_no_support", "a revealed move is in the vocabulary but no forme of the slot can hold it", slot);
  }
  for (const m of ev.moves) notes.push(placeable.includes(m) ? `move:${m}` : `move_skipped:${m}`);
  if (ev.transformed) notes.push("moves_skipped:transform");

  // An ability the Pokemon showed itself that the own request's flags rule out: the flags and the reveal cannot both hold.
  if (trap !== null && ev.ability !== null && trap.has(ev.ability)) throw new BeliefError("evidence_has_no_support", "the revealed ability contradicts the own request's trapping flags", slot);

  // The ability list in force per forme, and what the revealed ability says about each.
  const abilityOf = formes.map((f) => {
    const entry = entries.get(f.id) as RosterEntry;
    const mega = isMega(species, f, entry);
    const evolved = mega && (ev.mega || ev.shown === f.id);
    const listed = mega && !evolved ? (f.holder_abilities === undefined ? baseAbilities : abilityDist(f.holder_abilities.entries, f.holder_abilities.residual, speciesEntry.abilities)) : abilityDist(f.abilities.entries, f.abilities.residual, entry.abilities);
    // The own request's trapping flags: abilities the engine says the shown foe cannot hold get weight 0 (the forme's weight is scaled by the share left).
    const dist = trap === null ? listed : restrict(listed, trap);
    const verdict = ev.ability === null ? "free" : evolved && preAbilities.has(ev.ability) ? "ambiguous" : weightOf(dist, ev.ability) > 0n ? "fixed" : "excluded";
    return { f, mega, dist, verdict, share: dist.total === listed.total ? ONE : ratio(dist.total, listed.total) };
  });
  const usable = ev.ability !== null && abilityOf.some((a) => a.verdict !== "excluded");
  if (ev.ability !== null) {
    const kinds = abilityOf.map((a) => a.verdict);
    notes.push(!usable ? `ability_skipped:${ev.ability}` : kinds.includes("fixed") ? `ability:${ev.ability}` : `ability_ambiguous:${ev.ability}`);
  }
  const itemUsable = ev.item !== null && ctx.items.has(ev.item);
  if (ev.item !== null) notes.push(itemUsable ? `item:${ev.item}` : `item_skipped:${ev.item}`);

  const work: FormeWork[] = [];
  for (const { f, mega, dist, verdict, share } of abilityOf) {
    const entry = entries.get(f.id) as RosterEntry;
    const pool = poolOf(f.id);
    if (dist.total === 0n) continue;
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
    const weight = mul(mul(mul(ratio(BigInt(f.weight), BigInt(species.denominator)), abilityLikelihood), share), model.likelihood);
    work.push({ id: f.id, mega, weight, abilities, spreads, moves: model.model, items });
  }
  if (work.length === 0) throw new BeliefError("evidence_has_no_support", "the evidence leaves no forme of the slot with a positive weight", slot);
  return work;
}

/** The options of a distribution that are not in `excluded`. */
function restrict(dist: Dist<string>, excluded: ReadonlySet<string>): Dist<string> {
  const options = dist.options.filter((o) => !excluded.has(o.id));
  return options.length === dist.options.length ? dist : { total: options.reduce((sum, o) => sum + o.weight, 0n), options };
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

interface Chosen {
  readonly species: PriorSpecies;
  readonly ev: SlotEvidence;
  /** The species (or one of its formes) has an identity-hiding ability in the vocabulary. */
  readonly hides: boolean;
  /** Abilities the own request's trapping flags exclude for this slot, null when none. */
  readonly trap: ReadonlySet<string> | null;
  /** The fixed-word notes of those exclusions. */
  readonly trapNotes: readonly string[];
}

/** One slot's evidence work, or the refusal it led to, per (slot, evidence). */
type Memo = Map<string, { readonly work: FormeWork[]; readonly notes: readonly string[] } | BeliefError>;

interface Solved {
  readonly slots: SlotBelief[];
  /** The Item Clause DP total over the integer cells. */
  readonly normaliser: bigint;
  /** The product of the slots' integerising scales: the joint weight of the assignment is normaliser / scale. */
  readonly scale: bigint;
}

/**
 * The evidence of every slot when the shown positions `shown` are assigned to the slots `target`
 * (aligned). The shown forme, level and gender stay with the position (they are the details of the
 * Pokemon it is shown as); the moves, item, ability, Transform and Mega evidence go to the assigned slot.
 */
function attribute(own: readonly SlotEvidence[], shown: readonly number[], target: readonly number[]): { evs: SlotEvidence[]; notes: string[][] } {
  const source = new Map(target.map((slot, k) => [slot, shown[k] as number]));
  const notes: string[][] = own.map(() => []);
  const evs = own.map((e, s): SlotEvidence => {
    const from = source.get(s);
    if (from === s) return e;
    if (from !== undefined) {
      const f = own[from] as SlotEvidence;
      (notes[s] as string[]).push(`disguise_of:${from}`);
      return { ...e, settled: true, item: f.item, ability: f.ability, moves: f.moves, transformed: f.transformed, mega: f.mega, megaStone: f.megaStone };
    }
    if (!e.settled) return e;
    (notes[s] as string[]).push(`disguised_by:${target[shown.indexOf(s)] as number}`);
    return { ...e, settled: false, item: null, ability: null, moves: [], transformed: false, mega: false, megaStone: null };
  });
  return { evs, notes };
}

function solve(ctx: Context, chosen: readonly Chosen[], memo: Memo, known: ReadonlySet<string> | null, evs: readonly SlotEvidence[], extra: readonly (readonly string[])[], required: ReadonlySet<number>, picked: number, clause: boolean): Solved | null {
  const n = chosen.length;
  const got = evs.map((ev, i) => workOf(chosen, ctx, memo, known, i, ev));
  const works = got.map((g) => g.work);

  // Each slot's cells on one integer scale, then the Item Clause DP over all slots.
  let scale = 1n;
  const cells = works.map((work) => {
    const raw = work.flatMap((w) => [...w.items].map(([item, p]) => ({ forme: w.id, item, p: mul(w.weight, p) }))).filter((c) => !isZero(c.p));
    const integers = integerise(raw.map((c) => c.p));
    scale *= integers.scale;
    return raw.map((c, k): Cell => ({ forme: c.forme, item: c.item, weight: integers.weights[k] as bigint }));
  });
  const itemIds = [...new Set(cells.flatMap((slot) => slot.flatMap((c) => (c.item === null ? [] : [c.item]))))].sort();
  const weight = (slot: readonly Cell[], item: string | null): bigint => slot.filter((c) => c.item === item).reduce((s, c) => s + c.weight, 0n);
  const solved = solveItemClause({ items: cells.map((slot) => itemIds.map((id) => weight(slot, id))), none: cells.map((slot) => weight(slot, null)) }, clause);
  const Z = solved.normaliser;
  if (Z === 0n) return null;

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
    const ev = evs[i] as SlotEvidence;
    const species = (chosen[i] as Chosen).species;
    const isSettled = required.has(i);
    return {
      index: i,
      species: species.id,
      shown: { name: ev.name, species: ev.shown, level: ev.level, gender: ev.gender },
      settled: isSettled,
      p_brought: isSettled ? ONE : ratio(BigInt(picked - required.size), BigInt(n - required.size)),
      formes,
      abilities,
      items,
      moves,
      spreads,
      mode: { forme: mode(formes.map((f) => ({ id: f.id, p: f.p }))), ability: mode(abilities), item: mode(items) },
      cells: slotCells,
      constraints: [`species:${species.id}`, ...(ev.shown === null ? [] : [`forme:${ev.shown}`]), ...(ev.level === null ? [] : [`level:${ev.level}`]), ...(ev.gender === null ? [] : [`gender:${ev.gender}`]), ...(ev.mega ? [`mega:${ev.megaStone ?? "unknown"}`] : []), ...(isSettled ? ["settled"] : []), ...(extra[i] as readonly string[]), ...(got[i] as { notes: readonly string[] }).notes],
    };
  });
  return { slots, normaliser: Z, scale };
}

function workOf(chosen: readonly Chosen[], ctx: Context, memo: Memo, known: ReadonlySet<string> | null, i: number, ev: SlotEvidence): { work: FormeWork[]; notes: readonly string[] } {
  const trap = (chosen[i] as Chosen).trap;
  const key = `${i}|${JSON.stringify({ ...ev, settled: false })}|${trap === null ? "" : [...trap].sort().join(",")}`;
  let hit = memo.get(key);
  if (hit === undefined) {
    const notes: string[] = [];
    try {
      notes.push(...(chosen[i] as Chosen).trapNotes);
      hit = { work: slotWork((chosen[i] as Chosen).species, ev, ctx, i, notes, known, trap), notes };
    } catch (e) {
      if (!(e instanceof BeliefError)) throw e;
      hit = e;
    }
    memo.set(key, hit);
  }
  if (hit instanceof BeliefError) throw hit;
  return hit;
}

/** The belief mixed over the assignments: each marginal is the sum of the assignments' marginals weighted by p. */
function mixSlot(i: number, list: readonly DisguiseAssignment[]): SlotMarginal {
  const slotsOf = list.map((a) => a.slots[i] as SlotBelief);
  const first = slotsOf[0] as SlotBelief;
  const fold = <K extends string | null>(pick: (s: SlotBelief) => readonly Marginal<K>[]): Marginal<K>[] => {
    const out = new Map<K, Ratio>();
    slotsOf.forEach((s, k) => {
      for (const m of pick(s)) out.set(m.id, add(out.get(m.id) ?? ZERO, mul((list[k] as DisguiseAssignment).p, m.p)));
    });
    return marginalList(out);
  };
  const formeP = new Map<string, Ratio>();
  const megaOf = new Map<string, boolean>();
  slotsOf.forEach((s, k) => {
    for (const f of s.formes) {
      formeP.set(f.id, add(formeP.get(f.id) ?? ZERO, mul((list[k] as DisguiseAssignment).p, f.p)));
      megaOf.set(f.id, f.mega);
    }
  });
  const formes = [...formeP].map(([id, p]) => ({ id, mega: megaOf.get(id) as boolean, p })).sort((a, b) => compare(a.id, b.id));
  const abilities = fold((s) => s.abilities);
  const items = fold((s) => s.items);
  const common = first.constraints.filter((c) => slotsOf.every((s) => s.constraints.includes(c)));
  return {
    index: i,
    species: first.species,
    shown: first.shown,
    settled: slotsOf.every((s) => s.settled),
    p_brought: slotsOf.reduce((sum, s, k) => add(sum, mul((list[k] as DisguiseAssignment).p, s.p_brought)), ZERO),
    abilities,
    items,
    moves: fold((s) => s.moves),
    spreads: fold((s) => s.spreads),
    mode: { forme: mode(formes), ability: mode(abilities), item: mode(items) },
    formes,
    constraints: common,
  };
}

export function belief(input: BeliefInput): OpponentBelief {
  const { prior, vocabulary, config, opponent, flags, trapping } = input;
  if (flags.length > 0) throw new BeliefError("reader_flagged", "the reader flagged a line of this channel");
  if (prior.format_id !== vocabulary.format_id || prior.engine_data_hash !== vocabulary.engine_data_hash) {
    throw new BeliefError("format_mismatch", "the usage prior is for another format or engine data than the vocabulary");
  }
  if (prior.vocabulary_hash !== vocabulary.vocabulary_hash) {
    throw new BeliefError("format_mismatch", "the usage prior was derived against another team vocabulary than this one");
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

  let chosen = mons.map((mon, slot): Chosen => {
    const species = speciesById.get(mon.base_species) ?? (mon.species === null ? undefined : speciesById.get(mon.species));
    if (species === undefined) throw new BeliefError("species_not_in_prior", "a previewed species has no entry in the usage prior", slot);
    const ids = [species.id, ...species.formes.map((f) => f.id)];
    return { species, ev: slotEvidence(mon), hides: ids.some((id) => ctx.roster.get(id)?.abilities.some((a) => hiding.has(a))), trap: null, trapNotes: [] };
  });

  // The shown positions (in play, so in the brought set if they are what they seem) and the hiding members still unrevealed.
  const own = chosen.map((c) => c.ev);
  const shown = own.flatMap((e, i) => (e.settled ? [i] : []));
  if (shown.length > picked) throw new BeliefError("brought_exceeds_picked", "more Pokemon have been in play than the picked team size");
  const hidingSlots = chosen.flatMap((c, i) => (c.hides ? [i] : []));
  const unsettled = hidingSlots.filter((i) => !(own[i] as SlotEvidence).settled);
  const mixed = unsettled.length > 0 && shown.length > 0;

  // The own request's trapping flags (ticket #1127). Not under a disguise mixture: the shown foe may be another member.
  let trapConstraints: readonly string[] = [];
  if (trapping != null && trapping.observed.length > 0) {
    const found = trapExclusions(trapping, mons.map((mon, index): TrapSlot => ({ index, ...trapEvidence(mon, ctx.roster, !mixed) })));
    trapConstraints = found.constraints;
    chosen = chosen.map((c, i) => ({ ...c, trap: found.excluded.get(i) ?? null, trapNotes: found.notes.get(i) ?? [] }));
  }

  const memo: Memo = new Map();
  const known = mixed ? new Set(vocabulary.roster.flatMap((r) => r.moves)) : null;

  // Which slot each shown position may belong to: itself, or an unsettled hiding member, when the evidence can sit there.
  let targets: number[][] = [shown];
  if (mixed) {
    const options = shown.map((p) => [p, ...unsettled].filter((s) => {
      try {
        const { evs } = attribute(own, [p], [s]);
        workOf(chosen, ctx, memo, known, s, evs[s] as SlotEvidence);
        return true;
      } catch (e) {
        if (!(e instanceof BeliefError)) throw e;
        return false;
      }
    }));
    if (!hasMatching(options)) throw new BeliefError("evidence_has_no_support", "no assignment of the shown Pokemon to the preview slots explains the evidence");
    const list = enumerate(options, new Set(unsettled), config.max_disguise_assignments);
    if (list === null) throw new BeliefError("disguise_assignments_exceeded", "more disguise assignments than the reviewed bound");
    targets = list;
  }

  const solved: { target: readonly number[]; result: Solved; extra: readonly (readonly string[])[] }[] = [];
  let firstError: BeliefError | null = null;
  for (const target of targets) {
    const { evs, notes } = attribute(own, shown, target);
    try {
      const result = solve(ctx, chosen, memo, known, evs, notes, new Set(target), picked, clause === 1);
      if (result !== null) solved.push({ target, result, extra: notes });
    } catch (e) {
      // With one candidate the refusal is the belief's; among several, an assignment the evidence rules out is dropped.
      if (!(e instanceof BeliefError) || !mixed || (e.code !== "evidence_has_no_support" && e.code !== "forme_not_in_prior")) throw e;
      firstError ??= e;
    }
  }
  if (solved.length === 0) {
    throw firstError ?? new BeliefError("evidence_has_no_support", "the evidence and the Item Clause leave no world with a positive weight");
  }

  // Integer weights on one scale: an assignment's joint weight is normaliser / scale, so multiply by lcm(scales) / scale.
  const common = solved.reduce((l, s) => lcm(l, s.result.scale), 1n);
  const weights = solved.map((s) => s.result.normaliser * (common / s.result.scale));
  const total = weights.reduce((a, b) => a + b, 0n);
  const assignments: DisguiseAssignment[] = solved.map((s, k) => ({
    moved: shown.flatMap((position, j) => ((s.target[j] as number) === position ? [] : [{ position, slot: s.target[j] as number }])),
    weight: weights[k] as bigint,
    p: ratio(weights[k] as bigint, total),
    normaliser: s.result.normaliser,
    required: [...s.target].sort((a, b) => a - b),
    slots: s.result.slots,
  }));

  const certain = assignments.reduce<number[]>((acc, a) => acc.filter((i) => a.required.includes(i)), assignments[0]?.required.slice() ?? []);
  return {
    format: BELIEF_FORMAT,
    slots: mons.map((_, i) => mixSlot(i, assignments)),
    assignments,
    disguise: { hiding: hidingSlots, unsettled, positions: mixed ? shown : [], total },
    itemClause: { enforced: clause === 1 },
    brought: { picked, settled: certain, subsets: subsetCount(mons.length, picked, shown.length) },
    sampling: { level: vocabulary.rules.adjust_level, max_legality_redraws: config.max_legality_redraws },
    constraints: [clause === 1 ? "item_clause:1" : "item_clause:absent", `brought:picked=${picked}`, ...trapConstraints, ...(assignments.length > 1 ? [`disguise:assignments=${assignments.length}`] : [])],
  };
}
