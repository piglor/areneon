// `sample` and `sampleWorlds` (ADR-025 section 4): whole opponent teams drawn from the belief, exactly.
//
// A world is a function of the belief and a 32-byte stream alone. The draw order is fixed: the disguise
// assignment, the brought set, the Item Clause DP backward pass (an item or none for every slot, brought
// or not, because the rule holds across all six), then for each brought slot in preview order its forme,
// ability, spread and moves. Every integer below a total comes from `BitStream.below` (exact rejection
// sampling over L = bitlen(T) + 64 bits; no float, no reduced weight). The tables are built once per
// belief and reused by every draw.
//
// Legality is the target format's own validator, behind the `SetLegality` port the composition root
// supplies (the bridge implements it with `validateSet` and the canonical paste); the package is
// engine-free. A world with an illegal set is redrawn from the next counter block, at most
// `max_legality_redraws` times, and the redraw count is returned. The prior's "other spread" entry has
// no nature and Stat Points of its own: the `SpreadSampler` port (implemented beside the generator,
// bound by `generator_commit` in a generator-matched prior) draws one from the same stream.

import { BeliefError } from "./error.ts";
import { type BlockSource, BitStream, blocksOf, worldStream } from "./draw.ts";
import { parseSpreadKey } from "./fields.ts";
import { type ClauseTables, clauseTables } from "./item-clause.ts";
import type { MoveModel } from "./moves.ts";
import { type Ratio, integerise, ratio } from "./ratio.ts";
import type { Cell, FormeBelief, OpponentBelief } from "./types.ts";

export const STAT_IDS = ["hp", "atk", "def", "spa", "spd", "spe"] as const;
export type StatPoints = Readonly<Record<(typeof STAT_IDS)[number], number>>;

/** What the channel shows of a Pokemon: its nickname, the forme it is shown as, its level and gender. */
export interface ShownIdentifiers {
  readonly name: string | null;
  readonly species: string | null;
  readonly level: number | null;
  readonly gender: string | null;
}

/** One brought Pokemon's set. Ids are the vocabulary's; `forme` is the Mega forme for a Mega slot (its item is the stone). */
export interface WorldSet {
  /** The preview species id. */
  readonly species: string;
  readonly forme: string;
  /** An item id, or null for no item. */
  readonly item: string | null;
  /** The pre-Evolution ability for a Mega forme not yet evolved. */
  readonly ability: string;
  readonly nature: string;
  readonly stat_points: StatPoints;
  /** Up to four move ids, sorted. */
  readonly moves: readonly string[];
  readonly level: number;
  readonly gender: string | null;
  readonly shown: ShownIdentifiers;
}

export interface WorldSlot {
  /** The 0-based position in the opponent's preview roster. */
  readonly index: number;
  /** Exactly `picked_team_size` slots are brought. */
  readonly brought: boolean;
  /** The set of a brought slot, null otherwise. */
  readonly set: WorldSet | null;
}

export interface OpponentWorld {
  readonly slots: readonly WorldSlot[];
}

/** The target format's judgement of one set: the pinned validator takes it and it is the engine's canonical paste. */
export interface SetLegality {
  legal(set: WorldSet): boolean;
}

/** Draws the nature and Stat Points of the prior's "other spread" entry from the world's stream (`below(T)` is uniform in [0, T)). */
export interface SpreadSampler {
  other(species: string, forme: string, below: (total: bigint) => bigint): { readonly nature: string; readonly stat_points: StatPoints };
}

export interface SamplerPorts {
  readonly legality: SetLegality;
  /** Needed only when a sampled spread can be the prior's "other spread" entry. */
  readonly spreads?: SpreadSampler;
}

export interface SampledWorld {
  readonly world: OpponentWorld;
  /** How many times the world was redrawn because a set was illegal. */
  readonly redraws: number;
}

export interface SampledWorlds {
  /** World i is `sample(belief, w_i)`; each weighs 1/n. A consumer that redraws asks for further indices. */
  readonly worlds: readonly (SampledWorld & { readonly index: number; readonly weight: Ratio })[];
  /** The redraws summed over the worlds. */
  readonly redraws: number;
}

export interface Sampler {
  sample(belief: OpponentBelief, stream: Uint8Array): SampledWorld;
  sampleWorlds(belief: OpponentBelief, stream: Uint8Array, n: number): SampledWorlds;
}

interface PreparedAssignment {
  readonly itemIds: readonly string[];
  /** at[i][k]: the weight of slot i holding item k, summed over its cells. */
  readonly at: readonly (readonly bigint[])[];
  readonly tables: ClauseTables | null;
}

interface Prepared {
  readonly assignments: readonly PreparedAssignment[];
}

const cache = new WeakMap<OpponentBelief, Prepared>();

/** The tables of a belief, built once (one per disguise assignment) and kept for every later draw. */
export function prepared(b: OpponentBelief): Prepared {
  const hit = cache.get(b);
  if (hit !== undefined) return hit;
  const built: Prepared = {
    assignments: b.assignments.map((a): PreparedAssignment => {
      const itemIds = [...new Set(a.slots.flatMap((s) => s.cells.flatMap((c) => (c.item === null ? [] : [c.item]))))].sort();
      const sum = (cells: readonly Cell[], item: string | null): bigint => cells.filter((c) => c.item === item).reduce((t, c) => t + c.weight, 0n);
      const at = a.slots.map((s) => itemIds.map((id) => sum(s.cells, id)));
      const none = a.slots.map((s) => sum(s.cells, null));
      return { itemIds, at, tables: b.itemClause.enforced ? clauseTables({ items: at, none }) : null };
    }),
  };
  cache.set(b, built);
  return built;
}

interface MoveTables {
  readonly rest: readonly { readonly id: string; readonly odds: bigint }[];
  /** suffix[j][m]: the sum over m-subsets of rest[j..] of the product of their odds. */
  readonly suffix: readonly (readonly bigint[])[];
}

const moveTables = new WeakMap<MoveModel, MoveTables>();

function movesOf(model: MoveModel): MoveTables {
  const hit = moveTables.get(model);
  if (hit !== undefined) return hit;
  const rest = model.candidates.filter((c) => !model.revealed.includes(c.id));
  const degree = Math.max(...model.outcomes.map((o) => o.size)) - model.forced.length - model.revealed.length;
  const suffix: bigint[][] = [[1n, ...Array.from({ length: degree }, () => 0n)]];
  for (let j = rest.length - 1; j >= 0; j -= 1) {
    const after = suffix[0] as bigint[];
    suffix.unshift(after.map((value, m) => value + (m === 0 ? 0n : (rest[j] as { odds: bigint }).odds * (after[m - 1] as bigint))));
  }
  const built = { rest, suffix };
  moveTables.set(model, built);
  return built;
}

/** The moves of one set: the count from the posterior, then the conditional Poisson design over the open candidates, sequentially and exactly. */
function drawMoves(model: MoveModel, bits: BitStream): string[] {
  const { weights } = integerise(model.outcomes.map((o) => o.weight));
  const size = (model.outcomes[bits.choose(weights)] as { size: number }).size;
  let need = size - model.forced.length - model.revealed.length;
  const { rest, suffix } = movesOf(model);
  const drawn: string[] = [];
  for (let j = 0; j < rest.length && need > 0; j += 1) {
    const total = (suffix[j] as bigint[])[need] as bigint;
    const include = (rest[j] as { odds: bigint }).odds * ((suffix[j + 1] as bigint[])[need - 1] as bigint);
    if (bits.choose([total - include, include]) === 1) {
      drawn.push((rest[j] as { id: string }).id);
      need -= 1;
    }
  }
  return [...new Set([...model.forced, ...model.revealed, ...drawn])].sort();
}

const pickOption = <T>(options: readonly { readonly id: T; readonly weight: bigint }[], bits: BitStream): T => (options[bits.choose(options.map((o) => o.weight))] as { id: T }).id;

function draw(b: OpponentBelief, tables: Prepared, ports: SamplerPorts, bits: BitStream): OpponentWorld {
  // 1. The disguise assignment.
  const which = bits.choose(b.assignments.map((a) => a.weight));
  const a = b.assignments[which] as OpponentBelief["assignments"][number];
  const prep = tables.assignments[which] as PreparedAssignment;

  // 2. The brought set: the slots every consistent world holds, then a uniform draw of the rest.
  const n = a.slots.length;
  const brought = new Set(a.required);
  const free = a.slots.map((_, i) => i).filter((i) => !brought.has(i));
  for (let need = b.brought.picked - brought.size; need > 0; need -= 1) {
    const [pick] = free.splice(Number(bits.below(BigInt(free.length))), 1);
    brought.add(pick as number);
  }

  // 3. The Item Clause DP backward pass: which slot holds each item (none for the rest).
  const itemOf: (string | null)[] = a.slots.map(() => null);
  if (prep.tables !== null) {
    const backward = prep.tables.backward;
    let open = (1 << n) - 1;
    prep.itemIds.forEach((id, k) => {
      const after = backward[k + 1] as readonly bigint[];
      const slots: number[] = [];
      const weights: bigint[] = [after[open] as bigint];
      for (let i = 0; i < n; i += 1) {
        const w = (prep.at[i] as readonly bigint[])[k] as bigint;
        if (open & (1 << i) && w > 0n) {
          slots.push(i);
          weights.push(w * (after[open & ~(1 << i)] as bigint));
        }
      }
      const pick = bits.choose(weights);
      if (pick > 0) {
        const slot = slots[pick - 1] as number;
        itemOf[slot] = id;
        open &= ~(1 << slot);
      }
    });
  }

  // 4. Each brought slot in preview order: forme (with its item), ability, spread, moves.
  const slots = a.slots.map((slot, i): WorldSlot => {
    if (!brought.has(i)) return { index: i, brought: false, set: null };
    const cells = prep.tables === null ? slot.cells : slot.cells.filter((c) => c.item === itemOf[i]);
    const cell = cells[bits.choose(cells.map((c) => c.weight))] as Cell;
    const forme = slot.formes.find((f) => f.id === cell.forme) as FormeBelief;
    const ability = pickOption(forme.abilities.options, bits);
    const key = pickOption(forme.spreads.options, bits);
    const known = parseSpreadKey(key);
    if (known === null && ports.spreads === undefined) throw new BeliefError("spread_sampler_required", "a sampled spread is the prior's other-spread entry and no SpreadSampler was given", i);
    const spread = known ?? (ports.spreads as SpreadSampler).other(slot.species, forme.id, (total) => bits.below(total));
    const shown = (b.slots[i] as OpponentBelief["slots"][number]).shown;
    return {
      index: i,
      brought: true,
      set: {
        species: slot.species,
        forme: forme.id,
        item: cell.item,
        ability,
        nature: spread.nature,
        stat_points: spread.stat_points,
        moves: drawMoves(forme.moves, bits),
        level: shown.level ?? b.sampling.level,
        gender: shown.gender,
        shown: { name: shown.name, species: shown.species, level: shown.level, gender: shown.gender },
      },
    };
  });
  return { slots };
}

function run(b: OpponentBelief, source: BlockSource, ports: SamplerPorts): SampledWorld {
  const tables = prepared(b);
  const bits = new BitStream(source);
  for (let redraws = 0; ; redraws += 1) {
    const world = draw(b, tables, ports, bits);
    if (world.slots.every((s) => s.set === null || ports.legality.legal(s.set))) return { world, redraws };
    if (redraws >= b.sampling.max_legality_redraws) throw new BeliefError("legality_redraws_exhausted", "every redraw of the world held a set the target format's validator refuses");
    bits.restart();
  }
}

/** A sampler bound to the legality port (and, for priors with an "other spread" entry, the spread port). */
export function createSampler(ports: SamplerPorts): Sampler {
  const sample = (belief: OpponentBelief, stream: Uint8Array): SampledWorld => run(belief, blocksOf(stream), ports);
  return {
    sample,
    sampleWorlds(belief, stream, n) {
      if (!Number.isSafeInteger(n) || n < 1) throw new RangeError("the number of worlds is a positive integer");
      const worlds = Array.from({ length: n }, (_, index) => ({ index, weight: ratio(1n, BigInt(n)), ...sample(belief, worldStream(stream, index)) }));
      return { worlds, redraws: worlds.reduce((t, w) => t + w.redraws, 0) };
    },
  };
}
