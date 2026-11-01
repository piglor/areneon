// The opponent sheet block of the closed regime (ADR-027 section 2, ADR-025 erratum E4 point 1): what the
// opponent belief makes probable about each opponent preview slot, as the integers of the `OpponentSheet`
// contract. A pure function of the belief at the request, of the reader's opponent half that produced it
// (only to tell a REVEALED fact from a predicted one) and of the reviewed `k` values: no sampled world, seed,
// `topWorlds`, real team or oracle is an input (`test/sheet.test.ts` holds the type test).
//
// Rounding: every ppm is the exact rational times 10^6, rounded half up, each entry ALONE (so the entries of a
// field may sum to a few ppm above 10^6); `other_ppm` is max(0, 10^6 - the sum of the kept entries).
// Entries are sorted by ppm descending then id ascending (a null item id, "holds no item", last) and a rounded
// zero is dropped before the cut to the top `k`.
//
// Sources: `revealed` only when the reader shows the fact AND the belief has it at probability 1 (an item or
// ability the Pokemon received is no fact of its set; a Mega Stone is the item of an evolved Mega); `predicted`
// otherwise; `unknown` for a nature field that no named spread backs. While an identity-hiding member is unsettled
// and a position is shown, the whole block is `unknown` with the fixed refusal `disguise_unsettled`: the exact
// mixture of the belief says where the evidence goes, but a position's identity is what the policy would need to
// read the block against, and the observation cannot carry it.

import type { PublicSide } from "@areneon/public-state";
import type { OpponentSheet, SheetFieldSource, SheetIdEntry, SheetIdField, SheetItemEntry, SheetItemField, SheetMoveEntry, SheetMoveField, SheetSlot } from "@areneon/schema/types/v1/opponent-sheet.ts";
import { digest } from "./canonical.ts";
import { slotEvidence } from "./evidence.ts";
import { parseSpreadKey } from "./fields.ts";
import { add, type Ratio } from "./ratio.ts";
import type { Marginal, OpponentBelief } from "./types.ts";
import { BeliefError, type BeliefErrorCode } from "./error.ts";

/** Bumped when the builder computes a value differently (rounding, ordering, sources, the cut). Part of the feature set hash. */
export const OPPONENT_SHEET_FORMAT = "opponent-sheet-builder/1";

/** The feature set the block belongs to; the block's `source.hash` is its hash. */
export const SHEET_FEATURE_SET = "opponent-sheet/1";

/** The reviewed `k` values of the first feature set (E4 point 1). */
export const SHEET_K = { item: 4, ability: 3, nature: 3, moves: 8 } as const;

export interface SheetK {
  readonly item: number;
  readonly ability: number;
  readonly nature: number;
  readonly moves: number;
}

const PPM = 1_000_000n;

/** The fixed codes of a refused block: the belief's own codes a consumer falls back on, and the sheet's `disguise_unsettled`. */
export const SHEET_REFUSAL_CODES = [
  "reader_flagged",
  "team_size",
  "species_not_in_prior",
  "forme_not_in_prior",
  "brought_exceeds_picked",
  "disguise_assignments_exceeded",
  "evidence_has_no_support",
  "legality_redraws_exhausted",
  "empty_distribution",
  "disguise_unsettled",
] as const;

export type SheetRefusalCode = (typeof SHEET_REFUSAL_CODES)[number];

/** The belief codes that say the run is set up wrongly (they propagate, never a refused block). */
export const SHEET_DEFECT_CODES: ReadonlySet<BeliefErrorCode> = new Set<BeliefErrorCode>(["format_mismatch", "item_clause_unsupported", "spread_sampler_required"]);

/** The refusal code of a belief error that becomes a refused block, else null (a defect). */
export const sheetRefusalOf = (error: unknown): SheetRefusalCode | null => {
  if (!(error instanceof BeliefError)) return null;
  return (SHEET_REFUSAL_CODES as readonly string[]).includes(error.code) ? (error.code as SheetRefusalCode) : null;
};

/** round half up of `r * 10^6`, exact (`r` is non-negative). */
export const roundPpm = (r: Ratio): number => Number((2n * r.num * PPM + r.den) / (2n * r.den));

/** SHA-256 of the JCS of the feature set's identity: the belief identity, the `k` values, the rounding rule and the builder format. */
export function sheetFeatureSetHash(belief_identity_hash: string, k: SheetK = SHEET_K): string {
  if (!/^[0-9a-f]{64}$/.test(belief_identity_hash)) throw new RangeError("belief_identity_hash is a SHA-256 in lower-case hexadecimal");
  return digest({ feature_set: SHEET_FEATURE_SET, opponent_sheet_format: OPPONENT_SHEET_FORMAT, belief_identity_hash, k_item: k.item, k_ability: k.ability, k_nature: k.nature, k_moves: k.moves, rounding: "half-up" });
}

const byPpm = <T extends string | null>(a: { id: T; v: number }, b: { id: T; v: number }): number => b.v - a.v || (a.id === b.id ? 0 : a.id === null ? 1 : b.id === null ? -1 : a.id < b.id ? -1 : 1);

/** The entries of a field: rounded alone, zeros dropped, sorted, cut to `k`. */
function top<T extends string | null>(marginals: readonly { id: T; p: Ratio }[], k: number): { id: T; v: number }[] {
  return marginals
    .map((m) => ({ id: m.id, v: roundPpm(m.p) }))
    .filter((m) => m.v > 0)
    .sort(byPpm)
    .slice(0, k);
}

const otherOf = (entries: readonly { v: number }[]): number => Math.max(0, Number(PPM) - entries.reduce((s, e) => s + e.v, 0));

const certain = (p: Ratio): boolean => p.num === p.den;

/** A fact is revealed when the reader shows it and the belief has it, alone among the options with a positive probability, at probability exactly 1. */
const isRevealed = (marginals: readonly { id: string | null; p: Ratio }[], fact: string | null): boolean => {
  const live = marginals.filter((m) => m.p.num !== 0n);
  const only = live[0];
  return fact !== null && live.length === 1 && only !== undefined && only.id === fact && certain(only.p);
};

function idField(marginals: readonly { id: string; p: Ratio }[], k: number, revealed: string | null): SheetIdField {
  const kept = top(marginals, k);
  if (kept.length === 0) return { source: "unknown", entries: [], other_ppm: 0 };
  const entries: SheetIdEntry[] = kept.map((e) => ({ id: e.id, ppm: e.v }));
  const source: SheetFieldSource = isRevealed(marginals, revealed) ? "revealed" : "predicted";
  return { source, entries, other_ppm: otherOf(kept) };
}

function itemField(marginals: readonly Marginal<string | null>[], k: number, revealed: string | null): SheetItemField {
  const kept = top(marginals, k);
  if (kept.length === 0) return { source: "unknown", entries: [], other_ppm: 0 };
  const entries: SheetItemEntry[] = kept.map((e) => ({ id: e.id, ppm: e.v }));
  const source: SheetFieldSource = isRevealed(marginals, revealed) ? "revealed" : "predicted";
  return { source, entries, other_ppm: otherOf(kept) };
}

/** The nature marginal of a slot: the named spreads' natures summed; the "other spread" entry names no nature and is the rest. */
function natureMarginals(spreads: readonly Marginal<string>[]): { id: string; p: Ratio }[] {
  const sums = new Map<string, Ratio>();
  for (const s of spreads) {
    const parsed = parseSpreadKey(s.id);
    if (parsed === null) continue;
    const had = sums.get(parsed.nature);
    sums.set(parsed.nature, had === undefined ? s.p : add(had, s.p));
  }
  return [...sums].map(([id, p]) => ({ id, p }));
}

function moveField(marginals: readonly Marginal<string>[], k: number, revealedMoves: readonly string[]): SheetMoveField {
  const kept = top(marginals, k);
  const known = Math.min(4, revealedMoves.length);
  if (kept.length === 0) return { source: "unknown", entries: [], slots_known: 0 };
  const entries: SheetMoveEntry[] = kept.map((e) => ({ id: e.id, inclusion_ppm: e.v }));
  const source: SheetFieldSource = known === 4 ? "revealed" : "predicted";
  return { source, entries, slots_known: known };
}

/** The block of a refused belief: every field unknown, with the fixed code. `species` is each preview entry's species id. */
export function refusedSheet(opponent: PublicSide, code: SheetRefusalCode, hash: string): OpponentSheet {
  const slots = opponent.roster.map(
    (mon): SheetSlot => ({
      species: mon.base_species,
      brought_ppm: 0,
      brought_source: "unknown",
      item: { source: "unknown", entries: [], other_ppm: 0 },
      ability: { source: "unknown", entries: [], other_ppm: 0 },
      nature: { source: "unknown", entries: [], other_ppm: 0 },
      moves: { source: "unknown", entries: [], slots_known: 0 },
    }),
  );
  return { source: { kind: "belief", hash, refusal: code }, slots: slots as OpponentSheet["slots"] };
}

/**
 * The closed-regime block of the belief at a request. `opponent` is the reader's opponent half the belief was computed from.
 * Throws a `RangeError` for a belief that does not belong to that roster; returns a refused block for an unsettled
 * identity-hiding member with a shown position (`disguise_unsettled`).
 */
export function opponentSheet(belief: OpponentBelief, opponent: PublicSide, hash: string, k: SheetK = SHEET_K): OpponentSheet {
  if (belief.slots.length !== opponent.roster.length) throw new RangeError("the belief and the opponent roster are of different sizes");
  if (belief.disguise.unsettled.length > 0 && belief.disguise.positions.length > 0) return refusedSheet(opponent, "disguise_unsettled", hash);
  const slots = belief.slots.map((slot, i): SheetSlot => {
    const ev = slotEvidence(opponent.roster[i] as PublicSide["roster"][number]);
    const stone = ev.mega ? ev.megaStone : null;
    return {
      species: slot.species,
      brought_ppm: roundPpm(slot.p_brought),
      brought_source: slot.settled ? "revealed" : "predicted",
      item: itemField(slot.items, k.item, ev.item ?? stone),
      ability: idField(slot.abilities, k.ability, ev.ability),
      nature: idField(natureMarginals(slot.spreads), k.nature, null),
      moves: moveField(slot.moves, k.moves, ev.moves),
    };
  });
  return { source: { kind: "belief", hash }, slots: slots as OpponentSheet["slots"] };
}
