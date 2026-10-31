// The reader rules of the `opponent_sheet` block that the contract profile cannot say (ADR-027 section 2 and 5,
// ADR-025 erratum E4; the shapes are `packages/schema/contracts/v1/opponent-sheet.schema.json`, copied into
// PolicyQuery and TrainingExample). Pure and engine-free; the Python reader
// (`areneon_contracts/opponent_sheet.py`) applies the same rules and the shared corpus holds both to it.
// A rule is returned as fixed text naming a slot and a field, never a value, or null.
//
// The rules (closed regime; the open regime's `engine-sheet` kind is checked for the shape ADR-027 gives it):
//  - an entry list is sorted by ppm descending then id ascending (a null item id last) with unique ids;
//  - other_ppm is max(0, 1000000 - the sum of the entries) and the sum is at most 1000000 plus one per entry
//    (each entry is rounded alone); a field of source unknown has no entries and other_ppm 0 (moves: slots_known 0);
//  - a field of source revealed holds its value alone at 1000000 ppm (moves: slots_known 4 and four entries at
//    1000000); a move list holds at least slots_known entries at 1000000;
//  - brought_source revealed means 1000000 ppm and unknown means 0;
//  - the source kind fits the regime: belief has no field of source sheet; revealed-only has no predicted field;
//    engine-sheet and oracle-diagnostic hold every item, ability, nature and move field at source sheet; only a
//    belief block has a refusal, and a refused block is unknown everywhere with brought_ppm 0.

import { ajvFor } from "@areneon/schema/ajv";
import sheetSchema from "@areneon/schema/contracts/v1/opponent-sheet.schema.json" with { type: "json" };
import type { OpponentSheet, SheetFieldSource } from "@areneon/schema/types/v1/opponent-sheet.ts";

const FULL = 1_000_000;

type Entry = { id: string | null; v: number };

const order = (a: Entry, b: Entry): number => b.v - a.v || (a.id === b.id ? 0 : a.id === null ? 1 : b.id === null ? -1 : a.id < b.id ? -1 : 1);

/** The problem of one entry list, as the field it is in. */
function listProblem(where: string, source: SheetFieldSource, entries: readonly Entry[]): string | null {
  if (source === "unknown") return entries.length === 0 ? null : `${where}: a field of source unknown has no entries`;
  if (entries.length === 0) return `${where}: a field with a source other than unknown has entries`;
  for (let i = 1; i < entries.length; i++) {
    if (order(entries[i - 1] as Entry, entries[i] as Entry) >= 0) return `${where}: the entries must be sorted by ppm descending then id, and unique`;
  }
  return null;
}

function idFieldProblem(where: string, field: { source: SheetFieldSource; entries: readonly { id: string | null; ppm: number }[]; other_ppm: number }): string | null {
  const entries = field.entries.map((e) => ({ id: e.id, v: e.ppm }));
  const problem = listProblem(where, field.source, entries);
  if (problem !== null) return problem;
  if (field.source === "unknown") return field.other_ppm === 0 ? null : `${where}: a field of source unknown has other_ppm 0`;
  const sum = entries.reduce((s, e) => s + e.v, 0);
  if (sum > FULL + entries.length) return `${where}: the entries sum to more than 1000000 plus one per entry`;
  if (field.other_ppm !== Math.max(0, FULL - sum)) return `${where}: other_ppm must be max(0, 1000000 - the sum of the entries)`;
  if (field.source === "revealed" && !(entries.length === 1 && entries[0]?.v === FULL)) return `${where}: a revealed field holds its one value at 1000000 ppm`;
  return null;
}

function movesProblem(where: string, field: { source: SheetFieldSource; entries: readonly { id: string; inclusion_ppm: number }[]; slots_known: number }): string | null {
  const entries = field.entries.map((e) => ({ id: e.id, v: e.inclusion_ppm }));
  const problem = listProblem(where, field.source, entries);
  if (problem !== null) return problem;
  if (field.source === "unknown") return field.slots_known === 0 ? null : `${where}: a field of source unknown has slots_known 0`;
  const certain = entries.filter((e) => e.v === FULL).length;
  if (certain < field.slots_known) return `${where}: slots_known counts revealed moves, which are held at 1000000 ppm`;
  if (field.source === "revealed" && !(field.slots_known === 4 && entries.slice(0, 4).every((e) => e.v === FULL))) return `${where}: a revealed move field has all four moves known`;
  return null;
}

/** The first problem of a block, or null. */
export function opponentSheetProblem(block: OpponentSheet): string | null {
  const { kind, refusal } = block.source;
  if (refusal !== undefined && kind !== "belief") return "only a belief block can carry a refusal";
  for (const [i, slot] of block.slots.entries()) {
    const at = `slot ${i}`;
    const fields = [slot.item.source, slot.ability.source, slot.nature.source, slot.moves.source];
    if (slot.brought_source === "unknown" ? slot.brought_ppm !== 0 : slot.brought_source === "revealed" && slot.brought_ppm !== FULL) return `${at}: brought_ppm does not fit brought_source`;
    if (refusal !== undefined && !(slot.brought_ppm === 0 && slot.brought_source === "unknown" && fields.every((f) => f === "unknown"))) return `${at}: a refused block is unknown everywhere`;
    if (kind === "belief" && fields.includes("sheet")) return `${at}: a belief block has no field of source sheet`;
    if (kind === "revealed-only" && (fields.includes("sheet") || fields.includes("predicted") || slot.brought_source !== "unknown")) return `${at}: a revealed-only block has revealed or unknown fields only`;
    if ((kind === "engine-sheet" || kind === "oracle-diagnostic") && !fields.every((f) => f === "sheet")) return `${at}: a sheet block holds every item, ability, nature and move field at source sheet`;
    if (kind === "engine-sheet" && slot.brought_source === "revealed") return `${at}: an engine-sheet block takes brought_ppm from the belief or leaves it unknown`;
    const problem =
      idFieldProblem(`${at} item`, slot.item) ?? idFieldProblem(`${at} ability`, slot.ability) ?? idFieldProblem(`${at} nature`, slot.nature) ?? movesProblem(`${at} moves`, slot.moves);
    if (problem !== null) return problem;
  }
  return null;
}

/** A block that does not match its contract or breaks a reader rule. */
export class OpponentSheetError extends Error {
  override name = "OpponentSheetError";
}

const validSheet = ajvFor("contract").compile(sheetSchema);

/** The `OpponentSheet` contract's reader: the schema, then the rules the profile cannot say. */
export function readOpponentSheet(value: unknown): OpponentSheet {
  if (!validSheet(value)) throw new OpponentSheetError(`not an OpponentSheet: ${JSON.stringify(validSheet.errors)}`);
  const block = structuredClone(value) as unknown as OpponentSheet;
  const problem = opponentSheetProblem(block);
  if (problem !== null) throw new OpponentSheetError(problem);
  return block;
}
