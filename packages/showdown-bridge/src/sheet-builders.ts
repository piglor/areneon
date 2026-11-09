// The two opponent_sheet builders that need no belief (ADR-027 section 2, section 5, #1151).
//
//  - `sheetFromChannel(channel, record)` is the only constructor of an `engine-sheet/1` block (the open regime): the
//    opponent's engine-sent `|showteam|` sheet, read by the shared reader from the player's OWN channel, becomes the
//    block with every item, ability, nature and move field at source `sheet`. `brought_ppm` is unknown until the
//    belief fills it (slice 6). It refuses a closed-format record and a channel without both sheets.
//  - `revealedOnlySheet(opponent)` is the closed regime's `revealed-only/1` block: what the battle has shown at
//    probability 1 (source `revealed`), everything else unknown. There is no predicted field and no brought share.
//
// Both are pure functions of the reader's state (the player's own view): no team, omniscient log, hidden verdict,
// battle record or seed has a parameter (a type test), and the module imports no engine, runner or Node module but
// the SHA-256 for the block's source hash. The shape and reader rules are `opponent-sheet.ts`'s.

import { createHash } from "node:crypto";
import { interpretChannel, type PublicFlag, type PublicSide } from "@areneon/public-state";
import type { OpponentSheet } from "@areneon/schema/types/v1/opponent-sheet.ts";
import { jcs } from "./canonical-json.ts";
import type { PlayerChannel } from "./observation.ts";
import { teamSheetsOf } from "./rules.ts";

/** The block's source ids (`source.hash` is the SHA-256 of the JCS of `{feature_set, sheet_builders_format}`). */
export const ENGINE_SHEET_FEATURE_SET = "engine-sheet/1";
export const REVEALED_ONLY_FEATURE_SET = "revealed-only/1";
/** Bump when either builder computes a value differently for the same input. */
export const SHEET_BUILDERS_FORMAT = "sheet-builders/1";

const FULL = 1_000_000;
const MAX_MOVES = 4;

const hashOf = (feature_set: string): string => createHash("sha256").update(jcs({ feature_set, sheet_builders_format: SHEET_BUILDERS_FORMAT })).digest("hex");
export const ENGINE_SHEET_HASH = hashOf(ENGINE_SHEET_FEATURE_SET);
export const REVEALED_ONLY_HASH = hashOf(REVEALED_ONLY_FEATURE_SET);

/** A channel or record this builder cannot give a block for; the message names a rule and quotes no value. */
export class SheetBuildError extends Error {
  override name = "SheetBuildError";
}

type Slot = OpponentSheet["slots"][number];
const unknownField = { source: "unknown" as const, entries: [], other_ppm: 0 };
const byId = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);

/**
 * The `engine-sheet/1` block at the request a player's own channel ends in. `record` is the format's rules record
 * (only `team_sheets` is read): the open regime alone has an engine sheet.
 */
export function sheetFromChannel(channel: PlayerChannel, record: { readonly team_sheets?: "closed" | "open" }): OpponentSheet {
  // The regime comes from the format's rules record, never from the channel: a channel cannot say which format it is.
  if (teamSheetsOf(record) !== "open") throw new SheetBuildError("an engine sheet exists in an open-sheet format only");
  const run = interpretChannel(channel.player, channel.lines);
  const frame = run.frames[run.frames.length - 1];
  if (frame === undefined || frame.kind !== "request") throw new SheetBuildError("the channel does not end in a request");
  if (run.flags.length > 0) throw new SheetBuildError("the reader flagged a line of the channel");
  const { roster, sheet } = frame.state.opponent;
  if (sheet === undefined) throw new SheetBuildError("the channel holds no opponent team sheet");
  // The reader flags a sheet that is not the team preview's (a different count included), which is refused above.
  if (roster.length !== sheet.length) throw new SheetBuildError("the opponent sheet does not list one Pokemon per preview slot");
  const slots = sheet.map((mon, i): Slot => {
    const base = roster[i]?.base_species;
    if (base === undefined || base === "") throw new SheetBuildError("a sheet Pokemon has no preview entry");
    if (mon.nature === null) throw new SheetBuildError("a sheet Pokemon has no nature");
    const moves = [...new Set(mon.moves)].sort(byId);
    if (moves.length !== mon.moves.length || moves.length < 1 || moves.length > MAX_MOVES) throw new SheetBuildError("a sheet Pokemon holds one to four distinct moves");
    return {
      species: base,
      brought_ppm: 0,
      brought_source: "unknown",
      item: { source: "sheet", entries: [{ id: mon.item, ppm: FULL }], other_ppm: 0 },
      ability: { source: "sheet", entries: [{ id: mon.ability, ppm: FULL }], other_ppm: 0 },
      nature: { source: "sheet", entries: [{ id: mon.nature, ppm: FULL }], other_ppm: 0 },
      moves: { source: "sheet", entries: moves.map((id) => ({ id, inclusion_ppm: FULL })), slots_known: moves.length },
    };
  });
  return { source: { kind: "engine-sheet", hash: ENGINE_SHEET_HASH }, slots: slots as OpponentSheet["slots"] };
}

/** The `revealed-only/1` block of the reader's opponent half: revealed facts at 1,000,000 ppm, all else unknown. */
export function revealedOnlySheet(opponent: Pick<PublicSide, "roster">, flags: readonly PublicFlag[] = []): OpponentSheet {
  // Fail closed, as the belief does (`reader_flagged`): a line the reader could not take, or an Illusion twin it could not
  // reconcile, leaves a state that is not the facts. Slots are then exactly the non-phantom roster.
  if (flags.length > 0 || opponent.roster.some((mon) => mon.phantom === true)) throw new SheetBuildError("reader_flagged");
  const slots = opponent.roster.map((mon): Slot => {
    const transformed = mon.volatiles.includes("transform");
    // The same evidence the belief reads: an item or ability a Pokemon received or copied is not its own.
    const item = mon.item_known && mon.item !== null && !mon.item_received ? mon.item : mon.mega ? mon.mega_stone : null;
    const ability = mon.ability_received ? null : mon.ability;
    const moves = transformed ? [] : [...new Set(mon.moves_revealed)].sort(byId);
    return {
      species: mon.base_species,
      brought_ppm: 0,
      brought_source: "unknown",
      item: item === null ? { ...unknownField } : { source: "revealed", entries: [{ id: item, ppm: FULL }], other_ppm: 0 },
      ability: ability === null ? { ...unknownField } : { source: "revealed", entries: [{ id: ability, ppm: FULL }], other_ppm: 0 },
      nature: { ...unknownField },
      // The contract's revealed move field is a complete one (all four known); a partial list is left unknown,
      // since revealed-only has no predicted source. The revealed moves stay in the observation's log.
      moves:
        moves.length === MAX_MOVES
          ? { source: "revealed", entries: moves.map((id) => ({ id, inclusion_ppm: FULL })), slots_known: MAX_MOVES }
          : { source: "unknown", entries: [], slots_known: 0 },
    };
  });
  return { source: { kind: "revealed-only", hash: REVEALED_ONLY_HASH }, slots: slots as OpponentSheet["slots"] };
}
