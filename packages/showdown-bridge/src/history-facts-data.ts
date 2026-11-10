// The reviewed engine data the history facts take (`HistoryFactsData`, ADR-022 erratum E-22.5.2, slice H1), read from
// the pinned engine's own source files, the committed engine-data table and the committed volatile copy table: the
// stall moves and the stall condition's counter constants, the `[from]` texts of the move lines the engine runs as
// move actions of their own, the timed slot conditions (Wish, Future Sight) with the turn residuals each lasts, and
// the Baton Pass / Shed Tail copy table. No rule is written in code; a source that no longer has the shape this
// reader expects is a typed failure. The Python reducer loads the same values from the exported data file
// `ml/areneon_ml/reducer/data/history-facts.<format>.json` (`scripts/export-history-facts-data.ts`).

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { HistoryFactsData } from "@areneon/public-state";
import { jcs, type Json } from "./canonical-json.ts";
import { DEFAULT_SHOWDOWN_DIR } from "./engine.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "./golden.ts";
import { readVolatileCopy, VOLATILE_COPY_DIR, volatileCopyFile } from "./volatile-copy.ts";
import { actionSources, allySideStallMoves, type FactsTable, required, stallMoves } from "./world-facts-data.ts";

export class HistoryFactsDataError extends Error {
  override name = "HistoryFactsDataError";
}

/** Fails closed: an extractor that finds nothing in a source that has it at this pin means the source's shape changed. */
function found<T>(list: T[], what: string, applicable = true): T[] {
  if (applicable && list.length === 0) throw new HistoryFactsDataError(`no ${what} found in the engine source`);
  return list;
}

/** The stall condition's counter: the start value, the factor of each further use and the ceiling, read from the pinned `stall` condition. */
export function stallCounter(dir: string): HistoryFactsData["stall_counter"] {
  const source = readFileSync(join(dir, "data", "conditions.ts"), "utf8");
  const block = required(/\n\tstall: \{([\s\S]*?)\n\t\},/.exec(source)?.[1], "stall condition");
  return {
    start: Number(required(/this\.effectState\.counter = (\d+)/.exec(block)?.[1], "stall counter start")),
    factor: Number(required(/this\.effectState\.counter \*= (\d+)/.exec(block)?.[1], "stall counter factor")),
    max: Number(required(/counterMax: (\d+)/.exec(block)?.[1], "stall counterMax")),
  };
}

/** The ability action sources the engine runs with `externalMove: true` (Dancer): they deduct no PP. */
export function externalActionSources(dir: string, table: FactsTable): string[] {
  const actions = readFileSync(join(dir, "sim", "battle-actions.ts"), "utf8");
  const external = new Set([...actions.matchAll(/this\.runMove\([^)]*sourceEffect: this\.dex\.abilities\.get\('([a-z0-9]+)'\), externalMove: true/g)].map((match) => match[1]));
  return found(table.abilities.filter((ability) => external.has(ability.id)).map((ability) => `ability: ${ability.name}`).sort(), "external action source", table.abilities.length > 0);
}

/** The `[from] move: M` texts of the `-enditem` lines that set the user's `lastItem` (Fling's condition does it for the thrower). */
export function itemUseSources(dir: string, table: FactsTable): string[] {
  const source = readFileSync(join(dir, "data", "moves.ts"), "utf8");
  const names = new Map(table.moves.map((m) => [m.id, m.name] as const));
  const out: string[] = [];
  for (const block of source.split(/\n\t(?=[a-z0-9]+: \{)/)) {
    const id = /^([a-z0-9]+): \{/.exec(block)?.[1];
    if (id !== undefined && names.has(id) && /pokemon\.lastItem = item\.id;[\s\S]*?'-enditem', pokemon, item\.name, '\[from\] move: /.test(block)) out.push(`move: ${names.get(id)}`);
  }
  return found(out.sort(), "item use source", table.moves.length > 0);
}

/**
 * The timed slot conditions of the format's moves. A slot condition is timed when the engine removes it from its own
 * residual after a number of turns: Future Sight and Doom Desire (`futuremove`, `endingTurn = (this.turn - 1) + K`
 * checked with `<`) and Wish (`startingTurn` checked with `<=`: it ends at the first residual after its start turn,
 * which is a span of two). Any other shape of these two conditions is refused; the other slot conditions (Healing
 * Wish, Lunar Dance, Revival Blessing) end on a switch-in and are not timed.
 */
export function slotConditions(dir: string, table: FactsTable): HistoryFactsData["slot_conditions"] {
  const moves = readFileSync(join(dir, "data", "moves.ts"), "utf8");
  const conditions = readFileSync(join(dir, "data", "conditions.ts"), "utf8");
  const ids = new Set(table.moves.map((m) => m.id));
  const future: string[] = [];
  const wish: string[] = [];
  for (const block of moves.split(/\n\t(?=[a-z0-9]+: \{)/)) {
    const id = /^([a-z0-9]+): \{/.exec(block)?.[1];
    if (id === undefined || !ids.has(id)) continue;
    if (/addSlotCondition\(target, 'futuremove'\)/.test(block)) future.push(id);
    if (/slotCondition: 'Wish'/.test(block)) {
      if (!/getOverflowedTurnCount\(\) <= this\.effectState\.startingTurn/.test(block)) throw new HistoryFactsDataError("the Wish condition no longer ends at the first residual after its start turn");
      wish.push(id);
    }
  }
  const out: { id: string; moves: string[]; residual_turns: number }[] = [];
  if (future.length > 0) {
    const span = required(/endingTurn = \(this\.turn - 1\) \+ (\d+)/.exec(conditions)?.[1], "futuremove ending turn");
    if (!/getOverflowedTurnCount\(\) < this\.effectState\.endingTurn\) return/.test(conditions)) throw new HistoryFactsDataError("the futuremove condition no longer ends at its ending turn");
    // `endingTurn = (turn - 1) + K` is checked against the turn count at each residual: it lands in the residual of the (K + 1)th turn, counting the turn it began in.
    out.push({ id: "futuremove", moves: future.sort(), residual_turns: Number(span) + 1 });
  }
  if (wish.length > 0) out.push({ id: "wish", moves: wish.sort(), residual_turns: 2 });
  return out.sort((a, b) => (a.id < b.id ? -1 : 1));
}

/**
 * The reasons of `|cant|` lines whose `BeforeMove` handler returns null: the engine then records "no option" for the action
 * (Stomping Tantrum does not count it as a failure). Read from the pinned conditions, abilities, items and moves: a handler
 * that prints `cant` for a reason and returns null.
 */
export function nullCantReasons(dir: string): string[] {
  const out = new Set<string>();
  for (const file of ["conditions", "abilities", "items", "moves"]) {
    const source = readFileSync(join(dir, "data", `${file}.ts`), "utf8");
    for (const match of source.matchAll(/onBeforeMove\([^)]*\) \{([\s\S]*?)\n\t+\},?\n/g)) {
      const body = match[1] as string;
      if (!/return null;/.test(body)) continue;
      for (const reason of body.matchAll(/this\.add\('cant', [a-zA-Z.]+, '([^']+)'\)/g)) out.add(reason[1] as string);
    }
  }
  return found([...out].sort(), "null BeforeMove reason");
}

/** The tags of the `-enditem` lines of eaten berries: the one `eatItem` prints and the ones the weakening berries print after a successful `eatItem()`. */
export function eatTags(dir: string): string[] {
  const out = new Set<string>();
  const pokemon = readFileSync(join(dir, "sim", "pokemon.ts"), "utf8");
  out.add(required(/eatItem\([\s\S]*?'-enditem', this, item, '\[(\w+)\]'/.exec(pokemon)?.[1], "eatItem end-item tag"));
  const items = readFileSync(join(dir, "data", "items.ts"), "utf8");
  for (const match of items.matchAll(/eatItem\(\)\) \{[^}]*?this\.add\('-enditem', target, this\.effect, '\[(\w+)\]'\)/g)) out.add(match[1] as string);
  return [...out].sort();
}

/** The `[from]` tag of the `-enditem` line a thrown berry's eating prints (`stealeat`: Bug Bite, Pluck), read from the pinned moves. */
export function stealEatTags(dir: string): string[] {
  const source = readFileSync(join(dir, "data", "moves.ts"), "utf8");
  const out = new Set([...source.matchAll(/'-enditem', \w+, [\w.]+, '\[from\] (\w+)', '\[move\] [^']+', `\[of\]/g)].map((m) => m[1] as string));
  return found([...out].sort(), "steal-eat tag");
}

/** The `[from]` tag of the `-enditem` line of a consumed gem (the move it boosted follows as `[move]`), read from the pinned Pokemon source. */
export function gemTags(dir: string): string[] {
  const source = readFileSync(join(dir, "sim", "pokemon.ts"), "utf8");
  const out = new Set([...source.matchAll(/'-enditem', this, item, '\[from\] (\w+)', `\[move\]/g)].map((m) => m[1] as string));
  return found([...out].sort(), "gem tag");
}

/** The items whose `-enditem` line is printed where the engine clears the item itself, without `useItem` or `eatItem` (so no `lastItem`). */
export function directEndItems(dir: string, table: FactsTable): string[] {
  const source = readFileSync(join(dir, "data", "items.ts"), "utf8");
  const ids = new Set((table.items ?? []).map((i) => i.id));
  const out: string[] = [];
  for (const block of source.split(/\n\t(?=[a-z0-9]+: \{)/)) {
    const id = /^([a-z0-9]+): \{/.exec(block)?.[1];
    if (id !== undefined && ids.has(id) && /this\.add\('-enditem', target, '[^']+'\);\s*target\.item = ''/.test(block)) out.push(id);
  }
  return found(out.sort(), "directly cleared item", ids.size > 0);
}

/** The berries with an `onEat` effect, read from the pinned items (a thrown one is eaten by the target). */
export function eatenBerries(dir: string, table: FactsTable): string[] {
  const source = readFileSync(join(dir, "data", "items.ts"), "utf8");
  const ids = new Set((table.items ?? []).map((i) => i.id));
  const out: string[] = [];
  for (const block of source.split(/\n\t(?=[a-z0-9]+: \{)/)) {
    const id = /^([a-z0-9]+): \{/.exec(block)?.[1];
    if (id !== undefined && ids.has(id) && /isBerry: true/.test(block) && /\n\t\tonEat\(/.test(block)) out.push(id);
  }
  return found(out.sort(), "eaten berry", ids.size > 0);
}

/**
 * The `[from]` texts of the `-item` lines whose code also clears a `lastItem`: the Pokemon's own (Recycle, Harvest:
 * the item line and the cleared field name the same variable) or another's (Pickup). Read from the pinned moves and abilities.
 */
export function lastItemClearers(dir: string, table: FactsTable): { restorers: string[]; takers: string[] } {
  const restorers = new Set<string>();
  const takers = new Set<string>();
  const kinds = [
    ["moves", new Set(table.moves.map((m) => m.id))],
    ["abilities", new Set(table.abilities.map((a) => a.id))],
  ] as const;
  for (const [file, ids] of kinds) {
    const source = readFileSync(join(dir, "data", `${file}.ts`), "utf8");
    for (const block of source.split(/\n\t(?=[a-z0-9]+: \{)/)) {
      const id = /^([a-z0-9]+): \{/.exec(block)?.[1];
      if (id === undefined || !ids.has(id)) continue;
      const cleared = /(\w+)\.lastItem = '';/.exec(block)?.[1];
      const shown = /'-item', (\w+), [^;]*?'\[from\] ((?:move|ability): [^']+)'/.exec(block);
      if (cleared === undefined || shown === null) continue;
      (cleared === shown[1] ? restorers : takers).add(shown[2] as string);
    }
  }
  const applicable = table.moves.length > 0 && table.abilities.length > 0;
  return { restorers: found([...restorers].sort(), "last-item restorer", applicable), takers: found([...takers].sort(), "last-item taker", applicable) };
}

/** The moves that break Protect-like volatiles (`breaksProtect`), restricted to the format's moves. */
export function protectBreakers(dir: string, table: FactsTable): string[] {
  const source = readFileSync(join(dir, "data", "moves.ts"), "utf8");
  const ids = new Set(table.moves.map((m) => m.id));
  const out: string[] = [];
  for (const block of source.split(/\n\t(?=[a-z0-9]+: \{)/)) {
    const id = /^([a-z0-9]+): \{/.exec(block)?.[1];
    if (id !== undefined && ids.has(id) && /breaksProtect: true/.test(block)) out.push(id);
  }
  return found(out.sort(), "Protect breaker", ids.size > 0);
}

/** The history facts' data for an engine checkout, an engine-data table and a format. */
export function readHistoryFactsData(table: FactsTable, formatId: string = GOLDEN_FORMAT, engineDir: string = process.env.SHOWDOWN_DIR || DEFAULT_SHOWDOWN_DIR): HistoryFactsData {
  const stall = stallMoves(engineDir, table);
  const copy = readVolatileCopy(JSON.parse(readFileSync(join(VOLATILE_COPY_DIR, volatileCopyFile(formatId)), "utf8")));
  return {
    stall_moves: stall,
    ally_side_moves: allySideStallMoves(table, stall),
    stall_counter: stallCounter(engineDir),
    action_sources: actionSources(engineDir, table),
    external_action_sources: externalActionSources(engineDir, table),
    item_use_sources: itemUseSources(engineDir, table),
    eat_tags: eatTags(engineDir),
    steal_eat_tags: stealEatTags(engineDir),
    gem_tags: gemTags(engineDir),
    last_item_restorers: lastItemClearers(engineDir, table).restorers,
    last_item_takers: lastItemClearers(engineDir, table).takers,
    eaten_berries: eatenBerries(engineDir, table),
    direct_end_items: directEndItems(engineDir, table),
    slot_conditions: slotConditions(engineDir, table),
    null_cant_reasons: nullCantReasons(engineDir),
    protect_breaker_moves: protectBreakers(engineDir, table),
    volatile_not_copied: copy.not_copied,
    shed_tail_copies: copy.shed_tail_copies,
  };
}

/** The data for the committed format: the golden engine-data table and the pinned checkout. */
export function goldenHistoryFactsData(): HistoryFactsData {
  return readHistoryFactsData(JSON.parse(readFileSync(join(GOLDEN_DIR, "engine-data.json"), "utf8")) as FactsTable);
}

// ---- the exported data file the Python reducer loads ----

export const HISTORY_FACTS_DIR = VOLATILE_COPY_DIR;
export const historyFactsFile = (formatId: string): string => `history-facts.${formatId}.json`;

/** The part of the data the reducer takes from this file (the stall moves and the copy table have their own files). */
export interface HistoryFactsDataFile {
  schema_version: 1;
  showdown_commit: string;
  format_id: string;
  stall_counter: HistoryFactsData["stall_counter"];
  action_sources: string[];
  external_action_sources: string[];
  item_use_sources: string[];
  eat_tags: string[];
  steal_eat_tags: string[];
  gem_tags: string[];
  last_item_restorers: string[];
  last_item_takers: string[];
  eaten_berries: string[];
  direct_end_items: string[];
  slot_conditions: { id: string; moves: string[]; residual_turns: number }[];
  null_cant_reasons: string[];
  protect_breaker_moves: string[];
  /** SHA-256 of the JCS bytes of the other fields. */
  history_facts_data_hash: string;
}

export function historyFactsDataHash(body: Omit<HistoryFactsDataFile, "history_facts_data_hash">): string {
  return createHash("sha256").update(jcs(body as unknown as Json)).digest("hex");
}

export function exportHistoryFactsData(commit: string, formatId: string, data: HistoryFactsData): HistoryFactsDataFile {
  const body = {
    schema_version: 1 as const,
    showdown_commit: commit,
    format_id: formatId,
    stall_counter: data.stall_counter,
    action_sources: [...data.action_sources],
    external_action_sources: [...(data.external_action_sources ?? [])],
    item_use_sources: [...data.item_use_sources],
    eat_tags: [...data.eat_tags],
    steal_eat_tags: [...data.steal_eat_tags],
    gem_tags: [...data.gem_tags],
    last_item_restorers: [...data.last_item_restorers],
    last_item_takers: [...data.last_item_takers],
    eaten_berries: [...data.eaten_berries],
    direct_end_items: [...data.direct_end_items],
    slot_conditions: data.slot_conditions.map((c) => ({ id: c.id, moves: [...c.moves], residual_turns: c.residual_turns })),
    null_cant_reasons: [...data.null_cant_reasons],
    protect_breaker_moves: [...data.protect_breaker_moves],
  };
  return { ...body, history_facts_data_hash: historyFactsDataHash(body) };
}
