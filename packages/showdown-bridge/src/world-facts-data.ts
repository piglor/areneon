// The reviewed engine data the world facts take (`WorldFactsData`, ADR-026 slice L), read from the pinned engine's own
// source files and the committed engine-data table: the stall moves, the side conditions Court Change swaps, the
// Toxic stage ceiling and the `[from]` texts of the move lines the engine runs as move actions of their own. The
// search actor (slice R) and the world-builder and parity tests read it through here, so there is one reader.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { WorldFactsData } from "@areneon/public-state";
import { DEFAULT_SHOWDOWN_DIR } from "./engine.ts";
import { GOLDEN_DIR } from "./golden.ts";

/** The slice of the engine-data table the reader needs. */
export interface FactsTable {
  terrains: readonly { id: string }[];
  moves: readonly { id: string; name: string }[];
  abilities: readonly { id: string; name: string }[];
}

/** The moves whose engine definition adds the `stall` volatile, read from the pinned source, restricted to the format's moves. */
function stallMoves(dir: string, table: FactsTable): string[] {
  const source = readFileSync(join(dir, "data", "moves.ts"), "utf8");
  const ids = new Set(table.moves.map((m) => m.id));
  const out: string[] = [];
  for (const block of source.split(/\n\t(?=[a-z0-9]+: \{)/)) {
    const id = /^([a-z0-9]+): \{/.exec(block)?.[1];
    if (id !== undefined && ids.has(id) && /addVolatile\('stall'\)/.test(block)) out.push(id);
  }
  return out.sort();
}

/** The side conditions Court Change swaps, read from the pinned move's own list. */
function swapConditions(dir: string): string[] {
  const source = readFileSync(join(dir, "data", "moves.ts"), "utf8");
  const block = /\n\tcourtchange: \{[\s\S]*?const sideConditions = \[([\s\S]*?)\];/.exec(source)?.[1] as string;
  return [...block.matchAll(/'([a-z0-9]+)'/g)].map((m) => m[1] as string).sort();
}

/** The Toxic stage ceiling, read from the pinned `tox` condition's own cap. */
function toxicStageMax(dir: string): number {
  const source = readFileSync(join(dir, "data", "conditions.ts"), "utf8");
  return Number(/\n\ttox: \{[\s\S]*?stage < (\d+)/.exec(source)?.[1]);
}

/**
 * The `[from]` texts of the move lines the engine runs as move actions of their own (`runMove`, which counts
 * `activeMoveActions`): a move block that hands a queued action to `prioritizeAction` with itself as the source effect (Round, Instruct) or calls
 * `runMove` (Pursuit), and the ability whose `runMove` call sits in `battle-actions.ts` (Dancer). Read from the
 * pinned engine's own source, as the stall moves are.
 */
function actionSources(dir: string, table: FactsTable): string[] {
  const source = readFileSync(join(dir, "data", "moves.ts"), "utf8");
  const names = new Map(table.moves.map((m) => [m.id, m.name] as const));
  const out: string[] = [];
  for (const block of source.split(/\n\t(?=[a-z0-9]+: \{)/)) {
    const id = /^([a-z0-9]+): \{/.exec(block)?.[1];
    if (id !== undefined && names.has(id) && /(prioritizeAction\([^;]*,\s*move\)|actions\.runMove\()/.test(block)) out.push(`move: ${names.get(id)}`);
  }
  const actions = readFileSync(join(dir, "sim", "battle-actions.ts"), "utf8");
  const dancing = new Set([...actions.matchAll(/this\.runMove\([^)]*sourceEffect: this\.dex\.abilities\.get\('([a-z0-9]+)'\)/g)].map((match) => match[1]));
  out.push(...table.abilities.filter((ability) => dancing.has(ability.id)).map((ability) => `ability: ${ability.name}`));
  return out.sort();
}

/** The world facts' data for an engine checkout and an engine-data table. */
export function readWorldFactsData(table: FactsTable, engineDir: string = process.env.SHOWDOWN_DIR || DEFAULT_SHOWDOWN_DIR): WorldFactsData {
  return {
    stall_moves: stallMoves(engineDir, table),
    terrains: table.terrains.map((t) => t.id),
    swap_side_conditions: swapConditions(engineDir),
    toxic_stage_max: toxicStageMax(engineDir),
    action_sources: actionSources(engineDir, table),
  };
}

/** The data for the committed format: the golden engine-data table and the pinned checkout. */
export function goldenWorldFactsData(): WorldFactsData {
  return readWorldFactsData(JSON.parse(readFileSync(join(GOLDEN_DIR, "engine-data.json"), "utf8")) as FactsTable);
}
