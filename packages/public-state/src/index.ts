// The shared player-perspective state reader (ADR-018 section 2). One clean-room reader turns ONE
// player's own channel into a state after every line; the replay pages inline the very same file
// (`interpreter.cjs`, a classic script) and the parity tests hold it equal to the Python reducer.
// This module is engine-free and imports nothing from the bridge. It reads a channel it is given and
// nothing else: it takes no team, no omniscient log and no seed.
import { createRequire } from "node:module";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

/** One Pokemon as the reader shows it (the reducer's snake_case shape). */
export interface PublicMon {
  own: boolean;
  name: string | null;
  base_species: string;
  species: string | null;
  temp_species: string | null;
  level: number | null;
  gender: string | null;
  shiny: boolean;
  /** Shown hit points: out of 100 for the opponent, exact for the player's own Pokemon. */
  hp_num: number | null;
  hp_den: number | null;
  status: string | null;
  fainted: boolean;
  brought: boolean | null;
  active_slot: string | null;
  boosts: Record<string, number>;
  volatiles: string[];
  moves_revealed: string[];
  item: string | null;
  item_known: boolean;
  item_lost: boolean;
  ability: string | null;
  ability_suppressed: boolean;
  mega: boolean;
  mega_stone: string | null;
  stats: Record<string, number> | null;
  moves_own: string[] | null;
  pp: Record<string, number>;
  can_mega_evo: boolean;
  phantom?: boolean;
}

export interface PublicSide {
  player: string;
  username: string | null;
  teamsize: number | null;
  roster: PublicMon[];
  slots: Record<string, number | null>;
  conditions: Record<string, number>;
}

export interface PublicState {
  player: string;
  format_name: string | null;
  gametype: string | null;
  turn: number;
  started: boolean;
  request_kind: string | null;
  weather: string | null;
  terrain: string | null;
  field_conditions: string[];
  winner: string | null;
  own: PublicSide;
  opponent: PublicSide;
  labels: Record<string, string>;
  omniscient?: boolean;
}

export interface PublicFrame {
  index: number;
  line: string;
  kind: string;
  caption: string | null;
  /** The reason the line could not be taken; null when it was. A flagged line is kept, never dropped. */
  flagged: string | null;
  request: number;
  state: PublicState;
}

export interface PublicFlag {
  index: number;
  line: string;
  reason: string;
}

export interface PublicDex {
  species?: Record<string, { name?: string; base: string; types: string[] }>;
  terrains?: string[];
}

export interface PublicRun {
  frames: PublicFrame[];
  flags: PublicFlag[];
  requests: unknown[];
}

export interface PublicInterpreter {
  feed(raw: string): { kind: string; caption: string | null; flagged: string | null };
  snapshot(): PublicState;
  requestCount(): number;
  requests(): unknown[];
}

export interface PublicStateApi {
  toId(text: string): string;
  parseLine(raw: string): { raw: string; kind: string; args: string[]; tags: Record<string, string> };
  createInterpreter(options: { player: string; dex?: PublicDex }): PublicInterpreter;
  interpretChannel(player: string, lines: readonly string[], dex?: PublicDex, omniscient?: false): PublicRun;
}

/** The reader's file: the page inlines these exact bytes, and the baseline identity (ADR-018 section 5) hashes them. */
export const INTERPRETER_PATH = join(dirname(fileURLToPath(import.meta.url)), "interpreter.cjs");

const api = createRequire(import.meta.url)("./interpreter.cjs") as PublicStateApi;

export const toId = api.toId;
export const parseLine = api.parseLine;
export const createInterpreter = api.createInterpreter;

/** The state after every line of one player's own channel (a flagged line is kept and listed in `flags`). */
export function interpretChannel(player: string, lines: readonly string[], dex?: PublicDex): PublicRun {
  return api.interpretChannel(player, lines, dex, false);
}

/** SHA-256 of the reader file's bytes: a change to the reader is a change to any baseline that uses it. */
export function readerSha256(): string {
  return createHash("sha256").update(readFileSync(INTERPRETER_PATH)).digest("hex");
}
