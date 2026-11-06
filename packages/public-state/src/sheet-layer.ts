// The sheet layer around the shared reader (ADR-027 section 5, #1150). The interpreter never sees a
// `|showteam|` line (its bytes and hash stay as they were); this layer takes the line out of the
// stream, checks it fail closed, and attaches the sheets to the states it returns.
import type { PublicDex, PublicSheetMon, PublicState } from "./index.ts";
import { parseSheet, SheetError } from "./sheet.ts";

const KIND = "|showteam";
const PREFIX = `${KIND}|`;

/** A line the interpreter must not see: the placeholder keeps line indices and request counts aligned. */
export const PLACEHOLDER = "|";

export const isShowteam = (raw: string): boolean => raw === KIND || raw.startsWith(PREFIX);

type Which = "own" | "opponent";

export class SheetTracker {
  readonly #player: string;
  readonly #other: string;
  readonly #species: NonNullable<PublicDex["species"]>;
  readonly #sheets: Partial<Record<Which, PublicSheetMon[]>> = {};
  #version = 0;
  #cache: { source: PublicState; version: number; result: PublicState } | undefined;

  constructor(player: string, dex: PublicDex | undefined) {
    this.#player = player;
    this.#other = player === "p1" ? "p2" : "p1";
    this.#species = dex?.species ?? {};
  }

  /**
   * Take one `|showteam|` line, given the state before it. Returns the reason it was refused (the line
   * is then kept and flagged, and no sheet is recorded), or null when it was taken.
   */
  take(raw: string, before: PublicState): string | null {
    const rest = raw.startsWith(PREFIX) ? raw.slice(PREFIX.length) : "";
    const bar = rest.indexOf("|");
    if (bar < 0) return "showteam needs at least 2 fields";
    const side = rest.slice(0, bar);
    const which: Which | null = side === this.#player ? "own" : side === this.#other ? "opponent" : null;
    if (which === null) return `unsupported side '${side}' in showteam`;
    if (before.started) return "a team sheet after |start";
    if (this.#sheets[which] !== undefined) return "a second team sheet for one side";
    let sheet: PublicSheetMon[];
    try {
      sheet = parseSheet(rest.slice(bar + 1));
    } catch (error) {
      if (error instanceof SheetError) return error.message;
      throw error;
    }
    const roster = before.opponent.roster;
    if (which === "opponent" && roster.length > 0) {
      const bases = sheet.map((m) => this.#species[m.species]?.base ?? m.species);
      if (bases.length !== roster.length || bases.some((base, i) => base !== roster[i]?.base_species)) {
        return "a team sheet that is not the team preview's";
      }
    }
    this.#sheets[which] = sheet;
    this.#version += 1;
    this.#cache = undefined;
    return null;
  }

  /** `state` with the sheets read so far attached; a state before any sheet is returned as it is. */
  attach(state: PublicState): PublicState {
    if (this.#version === 0) return state;
    if (this.#cache?.source === state && this.#cache.version === this.#version) return this.#cache.result;
    const result: PublicState = {
      ...state,
      own: this.#sheets.own === undefined ? state.own : { ...state.own, sheet: this.#sheets.own },
      opponent: this.#sheets.opponent === undefined ? state.opponent : { ...state.opponent, sheet: this.#sheets.opponent },
    };
    this.#cache = { source: state, version: this.#version, result };
    return result;
  }
}
