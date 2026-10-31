// One simulated turn of a search (ADR-026 sections 1 and 2): the world's battle is cloned, its random generator is
// replaced by the world's engine seed (`search-engine/1`, the same for every candidate, so the candidates meet the same
// draws), both sides' choices are made, and the engine runs the turn. What comes out is what the PLAYER would have seen:
// the lines of the turn as the engine sends this seat's stream (the secret half of its own lines, the public half of the
// opponent's), and the request the engine sends the seat next. Appended to the real channel they are the hypothetical
// channel of the leaf.
//
// The world is a battle the builder made from the player's own view plus a sampled opponent, so everything here is
// ours; nothing is read from, or written to, a live battle (the world passed in is never changed: every turn runs on
// `Battle.fromJSON` of its serialisation). The clone's `send` is the engine's own output callback (`restart`), so the
// lines and requests have the order the real streams give them.

import { enumerateCandidates } from "./choices.ts";
import type { Engine, SimBattle } from "./engine.ts";
import { decodeRequest } from "./request.ts";

/** The engine's own split regex (`sim/battle.ts` `extractChannelMessages`): a `|split|pN` line takes the next two lines, secret then shared. */
const SPLIT = /^\|split\|p([1234])\n(.*)\n(.*)|.+/gm;

/** One channel's view of a chunk of log: `seat` (1 or 2) sees the secret half of its own split lines and the shared half of the others'. `|t:|` lines are dropped as the runner drops them. */
export function playerView(text: string, seat: 1 | 2): string[] {
  const lines: string[] = [];
  for (const [line, side, secret, shared] of text.matchAll(SPLIT)) {
    const chosen = side === undefined ? line : Number(side) === seat ? secret : shared;
    if (!chosen || chosen.startsWith("|t:|")) continue;
    lines.push(chosen);
  }
  return lines;
}

/** The engine objects a turn reads, narrowed here. */
interface TurnBattle extends SimBattle {
  log: string[];
  sentLogPos: number;
  prng: unknown;
  winner?: string;
  restart(send: (type: string, data: string | string[]) => void): void;
  sendUpdates(): void;
  sides: (SimBattle["sides"][number] & { activeRequest: ({ wait?: boolean } & object) | null })[];
}

export type Outcome = "own" | "foe" | "tie";

export interface TurnResult {
  /** Why the turn could not be played: a side's choice was refused (an `|error|`), or the engine threw. */
  refused: string | null;
  /** The lines this seat saw, in the order the streams give them: the turn's log, then the seat's request(s). */
  lines: string[];
  /** The last request the engine sent this seat (JSON text), if any. */
  request: string | null;
  /** The game ended in this turn. */
  ended: Outcome | null;
  /** The battle after the turn (a clone of the world). */
  battle: SimBattle;
  /** The opponent seat's own view of the same turn, only when `playChoices` was asked for it (`withFoe`). */
  foe?: { lines: string[]; request: string | null };
}

/** What a turn plays: the seat's choice and the opponent's. A side that is waiting (it has no request to answer) takes none. */
export interface TurnChoices {
  own: string | null;
  foe: string | null;
}

/**
 * The world's battle, cloned once per use. `serialised` is the JSON text of `world.battle.toJSON()` (made once per world:
 * every candidate and reply of the world starts from the same text).
 */
export function cloneWorld(engine: Engine, serialised: string, seed: string): TurnBattle {
  const clone = engine.sim.Battle.fromJSON(serialised) as TurnBattle;
  clone.prng = new engine.sim.PRNG(seed);
  return clone;
}

/**
 * Play the choices on `clone`: the engine commits the turn when every side that has a request has chosen. Returns the
 * seat's view of what the engine sent. `clone` stays live, so a caller may continue it (a `wait` request is played on).
 */
export function playChoices(clone: TurnBattle, seat: "p1" | "p2", choices: TurnChoices, withFoe = false): TurnResult {
  const viewer = seat === "p1" ? 1 : 2;
  const foe = seat === "p1" ? "p2" : "p1";
  const lines: string[] = [];
  const foeLines: string[] = [];
  let foeRequest: string | null = null;
  let request: string | null = null;
  let refused: string | null = null;
  clone.restart((type, data) => {
    // The engine sends the turn's log as `update` (an array of lines) and each side's request or error as `sideupdate` (`<side>\n<line>`); `end` is ignored.
    if (type === "update") {
      lines.push(...playerView((data as string[]).join("\n"), viewer));
      if (withFoe) foeLines.push(...playerView((data as string[]).join("\n"), viewer === 1 ? 2 : 1));
    }
    if (type === "sideupdate") {
      const text = data as string;
      const at = text.indexOf("\n");
      if (withFoe && text.slice(0, at) === foe && !text.slice(at + 1).startsWith("|error|")) {
        foeLines.push(text.slice(at + 1));
        if (text.slice(at + 1).startsWith("|request|")) foeRequest = text.slice(at + 1 + "|request|".length);
      }
      if (text.slice(0, at) !== seat) return;
      const line = text.slice(at + 1);
      if (line.startsWith("|error|")) refused = line;
      else {
        lines.push(line);
        if (line.startsWith("|request|")) request = line.slice("|request|".length);
      }
    }
  });
  clone.sentLogPos = clone.log.length;
  try {
    // The seat chooses first, as a player does; the engine commits when the last side that must choose has.
    if (choices.own !== null && !clone.choose(seat, choices.own)) refused ??= "the engine refused the seat's choice";
    if (refused === null && choices.foe !== null && !clone.choose(foe, choices.foe)) refused ??= "the engine refused the opponent's choice";
    // The engine's stream calls this after every written line: it sends the log of the turn, then each side's new request.
    clone.sendUpdates();
  } catch (error) {
    refused ??= `the engine threw: ${(error as Error).message}`;
  }
  let ended: Outcome | null = null;
  if (clone.ended) ended = clone.winner === undefined || clone.winner === "" ? "tie" : clone.winner === seat ? "own" : "foe";
  return { refused, lines, request, ended, battle: clone, ...(withFoe ? { foe: { lines: foeLines, request: foeRequest } } : {}) };
}

/** The request (parsed) the engine holds for a side of the clone, or null. */
export const activeRequest = (clone: TurnBattle, side: string): ({ wait?: boolean } & object) | null => clone.sides.find((s) => s.id === side)?.activeRequest ?? null;

export type { TurnBattle };

/**
 * The opponent's choice at a forced switch of the clone: the first non-pending legal choice in canonical order (the `actor-rule`
 * choice). The options of a forced switch are switches, which the max-damage reply never plays; it is the engine's own request for
 * the opponent, so the choice is legal. Called only while the seat is waiting, so the opponent holds a forced-switch request.
 */
export function foeSwitchChoice(clone: TurnBattle, seat: "p1" | "p2"): string {
  const decoded = decodeRequest(JSON.stringify(activeRequest(clone, seat === "p1" ? "p2" : "p1")));
  return (enumerateCandidates(decoded).candidates.find((candidate) => !candidate.pending) as { choice: string }).choice;
}
