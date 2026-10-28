// World checks V2 and V4 (ADR-026 Evidence 1). Pure functions over plain data, so each is tested with
// doctored inputs as well as on built worlds.
//
// V2: the world's own request equals the real one on every reproduced field (conditions, stats, moves with
// PP and `disabled`, `trapped` / `maybeTrapped`, `canMegaEvo`), and `enumerateCandidates` gives the same
// candidate list with the same pending flags. V2 is evidence, not a defect (N6): a `maybeTrapped` request
// that the world's opponent cannot explain is a reason to draw the opponent's sets again.
// V4: field and side conditions equal the reader's, and their remaining turns are consistent with the log (a
// condition the log still shows has not run out; the engine's base length comes from the source's sampled item).

import { enumerateCandidates } from "./choices.ts";
import { decodeRequest } from "./request.ts";
import type { StateClass, StateItemId } from "./world-writers.ts";

type Json = null | boolean | number | string | Json[] | { [key: string]: Json };

/** A failed check, with what the redraw interface needs. */
export interface CheckFailure {
  code: "v2_request" | "v2_candidates" | "v4_conditions" | "v4_duration";
  detail: string;
  /** The state item whose writer the difference points at, null for a field the world gets from the engine (E) or the opponent's sets. */
  item: StateItemId | null;
  /** The class of that item; null when the difference is evidence about the opponent's unrevealed sets. */
  state_class: StateClass | "E" | null;
  redraw: boolean;
}

/** The request as a plain JSON value without what differs by transport: `rqid` and the side's name. */
export function plainRequest(request: unknown): Json {
  const copy = JSON.parse(JSON.stringify(request)) as { rqid?: unknown; side: { name?: unknown } };
  delete copy.rqid;
  delete copy.side.name;
  return copy as Json;
}

/** The first path where two JSON values differ, or null. Arrays and objects are compared key by key. */
export function firstDifference(a: Json, b: Json, path = ""): string | null {
  if (a === b) return null;
  if (typeof a !== "object" || typeof b !== "object" || a === null || b === null) return path === "" ? "/" : path;
  if (Array.isArray(a) !== Array.isArray(b)) return path;
  const keys = [...new Set([...Object.keys(a), ...Object.keys(b)])].sort();
  for (const key of keys) {
    const inA = Object.hasOwn(a, key);
    const inB = Object.hasOwn(b, key);
    const where = `${path}/${key}`;
    if (inA !== inB) return where;
    const found = firstDifference((a as Record<string, Json>)[key] as Json, (b as Record<string, Json>)[key] as Json, where);
    if (found !== null) return found;
  }
  return null;
}

/** Which state item (and class) a request field belongs to. */
function requestField(path: string): Pick<CheckFailure, "item" | "state_class" | "redraw"> {
  if (/\/(disabled|maybeDisabled|maybeLocked)$/.test(path)) return { item: "choice_lock", state_class: "W", redraw: false };
  if (/\/(trapped|maybeTrapped)$/.test(path)) return { item: null, state_class: null, redraw: true };
  if (/\/item$/.test(path)) return { item: "item_removed", state_class: "W", redraw: false };
  return { item: null, state_class: "E", redraw: false };
}

const WORDS: [RegExp, string][] = [
  [/\/(trapped|maybeTrapped)$/, "trapping flags"],
  [/\/(disabled|maybeDisabled|maybeLocked)$/, "disabled moves"],
  [/\/moves\//, "moves and PP"],
  [/\/stats\//, "stats"],
  [/\/condition$/, "conditions"],
  [/\/canMegaEvo$/, "Mega Evolution"],
];

/** V2, first half: the world's request equals the real one on every reproduced field. */
export function v2Request(world: unknown, real: unknown): CheckFailure | null {
  const path = firstDifference(plainRequest(world), plainRequest(real));
  if (path === null) return null;
  const words = WORDS.find(([pattern]) => pattern.test(path))?.[1] ?? "a request field";
  return { code: "v2_request", detail: `the world's own request differs from the real one in ${words}`, ...requestField(path) };
}

/** V2, second half: the same candidate list with the same pending flags. */
export function v2Candidates(world: unknown, real: unknown): CheckFailure | null {
  const list = (request: unknown) => enumerateCandidates(decodeRequest(JSON.stringify(request))).candidates.map((c) => `${c.choice}${c.pending ? "?" : ""}`);
  const a = list(world);
  const b = list(real);
  if (a.length === b.length && a.every((choice, index) => choice === b[index])) return null;
  return { code: "v2_candidates", detail: "the world offers other candidates than the real request", item: null, state_class: null, redraw: true };
}

/** The conditions of a world or of the reader, by id. */
export interface ConditionSnapshot {
  weather: string | null;
  terrain: string | null;
  field: readonly string[];
  own: Readonly<Record<string, number>>;
  opponent: Readonly<Record<string, number>>;
}

const sameKeys = (a: Readonly<Record<string, number>>, b: Readonly<Record<string, number>>): boolean => {
  const [ka, kb] = [Object.keys(a).sort(), Object.keys(b).sort()];
  return ka.length === kb.length && ka.every((key, index) => key === kb[index] && a[key] === b[key]);
};

/** V4, first half: field and side conditions (and hazard layers) equal the reader's. */
export function v4Conditions(world: ConditionSnapshot, shown: ConditionSnapshot): CheckFailure | null {
  const equal =
    world.weather === shown.weather &&
    world.terrain === shown.terrain &&
    [...world.field].sort().join() === [...shown.field].sort().join() &&
    sameKeys(world.own, shown.own) &&
    sameKeys(world.opponent, shown.opponent);
  return equal ? null : { code: "v4_conditions", detail: "the world's field or side conditions differ from the reader's", item: "condition_duration", state_class: "W", redraw: false };
}

/** A remaining duration the writers computed from the log. */
export interface Remainder {
  what: "weather" | "terrain" | "field" | "side" | "volatile";
  remaining: number;
}

/** V4, second half: every effect the log still shows has turns left (the log says it has not run out). A condition that has is evidence against the sampled item of its source. */
export function v4Durations(remainders: readonly Remainder[]): CheckFailure | null {
  const gone = remainders.find((entry) => entry.remaining < 1);
  if (gone === undefined) return null;
  const item: StateItemId = gone.what === "volatile" ? "volatile_duration" : "condition_duration";
  return { code: "v4_duration", detail: `a ${gone.what} the log still shows has run out under the world's sets`, item, state_class: "W", redraw: true };
}
