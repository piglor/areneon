// Outcome rows the engine's own data proves (ADR-022 erratum E-22.5.10, #1353): a move part whose answer cannot differ from
// "passes, hit 1,000,000" is produced without a probe query. The decision uses the engine's data for the move, the user's ability
// and item, and the declared state of the user, never a list of moves:
// - the move draws no accuracy (`accuracy: true`) and is no one-hit KO move;
// - the move has no handler of its own (no `on...` key, no `...Callback`), so nothing in it can fail, draw or read the queue before its hit step;
//   it is no stalling move, no charge move, and its priority is 0 (a foe's priority-blocking ability reads the priority);
// - it aims at no foe and no single Pokemon (`self`, a side, the field), so no immunity or protection step can remove its target;
// - the user's ability and item have no handler of the engine's move path (`MOVE_PATH_HANDLERS`);
// - no Pokemon of the query holds an ability or item with `onFractionalPriority` (the engine draws when it orders actions: `random_effect`). That is the
//   only foe-side effect looked for: it is enough today because the move has priority 0 and aims at no foe, so no foe ability or item reads it otherwise;
// - no Pokemon of the query has a status or a declared volatile, none of the declared facts is unknown and no slot condition is pending:
//   the engine's declaration of a sleep or a confusion draws (the whole query is then `random_effect`), and a fact of the user's may name a
//   condition whose handler blocks the move.
// Everything not proven is asked. A test over the committed corpora compares every skipped row with the probe's answer.
import { MOVE_PATH_HANDLERS } from "./outcome-config.ts";

/** The engine's target types that name no foe and no single Pokemon: the move's target steps cannot refuse it. */
export const UNTARGETED = ["self", "allySide", "allyTeam", "allies", "all"] as const;

type Data = Record<string, unknown>;

/** True when the engine data of an effect has a handler key (`onX`) or a callback key. */
const hasHandler = (data: Data): boolean => Object.keys(data).some((key) => /^on[A-Z]/.test(key) || key.endsWith("Callback"));
/** The events of the move path an ability or item can answer, with the `Any`, `Foe`, `Ally` and `Source` prefixes of the engine's handler names. */
const PATH_EVENTS = [...MOVE_PATH_HANDLERS.map((name) => name.slice(2)), "PrepareHit", "TryHit", "TryImmunity", "Immunity", "Invulnerability", "Accuracy"];
const hasPathHandler = (data: Data): boolean =>
  Object.keys(data).some((key) => {
    const match = /^on(?:Any|Foe|Ally|Source)?([A-Z]\w*)$/.exec(key);
    return match !== null && PATH_EVENTS.includes(match[1] as string);
  });

/** Does a declared block (a Pokemon's facts or the battle's) hold an unknown fact, a volatile or a pending slot condition. */
function declaresOrUnknown(block: object | undefined): boolean {
  if (block === undefined) return false;
  for (const [key, fact] of Object.entries(block as Record<string, unknown>)) {
    if (key === "sides" && Array.isArray(fact)) {
      if (fact.some((side) => declaresOrUnknown(side as object))) return true;
      continue;
    }
    const entry = fact as { kind?: string; value?: unknown } | undefined;
    if (entry === undefined || entry === null || typeof entry !== "object" || entry.kind === undefined) continue;
    if (entry.kind === "unknown") return true;
    if ((key === "volatiles" || key === "pending_slots") && Array.isArray(entry.value) && entry.value.length > 0) return true;
  }
  return false;
}

export interface SkipInput {
  /** The engine's data of the asked move. */
  move: Data;
  /** The engine's data of the user's ability and item (undefined for none). */
  ability: Data | undefined;
  item: Data | undefined;
  /** The engine's data of every Pokemon's ability and item in the query (the engine draws when it orders actions under a fractional-priority item). */
  everyone?: readonly Data[];
  /** Every Pokemon of the query as declared, and the query's battle history. */
  combatants: readonly { status: string | null; history: object | undefined }[];
  battle: object | undefined;
  /** The query's history is marked unknown as a whole. */
  marked: boolean;
}

/** True when the engine's data proves the asked move passes with chance 1 and hit chance 1: no query is needed. */
export function provenPasses({ move, ability, item, everyone = [], combatants, battle, marked }: SkipInput): boolean {
  if (everyone.some((effect) => effect.onFractionalPriority !== undefined)) return false;
  if (marked || declaresOrUnknown(battle) || combatants.some((each) => each.status !== null || declaresOrUnknown(each.history))) return false;
  if (move.accuracy !== true || move.ohko !== undefined || move.stallingMove !== undefined || move.priority !== 0) return false;
  if (!(UNTARGETED as readonly string[]).includes(move.target as string)) return false;
  if (hasHandler(move)) return false;
  const flags = move.flags as Record<string, unknown> | undefined;
  if (flags?.charge !== undefined || flags?.recharge !== undefined) return false;
  for (const effect of [ability, item]) if (effect !== undefined && hasPathHandler(effect)) return false;
  return true;
}
