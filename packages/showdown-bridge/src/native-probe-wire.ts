// The wire form of the native probe (the declared-state layer of the vendored fork's `ninjask::declared`, #1534): the sets
// the pinned validator accepted, the queries with each paste replaced by the index of a registered set, and the answers
// that come back. The native layer holds no table of game content and never judges a paste: this module sends only sets the
// pinned engine has validated (`probeEnvelope` ran), in the engine's own names.
//
// A query that holds something the native layer does not implement (a `history` block) is never sent: it is an abstention
// of the backend (`native-probe-backend.ts`), and the pinned probe answers it.
import type { PokemonSet } from "./engine.ts";
import type { MechanicsQuery } from "./mechanics-query.ts";
import type { Outcome } from "./mechanics-result.ts";

/** A set as the native layer registers it: engine names, Stat Points in the engine's stat order (HP, Atk, Def, SpA, SpD, Spe). */
export interface WireSet {
  species: string;
  item: string;
  ability: string;
  moves: string[];
  nature: string;
  gender: string;
  points: [number, number, number, number, number, number];
  level: number;
}

/** A validated set in the native layer's form. */
export function wireSet(set: PokemonSet): WireSet {
  const { hp, atk, def, spa, spd, spe } = set.evs;
  return { species: set.species, item: set.item, ability: set.ability, moves: [...set.moves], nature: set.nature, gender: set.gender, points: [hp, atk, def, spa, spd, spe], level: set.level };
}

type Kind = MechanicsQuery["query"];
type WithSet<T> = Omit<T, "set"> & { set: number };

/** A query as the native layer reads it: the contract's own shape with each `set` an index. */
export type WireQuery =
  | { kind: "stats"; set: number }
  | { kind: "damage"; attacker: WithSet<Extract<Kind, { kind: "damage" }>["attacker"]>; defender: WithSet<Extract<Kind, { kind: "damage" }>["defender"]>; move: string; state: Extract<Kind, { kind: "damage" }>["state"] }
  | { kind: "speed_order"; sides: WithSet<Extract<Kind, { kind: "speed_order" }>["sides"][number][number]>[][]; actions: Extract<Kind, { kind: "speed_order" }>["actions"]; state: Extract<Kind, { kind: "speed_order" }>["state"] };

/** Whether a query carries a history block (any fact the declared history declares), which the native layer does not implement. */
export function hasHistory(query: Kind): boolean {
  if (query.kind === "stats") return false;
  if (query.history !== undefined) return true;
  if (query.kind === "damage") return query.attacker.history !== undefined || query.defender.history !== undefined;
  if (query.kind === "speed_order") return query.sides.some((side) => side.some((c) => c.history !== undefined));
  // A kind the layer never answers (`move_outcome`) is no history block: the caller abstains on its kind.
  return false;
}

/** The query in wire form, or null for a kind the native layer does not answer (`move_outcome`). `ids` are the registered ids of the declared sets in the contract's position order. */
export function wireQuery(query: Kind, ids: readonly number[]): WireQuery | null {
  if (query.kind === "stats") return { kind: "stats", set: ids[0] as number };
  if (query.kind === "damage") {
    const { set: _a, history: _ah, ...attacker } = query.attacker;
    const { set: _d, history: _dh, ...defender } = query.defender;
    return { kind: "damage", attacker: { ...attacker, set: ids[0] as number }, defender: { ...defender, set: ids[1] as number }, move: query.move, state: query.state };
  }
  if (query.kind === "speed_order") {
    let next = 0;
    const sides = query.sides.map((side) =>
      side.map((combatant) => {
        const { set: _s, history: _h, ...rest } = combatant;
        return { ...rest, set: ids[next++] as number };
      }),
    );
    return { kind: "speed_order", sides, actions: query.actions, state: query.state };
  }
  return null;
}

/** One native outcome: the answer part of a result, an `unsupported` the layer decided, or an abstention. */
export type NativeOutcome = { kind: "computed"; answer: Extract<Outcome, { kind: "computed" }>["answer"] } | { kind: "unsupported"; reason: "query_kind" | "random_effect"; position: number | null } | { kind: "abstain"; entity: string; why: string };

export interface SetsResponse {
  failed: [number, string][];
}
export interface BatchResponse {
  results: NativeOutcome[];
}

/** A frame the native process answered with an error (or that is not a response at all): a defect of the transport, never an answer. */
export class NativeProtocolError extends Error {
  override name = "NativeProtocolError";
}

/** Parse a response frame; an `error` member or a missing member is a protocol error. */
export function readResponse<T extends object>(frame: Buffer, member: keyof T & string): T {
  let value: unknown;
  try {
    value = JSON.parse(frame.toString("utf8"));
  } catch {
    throw new NativeProtocolError("the native probe sent a frame that is not JSON");
  }
  if (typeof value !== "object" || value === null) throw new NativeProtocolError("the native probe sent a frame that is not an object");
  if ("error" in value) throw new NativeProtocolError(`the native probe refused the frame: ${String((value as { error: unknown }).error)}`);
  if (!(member in value)) throw new NativeProtocolError(`the native probe's response has no ${member}`);
  return value as T;
}
