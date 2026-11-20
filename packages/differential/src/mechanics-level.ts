// Level 1, the mechanics-query level (ADR-030 step 3's fidelity gate): `In` is a mechanics query, `Out` its MechanicsResult.
// The entity rule and the exercising rule are reviewed rules held here; the facts they read come from the bridge, which parses
// each query's own sets with the pinned engine (`importPaste`) and hands over a `MechanicsDescriptor` of engine ids, so no list
// of game content is written in this package.
//
// Entities of a query (ADR-028 section 5, ADR-030 rule 2): the query kind; for every Pokemon of the query its species forme,
// ability, item, the moves of its set, its declared status and the volatiles its history declares; the moves the query's
// actions use; the declared weather, terrain, Trick Room, side conditions and pending slot conditions.
//
// Exercising (ADR-026 slice F's rule: present AND read by the answer), from the ORACLE's outcome:
// - always the query kind;
// - a refusal (the probe threw) or a `refused` result exercises nothing else;
// - an `unsupported` result whose reason the move itself decides (`query_kind`: a status, fixed-damage or multi-hit move) also
//   exercises the moves the actions use; any other reason (a random effect an item or ability may cause, an unknown history fact,
//   the scenario's rule table) exercises the query kind alone;
// - a computed `stats` answer: the set's species forme;
// - a computed `damage` answer: the used move, both Pokemon's species forme, ability, item and status, the declared field and
//   side conditions;
// - a computed `speed_order` answer: the used moves, every combatant's species forme, ability, item and status, the field and
//   side conditions;
// - a computed `move_outcome` answer: as speed_order, plus every declared volatile and slot condition (the engine's move step
//   reads them).
// A set's moves that no action uses, and volatiles outside a move outcome, are present but not exercised (conservative: no
// evidence is claimed for what the answer may not have read).
import type { MechanicsQuery } from "@areneon/schema/types/v1/mechanics-query.ts";
import type { MechanicsResult } from "@areneon/schema/types/v1/mechanics-result.ts";
import type { Json } from "./canonical.ts";
import type { Level, Settled } from "./backend.ts";

/** One Pokemon of a query, in engine ids (the bridge reads them from the query's paste with the pinned engine). */
export interface MonFacts {
  species: string;
  ability: string;
  item: string | null;
  moves: string[];
  status: string | null;
  volatiles: string[];
}

/** What a mechanics query holds, in engine ids. */
export interface MechanicsDescriptor {
  kind: MechanicsQuery["query"]["kind"];
  mons: MonFacts[];
  /** The moves the query's actions use (a damage query's move; every action's move; the asked action's move). */
  used_moves: string[];
  weather: string | null;
  terrain: string | null;
  trick_room: boolean;
  /** Declared side conditions, by the query schema's names (`reflect`, `tailwind`, ...). */
  side_conditions: string[];
  /** Pending slot conditions the history declares. */
  slot_conditions: string[];
}

const ID = /^[a-z0-9_]+$/;

export class EntityError extends Error {
  override name = "EntityError";
}

const id = (kind: string, value: string): string => {
  if (!ID.test(value)) throw new EntityError(`an entity id is an engine id: ${kind}`);
  return `${kind}:${value}`;
};

const sorted = (values: Iterable<string>): string[] => [...new Set(values)].sort();

const monCore = (mon: MonFacts): string[] => [id("species", mon.species), id("ability", mon.ability), ...(mon.item === null ? [] : [id("item", mon.item)]), ...(mon.status === null ? [] : [id("status", mon.status)])];

const fieldOf = (d: MechanicsDescriptor): string[] => [
  ...(d.weather === null ? [] : [id("weather", d.weather)]),
  ...(d.terrain === null ? [] : [id("terrain", d.terrain)]),
  ...(d.trick_room ? ["field:trickroom"] : []),
  ...d.side_conditions.map((s) => id("side", s)),
];

/** Every entity of a query. */
export function mechanicsEntities(d: MechanicsDescriptor): string[] {
  return sorted([
    id("kind", d.kind),
    ...d.mons.flatMap((mon) => [...monCore(mon), ...mon.moves.map((m) => id("move", m)), ...mon.volatiles.map((v) => id("volatile", v))]),
    ...d.used_moves.map((m) => id("move", m)),
    ...fieldOf(d),
    ...d.slot_conditions.map((s) => id("slot", s)),
  ]);
}

/** The entities the oracle's outcome read (the rule above). */
export function mechanicsExercised(d: MechanicsDescriptor, oracle: Settled<MechanicsResult>): string[] {
  const kind = id("kind", d.kind);
  if (!oracle.ok || oracle.result.kind === "refused") return [kind];
  const moves = d.used_moves.map((m) => id("move", m));
  if (oracle.result.kind === "unsupported") return oracle.result.reason === "query_kind" ? sorted([kind, ...moves]) : [kind];
  if (d.kind === "stats") return sorted([kind, ...d.mons.map((mon) => id("species", mon.species))]);
  const read = [kind, ...moves, ...d.mons.flatMap(monCore), ...fieldOf(d)];
  if (d.kind === "move_outcome") read.push(...d.mons.flatMap((mon) => mon.volatiles.map((v) => id("volatile", v))), ...d.slot_conditions.map((s) => id("slot", s)));
  return sorted(read);
}

/** What the bridge supplies for level 1: the descriptor of a query, whether a result answers a query, and a refusal's fixed form. */
export interface MechanicsFacts<In> {
  describe(input: In): MechanicsDescriptor;
  /** The query's hash, which every MechanicsResult names (`query_hash`). */
  queryHash(input: In): string;
  errorKey(error: unknown): string;
}

/** The level-1 adapter over the bridge's facts. A descriptor is computed once per input object. */
export function mechanicsLevel<In extends object>(facts: MechanicsFacts<In>): Level<In, MechanicsResult> {
  const seen = new WeakMap<In, MechanicsDescriptor>();
  const describe = (input: In): MechanicsDescriptor => {
    let d = seen.get(input);
    if (d === undefined) {
      d = facts.describe(input);
      seen.set(input, d);
    }
    return d;
  };
  return {
    name: "mechanics-query",
    entities: (input) => mechanicsEntities(describe(input)),
    exercised: (input, oracle) => mechanicsExercised(describe(input), oracle),
    key: (input) => facts.queryHash(input),
    answers: (input, output) => output.query_hash === facts.queryHash(input),
    bytes: (output) => output as unknown as Json,
    errorKey: (error) => facts.errorKey(error),
  };
}
