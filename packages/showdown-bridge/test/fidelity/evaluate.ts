/* eslint-disable @typescript-eslint/no-explicit-any */
// Playing the oracle over decisions: one TurnRecord per decision and arm (ADR-026 slice F). An ARM is a world-writers
// record plus an optional id under test: the committed record (`base`), the committed record admitting one volatile
// (`volatile:<id>`), or admitting one start-trigger id (`trigger:<id>`). Differences are classified against the state
// items the built world holds.

import type { Engine } from "../../src/engine.ts";
import type { WorldContext, WorldRefusal } from "../../src/world-builder.ts";
import { createWorldBuilder } from "../../src/world-builder.ts";
import type { WorldWriters } from "../../src/world-writers.ts";
import { type Comparison, candidatesOf, compare, copyRandomState, type Decision, revive, stride } from "./oracle.ts";

export interface Options {
  /** Own candidates tried per decision in F2 (a fixed stride sample of the canonical list). */
  k: number;
  /** Opponent replies tried per own candidate in F2. */
  m: number;
}

export interface Arm {
  /** `base`, `volatile:<id>`, `trigger:<id>`, `trigger-probe`. */
  name: string;
  writers: WorldWriters;
  /** The id under test, for the arms that test one. */
  id: string | null;
  /** Names of the Pokemon that carry the id in the reader's state at the decision (the carriers whose action exercises it). */
  carriers?: (d: Decision) => string[];
}

export interface TurnRecord {
  battle: string;
  player: string;
  turn: number;
  arm: string;
  /** State items present in the built world (W writers that wrote, X items resampled), plus the arm's id. */
  items: string[];
  /** Some X item is resampled in this world: the turn is X-class and kept out of every other count. */
  x: boolean;
  /** Names of the carriers of the arm's id. */
  carriers: string[];
  f1: Comparison | null;
  /** One comparison per tried (own candidate, opponent reply) pair. */
  f2: Comparison[];
}

export type Outcome = { kind: "turn"; record: TurnRecord; written: readonly string[] } | { kind: "refused"; arm: string; refusal: WorldRefusal };

const cache = new WeakMap<Decision, { own: string[]; foe: string[] }>();

function candidates(engine: Engine, d: Decision): { own: string[]; foe: string[] } {
  let found = cache.get(d);
  if (found === undefined) {
    const fork = revive(engine, d.fork);
    const request = (id: string): unknown => fork.sides.find((s: any) => s.id === id).activeRequest;
    found = { own: candidatesOf(request(d.player)), foe: candidatesOf(request(d.player === "p1" ? "p2" : "p1")) };
    cache.set(d, found);
  }
  return found;
}

export function evaluate(engine: Engine, context: WorldContext, arm: Arm, d: Decision, options: Options, builders: Map<string, ReturnType<typeof createWorldBuilder>>): Outcome {
  let builder = builders.get(arm.name);
  if (builder === undefined) {
    builder = createWorldBuilder({ ...context, writers: arm.writers });
    builders.set(arm.name, builder);
  }
  const result = builder.build(d.input);
  if (result.kind !== "world") return { kind: "refused", arm: arm.name, refusal: result.refusal };
  const world = result.world;
  copyRandomState(world.battle, revive(engine, d.fork));
  const state = world.battle.toJSON();
  const items = [...world.written, ...world.resampled.map((r) => r.item)] as string[];
  if (arm.id !== null) items.push(arm.id);
  const record: TurnRecord = {
    battle: d.battle_id,
    player: d.player,
    turn: d.turn,
    arm: arm.name,
    items: [...new Set(items)].sort(),
    x: world.resampled.length > 0,
    carriers: arm.carriers?.(d) ?? [],
    f1: d.recorded === null ? null : compare(engine, d, state, d.recorded.own, d.recorded.foe),
    f2: [],
  };
  if (options.k > 0 && options.m > 0) {
    const cands = candidates(engine, d);
    for (const own of stride(cands.own, options.k)) for (const foe of stride(cands.foe, options.m)) record.f2.push(compare(engine, d, state, own, foe));
  }
  return { kind: "turn", record, written: world.written };
}
