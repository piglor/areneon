/* eslint-disable @typescript-eslint/no-explicit-any */
import { beforeAll, describe, expect, it } from "vitest";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { type Engine, loadEngine, type PokemonSet } from "../src/engine.ts";
import { loadMechanicsConfig } from "../src/mechanics-config.ts";
import { checkScenario } from "../src/mechanics-scenario.ts";
import { loadRandomCorpus, playRandomBattle, type RandomBattleFile } from "../src/random-corpus.ts";
import { randomLegalActor } from "../src/random-legal.ts";
import type { Actor } from "../src/runner.ts";
import { type WorldContext, createWorldBuilder } from "../src/world-builder.ts";
import { type WorldWriters, loadWorldWriters } from "../src/world-writers.ts";
import { inputAtLastRequest, toId } from "./world-builder-support.ts";

// ADR-026 slice W part 2, Done-when: each W writer has a unit test against a seeded engine battle. The oracle
// is the LIVE battle: `runBattle` plays a committed seeded battle (its random-legal actors), and at every own
// move request this test reads the live battle's state (a test-only read, as the fidelity oracle of slice F
// will fork it) and compares it with the world built from the player's own view and the opponent's TRUE
// brought sets. Every W item the world writes must equal the live engine's value; X items are only counted.

/** The committed record gives toxic_stage R (too few equal oracle turns); this live-state test is the evidence for the writer itself, so it compares with it as W. */
const liveRecord = (record: WorldWriters): WorldWriters => {
  const copy = structuredClone(record);
  (copy.items.find((item) => item.id === "toxic_stage") as { class: string }).class = "W";
  return copy;
};

let engine: Engine;
let context: WorldContext;
beforeAll(() => {
  engine = loadEngine();
  context = { engine, format_id: GOLDEN_FORMAT, scenario_id: checkScenario(engine, loadMechanicsConfig(GOLDEN_FORMAT)), hp_rounding: "ceil", writers: liveRecord(loadWorldWriters(engine, GOLDEN_FORMAT)) };
});

/** Which comparisons ran, by what they compared. */
const compared = new Map<string, number>();
const tick = (what: string): void => void compared.set(what, (compared.get(what) ?? 0) + 1);

function same(world: any, live: any, seat: number, where: string, extra: string[]): void {
  expect(world.turn, `${where}: turn`).toBe(live.turn);
  tick("turn");
  for (const side of [seat, 1 - seat]) {
    const [w, l] = [world.sides[side], live.sides[side]];
    expect(w.totalFainted, `${where}: totalFainted`).toBe(l.totalFainted);
    expect(!!w.faintedLastTurn, `${where}: faintedLastTurn`).toBe(!!l.faintedLastTurn);
    tick("side faint counters");
    if (l.faintedLastTurn) tick("faintedLastTurn");
    // The opponent's bench order is the world's draw: pair Pokemon by name.
    for (const lm of l.pokemon) {
      const wm = w.pokemon.find((m: any) => m.name === lm.name);
      expect(wm, `${where}: ${lm.name} is in the world`).toBeDefined();
      const at = `${where}: ${lm.name}`;
      expect(wm.item, `${at}: item`).toBe(lm.item);
      if (lm.item === "") tick("item gone");
      if (lm.fainted) {
        expect(wm.fainted, `${at}: fainted`).toBe(true);
        continue;
      }
      expect(wm.status, `${at}: status`).toBe(lm.status);
      // The engine resets the stage when a Pokemon switches IN, so a benched Pokemon's stage is never read: only a Pokemon in play is compared.
      if (lm.status === "tox") {
        if (lm.isActive) expect(wm.statusState.stage, `${at}: Toxic stage`).toBe(lm.statusState.stage);
        tick(lm.isActive ? "Toxic stage" : "Toxic on the bench");
      }
      if (lm.status !== "" && !lm.isActive) tick("status on the bench");
      if (!lm.isActive) continue;
      expect(wm.activeTurns, `${at}: activeTurns`).toBe(lm.activeTurns);
      expect(wm.activeMoveActions, `${at}: activeMoveActions`).toBe(lm.activeMoveActions);
      if (lm.activeTurns >= 3) tick("activeTurns >= 3");
      expect(wm.lastMove?.id ?? null, `${at}: lastMove`).toBe(lm.lastMove?.id ?? null);
      if (lm.lastMove !== null) tick("lastMove");
      for (const id of ["stall", "choicelock"]) {
        expect(!!wm.volatiles[id], `${at}: ${id}`).toBe(!!lm.volatiles[id]);
        if (lm.volatiles[id]) {
          tick(id);
          expect(wm.volatiles[id].duration, `${at}: ${id} duration`).toBe(lm.volatiles[id].duration);
          if (id === "stall") expect(wm.volatiles[id].counter, `${at}: stall counter`).toBe(lm.volatiles[id].counter);
          if (id === "choicelock") expect(wm.volatiles[id].move, `${at}: lock move`).toBe(lm.volatiles[id].move);
        }
      }
      if (lm.volatiles.stall && lm.volatiles.stall.counter > 3) tick("stall counter > 3");
      for (const id of extra) {
        expect(!!wm.volatiles[id], `${at}: volatile ${id}`).toBe(!!lm.volatiles[id]);
        if (!lm.volatiles[id]) continue;
        tick(`volatile ${id}`);
        expect(wm.volatiles[id].duration, `${at}: ${id} remaining turns`).toBe(lm.volatiles[id].duration);
        if (lm.volatiles[id].move !== undefined) expect(wm.volatiles[id].move, `${at}: ${id} move`).toBe(lm.volatiles[id].move);
      }
    }
  }
  const [wf, lf] = [world.field, live.field];
  expect(wf.weather, `${where}: weather`).toBe(lf.weather);
  expect(wf.terrain, `${where}: terrain`).toBe(lf.terrain);
  if (lf.weather !== "") {
    expect(wf.weatherState.duration, `${where}: weather duration`).toBe(lf.weatherState.duration);
    tick("weather duration");
  }
  if (lf.terrain !== "") {
    expect(wf.terrainState.duration, `${where}: terrain duration`).toBe(lf.terrainState.duration);
    tick("terrain duration");
  }
  expect(Object.keys(wf.pseudoWeather).sort(), `${where}: field conditions`).toEqual(Object.keys(lf.pseudoWeather).sort());
  for (const id of Object.keys(lf.pseudoWeather)) {
    expect(wf.pseudoWeather[id].duration, `${where}: ${id} duration`).toBe(lf.pseudoWeather[id].duration);
    tick("field condition duration");
  }
  for (const side of [seat, 1 - seat]) {
    const [w, l] = [world.sides[side].sideConditions, live.sides[side].sideConditions];
    expect(Object.keys(w).sort(), `${where}: side conditions`).toEqual(Object.keys(l).sort());
    for (const id of Object.keys(l)) {
      expect(w[id].duration, `${where}: ${id} duration`).toBe(l[id].duration);
      expect(w[id].layers, `${where}: ${id} layers`).toBe(l[id].layers);
      tick("side condition duration");
    }
  }
}

async function replay(spec: RandomBattleFile, writers: WorldWriters = context.writers, admitted: readonly string[] = []): Promise<{ built: number; refused: number }> {
  let captured: any = null;
  const Base = engine.sim.BattleStream as unknown as new () => { battle: unknown };
  const capture = (stream: unknown): void => {
    captured = stream;
  };
  class Spy extends Base {
    constructor() {
      super();
      capture(this);
    }
  }
  const spy: Engine = { ...engine, sim: { ...engine.sim, BattleStream: Spy as never } };
  const builder = createWorldBuilder({ ...context, engine: spy, writers });
  let built = 0;
  let refused = 0;
  const actorFor = (seed: number, player: "p1" | "p2"): Actor => {
    const inner = randomLegalActor(seed);
    return (request, ctx) => {
      if (ctx !== undefined && request.phase === "move" && captured?.battle != null) {
        const live = captured.battle as any;
        const foe = player === "p1" ? "p2" : "p1";
        const seat = player === "p1" ? 0 : 1;
        const names = new Set(live.sides[1 - seat].pokemon.map((m: any) => m.name));
        const foeSets = (spec.teams[foe] as PokemonSet[]).filter((set) => names.has(set.name === "" ? set.species : set.name));
        const input = inputAtLastRequest(player, ctx.channel, spec.teams[player] as PokemonSet[], foeSets);
        if (input !== null) {
          const result = builder.build(input);
          if (result.kind === "world") {
            built += 1;
            same(result.world.battle, live, seat, `${spec.id}/${player}/turn ${input.state.turn}`, [...admitted]);
          } else refused += 1;
        }
      }
      return inner(request, ctx);
    };
  };
  const record = await playRandomBattle(spy, spec, (seed) => actorFor(seed, spec.actor_seeds.p1 === seed ? "p1" : "p2"));
  expect(record.turns).toBeGreaterThan(0);
  return { built, refused };
}

describe("every W writer equals the live engine's state at the decision", () => {
  it("over the committed seeded battles: Toxic stage, bench status, activeTurns and activeMoveActions, last move, Protect counter, choice lock, faint counters, a gone item, and the remaining turns of weather, terrain, field and side conditions", async () => {
    let built = 0;
    let refused = 0;
    for (const spec of loadRandomCorpus()) {
      const r = await replay(spec);
      built += r.built;
      refused += r.refused;
    }
    console.log(`world writers live: ${built} worlds compared with the live battle, ${refused} refused; compared ${JSON.stringify([...compared])}`);
    expect(built).toBeGreaterThan(300);
    // "stall counter > 3" is no longer required (#1186): the one corpus turn that showed it (b18) also holds a recharging Pokemon, so its world is now refused. The writer applies the engine's own `addVolatile` once per use, so the counter follows the engine for any streak.
    for (const what of ["Toxic stage", "status on the bench", "activeTurns >= 3", "lastMove", "stall", "side faint counters", "faintedLastTurn", "item gone", "weather duration", "side condition duration"]) {
      expect(compared.get(what) ?? 0, what).toBeGreaterThan(0);
    }
  }, 600_000);
});

describe("the fixed-duration volatiles an admitting record lists keep the turns the log leaves them", () => {
  it("add the engine's condition and count its duration down from the turn the log dates it (admitted in a test record, not in the committed one)", async () => {
    const ids = ["encore", "taunt", "throatchop", "yawn"];
    const writers = structuredClone(context.writers);
    writers.admitted = ids.map((id) => ({ kind: "volatile" as const, id, class: "W" as const }));
    let built = 0;
    for (const spec of loadRandomCorpus()) built += (await replay(spec, writers, ids)).built;
    console.log(`world writers live (admitted volatiles): ${built} worlds; compared ${JSON.stringify([...compared].filter(([k]) => k.startsWith("volatile")))}`);
    for (const id of ["taunt"]) expect(compared.get(`volatile ${id}`) ?? 0, id).toBeGreaterThan(0);
  }, 600_000);
});

describe("the choice lock", () => {
  it("is added with the move the log shows for a holder of the only Choice item the format has, and the world's request then disables the other moves like the real one (V2)", async () => {
    let built = 0;
    const specs = loadRandomCorpus();
    for (const [index, spec] of specs.entries()) {
      // Give one Pokemon of each side the Choice item (and take it from whoever held it): the committed teams hold none of it in play.
      const teams = structuredClone(spec.teams) as Record<"p1" | "p2", PokemonSet[]>;
      for (const side of ["p1", "p2"] as const) {
        const team = teams[side];
        for (const set of team) if (toId(set.item) === "choicescarf") set.item = "";
        (team[(index + (side === "p1" ? 0 : 3)) % team.length] as PokemonSet).item = "Choice Scarf";
      }
      built += (await replay({ ...spec, teams })).built;
    }
    console.log(`world writers live (choice lock): ${built} worlds; compared ${JSON.stringify([...compared].filter(([k]) => k === "choicelock"))}`);
    expect(compared.get("choicelock") ?? 0).toBeGreaterThan(0);
  }, 600_000);
});
