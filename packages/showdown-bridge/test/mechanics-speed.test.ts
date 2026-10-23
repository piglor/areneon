/* eslint-disable @typescript-eslint/no-explicit-any */
import { beforeAll, describe, expect, it } from "vitest";
import { jcs, type Json } from "../src/canonical-json.ts";
import type { Engine } from "../src/engine.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { loadEngine } from "../src/index.ts";
import { loadMechanicsConfig, type MechanicsConfig, mechanicsConfigHash } from "../src/mechanics-config.ts";
import { MechanicsProbeError, probeMechanics } from "../src/mechanics-probe.ts";
import { MechanicsQueryError } from "../src/mechanics-query.ts";
import { readMechanicsResult } from "../src/mechanics-result.ts";
import { fullSpeedQuery, SPEED_FIXTURE, type SpeedCase } from "./mechanics-speed-cases.ts";

// ADR-016 slice 2: the speed_order probe against hand-worked fixtures. The
// expected numbers in fixtures/mechanics/<format>/speed/cases.json were
// worked on paper from the stat formula and the modifier rules (README.md
// beside it), not read from the probe.

let engine: Engine;
let config: MechanicsConfig;
beforeAll(() => {
  engine = loadEngine();
  config = loadMechanicsConfig(GOLDEN_FORMAT);
});

const probe = (speedCase: SpeedCase, change?: (query: Record<string, any>) => void, context = { engine, config }) => probeMechanics(context, fullSpeedQuery(SPEED_FIXTURE, speedCase, change));
const byId = (id: string): SpeedCase => SPEED_FIXTURE.cases.find((candidate) => candidate.id === id) as SpeedCase;

describe("the hand-worked fixtures", () => {
  it("covers the cases the ticket names", () => {
    const ids = SPEED_FIXTURE.cases.map((candidate) => candidate.id);
    expect(ids).toEqual(expect.arrayContaining(["priority-brackets", "exact-tie", "tailwind-doubles-speed", "trick-room-reverses", "trick-room-speed-over-1809-acts-last", "paralysis-halves-speed", "choice-scarf-multiplies-by-1-5", "mega-becomes-faster-and-moves-first", "quick-claw-is-unsupported"]));
    expect(new Set(ids).size).toBe(ids.length);
  });

  it.each(SPEED_FIXTURE.cases.map((speedCase) => [speedCase.id, speedCase] as const))("%s", (_id, speedCase) => {
    const { result, diagnostics } = probe(speedCase);
    expect(diagnostics).toEqual([]);
    expect(readMechanicsResult(result)).toEqual(result);
    expect(result.gap_ids).toEqual(speedCase.expected.gap_ids);
    expect(result.mechanics_config_hash).toBe(mechanicsConfigHash(config));
    if ("unsupported" in speedCase.expected) {
      expect(result).toMatchObject({ kind: "unsupported", ...speedCase.expected.unsupported });
      return;
    }
    if (result.kind !== "computed" || result.answer.kind !== "speed_order") throw new Error(`not computed: ${JSON.stringify(result)}`);
    expect(result.answer.order_basis).toBe("turn_start");
    expect(result.answer.actions.map((action) => ({ speed: action.speed, priority: action.priority, order: action.order }))).toEqual(speedCase.expected.actions);
    expect(result.answer.actions.map((action) => [action.side, action.position, action.mega])).toEqual(speedCase.query.actions.map((action) => [action.side, action.position, action.mega ?? false]));
    expect(result.answer.groups).toEqual(speedCase.expected.groups);
  });

  it("names the Mega row only when a Mega is involved", () => {
    expect(probe(byId("no-mega-control")).result.gap_ids).toEqual(["paralysis", "trick-room"]);
    expect(probe(byId("already-evolved")).result.gap_ids).toEqual(["mega-evolution", "paralysis", "trick-room"]);
  });

  it("gives the same JCS bytes on a rerun and carries the query's data class", () => {
    const run = (): string => jcs(probe(byId("mega-becomes-faster-and-moves-first")).result as unknown as Json);
    expect(run()).toBe(run());
    expect(probe(byId("tailwind-doubles-speed"), (query) => query).result.data_class).toBe("synthetic");
  });
});

describe("the engine ids the probe names exist in the pinned data", () => {
  it("side conditions and Trick Room are engine conditions", () => {
    const dex = (engine.sim.Dex as unknown as { mod(id: string): { conditions: { get(id: string): { exists: boolean; effectType: string } } } }).mod("champions");
    for (const id of ["auroraveil", "lightscreen", "reflect", "tailwind", "trickroom"]) expect(dex.conditions.get(id).exists, id).toBe(true);
  });
});

describe("what the declared state does not change", () => {
  it("screens, a terrain and a declared HP do not move the order", () => {
    const base = byId("choice-scarf-multiplies-by-1-5");
    const plain = probe(base).result;
    const loaded = probe(base, (query) => {
      query.state.terrain = "grassyterrain";
      query.state.sides = query.state.sides.map(() => ({ aurora_veil: true, light_screen: true, reflect: true, tailwind: false }));
      query.sides[0][0].hp = 40;
    }).result;
    expect(loaded).toMatchObject({ kind: "computed" });
    expect((loaded as { answer: unknown }).answer).toEqual((plain as { answer: unknown }).answer);
  });
});

describe("an engine draw is unsupported, never a guess", () => {
  it("a declared sleep draws its length: random_effect at the Pokemon", () => {
    const { result } = probe(byId("paralysis-halves-speed"), (query) => (query.sides[1][0].status = "slp"));
    expect(result).toMatchObject({ kind: "unsupported", reason: "random_effect", position: 1 });
  });
});

// Hawlucha with Limber (the assumed set of public-main-004's game 604): its ability blocks paralysis.
const LIMBER = "Hawlucha (F)  \nAbility: Limber  \nLevel: 50  \nEVs: 11 HP / 11 Atk / 11 Def / 11 SpA / 11 SpD / 11 Spe  \nHardy Nature  \n- Brick Break  \n\n";

describe("a declared status is an observed fact", () => {
  it("is taken whatever the type: Electric Pikachu declared paralysed (a Pokemon the type would shield) is computed", () => {
    const speedCase = byId("choice-scarf-multiplies-by-1-5");
    const { result } = probe({ ...speedCase, query: { ...speedCase.query, sides: [[{ set: "pikachu" }], speedCase.query.sides[1] as never] } }, (query) => (query.sides[0][0].status = "par"));
    expect(result).toMatchObject({ kind: "computed" });
  });
});

describe("a declared state or action the engine refuses is no result", () => {
  const refused: [string, string, (query: Record<string, any>) => void][] = [
    ["a weather that is not one", "not an engine weather", (query) => (query.state.weather = "tailwind")],
    ["a terrain that is not one", "not an engine terrain", (query) => (query.state.terrain = "sunnyday")],
    ["a status that is not one", "not an engine status", (query) => (query.sides[0][0].status = "sunnyday")],
    ["a status the Pokemon's ability blocks", "refused the declared status", (query) => Object.assign(query.sides[0][0], { status: "par", set: LIMBER })],
    ["an HP above the maximum", "exceeds the maximum", (query) => (query.sides[0][0].hp = 9999)],
    ["a move the engine does not have", "does not have", (query) => (query.actions[0].move = "nosuchmove")],
    ["a switch that Mega Evolves", "cannot Mega Evolve", (query) => Object.assign(query.actions[0], { move: null, mega: true })],
    ["a Mega Evolution without a Mega Stone", "cannot Mega Evolve", (query) => (query.actions[0].mega = true)],
    ["an already evolved Pokemon that cannot evolve", "cannot Mega Evolve", (query) => (query.sides[0][0].mega = true)],
    ["a move that needs an action of its own first", "needs more than a move", (query) => (query.actions[0].move = "focuspunch")],
  ];
  it.each(refused)("%s", (_name, words, change) => {
    // Pikachu is Electric: it cannot be paralysed, so the status case uses side 0 of this fixture
    const speedCase = byId("choice-scarf-multiplies-by-1-5");
    const query = (q: Record<string, any>): void => change(q);
    const build = () => probe({ ...speedCase, query: { ...speedCase.query, sides: [[{ set: "pikachu" }], speedCase.query.sides[1] as never] } }, query);
    expect(build).toThrow(MechanicsProbeError);
    expect(build).toThrow(words);
  });

  it("a priority that is not a whole number (a fractional-priority item)", () => {
    const real = engine.sim.Battle as unknown as new (...args: unknown[]) => { getActionSpeed(action: { choice: string; priority?: number }): void };
    const fractional = class extends real {
      override getActionSpeed(action: { choice: string; priority?: number }): void {
        super.getActionSpeed(action);
        if (action.choice === "move") action.priority = (action.priority ?? 0) - 0.1;
      }
    };
    const stand = { ...engine, sim: { ...engine.sim, Battle: fractional } } as unknown as Engine;
    expect(() => probe(byId("tailwind-doubles-speed"), undefined, { engine: stand, config })).toThrow("not a whole number");
  });
});

describe("the reader rules of a speed_order query", () => {
  const read = (change: (query: Record<string, any>) => void): unknown => fullSpeedQuery(SPEED_FIXTURE, byId("tailwind-doubles-speed"), change);
  it.each([
    ["an action for a Pokemon the sides do not declare", (query: Record<string, any>) => (query.actions[0].position = 1), "/actions/0 names a Pokemon the sides do not declare"],
    ["two actions for one Pokemon", (query: Record<string, any>) => (query.actions[1] = { ...query.actions[0] }), "/actions/1 repeats an acting Pokemon"],
    ["a Mega Evolution of a Pokemon already evolved", (query: Record<string, any>) => Object.assign(query.sides[0][0], { mega: true }) && (query.actions[0].mega = true), "/actions/0 Mega Evolves a Pokemon that already has"],
  ])("refuses %s", (_name, change, words) => {
    expect(() => read(change)).toThrow(MechanicsQueryError);
    expect(() => read(change)).toThrow(words);
  });
});
