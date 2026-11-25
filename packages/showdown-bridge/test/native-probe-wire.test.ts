import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import type { PokemonSet } from "../src/engine.ts";
import { type MechanicsQuery, readMechanicsQuery } from "../src/mechanics-query.ts";
import { hasHistory, NativeProtocolError, readResponse, wireQuery, wireSet } from "../src/native-probe-wire.ts";

const FORMAT = "gen9championsvgc2026regmc";
const fixture = JSON.parse(readFileSync(new URL(`../../../fixtures/mechanics/${FORMAT}/damage/cases.json`, import.meta.url), "utf8")) as { cases: { query: unknown }[] };
const damage = (): MechanicsQuery => structuredClone(readMechanicsQuery(fixture.cases[0]?.query)) as MechanicsQuery;

const set = { name: "", species: "Garchomp", item: "", ability: "Rough Skin", moves: ["Dragon Claw"], nature: "Adamant", gender: "M", evs: { hp: 32, atk: 32, def: 0, spa: 0, spd: 0, spe: 2 }, ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 }, level: 50 } as PokemonSet;

describe("the native probe's wire form", () => {
  it("sends a validated set by name with its Stat Points in the engine's order", () => {
    expect(wireSet(set)).toEqual({ species: "Garchomp", item: "", ability: "Rough Skin", moves: ["Dragon Claw"], nature: "Adamant", gender: "M", points: [32, 32, 0, 0, 0, 2], level: 50 });
  });

  it("numbers the sets of a damage query and drops what the layer does not read", () => {
    const query = damage().query;
    const wire = wireQuery(query, [7, 9]);
    expect(wire).toMatchObject({ kind: "damage", attacker: { set: 7 }, defender: { set: 9 }, move: "dragonclaw" });
    expect(JSON.stringify(wire)).not.toContain("Garchomp");
  });

  it("numbers the sets of a speed order in the contract's position order and a stats query's set", () => {
    const side = (set: string) => ({ set, boosts: { atk: 0, def: 0, spa: 0, spd: 0, spe: 0 }, status: null, hp: null, mega: false });
    const state = { weather: null, terrain: null, trick_room: false, sides: [{ aurora_veil: false, light_screen: false, reflect: false, tailwind: false }, { aurora_veil: false, light_screen: false, reflect: false, tailwind: false }] };
    const speed = { kind: "speed_order", sides: [[side("a"), side("b")], [side("c")]], actions: [{ side: 0, position: 0, move: null, mega: false }], state } as unknown as MechanicsQuery["query"];
    expect(wireQuery(speed, [4, 5, 6])).toMatchObject({ kind: "speed_order", sides: [[{ set: 4 }, { set: 5 }], [{ set: 6 }]] });
    expect(wireQuery({ kind: "stats", set: "x" }, [3])).toEqual({ kind: "stats", set: 3 });
  });

  it("does not send a move outcome", () => {
    const outcome = { kind: "move_outcome" } as unknown as MechanicsQuery["query"];
    expect(wireQuery(outcome, [])).toBeNull();
    expect(hasHistory(outcome)).toBe(false);
  });

  it("finds a history block at every place it can be", () => {
    const query = damage().query;
    expect(hasHistory(query)).toBe(false);
    const withBattle = structuredClone(query) as Extract<MechanicsQuery["query"], { kind: "damage" }>;
    (withBattle as { history?: unknown }).history = { turn: 1 };
    expect(hasHistory(withBattle)).toBe(true);
    const withAttacker = structuredClone(query) as Extract<MechanicsQuery["query"], { kind: "damage" }>;
    (withAttacker.attacker as { history?: unknown }).history = {};
    expect(hasHistory(withAttacker)).toBe(true);
    const withDefender = structuredClone(query) as Extract<MechanicsQuery["query"], { kind: "damage" }>;
    (withDefender.defender as { history?: unknown }).history = {};
    expect(hasHistory(withDefender)).toBe(true);
    expect(hasHistory({ kind: "stats", set: "x" })).toBe(false);
    const speed = { kind: "speed_order", sides: [[{ history: {} }], [{}]], actions: [], state: {} } as unknown as MechanicsQuery["query"];
    expect(hasHistory(speed)).toBe(true);
    const speedPlain = { kind: "speed_order", sides: [[{}], [{}]], actions: [], state: {} } as unknown as MechanicsQuery["query"];
    expect(hasHistory(speedPlain)).toBe(false);
  });

  it("reads a response frame and refuses what is not one", () => {
    const frame = (value: unknown) => Buffer.from(JSON.stringify(value));
    expect(readResponse<{ results: unknown[] }>(frame({ results: [] }), "results")).toEqual({ results: [] });
    expect(() => readResponse(Buffer.from("nope"), "results")).toThrow(NativeProtocolError);
    expect(() => readResponse(frame(3), "results")).toThrow("not an object");
    expect(() => readResponse(frame(null), "results")).toThrow("not an object");
    expect(() => readResponse(frame({ error: "bad frame" }), "results")).toThrow("refused the frame: bad frame");
    expect(() => readResponse(frame({ other: 1 }), "results")).toThrow("has no results");
  });
});
