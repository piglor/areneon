// The level-1 entity and exercising rule, held on hand-built descriptors.
import { describe, expect, it } from "vitest";
import type { MechanicsResult } from "@areneon/schema/types/v1/mechanics-result.ts";
import { EntityError, type MechanicsDescriptor, mechanicsEntities, mechanicsExercised, mechanicsLevel, type MonFacts } from "../src/mechanics-level.ts";

const mon = (species: string, extra: Partial<MonFacts> = {}): MonFacts => ({ species, ability: `${species}ability`, item: null, moves: [], status: null, volatiles: [], ...extra });
const damage: MechanicsDescriptor = {
  kind: "damage",
  mons: [mon("garchomp", { item: "choiceband", moves: ["earthquake", "protect"], status: "brn", volatiles: ["taunt"] }), mon("rotomwash")],
  used_moves: ["earthquake"],
  weather: "raindance",
  terrain: null,
  trick_room: true,
  side_conditions: ["reflect"],
  slot_conditions: ["wish"],
};
const computed = (kind: string) => ({ ok: true as const, result: { kind: "computed", answer: { kind } } as unknown as MechanicsResult });
const of = (kind: string) => ({ ok: true as const, result: { kind } as unknown as MechanicsResult });

describe("mechanicsEntities", () => {
  it("names every species forme, ability, item, set move, status, volatile, used move, field and side and slot condition", () => {
    expect(mechanicsEntities(damage)).toEqual([
      "ability:garchompability",
      "ability:rotomwashability",
      "field:trickroom",
      "item:choiceband",
      "kind:damage",
      "move:earthquake",
      "move:protect",
      "side:reflect",
      "slot:wish",
      "species:garchomp",
      "species:rotomwash",
      "status:brn",
      "volatile:taunt",
      "weather:raindance",
    ]);
    expect(mechanicsEntities({ ...damage, weather: null, trick_room: false, terrain: "electricterrain", side_conditions: [], slot_conditions: [], mons: [mon("a")], used_moves: [] })).toEqual(["ability:aability", "kind:damage", "species:a", "terrain:electricterrain"]);
  });
  it("refuses an id that is not an engine id", () => {
    expect(() => mechanicsEntities({ ...damage, used_moves: ["Earthquake"] })).toThrow(EntityError);
  });
});

describe("mechanicsExercised", () => {
  it("reads only the kind on a refusal or a refused result, and the used moves on unsupported", () => {
    expect(mechanicsExercised(damage, { ok: false, error: new Error("x") })).toEqual(["kind:damage"]);
    expect(mechanicsExercised(damage, of("refused"))).toEqual(["kind:damage"]);
    const unsupported = (reason: string) => ({ ok: true as const, result: { kind: "unsupported", reason } as unknown as MechanicsResult });
    expect(mechanicsExercised(damage, unsupported("query_kind"))).toEqual(["kind:damage", "move:earthquake"]);
    for (const reason of ["random_effect", "history_unknown", "scenario_rule_table"]) expect(mechanicsExercised(damage, unsupported(reason))).toEqual(["kind:damage"]);
  });
  it("reads a damage answer's move, both Pokemon, the field and sides, never an unused set move or a volatile", () => {
    const read = mechanicsExercised(damage, computed("damage"));
    expect(read).toEqual(["ability:garchompability", "ability:rotomwashability", "field:trickroom", "item:choiceband", "kind:damage", "move:earthquake", "side:reflect", "species:garchomp", "species:rotomwash", "status:brn", "weather:raindance"]);
    expect(read).not.toContain("move:protect");
  });
  it("reads volatiles and slot conditions for a move outcome, species alone for stats", () => {
    expect(mechanicsExercised({ ...damage, kind: "move_outcome" }, computed("move_outcome"))).toEqual(expect.arrayContaining(["volatile:taunt", "slot:wish"]));
    expect(mechanicsExercised({ ...damage, kind: "speed_order" }, computed("speed_order"))).not.toContain("volatile:taunt");
    expect(mechanicsExercised({ ...damage, kind: "stats", used_moves: [] }, computed("stats"))).toEqual(["kind:stats", "species:garchomp", "species:rotomwash"]);
  });
});

describe("mechanicsLevel", () => {
  it("describes each input once and answers by the query hash", () => {
    let calls = 0;
    const level = mechanicsLevel<{ q: string }>({
      describe: () => {
        calls += 1;
        return damage;
      },
      queryHash: (input) => input.q,
      errorKey: (error) => (error as Error).message,
    });
    const input = { q: "h1" };
    expect(level.entities(input)).toContain("kind:damage");
    expect(level.exercised(input, of("refused"))).toEqual(["kind:damage"]);
    expect(calls).toBe(1);
    expect(level.key(input)).toBe("h1");
    expect(level.answers(input, { query_hash: "h1" } as MechanicsResult)).toBe(true);
    expect(level.answers(input, { query_hash: "h2" } as MechanicsResult)).toBe(false);
    expect(level.bytes({ query_hash: "h1" } as MechanicsResult)).toEqual({ query_hash: "h1" });
    expect(level.errorKey(new Error("w"))).toBe("w");
    expect(level.name).toBe("mechanics-query");
  });
});
