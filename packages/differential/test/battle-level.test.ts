// The battle-step entity and exercising rule, and the interaction classes, on hand-written steps.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { ClassesError, classesHash, classifyStep, readInteractionClasses } from "../src/battle-classes.ts";
import { stepFacts, teamEntities, toId } from "../src/battle-level.ts";
import type { BattleTrace } from "../src/battle-trace.ts";

const set = (name: string, species: string, ability: string, item: string, moves: string[]) => ({ name, species, ability, item, moves, nature: "Hardy", gender: "" as const, evs: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 }, ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 }, level: 50 });
const teams = { p1: [set("", "Garchomp", "Rough Skin", "Choice Scarf", ["Earthquake", "Protect"]), set("Nick", "Rotom-Wash", "Levitate", "", ["Hydro Pump"])], p2: [set("", "Kingambit", "", "Leftovers", ["Sucker Punch"])] };
const step = (omniscient: string[], draws: { method: string; args: string[]; result: string[] }[] = []) => ({ actions: [], draws, channels: { p1: [], p2: [] }, omniscient, requests: { p1: [], p2: [] }, state_sha256: "0".repeat(64) });
const trace = (steps: ReturnType<typeof step>[]) => ({ teams, steps }) as unknown as Pick<BattleTrace, "teams" | "steps">;

describe("the entity and exercising rule", () => {
  it("names every species, ability, item and move of both teams (empty ones aside)", () => {
    expect(teamEntities(trace([])).sort()).toEqual(["ability:levitate", "ability:roughskin", "item:choicescarf", "item:leftovers", "move:earthquake", "move:hydropump", "move:protect", "move:suckerpunch", "species:garchomp", "species:kingambit", "species:rotomwash"]);
    expect(toId("King's Shield")).toBe("kingsshield");
  });

  it("exercises what a step's lines read, with the carrier rule for the Pokemon that acted", () => {
    const facts = stepFacts(
      trace([
        step(["|switch|p1a: Garchomp|Garchomp, L50, M|100/100", "|switch|p1b: Nick|Rotom-Wash, L50|100/100", "|switch|p2a: Kingambit|Kingambit, L50, F|100/100"]),
        step([
          "|move|p1a: Garchomp|Earthquake|p2a: Kingambit|[spread] p2a,p1b",
          "|-immune|p1b: Nick|[from] ability: Levitate",
          "|cant|p2a: Kingambit|flinch",
          "|-start|p2a: Kingambit|Taunt",
          "|-activate|p1a: Garchomp|move: Protect",
          "|-sidestart|p1: Player|move: Tailwind",
          "|-sideend|p1: Player|Reflect",
          "|-fieldstart|move: Trick Room|[of] p1a: Garchomp",
          "|-weather|RainDance",
          "|-weather|none",
          "|-status|p2a: Kingambit|brn",
          "|-curestatus|p2a: Kingambit|brn",
          "|-mega|p1a: Garchomp|Garchomp|Garchompite",
          "|detailschange|p1a: Garchomp|Garchomp-Mega, L50, M",
          "|-ability|p1a: Garchomp|Sand Force",
          "|-enditem|p2a: Kingambit|Leftovers",
          "|-heal|p2a: Kingambit|50/100|[from] item: Leftovers",
          "|-damage|p1a: Garchomp|90/100|[from] psn",
          "|-start|p2a: Kingambit|",
          "|-sidestart|p2: Player|",
          "|-fieldstart|",
          "|-weather",
          "|move|p3a: Nobody|Splash|",
          "|turn|2",
        ]),
        step(["|cant|p1a: Garchomp|recharge", "|cant|p2b: Unknown|par"]),
      ]),
    );
    expect(facts[0]?.exercised).toEqual(["species:garchomp", "species:kingambit", "species:rotomwash"]);
    expect(facts[1]?.exercised).toEqual([
      "ability:levitate",
      "ability:roughskin",
      "ability:sandforce",
      "field:trickroom",
      "item:choicescarf",
      "item:garchompite",
      "item:leftovers",
      "move:earthquake",
      "move:protect",
      "move:splash",
      "side:reflect",
      "side:tailwind",
      "species:garchomp",
      "species:garchompmega",
      "species:kingambit",
      "status:brn",
      "volatile:psn",
      "volatile:taunt",
      "weather:raindance",
    ]);
    // A Pokemon the teams do not hold (p2b) has no set: only what its lines show; the shown forme follows a forme change.
    expect(facts[2]?.exercised).toEqual(["ability:roughskin", "item:choicescarf", "species:garchompmega"]);
    for (const f of facts) for (const id of f.exercised) expect(f.entities).toContain(id);
    expect(facts[2]?.entities).toContain("move:splash");
  });
});

describe("the interaction classes", () => {
  const file = JSON.parse(readFileSync(fileURLToPath(new URL("../../../config/differential/battle-interaction-classes.v1.json", import.meta.url)), "utf8")) as unknown;

  it("reads the committed file and hashes it", () => {
    const classes = readInteractionClasses(file);
    expect(classes.classes.map((c) => c.id)).toContain("speed_order_tie");
    expect(classesHash(classes)).toMatch(/^[0-9a-f]{64}$/);
  });

  it("tags a step with every class whose groups all match", () => {
    const classes = readInteractionClasses(file);
    expect(classifyStep(classes, step(["|move|p1a: X|Earthquake|p2a: Y|[spread] p2a,p2b", "|-activate|p2a: Y|move: Protect"], [{ method: "shuffle", args: ["2"], result: ["1", "0"] }]))).toEqual(["protection_multi_target", "speed_order_tie"]);
    expect(classifyStep(classes, step(["|upkeep"]))).toEqual([]);
    expect(classifyStep(classes, step(["|upkeep", "|-damage|p1a: X|10/100|[from] psn"]))).toEqual(["residual_order"]);
    expect(classifyStep(classes, step(["|switch|p1a: X|X, L50|100/100", "|-weather|SunnyDay|[from] ability: Drought|[of] p1a: X"]))).toEqual(["switch_in_trigger"]);
  });

  it("refuses a malformed file", () => {
    const good = { id: "a", description: "d", all: [[{ draw: "shuffle" }]] };
    const bad: unknown[] = [
      null,
      { schema_version: 2, data_class: "synthetic", classes: [] },
      { schema_version: 1, data_class: "synthetic", classes: [], extra: 1 },
      { schema_version: 1, data_class: "synthetic", classes: [{ ...good, id: "A" }] },
      { schema_version: 1, data_class: "synthetic", classes: [{ ...good, all: [] }] },
      { schema_version: 1, data_class: "synthetic", classes: [{ ...good, all: [[]] }] },
      { schema_version: 1, data_class: "synthetic", classes: [{ ...good, all: [[{ line: "" }]] }] },
      { schema_version: 1, data_class: "synthetic", classes: [{ ...good, all: [[{ line: "move", contains: 3 }]] }] },
      { schema_version: 1, data_class: "synthetic", classes: [{ ...good, all: [[{ draw: "shuffle", line: "x" }]] }] },
      { schema_version: 1, data_class: "synthetic", classes: [good, good] },
    ];
    for (const value of bad) expect(() => readInteractionClasses(value), JSON.stringify(value)).toThrow(ClassesError);
    expect(readInteractionClasses({ schema_version: 1, data_class: "synthetic", classes: [{ ...good, all: [[{ line: "move", from: "" }]] }] }).classes[0]?.all[0]).toEqual([{ line: "move", from: "" }]);
  });
});
