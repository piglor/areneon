import { beforeAll, describe, expect, it } from "vitest";
import type { Engine, SimFormat, SimRuleTable } from "../src/engine.ts";
import { GOLDEN_FORMAT, syntheticTeam } from "../src/golden.ts";
import { loadEngine } from "../src/index.ts";
import { loadMechanicsConfig, type MechanicsConfig } from "../src/mechanics-config.ts";
import { checkScenario, ScenarioError, scenarioFormatId, scenarioPokemon, startScenario } from "../src/mechanics-scenario.ts";

let engine: Engine;
let config: MechanicsConfig;
beforeAll(() => {
  engine = loadEngine();
  config = loadMechanicsConfig(GOLDEN_FORMAT);
});

interface Tamper {
  format?: (format: SimFormat) => void;
  table?: (table: SimRuleTable) => void;
}

/** An engine whose scenario format (the id with custom rules) answers as `tamper` says. */
function tampered(real: Engine, tamper: Tamper): Engine {
  const formats = real.sim.Dex.formats;
  const marked = new WeakMap<object, SimFormat>();
  const get = (id: string): SimFormat => {
    const format = formats.get(id);
    if (!id.includes("@@@")) return format;
    const copy = Object.create(format) as SimFormat;
    tamper.format?.(copy);
    marked.set(copy, format);
    return copy;
  };
  const getRuleTable = (format: SimFormat): SimRuleTable => {
    const real = marked.get(format);
    if (real === undefined) return formats.getRuleTable(format);
    const table = formats.getRuleTable(real);
    const copy = Object.assign(new Map(table) as SimRuleTable, table, { valueRules: new Map(table.valueRules) });
    tamper.table?.(copy);
    return copy;
  };
  return { ...real, sim: { ...real.sim, Dex: { ...real.sim.Dex, formats: { get, getRuleTable } } } };
}

const refusal = (run: () => unknown): string[] => {
  try {
    run();
  } catch (error) {
    if (error instanceof ScenarioError) return error.problems;
    throw error;
  }
  throw new Error("the scenario check did not refuse");
};

describe("the scenario format", () => {
  it("is the target minus the listed rules, in the engine's custom-rule syntax", () => {
    expect(scenarioFormatId(config)).toBe(`${GOLDEN_FORMAT}@@@!Min Team Size,!Open Team Sheets,!Picked Team Size,!Team Preview,!VGC Timer`);
  });

  it("starts the check clean on the pinned engine: exactly the listed keys are removed", () => {
    expect(checkScenario(engine, config)).toBe(scenarioFormatId(config));
    // The unmodified wrapper changes nothing.
    expect(checkScenario(tampered(engine, {}), config)).toBe(scenarioFormatId(config));
  });
});

describe("the start-up rule-table check refuses when anything differs", () => {
  it("a listed key the engine does not remove", () => {
    const wrong = { ...config, scenario: { ...config.scenario, removed_rule_keys: [...config.scenario.removed_rule_keys, "zzz"] } };
    expect(refusal(() => checkScenario(engine, wrong))[0]).toContain("not the listed");
  });

  it("a key the engine removes that the config does not list", () => {
    const wrong = { ...config, scenario: { ...config.scenario, removed_rule_keys: config.scenario.removed_rule_keys.slice(1) } };
    expect(refusal(() => checkScenario(engine, wrong))[0]).toContain("not the listed");
  });

  it("a removed rule that takes more than the listed keys with it", () => {
    const wrong = { ...config, scenario: { ...config.scenario, removed_rules: [...config.scenario.removed_rules, "Adjust Level"] } };
    const problems = refusal(() => checkScenario(engine, wrong));
    expect(problems.join(" ")).toContain("adjustlevel");
    expect(problems.join(" ")).toContain("adjustLevel differs");
  });

  it("a scenario format that does not exist", () => {
    expect(refusal(() => checkScenario(tampered(engine, { format: (f) => void Object.assign(f, { exists: false }) }), config))).toEqual(["the scenario format does not exist"]);
  });

  it("a custom rule the engine itself refuses", () => {
    const refuses = tampered(engine, { format: () => { throw new Error('Rule "!x" did nothing because "x" is not in effect'); } });
    expect(refusal(() => checkScenario(refuses, config))).toEqual(['the engine refuses the scenario format: Rule "!x" did nothing because "x" is not in effect']);
  });

  it.each([
    ["another mod", { format: (f: SimFormat) => void Object.assign(f, { mod: "gen9" }) }, "mod gen9 is not champions"],
    ["another game type", { format: (f: SimFormat) => void Object.assign(f, { gameType: "singles" }) }, "game type singles is not doubles"],
    ["battle options with a function", { format: (f: SimFormat) => void Object.assign(f, { battle: { trunc: (x: number) => x } }) }, "the battle options differ"],
    ["battle options with a value", { format: (f: SimFormat) => void Object.assign(f, { battle: { b: 1, a: 2 } }) }, "the battle options differ"],
    ["a key of its own", { table: (t: SimRuleTable) => void t.set("extra", "") }, "keys of its own: [extra]"],
    ["a rule from another source", { table: (t: SimRuleTable) => void t.set("itemclause", "elsewhere") }, "rule itemclause comes from another source"],
    ["a value rule that differs", { table: (t: SimRuleTable) => void t.valueRules.set("itemclause", "9") }, "value rule itemclause differs"],
    ["a level rule that differs", { table: (t: SimRuleTable) => void Object.assign(t, { minLevel: 7 }) }, "minLevel differs"],
    ["an EV limit that differs", { table: (t: SimRuleTable) => void Object.assign(t, { evLimit: 510 }) }, "evLimit differs"],
  ] as [string, Tamper, string][])("%s", (_name, tamper, message) => {
    expect(refusal(() => checkScenario(tampered(engine, tamper), config)).join(" | ")).toContain(message);
  });
});

describe("startScenario", () => {
  it("starts a battle at once, with the declared teams on both sides and no Team Preview", () => {
    const team = syntheticTeam().slice(0, 2);
    const battle = startScenario(engine, scenarioFormatId(config), { p1: team, p2: team });
    expect(battle.sides[0]?.requestState).toBe("move");
    expect(scenarioPokemon(battle, 0, 0).maxhp).toBeGreaterThan(0);
    expect(scenarioPokemon(battle, 1, 1).storedStats.spe).toBeGreaterThan(0);
  });
});
