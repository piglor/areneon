import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import type { Engine, SimFormat, SimRuleTable } from "../src/engine.ts";
import { GOLDEN_FORMAT, TEAMS_DIR } from "../src/golden.ts";
import { loadEngine, UnsupportedFormatError } from "../src/index.ts";
import { loadMechanicsConfig, type MechanicsConfig } from "../src/mechanics-config.ts";
import { MechanicsProbeError, probeMechanics } from "../src/mechanics-probe.ts";
import { readMechanicsQuery } from "../src/mechanics-query.ts";
import { exportPaste, importPaste } from "../src/teams.ts";

let engine: Engine;
let config: MechanicsConfig;
let own: string;
let foe: string;
let invalid: string;
beforeAll(() => {
  engine = loadEngine();
  config = loadMechanicsConfig(GOLDEN_FORMAT);
  const set = (file: string, index: number): string => exportPaste(engine, [importPaste(engine, readFileSync(join(TEAMS_DIR, file), "utf8"))[index] as never]);
  own = set("legal-synthetic.txt", 0);
  foe = set("legal-opponent.txt", 0);
  invalid = set("illegal-stat-points-over-32.txt", 0);
});

const wrap = (query: unknown, extra: Record<string, unknown> = {}) =>
  readMechanicsQuery({ schema_version: 1, data_class: "synthetic", format_id: GOLDEN_FORMAT, query, ...extra });
const boosts = { atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
const mon = (set: string) => ({ set, boosts, status: null, hp: null, mega: false });
const side = { aurora_veil: false, light_screen: false, reflect: false, tailwind: false };
const field = { weather: null, terrain: null, trick_room: false, sides: [side, side] };
const speedOrder = (a: string, b: string) =>
  wrap({ kind: "speed_order", sides: [[mon(a)], [mon(b)]], actions: [{ side: 0, position: 0, move: "protect", mega: false }], state: field });
const damage = (a: string, d: string) => wrap({ kind: "damage", attacker: mon(a), defender: mon(d), move: "protect", state: field });

describe("a refused set (ADR-016 figure 4: SetCheck)", () => {
  it("is invalid_set with the set's position, and the validator's words go to the diagnostics, never the result", () => {
    const outcome = probeMechanics({ engine, config }, wrap({ kind: "stats", set: invalid }));
    expect(outcome.result).toMatchObject({ kind: "refused", code: "invalid_set", position: 0 });
    expect(outcome.diagnostics.join(" ")).toContain("Stat Points");
    expect(JSON.stringify(outcome.result)).not.toContain("Stat Points");
  });

  it("counts positions as the contract says: a speed_order's Pokemon in order, a damage query's attacker then defender", () => {
    expect(probeMechanics({ engine, config }, speedOrder(own, invalid)).result).toMatchObject({ kind: "refused", position: 1 });
    expect(probeMechanics({ engine, config }, damage(own, invalid)).result).toMatchObject({ kind: "refused", position: 1 });
    expect(probeMechanics({ engine, config }, damage(invalid, foe)).result).toMatchObject({ kind: "refused", position: 0 });
  });

  it("refuses a paste that is not the engine's own form, and a paste of more than one Pokemon", () => {
    const noisy = probeMechanics({ engine, config }, wrap({ kind: "stats", set: own.replace("Level: 50", "Level: 50\nSPs: 1 HP") }));
    expect(noisy.result).toMatchObject({ kind: "refused", position: 0 });
    expect(noisy.diagnostics[0]).toContain("not reproduced");
    const team = readFileSync(join(TEAMS_DIR, "legal-synthetic.txt"), "utf8");
    const two = probeMechanics({ engine, config }, wrap({ kind: "stats", set: team.split(/\n\s*\n/).slice(0, 2).join("\n\n") }));
    expect(two.result).toMatchObject({ kind: "refused", position: 0 });
    expect(two.diagnostics).toEqual(["a set is one Pokémon, the paste has 2"]);
  });

  it("refuses before it looks at the scope", () => {
    expect(probeMechanics({ engine, config }, speedOrder(invalid, foe)).result.kind).toBe("refused");
  });
});

describe("unsupported results (ADR-016 figure 4: ScopeCheck)", () => {
  it.each(["speed_order", "damage"] as const)("a %s query is unsupported until the config lists it: no approximation", (kind) => {
    const { result, diagnostics } = probeMechanics({ engine, config }, kind === "damage" ? damage(own, foe) : speedOrder(own, foe));
    expect(result).toMatchObject({ kind: "unsupported", reason: "query_kind", position: null, gap_ids: [] });
    expect(diagnostics).toEqual([]);
  });

  it("listing a kind in the config never makes the probe handle it", () => {
    const greedy: MechanicsConfig = { ...config, scope: { query_kinds: [...config.scope.query_kinds, { kind: "damage", gap_ids: [] }] } };
    expect(() => probeMechanics({ engine, config: greedy }, damage(own, foe))).toThrow(MechanicsProbeError);
  });
});

/** An engine whose scenario format has a rule the target lacks, so the tables differ. */
function withOwnRule(real: Engine): Engine {
  const formats = real.sim.Dex.formats;
  const tampered = new WeakSet<object>();
  const get = (id: string): SimFormat => {
    const format = formats.get(id);
    if (!id.includes("@@@")) return format;
    const copy = Object.create(format) as SimFormat;
    tampered.add(copy);
    return copy;
  };
  const getRuleTable = (format: SimFormat): SimRuleTable => {
    const table = formats.getRuleTable(format);
    if (!tampered.has(format)) return table;
    const copy = new Map(table) as SimRuleTable;
    copy.set("extra", "");
    return Object.assign(copy, { valueRules: table.valueRules, adjustLevel: table.adjustLevel, evLimit: table.evLimit });
  };
  return { ...real, sim: { ...real.sim, Dex: { ...real.sim.Dex, formats: { get, getRuleTable } } } };
}

describe("the scenario check at start (ADR-016 figure 4: ScenarioCheck)", () => {
  it("refuses to run when the rule tables differ from the config's: unsupported, with the differences in the diagnostics", () => {
    const { result, diagnostics } = probeMechanics({ engine: withOwnRule(engine), config }, wrap({ kind: "stats", set: own }));
    expect(result).toMatchObject({ kind: "unsupported", reason: "scenario_rule_table", position: null });
    expect(diagnostics.join(" ")).toContain("keys of its own");
  });
});

describe("what the probe refuses to answer at all", () => {
  it("a config of another format", () => {
    expect(() => probeMechanics({ engine, config: { ...config, format_id: "gen9other" } }, wrap({ kind: "stats", set: own }))).toThrow("another format");
  });

  it("a format the rules snapshot does not support: there is no manifest to name", () => {
    const query = readMechanicsQuery({ schema_version: 1, data_class: "synthetic", format_id: "gen9ou", query: { kind: "stats", set: own } });
    expect(() => probeMechanics({ engine, config: { ...config, format_id: "gen9ou" } }, query)).toThrow(UnsupportedFormatError);
  });
});
