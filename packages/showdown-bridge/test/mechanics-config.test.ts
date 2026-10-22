import { mkdtempSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { jcs, type Json } from "../src/canonical-json.ts";
import type { Engine } from "../src/engine.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { loadEngine } from "../src/index.ts";
import {
  checkMechanicsConfig,
  loadMechanicsConfig,
  MECHANICS_CONFIG_DIR,
  type MechanicsConfig,
  MechanicsConfigError,
  mechanicsConfigHash,
  readMechanicsConfig,
} from "../src/mechanics-config.ts";
import { loadRulesSnapshot } from "../src/rules-load.ts";
import { sha256Hex } from "../src/rules.ts";

let engine: Engine;
let config: MechanicsConfig;
beforeAll(() => {
  engine = loadEngine();
  config = loadMechanicsConfig(GOLDEN_FORMAT);
});

/** The `stats` entry of the scope. */
const statsEntry = (c: MechanicsConfig): { gap_ids: string[] } => c.scope.query_kinds.find((entry) => entry.kind === "stats") as { gap_ids: string[] };

const edited = (change: (copy: MechanicsConfig) => void): MechanicsConfig => {
  const copy = structuredClone(config);
  change(copy);
  return copy;
};

describe("the committed config", () => {
  it("reads, and its hash is the SHA-256 of its JCS bytes", () => {
    expect(mechanicsConfigHash(config)).toBe(sha256Hex(jcs(config as unknown as Json)));
    expect(config.scope.query_kinds.map((entry) => entry.kind)).toEqual(["damage", "speed_order", "stats"]);
  });

  it("passes the check against the pinned engine and the gap ledger, for every config in config/mechanics", () => {
    const files = readdirSync(MECHANICS_CONFIG_DIR).filter((file) => file.endsWith(".json"));
    expect(files).toEqual([`${GOLDEN_FORMAT}.json`]);
    for (const file of files) expect(checkMechanicsConfig(engine, loadMechanicsConfig(file.replace(/\.json$/, "")))).toEqual([]);
  });

  it("lists twelve removed keys at this pin (ADR-016)", () => {
    expect(config.scenario.removed_rule_keys).toHaveLength(12);
  });
});

describe("readMechanicsConfig", () => {
  it("refuses a document that is not the contract", () => {
    expect(() => readMechanicsConfig({ ...config, extra: 1 })).toThrow(MechanicsConfigError);
    expect(() => readMechanicsConfig("x")).toThrow("invalid MechanicsConfig");
  });

  it.each([
    ["scenario.removed_rules", (c: MechanicsConfig) => c.scenario.removed_rules.reverse()],
    ["scenario.removed_rule_keys", (c: MechanicsConfig) => void c.scenario.removed_rule_keys.push("dctimerbank")],
    ["scope.query_kinds", (c: MechanicsConfig) => void c.scope.query_kinds.reverse()],
    ["ko_blockers.abilities", (c: MechanicsConfig) => void (c.ko_blockers.abilities = ["sturdy", "disguise"])],
    ["ko_blockers.items", (c: MechanicsConfig) => void (c.ko_blockers.items = ["focussash", "focussash"])],
    ["gap_ids of stats", (c: MechanicsConfig) => void (statsEntry(c).gap_ids = ["stat-points", "level"])],
  ] as [string, (c: MechanicsConfig) => void][])("refuses %s that is not sorted and unique", (name, change) => {
    expect(() => readMechanicsConfig(edited(change))).toThrow(`${name} must be sorted and unique`);
  });
});

describe("loadMechanicsConfig", () => {
  it("refuses a file that is the config of another format", () => {
    const dir = mkdtempSync(join(tmpdir(), "mechanics-config-"));
    writeFileSync(join(dir, "gen9other.json"), readFileSync(join(MECHANICS_CONFIG_DIR, `${GOLDEN_FORMAT}.json`)));
    expect(() => loadMechanicsConfig("gen9other", dir)).toThrow("config of another format");
  });
});

describe("checkMechanicsConfig names every problem", () => {
  it("a format the rules snapshot does not support", () => {
    expect(checkMechanicsConfig(engine, edited((c) => (c.format_id = "gen9ou")))).toEqual(["gen9ou is not a supported Areneon format"]);
  });

  it("a gap id that is not a ledger row, an item or ability that is not the engine's, and a scenario that differs", () => {
    const problems = checkMechanicsConfig(
      engine,
      edited((c) => {
        statsEntry(c).gap_ids = ["no-such-gap"];
        c.ko_blockers = { abilities: ["zzzability"], items: ["zzzitem"] };
        c.scenario.removed_rule_keys = c.scenario.removed_rule_keys.slice(1);
      }),
    );
    expect(problems).toHaveLength(4);
    expect(problems.join("\n")).toContain("stats: gap no-such-gap is not in the gap ledger");
    expect(problems.join("\n")).toContain("ko_blockers: zzzitem is not an item");
    expect(problems.join("\n")).toContain("ko_blockers: zzzability is not an ability");
    expect(problems.join("\n")).toContain("the target's keys missing from the scenario");
  });

  it("a speed_order scope needs the Mega row in the ledger, which the probe names when a Mega is involved", () => {
    const snapshot = structuredClone(loadRulesSnapshot(engine.commit));
    delete snapshot.gaps["mega-evolution"];
    expect(checkMechanicsConfig(engine, config, snapshot)).toContain("speed_order: gap mega-evolution is not in the gap ledger");
  });

  it("accepts engine ids as KO blockers", () => {
    expect(checkMechanicsConfig(engine, edited((c) => (c.ko_blockers = { abilities: ["disguise", "sturdy"], items: ["focussash"] })))).toEqual([]);
  });
});
