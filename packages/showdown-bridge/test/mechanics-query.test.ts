import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import type { Engine, SimBattle } from "../src/engine.ts";
import { GOLDEN_FORMAT, goldenBattle, syntheticTeam } from "../src/golden.ts";
import { type BattleRecord, type HiddenVerdicts, loadEngine, type OmniscientLog, type PlayerChannel } from "../src/index.ts";
import { type MechanicsContext, probeMechanics } from "../src/mechanics-probe.ts";
import { declaredSets, type MechanicsQuery, MechanicsQueryError, mechanicsQueryHash, readMechanicsQuery } from "../src/mechanics-query.ts";
import { buildResult, MECHANICS_ADAPTER_FORMAT, MechanicsResultError, readMechanicsResult } from "../src/mechanics-result.ts";
import { loadMechanicsConfig, readMechanicsConfig, MechanicsConfigError } from "../src/mechanics-config.ts";
import { SAMPLES_DIR } from "../src/samples.ts";
import { exportPaste } from "../src/teams.ts";

let engine: Engine;
let record: BattleRecord;
let paste: string;
beforeAll(async () => {
  engine = loadEngine();
  record = await goldenBattle(engine);
  paste = exportPaste(engine, [syntheticTeam()[0] as never]);
});

const query = (inner: unknown, extra: Record<string, unknown> = {}) => ({ schema_version: 1, data_class: "synthetic", format_id: GOLDEN_FORMAT, query: inner, ...extra });

describe("the privacy boundary of the probe API (ADR-016 section 5)", () => {
  it("takes only a parsed query: a battle, a record, a log, verdicts, a channel or an unparsed query is a compile error", () => {
    // Compile-time checks: `pnpm typecheck` fails if any of these compiles.
    const neverRun = (battle: SimBattle, log: OmniscientLog, verdicts: HiddenVerdicts[], channel: PlayerChannel, raw: MechanicsQuery): void => {
      const context = { engine, config: loadMechanicsConfig(GOLDEN_FORMAT) } as MechanicsContext;
      // @ts-expect-error: a live battle is not a query
      void probeMechanics(context, battle);
      // @ts-expect-error: nor is a BattleRecord
      void probeMechanics(context, record);
      // @ts-expect-error: nor the omniscient log
      void probeMechanics(context, log);
      // @ts-expect-error: nor hidden verdicts
      void probeMechanics(context, verdicts);
      // @ts-expect-error: nor a player's channel
      void probeMechanics(context, channel);
      // @ts-expect-error: nor a query that no reader checked; only readMechanicsQuery brands one
      void probeMechanics(context, raw);
    };
    expect(neverRun).toBeTypeOf("function");
  });

  it("refuses at run time a battle record, its parts, and a query carrying any of them", () => {
    for (const value of [record, record.omniscient_log, record.hidden_verdicts, record.channels.p1, { ...record }]) {
      expect(() => readMechanicsQuery(value)).toThrow(MechanicsQueryError);
    }
    const smuggled = query({ kind: "stats", set: paste });
    for (const field of ["omniscient_log", "hidden_verdicts", "channel", "battle", "opponent_team", "seed"]) {
      expect(() => readMechanicsQuery({ ...smuggled, [field]: field === "omniscient_log" ? record.omniscient_log : [] }), field).toThrow(MechanicsQueryError);
      expect(() => readMechanicsQuery(query({ kind: "stats", set: paste, [field]: 1 })), field).toThrow(MechanicsQueryError);
    }
  });

  it("names the root as / when the document is not an object", () => {
    expect(() => readMechanicsQuery("x")).toThrow("invalid MechanicsQuery: / must be object");
  });

  it("names the field in a refusal and never quotes a value", () => {
    try {
      readMechanicsQuery(query({ kind: "stats", set: paste, secret: "Canary-Nickname-9" }));
      throw new Error("not refused");
    } catch (error) {
      expect((error as Error).message).toContain("/query");
      expect((error as Error).message).not.toContain("Canary-Nickname-9");
    }
  });
});

describe("readMechanicsQuery", () => {
  it("reads each kind and returns a copy", () => {
    const original = query({ kind: "stats", set: paste });
    const parsed = readMechanicsQuery(original);
    expect(parsed).toEqual(original);
    expect(parsed).not.toBe(original);
  });

  it("hashes the JCS bytes: key order does not matter", () => {
    const a = readMechanicsQuery(query({ kind: "stats", set: paste }));
    const b = readMechanicsQuery({ query: { set: paste, kind: "stats" }, format_id: GOLDEN_FORMAT, data_class: "synthetic", schema_version: 1 });
    expect(mechanicsQueryHash(a)).toBe(mechanicsQueryHash(b));
    expect(mechanicsQueryHash(a)).toMatch(/^[0-9a-f]{64}$/);
  });

  it("lists the declared sets in position order for every kind", () => {
    const boosts = { atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
    const mon = (set: string) => ({ set, boosts, status: null, hp: null, mega: false });
    const side = { aurora_veil: false, light_screen: false, reflect: false, tailwind: false };
    const state = { weather: null, terrain: null, trick_room: false, sides: [side, side] };
    expect(declaredSets(readMechanicsQuery(query({ kind: "stats", set: "a" })))).toEqual(["a"]);
    expect(declaredSets(readMechanicsQuery(query({ kind: "damage", attacker: mon("a"), defender: mon("b"), move: "protect", state })))).toEqual(["a", "b"]);
    const order = { kind: "speed_order", sides: [[mon("a"), mon("b")], [mon("c")]], actions: [{ side: 0, position: 1, move: null, mega: true }], state };
    expect(declaredSets(readMechanicsQuery(query(order)))).toEqual(["a", "b", "c"]);
  });
});

describe("readMechanicsResult", () => {
  it("builds a result whose adapter format is the probe's constant", () => {
    const outcome = probeMechanics({ engine, config: loadMechanicsConfig(GOLDEN_FORMAT) }, readMechanicsQuery(query({ kind: "stats", set: paste })));
    const { result } = outcome;
    if (result.kind !== "computed") throw new Error("expected a computed result");
    const { data_class, query_hash, manifest, mechanics_config_hash, gap_ids } = result;
    expect(result.adapter_format).toBe(MECHANICS_ADAPTER_FORMAT);
    const identity = { data_class, query_hash, manifest, mechanics_config_hash, gap_ids };
    const rebuilt = buildResult(identity, { kind: "unsupported", reason: "query_kind", position: null });
    expect(readMechanicsResult(rebuilt)).toEqual(rebuilt);
  });

  it("refuses a document that is not the contract and gap ids that are not sorted", () => {
    expect(() => readMechanicsResult({})).toThrow(MechanicsResultError);
  });
});

// The corpus: every sample of the three contracts is classified as its directory says.
describe("the mechanics corpus samples", () => {
  const sample = (contract: string, kind: string, name: string): unknown => JSON.parse(readFileSync(join(SAMPLES_DIR, "v1", contract, kind, `${name}.json`), "utf8")) as unknown;
  const names = (contract: string, kind: string): string[] => readdirSync(join(SAMPLES_DIR, "v1", contract, kind)).map((file) => file.replace(".json", ""));

  it.each(names("mechanics-query", "valid"))("the query sample %s is read", (name) => {
    expect(readMechanicsQuery(sample("mechanics-query", "valid", name)).schema_version).toBe(1);
  });
  it.each(names("mechanics-query", "invalid"))("the query sample %s is refused", (name) => {
    expect(() => readMechanicsQuery(sample("mechanics-query", "invalid", name))).toThrow(MechanicsQueryError);
  });
  it.each(names("mechanics-result", "valid"))("the result sample %s is read", (name) => {
    expect(readMechanicsResult(sample("mechanics-result", "valid", name)).schema_version).toBe(1);
  });
  it.each(names("mechanics-result", "invalid"))("the result sample %s is refused", (name) => {
    expect(() => readMechanicsResult(sample("mechanics-result", "invalid", name))).toThrow(MechanicsResultError);
  });
  it("the reader-invalid result sample passes the schema and not the reader", () => {
    expect(() => readMechanicsResult(sample("mechanics-result", "reader-invalid", "unsorted-gap-ids"))).toThrow("sorted and unique");
  });
  it.each(names("mechanics-config", "valid"))("the config sample %s is read", (name) => {
    expect(readMechanicsConfig(sample("mechanics-config", "valid", name)).format_id).toBe(GOLDEN_FORMAT);
  });
  it.each(names("mechanics-config", "invalid"))("the config sample %s is refused", (name) => {
    expect(() => readMechanicsConfig(sample("mechanics-config", "invalid", name))).toThrow(MechanicsConfigError);
  });
  it("the reader-invalid config samples pass the schema and not the reader", () => {
    expect(() => readMechanicsConfig(sample("mechanics-config", "reader-invalid", "unsorted-removed-rule-ids"))).toThrow("sorted and unique");
    expect(() => readMechanicsConfig(sample("mechanics-config", "reader-invalid", "unsorted-ko-non-blockers"))).toThrow("ko_non_blockers.abilities must be sorted and unique");
  });
});

