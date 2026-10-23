import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  buildLeagueRunRecord,
  checkLeagueRunRecord,
  FORFEIT_CAUSES,
  LEAGUE_SEED_DERIVATION,
  LeagueRunRecordError,
  leagueRunRecordHash,
  readLeagueRunRecord,
  readTrainingRunConfig,
  runConfigHash,
} from "../src/index.ts";
import { SAMPLES_DIR } from "../src/samples.ts";

const sample = (contract: string, name: string): Record<string, unknown> =>
  JSON.parse(readFileSync(join(SAMPLES_DIR, `v1/${contract}/valid/${name}.json`), "utf8")) as Record<string, unknown>;

const readerInvalid = (name: string): Record<string, unknown> =>
  JSON.parse(readFileSync(join(SAMPLES_DIR, `v1/league-run-record/reader-invalid/${name}.json`), "utf8")) as Record<string, unknown>;

const config = () => readTrainingRunConfig(sample("training-run-config", "tiny"));

describe("the LeagueRunRecord reader (ADR-009 §3)", () => {
  it("reads the valid samples and hands back a copy", () => {
    const raw = sample("league-run-record", "tiny");
    const record = readLeagueRunRecord(raw);
    expect(record).toEqual(raw);
    expect(record).not.toBe(raw);
    expect(readLeagueRunRecord(sample("league-run-record", "self-play-only")).forfeits).toBe(0);
  });

  it("reads the cause totals and refuses the ones the profile cannot check", () => {
    expect(readLeagueRunRecord(sample("league-run-record", "mixed-causes")).forfeit_causes).toEqual([{ cause: "crash", count: 1 }, { cause: "latency_overrun", count: 2 }]);
    expect(() => readLeagueRunRecord(readerInvalid("causes-do-not-sum"))).toThrow(/sum to forfeits/);
    expect(() => readLeagueRunRecord(readerInvalid("unsorted-causes"))).toThrow(/sorted by cause/);
    expect(() => readLeagueRunRecord(readerInvalid("duplicate-cause"))).toThrow(/sorted by cause/);
  });

  it("reads the damage-aware baseline entry and refuses the ones the profile cannot check", () => {
    const record = readLeagueRunRecord(sample("league-run-record", "damage-aware"));
    expect(record.baselines).toHaveLength(1);
    expect(record.baselines?.[0]).toMatchObject({ policy_id: "damage-aware@v1", games: 3 });
    expect(readLeagueRunRecord(sample("league-run-record", "tiny")).baselines).toBeUndefined();
    for (const [name, message] of [
      ["baselines-unsorted", /baselines must be sorted/],
      ["baseline-duplicate", /baselines must be sorted/],
      ["baseline-games-disagree", /games must equal/],
      ["baseline-not-an-opponent", /games must equal/],
      ["baseline-probe-runs-exceed-queries", /cannot exceed its queries/],
      ["baseline-unsupported-exceed-queries", /cannot exceed its queries/],
      ["baseline-decisions-exceed", /cannot exceed its decisions/],
    ] as const) {
      expect(() => readLeagueRunRecord(readerInvalid(name)), name).toThrow(message);
    }
  });

  it("keeps the schema's cause enum equal to the harness's ForfeitCause list", () => {
    const schema = JSON.parse(readFileSync(join(SAMPLES_DIR, "../../packages/schema/contracts/v1/league-run-record.schema.json"), "utf8")) as {
      $defs: { ForfeitCauseCount: { properties: { cause: { enum: string[] } } } };
    };
    expect([...schema.$defs.ForfeitCauseCount.properties.cause.enum].sort()).toEqual([...FORFEIT_CAUSES].sort());
  });

  it("refuses a record the schema refuses and an unknown derivation", () => {
    expect(() => readLeagueRunRecord({ ...sample("league-run-record", "tiny"), notes: "x" })).toThrow(LeagueRunRecordError);
    expect(() => readLeagueRunRecord(null)).toThrow(/invalid LeagueRunRecord/);
  });

  it("hashes the JCS bytes: key order does not matter, a changed count does", async () => {
    const record = readLeagueRunRecord(sample("league-run-record", "tiny"));
    const reordered = readLeagueRunRecord(Object.fromEntries(Object.entries(record).reverse()));
    expect(await leagueRunRecordHash(reordered)).toBe(await leagueRunRecordHash(record));
    expect(await leagueRunRecordHash(record)).toMatch(/^[0-9a-f]{64}$/);
    expect(await leagueRunRecordHash({ ...record, generation: 2 })).not.toBe(await leagueRunRecordHash(record));
  });

  it("belongs to a config by its run key and its hash", async () => {
    const record = readLeagueRunRecord(sample("league-run-record", "tiny"));
    await expect(checkLeagueRunRecord(record, config())).resolves.toBeUndefined();
    await expect(checkLeagueRunRecord({ ...record, run_id: "other" }, config())).rejects.toThrow(/run key/);
    await expect(checkLeagueRunRecord({ ...record, run_config_hash: "0".repeat(64) }, config())).rejects.toThrow(/config's hash/);
  });
});

describe("buildLeagueRunRecord", () => {
  const pair = `policy:${"a".repeat(64)}`;
  const counts = (queries: number) => ({ decisions: 10, preview_decisions: 1, scored_decisions: 8, queries, probe_runs: queries - 1, unsupported_queries: 2, reader_fallbacks: 0, assumed_set_fallbacks: 1, epsilon_plays: 0 });

  it("sums a damage-aware baseline's per-game counts under its identity hash, and only then writes baselines", async () => {
    const games = [
      { opponent: "damage-aware@v1", outcome: "played" as const },
      { opponent: "damage-aware@v1", outcome: "played" as const },
      { opponent: pair, outcome: "played" as const },
    ];
    const reports = [
      { policy_id: "damage-aware@v1", identity_hash: "c".repeat(64), counts: counts(5) },
      { policy_id: "damage-aware@v1", identity_hash: "c".repeat(64), counts: counts(7) },
    ];
    const facts = { config: config(), generation: 1, pair_policy_id: pair, plan: [], games };
    const record = await buildLeagueRunRecord({ ...facts, baseline_reports: reports });
    expect(record.baselines).toEqual([{ policy_id: "damage-aware@v1", identity_hash: "c".repeat(64), games: 2, counts: { ...counts(12), decisions: 20, preview_decisions: 2, scored_decisions: 16, probe_runs: 10, unsupported_queries: 4, assumed_set_fallbacks: 2 } }]);
    expect("baselines" in (await buildLeagueRunRecord(facts))).toBe(false);
    expect("baselines" in (await buildLeagueRunRecord({ ...facts, baseline_reports: [] }))).toBe(false);
    await expect(
      buildLeagueRunRecord({ ...facts, baseline_reports: [reports[0] as (typeof reports)[0], { ...(reports[1] as (typeof reports)[0]), identity_hash: "d".repeat(64) }] }),
    ).rejects.toThrow(/two identity hashes/);
  });

  it("counts what a generation played and names the config, derivation and plan", async () => {
    const plan = [{ game: 0 }, { game: 1 }, { game: 2 }];
    const record = await buildLeagueRunRecord({
      config: config(),
      generation: 2,
      pair_policy_id: pair,
      plan,
      games: [
        { opponent: "random-legal@tiny", outcome: "played" },
        { opponent: pair, outcome: "forfeit", forfeit: { cause: "crash" } },
        { opponent: pair, outcome: "played" },
      ],
    });
    expect(record.forfeit_causes).toEqual([{ cause: "crash", count: 1 }]);
    expect(record).toMatchObject({
      run_id: "tiny-run",
      generation: 2,
      seed_derivation: LEAGUE_SEED_DERIVATION,
      games: 3,
      played: 2,
      forfeits: 1,
      episodes: 4,
      opponents: [
        { opponent_policy_id: pair, games: 2 },
        { opponent_policy_id: "random-legal@tiny", games: 1 },
      ],
    });
    expect(record.run_config_hash).toBe(await runConfigHash(config()));
    expect(record.plan_hash).toMatch(/^[0-9a-f]{64}$/);
    await checkLeagueRunRecord(record, config());
  });

  it("totals the forfeits by cause, sorted, and writes none for a forfeit-free generation", async () => {
    const forfeit = (cause: "crash" | "latency_overrun") => ({ opponent: pair, outcome: "forfeit" as const, forfeit: { cause } });
    const base = { config: config(), generation: 1, pair_policy_id: pair, plan: [] };
    const record = await buildLeagueRunRecord({
      ...base,
      games: [forfeit("latency_overrun"), forfeit("crash"), forfeit("latency_overrun"), { opponent: pair, outcome: "played" }],
    });
    expect(record.forfeit_causes).toEqual([{ cause: "crash", count: 1 }, { cause: "latency_overrun", count: 2 }]);
    const clean = await buildLeagueRunRecord({ ...base, games: [{ opponent: pair, outcome: "played" }] });
    expect(clean).not.toHaveProperty("forfeit_causes");
  });

  it("refuses a forfeit that names no cause: the totals would not sum", async () => {
    await expect(
      buildLeagueRunRecord({ config: config(), generation: 1, pair_policy_id: pair, plan: [], games: [{ opponent: pair, outcome: "forfeit" }, { opponent: pair, outcome: "played" }] }),
    ).rejects.toThrow(/invalid LeagueRunRecord/);
  });

  it("refuses a generation that played nothing", async () => {
    await expect(buildLeagueRunRecord({ config: config(), generation: 1, pair_policy_id: pair, plan: [], games: [] })).rejects.toThrow(LeagueRunRecordError);
  });
});
