import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  buildLeagueRunRecord,
  checkLeagueRunRecord,
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

const config = () => readTrainingRunConfig(sample("training-run-config", "tiny"));

describe("the LeagueRunRecord reader (ADR-009 §3)", () => {
  it("reads the valid samples and hands back a copy", () => {
    const raw = sample("league-run-record", "tiny");
    const record = readLeagueRunRecord(raw);
    expect(record).toEqual(raw);
    expect(record).not.toBe(raw);
    expect(readLeagueRunRecord(sample("league-run-record", "self-play-only")).forfeits).toBe(0);
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

  it("counts what a generation played and names the config, derivation and plan", async () => {
    const plan = [{ game: 0 }, { game: 1 }, { game: 2 }];
    const record = await buildLeagueRunRecord({
      config: config(),
      generation: 2,
      pair_policy_id: pair,
      plan,
      games: [
        { opponent: "random-legal@tiny", outcome: "played" },
        { opponent: pair, outcome: "forfeit" },
        { opponent: pair, outcome: "played" },
      ],
    });
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

  it("refuses a generation that played nothing", async () => {
    await expect(buildLeagueRunRecord({ config: config(), generation: 1, pair_policy_id: pair, plan: [], games: [] })).rejects.toThrow(LeagueRunRecordError);
  });
});
