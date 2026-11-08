import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { loadEngine } from "../src/engine.ts";
import { GenerationError, type PoolRequest, trainingTeamPool, type TrainingPool } from "../src/league-generation.ts";
import { poolRequestKey, POOL_FILE_VERSION, readPoolFile, waitForPoolFile, writePoolFile } from "../src/pool-share.ts";

const request: PoolRequest = { format_id: "gen9championsvgc2026regmc", generation: { excluded_abilities: [] } };
let dir: string;
let pool: TrainingPool;
beforeAll(() => {
  dir = mkdtempSync(join(tmpdir(), "areneon-pool-share-"));
  pool = trainingTeamPool(loadEngine(), request.generation.excluded_abilities);
});
afterAll(() => rmSync(dir, { recursive: true, force: true }));

/** A pool file changed by `change`, as a hostile or stale writer could leave it. */
const tampered = (name: string, change: (value: Record<string, unknown>) => void): string => {
  const file = join(dir, `${name}.json`);
  writePoolFile(file, pool, request);
  const value = JSON.parse(readFileSync(file, "utf8")) as Record<string, unknown>;
  change(value);
  writeFileSync(file, JSON.stringify(value));
  return file;
};

describe("the shared team pool (#1192)", () => {
  it("reads back the very pool that was written: the same teams in the same order with the same key order, and the synthetic set rebuilt from the hashes", () => {
    const file = join(dir, "pool.json");
    writePoolFile(file, pool, request);
    const back = readPoolFile(file, request);
    expect(back.hashes).toEqual([...pool.hashes]);
    expect(JSON.stringify(back.teams)).toBe(JSON.stringify(pool.teams)); // key order too
    expect([...back.known_synthetic].sort()).toEqual([...pool.known_synthetic].sort());
  });

  it("binds the file to the request it was built for: format, dataset and excluded abilities", () => {
    const file = join(dir, "bound.json");
    writePoolFile(file, pool, request);
    const key = poolRequestKey(request);
    expect(poolRequestKey({ ...request, format_id: "gen9other" })).not.toBe(key);
    expect(poolRequestKey({ ...request, team_dataset: { path: "d", dataset_hash: "a".repeat(64) } })).not.toBe(key);
    expect(poolRequestKey({ ...request, generation: { excluded_abilities: ["Illusion"] } })).not.toBe(key);
    expect(() => readPoolFile(file, { ...request, generation: { excluded_abilities: ["Illusion"] } })).toThrow(GenerationError);
    expect(() => readPoolFile(file, { ...request, generation: { excluded_abilities: ["Illusion"] } })).toThrow(/another format, dataset or excluded-ability list/);
  });

  it("refuses a file that is missing, damaged, of another version, unsorted, repeated, empty, ragged or holds a team that is not the one listed", () => {
    expect(() => readPoolFile(join(dir, "none.json"), request)).toThrow(/cannot be read/);
    const junk = join(dir, "junk.json");
    writeFileSync(junk, "not json");
    expect(() => readPoolFile(junk, request)).toThrow(/cannot be read/);
    expect(() => readPoolFile(tampered("version", (v) => (v.schema_version = POOL_FILE_VERSION + 1)), request)).toThrow(/layout version/);
    expect(() => readPoolFile(tampered("order", (v) => (v.hashes = [...(v.hashes as string[])].reverse())), request)).toThrow(/not sorted and unique/);
    expect(() => readPoolFile(tampered("twice", (v) => (v.hashes = (v.hashes as string[]).map(() => (v.hashes as string[])[0]))), request)).toThrow(/not sorted and unique/);
    expect(() => readPoolFile(tampered("number", (v) => (v.hashes = (v.hashes as unknown[]).map((_, i) => i))), request)).toThrow(/not sorted and unique/);
    expect(() => readPoolFile(tampered("empty", (v) => ((v.hashes = []), (v.teams = []))), request)).toThrow(/empty or its hashes and teams differ/);
    expect(() => readPoolFile(tampered("short", (v) => (v.teams = (v.teams as unknown[]).slice(1))), request)).toThrow(/differ in number/);
    expect(() => readPoolFile(tampered("swapped", (v) => (v.teams = [...(v.teams as unknown[])].reverse())), request)).toThrow(/does not hash to its listed hash/);
    expect(() => readPoolFile(tampered("shape", (v) => ((v.teams as unknown[])[0] = "x")), request)).toThrow(/does not hash to its listed hash/);
    expect(() => readPoolFile(tampered("edited", (v) => {
      const first = ((v.teams as { moves: string[] }[][])[0] as { moves: string[] }[])[0] as { moves: string[] };
      first.moves = ["tackle"];
    }), request)).toThrow(/does not hash to its listed hash/);
  });

  it("waits for a file that is written later, and gives up when it never comes", async () => {
    const file = join(dir, "later.json");
    setTimeout(() => writePoolFile(file, pool, request), 60);
    const back = await waitForPoolFile(file, request, { poll_ms: 10, timeout_ms: 5000 });
    expect(back.hashes).toEqual([...pool.hashes]);
    await expect(waitForPoolFile(join(dir, "never.json"), request, { poll_ms: 5, timeout_ms: 30 })).rejects.toThrow(/did not appear in time/);
  });
});
