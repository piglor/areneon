import { chmodSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { loadEngine } from "../src/engine.ts";
import { GenerationError, type PoolRequest, trainingTeamPool, type TrainingPool } from "../src/league-generation.ts";
import { obtainPool, poolRequestKey, POOL_FILE_VERSION, readPoolFile, writePoolFile } from "../src/pool-share.ts";

const request: PoolRequest = { format_id: "gen9championsvgc2026regmc", generation: { excluded_abilities: [] } };
let dir: string;
let pool: TrainingPool;
beforeAll(() => {
  dir = mkdtempSync(join(tmpdir(), "areneon-pool-share-"));
  pool = trainingTeamPool(loadEngine(), request.generation.excluded_abilities);
});
afterAll(() => rmSync(dir, { recursive: true, force: true }));

const sha = (file: string): string => createHash("sha256").update(readFileSync(file)).digest("hex");
/** Read a file with the digest of the bytes it holds now, as a starter that vouched for them would send. */
const readAs = (file: string, asked: PoolRequest = request): TrainingPool => readPoolFile(file, asked, sha(file));

/** A pool file changed by `change`, as a hostile or stale writer could leave it (and then vouched for, so the content checks are reached). */
const tampered = (name: string, change: (value: Record<string, unknown>) => void): string => {
  const file = join(dir, `${name}.json`);
  writePoolFile(file, pool, request);
  const value = JSON.parse(readFileSync(file, "utf8")) as Record<string, unknown>;
  change(value);
  writeFileSync(file, JSON.stringify(value));
  chmodSync(file, 0o600);
  return file;
};

describe("the shared team pool (#1192)", () => {
  it("reads back the very pool that was written: the same teams in the same order with the same key order, and the synthetic set rebuilt from the hashes", () => {
    const file = join(dir, "pool.json");
    const digest = writePoolFile(file, pool, request);
    expect(digest).toBe(sha(file));
    expect(statSync(file).mode & 0o777).toBe(0o600);
    const back = readPoolFile(file, request, digest);
    expect(back.hashes).toEqual([...pool.hashes]);
    expect(JSON.stringify(back.teams)).toBe(JSON.stringify(pool.teams)); // key order too
    expect([...back.known_synthetic].sort()).toEqual([...pool.known_synthetic].sort());
  });

  it("takes only the file its starter vouched for: other bytes, another owner and another mode are refused", () => {
    const file = join(dir, "vouched.json");
    const digest = writePoolFile(file, pool, request);
    expect(() => readPoolFile(file, request, "0".repeat(64))).toThrow(/not the file its starter vouched for/);
    expect(() => readPoolFile(file, request, digest, (process.getuid?.() ?? 0) + 1)).toThrow(/not owned by this user with mode 0600/);
    chmodSync(file, 0o644);
    expect(() => readPoolFile(file, request, digest)).toThrow(/not owned by this user with mode 0600/);
    chmodSync(file, 0o600);
    writeFileSync(file, `${readFileSync(file, "utf8")} `); // a changed byte after the vouching
    expect(() => readPoolFile(file, request, digest)).toThrow(/not the file its starter vouched for/);
  });

  it("binds the file to the request it was built for: format, dataset and excluded abilities (the abilities as a set)", () => {
    const file = join(dir, "bound.json");
    writePoolFile(file, pool, request);
    const key = poolRequestKey(request);
    expect(poolRequestKey({ ...request, format_id: "gen9other" })).not.toBe(key);
    expect(poolRequestKey({ ...request, team_dataset: { path: "d", dataset_hash: "a".repeat(64) } })).not.toBe(key);
    expect(poolRequestKey({ ...request, generation: { excluded_abilities: ["Illusion"] } })).not.toBe(key);
    expect(poolRequestKey({ ...request, generation: { excluded_abilities: ["Illusion", "Trace"] } })).toBe(poolRequestKey({ ...request, generation: { excluded_abilities: ["Trace", "Illusion"] } }));
    const other = { ...request, generation: { excluded_abilities: ["Illusion"] } };
    expect(() => readAs(file, other)).toThrow(GenerationError);
    expect(() => readAs(file, other)).toThrow(/another format, dataset or excluded-ability list/);
  });

  it("refuses a file that is missing, damaged, of another version, unsorted, repeated, empty, ragged or holds a team that is not the one listed", () => {
    expect(() => readPoolFile(join(dir, "none.json"), request, "0".repeat(64))).toThrow(/cannot be read/);
    const junk = join(dir, "junk.json");
    writeFileSync(junk, "not json", { mode: 0o600 });
    expect(() => readAs(junk)).toThrow(/cannot be read/);
    expect(() => readAs(tampered("version", (v) => (v.schema_version = POOL_FILE_VERSION + 1)))).toThrow(/layout version/);
    expect(() => readAs(tampered("order", (v) => (v.hashes = [...(v.hashes as string[])].reverse())))).toThrow(/not sorted and unique/);
    expect(() => readAs(tampered("twice", (v) => (v.hashes = (v.hashes as string[]).map(() => (v.hashes as string[])[0]))))).toThrow(/not sorted and unique/);
    expect(() => readAs(tampered("number", (v) => (v.hashes = (v.hashes as unknown[]).map((_, i) => i))))).toThrow(/not sorted and unique/);
    expect(() => readAs(tampered("empty", (v) => ((v.hashes = []), (v.teams = []))))).toThrow(/empty or its hashes and teams differ/);
    expect(() => readAs(tampered("short", (v) => (v.teams = (v.teams as unknown[]).slice(1))))).toThrow(/differ in number/);
    expect(() => readAs(tampered("swapped", (v) => (v.teams = [...(v.teams as unknown[])].reverse())))).toThrow(/does not hash to its listed hash/);
    expect(() => readAs(tampered("shape", (v) => ((v.teams as unknown[])[0] = "x")))).toThrow(/does not hash to its listed hash/);
    expect(() =>
      readAs(
        tampered("edited", (v) => {
          const first = ((v.teams as { moves: string[] }[][])[0] as { moves: string[] }[])[0] as { moves: string[] };
          first.moves = ["tackle"];
        }),
      ),
    ).toThrow(/does not hash to its listed hash/);
  });

  it("obtains the pool by loading it (and vouching for the file it leaves), or by reading what its starter left once the digest comes", async () => {
    const file = join(dir, "obtained.json");
    let loads = 0;
    const load = (): TrainingPool => {
      loads++;
      return pool;
    };
    const vouched: string[] = [];
    const written = await obtainPool(request, load, { write: file, vouch: (digest) => vouched.push(digest) });
    expect(written.pool.hashes).toEqual([...pool.hashes]);
    expect(vouched).toEqual([sha(file)]);
    expect(written.digest).toBe(sha(file));
    expect((await obtainPool(request, load, {})).digest).toBeUndefined();
    expect(loads).toBe(2);
    let send: (digest: string) => void = () => undefined;
    const arriving = new Promise<string>((resolve) => (send = resolve));
    const reading = obtainPool(request, load, { read: { file, digest: arriving } });
    send(sha(file)); // the reader holds until its starter's line arrives
    expect((await reading).pool.hashes).toEqual([...pool.hashes]);
    expect(loads).toBe(2); // a reader does not load
    await expect(obtainPool(request, load, { read: { file, digest: Promise.resolve("0".repeat(64)) } })).rejects.toThrow(/not the file its starter vouched for/);
    expect(() => obtainPool(request, () => { throw new GenerationError("empty"); }, {})).toThrow("empty"); // a load that fails fails before anything starts
  });
});
