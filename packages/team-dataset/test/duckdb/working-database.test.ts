import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { importPaste, recordHash, type TeamRecord } from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it } from "vitest";
import { buildWorkingDatabase, DUCKDB_SCHEMA_VERSION, runTeamsCli, SQL_SCRIPT, TABLE_FILES } from "../../src/duckdb/index.ts";
import { ShardError } from "../../src/shards/columns.ts";
import { convertDataset } from "../../src/shards/convert.ts";
import { DuckdbWriter, locateDuckdb, readPin, type Runner } from "../../src/shards/duckdb.ts";
import { engine } from "../support.ts";
import { DATASET_DIR, REPO } from "../shards/paths.ts";
import { fixtureRecords, labelSet, manifest, SHARD_KEY, standInRunner } from "../shards/support.ts";

const scratch = mkdtempSync(join(tmpdir(), "working-db-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));
let n = 0;
const dir = (): string => join(scratch, `d${n++}`);
const SPLITS = ["dev", "validation"] as const;
const README = readFileSync(new URL("../../README.md", import.meta.url), "utf8");

/** The fixtures' teams with the sets the pinned engine reads from their pastes, the oracle for every expectation. */
const teams = SPLITS.flatMap((split) => fixtureRecords(split)).map((record: TeamRecord) => ({ record, sets: importPaste(engine, record.paste) }));
const count = (keep: (t: (typeof teams)[number]) => boolean): number => teams.filter(keep).length;

const fakeBinary = join(scratch, "duckdb");
writeFileSync(fakeBinary, "stand-in\n");

/** The stand-in tool plus a `script` hook that sees the files the SQL script would load. */
function scriptRunner(onScript: (cwd: string, sql: string, database: string) => void): Runner {
  const base = standInRunner();
  return ((file, args, input, env, cwd, timeout) => {
    if (args[0] === "--version" || !args.includes("-batch") || args.length < 3) return base(file, args, input, env, cwd, timeout);
    onScript(cwd as string, input, args[2] as string);
    writeFileSync(args[2] as string, "database\n");
    return { status: 0, stdout: "", stderr: "" };
  }) as Runner;
}

const snapshotWith = async (writer: DuckdbWriter, out: string, shardRows = 50) =>
  convertDataset({ datasetDir: DATASET_DIR, outDir: out, labelSet, shardKey: SHARD_KEY, snapshotId: "working-db-test", splits: SPLITS, writer, shardRows });

describe("buildWorkingDatabase (stand-in tool: the files the SQL script loads)", () => {
  it("verifies the shards, expands every paste and hands the six tables to the script", async () => {
    const snapshotDir = dir();
    const { snapshot } = await snapshotWith(DuckdbWriter.open(fakeBinary, standInRunner()), snapshotDir);
    const seen: Record<string, Record<string, unknown>[]> = {};
    let script = "";
    const writer = DuckdbWriter.open(fakeBinary, scriptRunner((cwd, sql) => {
      script = sql;
      for (const name of TABLE_FILES) seen[name] = readFileSync(join(cwd, `${name}.jsonl`), "utf8").split("\n").filter((l) => l !== "").map((l) => JSON.parse(l) as Record<string, unknown>);
    }));
    const out = join(dir(), "teams.duckdb");
    const result = await buildWorkingDatabase({ snapshotFile: join(snapshotDir, "snapshot.json"), shardsDir: snapshotDir, labelSet, datasetManifest: manifest, engine, writer, outFile: out, shardRows: 50 });
    expect(existsSync(out)).toBe(true);
    expect(script).toBe(readFileSync(SQL_SCRIPT, "utf8"));
    expect(result).toMatchObject({ teams: 240, sets: 240 * 6 });
    expect(seen.teams).toHaveLength(240);
    expect(seen.shards).toHaveLength(snapshot.shards.length);
    expect(seen.labels).toHaveLength(240 * labelSet.labels.length);
    expect(seen.sets).toHaveLength(result.sets);
    expect(seen.moves).toHaveLength(result.moves);
    expect(seen.snapshot).toHaveLength(1);
    expect(seen.snapshot?.[0]).toMatchObject({ snapshot_id: "working-db-test", teams: 240, format_id: manifest.generator.manifest.format_id, parent_snapshot_hash: null });
    expect(new Set(seen.teams?.map((t) => t.record_hash))).toEqual(new Set(teams.map((t) => recordHash(t.record))));
    const first = teams[0] as (typeof teams)[number];
    const sets = (seen.sets ?? []).filter((s) => s.record_hash === recordHash(first.record));
    expect(sets.map((s) => s.species)).toEqual(first.sets.map((s) => s.species));
    expect(sets.map((s) => s.sp_spe)).toEqual(first.sets.map((s) => s.evs.spe));
    const moves = (seen.moves ?? []).filter((m) => m.record_hash === recordHash(first.record) && m.slot === 0);
    expect(moves.map((m) => m.move)).toEqual((first.sets[0] as { moves: string[] }).moves);
  });

  it("refuses a damaged shard before the script runs, and leaves no database", async () => {
    const snapshotDir = dir();
    const { snapshot } = await snapshotWith(DuckdbWriter.open(fakeBinary, standInRunner()), snapshotDir);
    const file = join(snapshotDir, (snapshot.shards[0] as { object_key: string }).object_key);
    const bytes = readFileSync(file);
    bytes[bytes.length >> 1] = (bytes[bytes.length >> 1] as number) ^ 0xff;
    writeFileSync(file, bytes);
    let ran = false;
    const writer = DuckdbWriter.open(fakeBinary, scriptRunner(() => (ran = true)));
    const out = join(dir(), "teams.duckdb");
    await expect(buildWorkingDatabase({ snapshotFile: join(snapshotDir, "snapshot.json"), shardsDir: snapshotDir, labelSet, datasetManifest: manifest, engine, writer, outFile: out, shardRows: 50 })).rejects.toBeInstanceOf(ShardError);
    expect(ran).toBe(false);
    expect(existsSync(out)).toBe(false);
    expect(readdirSync(join(out, "..")).filter((f) => f.includes("partial"))).toEqual([]);
  });

  it("refuses a snapshot whose records carry another label set, a missing dataset manifest for merged strata, and an existing output", async () => {
    const snapshotDir = dir();
    await snapshotWith(DuckdbWriter.open(fakeBinary, standInRunner()), snapshotDir);
    const writer = DuckdbWriter.open(fakeBinary, scriptRunner(() => undefined));
    const base = { snapshotFile: join(snapshotDir, "snapshot.json"), shardsDir: snapshotDir, engine, writer, shardRows: 50 };
    const other = { ...labelSet, label_set_id: `${labelSet.label_set_id}-other` };
    await expect(buildWorkingDatabase({ ...base, labelSet: other, datasetManifest: manifest, outFile: join(dir(), "a.duckdb") })).rejects.toBeInstanceOf(ShardError);
    await expect(buildWorkingDatabase({ ...base, labelSet, outFile: join(dir(), "b.duckdb") })).rejects.toBeInstanceOf(ShardError);
    const takenDir = dir();
    mkdirSync(takenDir, { recursive: true });
    const taken = join(takenDir, "c.duckdb");
    writeFileSync(taken, "x");
    await expect(buildWorkingDatabase({ ...base, labelSet, datasetManifest: manifest, outFile: taken })).rejects.toThrow(/never over one/);
    expect(readFileSync(taken, "utf8")).toBe("x");
  });
});

describe("DuckdbWriter.runScript", () => {
  it("forwards a failing script's status and text, and passes the database and working directory", () => {
    const calls: { args: readonly string[]; cwd: string | undefined }[] = [];
    const runner = ((_file, args, _input, _env, cwd) => {
      calls.push({ args, cwd });
      return args[0] === "--version" ? { status: 0, stdout: "v1.5.6 (x) 1\n", stderr: "" } : { status: 1, stdout: "", stderr: " syntax error " };
    }) as Runner;
    const writer = DuckdbWriter.open(fakeBinary, runner);
    expect(() => writer.runScript("SELECT 1;", "/tmp/x.duckdb", "/tmp")).toThrow(/DuckDB failed \(1\): syntax error/);
    expect(calls.at(-1)).toEqual({ args: ["-bail", "-batch", "/tmp/x.duckdb"], cwd: "/tmp" });
  });
});

describe("the teams command line", () => {
  const io = () => {
    const lines: { out: string[]; err: string[] } = { out: [], err: [] };
    return { lines, io: { out: (l: string) => lines.out.push(l), err: (l: string) => lines.err.push(l), env: { PATH: "" } } };
  };
  it("is a usage error without a DuckDB tool, without --out, with a wrong extension and for another command", async () => {
    const a = io();
    expect(await runTeamsCli(["duckdb", "--snapshot", "x", "--out", "t.duckdb"], a.io)).toBe(2);
    expect(a.lines.err[0]).toMatch(/no DuckDB command-line tool/);
    const withTool = { ...io().io, env: { DUCKDB_BIN: fakeBinary } };
    expect(await runTeamsCli(["duckdb", "--snapshot", "x"], withTool)).toBe(2);
    expect(await runTeamsCli(["duckdb", "--snapshot", "x", "--out", "t.db"], withTool)).toBe(2);
    expect(await runTeamsCli(["nope"], withTool)).toBe(2);
    expect(await runTeamsCli(["duckdb", "--snapshot", join(scratch, "missing"), "--out", "t.duckdb"], withTool)).toBe(2);
  });
  it("builds from a snapshot directory and prints the counts; a bad shard is exit 1", async () => {
    const snapshotDir = dir();
    await snapshotWith(DuckdbWriter.open(fakeBinary, standInRunner()), snapshotDir);
    const out = join(dir(), "teams.duckdb");
    const c = io();
    const runner = scriptRunner(() => undefined);
    const argv = ["duckdb", "--snapshot", snapshotDir, "--out", out, "--dataset", DATASET_DIR, "--shard-rows", "50"];
    expect(await runTeamsCli(argv, { ...c.io, env: { DUCKDB_BIN: fakeBinary }, runner })).toBe(0);
    expect(JSON.parse(c.lines.out[0] as string)).toMatchObject({ database: out, teams: 240 });
    const again = io();
    expect(await runTeamsCli(argv, { ...again.io, env: { DUCKDB_BIN: fakeBinary }, runner })).toBe(1);
    expect(again.lines.err[0]).toMatch(/^refused:/);
  });
});

describe("the teams command line, other paths", () => {
  const quiet = (env: Record<string, string>, runner?: Runner) => ({ out: () => undefined, err: () => undefined, env, ...(runner === undefined ? {} : { runner }) });
  it("takes a snapshot.json with --shards and --labels, refuses a binary that is not the pinned one, and rethrows what it does not know", async () => {
    const snapshotDir = dir();
    await snapshotWith(DuckdbWriter.open(fakeBinary, standInRunner()), snapshotDir);
    const runner = scriptRunner(() => undefined);
    const common = ["duckdb", "--snapshot", join(snapshotDir, "snapshot.json"), "--shards", snapshotDir, "--labels", join(REPO, "config/teams", manifest.generator.manifest.format_id, "labels.json"), "--dataset", DATASET_DIR, "--shard-rows", "50"];
    const out = join(dir(), "file.duckdb");
    expect(await runTeamsCli([...common, "--out", out], quiet({ DUCKDB_BIN: fakeBinary }, runner))).toBe(0);
    expect(await runTeamsCli([...common, "--out", join(dir(), "pinned.duckdb"), "--require-pinned"], quiet({ DUCKDB_BIN: fakeBinary }, runner))).toBe(1);
    expect(await runTeamsCli([...common, "--out", join(dir(), "rows.duckdb"), "--shard-rows", "0"], quiet({ DUCKDB_BIN: fakeBinary }, runner))).toBe(2);
    // a snapshot.json beside its shards, no --shards, no --dataset: the merged strata of the fixtures cannot be verified without the manifest
    expect(await runTeamsCli(["duckdb", "--snapshot", join(snapshotDir, "snapshot.json"), "--out", join(dir(), "nodataset.duckdb"), "--shard-rows", "50"], quiet({ DUCKDB_BIN: fakeBinary }, runner))).toBe(1);
    const broken = dir();
    mkdirSync(broken, { recursive: true });
    writeFileSync(join(broken, "snapshot.json"), "{");
    expect(await runTeamsCli(["duckdb", "--snapshot", broken, "--out", join(dir(), "x.duckdb")], quiet({ DUCKDB_BIN: fakeBinary }, runner))).toBe(2);
    const boom: Runner = () => {
      throw new Error("boom");
    };
    await expect(runTeamsCli([...common, "--out", join(dir(), "boom.duckdb")], quiet({ DUCKDB_BIN: fakeBinary }, boom))).rejects.toThrow("boom");
  });
});

describe("the SQL script and the repository guards", () => {
  it("creates the six tables of the layout and stamps its schema version", () => {
    const sql = readFileSync(SQL_SCRIPT, "utf8");
    for (const table of ["snapshot", "shards", "teams", "labels", "sets", "moves"]) expect(sql).toContain(`CREATE TABLE ${table} (`);
    expect(sql).toContain(`${DUCKDB_SCHEMA_VERSION} AS duckdb_schema_version`);
    expect(SQL_SCRIPT.pathname).toMatch(/working-db\.v1\.sql$/);
    for (const name of TABLE_FILES) expect(sql).toContain(`'${name}.jsonl'`);
  });

  it("keeps .duckdb files out of the repository: none tracked, the pattern ignored, the binary guard names it", () => {
    const tracked = execFileSync("git", ["ls-files", "-z"], { cwd: REPO, encoding: "utf8", maxBuffer: 1 << 26 }).split("\0");
    expect(tracked.filter((f) => /\.duckdb(\.wal)?$/.test(f))).toEqual([]);
    for (const path of ["teams.duckdb", "packages/team-dataset/teams.duckdb", "private-data/x/teams.duckdb.wal"]) {
      expect(spawnSync("git", ["check-ignore", "-q", path], { cwd: REPO }).status, path).toBe(0);
    }
    const guard = readFileSync(join(REPO, "scripts/check_binary_files.py"), "utf8");
    expect(guard).toContain('".duckdb"');
    expect(guard).toContain('".duckdb.wal"');
  });
});

// The real tool: CI installs the pinned one and sets ARENEON_REQUIRE_DUCKDB=1, where nothing below may be skipped.
const binary = locateDuckdb();
it.runIf(process.env.ARENEON_REQUIRE_DUCKDB === "1")("the working database tests have the pinned DuckDB tool", () => {
  expect(binary).toBeDefined();
  DuckdbWriter.open(binary as string).requirePinned(readPin());
});

describe.skipIf(binary === undefined)("the working database with the real DuckDB tool", () => {
  const snapshotDir = dir();
  const database = join(snapshotDir, "teams.duckdb");
  const outDir = join(snapshotDir, "exports");
  const query = (sql: string): Record<string, unknown>[] => {
    const text = execFileSync(binary as string, ["-readonly", "-json", database, "-c", sql], { cwd: snapshotDir, encoding: "utf8" }).trim();
    return text === "" ? [] : (JSON.parse(text) as Record<string, unknown>[]);
  };
  const byTeam = (keep: (t: (typeof teams)[number]) => boolean) => SPLITS.map((split) => ({ split, teams: count((t) => t.record.origin.split === split && keep(t)) })).filter((r) => r.teams > 0);

  let started: ReturnType<typeof buildWorkingDatabase> | undefined;
  const build = (): ReturnType<typeof buildWorkingDatabase> => (started ??= (async () => {
    const writer = DuckdbWriter.open(binary as string);
    await snapshotWith(writer, snapshotDir);
    return buildWorkingDatabase({ snapshotFile: join(snapshotDir, "snapshot.json"), shardsDir: snapshotDir, labelSet, datasetManifest: manifest, engine, writer, outFile: database, shardRows: 50 });
  })());

  it("loads every table, with the schema version in the snapshot table", async () => {
    const result = await build();
    expect(result.teams).toBe(240);
    expect(query("SELECT count(*) AS n FROM teams")).toEqual([{ n: 240 }]);
    expect(query("SELECT count(*) AS n FROM sets")).toEqual([{ n: result.sets }]);
    expect(query("SELECT count(*) AS n FROM moves")).toEqual([{ n: result.moves }]);
    expect(query("SELECT count(*) AS n FROM labels")).toEqual([{ n: 240 * labelSet.labels.length }]);
    expect(query("SELECT CAST(sum(rows) AS BIGINT) AS n FROM shards")).toEqual([{ n: 240 }]);
    expect(query("SELECT duckdb_schema_version AS v, teams, snapshot_id FROM snapshot")).toEqual([{ v: DUCKDB_SCHEMA_VERSION, teams: 240, snapshot_id: "working-db-test" }]);
    expect(query("SELECT count(DISTINCT paste) AS n FROM teams")).toEqual([{ n: 240 }]);
  });

  it("is a derived file the guards would refuse", async () => {
    await build();
    expect(readFileSync(database).subarray(8, 12).toString("latin1")).toBe("DUCK");
    // the file lives in a temp directory; the same name inside the repository is ignored
    expect(spawnSync("git", ["check-ignore", "-q", basename(database)], { cwd: REPO }).status).toBe(0);
  });

  it("refuses to build over an existing file", async () => {
    await build();
    await expect(buildWorkingDatabase({ snapshotFile: join(snapshotDir, "snapshot.json"), shardsDir: snapshotDir, labelSet, datasetManifest: manifest, engine, writer: DuckdbWriter.open(binary as string), outFile: database, shardRows: 50 })).rejects.toThrow(/never over one/);
  });

  it("builds through the command line too", async () => {
    await build();
    const out = join(dir(), "cli.duckdb");
    const lines: string[] = [];
    const code = await runTeamsCli(["duckdb", "--snapshot", snapshotDir, "--out", out, "--dataset", DATASET_DIR, "--shard-rows", "50", "--duckdb", binary as string], { out: (l) => lines.push(l), err: (l) => lines.push(l) });
    expect(lines).toHaveLength(1);
    expect(code).toBe(0);
    expect(JSON.parse(lines[0] as string)).toMatchObject({ teams: 240 });
  });

  // Every SQL recipe of the README runs here, against the expectations the engine-read fixtures give.
  const recipes = [...README.matchAll(/<!-- recipe: (\S+) -->\n```sql\n([\s\S]*?)```/g)].map((m) => ({ name: m[1] as string, sql: (m[2] as string).replaceAll("SNAPSHOT_DIR", snapshotDir).replaceAll("OUT_DIR", outDir) }));
  const recipe = (name: string): string => recipes.find((r) => r.name === name)?.sql ?? expect.fail(`no recipe ${name} in the README`);
  const expectations: Record<string, () => unknown> = {
    "filter-species": () => query(recipe("filter-species")),
    "filter-item": () => query(recipe("filter-item")),
    "filter-move": () => query(recipe("filter-move")),
    "filter-label": () => query(recipe("filter-label")),
    "read-parquet": () => query(recipe("read-parquet")),
    "export-selection": () => {
      execFileSync("mkdir", ["-p", outDir]);
      query(recipe("export-selection"));
      return query(`SELECT count(*) AS n FROM read_parquet('${outDir}/selection.parquet')`);
    },
  };
  const expected: Record<string, unknown> = {
    "filter-species": byTeam((t) => t.sets.some((s) => s.species === "Gallade")),
    "filter-item": byTeam((t) => t.sets.some((s) => s.item === "Wise Glasses")),
    "filter-move": byTeam((t) => t.sets.some((s) => s.moves.includes("Trick Room"))),
    "filter-label": byTeam((t) => t.record.labels.some((l) => l.label_id === "redirection" && l.value === "yes")),
    "read-parquet": SPLITS.map((split) => ({ split, teams: count((t) => t.record.origin.split === split) })),
    "export-selection": [{ n: count((t) => ["fake_out", "mega_stone"].every((id) => t.record.labels.some((l) => l.label_id === id && l.value === "yes"))) }],
  };

  it("has exactly the README's runnable recipes under test, and documents the ones it cannot run", () => {
    expect(recipes.map((r) => r.name).sort()).toEqual(Object.keys(expectations).sort());
    expect(README).toMatch(/<!-- recipe-not-run: r2 -->/);
    expect(README).toContain("duckdb -ui teams.duckdb");
  });

  for (const name of Object.keys(expectations)) {
    it(`recipe ${name}`, async () => {
      await build();
      const rows = (expectations[name] as () => unknown)();
      expect(rows).toEqual(expected[name]);
      expect(JSON.stringify(rows).length).toBeGreaterThan(2);
    });
  }
});
