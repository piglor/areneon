import { createHash } from "node:crypto";
import { chmodSync, existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { columnsFor, ShardError } from "../../src/shards/columns.ts";
import { convertDataset, verifySnapshot } from "../../src/shards/convert.ts";
import { type DuckdbPin, DuckdbWriter, locateDuckdb, PIN_FILE, platformKey, readPin, WITHHELD_MESSAGE } from "../../src/shards/duckdb.ts";
import { readShard } from "../../src/shards/reader.ts";
import { DATASET_DIR } from "./paths.ts";
import { at, expectationOf, fixtureRecords, labelSet, plan, SHARD_KEY, shardOrder, standInRunner } from "./support.ts";

const scratch = mkdtempSync(join(tmpdir(), "duckdb-writer-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));
const fakeBinary = join(scratch, "duckdb");
writeFileSync(fakeBinary, "not really duckdb\n");
const fakeSha = createHash("sha256").update(readFileSync(fakeBinary)).digest("hex");

describe("the pin", () => {
  it("is the committed one: an exact version, a release URL and SHA-256 per platform", () => {
    const pin = readPin();
    expect(pin.name).toBe("duckdb");
    expect(Object.keys(pin.assets).sort()).toEqual(["linux-amd64", "linux-arm64"]);
    for (const [platform, asset] of Object.entries(pin.assets)) {
      expect(asset.url).toBe(`https://github.com/duckdb/duckdb/releases/download/v${pin.version}/duckdb_cli-${platform}.zip`);
    }
  });
  it("refuses a pin of another shape", () => {
    const pin = readPin();
    const bad = (change: (p: DuckdbPin) => void): string => {
      const copy = structuredClone(pin);
      change(copy);
      const file = join(scratch, "pin.json");
      writeFileSync(file, JSON.stringify(copy));
      try {
        readPin(file);
        return "read";
      } catch (error) {
        expect(error).toBeInstanceOf(ShardError);
        return "refused";
      }
    };
    expect(bad(() => undefined)).toBe("read");
    expect(bad((p) => (p.version = "latest"))).toBe("refused");
    expect(bad((p) => ((p.assets["linux-amd64"] as { zip_sha256: string }).zip_sha256 = "abc"))).toBe("refused");
    expect(bad((p) => ((p.assets["linux-amd64"] as { url: string }).url = "https://example.com/duckdb.zip"))).toBe("refused");
    expect(bad((p) => ((p as { schema_version: number }).schema_version = 2))).toBe("refused");
    expect(readFileSync(PIN_FILE, "utf8")).toContain("\"version\"");
  });
  it("names the platform the way the release assets do", () => {
    expect(platformKey("linux", "x64")).toBe("linux-amd64");
    expect(platformKey("linux", "arm64")).toBe("linux-arm64");
  });
});

describe("locateDuckdb", () => {
  it("prefers DUCKDB_BIN, then searches the PATH, then gives up", () => {
    expect(locateDuckdb({ DUCKDB_BIN: "/opt/duckdb", PATH: scratch })).toBe("/opt/duckdb");
    chmodSync(fakeBinary, 0o755);
    expect(locateDuckdb({ PATH: `:${join(scratch, "nowhere")}:${scratch}` })).toBe(fakeBinary);
    expect(locateDuckdb({ DUCKDB_BIN: "", PATH: join(scratch, "nowhere") })).toBeUndefined();
    expect(locateDuckdb({})).toBeUndefined();
  });
});

describe("DuckdbWriter with a real child process", () => {
  it("runs the program it is given, reads its version, and reports a program that cannot start or fails", () => {
    const script = join(scratch, "fake-duckdb.sh");
    writeFileSync(script, "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo 'v9.8.7 (Test) abc'; exit 0; fi\ncat > /dev/null\nexit 3\n");
    chmodSync(script, 0o755);
    const writer = DuckdbWriter.open(script);
    expect(writer.identity.version).toBe("9.8.7");
    expect(() => writer.writeShard("rows", "out", columnsFor(labelSet))).toThrow(/DuckDB failed \(3\)/);
    expect(() => DuckdbWriter.open(join(scratch, "does-not-exist"))).toThrow(/did not answer --version as DuckDB: .*ENOENT/);
  });
});

describe("DuckdbWriter with a stand-in tool", () => {
  it("takes its identity from the binary that answered: version and SHA-256", () => {
    const writer = DuckdbWriter.open(fakeBinary, standInRunner("v1.5.6 (Variegata) 069cc9f9b5"));
    expect(writer.identity).toEqual({ name: "duckdb", version: "1.5.6", binary_sha256: fakeSha });
  });
  it("refuses a program that does not answer as DuckDB", () => {
    expect(() => DuckdbWriter.open(fakeBinary, () => ({ status: 1, stdout: "", stderr: "boom" }))).toThrow(/did not answer --version as DuckDB: boom/);
    expect(() => DuckdbWriter.open(fakeBinary, () => ({ status: 0, stdout: "GNU bash 5\n", stderr: "" }))).toThrow(/did not answer/);
  });
  it("requires the pinned version and the pinned binary", () => {
    const writer = DuckdbWriter.open(fakeBinary, standInRunner());
    const pin: DuckdbPin = { schema_version: 1, name: "duckdb", version: "1.5.6", assets: { "linux-amd64": { url: "u", zip_sha256: "0".repeat(64), binary_sha256: fakeSha } } };
    writer.requirePinned(pin, "linux-amd64");
    expect(() => writer.requirePinned(pin, "linux-arm64")).toThrow(/no binary for linux-arm64/);
    expect(() => writer.requirePinned({ ...pin, version: "1.5.7" }, "linux-amd64")).toThrow(/not the pinned 1.5.7/);
    expect(() => writer.requirePinned({ ...pin, assets: { "linux-amd64": { ...(pin.assets["linux-amd64"] as DuckdbPin["assets"][string]), binary_sha256: "1".repeat(64) } } }, "linux-amd64")).toThrow(/SHA-256 .* is not the pinned/);
    expect(() => writer.requirePinned(pin)).not.toThrow(/no binary for/);
  });
  it("sends one single-threaded COPY with typed columns, zstd and the schema version, in a scratch HOME", () => {
    const runner = standInRunner();
    const writer = DuckdbWriter.open(fakeBinary, runner);
    const rows = join(scratch, "rows'x.jsonl");
    writeFileSync(rows, shardOrder(fixtureRecords("dev").slice(0, 3)).map((s) => `${JSON.stringify(s.row)}\n`).join(""));
    const out = join(scratch, "o.parquet");
    writer.writeShard(rows, out, columnsFor(labelSet), { rowGroupSize: 2, compressionLevel: 9 });
    const call = runner.calls.at(-1) as { args: readonly string[]; input: string; env: Record<string, string> };
    expect(call.args).toEqual(["-bail", "-batch"]);
    expect(call.input).toContain("SET threads = 1;");
    expect(call.input).toContain("SET preserve_insertion_order = true;");
    expect(call.input).toContain("COMPRESSION zstd, COMPRESSION_LEVEL 9, ROW_GROUP_SIZE 2");
    expect(call.input).toContain("KV_METADATA {areneon_parquet_schema_version: '1'}");
    expect(call.input).toContain("'label:fake_out': 'VARCHAR'");
    expect(call.input).toContain("'labels': 'STRUCT(label_id VARCHAR, value VARCHAR)[]'");
    expect(call.input).toContain("rows''x.jsonl");
    expect(call.env.PATH).toBe("");
    expect(call.env.HOME).toMatch(/duckdb-home-/);
    expect(readFileSync(out).byteLength).toBeGreaterThan(0);
  });
  it("refuses bad options, control characters in paths, and a failing tool", () => {
    const writer = DuckdbWriter.open(fakeBinary, standInRunner());
    const cols = columnsFor(labelSet);
    expect(() => writer.writeShard("a", "b", cols, { rowGroupSize: 0 })).toThrow(/row group size/);
    expect(() => writer.writeShard("a", "b", cols, { compressionLevel: 99 })).toThrow(/compression level/);
    expect(() => writer.writeShard("a\nb", "b", cols)).toThrow(/control character/);
    const failing = DuckdbWriter.open(fakeBinary, (_f, args) => (args[0] === "--version" ? { status: 0, stdout: "v1.5.6 (x)\n", stderr: "" } : { status: 1, stdout: "", stderr: "IO Error: no such file\n" }));
    expect(() => failing.writeShard("a", "b", cols)).toThrow(/DuckDB failed \(1\): IO Error: no such file/);
  });
  it("on the private path runs in a scratch directory of the caller's, with a minimal environment, and withholds the tool's stderr", () => {
    const dir = mkdtempSync(join(scratch, "iso-"));
    const cols = columnsFor(labelSet);
    const runner = standInRunner();
    const writer = DuckdbWriter.open(fakeBinary, runner);
    const rows = join(dir, "r.jsonl");
    writeFileSync(rows, shardOrder(fixtureRecords("dev").slice(0, 3)).map((s) => `${JSON.stringify(s.row)}\n`).join(""));
    writer.writeShard(rows, join(dir, "o.parquet"), cols, { isolated: { dir } });
    const call = at(runner.calls, -1);
    expect(call.env).toEqual({ PATH: "", HOME: call.cwd, TMPDIR: call.cwd, DUCKDB_HISTORY: "/dev/null" });
    expect(call.cwd?.startsWith(dir)).toBe(true);
    expect(existsSync(call.cwd as string)).toBe(false);
    const failing = DuckdbWriter.open(fakeBinary, (_f, args) => (args[0] === "--version" ? { status: 0, stdout: "v1.5.6 (x)\n", stderr: "" } : { status: 1, stdout: "", stderr: "Conversion Error: PASTE-TEXT" }));
    expect(() => failing.writeShard("a", "b", cols, { isolated: { dir } })).toThrow(WITHHELD_MESSAGE);
    expect(() => failing.writeShard("a", "b", cols, { isolated: { dir } })).not.toThrow(/PASTE-TEXT/);
    expect(readdirSync(dir).sort()).toEqual(["o.parquet", "r.jsonl"]);
  });
});

// The same checks with a real DuckDB command-line tool, where one is installed
// (DUCKDB_BIN or the PATH); CI installs the pinned one. A second version in
// DUCKDB_BIN_ALT makes the writer-version test cross versions.
const binary = locateDuckdb();
const alternate = process.env.DUCKDB_BIN_ALT;

// CI sets ARENEON_REQUIRE_DUCKDB=1 after installing the pinned tool: there the tests below
// may not be skipped, and the tool must be the pinned one.
it.runIf(process.env.ARENEON_REQUIRE_DUCKDB === "1")("the DuckDB tool CI installed is the pinned one", () => {
  expect(binary).toBeDefined();
  DuckdbWriter.open(binary as string).requirePinned(readPin());
});

describe.skipIf(binary === undefined)("DuckdbWriter with the real tool", () => {
  const staged = shardOrder(fixtureRecords("dev"));
  const expected = expectationOf(staged, "dev");
  const rowsFile = join(scratch, "real-rows.jsonl");
  writeFileSync(rowsFile, staged.map((s) => `${JSON.stringify(s.row)}\n`).join(""));
  const write = (writer: DuckdbWriter, name: string, options = {}): Uint8Array => {
    const out = join(scratch, `${name}.parquet`);
    writer.writeShard(rowsFile, out, columnsFor(labelSet), options);
    return readFileSync(out);
  };

  it("writes a shard that reads back to the same records", async () => {
    const writer = DuckdbWriter.open(binary as string);
    const bytes = write(writer, "a");
    const read = await readShard(bytes, expected, plan);
    expect(read.records).toEqual(staged.map((s) => s.record));
  });

  it("writes different bytes at other settings and keeps the logical hash", async () => {
    const writer = DuckdbWriter.open(binary as string);
    const variants = [
      write(writer, "zstd"),
      write(writer, "zstd-9", { compressionLevel: 9 }),
      write(writer, "snappy", { compression: "snappy" }),
      write(writer, "small-groups", { rowGroupSize: 7 }),
      write(writer, "plain", { compression: "uncompressed" }),
    ];
    const digests = new Set(variants.map((b) => createHash("sha256").update(b).digest("hex")));
    expect(digests.size).toBeGreaterThan(1);
    for (const bytes of variants) expect((await readShard(bytes, expected, plan)).recordHashes).toEqual((await readShard(variants[0] as Uint8Array, expected, plan)).recordHashes);
  });

  it("writes the same bytes twice (the writer is deterministic on one machine)", () => {
    const writer = DuckdbWriter.open(binary as string);
    expect(createHash("sha256").update(write(writer, "x1")).digest("hex")).toBe(createHash("sha256").update(write(writer, "x2")).digest("hex"));
  });

  it.skipIf(alternate === undefined)("two DuckDB versions: different file bytes, identical logical hashes", async () => {
    const a = DuckdbWriter.open(binary as string);
    const b = DuckdbWriter.open(alternate as string);
    expect(a.identity.version).not.toBe(b.identity.version);
    const first = write(a, "va");
    const second = write(b, "vb");
    expect(createHash("sha256").update(first).digest("hex")).not.toBe(createHash("sha256").update(second).digest("hex"));
    const read1 = await readShard(first, expected, plan);
    const read2 = await readShard(second, expected, plan);
    expect(read2.recordHashes).toEqual(read1.recordHashes);
    expect(expected.logical_hash).toBe(expectationOf(staged, "dev").logical_hash);
  });

  it("converts a dataset: the logical identity equals the other writer's, the physical identity does not", async () => {
    const real = await convertDataset({ datasetDir: DATASET_DIR, outDir: join(scratch, "real-out"), labelSet, shardKey: SHARD_KEY, snapshotId: "real", splits: ["dev", "validation"], writer: DuckdbWriter.open(binary as string), shardRows: 50 });
    const standIn = await convertDataset({ datasetDir: DATASET_DIR, outDir: join(scratch, "stand-in-out"), labelSet, shardKey: SHARD_KEY, snapshotId: "real", splits: ["dev", "validation"], writer: DuckdbWriter.open(fakeBinary, standInRunner()), shardRows: 50 });
    expect(real.snapshot.logical_snapshot_hash).toBe(standIn.snapshot.logical_snapshot_hash);
    expect(real.snapshot.shards.map((x) => x.logical_hash)).toEqual(standIn.snapshot.shards.map((x) => x.logical_hash));
    expect(real.snapshot.shards.map((x) => x.file_sha256)).not.toEqual(standIn.snapshot.shards.map((x) => x.file_sha256));
    expect(await verifySnapshot({ snapshotFile: real.snapshotFile, shardsDir: join(scratch, "real-out"), datasetDir: DATASET_DIR, labelSet, shardRows: 50 })).toEqual({ dev: 120, validation: 120 });
  });

  it("reports an error for a rows file DuckDB cannot read", () => {
    const writer = DuckdbWriter.open(binary as string);
    expect(() => writer.writeShard(join(scratch, "missing.jsonl"), join(scratch, "m.parquet"), columnsFor(labelSet))).toThrow(/DuckDB failed/);
  });
});
