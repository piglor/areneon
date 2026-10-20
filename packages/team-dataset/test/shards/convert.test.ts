import { createHash } from "node:crypto";
import { cpSync, existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { jcs, readTeamSnapshotManifest, recordHash, snapshotHash, type TeamSnapshotManifest } from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it } from "vitest";
import { pretty } from "../../src/dataset-io.ts";
import type { HashIndex } from "../../src/dataset-reader.ts";
import { refuse, ShardError, shardRecordHash } from "../../src/shards/columns.ts";
import { convertDataset, datasetRecordsFile, fileLines, verifySnapshot } from "../../src/shards/convert.ts";
import { DuckdbWriter } from "../../src/shards/duckdb.ts";
import { cellLabels } from "../../src/labeller.ts";
import { shardAssigner, shardCount } from "../../src/shards/layout.ts";
import { buildSnapshotManifest, checkFixtureAgreement, datasetPlan, eachSnapshotShard, readSnapshotShards, shardsInDirectory } from "../../src/shards/snapshot.ts";
import { DATASET_DIR } from "./paths.ts";
import { expectationOf, fixtureRecords, labelSet, manifest, parquetOf, plan, shardOrder, SHARD_KEY, standInRunner } from "./support.ts";

const scratch = mkdtempSync(join(tmpdir(), "shards-convert-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));
let n = 0;
const dir = (): string => join(scratch, `d${n++}`);
const writerFile = join(scratch, "duckdb");
writeFileSync(writerFile, "stand-in\n");
const writerOf = (version = "v1.5.6 (Variegata) 0123456789") => DuckdbWriter.open(writerFile, standInRunner(version));

const SPLITS = ["dev", "validation"] as const;
const convert = (out: string, options: Partial<Parameters<typeof convertDataset>[0]> = {}) =>
  convertDataset({ datasetDir: DATASET_DIR, outDir: out, labelSet, shardKey: SHARD_KEY, snapshotId: "test-snapshot", splits: SPLITS, writer: writerOf(), shardRows: 50, ...options });

describe("convertDataset", () => {
  it("writes shards named by their SHA-256 and a snapshot manifest the manifest reader accepts", async () => {
    const out = dir();
    const { snapshot, snapshotFile } = await convert(out);
    expect(snapshot.shards).toHaveLength(6);
    expect(snapshot.shards.map((s) => s.rows).reduce((a, b) => a + b)).toBe(240);
    expect(snapshot.writer).toMatchObject({ name: "duckdb", version: "1.5.6" });
    expect(snapshot.dataset_hashes).toEqual([recordHash(manifest)]);
    expect(snapshot.manifest).toEqual(manifest.generator.manifest);
    expect(snapshot.vocabulary_hash).toBe(manifest.generator.vocabulary_hash);
    const files = readdirSync(join(out, "pub/v1/shards")).sort();
    expect(files).toEqual(snapshot.shards.map((s) => s.file_sha256).sort());
    for (const shard of snapshot.shards) {
      const bytes = readFileSync(join(out, shard.object_key));
      expect(createHash("sha256").update(bytes).digest("hex")).toBe(shard.file_sha256);
      expect(bytes.byteLength).toBe(shard.bytes);
    }
    expect(readFileSync(snapshotFile, "utf8")).toBe(pretty(snapshot));
    expect(readTeamSnapshotManifest(JSON.parse(readFileSync(snapshotFile, "utf8")))).toEqual(snapshot);
  });

  it("is deterministic: the same inputs and writer give the same files and manifest", async () => {
    const a = await convert(dir());
    const b = await convert(dir());
    expect(b.snapshot).toEqual(a.snapshot);
    expect(snapshotHash(b.snapshot)).toBe(snapshotHash(a.snapshot));
  });

  it("gives different physical identity and the same logical identity when the writer or its settings change", async () => {
    const base = await convert(dir());
    const otherVersion = await convert(dir(), { writer: writerOf("v1.6.0 (Other) 0123456789") });
    const otherSettings = await convert(dir(), { write: { rowGroupSize: 13, compression: "uncompressed" } });
    for (const other of [otherVersion, otherSettings]) {
      expect(other.snapshot.logical_snapshot_hash).toBe(base.snapshot.logical_snapshot_hash);
      expect(snapshotHash(other.snapshot)).not.toBe(snapshotHash(base.snapshot));
      expect(other.snapshot.shards.map((s) => s.logical_hash)).toEqual(base.snapshot.shards.map((s) => s.logical_hash));
    }
    expect(otherSettings.snapshot.shards.map((s) => s.file_sha256)).not.toEqual(base.snapshot.shards.map((s) => s.file_sha256));
    expect(otherVersion.snapshot.writer.version).toBe("1.6.0");
  });

  it("round-trips: the shards read back to the committed records, byte-identical in canonical form", async () => {
    const out = dir();
    const { snapshot, snapshotFile } = await convert(out);
    const read = await readSnapshotShards(snapshot, shardsInDirectory(out), plan, 50);
    for (const split of SPLITS) {
      const records = read.shards.filter((_, k) => snapshot.shards[k]?.split === split).flatMap((s) => s.records).sort((a, b) => a.origin.index - b.origin.index);
      const lines = readFileSync(join(DATASET_DIR, `teams-${split}.jsonl`), "utf8").split("\n").filter((l) => l !== "");
      expect(records.map((r) => jcs(r as never))).toEqual(lines);
    }
    expect(await verifySnapshot({ snapshotFile, shardsDir: out, datasetDir: DATASET_DIR, labelSet, shardRows: 50 })).toEqual({ dev: 120, validation: 120 });
  });

  it("agrees with the committed fixture's record hashes, split by split (and refuses when it does not)", async () => {
    const out = dir();
    const { snapshot } = await convert(out);
    const hashes = JSON.parse(readFileSync(join(DATASET_DIR, "hashes.json"), "utf8")) as HashIndex;
    const perSplit = { dev: [] as string[], validation: [] as string[] };
    for await (const { entry, read } of eachSnapshotShard(snapshot, shardsInDirectory(out), plan, 50)) perSplit[entry.split as "dev"].push(...read.recordHashes);
    for (const split of SPLITS) checkFixtureAgreement(split, perSplit[split], hashes);
    expect(() => checkFixtureAgreement("dev", perSplit.dev.slice(1), hashes)).toThrow(/the shards hold 119 records, the committed fixture 120/);
    expect(() => checkFixtureAgreement("dev", [...perSplit.dev.slice(1), "f".repeat(64)], hashes)).toThrow(/differs from the committed fixture's/);
    expect(() => checkFixtureAgreement("validation", perSplit.dev, hashes)).toThrow(ShardError);
  });

  it("converts one split, and the training split of a generator directory (records.<split>.jsonl)", async () => {
    const out = dir();
    const { snapshot } = await convert(out, { splits: ["validation"] });
    expect(new Set(snapshot.shards.map((s) => s.split))).toEqual(new Set(["validation"]));
    expect(datasetRecordsFile(DATASET_DIR, "dev")).toBe(join(DATASET_DIR, "teams-dev.jsonl"));
    expect(() => datasetRecordsFile(DATASET_DIR, "training")).toThrow(/no records file for the training split/);
  });

  it("refuses a records file that disagrees with the manifest", async () => {
    const copy = dir();
    cpSync(DATASET_DIR, copy, { recursive: true });
    const lines = readFileSync(join(copy, "teams-dev.jsonl"), "utf8").split("\n").filter((l) => l !== "");
    writeFileSync(join(copy, "teams-dev.jsonl"), `${lines.slice(1).join("\n")}\n`);
    await expect(convert(dir(), { datasetDir: copy })).rejects.toThrow(/holds 119 records, the manifest says 120/);
    writeFileSync(join(copy, "teams-dev.jsonl"), `${lines.map((l, k) => (k === 0 ? ` ${l}` : l)).join("\n")}\n`);
    await expect(convert(dir(), { datasetDir: copy })).rejects.toThrow(/not in canonical \(JCS\) form/);
    const surrogate = lines.map((l, k) => (k === 0 ? l.replace(/"paste":"/, '"paste":"\\ud800') : l));
    writeFileSync(join(copy, "teams-dev.jsonl"), `${surrogate.join("\n")}\n`);
    await expect(convert(dir(), { datasetDir: copy })).rejects.toThrow(/lone surrogate/);
    writeFileSync(join(copy, "teams-dev.jsonl"), '{"a":1}\n');
    await expect(convert(dir(), { datasetDir: copy })).rejects.toThrow(/line 1 is not a TeamRecord/);
    const other = lines.map((l) => l.replace('"split":"dev"', '"split":"validation"'));
    writeFileSync(join(copy, "teams-dev.jsonl"), `${other.join("\n")}\n`);
    await expect(convert(dir(), { datasetDir: copy })).rejects.toThrow(/its origin is fixtures-v1\/validation/);
    expect(existsSync(copy)).toBe(true);
  });

  it("refuses a split the manifest does not plan", async () => {
    const copy = dir();
    cpSync(DATASET_DIR, copy, { recursive: true });
    const edited = { ...manifest, splits: manifest.splits.filter((p) => p.split !== "validation") };
    writeFileSync(join(copy, "manifest.json"), pretty(edited));
    await expect(convert(dir(), { datasetDir: copy, splits: ["validation"] })).rejects.toThrow(/the manifest has no validation split/);
  });

  it("refuses a shard key that leaves a shard empty", async () => {
    await expect(convert(dir(), { shardRows: 1 })).rejects.toThrow(/received no row; use another shard key/);
    await expect(convert(dir(), { shardKey: "xyz" })).rejects.toThrow(/64 lowercase hex/);
  });

  it("refuses a shard the writer wrote wrongly", async () => {
    const broken = DuckdbWriter.open(writerFile, (file, args, input, env) => {
      const result = standInRunner()(file, args, input, env);
      const out = /TO '([^']*)'/.exec(input)?.[1];
      if (out !== undefined && result.status === 0) writeFileSync(out, "garbage");
      return result;
    });
    await expect(convert(dir(), { writer: broken })).rejects.toThrow(/not a readable Parquet file/);
  });
});

describe("verifySnapshot", () => {
  it("refuses a tampered shard file, a wrong size, a missing shard, and a snapshot of another dataset", async () => {
    const out = dir();
    const { snapshot, snapshotFile } = await convert(out);
    const check = (options: Partial<Parameters<typeof verifySnapshot>[0]> = {}) =>
      verifySnapshot({ snapshotFile, shardsDir: out, datasetDir: DATASET_DIR, labelSet, shardRows: 50, ...options });
    const first = snapshot.shards[0] as TeamSnapshotManifest["shards"][number];
    const file = join(out, first.object_key);
    const original = readFileSync(file);
    const flipped = Buffer.from(original);
    flipped[original.length >> 1] = (flipped[original.length >> 1] as number) ^ 0xff;
    writeFileSync(file, flipped);
    await expect(check()).rejects.toThrow(/SHA-256 is not/);
    writeFileSync(file, Buffer.concat([original, Buffer.from([0])]));
    await expect(check()).rejects.toThrow(/the file has \d+ bytes/);
    rmSync(file);
    await expect(check()).rejects.toThrow(/ENOENT/);
    writeFileSync(file, original);
    await expect(check()).resolves.toEqual({ dev: 120, validation: 120 });
    const stranger = { ...snapshot, dataset_hashes: ["0".repeat(64)] };
    writeFileSync(join(out, "other.json"), pretty({ ...stranger, logical_snapshot_hash: buildSnapshotManifest({ ...stranger, shards: snapshot.shards }).logical_snapshot_hash }));
    await expect(check({ snapshotFile: join(out, "other.json") })).rejects.toThrow(/does not list this dataset's hash/);
  });

  it("refuses records that are not the dataset's, or that differ from hashes.json", async () => {
    const out = dir();
    const { snapshotFile } = await convert(out);
    const copy = dir();
    cpSync(DATASET_DIR, copy, { recursive: true });
    const lines = readFileSync(join(copy, "teams-dev.jsonl"), "utf8").split("\n").filter((l) => l !== "");
    const swapped = [...lines];
    [swapped[0], swapped[1]] = [lines[1] as string, lines[0] as string];
    writeFileSync(join(copy, "teams-dev.jsonl"), `${swapped.join("\n")}\n`);
    await expect(verifySnapshot({ snapshotFile, shardsDir: out, datasetDir: copy, labelSet, shardRows: 50 })).rejects.toThrow(/record 0 of the shards is not the dataset's record 0/);
    writeFileSync(join(copy, "teams-dev.jsonl"), `${lines.join("\n")}\n`);
    const hashes = JSON.parse(readFileSync(join(copy, "hashes.json"), "utf8")) as HashIndex;
    hashes.splits.dev.record_hashes[0] = "0".repeat(64);
    writeFileSync(join(copy, "hashes.json"), JSON.stringify(hashes));
    await expect(verifySnapshot({ snapshotFile, shardsDir: out, datasetDir: copy, labelSet, shardRows: 50 })).rejects.toThrow(/differs from the committed fixture's/);
  });

  it("checks the shards against the dataset file alone when the dataset has no hashes.json", async () => {
    const out = dir();
    const { snapshotFile } = await convert(out);
    const copy = dir();
    cpSync(DATASET_DIR, copy, { recursive: true });
    rmSync(join(copy, "hashes.json"));
    await expect(verifySnapshot({ snapshotFile, shardsDir: out, datasetDir: copy, labelSet, shardRows: 50 })).resolves.toEqual({ dev: 120, validation: 120 });
  });

  it("refuses a manifest whose shard count is not ceil(rows / 2^18)", async () => {
    const out = dir();
    const { snapshotFile } = await convert(out);
    await expect(verifySnapshot({ snapshotFile, shardsDir: out, datasetDir: DATASET_DIR, labelSet })).rejects.toThrow(/120 rows need 1 shards, the manifest lists 3/);
  });
});

describe("origin indices", () => {
  const recordsWith = (change: (records: ReturnType<typeof fixtureRecords>) => void): string => {
    const copy = dir();
    cpSync(DATASET_DIR, copy, { recursive: true });
    const records = fixtureRecords("dev");
    change(records);
    writeFileSync(join(copy, "teams-dev.jsonl"), records.map((r) => `${jcs(r as never)}\n`).join(""));
    return copy;
  };
  const verifyDev = async (datasetDir: string) => {
    const out = dir();
    const { snapshotFile } = await convert(out, { datasetDir, splits: ["dev"] });
    return verifySnapshot({ snapshotFile, shardsDir: out, datasetDir, labelSet, shardRows: 50 });
  };
  it("refuses two records with one origin index, and a gap in the indices", async () => {
    await expect(verifyDev(recordsWith((r) => ((r[1] as { origin: { index: number } }).origin.index = 0)))).rejects.toThrow(/origin index 0 is in two records/);
    await expect(verifyDev(recordsWith((r) => ((r[1] as { origin: { index: number } }).origin.index = 500)))).rejects.toThrow(/origin indices of the shards' records have gaps/);
  });
});

describe("the layout rules of a snapshot", () => {
  it("refuses two shards of a split that hold the same shard index", async () => {
    const out = dir();
    const { snapshot } = await convert(out, { splits: ["validation"], shardRows: 40 });
    // Rebuild the validation split as two shards whose rows all belong to index 0 of 2.
    const assign = shardAssigner(SHARD_KEY, 2);
    const mine = shardOrder(fixtureRecords("validation")).filter((s) => assign(s.canonical_team_hash) === 0);
    const halves = [mine.slice(0, 41), mine.slice(41)];
    const entries = halves.map((half) => {
      const bytes = parquetOf(half.map((s) => s.row));
      const sha = createHash("sha256").update(bytes).digest("hex");
      writeFileSync(join(out, "pub/v1/shards", sha), bytes);
      return { object_key: `pub/v1/shards/${sha}`, file_sha256: sha, bytes: bytes.byteLength, ...expectationOf(half, "validation") };
    });
    const rebuilt = buildSnapshotManifest({ ...snapshot, shards: entries as never, dataset_hashes: snapshot.dataset_hashes });
    expect(mine.length).toBeGreaterThan(41);
    const entryCount = rebuilt.shards.length;
    expect(entryCount).toBe(2);
    expect(shardCount(mine.length, 40)).toBe(2);
    await expect(readSnapshotShards(rebuilt, shardsInDirectory(out), plan, 40)).rejects.toThrow(/shard index 0 is held by two shards of validation/);
  });
});

describe("datasetPlan", () => {
  it("numbers a cell by its merged stratum, and an unmerged cell by itself", () => {
    const merged = { ...manifest, merged_cells: [{ stratum: 0, cells: [0, 1] as [number, number] }] };
    const labelsOf = (cell: number) => cellLabels(labelSet, cell);
    expect(datasetPlan(merged, labelSet).stratumOf(labelsOf(1))).toBe(0);
    expect(datasetPlan(merged, labelSet).stratumOf(labelsOf(2))).toBe(2);
    expect(datasetPlan({ ...manifest, merged_cells: undefined } as never, labelSet).stratumOf(labelsOf(1))).toBe(1);
  });
});

describe("the fixture records and the shards", () => {
  it("have the record hash the bridge computes, with or without the fast path", () => {
    for (const split of SPLITS) for (const record of fixtureRecords(split)) expect(shardRecordHash(record)).toBe(recordHash(record));
  });
  it("hash awkward pastes the same way, and refuse a lone surrogate as JCS does", () => {
    const base = fixtureRecords("dev")[0] as ReturnType<typeof fixtureRecords>[number];
    for (const paste of ['quote " backslash \\ tab \t newline \n', "unicode   é \u{1f600} \u0007", "x".repeat(16384)]) {
      const record = { ...base, paste };
      expect(shardRecordHash(record)).toBe(recordHash(record));
    }
    expect(() => shardRecordHash({ ...base, paste: "\ud800" })).toThrow(/lone surrogate/);
    expect(() => recordHash({ ...base, paste: "\ud800" })).toThrow(/lone surrogate/);
  });
  it("let a line reader see every line, including a last line without a newline", async () => {
    const file = join(scratch, "lines.txt");
    writeFileSync(file, "a\nb\n\nc");
    const lines: string[] = [];
    for await (const line of fileLines(file)) lines.push(line);
    expect(lines).toEqual(["a", "b", "", "c"]);
    expect(refuse).toBeTypeOf("function");
  });
});
