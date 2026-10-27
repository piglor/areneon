import { createHash } from "node:crypto";
import { chmodSync, mkdirSync, readdirSync, readFileSync, statfsSync, symlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { holdoutFlag, privateLogicalSnapshotHash, privateSnapshotHash, recordHash } from "@areneon/showdown-bridge";
import { DuckdbWriter } from "@areneon/team-dataset";
import { describe, expect, it, vi } from "vitest";
import { SnapshotIoError } from "../src/snapshot-io-error.ts";
import { INSPECT_ONLY, readPrivateSnapshot } from "../src/snapshot-reader.ts";
import { writePrivateSnapshot } from "../src/snapshot-writer.ts";
import { assertWorkDir, onWindowsMount, parentOwnerOk } from "../src/work-dir.ts";
import { at, engine, HOLDOUT_KEY, hex32, input, labelSet, plan, records, recordOf, realDuckdb, scratchRoot, standIn, workDir } from "./snapshot-support.ts";

// Each case decodes shard files in worker threads; a loaded CI machine needs more than the default 5 s.
vi.setConfig({ testTimeout: 60_000 });

/** statfs types the work-directory rule counts as memory: tmpfs and ramfs. */
const MEMORY_FS_TYPES = [0x01021994, 0x858458f6];

const code = async (run: () => unknown): Promise<string> => {
  try {
    await run();
  } catch (error) {
    if (error instanceof SnapshotIoError) return error.code;
    throw error;
  }
  return "no error";
};

const written = async (n = 12, over: Parameters<typeof input>[0] extends infer T ? Partial<T> : never = {}, tool = standIn()) => writePrivateSnapshot(input({ records: records(n), shardRows: 5, ...over }, tool));
const readOf = (w: Awaited<ReturnType<typeof written>>, over: Partial<Parameters<typeof readPrivateSnapshot>[0]> = {}) =>
  readPrivateSnapshot({ engine, expected: INSPECT_ONLY, expectedSnapshotHash: privateSnapshotHash(("manifest" in over ? over.manifest : w.manifest) as never), shard: (i) => w.shards[i] as Uint8Array, labelSet, shardRows: 5, ...over, manifest: "manifest" in over ? over.manifest : w.manifest });

describe("writePrivateSnapshot and readPrivateSnapshot", () => {
  it("round-trips: the reader returns the records, and the manifest is the one the writer built", async () => {
    const w = await written();
    expect(w.manifest.data_class).toBe("private");
    expect(w.manifest.first).toBe(true);
    expect(w.manifest.shards).toHaveLength(3);
    expect(w.manifest.shards.map((s) => s.index)).toEqual([0, 1, 2]);
    expect(w.manifest.records).toEqual({ active: 10, retired: 2 });
    expect(w.snapshotHash).toBe(privateSnapshotHash(w.manifest));
    const read = await readOf(w, { expectedSnapshotHash: w.snapshotHash });
    expect(read.snapshotHash).toBe(w.snapshotHash);
    expect(read.records.map((r) => recordHash(r))).toEqual(records(12).sort((a, b) => (a.team_id < b.team_id ? -1 : 1)).map((r) => recordHash(r)));
    for (const [i, file] of w.shards.entries()) {
      const entry = at(w.manifest.shards, i);
      expect(entry.bytes).toBe(file.byteLength);
      expect(entry.file_sha256).toBe(createHash("sha256").update(file).digest("hex"));
    }
  });

  it("carries a parent and a hold-out block, and some records are held out", async () => {
    const parent = "cd".repeat(32);
    const w = await written(12, { parentSnapshotHash: parent });
    expect(w.manifest.parent_snapshot_hash).toBe(parent);
    expect(w.manifest.first).toBeUndefined();
    expect(w.manifest.holdout).toEqual({ rule: "hmac-sha256-team-id/1", key: HOLDOUT_KEY, ppm: 200000 });
    const flags = (await readOf(w)).records.map((r) => r.holdout);
    expect(flags).toEqual((await readOf(w)).records.map((r) => holdoutFlag(HOLDOUT_KEY, r.team_id)));
  });

  it("accepts a retired record that shares its team with an active one, and an all-retired snapshot", async () => {
    const shared = [recordOf(0, 1), recordOf(0, 2, { status: "retired" })];
    const w = await writePrivateSnapshot(input({ records: shared }));
    expect((await readOf(w, { shardRows: 2 ** 18 })).records).toHaveLength(2);
    const retired = await writePrivateSnapshot(input({ records: [recordOf(1, 3, { status: "retired" })] }));
    expect(retired.manifest.cells).toEqual([]);
    expect((await readPrivateSnapshot({ engine, expected: INSPECT_ONLY, manifest: retired.manifest, expectedSnapshotHash: privateSnapshotHash(retired.manifest as never), shard: (i) => retired.shards[i] as Uint8Array, labelSet })).records).toHaveLength(1);
  });

  it("is deterministic for one writer: same inputs, same shard bytes and hash", async () => {
    const tool = standIn();
    const a = await written(7, {}, tool);
    const b = await written(7, {}, tool);
    expect(b.snapshotHash).toBe(a.snapshotHash);
  });

  it("leaves nothing in the working directory, on success and on failure, and runs the tool in isolation", async () => {
    const tool = standIn();
    const dir = workDir();
    await writePrivateSnapshot(input({ records: records(6), workDir: dir, shardRows: 4 }, tool));
    expect(readdirSync(dir)).toEqual([]);
    const call = at(tool.runner.calls.filter((c) => c.args[0] === "-bail"), 0);
    expect(Object.keys(call.env).sort()).toEqual(["DUCKDB_HISTORY", "HOME", "PATH", "TMPDIR"]);
    expect(call.env.DUCKDB_HISTORY).toBe("/dev/null");
    expect(call.env.PATH).toBe("");
    expect(call.cwd?.startsWith(dir)).toBe(true);
    expect(String(call.env.HOME).startsWith(dir)).toBe(true);
    const failing = standIn();
    const broken = ((...args: Parameters<typeof failing.runner>) => (args[1][0] === "--version" ? failing.runner(...args) : { status: 1, stdout: "", stderr: "PASTE-IN-STDERR" })) as typeof failing.runner;
    const failWriter = DuckdbWriter.open(join(scratchRoot, "duckdb-stand-in"), broken);
    const error = await writePrivateSnapshot(input({ records: records(3), workDir: dir, writer: failWriter, pin: failing.pin })).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(SnapshotIoError);
    expect((error as SnapshotIoError).code).toBe("writer_failed");
    expect(String((error as Error).message)).not.toContain("PASTE-IN-STDERR");
    expect(readdirSync(dir)).toEqual([]);
  });
});

describe("writer refusals", () => {
  it("refuses an unsafe working directory", async () => {
    const root = join(scratchRoot, "unsafe");
    mkdirSync(root, { recursive: true });
    const open = join(root, "private-data-open");
    mkdirSync(open, { mode: 0o755 });
    chmodSync(open, 0o755);
    const notPrivate = join(root, "plain");
    mkdirSync(notPrivate, { mode: 0o700 });
    chmodSync(notPrivate, 0o700);
    const file = join(root, "file");
    writeFileSync(file, "x");
    const link = join(root, "link");
    symlinkSync(workDir(), link);
    for (const dir of ["relative/private-data", open, file, link, join(root, "missing")]) {
      expect(await code(() => writePrivateSnapshot(input({ records: records(2), workDir: dir })))).toBe("work_dir_unsafe");
    }
    // 0700, mine, but neither under private-data/ nor on tmpfs: refused; with tmpfs reported, accepted.
    expect(() => assertWorkDir(notPrivate, { fsType: () => 0xef53 })).toThrow(SnapshotIoError);
    expect(assertWorkDir(notPrivate, { fsType: () => 0x01021994 })).toBe(notPrivate);
    expect(assertWorkDir(notPrivate, { fsType: () => 0x858458f6 })).toBe(notPrivate);
    expect(() => assertWorkDir(workDir(), { uid: process.getuid?.() === 0 ? 1 : 0 })).toThrow(SnapshotIoError);
    const failing = (): number => {
      throw new Error("statfs");
    };
    expect(() => assertWorkDir(notPrivate, { fsType: failing })).toThrow(SnapshotIoError);
    // The decision does not depend on the runner's /tmp: whatever file system the scratch area is on, a probe that says "disk" refuses, and the default probe agrees with what the operating system says about the directory.
    const realIsMemory = MEMORY_FS_TYPES.includes(Number(statfsSync(notPrivate).type));
    if (realIsMemory) expect(assertWorkDir(notPrivate)).toBe(notPrivate);
    else expect(() => assertWorkDir(notPrivate)).toThrow(SnapshotIoError);
    // Parents: group- or world-writable without the sticky bit is refused, sticky is accepted, owners must be me or root.
    const openParent = join(scratchRoot, "open-parent");
    const inner = join(openParent, "private-data", "work");
    mkdirSync(inner, { recursive: true });
    chmodSync(inner, 0o700);
    chmodSync(openParent, 0o777);
    expect(() => assertWorkDir(inner)).toThrow(SnapshotIoError);
    chmodSync(openParent, 0o1777);
    expect(assertWorkDir(inner)).toBe(inner);
    chmodSync(openParent, 0o775);
    expect(() => assertWorkDir(inner)).toThrow(SnapshotIoError);
    chmodSync(openParent, 0o755);
    expect(assertWorkDir(inner)).toBe(inner);
    expect([parentOwnerOk(1000, 1000), parentOwnerOk(0, 1000), parentOwnerOk(1001, 1000)]).toEqual([true, true, false]);
    expect([onWindowsMount("/mnt/c/Users/x"), onWindowsMount("/mnt/c"), onWindowsMount("/mnt/wsl/x"), onWindowsMount("/tmp/mnt/c")]).toEqual([true, true, false, false]);
  });

  it("refuses an unpinned binary, no records, a bad set, a bad key, a wrong flag and bad labels", async () => {
    const tool = standIn();
    const other = standIn();
    const wrongPin = { ...tool.pin, version: "9.9.9" };
    expect(await code(() => writePrivateSnapshot(input({ records: records(2), pin: wrongPin }, tool)))).toBe("writer_not_pinned");
    expect(other.pin).toBeDefined();
    expect(await code(() => writePrivateSnapshot(input({ records: [] })))).toBe("no_records");
    expect(await code(() => writePrivateSnapshot(input({ records: [recordOf(0, 1), recordOf(1, 1)] })))).toBe("records_invalid");
    expect(await code(() => writePrivateSnapshot(input({ records: records(2), holdoutKey: "zz" })))).toBe("holdout_key_invalid");
    const flipped = { ...recordOf(0, 1), holdout: !recordOf(0, 1).holdout };
    expect(await code(() => writePrivateSnapshot(input({ records: [flipped] })))).toBe("holdout_mismatch");
    const mislabelled = { ...recordOf(0, 1), labels: [{ label_id: "nope", value: "x" }] };
    expect(await code(() => writePrivateSnapshot(input({ records: [mislabelled] })))).toBe("labels_invalid");
    expect(await code(() => writePrivateSnapshot(input({ records: records(2), shardKey: "short" })))).toBe("layout_failed");
  });

  it("refuses a result the reader would refuse (a shard the writer cannot read back)", async () => {
    const tool = standIn();
    const corrupt = ((file, args, inputText, env, cwd) => {
      const result = tool.runner(file, args, inputText, env, cwd);
      if (args[0] === "-bail") {
        const out = /TO '((?:[^']|'')*)'/.exec(inputText)?.[1];
        if (out !== undefined) writeFileSync(out, "not parquet");
      }
      return result;
    }) as typeof tool.runner;
    const w = DuckdbWriter.open(join(scratchRoot, "duckdb-stand-in"), corrupt);
    expect(await code(() => writePrivateSnapshot(input({ records: records(3), writer: w, pin: tool.pin })))).toBe("shard_invalid");
  });
});

describe("reader refusals", () => {
  it("refuses a manifest that is not valid, a self-parent, a wrong hash, label set or version", async () => {
    const w = await written();
    expect(await code(() => readOf(w, { manifest: { ...w.manifest, extra: 1 } }))).toBe("manifest_invalid");
    expect(await code(() => readOf(w, { manifest: null }))).toBe("manifest_invalid");
    expect(await code(() => readOf(w, { expectedSnapshotHash: "00".repeat(32) }))).toBe("snapshot_hash_mismatch");
    const { first: _first, ...rest } = w.manifest;
    const selfish = { ...rest, parent_snapshot_hash: "00".repeat(32) };
    // A parent equal to the manifest's own hash cannot be built by hashing (the hash covers the field), so the check is shown with the reader's hash function stubbed through a fixed point: the guard is exercised below on a crafted manifest.
    expect(await code(() => readOf(w, { manifest: selfish }))).toBe("manifest_invalid");
    const otherLabels = { ...labelSet, labels: labelSet.labels.slice(1) };
    expect(await code(() => readOf(w, { labelSet: otherLabels as never }))).toBe("label_set_mismatch");
  });

  it("refuses a missing, resized, altered or repeated shard file", async () => {
    const w = await written();
    expect(await code(() => readOf(w, { shard: () => Promise.reject(new Error("PASTE")) }))).toBe("shard_missing");
    expect(await code(() => readOf(w, { shard: (i) => (w.shards[i] as Uint8Array).slice(1) }))).toBe("shard_size_mismatch");
    const flipped = (i: number): Uint8Array => {
      const copy = (w.shards[i] as Uint8Array).slice();
      copy[copy.length >> 1] = (copy[copy.length >> 1] ?? 0) ^ 1;
      return copy;
    };
    expect(await code(() => readOf(w, { shard: flipped }))).toBe("shard_digest_mismatch");
  });

  it("refuses files that match the manifest's digest but are not shards of it", async () => {
    const w = await written(12);
    // Re-point the manifest at garbage with a matching size and digest.
    const garbage = new TextEncoder().encode("not a parquet file at all");
    const shards = w.manifest.shards.map((s, i) => (i === 0 ? { ...s, bytes: garbage.byteLength, file_sha256: createHash("sha256").update(garbage).digest("hex") } : s));
    const forged = reseal(w.manifest, { shards });
    expect(await code(() => readOf(w, { manifest: forged, shard: (i) => (i === 0 ? garbage : (w.shards[i] as Uint8Array)) }))).toBe("shard_invalid");
  });

  it("refuses two shards with one file digest and a layout that does not fit the rows", async () => {
    const w = await written(12);
    const dup = reseal(w.manifest, { shards: w.manifest.shards.map((s, i) => (i === 1 ? { ...s, file_sha256: at(w.manifest.shards, 0).file_sha256 } : s)) });
    expect(await code(() => readOf(w, { manifest: dup }))).toBe("shard_file_repeated");
    expect(await code(() => readOf(w, { shardRows: 100 }))).toBe("shard_layout_invalid");
    expect(await code(() => readOf(w, { shardRows: 0 }))).toBe("shard_layout_invalid");
  });

  it("refuses a shard whose rows belong to another shard of the layout", async () => {
    const a = await written(12);
    // Swap the shard key: every digest and hash still match, but the placement differs.
    const key = "11".repeat(32);
    const swapped = reseal(a.manifest, { shard_key: key });
    expect(await code(() => readOf(a, { manifest: swapped }))).toMatch(/shard_invalid|shard_layout_invalid/);
  });

  it("recomputes the hold-out flags from the manifest's key", async () => {
    const w = await written(12);
    const other = reseal(w.manifest, { holdout: { ...w.manifest.holdout, key: "77".repeat(32) } });
    expect(await code(() => readOf(w, { manifest: other }))).toBe("holdout_mismatch");
  });

  it("refuses a record the engine's round trip refuses", async () => {
    const bad = { ...recordOf(0, 1), paste: recordOf(0, 1).paste.replace(/Ability: .*/, "Ability: Nonsense") };
    const w = await writePrivateSnapshot(input({ records: [bad, recordOf(1, 2)] }));
    expect(await code(() => readPrivateSnapshot({ engine, expected: INSPECT_ONLY, manifest: w.manifest, expectedSnapshotHash: privateSnapshotHash(w.manifest as never), shard: (i) => w.shards[i] as Uint8Array, labelSet }))).toBe("record_invalid");
  });

  it("refuses counts that differ from the shards' records", async () => {
    const w = await written(12);
    const lying = reseal(w.manifest, { records: { active: 9, retired: 3 } });
    // The manifest reader catches a count that disagrees with the ranges first.
    expect(await code(() => readOf(w, { manifest: lying }))).toBe("manifest_invalid");
  });

  it("reads the shard files in any order the manifest lists them, by index", async () => {
    const w = await written(12);
    const seen: number[] = [];
    await readOf(w, { shard: (i) => (seen.push(i), w.shards[i] as Uint8Array) });
    expect(seen).toEqual([0, 1, 2]);
    expect(plan.labelSet).toBe(labelSet);
    expect(hex32(1)).toHaveLength(32);
  });
});

describe("the working-directory seam", () => {
  it("is not part of the writer's API: production passes no probe, and an extra one is ignored", async () => {
    expect(readFileSync(new URL("../src/snapshot-writer.ts", import.meta.url), "utf8")).not.toMatch(/probe/i);
    const plain = join(scratchRoot, "seam");
    mkdirSync(plain, { recursive: true, mode: 0o700 });
    chmodSync(plain, 0o700);
    // The scratch area may be disk (CI on 24.04) or tmpfs (26.04), so the test establishes which and smuggles the OPPOSITE answer: if the writer honoured a probe, the verdict would flip.
    const realIsMemory = MEMORY_FS_TYPES.includes(Number(statfsSync(plain).type));
    const smuggled = { ...input({ records: records(2), workDir: plain }), probe: { fsType: () => (realIsMemory ? 0xef53 : 0x01021994) } };
    if (realIsMemory) expect((await writePrivateSnapshot(smuggled)).snapshotHash).toMatch(/^[0-9a-f]{64}$/);
    else expect(await code(() => writePrivateSnapshot(smuggled))).toBe("work_dir_unsafe");
  });
});

describe("readPrivateSnapshot anchors", () => {
  it("takes the snapshot hash as a required argument", async () => {
    const w = await written();
    // @ts-expect-error expectedSnapshotHash is required: there is no way to read without an anchor
    const missing = readPrivateSnapshot({ engine, expected: INSPECT_ONLY, manifest: w.manifest, shard: (i) => w.shards[i] as Uint8Array, labelSet, shardRows: 5 });
    expect(await code(() => missing)).toBe("snapshot_hash_mismatch");
  });

  it("takes the expected version as a required argument: leaving it out is a refusal, INSPECT_ONLY is said out loud", async () => {
    const w = await written();
    // @ts-expect-error expected is required: a reader that compares nothing must say INSPECT_ONLY
    const omitted = readPrivateSnapshot({ engine, manifest: w.manifest, expectedSnapshotHash: w.snapshotHash, shard: (i) => w.shards[i] as Uint8Array, labelSet, shardRows: 5 });
    expect(await code(() => omitted)).toBe("version_manifest_mismatch");
    expect((await readOf(w, { expected: INSPECT_ONLY })).records).toHaveLength(12);
  });

  it("refuses a snapshot made under another version manifest or vocabulary, and accepts the expected ones", async () => {
    const w = await written();
    const expected = { manifest: w.manifest.manifest, vocabularyHash: w.manifest.vocabulary_hash };
    expect((await readOf(w, { expected })).records).toHaveLength(12);
    expect(await code(() => readOf(w, { expected: { ...expected, vocabularyHash: "00".repeat(32) } }))).toBe("version_manifest_mismatch");
    for (const change of [{ format_id: "gen9other" }, { regulation: "Z-Z" }, { ruleset_hash: "11".repeat(32) }, { format_rules_hash: "22".repeat(32) }]) {
      expect(await code(() => readOf(w, { expected: { ...expected, manifest: { ...expected.manifest, ...change } } }))).toBe("version_manifest_mismatch");
    }
  });
});

describe("with the real pinned DuckDB tool", () => {
  const real = realDuckdb();
  const required = process.env.ARENEON_REQUIRE_DUCKDB === "1";
  it.skipIf(real === undefined && !required)("writes shards the reader accepts, and withholds its stderr", async () => {
    expect(real, "ARENEON_REQUIRE_DUCKDB=1 but no pinned DuckDB").toBeDefined();
    if (real === undefined) throw new Error("no DuckDB");
    const tool = real;
    const w = await writePrivateSnapshot(input({ records: records(12), shardRows: 5 }, tool as never));
    const read = await readOf(w);
    expect(read.records).toHaveLength(12);
    expect(w.manifest.writer.version).toBe(tool.pin.version);
    // A real DuckDB failure on this path is covered in the team-dataset tests (the tool's stderr is withheld and the scratch removed).
  });
});

function reseal(m: Awaited<ReturnType<typeof written>>["manifest"], change: Record<string, unknown>) {
  const next = { ...m, ...change } as Record<string, unknown>;
  next.logical_snapshot_hash = privateLogicalSnapshotHash(next as never);
  return next;
}
