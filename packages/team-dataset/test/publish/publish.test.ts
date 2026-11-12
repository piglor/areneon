import { cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { InMemoryObjectStore, type ObjectStore, ObjectStoreError } from "@areneon/object-store";
import { jcsBytes, snapshotHash } from "@areneon/showdown-bridge";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { convertDataset } from "../../src/shards/convert.ts";
import { DuckdbWriter } from "../../src/shards/duckdb.ts";
import { runPublishCli } from "../../src/publish/command.ts";
import { snapshotKey, TEAMS_LAYOUT } from "../../src/publish/layout.ts";
import {
  assertPlatformsAgree,
  CHECKED_RANGES,
  checkPlan,
  comparePlatforms,
  parseReport,
  platformReport,
} from "../../src/publish/platform-check.ts";
import { buildPlan, parsePlan, PublishError, readPlan, readSnapshotDir, releaseFile, verifyDir, writePlan } from "../../src/publish/plan.ts";
import { uploadSnapshot } from "../../src/publish/upload.ts";
import { DATASET_DIR } from "../shards/paths.ts";
import { labelSet, SHARD_KEY, standInRunner } from "../shards/support.ts";

// a stuck async test fails after 30 s instead of holding the runner (a synchronous loop cannot be interrupted, so the code under test must not have one)
vi.setConfig({ testTimeout: 30_000 });

const scratch = mkdtempSync(join(tmpdir(), "teams-publish-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));
const writerFile = join(scratch, "duckdb");
writeFileSync(writerFile, "stand-in\n");

let base: string;
let n = 0;
const copy = (): string => {
  const out = join(scratch, `c${n++}`);
  cpSync(base, out, { recursive: true });
  return out;
};

beforeAll(async () => {
  base = join(scratch, "base");
  await convertDataset({
    datasetDir: DATASET_DIR,
    outDir: base,
    labelSet,
    shardKey: SHARD_KEY,
    snapshotId: "publish-test",
    splits: ["dev", "validation"],
    writer: DuckdbWriter.open(writerFile, standInRunner("v1.5.6 (Variegata) 0123456789")),
    shardRows: 50,
  });
}, 120_000);

const HEX = "ab".repeat(32);

describe("plan", () => {
  it("plans a snapshot directory: tag, hashes and every shard", () => {
    const plan = buildPlan(base);
    const manifest = readSnapshotDir(base);
    expect(plan.snapshot_hash).toBe(snapshotHash(manifest));
    expect(plan.tag).toBe(`teams-${plan.snapshot_hash}`);
    expect(plan.shards.map((s) => s.object_key)).toEqual(manifest.shards.map((s) => s.object_key));
  });

  it("writes the plan and the Release asset (the manifest's canonical bytes), and verifies them", () => {
    const dir = copy();
    const plan = writePlan(dir);
    expect(readFileSync(releaseFile(dir, plan))).toEqual(jcsBytes(readSnapshotDir(dir)));
    expect(readPlan(dir)).toEqual(plan);
    expect(verifyDir(dir)).toEqual(plan);
  });

  it("refuses a missing, altered or short shard, a missing manifest and a bad manifest", () => {
    const manifest = readSnapshotDir(base);
    const key = manifest.shards[0]?.object_key as string;
    const missing = copy();
    rmSync(join(missing, key));
    expect(() => buildPlan(missing)).toThrow(/missing/);
    const altered = copy();
    writeFileSync(join(altered, key), "x");
    expect(() => buildPlan(altered)).toThrow(/does not match its SHA-256/);
    const noManifest = join(scratch, "empty");
    mkdirSync(noManifest);
    expect(() => buildPlan(noManifest)).toThrow(/no snapshot.json/);
    const bad = copy();
    writeFileSync(join(bad, "snapshot.json"), "{}");
    expect(() => buildPlan(bad)).toThrow(/not a valid snapshot manifest/);
  });

  it("refuses a size the manifest does not state", () => {
    const dir = copy();
    const m = JSON.parse(readFileSync(join(dir, "snapshot.json"), "utf8")) as { shards: { bytes: number }[] };
    (m.shards[0] as { bytes: number }).bytes += 1;
    writeFileSync(join(dir, "snapshot.json"), JSON.stringify(m));
    expect(() => buildPlan(dir)).toThrow(/size|valid snapshot manifest/);
  });

  it("refuses an odd shard key", () => {
    const dir = copy();
    const m = JSON.parse(readFileSync(join(dir, "snapshot.json"), "utf8")) as { shards: { object_key: string }[] };
    (m.shards[0] as { object_key: string }).object_key = "../x";
    writeFileSync(join(dir, "snapshot.json"), JSON.stringify(m));
    expect(() => buildPlan(dir)).toThrow(PublishError);
  });

  it("verify refuses a plan that differs from the files now, and a directory with no plan", () => {
    const dir = copy();
    expect(() => verifyDir(dir)).toThrow(/run plan first/);
    const plan = writePlan(dir);
    const forged = { ...plan, logical_snapshot_hash: HEX };
    writeFileSync(join(dir, "plan.json"), JSON.stringify(forged));
    expect(() => verifyDir(dir)).toThrow(/differs/);
  });

  it("parsePlan fails closed on every malformed shape", () => {
    const plan = buildPlan(base);
    const s0 = plan.shards[0] as { object_key: string; sha256: string; bytes: number };
    const cases: unknown[] = [
      "not json",
      [],
      { ...plan, extra: 1 },
      { ...plan, schema_version: 2 },
      { ...plan, snapshot_hash: "x" },
      { ...plan, tag: "teams-x" },
      { ...plan, manifest_sha256: HEX },
      { ...plan, logical_snapshot_hash: "z" },
      { ...plan, shards: "x" },
      { ...plan, shards: [{ ...s0, extra: 1 }] },
      { ...plan, shards: [{ ...s0, object_key: "pub/v1/shards/zz" }] },
      { ...plan, shards: [{ ...s0, sha256: HEX }] },
      { ...plan, shards: [{ ...s0, bytes: 0 }] },
      { ...plan, shards: [s0, s0] },
    ];
    for (const c of cases) expect(() => parsePlan(typeof c === "string" ? c : JSON.stringify(c))).toThrow(PublishError);
    expect(parsePlan(JSON.stringify(plan))).toEqual(plan);
  });
});

describe("upload", () => {
  it("puts every shard then the manifest, and a retry finds them all present", async () => {
    const plan = buildPlan(base);
    const store = new InMemoryObjectStore(TEAMS_LAYOUT);
    const first = await uploadSnapshot(store, base, plan);
    expect(first.uploaded).toHaveLength(plan.shards.length + 1);
    expect(store.writes.at(-1)).toBe(snapshotKey(plan.snapshot_hash));
    expect(store.keys()).toHaveLength(plan.shards.length + 1);
    expect(Buffer.from((await store.get(snapshotKey(plan.snapshot_hash))) as Uint8Array)).toEqual(jcsBytes(readSnapshotDir(base)));
    const second = await uploadSnapshot(store, base, plan);
    expect(second).toEqual({ uploaded: [], present: first.uploaded });
  });

  it("refuses other bytes under a key, and passes other store errors on", async () => {
    const plan = buildPlan(base);
    const store = new InMemoryObjectStore(TEAMS_LAYOUT);
    store.seed((plan.shards[0] as { object_key: string }).object_key, Buffer.from("other"));
    await expect(uploadSnapshot(store, base, plan)).rejects.toThrow(/other bytes/);
    const denied = { putIfAbsent: () => Promise.reject(new ObjectStoreError("denied")) } as unknown as ObjectStore;
    await expect(uploadSnapshot(denied, base, plan)).rejects.toThrow(ObjectStoreError);
    const vanished = {
      putIfAbsent: () => Promise.reject(new ObjectStoreError("exists")),
      get: () => Promise.resolve(null),
    } as unknown as ObjectStore;
    await expect(uploadSnapshot(vanished, base, plan)).rejects.toThrow(/other bytes/);
  });

  it("the layout admits the two key kinds only and has no overwrite", async () => {
    const store = new InMemoryObjectStore(TEAMS_LAYOUT);
    await expect(store.putIfAbsent("pub/v1/other", Buffer.from("x"))).rejects.toThrow(ObjectStoreError);
    await expect(store.overwriteCurrent(Buffer.from("x"))).rejects.toThrow(ObjectStoreError);
    expect(await store.list("pub/v1/shards/")).toEqual([]);
    expect(TEAMS_LAYOUT.allowsListPrefix("pub/")).toBe(false);
  });
});

describe("two-platform check", () => {
  it("compares the whole pipeline up to the size bound", () => {
    expect(checkPlan(1000, 1000, HEX)).toEqual({ mode: "full" });
  });

  it("above it picks 16 distinct HMAC-chosen ranges, deterministic in the key", () => {
    const total = 2 ** 20 * 100 + 5;
    const a = checkPlan(total, 1000, HEX);
    expect(a).toEqual(checkPlan(total, 1000, HEX));
    expect(a.mode === "ranges" && a.ranges).toHaveLength(CHECKED_RANGES);
    if (a.mode !== "ranges") throw new Error("ranges expected");
    expect(new Set(a.ranges.map((r) => r.first)).size).toBe(CHECKED_RANGES);
    for (const r of a.ranges) expect(r.first % 2 ** 20).toBe(0);
    const b = checkPlan(total, 1000, "cd".repeat(32));
    expect(JSON.stringify(b)).not.toBe(JSON.stringify(a));
  });

  it("checks every range when there are few, the last one short", () => {
    const plan = checkPlan(10, 5, HEX, 4);
    expect(plan).toEqual({ mode: "ranges", ranges: [{ first: 0, count: 4 }, { first: 4, count: 4 }, { first: 8, count: 2 }] });
  });

  it("refuses bad arguments", () => {
    expect(() => checkPlan(0, 1, HEX)).toThrow(PublishError);
    expect(() => checkPlan(1, 0, HEX)).toThrow(PublishError);
    expect(() => checkPlan(1, 1, "x")).toThrow(PublishError);
    expect(() => checkPlan(1, 1, HEX, 0)).toThrow(PublishError);
  });

  it("reports and compares platforms", () => {
    const a = platformReport(base, "linux-amd64", [{ first: 4, count: 2, digest: HEX }, { first: 0, count: 2, digest: HEX }]);
    const b = platformReport(base, "linux-arm64", [{ first: 0, count: 2, digest: HEX }, { first: 4, count: 2, digest: HEX }]);
    expect(a.ranges.map((r) => r.first)).toEqual([0, 4]);
    expect(comparePlatforms(a, b)).toEqual([]);
    expect(() => assertPlatformsAgree(a, b)).not.toThrow();
    expect(parseReport(JSON.stringify(a))).toEqual(a);
    const bad = [
      comparePlatforms(a, { ...a }),
      comparePlatforms(a, { ...b, logical_snapshot_hash: "0".repeat(64) }),
      comparePlatforms(a, { ...b, shard_logical_hashes: [] }),
      comparePlatforms(a, { ...b, ranges: [] }),
      comparePlatforms(a, { ...b, ranges: [{ first: 1, count: 2, digest: HEX }, { first: 4, count: 2, digest: HEX }] }),
      comparePlatforms(a, { ...b, ranges: [{ first: 0, count: 2, digest: "0".repeat(64) }, { first: 4, count: 2, digest: HEX }] }),
    ];
    for (const problems of bad) expect(problems.length).toBeGreaterThan(0);
    expect(() => assertPlatformsAgree(a, { ...b, ranges: [] })).toThrow(/two-platform check failed/);
    expect(() => platformReport(base, "Bad Name")).toThrow(PublishError);
  });

  it("parseReport fails closed", () => {
    const a = platformReport(base, "linux-amd64");
    for (const c of ["x", "[]", JSON.stringify({ ...a, extra: 1 }), JSON.stringify({ ...a, ranges: [{ first: 1 }] }), JSON.stringify({ ...a, platform: "A" })]) {
      expect(() => parseReport(c)).toThrow(PublishError);
    }
  });
});

describe("command", () => {
  const io = (env: Record<string, string | undefined> = {}, openStore?: Parameters<typeof runPublishCli>[1]["openStore"]) => {
    const out: string[] = [];
    const err: string[] = [];
    return { out, err, io: { out: (l: string) => out.push(l), err: (l: string) => err.push(l), env, openStore } };
  };

  it("plans, verifies, reports and compares", async () => {
    const dir = copy();
    const a = io();
    expect(await runPublishCli(["plan", "--snapshot", dir], a.io)).toBe(0);
    expect(await runPublishCli(["verify", "--snapshot", dir], a.io)).toBe(0);
    const ra = join(scratch, "ra.json");
    const rb = join(scratch, "rb.json");
    expect(await runPublishCli(["report", "--snapshot", dir, "--platform", "p-a", "--out", ra], a.io)).toBe(0);
    expect(await runPublishCli(["report", "--snapshot", dir, "--platform", "p-b", "--out", rb], a.io)).toBe(0);
    expect(await runPublishCli(["compare", ra, rb], a.io)).toBe(0);
    expect(await runPublishCli(["compare", ra, ra], a.io)).toBe(1);
    expect(await runPublishCli(["compare", ra], a.io)).toBe(2);
  });

  it("uploads through the injected store, and skips cleanly without credentials", async () => {
    const dir = copy();
    writePlan(dir);
    const store = new InMemoryObjectStore(TEAMS_LAYOUT);
    const env = {
      AWS_ACCESS_KEY_ID: "AKIA0123456789ABCDEF",
      AWS_SECRET_ACCESS_KEY: "s".repeat(40),
      AWS_ENDPOINT_URL: "https://0123456789abcdef.r2.cloudflarestorage.com",
      AWS_DEFAULT_REGION: "auto",
    };
    const ok = io(env, () => store);
    expect(await runPublishCli(["upload", "--snapshot", dir, "--bucket", "teams-test"], ok.io)).toBe(0);
    expect(store.keys().length).toBeGreaterThan(1);
    const skipped = io({});
    expect(await runPublishCli(["upload", "--snapshot", dir, "--bucket", "b1", "--skip-without-store"], skipped.io)).toBe(0);
    expect(skipped.out[0]).toMatch(/skipped/);
    const missing = io({});
    expect(await runPublishCli(["upload", "--snapshot", dir, "--bucket", "teams-test"], missing.io)).toBe(1);
    expect(await runPublishCli(["upload", "--snapshot", dir], ok.io)).toBe(2);
  });

  it("opens the R2 client when no store is injected (a refused request is exit 1)", async () => {
    const dir = copy();
    writePlan(dir);
    vi.stubGlobal("fetch", () => {
      return Promise.resolve(new Response("", { status: 503 }));
    });
    try {
      const a = io({ AWS_ACCESS_KEY_ID: "AKIA0123456789ABCDEF", AWS_SECRET_ACCESS_KEY: "s".repeat(40), AWS_ENDPOINT_URL: "https://0123456789abcdef.r2.cloudflarestorage.com", AWS_DEFAULT_REGION: "auto" });
      expect(await runPublishCli(["upload", "--snapshot", dir, "--bucket", "teams-test"], a.io)).toBe(1);
      expect(a.err.join("\n")).toContain("object store refused");
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("exits 2 on a usage error and 1 on a refusal, and rethrows a defect", async () => {
    const a = io();
    expect(await runPublishCli([], a.io)).toBe(2);
    expect(await runPublishCli(["plan", "--nope"], a.io)).toBe(2);
    expect(await runPublishCli(["plan"], a.io)).toBe(2);
    expect(await runPublishCli(["verify", "--snapshot", join(scratch, "nowhere")], a.io)).toBe(1);
    const broken = io({ AWS_ACCESS_KEY_ID: "x" }, () => {
      throw new TypeError("defect");
    });
    const dir = copy();
    writePlan(dir);
    await expect(
      runPublishCli(["upload", "--snapshot", dir, "--bucket", "teams-test"], { ...broken.io, env: { AWS_ACCESS_KEY_ID: "AKIA0123456789ABCDEF", AWS_SECRET_ACCESS_KEY: "s".repeat(40), AWS_ENDPOINT_URL: "https://0123456789abcdef.r2.cloudflarestorage.com", AWS_DEFAULT_REGION: "auto" } }),
    ).rejects.toThrow(TypeError);
  });
});
