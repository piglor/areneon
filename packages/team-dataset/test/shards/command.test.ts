import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { runShardsCli, type ShardsIo } from "../../src/shards/command.ts";
import { CONFIG_DIR, DATASET_DIR } from "./paths.ts";
import { SHARD_KEY, standInRunner } from "./support.ts";

const scratch = mkdtempSync(join(tmpdir(), "shards-cli-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));
const tool = join(scratch, "duckdb");
writeFileSync(tool, "stand-in\n");
const labels = join(CONFIG_DIR, "labels.json");

function run(args: string[], env: Record<string, string | undefined> = {}) {
  const out: string[] = [];
  const err: string[] = [];
  const io: ShardsIo = { out: (l) => out.push(l), err: (l) => err.push(l), env, runner: standInRunner() };
  return runShardsCli(args, io).then((code) => ({ code, out, err }));
}
const writeArgs = (outDir: string, more: string[] = []) => ["write", "--dataset", DATASET_DIR, "--labels", labels, "--out", outDir, "--shard-key", SHARD_KEY, "--snapshot-id", "cli-test", "--duckdb", tool, "--splits", "dev,validation", "--shard-rows", "50", ...more];

describe("the shard command line", () => {
  it("writes a snapshot, then verifies it", async () => {
    const outDir = join(scratch, "ok");
    const written = await run(writeArgs(outDir, ["--row-group-size", "40", "--parent", "a".repeat(64)]));
    expect(written.err).toEqual([]);
    expect(written.code).toBe(0);
    const summary = JSON.parse(written.out[0] as string) as { snapshot_file: string; shards: number; logical_snapshot_hash: string; writer: { version: string } };
    expect(summary).toMatchObject({ shards: 6, writer: { name: "duckdb", version: "1.5.6" } });
    const snapshot = JSON.parse(readFileSync(summary.snapshot_file, "utf8")) as { parent_snapshot_hash: string; logical_snapshot_hash: string };
    expect(snapshot.parent_snapshot_hash).toBe("a".repeat(64));
    expect(snapshot.logical_snapshot_hash).toBe(summary.logical_snapshot_hash);
    const verified = await run(["verify", "--snapshot", summary.snapshot_file, "--shards", outDir, "--dataset", DATASET_DIR, "--labels", labels, "--shard-rows", "50"]);
    expect(verified).toEqual({ code: 0, out: [JSON.stringify({ verified: { dev: 120, validation: 120 } })], err: [] });
  });

  it("converts every split by default and takes the shard size from 2^18", async () => {
    const noSplits = writeArgs(join(scratch, "all")).filter((a, k, all) => a !== "--splits" && all[k - 1] !== "--splits");
    const result = await run(noSplits);
    expect(result.code).toBe(1);
    expect(result.err[0]).toMatch(/no records file for the training split/);
    const noRows = writeArgs(join(scratch, "rows")).filter((a, k, all) => a !== "--shard-rows" && all[k - 1] !== "--shard-rows");
    const written = await run(noRows);
    expect(written.code).toBe(0);
    expect(JSON.parse(written.out[0] as string)).toMatchObject({ shards: 2 });
  });

  it("finds the tool through DUCKDB_BIN and refuses to run without one", async () => {
    const args = writeArgs(join(scratch, "env")).filter((a, k, all) => a !== "--duckdb" && all[k - 1] !== "--duckdb");
    expect((await run(args, { DUCKDB_BIN: tool })).code).toBe(0);
    const none = await run(args, { PATH: "" });
    expect(none.code).toBe(2);
    expect(none.err[0]).toMatch(/usage: no DuckDB command-line tool/);
  });

  it("refuses an unpinned tool when a pinned one is required", async () => {
    const result = await run(writeArgs(join(scratch, "pinned"), ["--require-pinned"]));
    expect(result.code).toBe(1);
    expect(result.err[0]).toMatch(/refused: .*(not the pinned|no binary for)/);
  });

  it("refuses shards that are not the dataset's", async () => {
    const outDir = join(scratch, "bad");
    const written = await run(writeArgs(outDir));
    const file = (JSON.parse(written.out[0] as string) as { snapshot_file: string }).snapshot_file;
    const result = await run(["verify", "--snapshot", file, "--shards", outDir, "--dataset", DATASET_DIR, "--labels", labels]);
    expect(result.code).toBe(1);
    expect(result.err[0]).toMatch(/refused: dev: 120 rows need 1 shards/);
  });

  it("reports usage errors with exit code 2", async () => {
    for (const [args, message] of [
      [[], /usage: <write\|verify>/],
      [["frobnicate"], /usage: <write\|verify>/],
      [["write", "extra"], /usage: <write\|verify>/],
      [["write", "--duckdb", tool], /--dataset is required/],
      [["write", "--nope"], /error: .*nope/],
      [writeArgs(join(scratch, "u1"), ["--splits", "dev,dev"]), /--splits is a comma-separated list/],
      [writeArgs(join(scratch, "u2"), ["--splits", "test"]), /--splits is a comma-separated list/],
      [writeArgs(join(scratch, "u3"), ["--shard-rows", "0"]), /--shard-rows must be a positive whole number/],
      [writeArgs(join(scratch, "u4"), ["--row-group-size", "x"]), /--row-group-size must be a positive whole number/],
      [["verify", "--snapshot", join(scratch, "none.json"), "--shards", scratch, "--dataset", DATASET_DIR, "--labels", labels], /error: ENOENT/],
    ] as const) {
      const result = await run([...args]);
      expect(result.code, args.join(" ")).toBe(2);
      expect(result.err[0]).toMatch(message);
    }
  });
});
