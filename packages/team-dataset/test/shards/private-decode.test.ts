import { chmodSync, existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { ShardError } from "../../src/shards/columns.ts";
import { answerDecode, decodeIsolated, decodePrivateParquet, UNREADABLE } from "../../src/shards/private-decode.ts";
import { privateColumnsFor, privateRowOf } from "../../src/shards/private-shards.ts";
import { removeTree } from "../../src/shards/scratch.ts";
import { fixtureRecords, labelSet, parquetOf, plan } from "./support.ts";

const scratch = mkdtempSync(join(tmpdir(), "private-decode-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));

const columns = privateColumnsFor(labelSet);
const rows = fixtureRecords("dev").slice(0, 6).map((t, k) =>
  privateRowOf(
    {
      schema_version: 1,
      data_class: "private",
      team_id: k.toString(16).padStart(32, "0"),
      revision: 1,
      supersedes_record_hash: null,
      team_hash: t.team_hash,
      canonical_team_hash: t.canonical_team_hash,
      paste: t.paste,
      status: "active",
      tags: [],
      holdout: false,
      labels: t.labels,
    } as never,
    plan.stratumOf(t.labels),
  ),
);
const file = parquetOf(rows, columns);
const spec = { columns, rows: rows.length };

describe("the isolated Parquet decode", () => {
  it("returns the rows, in a worker thread and in process alike", async () => {
    const isolated = await decodeIsolated(file, spec);
    expect(isolated).toHaveLength(6);
    expect(await decodePrivateParquet(file, spec)).toEqual(isolated);
    expect(isolated.map((r) => r.team_id)).toEqual(rows.map((r) => r.team_id));
  });

  it("gives a refusal's fixed message and no library text, whatever the worker met", async () => {
    await expect(decodeIsolated(new TextEncoder().encode("PAR1 nonsense CanaryBytes7"), spec)).rejects.toThrow(UNREADABLE);
    await expect(decodeIsolated(file, { ...spec, rows: 5 })).rejects.toThrow(/row count/);
    const answer = await answerDecode(new Uint8Array(0), spec);
    expect(answer).toEqual({ ok: false, message: UNREADABLE });
  });

  it("gives up after the time limit and does not keep the worker", async () => {
    const started = Date.now();
    const error = await decodeIsolated(file, spec, 1).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ShardError);
    expect((error as Error).message).toBe(UNREADABLE);
    expect(Date.now() - started).toBeLessThan(5000);
  });
});

describe("the footer checks (run in process: the worker is not instrumented)", () => {
  const refusal = async (bytes: Uint8Array, over: Partial<typeof spec> = {}): Promise<string> => {
    try {
      await decodePrivateParquet(bytes, { ...spec, ...over });
    } catch (error) {
      expect(error).toBeInstanceOf(ShardError);
      return (error as Error).message;
    }
    return "read";
  };
  it("accepts the layout and refuses every departure from it with a fixed message", async () => {
    expect(await refusal(file)).toBe("read");
    expect(await refusal(parquetOf(rows, columns, { rowGroupSize: 2 }))).toBe("read");
    expect(await refusal(new TextEncoder().encode("PAR1 junk"))).toBe(UNREADABLE);
    expect(await refusal(parquetOf(rows, columns, { noVersion: true }))).toMatch(/schema version/);
    expect(await refusal(parquetOf(rows, columns, { version: "2" }))).toMatch(/schema version/);
    expect(await refusal(parquetOf(rows, columns, { without: ["paste"] }))).toBe("the file's columns are not the private layout's");
    expect(await refusal(parquetOf(rows, columns, { extra: [{ name: "extra", data: rows.map(() => "x") }] }))).toBe("the file's columns are not the private layout's");
    expect(await refusal(parquetOf(rows, columns, { labelExtra: true }))).toBe("column labels does not have the layout's type");
    expect(await refusal(parquetOf(rows.map((r) => ({ ...r, holdout: "true" })), columns.map((c) => (c.name === "holdout" ? { ...c, type: "VARCHAR" as const } : c))))).toBe("column holdout does not have the layout's type");
    expect(await refusal(parquetOf(rows.map((r) => ({ ...r, tags: "a" })), columns.map((c) => (c.name === "tags" ? { ...c, type: "VARCHAR" as const } : c))))).toBe("column tags does not have the layout's type");
    expect(await refusal(parquetOf(rows, columns, { indexAsDouble: true }), {})).toBe("read");
    expect(await refusal(file, { rows: 5 })).toBe("the file's row count is not the manifest's");
    // A footer that is fine over pages that are not.
    const damaged = file.slice();
    damaged.fill(0xff, 4, 60);
    expect(await refusal(damaged)).toBe(UNREADABLE);
  });
});

describe("removeTree", () => {
  it("removes a tree, also one it first has to reopen, and reports failure without a reason", () => {
    const dir = join(scratch, "tree");
    mkdirSync(join(dir, "locked", "inner"), { recursive: true });
    writeFileSync(join(dir, "locked", "inner", "file"), "x");
    writeFileSync(join(dir, "locked", "file"), "x");
    chmodSync(join(dir, "locked", "file"), 0o400);
    chmodSync(join(dir, "locked", "inner"), 0o500);
    chmodSync(join(dir, "locked"), 0o500);
    expect(removeTree(dir)).toBe(true);
    expect(existsSync(dir)).toBe(false);
    expect(removeTree(join(scratch, "never-there"))).toBe(true);
    let calls = 0;
    const stubborn = (): never => {
      calls += 1;
      throw new Error("EBUSY: secret path");
    };
    expect(removeTree(join(scratch, "x"), stubborn)).toBe(false);
    expect(calls).toBe(2);
    let first = true;
    expect(
      removeTree(join(scratch, "y"), () => {
        if (first) {
          first = false;
          throw new Error("EACCES");
        }
      }),
    ).toBe(true);
  });
});
