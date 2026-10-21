import { readdirSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";

// The writer's own scratch directory (rows and shard files) that cannot be
// removed: a finished write fails with the fixed code, a write that already
// failed keeps its own error, and neither says anything about the contents.

const stuck = vi.hoisted(() => ({ on: false }));
vi.mock("node:fs", async (original) => {
  const real = await original<typeof import("node:fs")>();
  return {
    ...real,
    rmSync: (path: string, options?: Parameters<typeof real.rmSync>[1]) => {
      if (stuck.on && /snapshot-[A-Za-z0-9]{6}$/.test(String(path))) throw new Error(`EBUSY ${String(path)} CanaryPath7`);
      return real.rmSync(path, options);
    },
  };
});

const { SnapshotIoError } = await import("../src/snapshot-io-error.ts");
const { writePrivateSnapshot } = await import("../src/snapshot-writer.ts");
const { input, records, standIn, workDir } = await import("./snapshot-support.ts");

const outcome = async (run: () => Promise<unknown>): Promise<InstanceType<typeof SnapshotIoError> | "no error"> => {
  try {
    await run();
  } catch (error) {
    expect(error).toBeInstanceOf(SnapshotIoError);
    return error as InstanceType<typeof SnapshotIoError>;
  }
  return "no error";
};

describe("a writer scratch directory that will not go", () => {
  it("fails a finished write with cleanup_failed and no path in the message", async () => {
    const dir = workDir();
    stuck.on = true;
    const error = await outcome(() => writePrivateSnapshot(input({ records: records(3), workDir: dir })));
    stuck.on = false;
    expect(error).not.toBe("no error");
    expect((error as InstanceType<typeof SnapshotIoError>).code).toBe("cleanup_failed");
    expect(String((error as Error).message) + String((error as Error).stack)).not.toContain("CanaryPath7");
    // What the stuck removal left is the caller's to remove; a later write is unaffected.
    expect(readdirSync(dir).length).toBe(1);
  });

  it("keeps the original error when the write failed first", async () => {
    const tool = standIn();
    const dir = workDir();
    stuck.on = true;
    const error = await outcome(() => writePrivateSnapshot(input({ records: records(3), workDir: dir, shardKey: "ab".repeat(32), write: { rowGroupSize: 0 } }, tool)));
    stuck.on = false;
    expect(error).not.toBe("no error");
    expect((error as InstanceType<typeof SnapshotIoError>).code).toBe("writer_failed");
  });
});
