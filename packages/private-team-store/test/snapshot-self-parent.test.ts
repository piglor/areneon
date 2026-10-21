import { describe, expect, it, vi } from "vitest";

// A snapshot hash covers the manifest, parent field included, so a manifest
// cannot name its own hash as parent by hashing; the reader still refuses it.
// The hash function is fixed here to a value the manifest can name.
const FIXED = "00".repeat(32);
vi.mock("@areneon/showdown-bridge", async (original) => ({ ...(await original<typeof import("@areneon/showdown-bridge")>()), privateSnapshotHash: () => FIXED }));

const { privateSnapshotHash } = await import("@areneon/showdown-bridge");
const { INSPECT_ONLY, readPrivateSnapshot } = await import("../src/snapshot-reader.ts");
const { writePrivateSnapshot } = await import("../src/snapshot-writer.ts");
const { at, engine, input, labelSet, recordOf } = await import("./snapshot-support.ts");

describe("a parent equal to the snapshot's own hash", () => {
  it("is refused by the reader", async () => {
    const w = await writePrivateSnapshot(input({ records: [recordOf(0, 1)], parentSnapshotHash: FIXED }));
    const error = await readPrivateSnapshot({ engine, expected: INSPECT_ONLY, manifest: w.manifest, expectedSnapshotHash: privateSnapshotHash(w.manifest as never), shard: (i) => at(w.shards, i), labelSet }).catch((e: unknown) => e);
    expect((error as { code: string }).code).toBe("parent_is_self");
  });
});
