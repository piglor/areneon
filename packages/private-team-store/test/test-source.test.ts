// The snapshot team source of the private test: the verified local copy of a
// snapshot, `all` active records or the hold-out ones, as teams. Stand-in
// records built from the committed synthetic teams; nothing real.
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { canonicalJson, holdoutFlag, PrivateTestRefusal, readTrainingRunConfig, teamVocabularyHash } from "@areneon/showdown-bridge";
import { loadFixtureInputs } from "@areneon/team-dataset";
import { describe, expect, it, vi } from "vitest";
import { writeLocalCopy } from "../src/local-copy.ts";
import { snapshotTeams } from "../src/test-source.ts";
import { writePrivateSnapshot } from "../src/snapshot-writer.ts";
import { engine, FORMAT_ID, HOLDOUT_KEY, hex32, input, recordOf, records, ROOT, standIn, workDir } from "./snapshot-support.ts";

vi.setConfig({ testTimeout: 60_000 });

const tiny = JSON.parse(readFileSync(join(ROOT, "fixtures/contracts/v1/training-run-config/valid/tiny.json"), "utf8")) as Record<string, unknown>;
const config = readTrainingRunConfig(tiny);
const vocabularyHash = teamVocabularyHash(loadFixtureInputs(engine, FORMAT_ID).vocabulary);

async function localCopy(n = 12, noHoldout = false): Promise<{ dir: string; hash: string; held: number; active: number }> {
  // With `noHoldout`, team ids whose flag is false, so the hold-out subset is empty.
  const ids = Array.from({ length: 200 }, (_, k) => k + 1).filter((id) => !holdoutFlag(HOLDOUT_KEY, hex32(id))).slice(0, n);
  const all = noHoldout ? ids.map((id, k) => recordOf(k, id)) : records(n);
  const w = await writePrivateSnapshot(input({ records: all, vocabularyHash }, standIn()));
  const dir = join(dirname(workDir()), "snapshots");
  mkdirSync(dir, { recursive: true, mode: 0o700 });
  writeLocalCopy(dir, w.snapshotHash, new TextEncoder().encode(canonicalJson(w.manifest as never)), w.shards);
  return { dir, hash: w.snapshotHash, held: all.filter((r) => r.status === "active" && r.holdout).length, active: all.filter((r) => r.status === "active").length };
}

const code = async (run: () => Promise<unknown>): Promise<string> => {
  try {
    await run();
  } catch (error) {
    if (error instanceof PrivateTestRefusal) return error.message;
    throw error;
  }
  return "no error";
};

describe("snapshotTeams", () => {
  it("reads the active records, or only the hold-out ones, from the verified local copy", async () => {
    const copy = await localCopy();
    expect(copy.held).toBeGreaterThan(0);
    const all = await snapshotTeams({ engine, config, snapshotsDir: copy.dir, snapshotHash: copy.hash, subset: "all" });
    expect(all.teams).toHaveLength(copy.active);
    expect(all.source).toEqual({ kind: "snapshot", snapshot_hash: copy.hash });
    expect(all.subset).toBe("all");
    const held = await snapshotTeams({ engine, config, snapshotsDir: copy.dir, snapshotHash: copy.hash, subset: "holdout" });
    expect(held.teams).toHaveLength(copy.held);
    expect(held.subset).toBe("holdout");
  });

  it("refuses with a fixed code: a copy that is not there, one outside private-data, a wrong hash and an empty subset", async () => {
    const copy = await localCopy();
    const ask = (over: Partial<Parameters<typeof snapshotTeams>[0]>) => snapshotTeams({ engine, config, snapshotsDir: copy.dir, snapshotHash: copy.hash, subset: "all", ...over });
    expect(await code(() => ask({ snapshotHash: "cd".repeat(32) }))).toBe("private-test: source_unreadable");
    for (const bad of ["../escape", "AB".repeat(32), "ab".repeat(31), ""]) expect(await code(() => ask({ snapshotHash: bad }))).toBe("private-test: source_unreadable");
    const plain = mkdtempSync(join(tmpdir(), "plain-"));
    try {
      expect(await code(() => ask({ snapshotsDir: plain }))).toBe("private-test: source_unreadable");
    } finally {
      rmSync(plain, { recursive: true, force: true });
    }
    // A manifest that does not hash to its name does not verify.
    const forged = join(copy.dir, "ef".repeat(32));
    mkdirSync(forged, { mode: 0o700 });
    writeFileSync(join(forged, "manifest.json"), readFileSync(join(copy.dir, copy.hash, "manifest.json")));
    expect(await code(() => ask({ snapshotHash: "ef".repeat(32) }))).toBe("private-test: snapshot_unverified");
    // A snapshot without an active hold-out record leaves nothing to test in that subset.
    const none = await localCopy(3, true);
    expect(none.held).toBe(0);
    expect(await code(() => snapshotTeams({ engine, config, snapshotsDir: none.dir, snapshotHash: none.hash, subset: "holdout" }))).toBe("private-test: source_empty");
  });
});
