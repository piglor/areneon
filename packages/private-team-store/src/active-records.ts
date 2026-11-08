// The one read of a pinned snapshot's verified local copy (`private-data/snapshots/<hash>/`)
// through the snapshot check (`readPrivateSnapshot`), returning its ACTIVE records. The private
// test's source and the loop's private pool both use it; each maps the two failures to its own
// fixed codes. A failure carries a code and nothing of a lower layer.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { type Engine, privateDataRoot, type PrivateTeamRecord, resolveFormat, teamVocabularyHash } from "@areneon/showdown-bridge";
import { loadFixtureInputs } from "@areneon/team-dataset";
import { readPrivateSnapshot } from "./snapshot-reader.ts";

export class ActiveRecordsError extends Error {
  readonly code: "source_unreadable" | "snapshot_unverified";
  constructor(code: "source_unreadable" | "snapshot_unverified") {
    super(code);
    this.name = "ActiveRecordsError";
    this.code = code;
  }
}

/** The active records of the snapshot `snapshotHash`, from its local copy under `snapshotsDir` (a `private-data` directory). */
export async function readActiveRecords(args: { engine: Engine; formatId: string; snapshotsDir: string; snapshotHash: string }): Promise<PrivateTeamRecord[]> {
  const { engine, formatId, snapshotsDir, snapshotHash } = args;
  if (!/^[0-9a-f]{64}$/.test(snapshotHash)) throw new ActiveRecordsError("source_unreadable");
  let dir: string;
  let manifest: unknown;
  try {
    privateDataRoot(snapshotsDir);
    dir = join(snapshotsDir, snapshotHash);
    manifest = JSON.parse(readFileSync(join(dir, "manifest.json"), "utf8"));
  } catch {
    throw new ActiveRecordsError("source_unreadable");
  }
  try {
    const inputs = loadFixtureInputs(engine, formatId);
    const read = await readPrivateSnapshot({
      engine,
      manifest,
      // A copy in a plain Uint8Array: the decode worker takes ownership of the buffer it is given.
      shard: (index) => new Uint8Array(readFileSync(join(dir, "shards", `${index}.parquet`))),
      labelSet: inputs.labelSet,
      expectedSnapshotHash: snapshotHash,
      expected: { manifest: resolveFormat(engine, formatId).manifest, vocabularyHash: teamVocabularyHash(inputs.vocabulary) },
    });
    return read.records.filter((record) => record.status === "active");
  } catch {
    throw new ActiveRecordsError("snapshot_unverified");
  }
}
