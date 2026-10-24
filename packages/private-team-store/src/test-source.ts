// The snapshot team source of the private test (ADR-015 erratum E1, item 2):
// a pinned snapshot's verified LOCAL copy (`private-data/snapshots/<hash>/`)
// read through the one snapshot check (`readPrivateSnapshot`), then the
// records of the asked subset as teams. `all` is every active record,
// `holdout` the active hold-out records (the flag recomputed from the
// manifest's key by the reader). Nothing is written and nothing leaves: every
// refusal is a fixed `PrivateTestRefusal` code, never text of a lower layer.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { type Engine, privateDataRoot, PrivateTestRefusal, privateTestTeams, resolveFormat, type TestTeams, type TrainingRunConfig, teamVocabularyHash } from "@areneon/showdown-bridge";
import { loadFixtureInputs } from "@areneon/team-dataset";
import { readPrivateSnapshot } from "./snapshot-reader.ts";

/** The records of one subset as teams, in `team_id` order (the reader's), from the local copy of `snapshotHash`. */
export async function snapshotTeams(args: { engine: Engine; config: TrainingRunConfig; snapshotsDir: string; snapshotHash: string; subset: "all" | "holdout" }): Promise<TestTeams> {
  const { engine, config, snapshotsDir, snapshotHash, subset } = args;
  if (!/^[0-9a-f]{64}$/.test(snapshotHash)) throw new PrivateTestRefusal("source_unreadable");
  let dir: string;
  let manifest: unknown;
  try {
    privateDataRoot(snapshotsDir); // the copy lies inside a private-data directory
    dir = join(snapshotsDir, snapshotHash);
    manifest = JSON.parse(readFileSync(join(dir, "manifest.json"), "utf8"));
  } catch {
    throw new PrivateTestRefusal("source_unreadable");
  }
  let records;
  try {
    const inputs = loadFixtureInputs(engine, config.format_id);
    const read = await readPrivateSnapshot({
      engine,
      manifest,
      // A copy in a plain Uint8Array: the decode worker takes ownership of the buffer it is given.
      shard: (index) => new Uint8Array(readFileSync(join(dir, "shards", `${index}.parquet`))),
      labelSet: inputs.labelSet,
      expectedSnapshotHash: snapshotHash,
      expected: { manifest: resolveFormat(engine, config.format_id).manifest, vocabularyHash: teamVocabularyHash(inputs.vocabulary) },
    });
    records = read.records.filter((record) => record.status === "active" && (subset === "all" || record.holdout));
  } catch {
    throw new PrivateTestRefusal("snapshot_unverified");
  }
  if (records.length === 0) throw new PrivateTestRefusal("source_empty");
  return { teams: privateTestTeams(engine, config.format_id, records.map((record) => record.paste)), source: { kind: "snapshot", snapshot_hash: snapshotHash }, subset };
}
