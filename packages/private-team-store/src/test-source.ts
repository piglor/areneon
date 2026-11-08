// The snapshot team source of the private test (ADR-015 erratum E1, item 2):
// a pinned snapshot's verified LOCAL copy (`private-data/snapshots/<hash>/`)
// read through the one snapshot check (`readPrivateSnapshot`), then the
// records of the asked subset as teams. `all` is every active record,
// `holdout` the active hold-out records (the flag recomputed from the
// manifest's key by the reader). Nothing is written and nothing leaves: every
// refusal is a fixed `PrivateTestRefusal` code, never text of a lower layer.

import { type Engine, PrivateTestRefusal, privateTestTeams, type TestTeams, type TrainingRunConfig } from "@areneon/showdown-bridge";
import { ActiveRecordsError, readActiveRecords } from "./active-records.ts";

/** The records of one subset as teams, in `team_id` order (the reader's), from the local copy of `snapshotHash`. */
export async function snapshotTeams(args: { engine: Engine; config: TrainingRunConfig; snapshotsDir: string; snapshotHash: string; subset: "all" | "holdout" }): Promise<TestTeams> {
  const { engine, config, snapshotsDir, snapshotHash, subset } = args;
  let records;
  try {
    const active = await readActiveRecords({ engine, formatId: config.format_id, snapshotsDir, snapshotHash });
    records = active.filter((record) => subset === "all" || record.holdout);
  } catch (error) {
    throw new PrivateTestRefusal(error instanceof ActiveRecordsError ? error.code : "snapshot_unverified");
  }
  if (records.length === 0) throw new PrivateTestRefusal("source_empty");
  return { teams: privateTestTeams(engine, config.format_id, records.map((record) => record.paste)), source: { kind: "snapshot", snapshot_hash: snapshotHash }, subset };
}
