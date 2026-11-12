// The key layout of the public teams bucket (ADR-013 section 6). Shards are
// content-addressed under `pub/v1/shards/<sha256>` (the key the snapshot
// manifest's `object_key` names) and each snapshot's manifest sits under
// `pub/v1/snapshots/<snapshot_hash>/snapshot.json`. The store has no
// overwritable key: every write is `If-None-Match: *`, and nothing is deleted.

import type { ObjectLayout } from "@areneon/object-store";

export const SHARDS_PREFIX = "pub/v1/shards/";
export const SNAPSHOTS_PREFIX = "pub/v1/snapshots/";

const SHARD_KEY = /^pub\/v1\/shards\/[0-9a-f]{64}$/;
const SNAPSHOT_KEY = /^pub\/v1\/snapshots\/[0-9a-f]{64}\/snapshot\.json$/;

export const snapshotKey = (snapshotHash: string): string => `${SNAPSHOTS_PREFIX}${snapshotHash}/snapshot.json`;

export const TEAMS_LAYOUT: ObjectLayout = {
  allowsKey: (key) => SHARD_KEY.test(key) || SNAPSHOT_KEY.test(key),
  allowsListPrefix: (prefix) => prefix === SHARDS_PREFIX || prefix === SNAPSHOTS_PREFIX,
  overwriteKey: null,
};
