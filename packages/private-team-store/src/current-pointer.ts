// `current.json` (ADR-015 section 1): the one mutable object of the bucket,
// outside `snapshots/`. It names the latest snapshot and that snapshot's
// parent, so an import can tell that the bucket still holds the state it was
// planned against. It is a convenience for the next import and for
// inspection; a run pins a `snapshot_hash` instead. It holds two hashes and
// nothing else.

import { canonicalJson } from "@areneon/showdown-bridge";
import { SnapshotIoError } from "./snapshot-io-error.ts";

export interface CurrentPointer {
  readonly schema_version: 1;
  readonly snapshot_hash: string;
  /** The hash of the snapshot this one was made from, or null for the first. */
  readonly parent_snapshot_hash: string | null;
}

const HASH = /^[0-9a-f]{64}$/;

export function encodeCurrent(pointer: CurrentPointer): Uint8Array {
  return new TextEncoder().encode(canonicalJson(pointer as unknown as Parameters<typeof canonicalJson>[0]));
}

/** Parse a stored `current.json`. Exactly the three keys, canonical bytes; anything else is `current_invalid`. */
export function readCurrentPointer(bytes: Uint8Array): CurrentPointer {
  try {
    const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    const value = JSON.parse(text) as Record<string, unknown>;
    const parent = value["parent_snapshot_hash"];
    const hash = value["snapshot_hash"];
    if (
      value["schema_version"] === 1 &&
      typeof hash === "string" &&
      HASH.test(hash) &&
      (parent === null || (typeof parent === "string" && HASH.test(parent))) &&
      Object.keys(value).length === 3 &&
      canonicalJson(value as Parameters<typeof canonicalJson>[0]) === text
    ) {
      return { schema_version: 1, snapshot_hash: hash, parent_snapshot_hash: parent };
    }
  } catch {
    // Falls through to the one refusal: what the parser said can quote the file.
  }
  throw new SnapshotIoError("current_invalid");
}
