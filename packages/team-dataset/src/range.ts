// A range of the draw stream, compared across platforms (ADR-013 section 2,
// "Two-platform check"). Draws are independent, so a range [first, first +
// count) can be computed alone. Its summary is one fixed-width record per draw
// (counter 8 bytes, status 1, cell 2, canonical hash 32: 43 bytes) and the
// range's digest is the SHA-256 of those records in counter order. Two
// platforms agree on a range exactly when their digests are equal, which is
// the byte-equality of the per-draw summaries without shipping them.
//
// status: 0 accepted, 1 rejected by the validator, 2 rejected by the paste
// round trip; the cell of a rejected draw is 0xffff.

import { createHash } from "node:crypto";
import type { Drawer } from "./draw.ts";
import type { StreamIdentity } from "./stream.ts";

export interface RangeSummary {
  /** SHA-256 of the range's summary records. */
  digest: string;
  accepted: number;
  rejected: number;
}

const RECORD = 43;

export function rangeSummary(drawer: Drawer, stream: StreamIdentity, first: number, count: number): RangeSummary {
  const hash = createHash("sha256");
  const record = Buffer.alloc(RECORD);
  let accepted = 0;
  for (let k = first; k < first + count; k++) {
    const draw = drawer.draw(stream, k);
    record.writeBigUInt64BE(BigInt(k), 0);
    record.writeUInt8(draw.status === "accepted" ? 0 : draw.reason === "invalid" ? 1 : 2, 8);
    record.writeUInt16BE(draw.status === "accepted" ? draw.cell : 0xffff, 9);
    record.write(draw.canonical_team_hash, 11, 32, "hex");
    hash.update(record);
    if (draw.status === "accepted") accepted++;
  }
  return { digest: hash.digest("hex"), accepted, rejected: count - accepted };
}
