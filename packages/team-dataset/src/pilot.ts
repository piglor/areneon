// The recorded pilot (ADR-013 section 2): `draws` pilot draws of the
// configured proposal, counted per base cell. It decides the D20 merge and the
// quotas, so it is a record of its own, tied to the config, label set and
// vocabulary it measured. A hit is a draw the engine accepted; a rejected draw
// counts toward `draws` but lands in no cell, so a cell's rate is
// hits / draws, the chance that one draw of the stream is a member.

import { cellCount } from "./labeller.ts";
import type { Drawer } from "./draw.ts";
import type { TeamLabelSet } from "@areneon/showdown-bridge";

export class PilotError extends Error {
  override name = "PilotError";
}

export interface Pilot {
  format_version: 1;
  config_hash: string;
  label_set_hash: string;
  vocabulary_hash: string;
  draws: number;
  rejected: number;
  /** Accepted draws per base cell, in cell order. */
  cell_hits: number[];
}

export interface PilotIdentity {
  config_hash: string;
  label_set_hash: string;
  vocabulary_hash: string;
}

/** Run `draws` pilot draws. */
export function runPilot(drawer: Drawer, identity: PilotIdentity, labelSet: TeamLabelSet, draws: number): Pilot {
  const cellHits = new Array<number>(cellCount(labelSet)).fill(0);
  let rejected = 0;
  for (let k = 0; k < draws; k++) {
    const draw = drawer.pilot(identity.config_hash, k);
    if (draw.status === "accepted") cellHits[draw.cell] = (cellHits[draw.cell] as number) + 1;
    else rejected++;
  }
  return { format_version: 1, ...identity, draws, rejected, cell_hits: cellHits };
}

const HASH = /^[0-9a-f]{64}$/;
const count = (v: unknown): v is number => typeof v === "number" && Number.isSafeInteger(v) && v >= 0;

/** Parse a stored pilot and check it belongs to this config, label set and vocabulary. */
export function readPilot(value: unknown, identity: PilotIdentity, labelSet: TeamLabelSet): Pilot {
  const pilot = value as Partial<Pilot> | null;
  if (typeof pilot !== "object" || pilot === null || Array.isArray(pilot)) throw new PilotError("the pilot is not an object");
  const keys = Object.keys(pilot).sort().join();
  if (keys !== "cell_hits,config_hash,draws,format_version,label_set_hash,rejected,vocabulary_hash") throw new PilotError(`the pilot has the fields ${keys}`);
  if (pilot.format_version !== 1) throw new PilotError("the pilot's format_version is not 1");
  for (const key of ["config_hash", "label_set_hash", "vocabulary_hash"] as const) {
    if (typeof pilot[key] !== "string" || !HASH.test(pilot[key])) throw new PilotError(`${key} is not a SHA-256`);
    if (pilot[key] !== identity[key]) throw new PilotError(`the pilot was recorded for another ${key.replace("_hash", "")}`);
  }
  const { draws, rejected, cell_hits: hits } = pilot;
  if (!count(draws) || draws < 1) throw new PilotError("draws is not a positive integer");
  if (!count(rejected)) throw new PilotError("rejected is not a count");
  if (!Array.isArray(hits) || hits.length !== cellCount(labelSet) || !hits.every(count)) throw new PilotError("cell_hits is not one count per cell");
  if (rejected + (hits as number[]).reduce((a, b) => a + b, 0) !== draws) throw new PilotError("rejected and the cell hits do not add up to draws");
  return pilot as Pilot;
}
