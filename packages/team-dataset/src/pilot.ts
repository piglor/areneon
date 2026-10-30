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
  /**
   * With a mixture proposal only (ADR-020 section 6): the same counts per component, in config order. The
   * component of pilot draw k is recomputed from its seed. Absent for a uniform config, so such a pilot keeps its bytes.
   */
  components?: PilotComponent[];
}

export interface PilotComponent {
  kind: string;
  draws: number;
  rejected: number;
  cell_hits: number[];
  /**
   * Component `usage` only (ADR-020 section 6, P3): the largest number of this component's pilot draws, accepted or not,
   * that produced one and the same canonical team. A count, a measured check beside the P3 bound; never gated.
   */
  max_team_count?: number;
}

export interface PilotIdentity {
  config_hash: string;
  label_set_hash: string;
  vocabulary_hash: string;
}

/** Run `draws` pilot draws. */
export function runPilot(drawer: Drawer, identity: PilotIdentity, labelSet: TeamLabelSet, draws: number): Pilot {
  const cellHits = new Array<number>(cellCount(labelSet)).fill(0);
  const kinds = drawer.components;
  const parts: PilotComponent[] | undefined = kinds?.map((kind) => ({ kind, draws: 0, rejected: 0, cell_hits: new Array<number>(cellHits.length).fill(0) }));
  let rejected = 0;
  const teams = new Map<string, number>();
  for (let k = 0; k < draws; k++) {
    const draw = drawer.pilot(identity.config_hash, k);
    const kind = parts === undefined ? undefined : drawer.pilotComponent(identity.config_hash, k);
    const part = parts?.find((p) => p.kind === kind);
    if (part !== undefined) part.draws++;
    if (kind === "usage") teams.set(draw.canonical_team_hash, (teams.get(draw.canonical_team_hash) ?? 0) + 1);
    if (draw.status === "accepted") {
      cellHits[draw.cell] = (cellHits[draw.cell] as number) + 1;
      if (part !== undefined) part.cell_hits[draw.cell] = (part.cell_hits[draw.cell] as number) + 1;
    } else {
      rejected++;
      if (part !== undefined) part.rejected++;
    }
  }
  const usage = parts?.find((p) => p.kind === "usage");
  if (usage !== undefined) usage.max_team_count = Math.max(0, ...teams.values());
  return { format_version: 1, ...identity, draws, rejected, cell_hits: cellHits, ...(parts === undefined ? {} : { components: parts }) };
}

const HASH = /^[0-9a-f]{64}$/;
const count = (v: unknown): v is number => typeof v === "number" && Number.isSafeInteger(v) && v >= 0;

/** Parse a stored pilot and check it belongs to this config, label set and vocabulary. */
export function readPilot(value: unknown, identity: PilotIdentity, labelSet: TeamLabelSet): Pilot {
  const pilot = value as Partial<Pilot> | null;
  if (typeof pilot !== "object" || pilot === null || Array.isArray(pilot)) throw new PilotError("the pilot is not an object");
  const keys = Object.keys(pilot).filter((key) => key !== "components").sort().join();
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
  if (pilot.components !== undefined) readPilotComponents(pilot as Pilot, labelSet);
  return pilot as Pilot;
}

/** The per-component counts add up to the pilot's own. */
function readPilotComponents(pilot: Pilot, labelSet: TeamLabelSet): void {
  const parts = pilot.components as PilotComponent[];
  if (!Array.isArray(parts) || parts.length === 0) throw new PilotError("components is not a non-empty list");
  const cells = cellCount(labelSet);
  const sum = new Array<number>(cells).fill(0);
  let draws = 0;
  let rejected = 0;
  for (const part of parts) {
    const fields = Object.keys(part).sort().join();
    if (fields !== "cell_hits,draws,kind,rejected" && fields !== "cell_hits,draws,kind,max_team_count,rejected") throw new PilotError("a pilot component has other fields than kind, draws, rejected, cell_hits and (usage) max_team_count");
    if (part.max_team_count !== undefined && (part.kind !== "usage" || !count(part.max_team_count) || part.max_team_count > part.draws)) throw new PilotError("max_team_count is for the usage component only and is a count of at most its draws");
    if (typeof part.kind !== "string" || !count(part.draws) || !count(part.rejected) || !Array.isArray(part.cell_hits) || part.cell_hits.length !== cells || !part.cell_hits.every(count)) {
      throw new PilotError("a pilot component is not a kind with counts and one hit count per cell");
    }
    if (part.rejected + part.cell_hits.reduce((a, b) => a + b, 0) !== part.draws) throw new PilotError(`component ${part.kind}: rejected and the cell hits do not add up to its draws`);
    draws += part.draws;
    rejected += part.rejected;
    part.cell_hits.forEach((n, cell) => (sum[cell] = (sum[cell] as number) + n));
  }
  if (draws !== pilot.draws || rejected !== pilot.rejected || sum.some((n, cell) => n !== pilot.cell_hits[cell])) throw new PilotError("the components do not add up to the pilot's draws, rejections and cell hits");
}
