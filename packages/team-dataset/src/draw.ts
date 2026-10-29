// One draw of the stream (ADR-013 section 2): propose a team from the draw's
// seed, put it in canonical order, round-trip it through the engine's paste
// import and export, ask the pinned validator, label it. The engine is the
// only legality authority: a draw is accepted exactly when `importPaste` takes
// its export back unchanged and `validateTeam` finds nothing wrong.

import {
  canonicalTeamHash,
  type Engine,
  exportPaste,
  importPaste,
  type PokemonSet,
  type ResolvedFormat,
  type TeamLabelSet,
  TeamImportError,
  teamHash,
  validateTeam,
} from "@areneon/showdown-bridge";
import { jcs } from "@areneon/showdown-bridge";
import { cellOf, type LabelValue, labelTeam } from "./labeller.ts";
import type { Proposal } from "./proposal.ts";
import { drawSeed, engineRng, pilotSeed, type StreamIdentity } from "./stream.ts";

/** Why a draw was not kept. */
export type RejectionReason = "round_trip" | "invalid";

export interface Accepted {
  status: "accepted";
  k: number;
  /** The team's identity regardless of set order. */
  canonical_team_hash: string;
  team_hash: string;
  /** The team as the engine exports it. */
  paste: string;
  labels: LabelValue[];
  /** The base cell number of the labels. */
  cell: number;
}

export interface Rejected {
  status: "rejected";
  k: number;
  reason: RejectionReason;
  /** The engine's first complaint, verbatim, or the import error. */
  detail: string;
  canonical_team_hash: string;
}

export type Draw = Accepted | Rejected;

const toId = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");
const codeUnits = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);

/** The canonical order of `canonicalTeamHash`: species id, ties by the set's JCS bytes. */
export function canonicalOrder(sets: readonly PokemonSet[]): PokemonSet[] {
  const keyed = sets.map((set) => ({ set, species: toId(set.species), bytes: jcs(set as never) }));
  keyed.sort((a, b) => codeUnits(a.species, b.species) || codeUnits(a.bytes, b.bytes));
  return keyed.map((k) => k.set);
}

export interface DrawerInput {
  engine: Engine;
  format: ResolvedFormat;
  proposal: Proposal;
  labelSet: TeamLabelSet;
  /** The HMAC key as 64 hex digits. */
  generationKey: string;
}

export class Drawer {
  readonly #input: DrawerInput;

  constructor(input: DrawerInput) {
    this.#input = input;
  }

  /** Draw `k` of the plan's stream. */
  draw(stream: StreamIdentity, k: number): Draw {
    return this.#judge(k, drawSeed(this.#input.generationKey, stream, k));
  }

  /** Pilot draw `k` of the config's pilot stream. */
  pilot(configHash: string, k: number): Draw {
    return this.#judge(k, pilotSeed(this.#input.generationKey, configHash, k));
  }

  /** The component draw `k` of `stream` takes (recomputed from its seed alone; "uniform" with no mixture). */
  component(stream: StreamIdentity, k: number): string {
    return this.#input.proposal.component(engineRng(this.#input.engine, drawSeed(this.#input.generationKey, stream, k)));
  }

  /** The component pilot draw `k` takes. */
  pilotComponent(configHash: string, k: number): string {
    return this.#input.proposal.component(engineRng(this.#input.engine, pilotSeed(this.#input.generationKey, configHash, k)));
  }

  /** The configured components, in config order; undefined with no mixture. */
  get components(): readonly string[] | undefined {
    return this.#input.proposal.componentKinds();
  }

  #judge(k: number, seed: string): Draw {
    const { engine, format, proposal, labelSet } = this.#input;
    const proposed = canonicalOrder(proposal.propose(engineRng(engine, seed)));
    const reject = (reason: RejectionReason, detail: string): Rejected => ({
      status: "rejected",
      k,
      reason,
      detail,
      canonical_team_hash: canonicalTeamHash(proposed),
    });
    const paste = exportPaste(engine, proposed);
    let sets: PokemonSet[];
    try {
      sets = importPaste(engine, paste);
    } catch (error) {
      if (!(error instanceof TeamImportError)) throw error;
      return reject("round_trip", error.message);
    }
    if (exportPaste(engine, sets) !== paste) return reject("round_trip", "the export of the imported team differs from the paste");
    // The identities and labels are read before the validator runs: it normalises the sets it is given in place.
    const canonical = canonicalTeamHash(sets);
    const positional = teamHash(sets);
    const labels = labelTeam(labelSet, sets);
    const verdict = validateTeam(engine, format, structuredClone(sets));
    if (!verdict.valid) return { status: "rejected", k, reason: "invalid", detail: verdict.problems[0] as string, canonical_team_hash: canonical };
    return { status: "accepted", k, canonical_team_hash: canonical, team_hash: positional, paste, labels, cell: cellOf(labelSet, labels) };
  }
}
