// The battle-step backend port (ADR-030 groundwork G2, ADR-028 section 5): level 2 of the differential harness. A backend is
// started on a batch of battles and stepped on a batch of (battle, actions, forced draws); every call returns one outcome per
// input, in input order, a refusal kept as that input's outcome (as level 1's `Backend.run`). The pinned Showdown, driven with a
// trace's draws forced through a recording random source, is the oracle backend (the bridge's `showdownBattleBackend`).
import type { Action, Draw, MaskEntry, PokemonSet } from "@areneon/schema/types/v1/battle-trace.ts";
import type { Settled } from "./backend.ts";

export type { Action, Draw, MaskEntry, PokemonSet };

export type Seat = "p1" | "p2";
export const SEATS: readonly Seat[] = ["p1", "p2"];

/** One battle to start: the seed names it, the teams are each seat's sets as brought; `draws` are the start's forced draws. */
export interface BattleStartInput {
  battle_id: string;
  format_id: string;
  seed: string;
  teams: Record<Seat, PokemonSet[]>;
  draws: Draw[];
}

/** One step of a started battle: what is written to it, in order, and the draws to answer its random source with, in order. */
export interface BattleStepInput {
  handle: string;
  actions: Action[];
  draws: Draw[];
}

/** One request a seat received in the step: its identity and the verdict of every candidate (absent for a wait request). */
export interface StepRequest {
  request_seq: number;
  request_hash: string;
  mask?: MaskEntry[];
}

/** What one step (or the start) gave: per seat the channel lines and requests, the draw requests made (with what was answered), the outcome so far. */
export interface BattleStepOutput {
  channels: Record<Seat, string[]>;
  requests: Record<Seat, StepRequest[]>;
  draws: Draw[];
  ended: boolean;
  winner: Seat | null;
  turns: number;
  /**
   * SHA-256 of the backend's own serialised state, when it has the engine's (`toJSON`). Compared only when given: a backend that
   * omits it gets no state check, so only the five ADR-028 items hold it to the engine.
   */
  state_sha256?: string;
}

export interface BattleStarted {
  handle: string;
  output: BattleStepOutput;
}

/**
 * A battle-step backend. `project` gives a seat's player-view projection of the backend's battle-state tensors (a `SeatView` of
 * `battle-projection.ts`: the fields the visibility classes let that seat read) after the last call; the harness compares it with
 * the reference projection when it is given a projection oracle (G4, #1464).
 */
export interface BattleBackend {
  readonly id: string;
  /** Bumped when the backend computes anything differently. */
  readonly backend_format: number;
  /** The frozen commit of its code (40 hex). */
  readonly backend_commit: string;
  start(batch: readonly BattleStartInput[]): Settled<BattleStarted>[];
  step(batch: readonly BattleStepInput[]): Settled<BattleStepOutput>[];
  /** Forget battles the harness is done with. */
  release(handles: readonly string[]): void;
  project?(handle: string, seat: Seat): unknown;
}
