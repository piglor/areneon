/* eslint-disable @typescript-eslint/no-explicit-any */
// The pinned Showdown as a battle-step backend (ADR-030 groundwork G2, ADR-028 section 5, #1462): the oracle of the battle-level
// differential. It drives the engine's own BattleStream exactly as `runBattle` writes to it (one `_write` per start chunk,
// choice and command, the same `>start` and `>player` lines), with the battle's random source replaced by the forcing source of
// `battle-trace.ts`: each call is answered from the step's forced draws when its method and arguments match the next one, else
// by the engine's own generator (which advances in lock step either way), and every call is reported. After each step it
// reports what the runner would have seen: each player's channel lines and requests, every request's identity and the mask of
// its candidates (the engine's `CloneJudge` on this battle, pending candidates with their hidden verdict, as the runner records
// them), the draw requests made, the outcome so far and the hash of the engine's serialised state.
//
// The shapes below are the differential core's `BattleBackend` port, declared structurally: the bridge does not import the core
// (`@areneon/differential`), the composition root joins the two. Isolation: see `battle-trace.ts`.
import type { Engine, PokemonSet, SimBattle } from "./engine.ts";
import { resolveFormat } from "./formats.ts";
import type { RulesSnapshot } from "./rules.ts";
import { addEmitted, type Action, type Draw, forcingSource, installSource, type MaskEntry, type Seat, stateHash, stepRequests, tracedStreamClass } from "./battle-trace.ts";
import { validateTeam } from "./teams.ts";

/** Bumped when the backend computes anything differently. */
export const REPLAY_FORMAT = 1;

export type Settled<T> = { ok: true; result: T } | { ok: false; error: unknown };

export interface ReplayStart {
  battle_id: string;
  format_id: string;
  seed: string;
  teams: Record<Seat, PokemonSet[]>;
  draws: Draw[];
}

export interface ReplayStep {
  handle: string;
  actions: Action[];
  draws: Draw[];
}

export interface ReplayRequest {
  request_seq: number;
  request_hash: string;
  mask?: MaskEntry[];
}

export interface ReplayOutput {
  channels: Record<Seat, string[]>;
  requests: Record<Seat, ReplayRequest[]>;
  draws: Draw[];
  ended: boolean;
  winner: Seat | null;
  turns: number;
  state_sha256: string;
}

export interface ShowdownBattleBackend {
  readonly id: string;
  readonly backend_format: number;
  readonly backend_commit: string;
  start(batch: readonly ReplayStart[]): Settled<{ handle: string; output: ReplayOutput }>[];
  step(batch: readonly ReplayStep[]): Settled<ReplayOutput>[];
  release(handles: readonly string[]): void;
}

export class ReplayError extends Error {
  override name = "ReplayError";
}

const SEATS: readonly Seat[] = ["p1", "p2"];

interface Battle {
  stream: any;
  forcing: ReturnType<typeof forcingSource> | null;
  seqs: Record<Seat, number>;
  winner: Seat | null;
  turns: number;
  out: { channels: Record<Seat, string[]>; omniscient: string[]; bodies: Record<Seat, string[]> };
  failure: unknown;
}

export interface ReplayOptions {
  snapshot?: RulesSnapshot;
  /** The name reports give it (default `showdown-battle`). */
  id?: string;
}

/** The pinned engine `engine` as a battle-step backend. Handles name battles of this backend object only. */
export function showdownBattleBackend(engine: Engine, options: ReplayOptions = {}): ShowdownBattleBackend {
  const battles = new Map<string, Battle>();
  let next = 0;

  const write = (b: Battle, chunk: string): void => {
    b.stream._write(chunk);
    if (b.failure !== undefined) throw new ReplayError("the engine failed on a write");
  };

  const output = (b: Battle): ReplayOutput => {
    const live = b.stream.battle as SimBattle;
    for (const line of b.out.omniscient) {
      if (line.startsWith("|turn|")) b.turns = Number(line.slice("|turn|".length));
      else if (line.startsWith("|win|")) {
        const name = line.slice("|win|".length);
        if (name !== "p1" && name !== "p2") throw new ReplayError("|win| names no seat");
        b.winner = name;
      }
    }
    const requests = stepRequests(engine, live, b.out.bodies, b.seqs);
    const result: ReplayOutput = {
      channels: { p1: b.out.channels.p1, p2: b.out.channels.p2 },
      requests,
      draws: (b.forcing as ReturnType<typeof forcingSource>).requests(),
      ended: live.ended,
      winner: live.ended ? b.winner : null,
      turns: b.turns,
      state_sha256: stateHash(live),
    };
    b.out = { channels: { p1: [], p2: [] }, omniscient: [], bodies: { p1: [], p2: [] } };
    return result;
  };

  const startOne = (input: ReplayStart): { handle: string; output: ReplayOutput } => {
    const resolved = resolveFormat(engine, input.format_id, options.snapshot);
    const played = { p1: structuredClone(input.teams.p1), p2: structuredClone(input.teams.p2) };
    for (const seat of SEATS) if (!validateTeam(engine, resolved, played[seat]).valid) throw new ReplayError(`the ${seat} team is refused by the validator`);
    const b: Battle = { stream: null, forcing: null, seqs: { p1: 0, p2: 0 }, winner: null, turns: 0, out: { channels: { p1: [], p2: [] }, omniscient: [], bodies: { p1: [], p2: [] } }, failure: undefined };
    const Traced = tracedStreamClass(engine, {
      forward: false,
      started(battle) {
        b.forcing = forcingSource(battle.prng);
        b.forcing.load(input.draws);
        installSource(battle, b.forcing.source);
      },
      message: (type, data) => addEmitted(b.out, type, data),
      error(error) {
        b.failure = error;
      },
    });
    b.stream = new Traced();
    write(b, [`>start ${JSON.stringify({ formatid: input.format_id, seed: input.seed })}`, ...SEATS.map((seat) => `>player ${seat} ${JSON.stringify({ name: seat, team: engine.sim.Teams.pack(played[seat]) })}`)].join("\n"));
    const handle = `${input.battle_id}#${next++}`;
    battles.set(handle, b);
    return { handle, output: output(b) };
  };

  const stepOne = (input: ReplayStep): ReplayOutput => {
    const b = battles.get(input.handle);
    if (b === undefined) throw new ReplayError("no battle has that handle");
    (b.forcing as ReturnType<typeof forcingSource>).load(input.draws);
    for (const action of input.actions) write(b, action.kind === "choice" ? `>${action.seat} ${action.choice}` : `>${action.command}`);
    return output(b);
  };

  const settle = <I, O>(fn: (input: I) => O) => (input: I): Settled<O> => {
    try {
      return { ok: true, result: fn(input) };
    } catch (error) {
      return { ok: false, error };
    }
  };

  return {
    id: options.id ?? "showdown-battle",
    backend_format: REPLAY_FORMAT,
    backend_commit: engine.commit,
    start: (batch) => batch.map(settle(startOne)),
    step: (batch) => batch.map(settle(stepOne)),
    release(handles) {
      for (const handle of handles) battles.delete(handle);
    },
  };
}
