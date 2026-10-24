// Engine re-simulation of one played A1 battle (ADR-017 section 4a, ticket #926): the pinned engine
// runs the battle again from its seed, both teams and both seats' recorded choices, and the result is
// returned as data. This module decides nothing: the caller (the training-data inspector) accepts the
// result only if both player channels equal the recorded ones. The omniscient log holds both sides'
// hidden information, so it is an evaluation artifact for local use: nothing here writes a file.
import type { Engine, PokemonSet } from "./engine.ts";
import { type BattleRecord, type Player, replayActor, runBattle } from "./runner.ts";

const PLAYERS: readonly Player[] = ["p1", "p2"];

export class ResimulationInputError extends Error {
  override name = "ResimulationInputError";
}

export interface ResimulationInput {
  format_id: string;
  /** `sodium,<32 hex>`: the seed the battle ran with. */
  seed: string;
  max_turns: number;
  teams: Record<Player, PokemonSet[]>;
  /** Every submitted choice of both seats, a refused one (`[Unavailable choice]`) included. */
  choices: { player: Player; request_seq: number; choice: string }[];
}

/** One decision as the engine took it again: the seat's legal set at that request (no verdicts). */
export interface ResimulatedDecision {
  player: Player;
  request_seq: number;
  choice: string;
  refused?: string;
  candidates: { choice: string; pending: boolean }[];
}

export interface Resimulation {
  winner: Player | null;
  turns: number;
  omniscient_log: string[];
  channels: Record<Player, string[]>;
  team_hashes: Record<Player, string>;
  decisions: ResimulatedDecision[];
}

const KEYS = ["choices", "format_id", "max_turns", "seed", "teams"];

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value);

/** Read the input strictly: a closed key set and the types it names. Messages quote no value. */
export function readResimulationInput(value: unknown): ResimulationInput {
  if (!isRecord(value) || Object.keys(value).sort().join() !== KEYS.join()) throw new ResimulationInputError("the input is not exactly the re-simulation request");
  const { format_id, seed, max_turns, teams, choices } = value;
  if (typeof format_id !== "string" || format_id === "") throw new ResimulationInputError("format_id is not a string");
  if (typeof seed !== "string" || !/^sodium,[0-9a-f]{32}$/.test(seed)) throw new ResimulationInputError("seed is not a sodium seed");
  if (!Number.isInteger(max_turns) || (max_turns as number) < 1) throw new ResimulationInputError("max_turns is not a positive integer");
  if (!isRecord(teams) || Object.keys(teams).sort().join() !== "p1,p2" || !PLAYERS.every((p) => Array.isArray(teams[p]) && (teams[p] as unknown[]).length > 0)) {
    throw new ResimulationInputError("teams are not one non-empty set list per seat");
  }
  if (!Array.isArray(choices)) throw new ResimulationInputError("choices is not a list");
  for (const item of choices) {
    if (
      !isRecord(item) ||
      Object.keys(item).sort().join() !== "choice,player,request_seq" ||
      !PLAYERS.includes(item.player as Player) ||
      !Number.isInteger(item.request_seq) ||
      (item.request_seq as number) < 1 ||
      typeof item.choice !== "string"
    ) {
      throw new ResimulationInputError("a choice is not exactly {player, request_seq, choice}");
    }
  }
  return value as unknown as ResimulationInput;
}

const order = (a: { player: string; request_seq: number }, b: { player: string; request_seq: number }): number =>
  a.player < b.player ? -1 : a.player > b.player ? 1 : a.request_seq - b.request_seq;

/**
 * Re-run the battle with `runBattle` and `replayActor`. Refused: a choice list that does not answer
 * the battle's requests one for one (a missing choice aborts the battle; a choice the battle never
 * asked for, or one the engine took differently, is refused here), and anything `runBattle` refuses.
 */
export async function resimulate(engine: Engine, input: ResimulationInput): Promise<Resimulation> {
  const given = [...input.choices].sort(order);
  const seen = new Set(given.map((c) => `${c.player}:${c.request_seq}`));
  if (seen.size !== given.length) throw new ResimulationInputError("two choices answer one request");
  const asRecorded: BattleRecord["choices"] = given.map((c) => ({ ...c }));
  const record = await runBattle({
    engine,
    formatId: input.format_id,
    seed: input.seed,
    teams: input.teams,
    actors: { p1: replayActor(asRecorded, "p1"), p2: replayActor(asRecorded, "p2") },
    maxTurns: input.max_turns,
  });
  const taken = record.choices.map(({ player, request_seq, choice }) => ({ player, request_seq, choice })).sort(order);
  if (JSON.stringify(taken) !== JSON.stringify(given.map(({ player, request_seq, choice }) => ({ player, request_seq, choice })))) {
    throw new ResimulationInputError("the recorded choices are not the choices the battle took");
  }
  const decisions = record.choices.map((c): ResimulatedDecision => {
    const legal = record.requests[c.player][c.request_seq - 1]?.legal;
    return {
      player: c.player,
      request_seq: c.request_seq,
      choice: c.choice,
      ...(c.refused === undefined ? {} : { refused: c.refused }),
      candidates: [
        ...(legal?.choices ?? []).map((choice) => ({ choice, pending: false })),
        ...(legal?.pending ?? []).map((choice) => ({ choice, pending: true })),
      ],
    };
  });
  return {
    winner: record.winner,
    turns: record.turns,
    omniscient_log: [...record.omniscient_log],
    channels: record.channels,
    team_hashes: record.team_hashes,
    decisions,
  };
}
