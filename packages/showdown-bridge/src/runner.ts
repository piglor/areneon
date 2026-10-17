import type { Engine, PokemonSet, SimStream } from "./engine.ts";
import { resolveFormat, type VersionManifest } from "./formats.ts";
import { type DecodedRequest, decodeRequest } from "./request.ts";

export type Player = "p1" | "p2";
const PLAYERS: readonly Player[] = ["p1", "p2"];

/**
 * Chooses for one player. It sees only that player's own request, never the
 * omniscient stream. The engine judges the choice: any `|error|` aborts the
 * battle rather than being retried.
 */
export type Actor = (request: DecodedRequest) => string | Promise<string>;

export interface BattleOptions {
  engine: Engine;
  formatId: string;
  /** Showdown PRNG seed, e.g. `sodium,<32 hex digits>`. */
  seed: string;
  teams: Record<Player, PokemonSet[]>;
  actors: Record<Player, Actor>;
  /** Force a tie once this turn starts. Defaults to 300. */
  maxTurns?: number;
}

export interface RequestRecord extends DecodedRequest {
  /** 1-based per-player position among that player's requests. */
  request_seq: number;
}

export interface BattleRecord {
  manifest: VersionManifest;
  seed: string;
  /** `p1`, `p2`, or `null` for a tie. */
  winner: Player | null;
  turns: number;
  /**
   * Omniscient sim log with wall-clock `|t:|` lines removed. It holds both
   * sides' hidden information: an evaluation artifact, never an actor input.
   */
  log: string[];
  requests: Record<Player, RequestRecord[]>;
  choices: { player: Player; request_seq: number; choice: string }[];
}

export class InvalidTeamError extends Error {
  override name = "InvalidTeamError";
  readonly player: Player;
  readonly problems: string[];
  constructor(player: Player, problems: string[]) {
    super(`${player} team rejected by the Showdown validator: ${problems.join("; ")}`);
    this.player = player;
    this.problems = problems;
  }
}

export class BattleAbortedError extends Error {
  override name = "BattleAbortedError";
}

/**
 * Run one headless battle on the in-process sim (runtime A1). Same engine,
 * format, seed, teams and choices always produce the same `log`.
 */
export async function runBattle(options: BattleOptions): Promise<BattleRecord> {
  const { engine, formatId, seed, teams, actors, maxTurns = 300 } = options;
  const { manifest } = resolveFormat(engine, formatId);
  const validator = engine.sim.TeamValidator.get(formatId);
  for (const player of PLAYERS) {
    const problems = validator.validateTeam(teams[player]);
    if (problems) throw new InvalidTeamError(player, problems);
  }

  const streams = engine.sim.getPlayerStreams(new engine.sim.BattleStream());
  const record: BattleRecord = {
    manifest,
    seed,
    winner: null,
    turns: 0,
    log: [],
    requests: { p1: [], p2: [] },
    choices: [],
  };
  let failure: Error | undefined;
  let ended = false;
  // Showdown's BattleStream crashes if ended twice, and drops writes after.
  const end = (): void => {
    if (ended) return;
    ended = true;
    streams.omniscient.writeEnd();
  };
  const abort = (error: Error): void => {
    failure ??= error;
    end();
  };

  const watchLog = async (): Promise<void> => {
    for await (const chunk of streams.omniscient) {
      for (const line of chunk.split("\n")) {
        if (line.startsWith("|t:|")) continue;
        record.log.push(line);
        if (line.startsWith("|turn|")) {
          record.turns = Number(line.slice("|turn|".length));
          if (record.turns > maxTurns) streams.omniscient.write(">forcetie");
        } else if (line.startsWith("|win|")) {
          record.winner = line.slice("|win|".length) as Player;
          end();
        } else if (line === "|tie") {
          end();
        }
      }
    }
  };

  const play = async (player: Player, stream: SimStream): Promise<void> => {
    for await (const chunk of stream) {
      for (const line of chunk.split("\n")) {
        if (line.startsWith("|error|")) {
          abort(new BattleAbortedError(`${player}: ${line.slice("|error|".length)}`));
        } else if (line.startsWith("|request|")) {
          const request = { ...decodeRequest(line.slice("|request|".length)), request_seq: 0 };
          request.request_seq = record.requests[player].push(request);
          if (request.phase === "wait") continue;
          try {
            const choice = await actors[player](request);
            if (ended) return;
            record.choices.push({ player, request_seq: request.request_seq, choice });
            stream.write(choice);
          } catch (error) {
            abort(error instanceof Error ? error : new Error(String(error)));
          }
        }
      }
    }
  };

  const done = Promise.all([watchLog(), play("p1", streams.p1), play("p2", streams.p2)]);
  const start = { formatid: formatId, seed };
  streams.omniscient.write(
    [
      `>start ${JSON.stringify(start)}`,
      ...PLAYERS.map(
        (player) =>
          `>player ${player} ${JSON.stringify({ name: player, team: engine.sim.Teams.pack(teams[player]) })}`,
      ),
    ].join("\n"),
  );
  await done;
  if (failure) throw failure;
  return record;
}
