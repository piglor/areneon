import {
  CloneJudge,
  judgeSubmission,
  type LegalChoiceSet,
  legalChoiceSet,
  type Verdict,
} from "./choice-judge.ts";
import type { Engine, PokemonSet, SimBattle, SimStream } from "./engine.ts";
import { resolveFormat, type VersionManifest } from "./formats.ts";
import type { FormatRulesRecord, RulesSnapshot } from "./rules.ts";
import { type DecodedRequest, decodeRequest } from "./request.ts";
import { validateTeam } from "./teams.ts";

export type Player = "p1" | "p2";
const PLAYERS: readonly Player[] = ["p1", "p2"];

/**
 * Chooses for one player. It sees only that player's own request and the
 * legal choice set built from it, never the omniscient stream or a verdict
 * on a pending candidate. The engine judges the choice. A pending candidate
 * refused as `[Unavailable choice]` brings a new request; any other
 * `|error|` aborts the battle rather than being retried.
 */
export type Actor = (request: RequestRecord) => string | Promise<string>;

export interface BattleOptions {
  engine: Engine;
  formatId: string;
  /** Showdown PRNG seed, e.g. `sodium,<32 hex digits>`. */
  seed: string;
  teams: Record<Player, PokemonSet[]>;
  actors: Record<Player, Actor>;
  /** Force a tie once this turn starts. Defaults to 300. */
  maxTurns?: number;
  /**
   * The rules snapshot to resolve the format against. Defaults to reading
   * and validating `rules/` again; a caller running many battles loads it
   * once and passes it here.
   */
  snapshot?: RulesSnapshot;
  /**
   * The `name` and `avatar` of each seat's `|player|` line, to reproduce a
   * recorded battle. Defaults to `p1` / `p2` and no avatar. Refused with a
   * `PlayersOptionError` unless valid (see `validatePlayers`).
   */
  players?: Record<Player, PlayerIdentity>;
}

export interface PlayerIdentity {
  name: string;
  /** Omitted or empty: no avatar on the `|player|` line. */
  avatar?: string;
}

export class PlayersOptionError extends Error {
  constructor(problem: string) {
    super(`invalid players option: ${problem}`);
    this.name = "PlayersOptionError";
  }
}

// Names: what the pinned server's name filter leaves of a name (at most 18
// UTF-16 units, no control or format characters, no `,` `[` `]`, one space at
// most between words, which it collapses), minus `|`, which would split a
// protocol line. Avatars: the server's ids and numbers (`102`, `cynthia`).
const NAME = /^[^|,[\]\p{Cc}\p{Cf}\p{Z}]([^|,[\]\p{Cc}\p{Cf}\p{Zl}\p{Zp}]*[^|,[\]\p{Cc}\p{Cf}\p{Z}])?$/u;
const AVATAR = /^[A-Za-z0-9-]{1,32}$/;
const idOf = (name: string): string => name.toLowerCase().replace(/[^a-z0-9]+/g, "");

/** Validate `players`: every name and avatar is protocol-safe, and the two names differ. */
export function validatePlayers(players: Record<Player, PlayerIdentity>): void {
  for (const player of PLAYERS) {
    const { name, avatar } = players[player];
    if (name.length > 18 || !NAME.test(name) || name.includes("  ") || idOf(name) === "") {
      throw new PlayersOptionError(`${player} name must be 1-18 characters, no \`|\`, no control characters, not blank`);
    }
    if (avatar !== undefined && avatar !== "" && !AVATAR.test(avatar)) {
      throw new PlayersOptionError(`${player} avatar must be 1-32 letters, digits or \`-\``);
    }
  }
  if (idOf(players.p1.name) === idOf(players.p2.name)) throw new PlayersOptionError("the two names must differ");
}

export interface RequestRecord extends DecodedRequest {
  /** 1-based per-player position among that player's requests. */
  request_seq: number;
  /**
   * What the player may choose (ADR-006), engine-judged on a clone of the
   * battle. Absent on `wait` requests.
   */
  legal?: LegalChoiceSet;
}

/**
 * The CloneJudge's verdicts on one request's pending candidates. They hold
 * hidden information (e.g. whether a foe's Shadow Tag traps), so they are
 * an evaluation artifact, never an actor input.
 */
export interface HiddenVerdicts {
  player: Player;
  request_seq: number;
  verdicts: Verdict[];
}

/**
 * The omniscient sim log: both sides' hidden information. A distinct type,
 * so it can never be passed where a PlayerChannel is expected (ADR-008 §5).
 */
export type OmniscientLog = string[] & { readonly __brand: "OmniscientLog" };

export interface BattleRecord {
  manifest: VersionManifest;
  /** The rules the battle ran under, embedded so replay is self-contained. */
  format_rules: FormatRulesRecord;
  /** Provenance only: the rules snapshot the format rules came from. */
  rules_snapshot_hash: string;
  seed: string;
  /** `p1`, `p2`, or `null` for a tie. */
  winner: Player | null;
  turns: number;
  /**
   * Omniscient sim log with wall-clock `|t:|` lines removed. It holds both
   * sides' hidden information: an evaluation artifact, never an actor input.
   */
  omniscient_log: OmniscientLog;
  requests: Record<Player, RequestRecord[]>;
  /**
   * Each player's own stream from `getPlayerStreams`, line by line and in
   * order, with `|t:|` dropped as in `omniscient_log`: what that player may
   * see (the engine already split the secrets), its own requests included.
   * The ObservationBuilder's only battle input (ADR-008 §5).
   */
  channels: Record<Player, string[]>;
  /** Every submitted choice, including those refused as `[Unavailable choice]`. */
  choices: { player: Player; request_seq: number; choice: string; refused?: string }[];
  /** Hidden state, like `omniscient_log`: never an actor input. */
  hidden_verdicts: HiddenVerdicts[];
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
 * Replay one player's recorded choices, matched by `request_seq`. Together
 * with the same engine, format, seed and teams this reproduces a battle.
 */
export function replayActor(choices: BattleRecord["choices"], player: Player): Actor {
  const bySeq = new Map(
    choices.filter((c) => c.player === player).map((c) => [c.request_seq, c.choice]),
  );
  return (request) => {
    const choice = bySeq.get(request.request_seq);
    if (choice === undefined) {
      throw new BattleAbortedError(`${player}: no recorded choice for this request`);
    }
    return choice;
  };
}

/**
 * Run one headless battle on the in-process sim (runtime A1). Same engine,
 * format, seed, teams and choices always produce the same `omniscient_log`.
 */
export async function runBattle(options: BattleOptions): Promise<BattleRecord> {
  const { engine, formatId, seed, teams, actors, maxTurns = 300, snapshot } = options;
  const identities = options.players ?? { p1: { name: "p1" }, p2: { name: "p2" } };
  validatePlayers(identities);
  const resolved = resolveFormat(engine, formatId, snapshot);
  for (const player of PLAYERS) {
    const verdict = validateTeam(engine, resolved, teams[player]);
    if (!verdict.valid) throw new InvalidTeamError(player, verdict.problems);
  }

  const battleStream = new engine.sim.BattleStream();
  const streams = engine.sim.getPlayerStreams(battleStream);
  const record: BattleRecord = {
    manifest: resolved.manifest,
    format_rules: resolved.rules,
    rules_snapshot_hash: resolved.rules_snapshot_hash,
    seed,
    winner: null,
    turns: 0,
    omniscient_log: [] as string[] as OmniscientLog,
    requests: { p1: [], p2: [] },
    channels: { p1: [], p2: [] },
    choices: [],
    hidden_verdicts: [],
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
        record.omniscient_log.push(line);
        if (line.startsWith("|turn|")) {
          record.turns = Number(line.slice("|turn|".length));
          if (record.turns > maxTurns) streams.omniscient.write(">forcetie");
        } else if (line.startsWith("|win|")) {
          const winner = line.slice("|win|".length);
          const seat = PLAYERS.find((player) => identities[player].name === winner);
          if (seat === undefined) abort(new Error(`|win| names no seat: ${winner}`));
          else record.winner = seat;
          end();
        } else if (line === "|tie") {
          end();
        }
      }
    }
  };

  const play = async (player: Player, stream: SimStream): Promise<void> => {
    // The last submission if it was a pending candidate, until the engine
    // answers it with a new request.
    let pendingSubmission: BattleRecord["choices"][number] | null = null;
    for await (const chunk of stream) {
      for (const line of chunk.split("\n")) {
        if (!line.startsWith("|t:|")) record.channels[player].push(line);
        if (line.startsWith("|error|")) {
          const error = line.slice("|error|".length);
          // A pending candidate may be refused for a hidden reason (ADR-006
          // rule 5); the updated request that follows is answered afresh.
          if (
            pendingSubmission !== null &&
            judgeSubmission(pendingSubmission.choice, error).revealed
          ) {
            pendingSubmission.refused = error;
            continue;
          }
          abort(new BattleAbortedError(`${player}: ${error}`));
        } else if (line.startsWith("|request|")) {
          pendingSubmission = null;
          const request: RequestRecord = {
            ...decodeRequest(line.slice("|request|".length)),
            request_seq: 0,
          };
          request.request_seq = record.requests[player].push(request);
          if (request.phase === "wait") continue;
          try {
            const battle = battleStream.battle as SimBattle;
            // A request still in the stream when the battle ends (e.g. a
            // forced tie) asks for nothing.
            if (battle.ended) continue;
            const judge = new CloneJudge(engine, battle, player);
            const legal = legalChoiceSet({ request, showdown_commit: engine.commit, judge });
            request.legal = legal;
            if (legal.pending.length > 0) {
              const verdicts = judge.judge(request.request_hash, legal.pending);
              record.hidden_verdicts.push({ player, request_seq: request.request_seq, verdicts });
            }
            const choice = await actors[player](request);
            if (ended) return;
            const entry = { player, request_seq: request.request_seq, choice };
            record.choices.push(entry);
            if (legal.pending.includes(choice)) pendingSubmission = entry;
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
          `>player ${player} ${JSON.stringify({
            name: identities[player].name,
            ...(identities[player].avatar ? { avatar: identities[player].avatar } : {}),
            team: engine.sim.Teams.pack(teams[player]),
          })}`,
      ),
    ].join("\n"),
  );
  await done;
  if (failure) throw failure;
  return record;
}
