// The A1 adapter for a PlayerChannel (ADR-008 §5), on Node: it checks a
// channel log before the ObservationBuilder may read it. The A2 adapter (server
// frames) comes with ADR-005's slices.

import { createHash } from "node:crypto";
import { ajvFor } from "@areneon/schema/ajv";
import battleSchema from "@areneon/schema/contracts/v1/battle-observation.schema.json" with { type: "json" };
import channelSchema from "@areneon/schema/contracts/v1/player-channel-log.schema.json" with { type: "json" };
import previewSchema from "@areneon/schema/contracts/v1/preview-observation.schema.json" with { type: "json" };
import type { Json } from "./canonical-json.ts";
import type { PlayerChannelLog } from "@areneon/schema/types/v1/player-channel-log.ts";
import {
  buildObservation,
  type Observation,
  ObservationError,
  type ObservationInput,
  type PlayerChannel,
  type Sha256,
} from "./observation.ts";
import { requestIdentityText } from "./request-bytes.ts";
import { decodeRequest } from "./request.ts";
import type { BattleRecord, Player } from "./runner.ts";
import type { BattleClass } from "./team-class.ts";

/** The Sha256 port on Node's `createHash` (A1, A2). */
export const nodeSha256: Sha256 = (bytes) => Promise.resolve(createHash("sha256").update(bytes).digest("hex"));

const contract = ajvFor("contract");
const validChannel = contract.compile(channelSchema);
const validPreview = contract.compile(previewSchema);
const validBattle = contract.compile(battleSchema);

/**
 * `buildObservation`, then full validation against the closed observation
 * schema (ADR-008 §5's "validate"), on Node.
 */
export async function observe(input: Omit<ObservationInput, "sha256">): Promise<Observation> {
  const observation = await buildObservation({ ...input, sha256: nodeSha256 });
  const valid = observation.kind === "preview" ? validPreview : validBattle;
  if (!valid(observation)) throw new ObservationError(`invalid observation: ${JSON.stringify(valid.errors)}`);
  return observation;
}

/**
 * A line only the opponent, or the omniscient log, could hold: an unresolved
 * `|split|` marker, or the opponent's exact HP. A player sees the opponent's
 * HP only as a share of 100; the exact value is the opponent's SECRET half.
 * This is the engine's convention for every supported format; a format that
 * reported HP otherwise (exact, or the 48-pixel scale) would fail closed
 * here until this check learned it. On the server path, chat (`|c|`) is
 * dropped by `stripServerLines` before this check (ADR-005 §5), so its free
 * text never trips it.
 */
function secretOfOpponent(line: string, player: string): boolean {
  if (line.startsWith("|split|")) return true;
  const opponent = player === "p1" ? "p2" : "p1";
  const fields = line.split("|");
  return (
    fields.some((f) => f.startsWith(`${opponent}a: `) || f.startsWith(`${opponent}b: `)) &&
    fields.some((f) => /^\d+\/(?!100(?:\s|$))\d+(?:\s|$)/.test(f))
  );
}

/**
 * Brand a channel log as a PlayerChannel once it matches PlayerChannelLog,
 * every `|request|` in it is of the engine's shape and for this player, and
 * no line is the opponent's SECRET half.
 */
export function playerChannel(log: unknown): PlayerChannel {
  if (!validChannel(log)) {
    throw new ObservationError(`not a PlayerChannelLog: ${JSON.stringify(validChannel.errors)}`);
  }
  const channel = log as unknown as PlayerChannelLog;
  for (const line of channel.lines) {
    if (secretOfOpponent(line, channel.player)) {
      throw new ObservationError(`${channel.player}'s channel holds a line only the opponent may see: ${line}`);
    }
    if (!line.startsWith("|request|")) continue;
    const { player } = decodeRequest(line.slice("|request|".length));
    if (player !== channel.player) {
      throw new ObservationError(`${channel.player}'s channel holds a request for ${player}`);
    }
  }
  return channel as PlayerChannel;
}

/**
 * A1: one player's channel of a recorded battle, up to and including its
 * `request_seq`-th request. Its class is the battle's, from both teams
 * (`battleClass`), never a label of the caller's.
 */
export function recordedChannel(
  record: BattleRecord,
  player: Player,
  request_seq: number,
  data_class: BattleClass,
): PlayerChannel {
  const lines = record.channels[player];
  let seen = 0;
  const end = lines.findIndex((line) => line.startsWith("|request|") && ++seen === request_seq);
  if (end < 0) throw new ObservationError(`${player} received no request ${request_seq}`);
  return playerChannel({ schema_version: 1, data_class, player, lines: lines.slice(0, end + 1) });
}

const REQUEST = "|request|";
const ERROR = "|error|";
/** Lines the server sends to the player alone and never adds to the room log it replays on a rejoin. */
const ALONE = [REQUEST, ERROR];

const isAlone = (line: string): boolean => ALONE.some((kind) => line.startsWith(kind));

/** A request line's rqid and its body without the rqid (the bytes `request_hash` covers). */
function identityOf(line: string): { rqid: unknown; body: string } {
  const request = JSON.parse(line.slice(REQUEST.length)) as { [key: string]: Json };
  return { rqid: request.rqid, body: requestIdentityText(request) };
}

/**
 * The channel of a player who reconnected (ADR-008 §3, §4): `earlier`, the
 * channel of the connection(s) before, joined with `rejoin`, the channel of
 * the connection after. This is how an observation is built after a rejoin.
 *
 * A rejoin channel alone has no Team Preview request: the server replays the
 * room log to a joining connection, and that log holds neither `|request|`
 * nor `|error|` lines (it sends those to the player alone and never adds
 * them to the log). So the rejoin's replay lines are the earlier channel's
 * lines and more, minus every player-alone line. The rule: the rejoin channel
 * carries the anchor of the connection it continues. The result is the
 * replay with the earlier channel's player-alone lines put back at the
 * places they had (the count of room lines before each), then the rejoin's
 * own. A first request of the rejoin that is the earlier channel's last
 * request sent again (same `rqid`, same body) is that request, not a new
 * one, and stays where the earlier channel had it. It need not sit at the
 * place the earlier one had: the server keeps each player's last request and
 * sends it after the replay, and the room may have moved on meanwhile (the
 * choice it answered was handled while the connection dropped, and the next
 * request was not yet stored when the player joined). The result is the one continuous channel of the battle:
 * equal to what A1 gives the player, and what an Episode stores (ADR-008 §5
 * and the glossary: one channel per player and battle).
 *
 * It uses only the two channels of this player, already checked by
 * `playerChannel`, and adds nothing of its own: no opponent line, no request
 * the player did not receive. It fails closed (`ObservationError`) unless
 * both are the same player's, the earlier channel's room lines are a
 * prefix of the rejoin's (so a replay of another battle, or a tampered one,
 * is refused), the rejoin's player-alone lines all come after its replay,
 * a re-sent request keeps its `rqid` (same body, other `rqid` at the
 * earlier one's place is refused; same `rqid`, other body is not a re-send
 * and fails the order below) and, on the server path, the `rqid`s of the
 * joined requests rise strictly (the server's one counter numbers every
 * request of both players, waits included)
 * present on every request. Across several reconnects, fold it: the result of one
 * join is the `earlier` of the next. The rqids are the server's (§4 leaves
 * them out of A1/A2 equality) and the request count follows the player's
 * own view.
 */
export function continueChannel(earlier: PlayerChannel, rejoin: PlayerChannel): PlayerChannel {
  if (earlier.player !== rejoin.player) {
    throw new ObservationError(`cannot continue ${earlier.player}'s channel with ${rejoin.player}'s`);
  }
  const room = (lines: readonly string[]): string[] => lines.filter((line) => !isAlone(line));
  const alone = (lines: readonly string[]): { at: number; line: string }[] => {
    let at = 0;
    return lines.flatMap((line) => (isAlone(line) ? [{ at, line }] : (at++, [])));
  };
  const before = room(earlier.lines);
  const replay = room(rejoin.lines);
  if (before.length > replay.length || before.some((line, i) => line !== replay[i])) {
    throw new ObservationError("the rejoin channel does not replay the earlier channel's room log");
  }
  const mine = alone(earlier.lines);
  const theirs = alone(rejoin.lines);
  const sent = mine.findLast((entry) => entry.line.startsWith(REQUEST));
  // The server sends a rejoin's player-alone lines after the whole replay, so
  // none of them can sit inside the room lines the player already had.
  if (theirs.some((entry) => entry.at < before.length)) {
    throw new ObservationError("the rejoin channel has a request or error line inside the room log the player already had");
  }
  const again = theirs[0];
  let dropped = 0;
  if (again?.line.startsWith(REQUEST) === true && sent !== undefined) {
    const a = identityOf(again.line);
    const b = identityOf(sent.line);
    const same = a.body === b.body;
    if (same && a.rqid !== undefined && a.rqid === b.rqid) {
      // The server keeps each player's last request and sends it again on a join, after the
      // replay. The room may have moved on since the player got it (the choice it answered
      // was handled while the connection dropped), so the place does not tell: the rqid and
      // the body do. The earlier channel's copy stays where the player had it.
      theirs.shift();
      dropped = 1;
    } else if (same && again.at === sent.at && (a.rqid !== undefined || b.rqid !== undefined)) {
      throw new ObservationError("the rejoin's first request has the body of the earlier last request but another rqid");
    }
  }
  const lines: string[] = [];
  const put = (list: { at: number; line: string }[], at: number): void => {
    for (const entry of list) if (entry.at === at) lines.push(entry.line);
  };
  replay.forEach((line, i) => {
    put(mine, i);
    put(theirs, i);
    lines.push(line);
  });
  put(mine, replay.length);
  put(theirs, replay.length);
  const requests = lines.filter((line) => line.startsWith(REQUEST));
  const expected = earlier.lines.filter((line) => line.startsWith(REQUEST)).length +
    rejoin.lines.filter((line) => line.startsWith(REQUEST)).length - dropped;
  if (requests.length !== expected) {
    throw new ObservationError("the joined channel's request sequence is not contiguous");
  }
  // `rqid` exists only on the server path (A2): an A1 channel has none, and a
  // channel must not mix the two.
  const rqids = requests.map((line) => identityOf(line).rqid);
  if (rqids.some((rqid) => rqid !== undefined)) {
    let last = -Infinity;
    for (const rqid of rqids) {
      if (typeof rqid !== "number" || !(rqid > last)) {
        throw new ObservationError("the rqid of the joined requests must be present and rise strictly");
      }
      last = rqid;
    }
  }
  return playerChannel({
    schema_version: 1,
    data_class: earlier.data_class === "private" || rejoin.data_class === "private" ? "private" : "synthetic",
    player: earlier.player,
    lines,
  });
}
