// The A1 adapter for a PlayerChannel (ADR-008 §5), on Node: it checks a
// channel log before the ObservationBuilder may read it. The A2 and advisor
// adapters (server frames) come with ADR-005's and ADR-004's slices.

import { createHash } from "node:crypto";
import { ajvFor } from "@areneon/schema/ajv";
import channelSchema from "@areneon/schema/contracts/v1/player-channel-log.schema.json" with { type: "json" };
import type { PlayerChannelLog } from "@areneon/schema/types/v1/player-channel-log.ts";
import { ObservationError, type PlayerChannel, type Sha256 } from "./observation.ts";
import { decodeRequest } from "./request.ts";
import type { BattleRecord, Player } from "./runner.ts";

/** The Sha256 port on Node's `createHash` (A1, A2). */
export const nodeSha256: Sha256 = (bytes) => Promise.resolve(createHash("sha256").update(bytes).digest("hex"));

const validChannel = ajvFor("contract").compile(channelSchema);

/**
 * Brand a channel log as a PlayerChannel once it matches PlayerChannelLog
 * and every `|request|` in it is of the engine's shape and for this player.
 */
export function playerChannel(log: unknown): PlayerChannel {
  if (!validChannel(log)) {
    throw new ObservationError(`not a PlayerChannelLog: ${JSON.stringify(validChannel.errors)}`);
  }
  const channel = log as unknown as PlayerChannelLog;
  for (const line of channel.lines) {
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
 * `request_seq`-th request.
 */
export function recordedChannel(
  record: BattleRecord,
  player: Player,
  request_seq: number,
  data_class: PlayerChannelLog["data_class"],
): PlayerChannel {
  const lines = record.channels[player];
  let seen = 0;
  const end = lines.findIndex((line) => line.startsWith("|request|") && ++seen === request_seq);
  if (end < 0) throw new ObservationError(`${player} received no request ${request_seq}`);
  return playerChannel({ schema_version: 1, data_class, player, lines: lines.slice(0, end + 1) });
}
