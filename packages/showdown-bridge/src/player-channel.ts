// The A1 adapter for a PlayerChannel (ADR-008 §5), on Node: it checks a
// channel log before the ObservationBuilder may read it. The A2 and advisor
// adapters (server frames) come with ADR-005's and ADR-004's slices.

import { createHash } from "node:crypto";
import { ajvFor } from "@areneon/schema/ajv";
import battleSchema from "@areneon/schema/contracts/v1/battle-observation.schema.json" with { type: "json" };
import channelSchema from "@areneon/schema/contracts/v1/player-channel-log.schema.json" with { type: "json" };
import previewSchema from "@areneon/schema/contracts/v1/preview-observation.schema.json" with { type: "json" };
import type { PlayerChannelLog } from "@areneon/schema/types/v1/player-channel-log.ts";
import {
  buildObservation,
  type Observation,
  ObservationError,
  type ObservationInput,
  type PlayerChannel,
  type Sha256,
} from "./observation.ts";
import { decodeRequest } from "./request.ts";
import type { BattleRecord, Player } from "./runner.ts";

/** The Sha256 port on Node's `createHash` (A1, A2). */
export const nodeSha256: Sha256 = (bytes) => Promise.resolve(createHash("sha256").update(bytes).digest("hex"));

const contract = ajvFor("contract");
const validChannel = contract.compile(channelSchema);
const validPreview = contract.compile(previewSchema);
const validBattle = contract.compile(battleSchema);

/**
 * `buildObservation`, then full validation against the closed observation
 * schema (ADR-008 §5's "validate"), on Node. The bundled advisor path needs
 * precompiled validators instead, since MV3 forbids Ajv's runtime compile.
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
