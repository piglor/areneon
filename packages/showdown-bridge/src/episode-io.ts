// Writing and reading Episodes on Node (ADR-008 §3, §5): from an A1 battle
// record, and back from stored JSON, validated against the schema of the
// version it was written with.

import { ajvFor } from "@areneon/schema/ajv";
import episodeSchema from "@areneon/schema/contracts/v1/episode.schema.json" with { type: "json" };
import type { PrivateTeamSpec } from "@areneon/schema/types/v1/private-team-spec.ts";
import { type Episode, rightsFor } from "./episode.ts";
import { buildObservation } from "./observation.ts";
import { nodeSha256, playerChannel } from "./player-channel.ts";
import type { BattleRecord, Player } from "./runner.ts";

export class EpisodeError extends Error {
  override name = "EpisodeError";
}

/** One validator per schema_version an Episode can be stored in. */
const READERS: Readonly<Record<number, ReturnType<ReturnType<typeof ajvFor>["compile"]>>> = {
  1: ajvFor("contract").compile(episodeSchema),
};

/**
 * Read a stored Episode. It must match the schema of its own
 * `schema_version`, and its rights must go with its data class. Every
 * decision's observation is rebuilt from the channel by the
 * ObservationBuilder (so the team must be the player's own) and must have
 * the decision's identity. A version with no reader is refused; upcasting to
 * a newer version is a read-time view added with that version (ADR-008 §3).
 */
export async function readEpisode(value: unknown): Promise<Episode> {
  const version = (value as { schema_version?: unknown } | null)?.schema_version;
  const validate = typeof version === "number" ? READERS[version] : undefined;
  if (!validate) throw new EpisodeError(`no reader for an Episode with schema_version ${JSON.stringify(version)}`);
  if (!validate(value)) throw new EpisodeError(`invalid Episode: ${JSON.stringify(validate.errors)}`);
  const episode = value as Episode;
  if (episode.rights !== rightsFor(episode.data_class)) {
    throw new EpisodeError(`${episode.data_class} data cannot carry ${episode.rights} rights`);
  }
  if (episode.team.data_class === "private" && episode.data_class !== "private") {
    throw new EpisodeError("an Episode with a private team is private");
  }
  const channel = playerChannel({
    schema_version: 1,
    data_class: episode.data_class,
    player: episode.player,
    lines: episode.channel,
  });
  for (const { identity } of episode.decisions) {
    let seen = 0;
    const end = channel.lines.findIndex((line) => line.startsWith("|request|") && ++seen === identity.request_seq);
    const observation =
      end < 0
        ? undefined
        : await buildObservation({
            channel: { ...channel, lines: channel.lines.slice(0, end + 1) },
            team: episode.team,
            manifest: episode.provenance.manifest,
            sha256: nodeSha256,
          }).catch((error: unknown) => {
            throw new EpisodeError(`decision ${identity.request_seq}: ${(error as Error).message}`);
          });
    if (observation?.identity.request_hash !== identity.request_hash) {
      throw new EpisodeError(`decision ${identity.request_seq} does not answer that request of the channel`);
    }
  }
  return episode;
}

export interface EpisodeOptions {
  /** The player's own team, as they brought it. */
  team: PrivateTeamSpec;
  behaviour_policy_id: string;
}

/**
 * A1: one player's Episode of a recorded battle. Only that player's channel,
 * requests and choices go in; A1 has no timer, so no decision carries a
 * deadline record.
 */
export function episodeFromRecord(
  record: BattleRecord,
  player: Player,
  { team, behaviour_policy_id }: EpisodeOptions,
): Promise<Episode> {
  const decisions = record.choices
    .filter((choice) => choice.player === player)
    .map(({ request_seq, choice, refused }) => {
      const request = record.requests[player][request_seq - 1] as BattleRecord["requests"][Player][number];
      return {
        identity: { request_seq, request_hash: request.request_hash },
        choice,
        ...(refused === undefined ? {} : { refused }),
        source_policy_id: behaviour_policy_id,
      };
    });
  return readEpisode({
    schema_version: 1,
    data_class: team.data_class,
    rights: rightsFor(team.data_class),
    provenance: { runtime: "A1", behaviour_policy_id, manifest: record.manifest },
    player,
    team,
    channel: record.channels[player],
    decisions,
    result: { winner: record.winner, turns: record.turns },
  });
}
