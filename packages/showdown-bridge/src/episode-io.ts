// Writing and reading Episodes on Node (ADR-008 §3, §5): from an A1 battle
// record, and back from stored JSON, validated against the schema of the
// version it was written with.

import { ajvFor } from "@areneon/schema/ajv";
import episodeSchema from "@areneon/schema/contracts/v1/episode.schema.json" with { type: "json" };
import type { PrivateTeamSpec } from "@areneon/schema/types/v1/private-team-spec.ts";
import { type Episode, rightsFor } from "./episode.ts";
import { buildObservation, type PlayerChannel } from "./observation.ts";
import { nodeSha256, playerChannel } from "./player-channel.ts";
import type { BattleRecord, Player } from "./runner.ts";
import { higherClass, type SyntheticTeamHashes, teamHash } from "./team-class.ts";

export class EpisodeError extends Error {
  override name = "EpisodeError";
}

/** One validator per schema_version an Episode can be stored in. */
const READERS: Readonly<Record<number, ReturnType<ReturnType<typeof ajvFor>["compile"]>>> = {
  1: ajvFor("contract").compile(episodeSchema),
};

/** How the channel says the battle ended: its `|win|` or `|tie` line and last turn. */
function resultOf(lines: readonly string[]): Episode["result"] {
  const names = new Map<string, "p1" | "p2">();
  let turns = 0;
  let winner: Episode["result"]["winner"] | undefined;
  for (const line of lines) {
    const [, kind, a, b] = line.split("|");
    if (kind === "player" && (a === "p1" || a === "p2") && b) names.set(b, a);
    else if (kind === "turn") turns = Number(a);
    else if (kind === "win") {
      const seat = names.get(a as string) ?? (a === "p1" || a === "p2" ? a : undefined);
      if (seat === undefined) throw new EpisodeError(`the channel's winner ${a} is not one of its players`);
      winner = seat;
    }
    else if (kind === "tie") winner = null;
  }
  if (winner === undefined) throw new EpisodeError("the channel does not end the battle");
  return { winner, turns };
}

/** The channel cut after its `seq`-th request. */
function cut(channel: PlayerChannel, seq: number): PlayerChannel | undefined {
  let seen = 0;
  const end = channel.lines.findIndex((line) => line.startsWith("|request|") && ++seen === seq);
  return end < 0 ? undefined : { ...channel, lines: channel.lines.slice(0, end + 1) };
}

/**
 * Read a stored Episode, as a copy no later change to `value` reaches.
 * - It must match the schema of its own `schema_version`. A version with no
 *   reader is refused; upcasting to a newer version is a read-time view
 *   added with that version (ADR-008 §3).
 * - Its rights must go with its data class, and a private team makes it
 *   private.
 * - Its channel must be the player's own: own requests, no opponent SECRET
 *   line. Its result must be the channel's own ending.
 * - Its team must be the player's own. This is checked by building the Team
 *   Preview observation, then every decision's, which must carry the
 *   decision's identity, in request order.
 * - On A2, every decision carries its deadline record and timer context, the
 *   never-auto-choose evidence ADR-005 needs.
 *
 * - A team labelled `synthetic` must hash to a member of `known_synthetic`
 *   (required: omitting it would leave the class unchecked). The stored Episode does not name the opponent's team or class, so
 *   the opponent half of the both-teams rule is checked when it is written
 *   (`episodeFromRecord`), not here.
 *
 * Every refusal is an EpisodeError.
 */
export async function readEpisode(
  value: unknown,
  { known_synthetic }: { known_synthetic: SyntheticTeamHashes },
): Promise<Episode> {
  try {
    const episode = await read(structuredClone(value));
    assertProven(episode.team, known_synthetic);
    return episode;
  } catch (error) {
    if (error instanceof EpisodeError) throw error;
    throw new EpisodeError(`invalid Episode: ${(error as Error).message}`);
  }
}

async function read(value: unknown): Promise<Episode> {
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
  const channel = playerChannel({ schema_version: 1, data_class: episode.data_class, player: episode.player, lines: episode.channel });
  const ended = resultOf(channel.lines);
  if (ended.winner !== episode.result.winner || ended.turns !== episode.result.turns) {
    throw new EpisodeError(`the result ${JSON.stringify(episode.result)} is not the channel's ${JSON.stringify(ended)}`);
  }
  const observe = (prefix: PlayerChannel) =>
    buildObservation({ channel: prefix, team: episode.team, manifest: episode.provenance.manifest, sha256: nodeSha256 });
  const preview = cut(channel, 1);
  if (!preview) throw new EpisodeError("the channel holds no request");
  await observe(preview);
  let last = 0;
  for (const decision of episode.decisions) {
    const { identity } = decision;
    if (identity.request_seq <= last) throw new EpisodeError(`decision ${identity.request_seq} is out of request order`);
    last = identity.request_seq;
    const prefix = cut(channel, identity.request_seq);
    const observation = prefix && (await observe(prefix));
    if (observation?.identity.request_hash !== identity.request_hash) {
      throw new EpisodeError(`decision ${identity.request_seq} does not answer that request of the channel`);
    }
    if (episode.provenance.runtime === "A2" && (!decision.deadline || !decision.decision_context)) {
      throw new EpisodeError(`A2 decision ${identity.request_seq} lacks its deadline record or timer context`);
    }
  }
  return episode;
}

/** A team labelled `synthetic` must be one the source knows is synthetic. */
function assertProven(team: PrivateTeamSpec, known: SyntheticTeamHashes): void {
  if (team.data_class === "synthetic" && !known.has(teamHash(team.team))) {
    throw new EpisodeError("a team labelled synthetic is not a known synthetic team");
  }
}

export interface EpisodeOptions {
  /** The player's own team, as they brought it. */
  team: PrivateTeamSpec;
  /** The opponent's team as the battle ran with it, for its class only: it is never stored. */
  opponent_team: PrivateTeamSpec;
  /** The synthetic teams there are (hashes); a `synthetic` label outside it is refused. */
  known_synthetic: SyntheticTeamHashes;
  behaviour_policy_id: string;
}

/**
 * A1: one player's Episode of a recorded battle. Only that player's channel,
 * requests and choices go in; A1 has no timer, so no decision carries a
 * deadline record.
 *
 * The class is the higher of BOTH teams' (ADR-008 §5): the stored channel
 * shows what the opponent revealed. Each team must hash to the entry the
 * battle ran with (`record.team_hashes`), and a `synthetic` label must be
 * proven by `known_synthetic`; anything else is refused.
 */
export async function episodeFromRecord(
  record: BattleRecord,
  player: Player,
  { team, opponent_team, known_synthetic, behaviour_policy_id }: EpisodeOptions,
): Promise<Episode> {
  const rival = player === "p1" ? "p2" : "p1";
  for (const [who, spec] of [[player, team], [rival, opponent_team]] as const) {
    if (teamHash(spec.team) !== record.team_hashes[who]) {
      throw new EpisodeError(`the team given for ${who} is not the one ${who} played the battle with`);
    }
    assertProven(spec, known_synthetic);
  }
  const data_class = higherClass(team.data_class, opponent_team.data_class);
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
  // Read back through the reader, on a copy: the record stays the caller's.
  return await readEpisode({
    schema_version: 1,
    data_class,
    rights: rightsFor(data_class),
    provenance: { runtime: "A1", behaviour_policy_id, manifest: record.manifest },
    player,
    team,
    channel: record.channels[player],
    decisions,
    result: { winner: record.winner, turns: record.turns },
  }, { known_synthetic });
}
