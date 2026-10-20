// Writing and reading Episodes on Node (ADR-008 §3, §5): from an A1 battle
// record, and back from stored JSON, validated against the schema of the
// version it was written with.

import { ajvFor } from "@areneon/schema/ajv";
import episodeSchema from "@areneon/schema/contracts/v1/episode.schema.json" with { type: "json" };
import type { PrivateTeamSpec } from "@areneon/schema/types/v1/private-team-spec.ts";
import { type CollectionInput, type Episode, rightsFor } from "./episode.ts";
import { buildObservation, type Observation, type PlayerChannel } from "./observation.ts";
import { nodeSha256, playerChannel } from "./player-channel.ts";
import type { BattleRecord, Player } from "./runner.ts";
import { higherClass, type StampedTeam, type SyntheticTeamHashes, teamHash } from "./team-class.ts";

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

/** The player's observation at their `seq`-th request, built from that prefix of their own channel. */
async function observeAt(episode: Episode, channel: PlayerChannel, seq: number): Promise<Observation | undefined> {
  const prefix = cut(channel, seq);
  return prefix && buildObservation({ channel: prefix, team: episode.team, manifest: episode.provenance.manifest, sha256: nodeSha256 });
}

/**
 * The observation of each decision of a stored Episode, in decision order,
 * rebuilt from the Episode's own channel. Call it on an Episode `readEpisode`
 * has checked: that is what guarantees every decision has its request.
 */
export async function decisionObservations(episode: Episode): Promise<Observation[]> {
  const channel = playerChannel({ schema_version: 1, data_class: episode.data_class, player: episode.player, lines: episode.channel });
  const out: Observation[] = [];
  for (const { identity } of episode.decisions) out.push((await observeAt(episode, channel, identity.request_seq)) as Observation);
  return out;
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
 * - A `collection` record (ADR-009 §4) is checked against the class: the
 *   Episode's data_class is the higher of its team's and the record's
 *   `opponent_team_data_class` (this refuses a stored Episode relabelled
 *   synthetic), and a sparring Episode is private.
 * - It is also checked against the runtime, which the schema cannot: a
 *   `league` or `evaluation` battle on A1 has the seed it ran with, and one
 *   on A2 has none (a challenge cannot be seeded).
 *
 * - A team labelled `synthetic` must hash to a member of `known_synthetic`
 *   (required: omitting it would leave the class unchecked). The stored Episode
 *   does not name the opponent's team, so the opponent's half of the rule
 *   is checked here only through `opponent_team_data_class`, which
 *   `episodeFromRecord` sets from the opponent's stamped team. An Episode with
 *   no collection record has no opponent class to check.
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
  assertCollection(episode);
  const channel = playerChannel({ schema_version: 1, data_class: episode.data_class, player: episode.player, lines: episode.channel });
  const ended = resultOf(channel.lines);
  if (ended.winner !== episode.result.winner || ended.turns !== episode.result.turns) {
    throw new EpisodeError(`the result ${JSON.stringify(episode.result)} is not the channel's ${JSON.stringify(ended)}`);
  }
  const preview = cut(channel, 1);
  if (!preview) throw new EpisodeError("the channel holds no request");
  await observeAt(episode, channel, 1);
  let last = 0;
  for (const decision of episode.decisions) {
    const { identity } = decision;
    if (identity.request_seq <= last) throw new EpisodeError(`decision ${identity.request_seq} is out of request order`);
    last = identity.request_seq;
    const observation = await observeAt(episode, channel, identity.request_seq);
    if (observation?.identity.request_hash !== identity.request_hash) {
      throw new EpisodeError(`decision ${identity.request_seq} does not answer that request of the channel`);
    }
    if (episode.provenance.runtime === "A2" && (!decision.deadline || !decision.decision_context)) {
      throw new EpisodeError(`A2 decision ${identity.request_seq} lacks its deadline record or timer context`);
    }
  }
  return episode;
}

/**
 * What a collection record must agree with: the Episode's class (the higher of
 * the own team's and the opponent's recorded class; a sparring battle against
 * a human is private) and the runtime that played the battle (its seed).
 */
function assertCollection({ data_class, team, provenance: { runtime, collection } }: Episode): void {
  if (collection === undefined) return;
  if (collection.kind === "sparring") {
    if (data_class !== "private") throw new EpisodeError("a sparring Episode is private: it records a battle against a human");
    return;
  }
  const expected = higherClass(team.data_class, collection.opponent_team_data_class);
  if (data_class !== expected) {
    throw new EpisodeError(
      `the Episode is labelled ${data_class}, but its own team is ${team.data_class} and its opponent's team is ${collection.opponent_team_data_class}: the class must be ${expected}`,
    );
  }
  if (runtime === "A1" && collection.seed === null) {
    throw new EpisodeError(`an A1 ${collection.kind} collection record needs the seed the battle ran with`);
  }
  if (runtime === "A2" && collection.seed !== null) {
    throw new EpisodeError(`an A2 ${collection.kind} collection record has no seed: a challenge cannot be seeded`);
  }
}

/** A team labelled `synthetic` must be one the source knows is synthetic. */
function assertProven(team: PrivateTeamSpec, known: SyntheticTeamHashes): void {
  if (team.data_class === "synthetic" && !known.has(teamHash(team.team))) {
    throw new EpisodeError("a team labelled synthetic is not a known synthetic team");
  }
}

export interface EpisodeOptions {
  /** The player's own team, as they brought it. */
  team: StampedTeam;
  /** The opponent's team as the battle ran with it, for its class only: it is never stored. */
  opponent_team: StampedTeam;
  /** The synthetic teams there are (hashes); a `synthetic` label outside it is refused. */
  known_synthetic: SyntheticTeamHashes;
  behaviour_policy_id: string;
  /**
   * How the battle was collected, when it is known (ADR-009 §4). A `league` or
   * `evaluation` record carries the battle's own seed (`record.seed`); its
   * `opponent_team_data_class` is not the caller's to give: it is set from
   * `opponent_team`.
   */
  collection?: CollectionInput;
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
  { team, opponent_team, known_synthetic, behaviour_policy_id, collection }: EpisodeOptions,
): Promise<Episode> {
  if (record.forfeit !== undefined) {
    throw new EpisodeError("a forfeited game is not an Episode: it is not training data");
  }
  const rival = player === "p1" ? "p2" : "p1";
  for (const [who, spec] of [[player, team], [rival, opponent_team]] as const) {
    if (teamHash(spec.team) !== record.team_hashes[who]) {
      throw new EpisodeError(`the team given for ${who} is not the one ${who} played the battle with`);
    }
    assertProven(spec, known_synthetic);
  }
  if (collection && collection.kind !== "sparring" && collection.seed !== record.seed) {
    throw new EpisodeError("the collection record's seed is not the seed the battle ran with");
  }
  const stored =
    collection && collection.kind !== "sparring"
      ? { ...collection, opponent_team_data_class: opponent_team.data_class }
      : collection;
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
    provenance: { runtime: "A1", behaviour_policy_id, manifest: record.manifest, ...(stored && { collection: stored }) },
    player,
    team,
    channel: record.channels[player],
    decisions,
    result: { winner: record.winner, turns: record.turns },
  }, { known_synthetic });
}
