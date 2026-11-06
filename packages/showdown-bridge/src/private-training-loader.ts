// The private training loader (ADR-015 erratum E1 items 5 and 6, #1062): the one way a PRIVATE run's own
// Episodes become a `private` TrainingSet. It is a separate entry used only by the private run driver;
// `loadTrainingSet` is unchanged and keeps refusing every private Episode (the public run never sees one).
//
// It accepts only `private` / `owner-private` `league` Episodes of the private run (`collection.run_id` equals the
// run key), read through `readEpisode` (so the class floor and every other Episode rule hold). It refuses
// `evaluation` and `sparring` Episodes, a missing collection record, an Episode of any other run, a synthetic
// Episode (a private run's Episodes are all private: the writer takes a class floor), an Episode played by another
// policy than the one being trained, a hash an EvaluationRunManifest lists, a repeated Episode or manifest, and an
// Episode whose own or opponent team is in the hold-out set (the caller supplies its hashes; the hold-out set's
// construction is not here).
//
// Every refusal is a `PrivateTrainingLoaderError` with a code and a FIXED message: it names at most the position of
// the Episode in the caller's list and never a team, a species, a hash, a run key or a text of the Episode, so
// nothing private reaches a log through a refusal (the canary test). The stream and TrainingSet bytes are those of
// the public loader's writer, with `data_class` private.

import type { PokemonSet } from "./engine.ts";
import { jcs, type Json } from "./canonical-json.ts";
import { type Episode, episodeHash } from "./episode.ts";
import { readEpisode } from "./episode-io.ts";
import { nodeSha256 } from "./player-channel.ts";
import { canonicalTeamHash } from "./team-records.ts";
import { teamHash } from "./team-class.ts";
import type { ChunkLimits } from "./chunked-file.ts";
import {
  type EpisodeFeed,
  readEvaluationRunManifest,
  streamTraining,
  type TrainingExample,
  type TrainingInput,
  type TrainingSet,
  type TrainingStream,
} from "./training-loader.ts";

/** Why the private loader refused; the message of each is fixed. */
export const PRIVATE_LOADER_REFUSALS = {
  episode_invalid: "an Episode is refused by the Episode reader",
  no_collection: "an Episode has no collection record",
  evaluation_episode: "an evaluation Episode never trains",
  sparring_episode: "a sparring Episode never trains",
  not_private: "a private run trains on private Episodes only",
  other_run: "an Episode belongs to another run",
  other_policy: "an Episode was not played by the policy being trained",
  holdout_team: "an Episode holds a team of the hold-out set",
  opponent_hash_invalid: "an opponent team hash is not a SHA-256 digest",
  listed_by_manifest: "an EvaluationRunManifest lists an Episode",
  repeated_episode: "an Episode is given twice",
  manifest_invalid: "an EvaluationRunManifest is invalid",
  manifest_repeated: "an EvaluationRunManifest is given twice",
  run_key_invalid: "the private run key is not an id token",
  no_episode: "there is no Episode to load",
  stream_too_long: "the stream does not fit one string",
} as const;

export type PrivateLoaderRefusal = keyof typeof PRIVATE_LOADER_REFUSALS;

export class PrivateTrainingLoaderError extends Error {
  override name = "PrivateTrainingLoaderError";
  readonly code: PrivateLoaderRefusal;
  /** The 0-based position in the caller's Episode (or manifest) list, when the refusal is about one. */
  readonly position: number | null;
  constructor(code: PrivateLoaderRefusal, position: number | null = null) {
    super(position === null ? PRIVATE_LOADER_REFUSALS[code] : `${PRIVATE_LOADER_REFUSALS[code]} (position ${position})`);
    this.code = code;
    this.position = position;
  }
}

/** One stored Episode and the hash of the team the opponent played it against (an Episode does not store it). */
export interface PrivateEpisodeInput {
  /** The Episode as stored: checked, never trusted. */
  episode: unknown;
  /** The opponent's team hash, positional (`teamHash`) or canonical (`canonicalTeamHash`), for the hold-out check. */
  opponent_team_hash: string;
}

export interface PrivateTrainingInput extends Omit<TrainingInput, "episodes"> {
  episodes: readonly PrivateEpisodeInput[];
  /** The private run's key: every Episode's `collection.run_id` must equal it. */
  run_key: string;
  /**
   * Hashes of the hold-out teams (positional or canonical). An Episode whose own team or whose opponent's team hashes
   * into the set is refused. Built by the hold-out ticket, supplied by the caller; empty means no hold-out.
   */
  holdout_team_hashes: ReadonlySet<string>;
}

const ID_TOKEN = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
const DIGEST = /^[0-9a-f]{64}$/;

/**
 * A private run's TrainingSet and stream from its own Episodes. `known_synthetic` is `readEpisode`'s requirement: a team
 * labelled synthetic must hash into it, whatever the run's class. See the file comment for the refusals.
 */
export async function loadPrivateTrainingSet(input: PrivateTrainingInput, limits: ChunkLimits = {}): Promise<TrainingStream> {
  if (!ID_TOKEN.test(input.run_key)) throw new PrivateTrainingLoaderError("run_key_invalid");
  const manifests = new Set<string>();
  for (const [index, value] of input.evaluation_manifests.entries()) {
    let digest: string;
    try {
      digest = await nodeSha256(new TextEncoder().encode(jcs(readEvaluationRunManifest(value) as unknown as Json)));
    } catch {
      throw new PrivateTrainingLoaderError("manifest_invalid", index);
    }
    if (manifests.has(digest)) throw new PrivateTrainingLoaderError("manifest_repeated", index);
    manifests.add(digest);
  }
  const feed = privateFeed(input);
  const examples: TrainingExample[] = [];
  const pieces: string[] = [];
  let chars = 0;
  const set: TrainingSet = await streamTraining(
    input,
    feed,
    limits,
    (example, line) => {
      examples.push(example);
      pieces.push(line);
      chars += line.length;
      // The stream as one string must fit one (as for the public loader); a longer stream is written in chunks, not here.
      if (limits.max_string_chars !== undefined && chars > limits.max_string_chars) {
        throw new PrivateTrainingLoaderError("stream_too_long");
      }
    },
    "private",
  );
  return { examples, stream: pieces.join(""), set };
}

function privateFeed(input: PrivateTrainingInput): EpisodeFeed {
  return async function* (evaluated) {
    const accepted = new Map<string, Episode>();
    for (const [index, item] of input.episodes.entries()) {
      const episode = await readPrivate(item, index, input);
      const hash = await episodeHash(episode, nodeSha256);
      if (evaluated.has(hash)) throw new PrivateTrainingLoaderError("listed_by_manifest", index);
      if (accepted.has(hash)) throw new PrivateTrainingLoaderError("repeated_episode", index);
      accepted.set(hash, episode);
    }
    if (accepted.size === 0) throw new PrivateTrainingLoaderError("no_episode");
    for (const hash of [...accepted.keys()].sort()) yield [hash, accepted.get(hash) as Episode];
  };
}

async function readPrivate(item: PrivateEpisodeInput, index: number, input: PrivateTrainingInput): Promise<Episode> {
  const episode = await readEpisode(item.episode, { known_synthetic: input.known_synthetic }).catch(() => {
    throw new PrivateTrainingLoaderError("episode_invalid", index); // the reader's text is dropped: it can quote a value
  });
  const collection = episode.provenance.collection;
  if (collection === undefined) throw new PrivateTrainingLoaderError("no_collection", index);
  if (collection.kind === "evaluation") throw new PrivateTrainingLoaderError("evaluation_episode", index);
  if (collection.kind === "sparring") throw new PrivateTrainingLoaderError("sparring_episode", index);
  if (episode.data_class !== "private" || episode.rights !== "owner-private") throw new PrivateTrainingLoaderError("not_private", index);
  if (collection.run_id !== input.run_key) throw new PrivateTrainingLoaderError("other_run", index);
  if (episode.provenance.behaviour_policy_id !== input.behaviour_policy_id) throw new PrivateTrainingLoaderError("other_policy", index);
  if (typeof item.opponent_team_hash !== "string" || !DIGEST.test(item.opponent_team_hash)) {
    throw new PrivateTrainingLoaderError("opponent_hash_invalid", index);
  }
  const own = episode.team.team as unknown as PokemonSet[];
  const hashes = [teamHash(own), canonicalTeamHash(own), item.opponent_team_hash];
  if (hashes.some((hash) => input.holdout_team_hashes.has(hash))) throw new PrivateTrainingLoaderError("holdout_team", index);
  return episode;
}
