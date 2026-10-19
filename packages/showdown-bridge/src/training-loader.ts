// The training loader (ADR-009 §4): the one way an Episode becomes training
// data. It fails closed. It reads each Episode through `readEpisode` (the
// schema, the class and rights pair, the player's own channel and team, the
// collection rules) and then accepts only a synthetic `league` Episode that
// no EvaluationRunManifest lists. Every refusal is a TrainingLoaderError;
// nothing is skipped, so a breach of the evaluation/training split is visible.
// It writes a stream of TrainingExample lines (no provenance, so no seed or
// opponent id can reach a model) and the TrainingSet manifest that Python
// checks against the stream.

import { ajvFor } from "@areneon/schema/ajv";
import evaluationRunManifestSchema from "@areneon/schema/contracts/v1/evaluation-run-manifest.schema.json" with { type: "json" };
import trainingSetSchema from "@areneon/schema/contracts/v1/training-set.schema.json" with { type: "json" };
import type { EvaluationRunManifest } from "@areneon/schema/types/v1/evaluation-run-manifest.ts";
import type { TrainingExample } from "@areneon/schema/types/v1/training-example.ts";
import type { TrainingSet } from "@areneon/schema/types/v1/training-set.ts";
import { jcs, type Json } from "./canonical-json.ts";
import { type Episode, episodeHash } from "./episode.ts";
import { decisionObservations, readEpisode } from "./episode-io.ts";
import { nodeSha256 } from "./player-channel.ts";
import type { SyntheticTeamHashes } from "./team-class.ts";

export type { EvaluationRunManifest, TrainingExample, TrainingSet };

export class TrainingLoaderError extends Error {
  override name = "TrainingLoaderError";
}

const validManifest = ajvFor("contract").compile(evaluationRunManifestSchema);
const validSet = ajvFor("contract").compile(trainingSetSchema);

/** True when `hashes` is strictly ascending: sorted and without a repeat. */
const strictlyAscending = (hashes: readonly string[]): boolean => hashes.every((h, i) => i === 0 || (hashes[i - 1] as string) < h);

/**
 * Read an EvaluationRunManifest: the schema, then what the profile cannot
 * say (it has no uniqueItems): the hashes are sorted and unique.
 */
export function readEvaluationRunManifest(value: unknown): EvaluationRunManifest {
  if (!validManifest(value)) throw new TrainingLoaderError(`invalid EvaluationRunManifest: ${JSON.stringify(validManifest.errors)}`);
  const manifest = structuredClone(value) as unknown as EvaluationRunManifest;
  if (!strictlyAscending(manifest.episode_hashes)) {
    throw new TrainingLoaderError(`the EvaluationRunManifest of run ${manifest.run_id} must list its Episode hashes sorted and unique`);
  }
  return manifest;
}

/** Read a TrainingSet: the schema, then sorted and unique hash lists. */
export function readTrainingSet(value: unknown): TrainingSet {
  if (!validSet(value)) throw new TrainingLoaderError(`invalid TrainingSet: ${JSON.stringify(validSet.errors)}`);
  const set = structuredClone(value) as unknown as TrainingSet;
  for (const field of ["episode_hashes", "evaluation_manifest_hashes"] as const) {
    if (!strictlyAscending(set[field])) throw new TrainingLoaderError(`the TrainingSet's ${field} must be sorted and unique`);
  }
  return set;
}

export interface TrainingInput {
  /** Stored Episodes, as read from JSON: each is checked, none is trusted. */
  episodes: readonly unknown[];
  /** Every EvaluationRunManifest there is (sealed ones are hash-only). */
  evaluation_manifests: readonly unknown[];
  /** The synthetic teams there are (hashes), as `readEpisode` requires. */
  known_synthetic: SyntheticTeamHashes;
}

export interface TrainingStream {
  /** The stream's records, in order: Episodes by hash, each Episode's decisions in request order. */
  examples: TrainingExample[];
  /** The stream as stored: the JCS bytes of each example, each line ended by a newline. */
  stream: string;
  set: TrainingSet;
}

const sha256Text = (text: string): Promise<string> => nodeSha256(new TextEncoder().encode(text));

/** The Episode's TrainingExamples: each decision's observation, and the choice. */
async function examplesOf(episode: Episode, hash: string): Promise<TrainingExample[]> {
  const observations = await decisionObservations(episode);
  return episode.decisions.map(({ choice }, i) => ({
    schema_version: 1,
    episode_hash: hash,
    observation: observations[i] as TrainingExample["observation"],
    choice,
  }));
}

/**
 * Build the training stream and its TrainingSet from stored Episodes.
 * Refused, each as a TrainingLoaderError:
 * - an Episode `readEpisode` refuses (its message is kept);
 * - one that is not synthetic with cc0-synthetic rights: private data is
 *   never loaded (it would make the stream private, which is not built yet);
 * - one with no `collection` record, or of kind `evaluation` or `sparring`:
 *   only `league` trains (C4);
 * - one whose hash an EvaluationRunManifest lists, which catches an
 *   evaluation Episode written under a wrong label;
 * - the same Episode or manifest twice, an invalid manifest, or no Episode.
 * A relabelled copy of an evaluation Episode has other bytes and so another
 * hash: only the storage partition catches it, not this check.
 */
export async function loadTrainingSet({ episodes, evaluation_manifests, known_synthetic }: TrainingInput): Promise<TrainingStream> {
  const manifestHashes = new Set<string>();
  const evaluated = new Set<string>();
  for (const value of evaluation_manifests) {
    const manifest = readEvaluationRunManifest(value);
    const hash = await sha256Text(jcs(manifest as unknown as Json));
    if (manifestHashes.has(hash)) throw new TrainingLoaderError(`the EvaluationRunManifest of run ${manifest.run_id} is given twice`);
    manifestHashes.add(hash);
    for (const listed of manifest.episode_hashes) evaluated.add(listed);
  }

  const accepted = new Map<string, Episode>();
  for (const [index, value] of episodes.entries()) {
    const episode = await readTraining(value, index, known_synthetic);
    const hash = await episodeHash(episode, nodeSha256);
    if (evaluated.has(hash)) throw new TrainingLoaderError(`Episode ${index} (${hash}) is listed in an EvaluationRunManifest: evaluation games never train`);
    if (accepted.has(hash)) throw new TrainingLoaderError(`Episode ${index} (${hash}) is given twice`);
    accepted.set(hash, episode);
  }
  if (accepted.size === 0) throw new TrainingLoaderError("no Episode to load");

  const hashes = [...accepted.keys()].sort();
  const examples: TrainingExample[] = [];
  for (const hash of hashes) examples.push(...(await examplesOf(accepted.get(hash) as Episode, hash)));
  const stream = examples.map((example) => `${jcs(example as unknown as Json)}\n`).join("");
  const set: TrainingSet = {
    schema_version: 1,
    data_class: "synthetic",
    episode_hashes: hashes as TrainingSet["episode_hashes"],
    evaluation_manifest_hashes: [...manifestHashes].sort(),
    examples: examples.length,
    stream_sha256: await sha256Text(stream),
  };
  return { examples, stream, set };
}

async function readTraining(value: unknown, index: number, known_synthetic: SyntheticTeamHashes): Promise<Episode> {
  const episode = await readEpisode(value, { known_synthetic }).catch((error: Error) => {
    throw new TrainingLoaderError(`Episode ${index} is refused: ${error.message}`, { cause: error });
  });
  const collection = episode.provenance.collection;
  if (collection === undefined) throw new TrainingLoaderError(`Episode ${index} has no collection record: it is not known to be training data`);
  if (collection.kind !== "league") {
    throw new TrainingLoaderError(`Episode ${index} was collected as ${collection.kind}: only a league Episode trains`);
  }
  if (episode.data_class !== "synthetic" || episode.rights !== "cc0-synthetic") {
    throw new TrainingLoaderError(`Episode ${index} is ${episode.data_class} with ${episode.rights} rights: only synthetic, cc0-synthetic data is loaded`);
  }
  return episode;
}
