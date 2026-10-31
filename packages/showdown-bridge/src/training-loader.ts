// The training loader (ADR-009 §4): the one way an Episode becomes training
// data. It fails closed. It reads each Episode through `readEpisode` (the
// schema, the class and rights pair, the player's own channel and team, the
// collection rules) and then accepts only a synthetic `league` Episode that
// no EvaluationRunManifest lists. Every refusal is a TrainingLoaderError;
// nothing is skipped, so a breach of the evaluation/training split is visible.
// It writes a stream of TrainingExample lines (no provenance, so no seed or
// opponent id can reach a model) and the TrainingSet manifest that Python
// checks against the stream.

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { ajvFor } from "@areneon/schema/ajv";
import evaluationRunManifestSchema from "@areneon/schema/contracts/v1/evaluation-run-manifest.schema.json" with { type: "json" };
import trainingExampleSchema from "@areneon/schema/contracts/v1/training-example.schema.json" with { type: "json" };
import trainingSetSchema from "@areneon/schema/contracts/v1/training-set.schema.json" with { type: "json" };
import type { EvaluationRunManifest } from "@areneon/schema/types/v1/evaluation-run-manifest.ts";
import type { TrainingExample } from "@areneon/schema/types/v1/training-example.ts";
import type { TrainingSet } from "@areneon/schema/types/v1/training-set.ts";
import { type FeatureConfigs, createFeatureMemo, FeatureBuildError } from "./candidate-features.ts";
import { buildFeatures } from "./feature-build.ts";
import { jcs, type Json } from "./canonical-json.ts";
import type { Engine } from "./engine.ts";
import { featureRowsProblem } from "./feature-rows.ts";
import { opponentSheetProblem } from "./opponent-sheet.ts";
import { type ChunkLimits, ChunkedFileWriter, checkStringLength } from "./chunked-file.ts";
import { enumerateCandidates, EnumerationError } from "./choices.ts";
import { type Episode, episodeHash } from "./episode.ts";
import { decisionObservations, readEpisode } from "./episode-io.ts";
import { nodeSha256, playerChannel } from "./player-channel.ts";
import { decodeRequest, RequestDecodeError } from "./request.ts";
import type { SyntheticTeamHashes } from "./team-class.ts";

export type { EvaluationRunManifest, TrainingExample, TrainingSet };

export class TrainingLoaderError extends Error {
  override name = "TrainingLoaderError";
}

const validManifest = ajvFor("contract").compile(evaluationRunManifestSchema);
const validSet = ajvFor("contract").compile(trainingSetSchema);
const validExample = ajvFor("contract").compile(trainingExampleSchema);

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

/**
 * Read a TrainingSet: the schema, then what the profile cannot say: sorted
 * and unique hash lists, and a learner_ready set names the policy it trains
 * (ADR-010 §5; the examples' own fields are checked by the stream check).
 */
export function readTrainingSet(value: unknown): TrainingSet {
  if (!validSet(value)) throw new TrainingLoaderError(`invalid TrainingSet: ${JSON.stringify(validSet.errors)}`);
  const set = structuredClone(value) as unknown as TrainingSet;
  for (const field of ["episode_hashes", "evaluation_manifest_hashes"] as const) {
    if (!strictlyAscending(set[field])) throw new TrainingLoaderError(`the TrainingSet's ${field} must be sorted and unique`);
  }
  if (set.learner_ready === true && set.behaviour_policy_id === undefined) {
    throw new TrainingLoaderError("a learner_ready TrainingSet must name its behaviour_policy_id");
  }
  return set;
}

/**
 * Read a TrainingExample: the schema, then what the profile cannot say: when
 * it carries `candidates`, its `choice` is one of them (ADR-010 §5). Whether
 * a stream's examples all carry them is the stream check's (learner_ready).
 */
export function readTrainingExample(value: unknown): TrainingExample {
  if (!validExample(value)) throw new TrainingLoaderError(`invalid TrainingExample: ${JSON.stringify(validExample.errors)}`);
  const example = structuredClone(value) as unknown as TrainingExample;
  if (example.candidate_features !== undefined) {
    const problem = featureRowsProblem(example.candidate_features);
    if (problem !== null) throw new TrainingLoaderError(`${problem} (${example.episode_hash} #${example.observation.identity.request_seq})`);
  }
  if (example.opponent_sheet !== undefined) {
    const problem = opponentSheetProblem(example.opponent_sheet);
    if (problem !== null) throw new TrainingLoaderError(`opponent_sheet: ${problem} (${example.episode_hash} #${example.observation.identity.request_seq})`);
  }
  if (example.candidates !== undefined && !example.candidates.some((candidate) => candidate.choice === example.choice)) {
    throw new TrainingLoaderError(
      `the choice "${example.choice}" of ${example.episode_hash} #${example.observation.identity.request_seq} is not among its ${example.candidates.length} candidates`,
    );
  }
  return example;
}

export interface EvaluationRunInput {
  run_id: string;
  sealed: boolean;
  /** The run's Episodes. Each must be an `evaluation` Episode of this run and sealed flag. */
  episodes: readonly Episode[];
}

/**
 * The evaluation job's writer (ADR-009 §4, once per run after its last
 * Episode): the run's Episode hashes, sorted and unique, and nothing else,
 * so a sealed run's manifest is hash-only. An Episode that is not an
 * evaluation Episode of this run, or a repeated one, is refused.
 */
export async function evaluationRunManifest({ run_id, sealed, episodes }: EvaluationRunInput): Promise<EvaluationRunManifest> {
  const hashes = new Set<string>();
  for (const [index, episode] of episodes.entries()) {
    const collection = episode.provenance.collection;
    if (collection?.kind !== "evaluation" || collection.run_id !== run_id || collection.sealed !== sealed) {
      throw new TrainingLoaderError(`Episode ${index} is not an evaluation Episode of run ${run_id} with sealed ${sealed}`);
    }
    const hash = await episodeHash(episode, nodeSha256);
    if (hashes.has(hash)) throw new TrainingLoaderError(`Episode ${index} (${hash}) is given twice`);
    hashes.add(hash);
  }
  return readEvaluationRunManifest({ schema_version: 1, run_id, sealed, episode_hashes: [...hashes].sort() });
}

export interface TrainingInput {
  /** Stored Episodes, as read from JSON: each is checked, none is trusted. */
  episodes: readonly unknown[];
  /** Every EvaluationRunManifest there is (sealed ones are hash-only). */
  evaluation_manifests: readonly unknown[];
  /** The synthetic teams there are (hashes), as `readEpisode` requires. */
  known_synthetic: SyntheticTeamHashes;
  /**
   * The one policy whose own seats train (ADR-010 §1): an Episode whose
   * `provenance.behaviour_policy_id` differs is refused, so a frozen
   * opponent's or a baseline's decisions never enter the update. Recorded in
   * the TrainingSet.
   */
  behaviour_policy_id: string;
  /**
   * The run's candidate feature set (ADR-022 section 3), when it names one: every example then carries
   * `candidate_features` recomputed here with the same builder the play-time actor used, from the Episode's own
   * channel prefix, its own team and the example's candidates (features are derived, never stored in an Episode).
   * Absent, no example has the field and every byte of the stream is as before. A decision the builder cannot
   * feature is a `TrainingLoaderError` (the game that played it forfeited, so no Episode should hold it).
   */
  features?: {
    engine: Engine;
    configs: FeatureConfigs;
    /**
     * The rows another process already recomputed for an Episode, by its hash (#1185; see `feature-recompute.ts`): used
     * only when they are one entry per decision, otherwise the rows are recomputed here. They are the same function
     * of the same Episode, so the stream's bytes do not depend on whether this is given or what it returns.
     */
    precomputed?: (hash: string, request_seqs: readonly number[]) => readonly NonNullable<TrainingExample["candidate_features"]>[] | undefined;
  };
}

export interface TrainingStream {
  /** The stream's records, in order: Episodes by hash, each Episode's decisions in request order. */
  examples: TrainingExample[];
  /** The stream as stored: the JCS bytes of each example, each line ended by a newline. */
  stream: string;
  set: TrainingSet;
}

/**
 * Where a stream's lines go: a file written in chunks (`path`, the way a generation
 * is written: no string holds the whole stream), and the limits on string sizes.
 */
export interface StreamOptions extends ChunkLimits {
  path: string;
}

const sha256Text = (text: string): Promise<string> => nodeSha256(new TextEncoder().encode(text));

/** The game score from the player's view (ADR-010 §1). */
const outcomeOf = ({ player, result }: Episode): NonNullable<TrainingExample["outcome"]> =>
  result.winner === null ? "tie" : result.winner === player ? "win" : "loss";

/**
 * One decision as a learner-ready TrainingExample. The candidates are
 * `enumerateCandidates` on the observation's own request (the `|request|`
 * line that ends its log), in canonical order, with their pending flags;
 * no CloneJudge verdict is used. Refused: a request the enumerator or
 * decoder does not understand, a `choice` that is not among the candidates,
 * and an example the schema refuses (more candidates than N_max).
 */
export function trainingExample(
  { observation, choice, outcome, episode_hash, candidate_features }: Pick<TrainingExample, "observation" | "choice" | "episode_hash" | "candidate_features"> & { outcome: NonNullable<TrainingExample["outcome"]> },
): TrainingExample {
  const line = observation.log.at(-1) as string;
  let candidates;
  try {
    candidates = enumerateCandidates(decodeRequest(line.slice("|request|".length))).candidates;
  } catch (error) {
    if (!(error instanceof EnumerationError || error instanceof RequestDecodeError)) throw error;
    throw new TrainingLoaderError(`the request of ${episode_hash} #${observation.identity.request_seq} has no candidates: ${error.message}`, { cause: error });
  }
  return readTrainingExample({ schema_version: 1, episode_hash, observation, choice, candidates, ...(candidate_features === undefined ? {} : { candidate_features }), outcome });
}

/** The Episode's TrainingExamples, one per decision. */
async function examplesOf(episode: Episode, hash: string, features: TrainingInput["features"]): Promise<TrainingExample[]> {
  const observations = await decisionObservations(episode);
  const outcome = outcomeOf(episode);
  const memo = createFeatureMemo(); // one Episode is one game, as at play time; the memo never changes a byte
  const given = features?.precomputed?.(hash, observations.map((o) => o.identity.request_seq));
  const rows = given?.length === episode.decisions.length ? given : undefined;
  return episode.decisions.map(({ choice }, i) => {
    const observation = observations[i] as TrainingExample["observation"];
    return trainingExample({
      observation,
      choice,
      outcome,
      episode_hash: hash,
      ...(features === undefined ? {} : { candidate_features: rows?.[i] ?? featuresOf(episode, observation, hash, features, memo) }),
    });
  });
}

/** The recomputed rows of every decision of an Episode, in decision order: what the stream carries, computed on their own (#1185). */
export async function episodeFeatureRows(episode: Episode, hash: string, features: NonNullable<TrainingInput["features"]>): Promise<NonNullable<TrainingExample["candidate_features"]>[]> {
  const observations = await decisionObservations(episode);
  const memo = createFeatureMemo();
  return observations.map((observation) => featuresOf(episode, observation, hash, features, memo));
}

/** The rows of one decision, recomputed from what the play-time actor had: the channel up to the request, the own team, the candidates. */
function featuresOf(
  episode: Episode,
  observation: TrainingExample["observation"],
  hash: string,
  { engine, configs }: NonNullable<TrainingInput["features"]>,
  memo: ReturnType<typeof createFeatureMemo>,
): NonNullable<TrainingExample["candidate_features"]> {
  const line = observation.log.at(-1) as string;
  try {
    const candidates = enumerateCandidates(decodeRequest(line.slice("|request|".length))).candidates;
    const channel = playerChannel({ schema_version: 1, data_class: observation.data_class, player: observation.player, lines: observation.log });
    return buildFeatures(engine, configs, channel, episode.team.team, candidates, memo);
  } catch (error) {
    if (!(error instanceof FeatureBuildError || error instanceof EnumerationError || error instanceof RequestDecodeError)) throw error;
    throw new TrainingLoaderError(`the candidate features of ${hash} #${observation.identity.request_seq} cannot be built: ${error.message}`, { cause: error });
  }
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
 * - one played by another policy than `behaviour_policy_id` (ADR-010 §1);
 * - a decision whose `choice` is not among the candidates `enumerateCandidates`
 *   returns for its request (ADR-010 §5);
 * - the same Episode or manifest twice, an invalid manifest, or no Episode.
 * Every example carries `candidates` and `outcome`, so the set is `learner_ready`.
 * A relabelled copy of an evaluation Episode has other bytes and so another
 * hash: only the storage partition catches it, not this check.
 */
export async function loadTrainingSet(input: TrainingInput, limits: ChunkLimits = {}): Promise<TrainingStream> {
  const examples: TrainingExample[] = [];
  const pieces: string[] = [];
  let chars = 0;
  const set = await streamTraining(input, memoryFeed(input), limits, (example, line) => {
    examples.push(example);
    pieces.push(line);
    chars += line.length;
    // The stream as one string must fit one: a larger one is written with `writeTrainingStream`.
    if (limits.max_string_chars !== undefined && chars > limits.max_string_chars) {
      throw new TrainingLoaderError(`the stream is over ${limits.max_string_chars} characters: write it with writeTrainingStream, not as one string`);
    }
  });
  return { examples, stream: pieces.join(""), set };
}

/**
 * The same checks and the same bytes as `loadTrainingSet`, with the stream written
 * to `options.path` in chunks as the examples are made and none kept: neither the
 * stream nor the examples are ever held whole, so a stream over Node's string limit
 * (about 512 MB) is no problem (#998). The file is flushed to disk; the TrainingSet
 * (its `stream_sha256` is the file's digest) is returned. The file is left behind
 * when a check refuses: the caller discards the directory it wrote into.
 */
export async function writeTrainingStream(input: TrainingInput, options: StreamOptions): Promise<TrainingSet> {
  const writer = new ChunkedFileWriter(options.path, options);
  try {
    const set = await streamTraining(input, memoryFeed(input), options, (_example, line) => writer.write(line));
    writer.finish();
    return set;
  } finally {
    writer.close();
  }
}

/** An Episode stored as JSON under the hash it is addressed by (`episodes/<hash>.json` of a generation). */
export interface StoredEpisode {
  hash: string;
  path: string;
}

/** `TrainingInput` for Episodes that are on disk: they are read one at a time and none is kept (#1069). */
export interface StoredTrainingInput extends Omit<TrainingInput, "episodes"> {
  /** In any order; the stream lists them by hash. */
  episodes: readonly StoredEpisode[];
}

/**
 * `writeTrainingStream` for Episodes stored as files: the same checks, the same stream bytes and
 * the same TrainingSet as with the Episodes in memory, but only one Episode is held at a time, so
 * a generation's memory does not grow with its size (#1069). Each file must hold the Episode its
 * name addresses (its recomputed hash is compared), and a repeated hash is refused before any
 * file is read. Episodes are checked as their turn comes in hash order, so a bad Episode is found
 * after the stream is partly written, never before; the caller discards the file on a refusal.
 */
export async function writeStoredTrainingStream(input: StoredTrainingInput, options: StreamOptions): Promise<TrainingSet> {
  const writer = new ChunkedFileWriter(options.path, options);
  try {
    const set = await streamTraining(input, fileFeed(input), options, (_example, line) => writer.write(line));
    writer.finish();
    return set;
  } finally {
    writer.close();
  }
}

/** Every Episode as `[hash, Episode]`, validated, in the order the stream lists them (hash order). */
type EpisodeFeed = (evaluated: ReadonlySet<string>) => AsyncGenerator<[string, Episode]>;

function memoryFeed({ episodes, known_synthetic, behaviour_policy_id }: TrainingInput): EpisodeFeed {
  return async function* (evaluated) {
    const accepted = new Map<string, Episode>();
    for (const [index, value] of episodes.entries()) {
      const episode = await readTraining(value, index, known_synthetic, behaviour_policy_id);
      const hash = await episodeHash(episode, nodeSha256);
      if (evaluated.has(hash)) throw new TrainingLoaderError(`Episode ${index} (${hash}) is listed in an EvaluationRunManifest: evaluation games never train`);
      if (accepted.has(hash)) throw new TrainingLoaderError(`Episode ${index} (${hash}) is given twice`);
      accepted.set(hash, episode);
    }
    if (accepted.size === 0) throw new TrainingLoaderError("no Episode to load");
    for (const hash of [...accepted.keys()].sort()) yield [hash, accepted.get(hash) as Episode];
  };
}

function fileFeed({ episodes, known_synthetic, behaviour_policy_id }: StoredTrainingInput): EpisodeFeed {
  return async function* (evaluated) {
    const seen = new Set<string>();
    for (const [index, { hash }] of episodes.entries()) {
      if (evaluated.has(hash)) throw new TrainingLoaderError(`Episode ${index} (${hash}) is listed in an EvaluationRunManifest: evaluation games never train`);
      if (seen.has(hash)) throw new TrainingLoaderError(`Episode ${index} (${hash}) is given twice`);
      seen.add(hash);
    }
    if (seen.size === 0) throw new TrainingLoaderError("no Episode to load");
    const order = episodes.map((stored, index) => ({ ...stored, index })).sort((a, b) => (a.hash < b.hash ? -1 : 1));
    for (const { hash, path, index } of order) {
      let stored: unknown;
      try {
        stored = JSON.parse(readFileSync(path, "utf8")) as unknown;
      } catch (error) {
        throw new TrainingLoaderError(`Episode ${index} cannot be read from its file: ${(error as Error).name}`, { cause: error });
      }
      const episode = await readTraining(stored, index, known_synthetic, behaviour_policy_id);
      if ((await episodeHash(episode, nodeSha256)) !== hash) throw new TrainingLoaderError(`Episode ${index} is stored under a hash that is not its own`);
      yield [hash, episode];
    }
  };
}

/** The checks, the examples in stream order and the TrainingSet; `each` gets every example and its line. */
async function streamTraining(
  { evaluation_manifests, behaviour_policy_id, features }: Pick<TrainingInput, "evaluation_manifests" | "behaviour_policy_id" | "features">,
  feed: EpisodeFeed,
  limits: ChunkLimits,
  each: (example: TrainingExample, line: string) => void,
): Promise<TrainingSet> {
  const manifestHashes = new Set<string>();
  const evaluated = new Set<string>();
  for (const value of evaluation_manifests) {
    const manifest = readEvaluationRunManifest(value);
    const hash = await sha256Text(jcs(manifest as unknown as Json));
    if (manifestHashes.has(hash)) throw new TrainingLoaderError(`the EvaluationRunManifest of run ${manifest.run_id} is given twice`);
    manifestHashes.add(hash);
    for (const listed of manifest.episode_hashes) evaluated.add(listed);
  }

  const hashes: string[] = [];
  const digest = createHash("sha256");
  let count = 0;
  for await (const [hash, episode] of feed(evaluated)) {
    hashes.push(hash);
    for (const example of await examplesOf(episode, hash, features)) {
      const line = `${jcs(example as unknown as Json)}\n`;
      checkStringLength(line, limits, "a stream line");
      digest.update(line, "utf8");
      count++;
      each(example, line);
    }
  }
  return {
    schema_version: 1,
    // The highest class of the Episodes: readTraining refuses every private Episode, so it is synthetic.
    data_class: "synthetic",
    episode_hashes: hashes as TrainingSet["episode_hashes"],
    evaluation_manifest_hashes: [...manifestHashes].sort(),
    learner_ready: true,
    behaviour_policy_id,
    examples: count,
    stream_sha256: digest.digest("hex"),
  };
}

async function readTraining(value: unknown, index: number, known_synthetic: SyntheticTeamHashes, behaviour_policy_id: string): Promise<Episode> {
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
  if (episode.provenance.behaviour_policy_id !== behaviour_policy_id) {
    throw new TrainingLoaderError(`Episode ${index} was played by ${episode.provenance.behaviour_policy_id}, not ${behaviour_policy_id}: only the trained policy's own seats train`);
  }
  return episode;
}
