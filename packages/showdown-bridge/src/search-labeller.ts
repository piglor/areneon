// The off-path search labellers and the label stream (ADR-026 erratum E2 decisions 1, 3-6 and 10; SD-d, #1290).
//
// After a game, a hashed share of the pair's own Battle move decisions is labelled by the decision-time search, from the stored
// Episode alone: the Episode is read through `readEpisode` (inside `readTraining`), the own channel is cut at each selected
// request by `decisionObservations`, and `searchLabel` (the actor's own decision, under the label config) gives each searched
// candidate's q. Nothing here reads a `BattleRecord`, an omniscient log, a hidden verdict or the opponent's team (`LabelContext` has
// no field for any; a type test). Nothing the search does is played or enters an Episode or the PPO stream.
//
// A game process labels the Episodes it wrote (the way #1185's feature rows are recomputed), leaving one scratch file
// `<episode hash>.search.json` per Episode in a directory of the generation's shard scratch. The stream writer merges the files
// into `labels.jsonl` and `label-set.json` in Episode-hash then decision order. A missing or stale file is recomputed by the
// writer's process; more than 5% missing fails the generation as a harness incident. A file is a pure function of the Episode
// and the reviewed records, so the stream's bytes do not depend on which process labelled what.
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { TrainingExample } from "@areneon/schema/types/v1/training-example.ts";
import type { TrainingSet } from "@areneon/schema/types/v1/training-set.ts";
import { createFeatureMemo, FeatureBuildError, type FeatureConfigs, type FeatureSetId } from "./candidate-features.ts";
import { type Json, jcs } from "./canonical-json.ts";
import { enumerateCandidates } from "./choices.ts";
import type { Engine } from "./engine.ts";
import { type Episode, episodeHash } from "./episode.ts";
import { decisionObservations } from "./episode-io.ts";
import { buildFeatures, isFeatureSetId } from "./feature-build.ts";
import type { VersionManifest } from "./formats.ts";
import { leagueGameStreams } from "./league-seed.ts";
import { ChunkedFileWriter, type ChunkLimits, checkStringLength } from "./chunked-file.ts";
import { nodeSha256, playerChannel } from "./player-channel.ts";
import type { PlaySettings } from "./policy-protocol.ts";
import type { EvaluationClient, PolicyClient } from "./policy-worker.ts";
import { decodeRequest } from "./request.ts";
import type { ReplyModel } from "./reply-model.ts";
import { sha256Hex } from "./rules.ts";
import { searchActor, searchPolicy, type SearchSetup, worldRequestOf } from "./search-actor.ts";
import { evidenceClass, searchStream } from "./world-sampler.ts";
import { sheetRegimeOf } from "./team-sheet-binding.ts";
import type { SyntheticTeamHashes } from "./team-class.ts";
import { examplesOf, readTraining, readTrainingExample, readTrainingSet } from "./training-loader.ts";

/** The hash purpose of the selection (`generation.search_labels`). */
export const LABEL_PURPOSE = "search-label/1";
/** Bump when the scratch file or the way a label is made from a decision changes. */
export const LABEL_FILE_FORMAT = 1;
/** More than this share (percent) of missing scratch files fails the generation. */
export const MAX_MISSING_PERCENT = 5;

/** A labelling defect of the harness (a seed that is not the run's, a stale mass of missing files). Never a forfeit. */
export class LabelError extends Error {
  override name = "LabelError";
}

/** The reasons a selected decision gets no label, sorted when written (the keys of `search_labels.refused`). */
export const LABEL_REFUSALS = ["belief_refused", "below_min_valid", "features_refused", "no_prior_mass", "spread_below_min", "too_few_searched", "world_dropped"] as const;
export type LabelRefusal = (typeof LABEL_REFUSALS)[number];

/** The reviewed records and values of a run's labelling (`generation.search_labels` and `ppo.search_distill`). */
export interface LabelSetup {
  /** The search setup under the label config (`config/search/rerank-1-label.v1.json`). */
  setup: SearchSetup;
  /** The hash `generation.search_labels.search_config_hash` names (checked by the caller against the setup's config). */
  search_config_hash: string;
  share_ppm: number;
  start_generation: number;
  /** `ppo.search_distill.min_spread_ppm`: a label under it is refused here (the learner would not use it). */
  min_spread_ppm: number;
  pair: { preview: string; battle: string };
  manifest: VersionManifest;
  /** The candidate features of the root query, for a pair that reads `probe-damage/1`. */
  features?: FeatureConfigs;
  /** The run's play settings (`generation.play`): part of the search policy id. */
  play?: PlaySettings;
}

/** Everything a labeller reads besides one Episode: no field for a record, a log, a verdict or the opponent's team. */
export interface LabelContext {
  engine: Engine;
  labels: LabelSetup;
  /** The pair's worker: the prior (`evaluate`) and the leaf values. */
  client: PolicyClient & EvaluationClient & { beginGame?: () => Promise<void> };
  run_key: string;
  generation: number;
  /** Test seam: the reply model (default the setup's reviewed records). */
  reply_model?: ReplyModel;
}

const U64 = 1n << 64n;

/**
 * Whether the decision is selected for a label: the first 8 bytes of SHA-256 over the JCS of
 * `{run_key, generation, game, player, request_seq, purpose: "search-label/1"}`, read big-endian as n, satisfy
 * n * 10^6 < share * 2^64 (exact integers).
 */
export function labelSelected(parts: { run_key: string; generation: number; game: number; player: string; request_seq: number }, share_ppm: number): boolean {
  const n = BigInt(`0x${sha256Hex(jcs({ game: parts.game, generation: parts.generation, player: parts.player, purpose: LABEL_PURPOSE, request_seq: parts.request_seq, run_key: parts.run_key })).slice(0, 16)}`);
  return n * 1_000_000n < BigInt(share_ppm) * U64;
}

/** One selected decision of an Episode: a label, or the reason it has none; `class` is the decision's evidence class (a diagnostic only). */
export type LabelledDecision =
  | { request_seq: number; class: string; label: { prior_ppm: number[]; entries: { candidate: number; q_ppm: number }[] } }
  | { request_seq: number; class: string; refused: LabelRefusal };

/** The scratch file of one Episode. */
export interface EpisodeLabels {
  format: number;
  episode_hash: string;
  search_identity_hash: string;
  decisions: LabelledDecision[];
}

const labelsFile = (dir: string, hash: string): string => join(dir, `${hash}.search.json`);

/** The search policy id and identity of the pair under the label config, for the worker's stated feature set. */
export function labelPolicy(labels: LabelSetup, client: Pick<PolicyClient, "featureSet">): { policy_id: string; identity_hash: string } {
  const { policy_id, identity_hash } = searchPolicy(labels.setup, labels.manifest, labels.pair, "sample", labels.play ?? {}, client.featureSet?.feature_set_hash);
  return { policy_id, identity_hash };
}

/**
 * Label one stored Episode: every selected Battle move decision of the Episode's own seat. The Episode must be the one the context's
 * run played as `game` of `generation` (its collection seed is recomputed from `league-seed/1` and a mismatch is a `LabelError`).
 * A defect of the setup propagates; a decision the search cannot label is a counted refusal.
 */
export async function labelEpisode(context: LabelContext, episode: Episode, hash: string, game: number): Promise<EpisodeLabels> {
  const { engine, labels, client, run_key, generation } = context;
  const { setup, manifest } = labels;
  const collection = episode.provenance.collection;
  if (collection === undefined || collection.kind !== "league" || collection.run_id !== run_key) throw new LabelError("the Episode is not a league Episode of this run");
  const expected = leagueGameStreams(run_key, generation, game).seed;
  if (collection.seed !== expected) throw new LabelError("the Episode's seed is not the one league-seed/1 gives for this game");
  const { policy_id, identity_hash } = labelPolicy(labels, client);
  const player = episode.player;
  const observations = await decisionObservations(episode);
  const regime = sheetRegimeOf(manifest.showdown_commit, manifest.format_id);
  const battle = `label-${hash.slice(0, 24)}`;
  await client.beginGame?.();
  const actor = searchActor({
    setup,
    client,
    battle,
    player,
    team: episode.team,
    manifest,
    channel_class: episode.data_class,
    seed: collection.seed,
    game,
    policy_id,
    identity_hash,
    sampling: "sample",
    pair: labels.pair,
    ...(labels.play === undefined ? {} : { play: labels.play }),
    ...(labels.features === undefined ? {} : { features: { configs: labels.features } }),
    ...(context.reply_model === undefined ? {} : { reply_model: context.reply_model }),
  });
  const stated = client.featureSet ?? null;
  const memo = createFeatureMemo();
  const dropped = (): number => Object.values(actor.report().counts.worlds_dropped).reduce((a, b) => a + b, 0);
  const decisions: LabelledDecision[] = [];
  for (const observation of observations) {
    const line = observation.log.at(-1) as string;
    const request = decodeRequest(line.slice("|request|".length));
    if (request.phase !== "move") continue;
    const request_seq = observation.identity.request_seq;
    const candidates = enumerateCandidates(request).candidates;
    if (candidates.filter((candidate) => !candidate.pending).length < 2) continue;
    if (!labelSelected({ run_key, generation, game, player, request_seq }, labels.share_ppm)) continue;
    const world = worldRequestOf(setup.facts_data, player, { channel: observation.log, team: episode.team.team });
    const klass = evidenceClass(world);
    let candidate_features;
    if (labels.features !== undefined && stated !== null) {
      if (!isFeatureSetId(stated.feature_set)) throw new LabelError("the worker stated a candidate feature set this side cannot build");
      try {
        const channel = playerChannel({ schema_version: 1, data_class: observation.data_class, player, lines: observation.log }, regime);
        candidate_features = buildFeatures(engine, labels.features, channel, episode.team.team, candidates, memo, stated.feature_set as FeatureSetId);
      } catch (error) {
        if (!(error instanceof FeatureBuildError)) throw error;
        decisions.push({ request_seq, class: klass, refused: "features_refused" });
        continue;
      }
    }
    const prior = await client.evaluate(battle, {
      schema_version: 1,
      identity: observation.identity,
      observation,
      candidates: candidates.map(({ choice, pending }) => ({ choice, pending })),
      ...(candidate_features === undefined ? {} : { candidate_features }),
      sampling: { kind: "evaluate" },
      policy_id,
    } as Parameters<EvaluationClient["evaluate"]>[1]);
    const stream = searchStream({ seed: collection.seed, game, player, policy_id, request_seq });
    const before = dropped();
    const decision = await actor.label({ world, channel: observation.log, candidates, prior_ppm: prior.prior_ppm, stream });
    const refuse = (refused: LabelRefusal): void => void decisions.push({ request_seq, class: klass, refused });
    if (decision.kind === "refused") {
      // Selection kept only requests with two or more non-pending candidates, so `too_few_candidates` cannot be the reason.
      if (decision.reason === "too_few_candidates") throw new LabelError("the search found too few candidates in a decision the labeller selected");
      refuse(decision.reason);
      continue;
    }
    if (dropped() > before) {
      refuse("world_dropped");
      continue;
    }
    const entries = decision.entries.map(({ candidate, q_ppm }) => ({ candidate, q_ppm }));
    const qs = entries.map((entry) => entry.q_ppm);
    if (entries.length < 2) refuse("too_few_searched");
    else if (entries.reduce((sum, entry) => sum + (decision.prior_ppm[entry.candidate] as number), 0) < 1) refuse("no_prior_mass");
    else if (Math.max(...qs) - Math.min(...qs) < Math.max(1, labels.min_spread_ppm)) refuse("spread_below_min");
    else decisions.push({ request_seq, class: klass, label: { prior_ppm: [...decision.prior_ppm], entries } });
  }
  return { format: LABEL_FILE_FORMAT, episode_hash: hash, search_identity_hash: identity_hash, decisions };
}

/** Leave an Episode's labels in `dir`. */
export function writeEpisodeLabels(dir: string, found: EpisodeLabels): void {
  mkdirSync(dir, { recursive: true });
  writeFileSync(labelsFile(dir, found.episode_hash), jcs(found as unknown as Json));
}

/** The scratch file of an Episode when it is there, parses and was made under `identity_hash`; else undefined (it is recomputed). */
export function readEpisodeLabels(dir: string, hash: string, identity_hash: string): EpisodeLabels | undefined {
  try {
    const found = JSON.parse(readFileSync(labelsFile(dir, hash), "utf8")) as Partial<EpisodeLabels>;
    if (found.format !== LABEL_FILE_FORMAT || found.episode_hash !== hash || found.search_identity_hash !== identity_hash || !Array.isArray(found.decisions)) return undefined;
    return found as EpisodeLabels;
  } catch {
    return undefined;
  }
}

/** A stored Episode to label: its hash, its file and the plan game that played it. */
export interface LabelEpisodeRef {
  hash: string;
  path: string;
  game: number;
}

const errorFile = (dir: string, hash: string): string => join(dir, `${hash}.search.error`);

/** The class of the error a labeller left for an Episode (`<hash>.search.error`), or undefined. */
export function readLabelError(dir: string, hash: string): string | undefined {
  try {
    return readFileSync(errorFile(dir, hash), "utf8");
  } catch {
    return undefined;
  }
}

/** One labeller worker (a started policy client); a process labels with all of its workers, one Episode per worker at a time. */
type LabelClient = LabelContext["client"];

/** Spread `refs` over `clients` (Episode i to client i mod n) and run `each` on every one, the clients in parallel, a client's Episodes one after another. */
async function spread<T extends LabelEpisodeRef>(clients: readonly LabelClient[], refs: readonly T[], each: (client: LabelClient, ref: T) => Promise<boolean>, stopped: () => boolean): Promise<number> {
  const done = await Promise.all(
    clients.map(async (client, worker) => {
      let n = 0;
      for (const [index, ref] of refs.entries()) {
        if (index % clients.length !== worker) continue;
        if (stopped()) break;
        if (await each(client, ref)) n += 1;
      }
      return n;
    }),
  );
  return done.reduce((a, b) => a + b, 0);
}

/**
 * Label each Episode with every worker of the process (Episodes spread over `clients`, default the context's one; the files are the
 * same whatever the spread, because a label is a pure function of its Episode) and leave its file in `dir`. An Episode that cannot
 * be read or labelled gets no file but a `.search.error` file naming the error class (never content) and a line on stderr: the
 * stream writer recomputes it, and fails the generation if that fails again or too many are missing.
 */
export async function writeLabelFiles(
  context: LabelContext & { known_synthetic: SyntheticTeamHashes; behaviour_policy_id: string },
  episodes: readonly LabelEpisodeRef[],
  dir: string,
  stopped: () => boolean = () => false,
  clients: readonly LabelClient[] = [context.client],
): Promise<number> {
  const { identity_hash } = labelPolicy(context.labels, context.client);
  return spread(
    clients,
    episodes,
    async (client, ref) => {
      try {
        if (readEpisodeLabels(dir, ref.hash, identity_hash) !== undefined) return false;
        const episode = await readStored(context, ref);
        writeEpisodeLabels(dir, await labelEpisode({ ...context, client }, episode, ref.hash, ref.game));
        return true;
      } catch (error) {
        process.stderr.write(`search labels skipped for game ${ref.game}: ${(error as Error).name}\n`);
        mkdirSync(dir, { recursive: true });
        writeFileSync(errorFile(dir, ref.hash), (error as Error).name);
        return false;
      }
    },
    stopped,
  );
}

async function readStored(context: { known_synthetic: SyntheticTeamHashes; behaviour_policy_id: string }, ref: LabelEpisodeRef): Promise<Episode> {
  const episode = await readTraining(JSON.parse(readFileSync(ref.path, "utf8")) as unknown, ref.game, context.known_synthetic, context.behaviour_policy_id);
  if ((await episodeHash(episode, nodeSha256)) !== ref.hash) throw new LabelError("an Episode is stored under a hash that is not its own");
  return episode;
}

/** The counts of one generation's labelling (integers; nothing but counts and class names). */
export interface LabelTotals {
  selected: number;
  labelled: number;
  refused: Record<string, number>;
  /** Labels and refusals by the decision's evidence class (a diagnostic only). */
  classes: Record<string, number>;
  /** Episodes whose scratch file was missing or stale and was recomputed by the stream writer's process. */
  recomputed: number;
}

export interface LabelStreamResult {
  /** The label set, or null when no decision was labelled (then no label file is written). */
  set: TrainingSet | null;
  totals: LabelTotals;
}

export interface LabelStreamInput {
  context: LabelContext & { known_synthetic: SyntheticTeamHashes; behaviour_policy_id: string };
  episodes: readonly LabelEpisodeRef[];
  /** The scratch directory the game processes left their files in. */
  dir: string;
  /** The label stream's file. */
  path: string;
  limits?: ChunkLimits;
  /** The process's workers, over which missing files are recomputed (default the context's one). */
  clients?: readonly LabelClient[];
  /** The share of missing files that fails the generation (default `MAX_MISSING_PERCENT`; a test seam). */
  max_missing_percent?: number;
  /** The run's candidate features, for the examples' rows (as the PPO stream carries them). */
  features?: { engine: Engine; configs: FeatureConfigs; precomputed?: (hash: string, request_seqs: readonly number[]) => readonly NonNullable<TrainingExample["candidate_features"]>[] | undefined };
}

/**
 * Merge the Episodes' label files into the label stream (`path`) and its TrainingSet. Examples come in Episode-hash then decision
 * order, each the Episode's own TrainingExample with `search_target`, so the bytes depend on the Episodes and the records alone.
 * More than 5% of the files missing is a `LabelError`; the rest are recomputed here. With no label, nothing is written.
 */
export async function writeLabelStream(input: LabelStreamInput): Promise<LabelStreamResult> {
  const { context, dir, path } = input;
  const limits = input.limits ?? {};
  const { identity_hash } = labelPolicy(context.labels, context.client);
  const ordered = [...input.episodes].sort((a, b) => (a.hash < b.hash ? -1 : 1));
  const missing = ordered.filter((ref) => readEpisodeLabels(dir, ref.hash, identity_hash) === undefined);
  const allowed = input.max_missing_percent ?? MAX_MISSING_PERCENT;
  if (missing.length * 100 > ordered.length * allowed) {
    const first = missing.map((ref) => readLabelError(dir, ref.hash)).find((name) => name !== undefined);
    throw new LabelError(`${missing.length} of ${ordered.length} Episodes have no search label file: more than ${allowed}% missing is a harness incident${first === undefined ? "" : ` (first labeller error: ${first})`}`);
  }
  const totals: LabelTotals = { selected: 0, labelled: 0, refused: {}, classes: {}, recomputed: missing.length };
  const found = new Map<string, EpisodeLabels>();
  // The missing files are made again by all of the process's workers (a failure here is the generation's: it propagates).
  await spread(
    input.clients ?? [context.client],
    missing,
    async (client, ref) => {
      const labels = await labelEpisode({ ...context, client }, await readStored(context, ref), ref.hash, ref.game);
      writeEpisodeLabels(dir, labels);
      found.set(ref.hash, labels);
      return true;
    },
    () => false,
  );
  for (const ref of ordered) if (!found.has(ref.hash)) found.set(ref.hash, readEpisodeLabels(dir, ref.hash, identity_hash) as EpisodeLabels);
  const writer = new ChunkedFileWriter(path, limits);
  const digest = createHash("sha256");
  const withLabels: string[] = [];
  let count = 0;
  try {
    for (const ref of ordered) {
      const labels = found.get(ref.hash) as EpisodeLabels;
      const labelled = new Map<number, Extract<LabelledDecision, { label: unknown }>["label"]>();
      for (const decision of labels.decisions) {
        totals.selected += 1;
        const key = `${"label" in decision ? "labelled" : "refused"}.${decision.class}`;
        totals.classes[key] = (totals.classes[key] ?? 0) + 1;
        if ("label" in decision) labelled.set(decision.request_seq, decision.label);
        else totals.refused[decision.refused] = (totals.refused[decision.refused] ?? 0) + 1;
      }
      if (labelled.size === 0) continue;
      const episode = await readStored(context, ref);
      const examples = await examplesOf(episode, ref.hash, input.features, undefined, new Set(labelled.keys()));
      if (examples.length !== labelled.size) throw new LabelError("a label names a decision the Episode does not have");
      withLabels.push(ref.hash);
      for (const example of examples) {
        const label = labelled.get(example.observation.identity.request_seq) as NonNullable<ReturnType<typeof labelled.get>>;
        const labelledExample = readTrainingExample({ ...example, search_target: { prior_ppm: label.prior_ppm, entries: label.entries } });
        const line = `${jcs(labelledExample as unknown as Json)}\n`;
        checkStringLength(line, limits, "a label stream line");
        digest.update(line, "utf8");
        writer.write(line);
        count += 1;
      }
    }
    if (count > 0) writer.finish();
  } finally {
    writer.close();
  }
  if (count === 0) {
    rmSync(path, { force: true }); // a generation with no label has no label file (the schema's `labelled` is at least 1)
    return { set: null, totals };
  }
  totals.labelled = count;
  const { refused } = labelCounts(totals);
  const set = readTrainingSet({
    schema_version: 1,
    data_class: "synthetic",
    episode_hashes: withLabels,
    evaluation_manifest_hashes: [],
    learner_ready: true,
    behaviour_policy_id: context.behaviour_policy_id,
    examples: count,
    stream_sha256: digest.digest("hex"),
    search_labels: { search_identity_hash: identity_hash, labelled: count, refused },
  });
  return { set, totals };
}

/**
 * The generation's counts as `generation.json` states them (`search_labels`): how many decisions were selected, labelled and refused
 * (by reason, sorted) and how many Episodes the writer had to label again. Counts only; the diagnostic classes stay out.
 */
export function labelCounts(totals: LabelTotals): { selected: number; labelled: number; refused: { reason: string; count: number }[]; recomputed: number } {
  // `LABEL_REFUSALS` is sorted by code unit, the order the manifest's reader requires.
  const refused = LABEL_REFUSALS.filter((reason) => (totals.refused[reason] ?? 0) > 0).map((reason) => ({ reason, count: totals.refused[reason] as number }));
  return { selected: totals.selected, labelled: totals.labelled, refused, recomputed: totals.recomputed };
}
