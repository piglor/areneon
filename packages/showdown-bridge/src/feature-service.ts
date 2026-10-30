// The candidate feature builder as a service one thread owns (ADR-022 section 4, mitigation 3, ticket #1030).
// A `FeatureService` loads the pinned engine and the reviewed records once and answers `FeatureTask`s: the
// player's own channel lines, own team and candidates in, the rows of `candidateFeatures` out. It keeps one
// probe memo per game (a pure function of the query, so no row depends on it) and measures the time of each build.
// Nothing here reads an opponent team, a verdict or a seed: a task holds the same inputs `candidateFeatures` takes.
// The worker-thread pool (`feature-pool.ts`) runs one service per thread.
import { type CandidateFeatures, candidateFeatures, createFeatureMemo, type FeatureBuildCode, FeatureBuildError, type FeatureConfigs, type FeatureMemo } from "./candidate-features.ts";
import { type CandidateFeatures2, candidateFeatures2, type CostTrace, emptyTrace } from "./candidate-features-2.ts";
import type { Candidate } from "./choices.ts";
import { loadDamageAwareConfig } from "./damage-aware-config.ts";
import { type ProbeCounts } from "./damage-aware-view.ts";
import { type Engine, loadEngine, type PokemonSet } from "./engine.ts";
import { goldenTeamVocabulary, GOLDEN_FORMAT } from "./golden.ts";
import { loadMechanicsConfig } from "./mechanics-config.ts";
import { playerChannel } from "./player-channel.ts";
import { loadRulesSnapshot } from "./rules-load.ts";
import type { Player } from "./runner.ts";

/** What a service is built from. Plain data, so it crosses to a worker thread. */
export interface FeatureServiceSpec {
  format_id: string;
  data_class: "synthetic" | "private";
  /** The feature set the service builds; default `probe-damage/1`. */
  feature_set?: "probe-damage/1" | "probe-damage/2";
  /** Another built checkout than the default (the engine loader's argument). */
  showdown_dir?: string;
}

/** One decision to featurise. `game` keys the probe memo; `final` drops it after this task. */
export interface FeatureTask {
  game: string;
  player: Player;
  /** The player's own channel up to and including the request being answered. */
  lines: string[];
  team: PokemonSet[];
  candidates: Pick<Candidate, "choice" | "pending">[];
  final?: boolean;
}

export type FeatureOutcome = { kind: "features"; features: CandidateFeatures | CandidateFeatures2 } | { kind: "error"; code: FeatureBuildCode };

export interface FeatureResult {
  outcome: FeatureOutcome;
  /** Wall-clock milliseconds of the build in its thread; a measurement, never part of the rows. */
  ms: number;
  /** The probe counts this task added to its game's memo. */
  counts: ProbeCounts & { probe_refusals: number };
  /** Where a `probe-damage/2` build spent them, by part (absent for `probe-damage/1`). */
  parts?: CostTrace;
}

export interface FeatureService {
  run(task: FeatureTask): FeatureResult;
}

export function createFeatureService(spec: FeatureServiceSpec, engineOverride?: Engine): FeatureService {
  if (spec.format_id !== GOLDEN_FORMAT) throw new Error(`the feature builder has a vocabulary for ${GOLDEN_FORMAT} only, not ${spec.format_id}`);
  const engine = engineOverride ?? loadEngine(spec.showdown_dir);
  const vocabulary = goldenTeamVocabulary();
  const configs: FeatureConfigs = {
    vocabulary,
    config: loadDamageAwareConfig("v1", vocabulary),
    mechanics: loadMechanicsConfig(spec.format_id),
    snapshot: loadRulesSnapshot(engine.commit),
    data_class: spec.data_class,
  };
  const memos = new Map<string, FeatureMemo>();
  return {
    run(task) {
      let memo = memos.get(task.game);
      if (memo === undefined) memos.set(task.game, (memo = createFeatureMemo()));
      const before = { ...memo.counts };
      const started = performance.now();
      let outcome: FeatureOutcome;
      const parts = spec.feature_set === "probe-damage/2" ? emptyTrace() : undefined;
      try {
        const channel = playerChannel({ schema_version: 1, data_class: spec.data_class, player: task.player, lines: task.lines });
        outcome = { kind: "features", features: parts === undefined ? candidateFeatures(engine, configs, channel, task.team, task.candidates, memo) : candidateFeatures2(engine, configs, channel, task.team, task.candidates, memo, parts) };
      } catch (error) {
        if (!(error instanceof FeatureBuildError)) throw error;
        outcome = { kind: "error", code: error.code };
      }
      const ms = performance.now() - started;
      if (task.final === true) memos.delete(task.game);
      const counts = { queries: memo.counts.queries - before.queries, probe_runs: memo.counts.probe_runs - before.probe_runs, unsupported_queries: memo.counts.unsupported_queries - before.unsupported_queries, probe_refusals: memo.counts.probe_refusals - before.probe_refusals };
      return { outcome, ms, counts, ...(parts === undefined ? {} : { parts }) };
    },
  };
}

/** A message to a worker and its answer; the worker's wiring is `answerMessage` and nothing else. */
export type WorkerMessage = { id: number; task: FeatureTask };
export type WorkerAnswer = { id: number; result: FeatureResult } | { id: number; defect: string };

/** The answer to one message: a defect (anything but a refusal to featurise) is reported by its message, not thrown, so the pool can fail the run. */
export function answerMessage(service: FeatureService, message: WorkerMessage): WorkerAnswer {
  try {
    return { id: message.id, result: service.run(message.task) };
  } catch (error) {
    return { id: message.id, defect: error instanceof Error ? error.message : "a defect of the feature builder" };
  }
}
