// The candidate feature builder, tier A of ADR-022 (section 3): per-candidate damage rows from the pinned
// engine, ready to be carried beside a `PolicyQuery`. A pure, deterministic function of the player's own
// channel, the player's own team, the candidates and the configs the feature set's hash names. It reads
// the public state through the shared `packages/public-state` reader, builds each opposing active's assumed
// set with ADR-018's assumed-set builder and its reviewed config (revealed facts as revealed, the rest from
// the record; the real opponent team is never read) and asks the ADR-016 probe ONE `damage` query per
// distinct row: (acting slot, move position, target position, Mega flag), never one per joint candidate.
//
// Pending candidates feed rows like any other: a row exists for every move part some candidate uses.
//
// Row content is integers only (ADR-008 profile). An `unsupported` probe answer is an `unsupported` row with
// the probe's closed reason, never filled with a guess. `ohko_rolls` (ADR-022 section 3: "once ADR-016 slice
// 5 exists") is the probe's own count of the 16 no-crit rolls that reach the target's hit points, or null when
// the probe gives no count (`ko` unsupported); nothing is counted here.
//
// Excluded by construction (a type test shows it, as for `buildObservation`): the input is a `PlayerChannel`
// (the player's own checked stream, never an `OmniscientLog`), the player's own team and the candidate
// strings. No opponent team, hidden verdict, `BattleRecord` or seed is a parameter of anything here.
//
// Not here: incoming threats and turn order (`probe-damage/2`), the belief-based sets (ADR-025 slice 5),
// the transport field, the encoder and the worker pool.

import { createInterpreter, type PublicFlag, type PublicMon, readerSha256 } from "@areneon/public-state";
import { type Json, jcs } from "./canonical-json.ts";
import { type AssumedSet, AssumedSetError } from "./damage-aware-assumed-set.ts";
import { parseJointChoice } from "./damage-aware-choice.ts";
import { type DamageAwareConfig, damageAwareConfigHash } from "./damage-aware-config.ts";
import { PPM, rollsValue } from "./damage-aware-score.ts";
import { array, DamageAwareError, damageView, foesHit, object, ProbeRefusal, type ProbeMemo, type ProbeCounts, ReaderGap, text } from "./damage-aware-view.ts";
import type { Engine, PokemonSet } from "./engine.ts";
import type { Candidate } from "./choices.ts";
import { type VersionManifest, resolveFormat } from "./formats.ts";
import { type MechanicsConfig, mechanicsConfigHash } from "./mechanics-config.ts";
import { type MechanicsContext, type ProbeOutcome, probeMechanics } from "./mechanics-probe.ts";
import { type ParsedMechanicsQuery } from "./mechanics-query.ts";
import { MECHANICS_ADAPTER_FORMAT, type MechanicsResult } from "./mechanics-result.ts";
import type { PlayerChannel } from "./observation.ts";
import { loadRulesSnapshot } from "./rules-load.ts";
import type { RulesSnapshot } from "./rules.ts";
import { sha256Hex } from "./rules.ts";
import type { TeamVocabulary } from "./team-vocabulary.ts";

/** The feature set this module builds (ADR-022 section 3). A change of what a row holds is a new id. */
export const FEATURE_SET = "probe-damage/1";

/** Why a request has no features: the caller decides what to do (ADR-022 section 3); nothing is guessed. */
export type FeatureBuildCode = "reader_flagged" | "reader_gap" | "assumed_set" | "probe_refused" | "no_request";

export class FeatureBuildError extends Error {
  override name = "FeatureBuildError";
  readonly code: FeatureBuildCode;
  constructor(code: FeatureBuildCode) {
    super(MESSAGES[code]);
    this.code = code;
  }
}
const MESSAGES: Record<FeatureBuildCode, string> = {
  reader_flagged: "the state reader flagged a line since the previous request",
  reader_gap: "the state reader does not show an own Pokemon the request lists",
  assumed_set: "no legal assumed opponent set could be built",
  probe_refused: "the mechanics probe refused a query the builder declared",
  no_request: "the channel does not end in the request being answered",
};

/** The reviewed records the features depend on; every part is named by `feature_set_hash`. */
export interface FeatureConfigs {
  vocabulary: TeamVocabulary;
  /** ADR-018's assumption record (v1: field rules). */
  config: DamageAwareConfig;
  mechanics: MechanicsConfig;
  /** The rules snapshot; read from `rules/` once per builder call when absent. */
  snapshot?: RulesSnapshot;
  /** The class the probe queries carry (they are never stored). */
  data_class: "synthetic" | "private";
  /** Test seam: the probe (default `probeMechanics`). */
  probe?: (context: MechanicsContext, query: ParsedMechanicsQuery) => ProbeOutcome;
  /** Test seam (the negative control of the invariance test): the assumed opponent set of a public Pokemon (default `buildAssumedSet`). */
  assume?: (mon: PublicMon) => AssumedSet;
}

/** One row: a damaging move part of some candidate against one opposing target. */
export type FeatureRow = { slot: number; move: number; target: number; mega: boolean } & (
  | { kind: "computed"; mean_ppm: number; max_ppm: number; min_ppm: number; ohko_rolls: number | null }
  | { kind: "unsupported"; reason: Extract<MechanicsResult, { kind: "unsupported" }>["reason"] }
);

export interface CandidateFeatures {
  feature_set: typeof FEATURE_SET;
  feature_set_hash: string;
  /** In canonical order: slot, move position, target position, Mega flag (false first). */
  rows: FeatureRow[];
}

/** A game's memo of probe answers by query hash, with the counts that measure it. Create one per game. */
export interface FeatureMemo {
  results: ProbeMemo;
  counts: ProbeCounts;
}
export const createFeatureMemo = (): FeatureMemo => ({ results: new Map(), counts: { queries: 0, probe_runs: 0, unsupported_queries: 0 } });

/** The parts a feature row depends on besides the request (ADR-022 section 3). */
export interface FeatureSetIdentity {
  feature_set: typeof FEATURE_SET;
  mechanics_config_hash: string;
  assumed_set_config_hash: string;
  adapter_format: number;
  reader_sha256: string;
  version_manifest: VersionManifest;
}

export function featureSetIdentity(engine: Engine, configs: Pick<FeatureConfigs, "config" | "mechanics" | "snapshot">): FeatureSetIdentity {
  const snapshot = configs.snapshot ?? loadRulesSnapshot(engine.commit);
  return {
    feature_set: FEATURE_SET,
    mechanics_config_hash: mechanicsConfigHash(configs.mechanics),
    assumed_set_config_hash: damageAwareConfigHash(configs.config),
    adapter_format: MECHANICS_ADAPTER_FORMAT,
    reader_sha256: readerSha256(),
    version_manifest: resolveFormat(engine, configs.config.format_id, snapshot).manifest,
  };
}

/** `feature_set_hash`: SHA-256 of the JCS bytes of the identity. */
export const featureSetHash = (identity: FeatureSetIdentity): string => sha256Hex(jcs(identity as unknown as Json));

const REQUEST = "|request|";

/** The index of the last `|request|` line at or before `end` (exclusive), or -1. */
const requestBefore = (lines: readonly string[], end: number): number => {
  for (let index = end - 1; index >= 0; index--) if ((lines[index] as string).startsWith(REQUEST)) return index;
  return -1;
};

/** Parts per million of `value` out of `hp`, floored. */
const share = (value: number, hp: number): number => Math.floor((value * PPM) / hp);

/**
 * The rows of one decision: one damage query per distinct (slot, move position, target position, Mega flag) that
 * some candidate uses with a damaging move into an opposing active. `memo` is this game's: with or without it the
 * rows are the same bytes. A request with no move to score (Team Preview, a forced switch, a wait) has no rows.
 * Throws `FeatureBuildError` when the request cannot be featured (the caller decides what then).
 */
export function candidateFeatures(
  engine: Engine,
  configs: FeatureConfigs,
  channel: PlayerChannel,
  team: readonly Readonly<PokemonSet>[],
  candidates: readonly Pick<Candidate, "choice" | "pending">[],
  memo: FeatureMemo = createFeatureMemo(),
): CandidateFeatures {
  const snapshot = configs.snapshot ?? loadRulesSnapshot(engine.commit);
  const identity = featureSetIdentity(engine, { ...configs, snapshot });
  const out = (rows: FeatureRow[]): CandidateFeatures => ({ feature_set: FEATURE_SET, feature_set_hash: featureSetHash(identity), rows });

  const lines = channel.lines;
  const last = lines.length - 1;
  if (last < 0 || !(lines[last] as string).startsWith(REQUEST)) throw new FeatureBuildError("no_request");
  // `playerChannel` has checked every `|request|` line, so this parses.
  const body = object(JSON.parse((lines[last] as string).slice(REQUEST.length)) as unknown, "request");
  // Only a move request has a damaging move to featurise; the others have no active entries to read.
  if (body.teamPreview !== undefined || body.active === undefined) return out([]);

  const view = damageView({
    engine, vocabulary: configs.vocabulary, config: configs.config, v1: configs.config, mechanics: configs.mechanics, snapshot,
    data_class: configs.data_class, probe: configs.probe ?? probeMechanics, counts: memo.counts, memo: memo.results,
    engine_targets: true, // the engine's choice-location convention (#1180), whatever the assumed-set record says
    ...(configs.assume === undefined ? {} : { assume: configs.assume }),
  });

  // The reader, fed the whole channel; a line it could not take since the previous request makes the state untrustworthy (as for the baseline).
  const reader = createInterpreter({ player: channel.player });
  const since = requestBefore(lines, last);
  const flags: PublicFlag[] = [];
  lines.forEach((line, index) => {
    const reason = reader.feed(line).flagged;
    if (reason !== null && index > since) flags.push({ index, line, reason });
  });
  if (flags.length > 0) throw new FeatureBuildError("reader_flagged");
  const state = reader.snapshot();

  try {
    const own = view.ownSlots(body, state, team);
    const active = array(body.active, "active");
    let foes: ReturnType<typeof view.foesOf> | null = null;
    const foesNow = (): ReturnType<typeof view.foesOf> => (foes ??= view.foesOf(state, (mon) => view.assumedFor(mon, state)));
    const rows = new Map<string, FeatureRow>();

    for (const { choice } of candidates) {
      const parts = parseJointChoice(choice);
      if (parts.length !== active.length) throw new DamageAwareError("a choice has a different number of parts than the request has active Pokemon");
      parts.forEach((part, slot) => {
        if (part.kind !== "move") return;
        const moves = array(object(active[slot], "an active entry").moves, "moves");
        const slotMove = object(moves[part.index - 1], "a move of the request");
        const id = text(slotMove.id, "a move id");
        const data = view.dex.moves.get(id);
        if (!data.exists || data.category === "Status") return;
        const type = typeof slotMove.target === "string" ? slotMove.target : undefined;
        for (const foe of foesHit(type, part.target, foesNow)) {
          const key = `${slot}/${part.index}/${foe.loc}/${part.mega}`;
          if (rows.has(key)) continue;
          const mine = (own[slot] as (typeof own)[number]).fighter;
          const result = view.damageResult({ ...mine, mega: mine.mega || part.mega }, foe.fighter, id, state, true);
          rows.set(key, rowOf({ slot, move: part.index, target: foe.loc, mega: part.mega }, result));
        }
      });
    }
    return out([...rows.values()].sort((a, b) => a.slot - b.slot || a.move - b.move || a.target - b.target || Number(a.mega) - Number(b.mega)));
  } catch (error) {
    if (error instanceof ReaderGap) throw new FeatureBuildError("reader_gap");
    if (error instanceof AssumedSetError) throw new FeatureBuildError("assumed_set");
    if (error instanceof ProbeRefusal) throw new FeatureBuildError("probe_refused");
    throw error;
  }
}

/** One row from the probe's answer: integers from the engine's rolls, or the probe's closed reason. */
function rowOf(key: { slot: number; move: number; target: number; mega: boolean }, result: MechanicsResult | null): FeatureRow {
  if (result?.kind === "unsupported") return { ...key, kind: "unsupported", reason: result.reason };
  if (result?.kind !== "computed" || result.answer.kind !== "damage") throw new DamageAwareError("the probe gave no damage answer to a damage query");
  const { no_crit: rolls, hp_current: hp, ko } = result.answer;
  return {
    ...key,
    kind: "computed",
    mean_ppm: rollsValue(rolls, hp, 0),
    max_ppm: share(Math.min(Math.max(...rolls), hp), hp),
    min_ppm: share(Math.min(Math.min(...rolls), hp), hp),
    ohko_rolls: ko.kind === "counts" ? ko.ohko_no_crit : null,
  };
}
