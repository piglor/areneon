// The candidate feature builder, tier A, fifth feature set `probe-outcome-likely/1` (ADR-022 erratum E-22.7.1).
// It holds `probe-outcome/1` byte for byte (`rows`, the four arrays and `outcomes`: the same code, `buildArrays` with the history declarer and
// the outcome extra) plus the array `belief_threats` and the field `belief_threats_refused` in `probe-damage/3`'s row shape: what the moves a
// foe is LIKELY to hold (a committed move-frequency table, revealed moves removed and the rest renormalised) would do to each own Pokemon, with
// the table's inclusion of the move. The few moves priced per foe are chosen by expected threat (inclusion x damage), not by name, and their
// damage comes from a per-game cache of the probe's rolls (`likely-threats.ts`). The values are on ADR-018's assumed foe set plus the move,
// never facts about the real opponent; nothing of the opponent's unrevealed sets can enter. Pure and deterministic: the same bytes with a warm,
// a cold or no cache and in any candidate order.
import type { CandidateFeaturesOutcome } from "./candidate-features-outcome.ts";
import { lazyDeclarer, declarerOf, outcomeExtra } from "./candidate-features-outcome.ts";
import { buildArrays, type CostTrace } from "./candidate-features-2.ts";
import type { BeliefRefusal } from "./candidate-features-3.ts";
import { createFeatureMemo, type FeatureConfigs, type FeatureMemo, FEATURE_SET_LIKELY } from "./candidate-features.ts";
import type { Candidate } from "./choices.ts";
import { DamageAwareError } from "./damage-aware-view.ts";
import type { Engine, PokemonSet } from "./engine.ts";
import { type LikelyRow, likelyThreatRows } from "./likely-threats.ts";
import type { PlayerChannel } from "./observation.ts";

export { FEATURE_SET_LIKELY };
export type { LikelyRow };

export interface CandidateFeaturesLikely extends Omit<CandidateFeaturesOutcome, "feature_set"> {
  feature_set: typeof FEATURE_SET_LIKELY;
  belief_threats: LikelyRow[];
  /** Present exactly when the likelihood table could not answer this decision (the closed code): `belief_threats` is then empty. */
  belief_threats_refused?: BeliefRefusal;
}

/** The features of one decision under `probe-outcome-likely/1`. Throws `FeatureBuildError` as `candidateFeatures2` does. */
export function candidateFeaturesLikely(
  engine: Engine,
  configs: FeatureConfigs,
  channel: PlayerChannel,
  team: readonly Readonly<PokemonSet>[],
  candidates: readonly Pick<Candidate, "choice" | "pending">[],
  memo: FeatureMemo = createFeatureMemo(),
  trace?: CostTrace,
): CandidateFeaturesLikely {
  const outcome = configs.outcome;
  const likely = configs.likely;
  if (outcome === undefined) throw new DamageAwareError("probe-outcome-likely/1 needs the probe-outcome record");
  if (likely === undefined) throw new DamageAwareError("probe-outcome-likely/1 needs its likelihood record");
  const outcomes = outcomeExtra(engine, configs, outcome, memo);
  const out = buildArrays(
    engine,
    configs,
    channel,
    team,
    candidates,
    memo,
    trace,
    FEATURE_SET_LIKELY,
    (context) => ({ ...outcomes(context), ...context.measured("likely", () => likelyThreatRows(engine, configs, likely, memo, channel, context)) }),
    { outcomes: [], belief_threats: [] } as { outcomes: CandidateFeaturesOutcome["outcomes"]; belief_threats: LikelyRow[]; belief_threats_refused?: BeliefRefusal },
    lazyDeclarer(() => declarerOf(engine, configs, outcome, channel)),
  );
  return out as unknown as CandidateFeaturesLikely;
}
