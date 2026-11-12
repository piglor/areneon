// The candidate feature set a run reads, and the builder for it (ADR-022 erratum E-22.3). The play-time actor, the training
// loader and the row recomputation of the game processes all build rows through this one dispatch, so a set's rows are the
// same bytes wherever they are made. `probe-damage/1` is `candidateFeatures`, `probe-damage/2` is `candidateFeatures2`, `probe-damage/3` is `candidateFeatures3`,
// `probe-damage-belief/1` is `candidateFeaturesBelief`, `probe-outcome/1` is `candidateFeaturesOutcome`. `isFeatureSetId` does not accept `probe-outcome/1` yet: the wire contract, the run config
// enums and the encoder (E-22.5 slices H4 and H5) must read it first.
import { candidateFeatures, type CandidateFeatures, type FeatureConfigs, type FeatureMemo, featureSetHash, featureSetIdentity, type FeatureSetId, FEATURE_SET, FEATURE_SET_3, FEATURE_SET_BELIEF, FEATURE_SET_OUTCOME } from "./candidate-features.ts";
import { candidateFeaturesBelief, type CandidateFeaturesBelief } from "./candidate-features-belief.ts";
import { candidateFeatures2, type CandidateFeatures2 } from "./candidate-features-2.ts";
import { candidateFeatures3, type CandidateFeatures3 } from "./candidate-features-3.ts";
import { DamageAwareError } from "./damage-aware-view.ts";
import type { Candidate } from "./choices.ts";
import type { Engine, PokemonSet } from "./engine.ts";
import type { PlayerChannel } from "./observation.ts";

/** The object a decision carries under either set. */
export type AnyCandidateFeatures = CandidateFeatures | CandidateFeatures2 | CandidateFeatures3 | CandidateFeaturesBelief;

/** The set a run reads: the configs' own, `probe-damage/1` when they name none (so every earlier caller is unchanged). */
export const featureSetOf = (configs: Pick<FeatureConfigs, "feature_set">): FeatureSetId => configs.feature_set ?? FEATURE_SET;

/** Whether `name` is a feature set this side can build. */
export const isFeatureSetId = (name: string): name is FeatureSetId => name === FEATURE_SET || name === "probe-damage/2" || name === FEATURE_SET_3 || name === FEATURE_SET_BELIEF;

/** `feature_set_hash` of the set the configs name. */
export const configuredSetHash = (engine: Engine, configs: FeatureConfigs): string => featureSetHash(featureSetIdentity(engine, configs, featureSetOf(configs)));

/** The features of one decision under `set` (default: the configs' own). Throws `FeatureBuildError` as the builders do. */
export function buildFeatures(
  engine: Engine,
  configs: FeatureConfigs,
  channel: PlayerChannel,
  team: readonly Readonly<PokemonSet>[],
  candidates: readonly Pick<Candidate, "choice" | "pending">[],
  memo: FeatureMemo,
  set: FeatureSetId = featureSetOf(configs),
): AnyCandidateFeatures {
  if (set === "probe-damage/2") return candidateFeatures2(engine, configs, channel, team, candidates, memo);
  if (set === FEATURE_SET_3) return candidateFeatures3(engine, configs, channel, team, candidates, memo);
  // The wire, the loader and the sidecar carry `TrainingExample` rows, which have no `outcomes` and no `history_unknown` reason until slice H4 moves the
  // contract: this dispatch refuses the id so no caller writes a value the readers cannot take. The bench and the feature service call
  // `candidateFeaturesOutcome` directly. H4 replaces this throw with the call.
  if (set === FEATURE_SET_OUTCOME) throw new DamageAwareError("probe-outcome/1 cannot be carried on the wire until the contract has its outcomes (slice H4)");
  if (set === FEATURE_SET_BELIEF) return candidateFeaturesBelief(engine, configs, channel, team, candidates, memo);
  return candidateFeatures(engine, configs, channel, team, candidates, memo);
}
