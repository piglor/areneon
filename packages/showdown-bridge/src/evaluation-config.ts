// The evaluation config records (ADR-010 sections 6, 7): `EvaluationParameters`
// and `TrainingGateRule`. Their values live in reviewed JSON under
// `config/evaluation/`, never here. This module holds only what the contract
// profile cannot say: the arithmetic the ADR derives from those values and the
// outer admissibility rules, so an inconsistent record is refused (fail
// closed). Both records are referenced only by the SHA-256 of their JCS bytes.
//
// Engine-free and Node-only (`node:crypto`): the register verifier and the
// plan registration use it without loading the engine.

import { createHash } from "node:crypto";
import { ajvFor } from "@areneon/schema/ajv";
import parametersSchema from "@areneon/schema/contracts/v1/evaluation-parameters.schema.json" with { type: "json" };
import ruleSchema from "@areneon/schema/contracts/v1/training-gate-rule.schema.json" with { type: "json" };
import type { EvaluationParameters } from "@areneon/schema/types/v1/evaluation-parameters.ts";
import type { TrainingGateRule } from "@areneon/schema/types/v1/training-gate-rule.ts";
import { jcs, type Json } from "./canonical-json.ts";

export type { EvaluationParameters, TrainingGateRule };

/** The unit of every ratio in the records: parts per million. */
export const PPM = 1_000_000;

export class EvaluationConfigError extends Error {
  override name = "EvaluationConfigError";
}

/** SHA-256 (hex) of a JSON value's JCS bytes: how every record is referenced. */
export function jsonHash(value: unknown): string {
  return createHash("sha256").update(jcs(value as Json)).digest("hex");
}

const validParameters = ajvFor("contract").compile(parametersSchema);
const validRule = ajvFor("contract").compile(ruleSchema);

/** True when `items` is strictly ascending: sorted and without a repeat. */
export const strictlyAscending = (items: readonly (string | number)[]): boolean => items.every((x, i) => i === 0 || (items[i - 1] as string | number) < x);

// ---------------------------------------------------------------------------
// Normal quantiles. The half-width h follows from the target power through the
// standard normal quantile (ADR-010 section 6.1); no table of z values is kept.

/** erf(x), from the series with positive terms: stable for every x. */
function erf(x: number): number {
  const a = Math.abs(x);
  let term = a;
  let sum = a;
  for (let n = 1; n < 2000 && term > sum * 1e-18; n++) {
    term *= (2 * a * a) / (2 * n + 1);
    sum += term;
  }
  return Math.sign(x) * ((2 / Math.sqrt(Math.PI)) * Math.exp(-a * a) * sum);
}

/** The standard normal CDF. */
const cdf = (x: number): number => 0.5 * (1 + erf(x / Math.SQRT2));

/** The quantile z with P(Z <= z) = p, for 0 < p < 1, by bisection. */
export function normalQuantile(p: number): number {
  if (!(p > 0 && p < 1)) throw new RangeError(`normalQuantile: ${p} is not in (0, 1)`);
  let lo = -40;
  let hi = 40;
  for (let i = 0; i < 200; i++) {
    const mid = (lo + hi) / 2;
    if (cdf(mid) < p) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

// ---------------------------------------------------------------------------
// EvaluationParameters

/** alpha' = alpha / (m * B_sealed), in ppm. Refused unless it is a whole number of ppm. */
export function alphaPrimePpm(p: Pick<EvaluationParameters, "alpha_ppm" | "family_size" | "b_sealed">): number {
  const divisor = p.family_size * p.b_sealed;
  if (p.alpha_ppm % divisor !== 0) {
    throw new EvaluationConfigError(`alpha' = ${p.alpha_ppm} ppm / (${p.family_size} * ${p.b_sealed}) is not a whole number of ppm`);
  }
  return p.alpha_ppm / divisor;
}

/** cov_tol in ppm: the multiple of the Monte Carlo standard error sqrt(a'(1-a')/R_cov) of a coverage estimate. */
export function covTolPpm(p: EvaluationParameters): number {
  const a = alphaPrimePpm(p) / PPM;
  return Math.ceil(p.cov_tol_se_multiple * Math.sqrt((a * (1 - a)) / p.r_cov) * PPM);
}

/** True when every measured value is in and d_team is measured: the only state a plan may name. */
export function isComplete(p: EvaluationParameters): boolean {
  return p.n_teams_floor !== null && p.n_teams !== null && p.n_seeds !== null && p.d_team.basis === "measured";
}

/** Throws unless the parameters are complete (nothing Unknown is registered as if known). */
export function requireComplete(p: EvaluationParameters): void {
  const missing = [
    ...(["n_teams_floor", "n_teams", "n_seeds"] as const).filter((k) => p[k] === null),
    ...(p.d_team.basis === "measured" ? [] : ["d_team (still the starting value)"]),
  ];
  if (missing.length > 0) {
    throw new EvaluationConfigError(`the EvaluationParameters are provisional: ${missing.join(", ")} not measured yet`);
  }
}

const sortedUnique = (what: string, items: readonly string[]): void => {
  if (!strictlyAscending(items)) throw new EvaluationConfigError(`${what} must be sorted and unique`);
};

/**
 * Read an EvaluationParameters record: the schema, then the rules the profile
 * cannot state. Provisional records (measured values still null) are valid;
 * `requireComplete` is the gate for registering a plan.
 */
export function readEvaluationParameters(value: unknown): EvaluationParameters {
  if (!validParameters(value)) throw new EvaluationConfigError(`invalid EvaluationParameters: ${JSON.stringify(validParameters.errors)}`);
  const p = structuredClone(value) as unknown as EvaluationParameters;
  const alphaPrime = alphaPrimePpm(p);
  // The protocol's lower bounds: B_boot far above m*B/alpha; R_cov at least 1/alpha'.
  if (p.b_boot * p.alpha_ppm <= p.family_size * p.b_sealed * PPM) {
    throw new EvaluationConfigError(`b_boot ${p.b_boot} must exceed family_size * b_sealed / alpha = ${(p.family_size * p.b_sealed * PPM) / p.alpha_ppm}`);
  }
  if (p.r_cov * alphaPrime < PPM) throw new EvaluationConfigError(`r_cov ${p.r_cov} must be at least 1 / alpha' = ${PPM / alphaPrime}`);
  sortedUnique("pool.baselines", p.pool.baselines);
  sortedUnique("pool.seen_lineages", p.pool.seen_lineages);
  sortedUnique("pool.held_out_lineages", p.pool.held_out_lineages);
  const seen = new Set(p.pool.seen_lineages);
  const shared = p.pool.held_out_lineages.filter((l) => seen.has(l));
  if (shared.length > 0) throw new EvaluationConfigError(`a lineage cannot be both seen and held out: ${shared.join(", ")}`);
  if (p.n_teams !== null && (p.n_teams_floor === null || p.n_seeds === null)) {
    throw new EvaluationConfigError("n_teams is set, so n_teams_floor and n_seeds must be measured too");
  }
  if (p.n_teams !== null && p.n_teams_floor !== null && p.n_teams < p.n_teams_floor) {
    throw new EvaluationConfigError(`n_teams ${p.n_teams} is below its floor ${p.n_teams_floor}`);
  }
  return p;
}

// ---------------------------------------------------------------------------
// TrainingGateRule

export type Gate = TrainingGateRule["claims"][number]["gate"];

/** One gate claim with the numbers the rule derives for it. */
export interface DerivedClaim {
  claim_id: string;
  gate: Gate;
  opponent: string;
  /** theta_0: chance plus the gate's margin (G2, G3), or 0 for the paired difference (G3b). */
  margin_null_ppm: number;
  /** theta_1: the owner's alternative score or difference. */
  alternative_ppm: number;
}

/** The claims of the rule with their margin nulls and alternatives, in the rule's order. */
export function deriveClaims(rule: TrainingGateRule): DerivedClaim[] {
  return rule.claims.map(({ claim_id, gate, opponent }) => {
    if (gate === "G2") {
      return { claim_id, gate, opponent, margin_null_ppm: rule.chance_ppm + rule.margins_ppm.floor, alternative_ppm: rule.alternatives_ppm.floor };
    }
    if (gate === "G3") {
      return { claim_id, gate, opponent, margin_null_ppm: rule.chance_ppm + rule.margins_ppm.held_out, alternative_ppm: rule.alternatives_ppm.held_out };
    }
    return { claim_id, gate, opponent, margin_null_ppm: 0, alternative_ppm: rule.alternatives_ppm.progress_gain };
  });
}

/** Read a TrainingGateRule: the schema, then what the profile cannot state. */
export function readTrainingGateRule(value: unknown): TrainingGateRule {
  if (!validRule(value)) throw new EvaluationConfigError(`invalid TrainingGateRule: ${JSON.stringify(validRule.errors)}`);
  const rule = structuredClone(value) as unknown as TrainingGateRule;
  const ids = rule.claims.map((c) => c.claim_id);
  if (!strictlyAscending(ids)) throw new EvaluationConfigError("the claims must be sorted by claim_id and unique");
  if (!strictlyAscending(rule.selection.reference_fractions_ppm)) {
    throw new EvaluationConfigError("selection.reference_fractions_ppm must be strictly ascending");
  }
  sortedUnique("required_checks", rule.required_checks);
  for (const claim of deriveClaims(rule)) {
    if (claim.margin_null_ppm >= PPM) throw new EvaluationConfigError(`${claim.claim_id}: the margin null ${claim.margin_null_ppm} ppm leaves no room above it`);
    if (claim.alternative_ppm <= claim.margin_null_ppm) {
      throw new EvaluationConfigError(`${claim.claim_id}: the alternative ${claim.alternative_ppm} ppm must lie above the margin null ${claim.margin_null_ppm} ppm`);
    }
  }
  return rule;
}

/**
 * The largest half-width h (ppm) that passes a claim with the target power
 * (ADR-010 section 6.1, step 2): with SE = h / z(1 - alpha'/2), the claim
 * passes when the estimate minus z(1 - alpha') * SE exceeds the margin null,
 * so power `target` at the alternative needs
 * h <= z(1 - alpha'/2) / (z(1 - alpha') + z(target)) * (theta_1 - theta_0).
 */
export function maxHalfWidthPpm(alphaPrimePpmValue: number, targetPowerPpm: number, gapPpm: number): number {
  const a = alphaPrimePpmValue / PPM;
  const twoSided = normalQuantile(1 - a / 2);
  const oneSided = normalQuantile(1 - a);
  const power = normalQuantile(targetPowerPpm / PPM);
  return (twoSided / (oneSided + power)) * gapPpm;
}

/** The half-width the parameters set for a gate. */
export const halfWidthOf = (p: EvaluationParameters, gate: Gate): number => (gate === "G2" ? p.half_width_ppm.g2 : gate === "G3" ? p.half_width_ppm.g3 : p.half_width_ppm.g3b);

/**
 * The two records together (neither alone can say it): the family size is the
 * number of claims; G2 claims name each scripted baseline exactly once, G3 and
 * G3b claims a held-out lineage; and each gate's h is within what the target
 * power allows at the owner's alternative.
 */
export function checkGateConsistency(parameters: EvaluationParameters, rule: TrainingGateRule): void {
  if (rule.claims.length !== parameters.family_size) {
    throw new EvaluationConfigError(`the rule lists ${rule.claims.length} claims but the parameters' family_size is ${parameters.family_size}`);
  }
  const baselines = rule.claims.filter((c) => c.gate === "G2").map((c) => c.opponent);
  if (JSON.stringify([...baselines].sort()) !== JSON.stringify(parameters.pool.baselines)) {
    throw new EvaluationConfigError(`the G2 claims (${baselines.join(", ")}) must name each scripted baseline (${parameters.pool.baselines.join(", ")}) exactly once`);
  }
  const heldOut = new Set(parameters.pool.held_out_lineages);
  for (const claim of rule.claims.filter((c) => c.gate !== "G2")) {
    if (!heldOut.has(claim.opponent)) throw new EvaluationConfigError(`${claim.claim_id}: ${claim.opponent} is not a held-out lineage`);
  }
  const alphaPrime = alphaPrimePpm(parameters);
  for (const claim of deriveClaims(rule)) {
    const bound = maxHalfWidthPpm(alphaPrime, rule.target_power_ppm, claim.alternative_ppm - claim.margin_null_ppm);
    const h = halfWidthOf(parameters, claim.gate);
    if (h > bound) {
      throw new EvaluationConfigError(`${claim.claim_id}: h ${h} ppm exceeds ${Math.floor(bound)} ppm, the most that reaches the target power at its alternative`);
    }
  }
}

/** Both records read, cross-checked and hashed: what a sealed-set commitment and a plan name. */
export interface EvaluationConfig {
  parameters: EvaluationParameters;
  rule: TrainingGateRule;
  parameters_hash: string;
  rule_hash: string;
}

export function readEvaluationConfig(parametersValue: unknown, ruleValue: unknown): EvaluationConfig {
  const parameters = readEvaluationParameters(parametersValue);
  const rule = readTrainingGateRule(ruleValue);
  checkGateConsistency(parameters, rule);
  return { parameters, rule, parameters_hash: jsonHash(parameters), rule_hash: jsonHash(rule) };
}
