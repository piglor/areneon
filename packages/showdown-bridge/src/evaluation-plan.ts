// The sealed-set commitment (protocol L3) and the registered plan (protocol
// I2) of ADR-010: the data model only. Both hold hashes and counts, never
// sealed content: the teams, the salt and the key stay in the isolated store
// and appear only in a reveal on retirement (L5). The readers check what the
// contract profile cannot say, against the config records the documents name.
//
// Engine-free and Node-only, like `evaluation-config.ts`.

import { createHash, randomBytes } from "node:crypto";
import { ajvFor } from "@areneon/schema/ajv";
import planSchema from "@areneon/schema/contracts/v1/evaluation-plan.schema.json" with { type: "json" };
import commitmentSchema from "@areneon/schema/contracts/v1/sealed-set-commitment.schema.json" with { type: "json" };
import type { EvaluationPlan } from "@areneon/schema/types/v1/evaluation-plan.ts";
import type { SealedSetCommitment } from "@areneon/schema/types/v1/sealed-set-commitment.ts";
import { jcs, type Json } from "./canonical-json.ts";
import { alphaPrimePpm, deriveClaims, domainHash, type EvaluationConfig, EvaluationConfigError, requireComplete, strictlyAscending } from "./evaluation-config.ts";

export type { EvaluationPlan, SealedSetCommitment };

const validCommitment = ajvFor("contract").compile(commitmentSchema);
const validPlan = ajvFor("contract").compile(planSchema);

/** SHA-256 of the JCS bytes of a commitment record or a plan: how the register names them. */
export const commitmentHash = (commitment: SealedSetCommitment): string => domainHash("sealed-set-commitment", commitment);
export const planHash = (plan: EvaluationPlan): string => domainHash("evaluation-plan", plan);

const sha256Text = (text: string): string => createHash("sha256").update(text, "utf8").digest("hex");

const HEX256 = /^[0-9a-f]{64}$/;

/** L3: SHA-256(JCS({salt, teams})). The salt is 256 random bits, hex, so a short team list cannot be guessed by hashing candidates. */
export function setCommitment(salt: string, teams: readonly Json[]): string {
  if (!HEX256.test(salt)) throw new EvaluationConfigError("the salt must be 64 lowercase hex characters (256 random bits)");
  return sha256Text(jcs({ salt, teams: [...teams] }));
}

/** L3: the commitment to the secret seed key (64 lowercase hex characters). */
export function keyCommitment(key: string): string {
  if (!HEX256.test(key)) throw new EvaluationConfigError("the seed key must be 64 lowercase hex characters (256 random bits)");
  return sha256Text(key);
}

/** 256 random bits as 64 hex characters: a salt or a seed key. */
export const generateSecret = (): string => randomBytes(32).toString("hex");

/**
 * L5: whether a published reveal opens a commitment, and is the set it
 * declared: the hashes match, the number of teams is `team_count`, and the
 * strata the teams fall into (by `stratumOf`, the reviewed labeller) are
 * exactly `stratum_counts`, each holding at least the PARAMETERS' `n_stratum_min`
 * (erratum E1; the caller cannot lower it). `stratumOf` is the reviewed
 * labeller of ADR-010 O6 (a data record naming engine ids; no ids in code); its
 * identity hash is meant to be pinned in the registered plan, which has no field
 * for it yet (a follow-up on #846). Anyone can run it once the set retires.
 */
export function checkReveal(
  commitment: SealedSetCommitment,
  reveal: { salt: string; teams: readonly Json[]; key: string },
  stratumOf: (team: Json) => string,
  config: EvaluationConfig,
): boolean {
  if (setCommitment(reveal.salt, reveal.teams) !== commitment.set_commitment || keyCommitment(reveal.key) !== commitment.key_commitment) return false;
  if (reveal.teams.length !== commitment.team_count) return false;
  const counts = new Map<string, number>();
  for (const team of reveal.teams) {
    const stratum = stratumOf(team);
    counts.set(stratum, (counts.get(stratum) ?? 0) + 1);
  }
  const declared = commitment.stratum_counts;
  return counts.size === declared.length && declared.every((s) => counts.get(s.stratum) === s.count && s.count >= config.parameters.n_stratum_min);
}

/** Read a SealedSetCommitment: the schema, then the sorted-and-unique strata and their sum. */
export function readSealedSetCommitment(value: unknown): SealedSetCommitment {
  if (!validCommitment(value)) throw new EvaluationConfigError(`invalid SealedSetCommitment: ${JSON.stringify(validCommitment.errors)}`);
  const c = structuredClone(value) as unknown as SealedSetCommitment;
  if (!strictlyAscending(c.stratum_counts.map((s) => s.stratum))) throw new EvaluationConfigError("the strata must be sorted by label and unique");
  const total = c.stratum_counts.reduce((sum, s) => sum + s.count, 0);
  if (total !== c.team_count) throw new EvaluationConfigError(`the strata hold ${total} teams but team_count is ${c.team_count}`);
  return c;
}

/**
 * The commitment against the config records it names (erratum E1 included):
 * the hashes match, the parameters are complete, the count is n_teams, the
 * budget is b_sealed and every stratum holds at least n_stratum_min teams.
 */
export function checkCommitment(commitment: SealedSetCommitment, config: EvaluationConfig): void {
  const { parameters: p } = config;
  requireComplete(p);
  if (commitment.evaluation_parameters_hash !== config.parameters_hash) throw new EvaluationConfigError("the commitment names other EvaluationParameters than the ones given");
  if (commitment.training_gate_rule_hash !== config.rule_hash) throw new EvaluationConfigError("the commitment names another TrainingGateRule than the one given");
  if (commitment.team_count !== p.n_teams) throw new EvaluationConfigError(`team_count ${commitment.team_count} differs from the parameters' n_teams ${p.n_teams}`);
  if (commitment.b_sealed !== p.b_sealed) throw new EvaluationConfigError(`b_sealed ${commitment.b_sealed} differs from the parameters' ${p.b_sealed}`);
  const thin = commitment.stratum_counts.filter((s) => s.count < p.n_stratum_min);
  if (thin.length > 0) {
    throw new EvaluationConfigError(`strata below n_stratum_min ${p.n_stratum_min}: ${thin.map((s) => `${s.stratum} (${s.count})`).join(", ")}`);
  }
}

/** Read an EvaluationPlan: the schema, then roles in order and opponents sorted and unique. */
export function readEvaluationPlan(value: unknown): EvaluationPlan {
  if (!validPlan(value)) throw new EvaluationConfigError(`invalid EvaluationPlan: ${JSON.stringify(validPlan.errors)}`);
  const plan = structuredClone(value) as unknown as EvaluationPlan;
  for (const [what, pair] of [["candidate", plan.candidate], ["progress_reference", plan.progress_reference]] as const) {
    if (pair[0].role !== "battle" || pair[1].role !== "preview") throw new EvaluationConfigError(`${what} must list the battle checkpoint, then the preview checkpoint`);
  }
  if (!strictlyAscending(plan.opponents.map((o) => o.opponent_id))) throw new EvaluationConfigError("the opponents must be sorted by opponent_id and unique");
  for (const o of plan.opponents) {
    if ((o.pool === "B") !== (o.lineage_id === undefined)) throw new EvaluationConfigError(`opponent ${o.opponent_id}: a scripted baseline has no lineage and every other opponent has one`);
  }
  if (new Set(plan.claims.map((c) => c.claim_id)).size !== plan.claims.length) throw new EvaluationConfigError("a claim is listed twice");
  return plan;
}

/**
 * The plan against the records it names. Refused: provisional parameters;
 * other records or another commitment than the plan's hashes; levels or sizes
 * that differ from the parameters' or the commitment's; a family that differs
 * from the rule's claims; an opponent of the wrong pool or lineage; the
 * candidate among its own opponents.
 */
export function checkPlan(plan: EvaluationPlan, context: { config: EvaluationConfig; commitment: SealedSetCommitment }): void {
  const { config, commitment } = context;
  const p = config.parameters;
  checkCommitment(commitment, config);
  if (plan.set_id !== commitment.set_id) throw new EvaluationConfigError(`the plan is for set ${plan.set_id} but the commitment is for ${commitment.set_id}`);
  if (plan.set_commitment_hash !== commitmentHash(commitment)) throw new EvaluationConfigError("the plan names another SealedSetCommitment than the one given");
  if (plan.evaluation_parameters_hash !== config.parameters_hash) throw new EvaluationConfigError("the plan names other EvaluationParameters than the ones given");
  if (plan.training_gate_rule_hash !== config.rule_hash) throw new EvaluationConfigError("the plan names another TrainingGateRule than the one given");
  const expected = { alpha_ppm: p.alpha_ppm, alpha_prime_ppm: alphaPrimePpm(p), b_boot: p.b_boot };
  for (const [field, value] of Object.entries(expected)) {
    if (plan[field as keyof typeof expected] !== value) throw new EvaluationConfigError(`${field} ${plan[field as keyof typeof expected]} differs from the parameters' ${value}`);
  }
  if (plan.block_plan.n_teams !== p.n_teams || plan.block_plan.n_seeds !== p.n_seeds) {
    throw new EvaluationConfigError(`the block plan (${plan.block_plan.n_teams} teams, ${plan.block_plan.n_seeds} seeds) differs from the parameters' (${p.n_teams}, ${p.n_seeds})`);
  }
  const derived = deriveClaims(config.rule);
  if (plan.claims.length !== derived.length) throw new EvaluationConfigError(`the plan lists ${plan.claims.length} claims, the rule ${derived.length}`);
  const opponents = new Map(plan.opponents.map((o) => [o.opponent_id, o]));
  for (const [i, claim] of derived.entries()) {
    const planned = plan.claims[i] as EvaluationPlan["claims"][number];
    if (planned.claim_id !== claim.claim_id || planned.gate !== claim.gate || planned.margin_null_ppm !== claim.margin_null_ppm) {
      throw new EvaluationConfigError(`claim ${i} (${planned.claim_id}) differs from the rule's ${claim.claim_id}`);
    }
    const opponent = opponents.get(planned.opponent_id);
    if (opponent === undefined) throw new EvaluationConfigError(`${claim.claim_id}: opponent ${planned.opponent_id} is not among the plan's opponents`);
    const fits = claim.gate === "G2" ? opponent.pool === "B" && opponent.opponent_id === claim.opponent : opponent.pool === "F" && opponent.lineage_id === claim.opponent;
    if (!fits) {
      throw new EvaluationConfigError(`${claim.claim_id}: opponent ${opponent.opponent_id} is not ${claim.gate === "G2" ? `the scripted baseline ${claim.opponent}` : `a frozen checkpoint of the held-out lineage ${claim.opponent}`}`);
    }
  }
  const own = new Set([...plan.candidate, ...plan.progress_reference].map((c) => c.checkpoint_id));
  for (const o of plan.opponents) {
    if (own.has(o.opponent_id)) throw new EvaluationConfigError(`${o.opponent_id} is both a candidate or progress reference and an opponent`);
  }
}
