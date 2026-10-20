// The evaluation contracts' share of the sample corpus (ADR-008 §2) and the
// fixtures the tests build on. Engine-free. The two config records are read
// from `config/evaluation/` (their committed v1 values are data); every other
// value here is a placeholder that pins a shape, never a real sealed set:
// hashes are repeated characters, the sealed "teams" are never built.

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { type EvaluationConfig, PARAMETERS_FILE, readEvaluationConfig, RULE_FILE } from "./evaluation-config.ts";
import { commitmentHash, planHash, type EvaluationPlan, type SealedSetCommitment } from "./evaluation-plan.ts";
import { appendEntry, type EvaluationRegisterEntry } from "./evaluation-register.ts";
import { sealedManifestHash, type SealedTeamSetManifest } from "./sealed-records.ts";

export { CONFIG_DIR, PARAMETERS_FILE, RULE_FILE } from "./evaluation-config.ts";

const readJson = (path: string): Record<string, unknown> => JSON.parse(readFileSync(path, "utf8")) as Record<string, unknown>;

/** A copy of `value` with `change` applied: one mutation per sample. */
function mutated<T>(value: T, change: (copy: Record<string, unknown>) => void): Record<string, unknown> {
  const copy = structuredClone(value) as unknown as Record<string, unknown>;
  change(copy);
  return copy;
}

const hexOf = (c: string): string => c.repeat(64);

export interface EvaluationFixtures {
  /** The committed v1 records: the parameters are provisional (measured values still null). */
  provisional: Record<string, unknown>;
  rule: Record<string, unknown>;
  /** Illustrative measured values, for tests and samples only. */
  complete: Record<string, unknown>;
  config: EvaluationConfig;
  commitment: SealedSetCommitment;
  plan: EvaluationPlan;
  /** A register that commits a set, registers its plan, re-checks, reports and retires. */
  chain: { text: string; entries: EvaluationRegisterEntry[] };
}

export function evaluationFixtures(): EvaluationFixtures {
  const provisional = readJson(PARAMETERS_FILE);
  const rule = readJson(RULE_FILE);
  const complete = mutated(provisional, (p) => {
    p.n_teams_floor = 24;
    p.n_teams = 32;
    p.n_seeds = 1;
    p.d_team = { differing_fields: 2, basis: "measured" };
  });
  const config = readEvaluationConfig(complete, rule);
  const commitment: SealedSetCommitment = {
    schema_version: 1,
    set_id: "set-1",
    generation: 1,
    set_commitment: hexOf("1"),
    key_commitment: hexOf("2"),
    team_count: 32,
    stratum_counts: ["a", "b", "c", "d"].map((x) => ({ stratum: `speed=${x}`, count: 8 })) as SealedSetCommitment["stratum_counts"],
    evaluation_parameters_hash: config.parameters_hash,
    training_gate_rule_hash: config.rule_hash,
    b_sealed: 1,
  };
  const pair = (a: string, b: string) => [
    { role: "battle" as const, checkpoint_id: hexOf(a), provenance_record_hash: hexOf(b) },
    { role: "preview" as const, checkpoint_id: hexOf(b), provenance_record_hash: hexOf(a) },
  ];
  const heldOut = hexOf("a");
  const plan: EvaluationPlan = {
    schema_version: 1,
    set_id: "set-1",
    set_commitment_hash: commitmentHash(commitment),
    evaluation_parameters_hash: config.parameters_hash,
    training_gate_rule_hash: config.rule_hash,
    candidate: pair("3", "4") as unknown as EvaluationPlan["candidate"],
    progress_reference: pair("5", "6") as unknown as EvaluationPlan["progress_reference"],
    opponents: [
      { opponent_id: heldOut, definition_hash: hexOf("7"), pool: "F", lineage_id: "l2" },
      { opponent_id: "actor-rule", definition_hash: hexOf("8"), pool: "B" },
      { opponent_id: "random-legal", definition_hash: hexOf("9"), pool: "B" },
    ],
    claims: [
      { claim_id: "g2-actor-rule", gate: "G2", opponent_id: "actor-rule", direction: "greater", margin_null_ppm: 750000 },
      { claim_id: "g2-random-legal", gate: "G2", opponent_id: "random-legal", direction: "greater", margin_null_ppm: 750000 },
      { claim_id: "g3-held-out-l2", gate: "G3", opponent_id: heldOut, direction: "greater", margin_null_ppm: 550000 },
      { claim_id: "g3b-progress-l2", gate: "G3b", opponent_id: heldOut, direction: "greater", margin_null_ppm: 0 },
    ],
    alpha_ppm: 50000,
    alpha_prime_ppm: 12500,
    b_boot: 20000,
    interval: { method: "percentile_bootstrap", fallback: "bca" },
    block_plan: { n_teams: 32, n_seeds: 1 },
    learner_step_budget: 1000000,
    a2_matches_min: 10,
    harness_commit: "c".repeat(40),
    report_template_hash: hexOf("d"),
    showdown_commit: "7244fe8e28fa8bc4d0809ae30d183c608b62bb2b",
    format_id: "gen9championsvgc2026regmc",
    format_rules_hash: hexOf("e"),
  };
  const at = (n: number) => `2026-11-0${n}T10:00:00Z`;
  let text = "";
  const entries: EvaluationRegisterEntry[] = [];
  const add = (draft: Parameters<typeof appendEntry>[1]) => {
    const done = appendEntry(text, draft);
    text = done.text;
    entries.push(done.entry);
  };
  add({ kind: "set_commitment", recorded_at: at(1), operator: "owner", commitment });
  add({ kind: "plan_registration", recorded_at: at(2), operator: "owner", plan });
  add({ kind: "l2a_recheck", recorded_at: at(3), operator: "evaluation-job", set_id: "set-1", plan_hash: planHash(plan), result: "pass" });
  add({ kind: "run_outcome", recorded_at: at(4), operator: "evaluation-job", plan_hash: planHash(plan), outcome: "reported", outcome_hash: hexOf("f") });
  add({ kind: "set_retirement", recorded_at: at(5), operator: "owner", set_id: "set-1", reason: "budget_spent", reveal_hash: hexOf("0") });
  return { provisional, rule, complete, config, commitment, plan, chain: { text, entries } };
}

/** The evaluation contracts' valid, invalid and reader-invalid samples, keyed `<contract>/<case>`. */
export function evaluationSamples(): { valid: Record<string, unknown>; invalid: Record<string, unknown>; readerInvalid: Record<string, unknown> } {
  const { provisional, rule, complete, commitment, plan, chain } = evaluationFixtures();
  const entry = (kind: string): EvaluationRegisterEntry => chain.entries.find((e) => e.kind === kind) as EvaluationRegisterEntry;
  const incident = appendEntry(`${chain.text.split("\n").slice(0, 3).join("\n")}\n`, {
    kind: "incident",
    recorded_at: "2026-11-04T10:00:00Z",
    operator: "owner",
    set_id: "set-1",
    cause: "sealed_artifact_read",
    description: "A training container listed the store mount.",
    affected_candidates: [hexOf("3")],
  }).entry;
  const run = entry("run_outcome");
  const head = { schema_version: 1, seq: chain.entries.length, entry_hash: hexOf("a") };
  const params = (change: (p: Record<string, unknown>) => void) => mutated(complete, change);
  const withRule = (change: (r: Record<string, unknown>) => void) => mutated(rule, change);
  const withCommitment = (change: (c: Record<string, unknown>) => void) => mutated(commitment, change);
  const withPlan = (change: (p: Record<string, unknown>) => void) => mutated(plan, change);
  const withEntry = (e: EvaluationRegisterEntry, change: (c: Record<string, unknown>) => void) => mutated(e, change);
  const claimsOf = (r: Record<string, unknown>) => r.claims as Record<string, unknown>[];

  const setManifest = sealedSetManifestSample(commitment);
  const retired = {
    schema_version: 1,
    data_class: "synthetic",
    set_id: "set-1",
    set_manifest_hash: sealedManifestHash(setManifest),
    reveal: { key: hexOf("4"), salt: hexOf("5") },
    files: { teams_sha256: hexOf("6"), rejections_sha256: hexOf("7"), discards_sha256: hexOf("8") },
  };
  const withSet = (change: (m: Record<string, unknown>) => void) => mutated(setManifest, change);
  const withRetired = (change: (m: Record<string, unknown>) => void) => mutated(retired, change);

  const valid: Record<string, unknown> = {
    "evaluation-parameters/provisional": provisional,
    "evaluation-parameters/complete": complete,
    "training-gate-rule/v1": rule,
    "sealed-set-commitment/four-strata": commitment,
    "evaluation-plan/registered": plan,
    "sealed-team-set-manifest/four-strata": setManifest,
    "retired-sealed-manifest/revealed": retired,
    "evaluation-register-entry/set-commitment": entry("set_commitment"),
    "evaluation-register-entry/plan-registration": entry("plan_registration"),
    "evaluation-register-entry/l2a-recheck": entry("l2a_recheck"),
    "evaluation-register-entry/run-outcome": run,
    "evaluation-register-entry/set-retirement": entry("set_retirement"),
    "evaluation-register-entry/incident": incident,
    "evaluation-register-head/head": head,
  };
  const readerInvalid: Record<string, unknown> = {
    // Schema-valid, refused by readEvaluationParameters.
    "evaluation-parameters/alpha-not-whole-ppm": params((p) => (p.alpha_ppm = 50001)),
    "evaluation-parameters/b-boot-too-small": params((p) => (p.b_boot = 80)),
    "evaluation-parameters/r-cov-too-small": params((p) => (p.r_cov = 79)),
    "evaluation-parameters/unsorted-baselines": params((p) => ((p.pool as Record<string, unknown>).baselines = ["random-legal", "actor-rule"])),
    "evaluation-parameters/seen-and-held-out": params((p) => ((p.pool as Record<string, unknown>).held_out_lineages = ["l1"])),
    "evaluation-parameters/n-teams-below-floor": params((p) => (p.n_teams_floor = 40)),
    "evaluation-parameters/n-teams-without-floor": params((p) => (p.n_teams_floor = null)),
    // Schema-valid, refused by readTrainingGateRule.
    "training-gate-rule/unsorted-claims": withRule((r) => claimsOf(r).reverse()),
    "training-gate-rule/alternative-below-null": withRule((r) => ((r.alternatives_ppm as Record<string, unknown>).floor = 700000)),
    // Schema-valid, refused by readSealedSetCommitment.
    "sealed-set-commitment/strata-sum-mismatch": withCommitment((c) => (c.team_count = 33)),
    "sealed-set-commitment/unsorted-strata": withCommitment((c) => (c.stratum_counts as unknown[]).reverse()),
    // Schema-valid, refused by readSealedTeamSetManifest and readRetiredSealedManifest.
    "sealed-team-set-manifest/unsorted-strata": withSet((m) => (m.strata = [...(setManifest.strata as unknown[])].reverse())),
    "sealed-team-set-manifest/quota-sum-mismatch": withSet((m) => (m.team_count = 33)),
    "retired-sealed-manifest/seed-is-salt": withRetired((m) => (m.reveal = { key: hexOf("4"), salt: hexOf("4") })),
    // Schema-valid, refused by readEvaluationPlan.
    "evaluation-plan/candidate-roles-swapped": withPlan((p) => (p.candidate = [...(plan.candidate as unknown as unknown[])].reverse())),
    "evaluation-plan/unsorted-opponents": withPlan((p) => (p.opponents = [...plan.opponents].reverse())),
    "evaluation-plan/baseline-with-lineage": withPlan((p) => ((p.opponents as Record<string, unknown>[])[1] as Record<string, unknown>).lineage_id = "l1"),
    // Schema-valid, refused by readEntry.
    "evaluation-register-entry/reported-without-outcome-hash": withEntry(run, (e) => delete e.outcome_hash),
    "evaluation-register-entry/failed-without-reason": withEntry(run, (e) => {
      e.outcome = "failed";
      delete e.outcome_hash;
    }),
    "evaluation-register-entry/impossible-date": withEntry(run, (e) => (e.recorded_at = "9999-99-99T99:99:99Z")),
    "evaluation-register-entry/unsorted-affected-candidates": withEntry(incident, (e) => (e.affected_candidates = [hexOf("4"), hexOf("3")])),
  };
  const invalid: Record<string, unknown> = {
    "evaluation-parameters/alpha-as-float": params((p) => (p.alpha_ppm = 0.05)),
    "evaluation-parameters/no-b-sealed": params((p) => delete p.b_sealed),
    "evaluation-parameters/ppm-above-one": params((p) => ((p.half_width_ppm as Record<string, unknown>).g3 = 1000001)),
    "evaluation-parameters/zero-family": params((p) => (p.family_size = 0)),
    "evaluation-parameters/stratum-min-one": params((p) => (p.n_stratum_min = 1)),
    "evaluation-parameters/extra-field": params((p) => (p.default = 1)),
    "evaluation-parameters/next-version": params((p) => (p.schema_version = 2)),
    "training-gate-rule/unknown-test": withRule((r) => (r.test = "holm")),
    "training-gate-rule/certain-power": withRule((r) => (r.target_power_ppm = 1000000)),
    "training-gate-rule/exploiters-gate": withRule((r) => (r.exploiters_gate = true)),
    "training-gate-rule/unknown-gate": withRule((r) => (claimsOf(r)[0] as Record<string, unknown>).gate = "G9"),
    "sealed-set-commitment/short-hash": withCommitment((c) => (c.set_commitment = "abc")),
    "sealed-set-commitment/sealed-teams": withCommitment((c) => (c.teams = [])),
    "sealed-set-commitment/salt": withCommitment((c) => (c.salt = hexOf("5"))),
    "sealed-set-commitment/no-count": withCommitment((c) => delete ((c.stratum_counts as Record<string, unknown>[])[0] as Record<string, unknown>).count),
    "sealed-team-set-manifest/seed-field": withSet((m) => (m.key = hexOf("4"))),
    "sealed-team-set-manifest/teams": withSet((m) => (m.teams = [])),
    "sealed-team-set-manifest/real-class": withSet((m) => (m.data_class = "real")),
    "sealed-team-set-manifest/no-commitment": withSet((m) => delete m.set_commitment),
    "sealed-team-set-manifest/float-margin": withSet((m) => (m.sealed_margin_permille = 0.5)),
    "retired-sealed-manifest/no-seed": withRetired((m) => (m.reveal = { salt: hexOf("5") })),
    "retired-sealed-manifest/teams": withRetired((m) => (m.teams = [])),
    "retired-sealed-manifest/short-salt": withRetired((m) => (m.reveal = { key: hexOf("4"), salt: "abc" })),
    "evaluation-plan/direction-less": withPlan((p) => ((p.claims as Record<string, unknown>[])[0] as Record<string, unknown>).direction = "less"),
    "evaluation-plan/extra-field": withPlan((p) => (p.seed_material = hexOf("6"))),
    "evaluation-plan/one-checkpoint": withPlan((p) => (p.candidate = (plan.candidate as unknown as unknown[]).slice(0, 1))),
    "evaluation-plan/short-harness-commit": withPlan((p) => (p.harness_commit = "abc")),
    "evaluation-register-entry/unknown-kind": withEntry(run, (e) => (e.kind = "note")),
    "evaluation-register-entry/seq-zero": withEntry(run, (e) => (e.seq = 0)),
    "evaluation-register-entry/bad-timestamp": withEntry(run, (e) => (e.recorded_at = "yesterday")),
    "evaluation-register-entry/extra-field": withEntry(run, (e) => (e.note = "x")),
    "evaluation-register-entry/unknown-outcome": withEntry(run, (e) => (e.outcome = "won")),
    "evaluation-register-head/seq-zero": { ...head, seq: 0 },
    "evaluation-register-head/short-hash": { ...head, entry_hash: "abc" },
    "evaluation-register-head/extra-field": { ...head, secret: "x" },
  };
  return { valid, invalid, readerInvalid };
}

/** A SealedTeamSetManifest for the sample commitment: four strata of eight, no discards. Illustrative values only. */
function sealedSetManifestSample(commitment: SealedSetCommitment): SealedTeamSetManifest {
  return {
    schema_version: 1,
    data_class: "synthetic",
    set_id: commitment.set_id,
    generation: commitment.generation,
    plan_id: "sealed-plan-1",
    generator: {
      generator_commit: "a".repeat(40),
      config_hash: hexOf("a"),
      label_set_hash: hexOf("b"),
      vocabulary_hash: hexOf("c"),
      manifest: {
        showdown_commit: "7244fe8e28fa8bc4d0809ae30d183c608b62bb2b",
        format_id: "gen9championsvgc2026regmc",
        regulation: "M-C",
        ruleset_hash: hexOf("d"),
        format_rules_hash: hexOf("e"),
      },
    },
    d_max: 120000,
    sealed_margin_permille: 200,
    d_team_differing_fields: 2,
    pilot_draws: 40000,
    team_count: commitment.team_count,
    strata: [0, 1, 2, 3].map((stratum) => ({ stratum, pilot_hits: 9000, quota: 8 })) as SealedTeamSetManifest["strata"],
    reference_pool_hash: hexOf("f"),
    rejection_log: { log_sha256: hexOf("9"), draws: 900, rejections: 70, duplicates: 2, near_duplicates: 5, overflows: 791 },
    discards: { count: 0, log_sha256: createHash("sha256").update("").digest("hex") },
    set_commitment: commitment.set_commitment,
    key_commitment: commitment.key_commitment,
  };
}
