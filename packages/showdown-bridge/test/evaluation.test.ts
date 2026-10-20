// The edits below reach into nested JSON documents by path; `any` keeps each mutation one line.
/* eslint-disable @typescript-eslint/no-explicit-any */
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  alphaPrimePpm,
  checkGateConsistency,
  covTolPpm,
  deriveClaims,
  EvaluationConfigError,
  halfWidthOf,
  isComplete,
  jsonHash,
  maxHalfWidthPpm,
  normalQuantile,
  readEvaluationConfig,
  readEvaluationParameters,
  readTrainingGateRule,
  requireComplete,
} from "../src/evaluation-config.ts";
import type { Json } from "../src/canonical-json.ts";
import { runEvaluationCli } from "../src/evaluation-cli.ts";
import {
  checkCommitment,
  checkPlan,
  checkReveal,
  commitmentHash,
  keyCommitment,
  planHash,
  readEvaluationPlan,
  readSealedSetCommitment,
  setCommitment,
} from "../src/evaluation-plan.ts";
import { appendEntry, assertPlanPublished, checkHeads, entryHash, GENESIS, readEntry, readHead, RegisterError, verifyRegister } from "../src/evaluation-register.ts";
import { evaluationFixtures, PARAMETERS_FILE, RULE_FILE } from "../src/evaluation-samples.ts";

const fx = evaluationFixtures();
const copy = <T>(v: T): T => structuredClone(v);
const edit = (v: Record<string, unknown>, change: (c: Record<string, any>) => void): Record<string, unknown> => {
  const c = copy(v);
  change(c);
  return c;
};
const lines = (text: string): string[] => text.slice(0, -1).split("\n");
const hex = (c: string): string => c.repeat(64);

describe("normalQuantile", () => {
  it.each([
    [0.5, 0],
    [0.975, 1.959964],
    [0.99375, 2.497705],
    [0.9875, 2.241403],
    [0.8, 0.841621],
    [0.2, -0.841621],
  ])("is the standard normal quantile of %s", (p, z) => {
    expect(normalQuantile(p)).toBeCloseTo(z, 5);
  });
  it("refuses 0 and 1", () => {
    expect(() => normalQuantile(0)).toThrow(RangeError);
    expect(() => normalQuantile(1)).toThrow(RangeError);
  });
});

describe("EvaluationParameters and TrainingGateRule (ADR-010 sections 6 and 7)", () => {
  it("the committed v1 records read, agree with each other and are provisional", () => {
    const config = readEvaluationConfig(fx.provisional, fx.rule);
    expect(alphaPrimePpm(config.parameters)).toBe(12500);
    expect(isComplete(config.parameters)).toBe(false);
    expect(() => requireComplete(config.parameters)).toThrow(/n_teams_floor, n_teams, n_seeds, d_team/);
    expect(config.parameters_hash).toBe(jsonHash(fx.provisional));
    expect(config.rule_hash).toBe(jsonHash(fx.rule));
  });

  it("complete parameters pass requireComplete", () => {
    expect(isComplete(fx.config.parameters)).toBe(true);
    expect(() => requireComplete(fx.config.parameters)).not.toThrow();
  });

  it("derives the coverage tolerance from the record (ADR-010 section 6.1 step 4: about 0.0035)", () => {
    expect(covTolPpm(fx.config.parameters)).toBeGreaterThan(3400);
    expect(covTolPpm(fx.config.parameters)).toBeLessThan(3700);
  });

  it("derives the margin nulls and alternatives of the gate claims", () => {
    expect(deriveClaims(fx.config.rule).map((c) => [c.claim_id, c.margin_null_ppm, c.alternative_ppm])).toEqual([
      ["g2-actor-rule", 750000, 850000],
      ["g2-random-legal", 750000, 850000],
      ["g3-held-out-l2", 550000, 600000],
      ["g3b-progress-l2", 0, 50000],
    ]);
  });

  it("derives the half-widths of section 6.1: at most 0.0405 for the 0.05 gaps and 0.0810 for 0.10", () => {
    expect(maxHalfWidthPpm(12500, 800000, 50000)).toBeCloseTo(40507, -1);
    expect(maxHalfWidthPpm(12500, 800000, 100000)).toBeCloseTo(81014, -1);
    expect(halfWidthOf(fx.config.parameters, "G2")).toBe(80000);
    expect(halfWidthOf(fx.config.parameters, "G3")).toBe(40000);
    expect(halfWidthOf(fx.config.parameters, "G3b")).toBe(40000);
  });

  it.each([
    ["a schema violation", (p: Record<string, any>) => (p.alpha_ppm = 0.05), /invalid EvaluationParameters/],
    ["alpha' that is not whole ppm", (p: Record<string, any>) => (p.alpha_ppm = 50001), /not a whole number of ppm/],
    ["b_boot at or below m*B/alpha", (p: Record<string, any>) => (p.b_boot = 80), /must exceed/],
    ["r_cov below 1/alpha'", (p: Record<string, any>) => (p.r_cov = 79), /at least 1 \/ alpha'/],
    ["unsorted baselines", (p: Record<string, any>) => (p.pool.baselines = ["random-legal", "actor-rule"]), /pool.baselines must be sorted and unique/],
    ["duplicate lineages", (p: Record<string, any>) => (p.pool.seen_lineages = ["l1", "l1"]), /pool.seen_lineages/],
    ["unsorted held-out lineages", (p: Record<string, any>) => (p.pool.held_out_lineages = ["l3", "l2"]), /pool.held_out_lineages/],
    ["a lineage both seen and held out", (p: Record<string, any>) => (p.pool.held_out_lineages = ["l1"]), /both seen and held out: l1/],
    ["n_teams without its floor", (p: Record<string, any>) => (p.n_teams_floor = null), /n_teams is set/],
    ["n_teams without seeds", (p: Record<string, any>) => (p.n_seeds = null), /n_teams is set/],
    ["n_teams below its floor", (p: Record<string, any>) => (p.n_teams_floor = 40), /below its floor 40/],
  ])("readEvaluationParameters refuses %s", (_what, change, message) => {
    expect(() => readEvaluationParameters(edit(fx.complete, change))).toThrow(EvaluationConfigError);
    expect(() => readEvaluationParameters(edit(fx.complete, change))).toThrow(message);
  });

  it.each([
    ["a schema violation", (r: Record<string, any>) => (r.test = "holm"), /invalid TrainingGateRule/],
    ["unsorted claims", (r: Record<string, any>) => r.claims.reverse(), /sorted by claim_id/],
    ["a repeated claim", (r: Record<string, any>) => (r.claims[1] = r.claims[0]), /sorted by claim_id and unique/],
    ["unsorted reference fractions", (r: Record<string, any>) => (r.selection.reference_fractions_ppm = [500000, 250000]), /strictly ascending/],
    ["unsorted checks", (r: Record<string, any>) => r.required_checks.reverse(), /required_checks must be sorted/],
    ["an alternative below its null", (r: Record<string, any>) => (r.alternatives_ppm.floor = 700000), /must lie above the margin null/],
    ["a G3b alternative of zero", (r: Record<string, any>) => (r.alternatives_ppm.progress_gain = 0), /must lie above the margin null 0/],
    ["a margin that leaves no room", (r: Record<string, any>) => (r.margins_ppm.floor = 500000), /leaves no room/],
  ])("readTrainingGateRule refuses %s", (_what, change, message) => {
    expect(() => readTrainingGateRule(edit(fx.rule, change))).toThrow(EvaluationConfigError);
    expect(() => readTrainingGateRule(edit(fx.rule, change))).toThrow(message);
  });

  it.each([
    ["a family size that differs from the claims", (p: Record<string, any>) => (p.family_size = 8), /family_size is 8/],
    ["a baseline without its G2 claim", (p: Record<string, any>) => (p.pool.baselines = ["actor-rule", "random-legal", "zz"]), /exactly once/],
    ["a half-width above what the power allows (G3)", (p: Record<string, any>) => (p.half_width_ppm.g3 = 41000), /g3-held-out-l2: h 41000 ppm exceeds 40507/],
    ["a half-width above what the power allows (G2)", (p: Record<string, any>) => (p.half_width_ppm.g2 = 82000), /g2-actor-rule: h 82000 ppm exceeds 81014/],
    ["a half-width above what the power allows (G3b)", (p: Record<string, any>) => (p.half_width_ppm.g3b = 50000), /g3b-progress-l2/],
    ["a held-out lineage the claims do not use", (p: Record<string, any>) => (p.pool.held_out_lineages = ["l3"]), /l2 is not a held-out lineage/],
  ])("the two records together refuse %s", (_what, change, message) => {
    const parameters = readEvaluationParameters(edit(fx.complete, change));
    expect(() => checkGateConsistency(parameters, fx.config.rule)).toThrow(message);
  });

  it("a higher target power or a smaller gap needs a narrower h", () => {
    const rule = readTrainingGateRule(edit(fx.rule, (r) => (r.target_power_ppm = 900000)));
    expect(() => checkGateConsistency(fx.config.parameters, rule)).toThrow(/exceeds/);
    const gap = readTrainingGateRule(edit(fx.rule, (r) => (r.alternatives_ppm.held_out = 580000)));
    expect(() => checkGateConsistency(fx.config.parameters, gap)).toThrow(/g3-held-out-l2/);
  });

  it("changing any value changes the hash a plan binds", () => {
    const changed = readEvaluationConfig(edit(fx.complete, (p) => (p.max_turns = 301)), fx.rule);
    expect(changed.parameters_hash).not.toBe(fx.config.parameters_hash);
    expect(changed.rule_hash).toBe(fx.config.rule_hash);
  });

  it("holds the committed records as data: the files the CLI reads are the ones the fixtures use", () => {
    expect(JSON.parse(readFileSync(PARAMETERS_FILE, "utf8"))).toEqual(fx.provisional);
    expect(JSON.parse(readFileSync(RULE_FILE, "utf8"))).toEqual(fx.rule);
    expect(fx.provisional.max_turns).toBe(300);
  });
});

describe("the sealed-set commitment and the plan (protocol L3, I2)", () => {
  it("commits to {salt, teams} and the key by hash; a reveal opens it and nothing else does", () => {
    const salt = hex("a");
    const key = hex("b");
    const teams: Json[] = [{ species: "x" }, { species: "y" }];
    const commitment = { ...fx.commitment, set_commitment: setCommitment(salt, teams), key_commitment: keyCommitment(key) };
    expect(checkReveal(commitment, { salt, teams, key })).toBe(true);
    expect(checkReveal(commitment, { salt, teams: [teams[1] as Json, teams[0] as Json], key })).toBe(false);
    expect(checkReveal(commitment, { salt: hex("c"), teams, key })).toBe(false);
    expect(checkReveal(commitment, { salt, teams, key: hex("c") })).toBe(false);
  });

  it("refuses a short salt or key (a short list could be guessed by hashing candidates)", () => {
    expect(() => setCommitment("abc", [])).toThrow(/64 lowercase hex/);
    expect(() => keyCommitment("abc")).toThrow(/64 lowercase hex/);
  });

  it("reads the commitment and checks it against the records (E1: n_stratum_min in each stratum)", () => {
    const commitment = readSealedSetCommitment(fx.commitment);
    expect(() => checkCommitment(commitment, fx.config)).not.toThrow();
    const thin = readSealedSetCommitment(edit(fx.commitment as never, (c) => {
      c.stratum_counts = [{ stratum: "a", count: 31 }, { stratum: "b", count: 1 }];
    }));
    expect(() => checkCommitment(thin, fx.config)).toThrow(/below n_stratum_min 2: b \(1\)/);
  });

  it.each([
    ["provisional parameters", () => checkCommitment(fx.commitment, readEvaluationConfig(fx.provisional, fx.rule)), /provisional/],
    ["other parameters", () => checkCommitment({ ...fx.commitment, evaluation_parameters_hash: hex("1") }, fx.config), /other EvaluationParameters/],
    ["another rule", () => checkCommitment({ ...fx.commitment, training_gate_rule_hash: hex("1") }, fx.config), /another TrainingGateRule/],
    ["a team count that is not n_teams", () => checkCommitment({ ...fx.commitment, team_count: 33, stratum_counts: [{ stratum: "a", count: 33 }] }, fx.config), /differs from the parameters' n_teams 32/],
    ["a budget that is not b_sealed", () => checkCommitment({ ...fx.commitment, b_sealed: 2 }, fx.config), /b_sealed 2/],
    ["strata that do not sum", () => readSealedSetCommitment({ ...fx.commitment, team_count: 33 }), /hold 32 teams but team_count is 33/],
    ["unsorted strata", () => readSealedSetCommitment({ ...fx.commitment, stratum_counts: [...fx.commitment.stratum_counts].reverse() }), /sorted by label/],
    ["a schema violation", () => readSealedSetCommitment({ ...fx.commitment, teams: [] }), /invalid SealedSetCommitment/],
  ])("refuses %s", (_what, run, message) => {
    expect(run).toThrow(EvaluationConfigError);
    expect(run).toThrow(message);
  });

  it("a plan that matches its records checks, and its hash is the JCS digest", () => {
    const plan = readEvaluationPlan(fx.plan);
    expect(() => checkPlan(plan, { config: fx.config, commitment: fx.commitment })).not.toThrow();
    expect(planHash(plan)).toBe(jsonHash(fx.plan));
  });

  it.each([
    ["roles swapped", (p: Record<string, any>) => p.candidate.reverse(), /battle checkpoint, then the preview/],
    ["unsorted opponents", (p: Record<string, any>) => p.opponents.reverse(), /sorted by opponent_id/],
    ["a baseline with a lineage", (p: Record<string, any>) => (p.opponents[1].lineage_id = "l1"), /no lineage/],
    ["a frozen opponent without a lineage", (p: Record<string, any>) => delete p.opponents[0].lineage_id, /no lineage/],
    ["a claim twice", (p: Record<string, any>) => (p.claims[1] = p.claims[0]), /listed twice/],
    ["a schema violation", (p: Record<string, any>) => (p.key = hex("1")), /invalid EvaluationPlan/],
  ])("readEvaluationPlan refuses %s", (_what, change, message) => {
    expect(() => readEvaluationPlan(edit(fx.plan as never, change))).toThrow(message);
  });

  const withPlan = (change: (p: Record<string, any>) => void) => readEvaluationPlan(edit(fx.plan as never, change));
  it.each([
    ["another set", (p: Record<string, any>) => (p.set_id = "set-2"), /for set set-2/],
    ["another commitment", (p: Record<string, any>) => (p.set_commitment_hash = hex("1")), /another SealedSetCommitment/],
    ["other parameters", (p: Record<string, any>) => (p.evaluation_parameters_hash = hex("1")), /other EvaluationParameters/],
    ["another rule", (p: Record<string, any>) => (p.training_gate_rule_hash = hex("1")), /another TrainingGateRule/],
    ["another alpha", (p: Record<string, any>) => (p.alpha_ppm = 100000), /alpha_ppm 100000 differs/],
    ["another alpha'", (p: Record<string, any>) => (p.alpha_prime_ppm = 50000), /alpha_prime_ppm 50000 differs/],
    ["another b_boot", (p: Record<string, any>) => (p.b_boot = 10000), /b_boot 10000 differs/],
    ["another block plan", (p: Record<string, any>) => (p.block_plan.n_seeds = 2), /block plan/],
    ["a claim missing", (p: Record<string, any>) => p.claims.pop(), /lists 3 claims, the rule 4/],
    ["a claim that differs", (p: Record<string, any>) => (p.claims[0].margin_null_ppm = 700000), /claim 0 \(g2-actor-rule\) differs/],
    ["a claim against an unknown opponent", (p: Record<string, any>) => (p.claims[2].opponent_id = "nobody"), /not among the plan's opponents/],
    ["a G2 claim against a frozen opponent", (p: Record<string, any>) => (p.claims[0].opponent_id = hex("a")), /not the scripted baseline actor-rule/],
    ["a G3 claim against another lineage", (p: Record<string, any>) => (p.opponents[0].lineage_id = "l1"), /not a frozen checkpoint of the held-out lineage l2/],
    ["a candidate that is its own opponent", (p: Record<string, any>) => (p.opponents[0].opponent_id = hex("3")), /both a candidate/],
  ])("checkPlan refuses %s", (_what, change, message) => {
    const plan = withPlan((p) => {
      change(p);
      if (message.source.includes("both a candidate")) p.claims.forEach((c: Record<string, any>) => c.opponent_id === hex("a") && (c.opponent_id = hex("3")));
    });
    expect(() => checkPlan(plan, { config: fx.config, commitment: fx.commitment })).toThrow(EvaluationConfigError);
    expect(() => checkPlan(plan, { config: fx.config, commitment: fx.commitment })).toThrow(message);
  });
});

describe("the evaluation register (protocol I6, ADR-010 O1)", () => {
  const { text, entries } = fx.chain;
  const rebuild = (changed: string[]): string => `${changed.join("\n")}\n`;
  const entryAt = (n: number): Record<string, any> => JSON.parse(lines(text)[n - 1] as string);
  const asLine = (e: Record<string, unknown>): string => JSON.stringify(sortKeys(e));
  const sortKeys = (v: any): any => (Array.isArray(v) ? v.map(sortKeys) : v && typeof v === "object" ? Object.fromEntries(Object.keys(v).sort().map((k) => [k, sortKeys(v[k])])) : v);

  it("a test plan registers: the chain links, hashes and counts", () => {
    const state = verifyRegister(text);
    expect(state.entries.map((e) => e.kind)).toEqual(["set_commitment", "plan_registration", "l2a_recheck", "run_outcome", "set_retirement"]);
    expect(state.entries.map((e) => e.seq)).toEqual([1, 2, 3, 4, 5]);
    expect(entries[0]?.prev_hash).toBe(GENESIS);
    entries.forEach((e, i) => expect(e.prev_hash).toBe(i === 0 ? GENESIS : entryHash(entries[i - 1] as never)));
    expect(state.head).toEqual({ schema_version: 1, seq: 5, entry_hash: entryHash(entries[4] as never) });
    const set = state.sets.get("set-1");
    expect(set?.retired).toBe(true);
    expect(set?.plans[0]?.plan_hash).toBe(planHash(fx.plan));
  });

  it("an empty register has no head", () => {
    expect(verifyRegister("")).toMatchObject({ head: null, entries: [] });
  });

  it("refuses a changed entry, a dropped entry, a reordered pair and an appended-over history", () => {
    const l = lines(text);
    const changed = entryAt(3);
    changed.operator = "someone-else";
    expect(() => verifyRegister(rebuild([l[0] as string, l[1] as string, asLine(changed), ...l.slice(3)]))).toThrow(/line 4: prev_hash/);
    expect(() => verifyRegister(rebuild([l[0] as string, ...l.slice(2)]))).toThrow(/line 2: seq 3 where 2 is due/);
    expect(() => verifyRegister(rebuild([l[0] as string, l[2] as string, l[1] as string, ...l.slice(3)]))).toThrow(/line 2/);
  });

  it("refuses a torn last line, a non-JSON line, a line that is not JCS and a schema violation", () => {
    expect(() => verifyRegister(text.slice(0, -1))).toThrow(/must end with a newline/);
    expect(() => verifyRegister("{oops\n")).toThrow(/line 1: not JSON/);
    expect(() => verifyRegister(`${JSON.stringify(entryAt(1), null, 1).replaceAll("\n", "")}\n`)).toThrow(/not the entry's JCS bytes/);
    expect(() => verifyRegister(`${asLine({ ...entryAt(1), kind: "note" })}\n`)).toThrow(/line 1: invalid EvaluationRegisterEntry/);
  });

  it("names the line when a record inside an entry is refused", () => {
    const bad = { ...entryAt(1), commitment: { ...fx.commitment, team_count: 33 } };
    expect(() => verifyRegister(`${asLine(bad)}\n`)).toThrow(/line 1: .*hold 32 teams but team_count is 33/);
    const badPlan = { ...entryAt(2), plan: { ...fx.plan, claims: [fx.plan.claims[0], fx.plan.claims[0]] } };
    expect(() => verifyRegister(`${asLine(badPlan)}\n`)).toThrow(/line 1: .*listed twice/);
  });

  it("refuses a recorded_at that goes back", () => {
    const first = appendEntry("", { kind: "set_commitment", recorded_at: "2026-11-02T10:00:00Z", operator: "owner", commitment: fx.commitment });
    expect(() => appendEntry(first.text, { kind: "plan_registration", recorded_at: "2026-11-01T10:00:00Z", operator: "owner", plan: fx.plan })).toThrow(/is before the previous entry/);
  });

  const upTo = (n: number): string => `${lines(text).slice(0, n).join("\n")}\n`;
  const at = "2026-11-06T10:00:00Z";
  const draft = (d: Record<string, unknown>) => ({ recorded_at: at, operator: "owner", ...d }) as never;

  it.each([
    ["a plan before its set is committed", 0, { kind: "plan_registration", plan: fx.plan }, /has no commitment/],
    ["a set committed twice", 1, { kind: "set_commitment", commitment: fx.commitment }, /committed twice/],
    ["a set that reuses a commitment", 1, { kind: "set_commitment", commitment: { ...fx.commitment, set_id: "set-2" } }, /reuses the commitment/],
    ["a second plan beyond the budget", 2, { kind: "plan_registration", plan: { ...fx.plan, a2_matches_min: 11 } }, /allows 1 run/],
    ["the same plan twice", 2, { kind: "plan_registration", plan: fx.plan }, /registered twice|allows 1 run/],
    ["a plan naming another commitment", 1, { kind: "plan_registration", plan: { ...fx.plan, set_commitment_hash: hex("1") } }, /does not name set set-1's commitment/],
    ["a run without a re-check", 2, { kind: "run_outcome", plan_hash: planHash(fx.plan), outcome: "reported", outcome_hash: hex("f") }, /no passing L2a re-check/],
    ["a run of an unregistered plan", 3, { kind: "run_outcome", plan_hash: hex("1"), outcome: "failed", reason: "x" }, /not registered/],
    ["a second outcome", 4, { kind: "run_outcome", plan_hash: planHash(fx.plan), outcome: "aborted", reason: "again" }, /already has an outcome/],
    ["a re-check of another set's plan", 2, { kind: "l2a_recheck", set_id: "set-2", result: "pass" }, /no commitment/],
    ["a re-check of an unregistered plan", 2, { kind: "l2a_recheck", set_id: "set-1", plan_hash: hex("1"), result: "pass" }, /not registered/],
    ["a re-check after retirement", 5, { kind: "l2a_recheck", set_id: "set-1", result: "pass" }, /is retired/],
    ["a plan after retirement", 5, { kind: "plan_registration", plan: fx.plan }, /is retired/],
    ["a second retirement", 5, { kind: "set_retirement", set_id: "set-1", reason: "withdrawn" }, /already retired/],
    ["a budget_spent retirement with budget left", 2, { kind: "set_retirement", set_id: "set-1", reason: "budget_spent" }, /budget left or a run without an outcome/],
    ["a contaminated retirement of a clean set", 2, { kind: "set_retirement", set_id: "set-1", reason: "contaminated" }, /not contaminated/],
    ["an incident on an unknown set", 1, { kind: "incident", set_id: "nope", cause: "plan_deviation", description: "x", affected_candidates: [] }, /no commitment/],
  ])("refuses %s", (_what, n, d, message) => {
    expect(() => appendEntry(n === 0 ? "" : upTo(n), draft(d as Record<string, unknown>))).toThrow(RegisterError);
    expect(() => appendEntry(n === 0 ? "" : upTo(n), draft(d as Record<string, unknown>))).toThrow(message);
  });

  it("refuses a candidate that already has a plan on the set", () => {
    const wide = { ...fx.commitment, b_sealed: 2 };
    const first = appendEntry("", { kind: "set_commitment", recorded_at: at, operator: "owner", commitment: wide });
    const plan = { ...fx.plan, set_commitment_hash: commitmentHash(wide) };
    const second = appendEntry(first.text, draft({ kind: "plan_registration", plan }));
    expect(() => appendEntry(second.text, draft({ kind: "plan_registration", plan: { ...plan, a2_matches_min: 11 } }))).toThrow(/already has a plan/);
  });

  it("a failed re-check contaminates the set: only an aborted run may follow, and its retirement is contaminated", () => {
    let current = upTo(2);
    current = appendEntry(current, draft({ kind: "l2a_recheck", set_id: "set-1", plan_hash: planHash(fx.plan), result: "fail" })).text;
    expect(verifyRegister(current).sets.get("set-1")).toMatchObject({ contaminated: true, retired: true });
    expect(() => appendEntry(current, draft({ kind: "run_outcome", plan_hash: planHash(fx.plan), outcome: "failed", reason: "x" }))).toThrow(/contaminated: only an aborted run/);
    current = appendEntry(current, draft({ kind: "run_outcome", plan_hash: planHash(fx.plan), outcome: "aborted", reason: "re-check failed" })).text;
    expect(() => appendEntry(current, draft({ kind: "set_retirement", set_id: "set-1", reason: "withdrawn" }))).toThrow(/is contaminated, so its retirement reason/);
    expect(verifyRegister(appendEntry(current, draft({ kind: "set_retirement", set_id: "set-1", reason: "contaminated" })).text).entries).toHaveLength(5);
  });

  it("an incident marks the set contaminated and retired at once", () => {
    const current = appendEntry(upTo(2), draft({ kind: "incident", set_id: "set-1", cause: "sealed_artifact_read", description: "x", affected_candidates: [hex("3")] })).text;
    expect(verifyRegister(current).sets.get("set-1")).toMatchObject({ contaminated: true, retired: true });
  });

  it("a plan-bound re-check of the wrong set is refused", () => {
    const other = { ...fx.commitment, set_id: "set-2", set_commitment: hex("7"), key_commitment: hex("8") };
    const current = appendEntry(upTo(2), draft({ kind: "set_commitment", commitment: other })).text;
    expect(() => appendEntry(current, draft({ kind: "l2a_recheck", set_id: "set-2", plan_hash: planHash(fx.plan), result: "pass" }))).toThrow(/belongs to set set-1, not set-2/);
  });

  it("readEntry refuses what the profile cannot say", () => {
    const run = entryAt(4);
    expect(() => readEntry({ ...run, outcome_hash: undefined })).toThrow(/needs its outcome_hash/);
    expect(() => readEntry({ ...run, outcome: "failed", outcome_hash: undefined })).toThrow(/needs its reason/);
    expect(() => readEntry({ ...run, outcome: "failed", reason: "x" })).toThrow(/has no outcome_hash/);
    const { schema_version, seq, prev_hash, recorded_at, operator } = run;
    const incident = { schema_version, seq, prev_hash, recorded_at, operator, kind: "incident", set_id: "set-1", cause: "plan_deviation", description: "x" };
    expect(readEntry({ ...incident, affected_candidates: [hex("3"), hex("4")] }).kind).toBe("incident");
    expect(() => readEntry({ ...incident, affected_candidates: [hex("4"), hex("3")] })).toThrow(/sorted and unique/);
  });
});

describe("published chain heads (O1)", () => {
  const { text, entries } = fx.chain;
  const state = verifyRegister(text);
  const head = (seq: number, hash = entryHash(entries[seq - 1] as never)) => ({ schema_version: 1, seq, entry_hash: hash });

  it("reports how far the heads reach", () => {
    expect(checkHeads(state, [])).toEqual({ published_through: 0, unpublished: 5 });
    expect(checkHeads(state, [head(2), head(3)])).toEqual({ published_through: 3, unpublished: 2 });
    expect(checkHeads(state, [head(5)])).toEqual({ published_through: 5, unpublished: 0 });
  });

  it("refuses a register rolled back below a published head", () => {
    const short = verifyRegister(`${lines(text).slice(0, 3).join("\n")}\n`);
    expect(() => checkHeads(short, [head(5)])).toThrow(/rollback: a head is published at position 5 but the register has 3/);
  });

  it("refuses a rewritten history and a fork behind a published head", () => {
    const l = lines(text);
    const forkedEntry = { ...JSON.parse(l[3] as string), outcome_hash: hex("1") };
    const sortKeys = (v: any): any => (Array.isArray(v) ? v.map(sortKeys) : v && typeof v === "object" ? Object.fromEntries(Object.keys(v).sort().map((k) => [k, sortKeys(v[k])])) : v);
    const fork = verifyRegister(`${[...l.slice(0, 3), JSON.stringify(sortKeys(forkedEntry))].join("\n")}\n`);
    expect(() => checkHeads(fork, [head(4)])).toThrow(/fork or rewrite: position 4/);
    expect(() => checkHeads(state, [head(2, hex("1"))])).toThrow(/fork or rewrite: position 2/);
    expect(() => checkHeads(state, [head(2), head(2)])).toThrow(/two published heads at position 2/);
  });

  it("a plan may run only when its own entry is published (I2)", () => {
    const hash = planHash(fx.plan);
    expect(() => assertPlanPublished(state, checkHeads(state, []), hash)).toThrow(/merge the chain head first/);
    expect(assertPlanPublished(state, checkHeads(state, [head(2)]), hash)).toBe(2);
    expect(() => assertPlanPublished(state, checkHeads(state, [head(2)]), hex("1"))).toThrow(/not in the register/);
  });

  it("readHead refuses a head with extra fields", () => {
    expect(() => readHead({ ...head(1), secret: "x" })).toThrow(/invalid EvaluationRegisterHead/);
  });
});

describe("the evaluation-register CLI", () => {
  const dir = mkdtempSync(join(tmpdir(), "eval-register-"));
  const p = (name: string): string => join(dir, name);
  const write = (name: string, value: unknown): string => {
    writeFileSync(p(name), JSON.stringify(value));
    return p(name);
  };
  const clock = () => new Date("2026-11-07T08:09:10.123Z");
  const run = (...args: string[]) => runEvaluationCli(args, clock);
  const records = ["--parameters", write("params.json", fx.complete), "--rule", write("rule.json", fx.rule)];
  const register = p("register.jsonl");

  it("config-check prints the hashes and refuses a provisional record under --complete", () => {
    const ok = run("config-check", ...records);
    expect(ok.code).toBe(0);
    expect(JSON.parse(ok.out)).toEqual({ parameters_hash: fx.config.parameters_hash, rule_hash: fx.config.rule_hash, alpha_prime_ppm: 12500, complete: true });
    const provisional = run("config-check", "--parameters", PARAMETERS_FILE, "--rule", RULE_FILE);
    expect(JSON.parse(provisional.out).complete).toBe(false);
    const refused = run("config-check", "--parameters", PARAMETERS_FILE, "--rule", RULE_FILE, "--complete");
    expect(refused.code).toBe(1);
    expect(refused.err).toMatch(/provisional/);
  });

  it("commits a set, registers a plan, appends the run entries and verifies, end to end", () => {
    const commit = run("commit-set", "--register", register, "--commitment", write("commitment.json", fx.commitment), ...records, "--operator", "owner");
    expect(commit.code, commit.err).toBe(0);
    expect(JSON.parse(commit.out).head.seq).toBe(1);
    const registered = run("register-plan", "--register", register, "--plan", write("plan.json", fx.plan), ...records, "--operator", "owner", "--recorded-at", "2026-11-07T09:00:00Z");
    expect(registered.code, registered.err).toBe(0);
    expect(JSON.parse(registered.out).plan_hash).toBe(planHash(fx.plan));
    const recheck = run("append", "--register", register, "--entry", write("recheck.json", { kind: "l2a_recheck", set_id: "set-1", plan_hash: planHash(fx.plan), result: "pass" }), "--operator", "evaluation-job", "--recorded-at", "2026-11-07T10:00:00Z");
    expect(recheck.code, recheck.err).toBe(0);

    const verified = JSON.parse(run("verify", "--register", register).out);
    expect(verified).toMatchObject({ entries: 3, sets: 1, published_through: null });
    const text = readFileSync(register, "utf8");
    expect(text.split("\n")).toHaveLength(4);
    expect(JSON.parse(lines(text)[0] as string).recorded_at).toBe("2026-11-07T08:09:10Z");
    expect(JSON.parse(run("head", "--register", register).out)).toEqual(verified.head);
  });

  it("refuses to register a plan that does not match its records, and leaves the register untouched", () => {
    const before = readFileSync(register, "utf8");
    const bad = run("register-plan", "--register", register, "--plan", write("bad-plan.json", { ...fx.plan, alpha_ppm: 100000 }), ...records, "--operator", "owner");
    expect(bad.code).toBe(1);
    expect(bad.err).toMatch(/alpha_ppm 100000 differs/);
    const unknown = run("register-plan", "--register", register, "--plan", write("set2.json", { ...fx.plan, set_id: "set-2" }), ...records, "--operator", "owner");
    expect(unknown.err).toMatch(/set set-2 has no commitment in the register/);
    const provisional = run("register-plan", "--register", register, "--plan", p("plan.json"), "--parameters", PARAMETERS_FILE, "--rule", RULE_FILE, "--operator", "owner");
    expect(provisional.err).toMatch(/provisional/);
    expect(readFileSync(register, "utf8")).toBe(before);
  });

  it("append refuses the kinds that carry records and an entry the chain refuses", () => {
    const records2 = run("append", "--register", register, "--entry", write("c.json", { kind: "set_commitment" }), "--operator", "o");
    expect(records2.code).toBe(2);
    expect(records2.err).toMatch(/use commit-set/);
    expect(run("append", "--register", register, "--entry", write("pr.json", { kind: "plan_registration" }), "--operator", "o").err).toMatch(/use register-plan/);
    const refused = run("append", "--register", register, "--entry", write("r.json", { kind: "run_outcome", plan_hash: hex("1"), outcome: "failed", reason: "x" }), "--operator", "o", "--recorded-at", "2026-11-08T10:00:00Z");
    expect(refused.code).toBe(1);
    expect(refused.err).toMatch(/is not registered before this entry/);
    expect(run("append", "--register", register, "--entry", write("arr.json", []), "--operator", "o").code).toBe(2);
  });

  it("verify checks the published heads and --published gates a run on its plan's head", () => {
    const heads = p("heads");
    mkdirSync(heads);
    const state = verifyRegister(readFileSync(register, "utf8"));
    const headOf = (seq: number) => ({ schema_version: 1, seq, entry_hash: state.hashes[seq - 1] });
    const publish = (seq: number) => writeFileSync(join(heads, `${String(seq).padStart(10, "0")}.json`), JSON.stringify(headOf(seq)));
    publish(1);
    expect(JSON.parse(run("verify", "--register", register, "--heads", heads).out)).toMatchObject({ published_through: 1, unpublished: 2 });
    const gated = run("verify", "--register", register, "--heads", heads, "--published", planHash(fx.plan));
    expect(gated.code).toBe(1);
    expect(gated.err).toMatch(/merge the chain head first/);
    publish(2);
    expect(run("verify", "--register", register, "--heads", heads, "--published", planHash(fx.plan)).code).toBe(0);
    expect(run("verify", "--register", register, "--published", planHash(fx.plan)).code).toBe(2);
    expect(JSON.parse(run("heads-check", "--dir", heads).out)).toEqual({ heads: 2, latest: 2 });
    expect(JSON.parse(run("heads-check", "--dir", p("no-such-dir")).out)).toEqual({ heads: 0, latest: null });
  });

  it("verify refuses a rolled-back, a tampered and a forked register against the heads", () => {
    const heads = p("heads");
    const text = readFileSync(register, "utf8");
    writeFileSync(p("rolled-back.jsonl"), `${lines(text)[0]}\n`);
    expect(run("verify", "--register", p("rolled-back.jsonl"), "--heads", heads).err).toMatch(/rollback/);
    const tampered = lines(text);
    tampered[1] = (tampered[1] as string).replace("owner", "owneR");
    writeFileSync(p("tampered.jsonl"), `${tampered.join("\n")}\n`);
    expect(run("verify", "--register", p("tampered.jsonl"), "--heads", heads).code).toBe(1);
    // A fork: same first entry, a different second one, at a position a head covers.
    const fork = appendEntry(`${lines(text)[0]}\n`, { kind: "plan_registration", recorded_at: "2026-11-07T09:30:00Z", operator: "owner", plan: { ...fx.plan, a2_matches_min: 99 } });
    writeFileSync(p("fork.jsonl"), fork.text);
    expect(run("verify", "--register", p("fork.jsonl"), "--heads", heads).err).toMatch(/fork or rewrite: position 2/);
  });

  it("refuses a head file whose name is not its position", () => {
    const heads = p("misnamed");
    mkdirSync(heads);
    writeFileSync(join(heads, "a.json"), JSON.stringify({ schema_version: 1, seq: 1, entry_hash: hex("1") }));
    expect(run("heads-check", "--dir", heads).err).toMatch(/file name must be the zero-padded position/);
  });

  it("head of an empty register is refused; a missing register file reads as empty", () => {
    expect(run("head", "--register", p("none.jsonl")).err).toMatch(/empty/);
    expect(existsSync(p("none.jsonl"))).toBe(false);
    expect(run("verify", "--register", p("none.jsonl")).code).toBe(0);
  });

  it("lets an operating-system error through instead of reporting it as a refusal", () => {
    expect(() => run("commit-set", "--register", p("no-such-dir/register.jsonl"), "--commitment", p("commitment.json"), ...records, "--operator", "owner")).toThrow(/ENOENT/);
  });

  it("usage errors exit 2", () => {
    expect(run().code).toBe(2);
    expect(run("nope").err).toMatch(/unknown command nope/);
    expect(run("verify").err).toMatch(/--register is required/);
    expect(run("verify", "register").err).toMatch(/unexpected argument/);
    expect(run("verify", "--register").err).toMatch(/needs a value/);
    expect(run("config-check", "--parameters", p("missing.json"), "--rule", p("rule.json")).err).toMatch(/cannot read/);
  });

  it("uses the real clock by default", () => {
    expect(runEvaluationCli(["verify", "--register", p("none.jsonl")]).code).toBe(0);
    const stamped = runEvaluationCli(["commit-set", "--register", p("clock.jsonl"), "--commitment", p("commitment.json"), ...records, "--operator", "owner"]);
    expect(stamped.code).toBe(0);
    expect(JSON.parse(lines(readFileSync(p("clock.jsonl"), "utf8"))[0] as string).recorded_at).toMatch(/^20\d\d-\d\d-\d\dT\d\d:\d\d:\d\dZ$/);
  });
});

describe("the evaluation modules are engine-free", () => {
  it.each(["evaluation-config", "evaluation-plan", "evaluation-register", "evaluation-cli", "evaluation-samples"])("%s imports no engine, runner or network module", (name) => {
    const source = readFileSync(new URL(`../src/${name}.ts`, import.meta.url), "utf8");
    const imports = [...source.matchAll(/^import [^;]*from "([^"]+)"/gm)].map((m) => m[1] as string);
    for (const specifier of imports) {
      expect(specifier, `${name} imports ${specifier}`).toMatch(/^(node:(crypto|fs|os|path|url)|@areneon\/schema\/.*|\.\/(canonical-json|evaluation-[a-z]+)\.ts)$/);
    }
  });
});
