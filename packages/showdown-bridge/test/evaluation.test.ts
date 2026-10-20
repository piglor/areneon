// The edits below reach into nested JSON documents by path; `any` keeps each mutation one line.
/* eslint-disable @typescript-eslint/no-explicit-any */
import { execFileSync } from "node:child_process";
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
  domainHash,
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
  generateSecret,
  keyCommitment,
  planHash,
  readEvaluationPlan,
  readSealedSetCommitment,
  type SealedSetCommitment,
  setCommitment,
} from "../src/evaluation-plan.ts";
import { appendEntry, assertPlanPublished, checkHeads, checkHeadsAddOnly, entryHash, GENESIS, readEntry, readHead, RegisterError, verifyRegister } from "../src/evaluation-register.ts";
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
    expect(config.parameters_hash).toBe(domainHash("evaluation-parameters", fx.provisional));
    expect(config.rule_hash).toBe(domainHash("training-gate-rule", fx.rule));
    expect(domainHash("evaluation-plan", fx.provisional)).not.toBe(config.parameters_hash);
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
    ["a family without a G3b claim", (r: Record<string, any>) => r.claims.pop(), /must include a G3b claim/],
    ["a family without a G2 claim", (r: Record<string, any>) => (r.claims = r.claims.slice(2)), /must include a G2 claim/],
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

  it("pins the gate conditions G6 and G7 the ADR names, as data in the committed rule", () => {
    expect(fx.rule.required_checks).toEqual([
      "a1-a2-observation-equality",
      "batch-invariance",
      "candidate-parser-roundtrip",
      "d11-coverage",
      "encoder-equality",
      "enumerator-worst-case-oracle",
      "l2a-recheck",
      "privacy-tests",
    ]);
  });

  it("holds the committed records as data: the files the CLI reads are the ones the fixtures use", () => {
    expect(JSON.parse(readFileSync(PARAMETERS_FILE, "utf8"))).toEqual(fx.provisional);
    expect(JSON.parse(readFileSync(RULE_FILE, "utf8"))).toEqual(fx.rule);
    expect(fx.provisional.max_turns).toBe(300);
  });
});

describe("the sealed-set commitment and the plan (protocol L3, I2)", () => {
  it("commits to {salt, teams} and the key by hash; a reveal opens it only if it is the set declared", () => {
    const salt = hex("a");
    const key = hex("b");
    const teams: Json[] = Array.from({ length: 32 }, (_, i) => ({ species: `s${i}`, band: ["a", "b", "c", "d"][i % 4] as string }));
    const labels = { stratumOf: (team: Json) => `speed=${(team as { band: string }).band}`, n_stratum_min: 2 };
    const commitment = { ...fx.commitment, set_commitment: setCommitment(salt, teams), key_commitment: keyCommitment(key) };
    expect(checkReveal(commitment, { salt, teams, key }, labels)).toBe(true);
    expect(checkReveal(commitment, { salt, teams: [...teams].reverse(), key }, labels)).toBe(false);
    expect(checkReveal(commitment, { salt: hex("c"), teams, key }, labels)).toBe(false);
    expect(checkReveal(commitment, { salt, teams, key: hex("c") }, labels)).toBe(false);
  });

  it("a reveal of fewer teams, other strata or a thin stratum does not open a commitment it hashes to", () => {
    const salt = hex("a");
    const key = hex("b");
    const open = (teams: Json[], declared: SealedSetCommitment) => {
      const commitment = { ...declared, set_commitment: setCommitment(salt, teams), key_commitment: keyCommitment(key) };
      return checkReveal(commitment, { salt, teams, key }, { stratumOf: (t) => `speed=${(t as { band: string }).band}`, n_stratum_min: 2 });
    };
    const one: Json[] = [{ band: "a" }];
    expect(open(one, { ...fx.commitment, team_count: 32 })).toBe(false);
    const wrongStrata: Json[] = Array.from({ length: 32 }, () => ({ band: "z" }));
    expect(open(wrongStrata, fx.commitment)).toBe(false);
    const thin = { ...fx.commitment, team_count: 3, stratum_counts: [{ stratum: "speed=a", count: 2 }, { stratum: "speed=b", count: 1 }] } as SealedSetCommitment;
    expect(open([{ band: "a" }, { band: "a" }, { band: "b" }], thin)).toBe(false);
  });

  it("generates 256-bit secrets that the commitment functions accept", () => {
    const secret = generateSecret();
    expect(secret).toMatch(/^[0-9a-f]{64}$/);
    expect(generateSecret()).not.toBe(secret);
    expect(() => keyCommitment(secret)).not.toThrow();
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
    expect(planHash(plan)).toBe(domainHash("evaluation-plan", fx.plan));
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

  it("refuses a date that is only shaped like one", () => {
    expect(() => verifyRegister(`${asLine({ ...entryAt(1), recorded_at: "9999-99-99T99:99:99Z" })}\n`)).toThrow(/not a real UTC date/);
    expect(() => verifyRegister(`${asLine({ ...entryAt(1), recorded_at: "2026-02-30T10:00:00Z" })}\n`)).toThrow(/not a real UTC date/);
  });

  it("refuses a chain-valid line the records would not allow, once the verifier has the records", () => {
    const crafted = appendEntry("", { kind: "set_commitment", recorded_at: "2026-11-01T10:00:00Z", operator: "owner", commitment: { ...fx.commitment, b_sealed: 5, team_count: 3, stratum_counts: [{ stratum: "a", count: 3 }], evaluation_parameters_hash: hex("1") } }).text;
    expect(verifyRegister(crafted).entries).toHaveLength(1);
    expect(() => verifyRegister(crafted, [fx.config])).toThrow(/names records .* that were not given/);
    const sameRecords = appendEntry("", { kind: "set_commitment", recorded_at: "2026-11-01T10:00:00Z", operator: "owner", commitment: { ...fx.commitment, team_count: 3, stratum_counts: [{ stratum: "a", count: 3 }] } }).text;
    expect(() => verifyRegister(sameRecords, [fx.config])).toThrow(/differs from the parameters' n_teams/);
  });

  it("takes the run budget from the parameters, not from the commitment's own field", () => {
    // The commitment claims a budget of 2; the parameters (b_sealed 1) win once they are given.
    const greedy = { ...fx.commitment, b_sealed: 2 };
    const first = appendEntry("", { kind: "set_commitment", recorded_at: at, operator: "owner", commitment: greedy });
    const plan = { ...fx.plan, set_commitment_hash: commitmentHash(greedy) };
    const second = appendEntry(first.text, { kind: "plan_registration", recorded_at: at, operator: "owner", plan: { ...plan, a2_matches_min: 11 } });
    expect(() => appendEntry(second.text, { kind: "plan_registration", recorded_at: at, operator: "owner", plan: { ...plan, a2_matches_min: 12 } })).toThrow(/candidate .* already has a plan/);
    expect(() => verifyRegister(first.text, [fx.config])).toThrow(/differs from the parameters' 1/);
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

describe("head files are add-only (O1)", () => {
  const file = (seq: number, hash = hex("1")) => [`${String(seq).padStart(10, "0")}.json`, JSON.stringify({ schema_version: 1, seq, entry_hash: hash })] as const;
  const files = (...entries: (readonly [string, string])[]) => new Map(entries);

  it("accepts a new head beyond the published ones and refuses a modification, a deletion and a new head behind", () => {
    const base = files(file(2));
    expect(() => checkHeadsAddOnly(base, files(file(2), file(3)))).not.toThrow();
    expect(() => checkHeadsAddOnly(base, files(file(2, hex("2"))))).toThrow(/was modified/);
    expect(() => checkHeadsAddOnly(base, files())).toThrow(/was deleted/);
    expect(() => checkHeadsAddOnly(base, files(file(2), file(1)))).toThrow(/not beyond the published 2/);
  });

  it("refuses a file that is not a head or is named for another position", () => {
    expect(() => checkHeadsAddOnly(files(), files(["0000000001.json", "{oops"]))).toThrow(/0000000001.json/);
    expect(() => checkHeadsAddOnly(files(), files(["1.json", file(1)[1]]))).toThrow(/file name must be the zero-padded position/);
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
  const run = (...args: string[]) => runEvaluationCli(args, clock, dir);
  const records = ["--parameters", write("params.json", fx.complete), "--rule", write("rule.json", fx.rule)];
  const register = p("register.jsonl");
  const heads = p("heads");
  const hashes = (): string[] => verifyRegister(readFileSync(register, "utf8")).hashes;
  const publish = (seq: number, hash = hashes()[seq - 1]) => {
    mkdirSync(heads, { recursive: true });
    writeFileSync(join(heads, `${String(seq).padStart(10, "0")}.json`), JSON.stringify({ schema_version: 1, seq, entry_hash: hash }));
  };

  it("config-check prints the hashes and refuses a provisional record under --complete; the defaults are the committed v1 files", () => {
    const ok = run("config-check", ...records);
    expect(ok.code).toBe(0);
    expect(JSON.parse(ok.out)).toEqual({ parameters_hash: fx.config.parameters_hash, rule_hash: fx.config.rule_hash, alpha_prime_ppm: 12500, complete: true });
    const provisional = run("config-check");
    expect(JSON.parse(provisional.out).complete).toBe(false);
    expect(run("config-check", "--complete").err).toMatch(/provisional/);
    expect(run("config-check", ...records, "--parameters", PARAMETERS_FILE).code).toBe(2);
    expect(PARAMETERS_FILE).toMatch(/evaluation-parameters\.v1\.json$/);
    expect(RULE_FILE).toMatch(/training-gate-rule\.v1\.json$/);
  });

  it("a mistyped register path cannot start a chain: only init does, and only once", () => {
    for (const command of [
      ["commit-set", "--commitment", write("commitment.json", fx.commitment)],
      ["register-plan", "--plan", write("plan.json", fx.plan)],
      ["append", "--entry", write("e.json", { kind: "incident" })],
    ]) {
      const result = run(command[0] as string, "--register", p("typo.jsonl"), ...command.slice(1), ...records, "--operator", "owner");
      expect(result.code).toBe(1);
      expect(result.err).toMatch(/does not exist \(a new register is started only with init\)/);
    }
    expect(existsSync(p("typo.jsonl"))).toBe(false);
    expect(run("verify", "--register", p("typo.jsonl")).code).toBe(1);
    expect(run("head", "--register", p("typo.jsonl")).code).toBe(1);
    expect(run("init", "--register", register).code).toBe(0);
    expect(readFileSync(register, "utf8")).toBe("");
    expect(run("init", "--register", register).err).toMatch(/already exists/);
    expect(run("head", "--register", register, ...records).err).toMatch(/empty/);
  });

  it("commits a set and registers a plan, refusing what does not match the records", () => {
    const commit = run("commit-set", "--register", register, "--commitment", p("commitment.json"), ...records, "--operator", "owner");
    expect(commit.code, commit.err).toBe(0);
    expect(JSON.parse(commit.out).head.seq).toBe(1);
    const before = readFileSync(register, "utf8");
    const bad = run("register-plan", "--register", register, "--plan", write("bad-plan.json", { ...fx.plan, alpha_ppm: 100000 }), ...records, "--operator", "owner");
    expect(bad.code).toBe(1);
    expect(bad.err).toMatch(/alpha_ppm 100000 differs/);
    expect(run("register-plan", "--register", register, "--plan", write("set2.json", { ...fx.plan, set_id: "set-2" }), ...records, "--operator", "owner").err).toMatch(/set set-2 has no commitment/);
    expect(run("register-plan", "--register", register, "--plan", p("plan.json"), "--operator", "owner").err).toMatch(/names records .* that were not given/);
    expect(readFileSync(register, "utf8")).toBe(before);
    const registered = run("register-plan", "--register", register, "--plan", p("plan.json"), ...records, "--operator", "owner", "--recorded-at", "2026-11-07T09:00:00Z");
    expect(registered.code, registered.err).toBe(0);
    expect(JSON.parse(registered.out).plan_hash).toBe(planHash(fx.plan));
  });

  it("refuses to append a run entry for a plan whose head is not published, and accepts it once it is", () => {
    const recheck = write("recheck.json", { kind: "l2a_recheck", set_id: "set-1", plan_hash: planHash(fx.plan), result: "pass" });
    const append = (...extra: string[]) => run("append", "--register", register, "--entry", recheck, ...records, "--operator", "evaluation-job", "--recorded-at", "2026-11-07T10:00:00Z", ...extra);
    expect(append().code).toBe(2);
    expect(append().err).toMatch(/records a run: give --heads/);
    expect(append("--heads-ref", "main").err).toMatch(/--heads-ref needs --heads/);
    expect(append("--heads", heads).err).toMatch(/merge the chain head first/);
    publish(2);
    const done = append("--heads", heads);
    expect(done.code, done.err).toBe(0);
    const wrongPlan = write("run.json", { kind: "run_outcome", plan_hash: hex("1"), outcome: "failed", reason: "x" });
    expect(run("append", "--register", register, "--entry", wrongPlan, ...records, "--operator", "o", "--heads", heads).err).toMatch(/plan .* is not in the register/);
  });

  it("append refuses the kinds that carry records, a non-object and an entry the chain refuses", () => {
    const opts = ["--register", register, ...records, "--operator", "o", "--recorded-at", "2026-11-08T10:00:00Z"];
    expect(run("append", ...opts, "--entry", write("c.json", { kind: "set_commitment" })).err).toMatch(/use commit-set/);
    expect(run("append", ...opts, "--entry", write("pr.json", { kind: "plan_registration" })).err).toMatch(/use register-plan/);
    expect(run("append", ...opts, "--entry", write("arr.json", [])).code).toBe(2);
    const refused = run("append", ...opts, "--entry", write("inc.json", { kind: "incident", set_id: "nope", cause: "plan_deviation", description: "x", affected_candidates: [] }));
    expect(refused.code).toBe(1);
    expect(refused.err).toMatch(/has no commitment/);
  });

  it("verify checks every line against the records, and ties the commitment to the files given", () => {
    const verified = JSON.parse(run("verify", "--register", register, ...records).out);
    expect(verified).toMatchObject({ entries: 3, sets: 1, published_through: null });
    expect(JSON.parse(run("head", "--register", register, ...records).out)).toEqual(verified.head);
    expect(readFileSync(register, "utf8").split("\n")).toHaveLength(4);
    expect(JSON.parse(readFileSync(register, "utf8").split("\n")[0] as string).recorded_at).toBe("2026-11-07T08:09:10Z");
    // Without the records the set names, and with the committed (provisional) v1 files, the verifier refuses.
    expect(run("verify", "--register", register).err).toMatch(/names records .* that were not given/);
    // Editing a records file in place changes its hash: the commitment no longer ties to it.
    const edited = write("params-edited.json", { ...fx.complete, max_turns: 301 });
    expect(run("verify", "--register", register, "--parameters", edited, "--rule", p("rule.json")).err).toMatch(/names records .* that were not given/);
    expect(run("verify", "--register", register, "--parameters", edited, "--rule", p("rule.json"), ...records).code).toBe(0);
    expect(run("verify", "--register", register, ...records, "--rule", p("rule.json")).code).toBe(2);
  });

  it("verify checks the published heads; --published gates a run on its plan's head", () => {
    publish(1);
    rmHead(2);
    expect(JSON.parse(run("verify", "--register", register, ...records, "--heads", heads).out)).toMatchObject({ published_through: 1, unpublished: 2 });
    const gated = run("verify", "--register", register, ...records, "--heads", heads, "--published", planHash(fx.plan));
    expect(gated.code).toBe(1);
    expect(gated.err).toMatch(/merge the chain head first/);
    publish(2);
    expect(run("verify", "--register", register, ...records, "--heads", heads, "--published", planHash(fx.plan)).code).toBe(0);
    expect(run("verify", "--register", register, ...records, "--published", planHash(fx.plan)).code).toBe(2);
    expect(JSON.parse(run("heads-check", "--dir", heads).out)).toEqual({ heads: 2, latest: 2 });
    expect(JSON.parse(run("heads-check", "--dir", p("no-such-dir")).out)).toEqual({ heads: 0, latest: null });
  });

  function rmHead(seq: number): void {
    const file = join(heads, `${String(seq).padStart(10, "0")}.json`);
    if (existsSync(file)) execFileSync("rm", [file]);
  }

  it("verify refuses a rolled-back, a tampered and a forked register against the heads", () => {
    const text = readFileSync(register, "utf8");
    const rows = text.slice(0, -1).split("\n");
    writeFileSync(p("rolled-back.jsonl"), `${rows[0]}\n`);
    expect(run("verify", "--register", p("rolled-back.jsonl"), ...records, "--heads", heads).err).toMatch(/rollback/);
    const tampered = [...rows];
    tampered[1] = (tampered[1] as string).replace("owner", "owneR");
    writeFileSync(p("tampered.jsonl"), `${tampered.join("\n")}\n`);
    expect(run("verify", "--register", p("tampered.jsonl"), ...records, "--heads", heads).code).toBe(1);
    const fork = appendEntry(`${rows[0]}\n`, { kind: "plan_registration", recorded_at: "2026-11-07T09:30:00Z", operator: "owner", plan: { ...fx.plan, a2_matches_min: 99 } });
    writeFileSync(p("fork.jsonl"), fork.text);
    expect(run("verify", "--register", p("fork.jsonl"), ...records, "--heads", heads).err).toMatch(/fork or rewrite: position 2/);
  });

  it("an unmerged head in the working tree does not count when the heads are read from main", () => {
    const repo = p("repo");
    mkdirSync(join(repo, "evaluation/register-heads"), { recursive: true });
    const git = (...args: string[]) => execFileSync("git", ["-c", "user.name=t", "-c", "user.email=t@example.com", ...args], { cwd: repo, stdio: "pipe" });
    git("init", "-q", "-b", "main");
    const name = `${String(1).padStart(10, "0")}.json`;
    writeFileSync(join(repo, "evaluation/register-heads", name), JSON.stringify({ schema_version: 1, seq: 1, entry_hash: hashes()[0] }));
    git("add", "."); git("commit", "-q", "-m", "head 1");
    // A local, unmerged head 2 satisfies the working tree, not main.
    writeFileSync(join(repo, "evaluation/register-heads", `${String(2).padStart(10, "0")}.json`), JSON.stringify({ schema_version: 1, seq: 2, entry_hash: hashes()[1] }));
    const inRepo = (...args: string[]) => runEvaluationCli(args, clock, repo);
    const common = ["--register", register, ...records, "--heads", "evaluation/register-heads", "--published", planHash(fx.plan)];
    expect(inRepo("verify", ...common).code).toBe(0);
    const trusted = inRepo("verify", ...common, "--heads-ref", "main");
    expect(trusted.code).toBe(1);
    expect(trusted.err).toMatch(/merge the chain head first/);
    expect(inRepo("verify", ...common, "--heads-ref", "no-such-ref").err).toMatch(/git ls-tree no-such-ref failed/);
    // heads-check --base: head 2 is added beyond head 1 (ok); modifying head 1 is refused.
    expect(JSON.parse(inRepo("heads-check", "--dir", "evaluation/register-heads", "--base", "main").out)).toEqual({ heads: 2, latest: 2 });
    writeFileSync(join(repo, "evaluation/register-heads", name), JSON.stringify({ schema_version: 1, seq: 1, entry_hash: hex("1") }));
    expect(inRepo("heads-check", "--dir", "evaluation/register-heads", "--base", "main").err).toMatch(/was modified/);
  });

  it("refuses a head file whose name is not its position", () => {
    const misnamed = p("misnamed");
    mkdirSync(misnamed);
    writeFileSync(join(misnamed, "a.json"), JSON.stringify({ schema_version: 1, seq: 1, entry_hash: hex("1") }));
    expect(run("heads-check", "--dir", misnamed).err).toMatch(/file name must be the zero-padded position/);
  });

  it("gen-secret writes 256 random bits readable by the owner only, never overwrites, and prints only the hash", () => {
    const made = run("gen-secret", "--out", p("key"));
    expect(made.code).toBe(0);
    const secret = readFileSync(p("key"), "utf8");
    expect(secret).toMatch(/^[0-9a-f]{64}$/);
    expect(JSON.parse(made.out)).toEqual({ sha256: keyCommitment(secret) });
    expect(made.out).not.toContain(secret);
    expect(execFileSync("stat", ["-c", "%a", p("key")], { encoding: "utf8" }).trim()).toBe("600");
    expect(() => run("gen-secret", "--out", p("key"))).toThrow(/EEXIST/);
  });

  it("lets an operating-system error through instead of reporting it as a refusal", () => {
    expect(() => run("init", "--register", p("no-such-dir/register.jsonl"))).toThrow(/ENOENT/);
  });

  it("usage errors exit 2", () => {
    expect(run().code).toBe(2);
    expect(run("nope").err).toMatch(/unknown command nope/);
    expect(run("verify").err).toMatch(/--register is required/);
    expect(run("verify", "register").err).toMatch(/unexpected argument/);
    expect(run("verify", "--register").err).toMatch(/needs a value/);
    expect(run("config-check", "--parameters", p("missing.json"), "--rule", p("rule.json")).err).toMatch(/cannot read/);
  });

  it("uses the real clock and the process directory by default", () => {
    expect(runEvaluationCli(["init", "--register", p("clock.jsonl")]).code).toBe(0);
    const stamped = runEvaluationCli(["commit-set", "--register", p("clock.jsonl"), "--commitment", p("commitment.json"), ...records, "--operator", "owner"]);
    expect(stamped.code, stamped.err).toBe(0);
    expect(JSON.parse(readFileSync(p("clock.jsonl"), "utf8").split("\n")[0] as string).recorded_at).toMatch(/^20\d\d-\d\d-\d\dT\d\d:\d\d:\d\dZ$/);
  });
});

describe("the evaluation modules are engine-free", () => {
  it.each(["evaluation-config", "evaluation-plan", "evaluation-register", "evaluation-cli", "evaluation-samples"])("%s imports no engine, runner or network module", (name) => {
    const source = readFileSync(new URL(`../src/${name}.ts`, import.meta.url), "utf8");
    const imports = [...source.matchAll(/^import [^;]*from "([^"]+)"/gm)].map((m) => m[1] as string);
    for (const specifier of imports) {
      expect(specifier, `${name} imports ${specifier}`).toMatch(/^(node:(child_process|crypto|fs|os|path|url)|@areneon\/schema\/.*|\.\/(canonical-json|evaluation-[a-z]+)\.ts)$/);
    }
  });
});
