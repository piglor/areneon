import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { ajvFor } from "@areneon/schema/ajv";
import { beforeAll, describe, expect, it } from "vitest";
import { syntheticFixtureHashes } from "../src/golden.ts";
import { EngineDataError, readEngineData } from "../src/engine-data.ts";
import { contractVersions, EpisodeError, loadEngine, PolicyProtocolError, readHandshake, readPolicyQuery, readPolicyReply, readEpisode, readEvaluationRunManifest, readTrainingExample, readTrainingSet, TrainingLoaderError } from "../src/index.ts";
import { EvaluationConfigError, readEvaluationParameters, readEntry, RegisterError, readEvaluationPlan, readSealedSetCommitment, readTrainingGateRule, readEvaluationConfig } from "../src/index.ts";
import { contractSamples, SAMPLE_HASHES, SAMPLES_DIR } from "../src/samples.ts";

const SCHEMAS = new URL("../../schema/contracts/v1/", import.meta.url);
const committed = (file: string): unknown => JSON.parse(readFileSync(join(SAMPLES_DIR, file), "utf8"));

let fresh: Record<string, unknown>;
beforeAll(async () => {
  fresh = await contractSamples(loadEngine());
});

describe("the shared contract sample corpus (ADR-008 §2)", () => {
  it("matches a fresh capture: re-run scripts/capture-samples.ts and review the diff", () => {
    const files = readdirSync(join(SAMPLES_DIR, "v1"), { recursive: true, withFileTypes: true })
      .filter((e) => e.isFile())
      .map((e) => join(e.parentPath, e.name).slice(SAMPLES_DIR.length + 1));
    expect([...files, SAMPLE_HASHES].sort()).toEqual(Object.keys(fresh).sort());
    for (const [file, content] of Object.entries(fresh)) expect(committed(file), file).toEqual(content);
  });

  it("covers every contract, with a valid and an invalid case each", () => {
    const contracts = readdirSync(SCHEMAS).map((f) => f.replace(".schema.json", ""));
    for (const contract of contracts) {
      for (const kind of ["valid", "invalid"]) {
        expect(Object.keys(fresh).some((f) => f.startsWith(`v1/${contract}/${kind}/`)), `${contract} ${kind}`).toBe(true);
      }
    }
  });

  it("is classified by Ajv as its directory says: only invalid fails; reader-invalid is schema-valid", () => {
    // One parsed schema object per contract: Ajv compiles per object, so parsing it again for every sample recompiled it each time.
    const schemas = new Map<string, { $id: string }>();
    for (const [file, content] of Object.entries(fresh)) {
      if (file === SAMPLE_HASHES) continue;
      const [, contract, kind] = file.split("/") as [string, string, string];
      if (!schemas.has(contract)) schemas.set(contract, JSON.parse(readFileSync(new URL(`${contract}.schema.json`, SCHEMAS), "utf8")) as { $id: string });
      const schema = schemas.get(contract) as { $id: string };
      const profile = schema.$id.includes(":engine:") ? "engine-mirror" : "contract";
      expect(ajvFor(profile).validate(schema, content), file).toBe(kind !== "invalid");
    }
  }, 30_000); // headroom only: about 1 s locally; the default 5 s was already main's and a loaded CI runner came close

  // ADR-009 section 4, test 2: schema-valid, yet readEpisode refuses each, for the named rule.
  it.each([
    ["v1/episode/reader-invalid/sparring-synthetic.json", "a sparring Episode is private: it records a battle against a human"],
    [
      "v1/episode/reader-invalid/class-below-opponent.json",
      "the Episode is labelled synthetic, but its own team is synthetic and its opponent's team is private: the class must be private",
    ],
  ])("readEpisode refuses %s: %s", async (file, message) => {
    expect(Object.keys(fresh)).toContain(file);
    const known_synthetic = syntheticFixtureHashes(loadEngine());
    await expect(readEpisode(committed(file), { known_synthetic })).rejects.toThrow(EpisodeError);
    await expect(readEpisode(committed(file), { known_synthetic })).rejects.toThrow(message);
  });

  it("holds every reader-invalid sample under its contract's directory, and each is refused above", () => {
    const readerInvalid = Object.keys(fresh).filter((f) => f.includes("/reader-invalid/"));
    expect(readerInvalid.sort()).toEqual([
      "v1/checkpoint-record/reader-invalid/duplicate-training-sets.json",
      "v1/checkpoint-record/reader-invalid/exploiter-of-self.json",
      "v1/checkpoint-record/reader-invalid/opponent-is-self.json",
      "v1/checkpoint-record/reader-invalid/parent-is-self.json",
      "v1/checkpoint-record/reader-invalid/unsorted-league-opponents.json",
      "v1/engine-data/reader-invalid/dangling-type.json",
      "v1/engine-data/reader-invalid/duplicate-species.json",
      "v1/engine-data/reader-invalid/short-damage-row.json",
      "v1/engine-data/reader-invalid/unsorted-species.json",
      "v1/episode/reader-invalid/class-below-opponent.json",
      "v1/episode/reader-invalid/sparring-synthetic.json",
      "v1/evaluation-parameters/reader-invalid/alpha-not-whole-ppm.json",
      "v1/evaluation-parameters/reader-invalid/b-boot-too-small.json",
      "v1/evaluation-parameters/reader-invalid/n-teams-below-floor.json",
      "v1/evaluation-parameters/reader-invalid/n-teams-without-floor.json",
      "v1/evaluation-parameters/reader-invalid/r-cov-too-small.json",
      "v1/evaluation-parameters/reader-invalid/seen-and-held-out.json",
      "v1/evaluation-parameters/reader-invalid/unsorted-baselines.json",
      "v1/evaluation-plan/reader-invalid/baseline-with-lineage.json",
      "v1/evaluation-plan/reader-invalid/candidate-roles-swapped.json",
      "v1/evaluation-plan/reader-invalid/unsorted-opponents.json",
      "v1/evaluation-register-entry/reader-invalid/failed-without-reason.json",
      "v1/evaluation-register-entry/reader-invalid/impossible-date.json",
      "v1/evaluation-register-entry/reader-invalid/reported-without-outcome-hash.json",
      "v1/evaluation-register-entry/reader-invalid/unsorted-affected-candidates.json",
      "v1/evaluation-run-manifest/reader-invalid/duplicate.json",
      "v1/evaluation-run-manifest/reader-invalid/unsorted.json",
      "v1/policy-handshake/reader-invalid/duplicate-contracts.json",
      "v1/policy-handshake/reader-invalid/unsorted-checkpoints.json",
      "v1/policy-handshake/reader-invalid/unsorted-contracts.json",
      "v1/policy-query/reader-invalid/duplicate-candidate.json",
      "v1/policy-query/reader-invalid/identity-differs.json",
      "v1/sealed-set-commitment/reader-invalid/strata-sum-mismatch.json",
      "v1/sealed-set-commitment/reader-invalid/unsorted-strata.json",
      "v1/training-example/reader-invalid/choice-not-in-candidates.json",
      "v1/training-gate-rule/reader-invalid/alternative-below-null.json",
      "v1/training-gate-rule/reader-invalid/unsorted-claims.json",
      "v1/training-set/reader-invalid/duplicate-manifests.json",
      "v1/training-set/reader-invalid/learner-ready-no-policy.json",
      "v1/training-set/reader-invalid/unsorted-episodes.json",
    ]);
  });

  // ADR-009 section 4: the manifests' readers refuse what the profile cannot say.
  it.each([
    ["v1/evaluation-run-manifest/reader-invalid/unsorted.json", readEvaluationRunManifest, "sorted and unique"],
    ["v1/evaluation-run-manifest/reader-invalid/duplicate.json", readEvaluationRunManifest, "sorted and unique"],
    ["v1/training-set/reader-invalid/unsorted-episodes.json", readTrainingSet, "episode_hashes must be sorted and unique"],
    ["v1/training-example/reader-invalid/choice-not-in-candidates.json", readTrainingExample, "is not among its"],
    ["v1/training-set/reader-invalid/learner-ready-no-policy.json", readTrainingSet, "must name its behaviour_policy_id"],
    ["v1/training-set/reader-invalid/duplicate-manifests.json", readTrainingSet, "evaluation_manifest_hashes must be sorted and unique"],
  ])("the reader refuses %s", (file, reader, message) => {
    expect(Object.keys(fresh)).toContain(file);
    expect(() => reader(committed(file))).toThrow(TrainingLoaderError);
    expect(() => reader(committed(file))).toThrow(message);
  });

  // ADR-010 section 2: the worker protocol's readers refuse what the profile cannot say.
  it.each([
    ["v1/policy-query/reader-invalid/identity-differs.json", readPolicyQuery, "identity differs from its observation"],
    ["v1/policy-query/reader-invalid/duplicate-candidate.json", readPolicyQuery, "is listed twice"],
    ["v1/policy-handshake/reader-invalid/unsorted-contracts.json", readHandshake, "sorted by name"],
    ["v1/policy-handshake/reader-invalid/duplicate-contracts.json", readHandshake, "sorted by name"],
    ["v1/policy-handshake/reader-invalid/unsorted-checkpoints.json", readHandshake, "sorted by role"],
  ])("the policy reader refuses %s", (file, reader, message) => {
    expect(Object.keys(fresh)).toContain(file);
    expect(() => reader(committed(file))).toThrow(PolicyProtocolError);
    expect(() => reader(committed(file))).toThrow(message);
  });

  it("the policy readers accept the valid samples, and the hello lists this side's contracts", () => {
    for (const name of ["battle", "preview", "battle-a2"]) expect(readPolicyQuery(committed(`v1/policy-query/valid/${name}.json`)).schema_version).toBe(1);
    for (const name of ["recommendation", "unsupported-candidate", "invalid-query"]) {
      expect(readPolicyReply(committed(`v1/policy-reply/valid/${name}.json`)).query_id).toBe(1);
    }
    expect(readHandshake(committed("v1/policy-handshake/valid/hello.json")).message.contracts).toEqual(contractVersions());
    expect(readHandshake(committed("v1/policy-handshake/valid/ready.json")).message.kind).toBe("ready");
    expect(() => readPolicyReply({})).toThrow(PolicyProtocolError);
  });

  // ADR-010 section 3: the engine-data reader refuses what the profile cannot say.
  it.each([
    ["v1/engine-data/reader-invalid/unsorted-species.json", /species: .* repeated or out of order/],
    ["v1/engine-data/reader-invalid/duplicate-species.json", /species: .* repeated or out of order/],
    ["v1/engine-data/reader-invalid/dangling-type.json", /unknown type/],
    ["v1/engine-data/reader-invalid/short-damage-row.json", /damage_taken is not one entry per type/],
  ])("readEngineData refuses %s", (file, message) => {
    expect(Object.keys(fresh)).toContain(file);
    expect(() => readEngineData(committed(file))).toThrow(EngineDataError);
    expect(() => readEngineData(committed(file))).toThrow(message);
  });

  it("holds a valid engine-data sample the reader accepts", () => {
    expect(readEngineData(committed("v1/engine-data/valid/golden.json")).species).toHaveLength(2);
  });

  // ADR-010 sections 6, 7 and O1: the evaluation readers refuse what the profile cannot say.
  it.each([
    ["v1/evaluation-parameters/reader-invalid/alpha-not-whole-ppm.json", readEvaluationParameters, "not a whole number of ppm"],
    ["v1/evaluation-parameters/reader-invalid/b-boot-too-small.json", readEvaluationParameters, "must exceed"],
    ["v1/evaluation-parameters/reader-invalid/r-cov-too-small.json", readEvaluationParameters, "at least 1 / alpha'"],
    ["v1/evaluation-parameters/reader-invalid/unsorted-baselines.json", readEvaluationParameters, "sorted and unique"],
    ["v1/evaluation-parameters/reader-invalid/seen-and-held-out.json", readEvaluationParameters, "both seen and held out"],
    ["v1/evaluation-parameters/reader-invalid/n-teams-below-floor.json", readEvaluationParameters, "below its floor"],
    ["v1/evaluation-parameters/reader-invalid/n-teams-without-floor.json", readEvaluationParameters, "n_teams is set"],
    ["v1/training-gate-rule/reader-invalid/unsorted-claims.json", readTrainingGateRule, "sorted by claim_id"],
    ["v1/training-gate-rule/reader-invalid/alternative-below-null.json", readTrainingGateRule, "must lie above"],
    ["v1/sealed-set-commitment/reader-invalid/strata-sum-mismatch.json", readSealedSetCommitment, "hold 32 teams"],
    ["v1/sealed-set-commitment/reader-invalid/unsorted-strata.json", readSealedSetCommitment, "sorted by label"],
    ["v1/evaluation-plan/reader-invalid/candidate-roles-swapped.json", readEvaluationPlan, "battle checkpoint, then the preview"],
    ["v1/evaluation-plan/reader-invalid/unsorted-opponents.json", readEvaluationPlan, "sorted by opponent_id"],
    ["v1/evaluation-plan/reader-invalid/baseline-with-lineage.json", readEvaluationPlan, "no lineage"],
  ])("the evaluation reader refuses %s", (file, reader, message) => {
    expect(Object.keys(fresh)).toContain(file);
    expect(() => reader(committed(file))).toThrow(EvaluationConfigError);
    expect(() => reader(committed(file))).toThrow(message);
  });

  it.each([
    ["v1/evaluation-register-entry/reader-invalid/reported-without-outcome-hash.json", "needs its outcome_hash"],
    ["v1/evaluation-register-entry/reader-invalid/failed-without-reason.json", "needs its reason"],
    ["v1/evaluation-register-entry/reader-invalid/unsorted-affected-candidates.json", "sorted and unique"],
    ["v1/evaluation-register-entry/reader-invalid/impossible-date.json", "not a real UTC date"],
  ])("readEntry refuses %s", (file, message) => {
    expect(Object.keys(fresh)).toContain(file);
    expect(() => readEntry(committed(file))).toThrow(RegisterError);
    expect(() => readEntry(committed(file))).toThrow(message);
  });

  it("the valid evaluation samples read, and the provisional parameters pair with the rule", () => {
    const rule = committed("v1/training-gate-rule/valid/v1.json");
    expect(readEvaluationConfig(committed("v1/evaluation-parameters/valid/provisional.json"), rule).parameters.max_turns).toBe(300);
    expect(readEvaluationConfig(committed("v1/evaluation-parameters/valid/complete.json"), rule).parameters.n_teams).toBe(32);
    expect(readEvaluationPlan(committed("v1/evaluation-plan/valid/registered.json")).set_id).toBe("set-1");
    expect(readSealedSetCommitment(committed("v1/sealed-set-commitment/valid/four-strata.json")).team_count).toBe(32);
    for (const kind of ["set-commitment", "plan-registration", "l2a-recheck", "run-outcome", "set-retirement", "incident"]) {
      expect(readEntry(committed(`v1/evaluation-register-entry/valid/${kind}.json`)).schema_version).toBe(1);
    }
  });
});
