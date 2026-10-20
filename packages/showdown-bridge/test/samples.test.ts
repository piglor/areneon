import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { ajvFor } from "@areneon/schema/ajv";
import { beforeAll, describe, expect, it } from "vitest";
import { syntheticFixtureHashes } from "../src/golden.ts";
import { EpisodeError, loadEngine, readEpisode, readEvaluationRunManifest, readTrainingExample, readTrainingSet, TrainingLoaderError } from "../src/index.ts";
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
    for (const [file, content] of Object.entries(fresh)) {
      if (file === SAMPLE_HASHES) continue;
      const [, contract, kind] = file.split("/") as [string, string, string];
      const schema = JSON.parse(readFileSync(new URL(`${contract}.schema.json`, SCHEMAS), "utf8")) as { $id: string };
      const profile = schema.$id.includes(":engine:") ? "engine-mirror" : "contract";
      expect(ajvFor(profile).validate(schema, content), file).toBe(kind !== "invalid");
    }
  });

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
      "v1/episode/reader-invalid/class-below-opponent.json",
      "v1/episode/reader-invalid/sparring-synthetic.json",
      "v1/evaluation-run-manifest/reader-invalid/duplicate.json",
      "v1/evaluation-run-manifest/reader-invalid/unsorted.json",
      "v1/training-example/reader-invalid/choice-not-in-candidates.json",
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
});
