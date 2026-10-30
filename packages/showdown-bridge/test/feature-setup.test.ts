// The run config names the candidate feature set (ADR-022 sections 3 and 6, slice D3): `featureSetup` checks the named
// hash against this checkout's records, and a generation given the set builds the rows at play time and recomputes
// them into the stream.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { actorRule } from "../src/actor-rule.ts";
import { type Json, jcs } from "../src/canonical-json.ts";
import { loadEngine } from "../src/engine.ts";
import { FEATURE_SET_BELIEF, featureSetHash, featureSetIdentity } from "../src/candidate-features.ts";
import type { BeliefWorldsFactory } from "../src/damage-aware-worlds.ts";
import { configV2, fakePort } from "./damage-aware-world-support.ts";
import { featureSetup, GenerationError, playGeneration, trainingTeamPool } from "../src/league-generation.ts";
import type { PolicyClient } from "../src/policy-worker.ts";
import type { PolicyQueryInput } from "../src/policy-protocol.ts";
import { readTrainingExample } from "../src/training-loader.ts";
import { SAMPLES_DIR } from "../src/samples.ts";
import { readTrainingRunConfig, type TrainingRunConfig } from "../src/training-run-config.ts";

const engine = loadEngine();
const tiny = JSON.parse(readFileSync(join(SAMPLES_DIR, "v1/training-run-config/valid/tiny.json"), "utf8")) as Record<string, unknown>;
const withFeatures = (hash: string, set = "probe-damage/1"): TrainingRunConfig => {
  const raw = structuredClone(tiny) as { model: { encoder: { options?: unknown } } };
  raw.model.encoder.options = { candidate_features: { feature_set: set, feature_set_hash: hash } };
  return readTrainingRunConfig(raw);
};
const plain = readTrainingRunConfig(tiny);
const POLICY = `policy:${"c".repeat(64)}`;

/** This checkout's feature set hash, as the refusal of a wrong one states it. */
const hashOf = (): string => {
  try {
    featureSetup(engine, withFeatures("0".repeat(64)));
  } catch (error) {
    return /computes as ([0-9a-f]{64})/.exec((error as Error).message)?.[1] ?? "";
  }
  return "";
};

describe("featureSetup", () => {
  it("is undefined for a config that names no set, so such a run changes in no byte", () => {
    expect(featureSetup(engine, plain)).toBeUndefined();
  });

  it("refuses a hash this checkout does not compute, naming both", () => {
    expect(() => featureSetup(engine, withFeatures("0".repeat(64)))).toThrow(GenerationError);
    expect(() => featureSetup(engine, withFeatures("0".repeat(64)))).toThrow(/which this checkout computes as [0-9a-f]{64}/);
  });

  it("returns the records and the statement the worker must make when the hash is this checkout's", () => {
    const hash = hashOf();
    const setup = featureSetup(engine, withFeatures(hash)) as NonNullable<ReturnType<typeof featureSetup>>;
    expect(setup.statement).toEqual({ feature_set: "probe-damage/1", feature_set_hash: hash });
    expect(featureSetHash(featureSetIdentity(engine, setup.configs))).toBe(hash);
    expect(setup.configs.data_class).toBe("synthetic");
  });

  it("builds probe-damage-belief/1 from the committed v2 record through the composition root's factory, and fails closed without one", () => {
    const calls: unknown[] = [];
    const factory: BeliefWorldsFactory = (context) => {
      calls.push(context.config.schema_version);
      return fakePort();
    };
    const attempt = (): string => {
      try {
        featureSetup(engine, withFeatures("0".repeat(64), FEATURE_SET_BELIEF), factory);
      } catch (error) {
        return /computes as ([0-9a-f]{64})/.exec((error as Error).message)?.[1] ?? "";
      }
      return "";
    };
    const hash = attempt();
    expect(hash).toMatch(/^[0-9a-f]{64}$/);
    expect(calls).toEqual([2]);
    const setup = featureSetup(engine, withFeatures(hash, FEATURE_SET_BELIEF), factory) as NonNullable<ReturnType<typeof featureSetup>>;
    expect(setup.statement).toEqual({ feature_set: FEATURE_SET_BELIEF, feature_set_hash: hash });
    expect(setup.configs.belief?.worlds).toBe(configV2.assumed_set.worlds);
    expect(hash).not.toBe(hashOf());
    expect(() => featureSetup(engine, withFeatures(hash, FEATURE_SET_BELIEF))).toThrow(/belief port of the composition root/);
  });

  it("refuses a format whose vocabulary it does not hold", () => {
    const other = { ...withFeatures("0".repeat(64)), format_id: "gen9othermod" } as TrainingRunConfig;
    expect(() => featureSetup(engine, other)).toThrow(/vocabulary for/);
  });
});

describe("a generation that reads a feature set", () => {
  /** A stand-in worker that states the set and plays the actor rule over each query's candidates. */
  function standIn(statement: { feature_set: string; feature_set_hash: string } | null): PolicyClient & { beginGame: () => Promise<void>; queries: PolicyQueryInput[] } {
    const queries: PolicyQueryInput[] = [];
    return {
      queries,
      featureSet: statement,
      beginGame: () => Promise.resolve(),
      ask(_battle, query) {
        queries.push(query);
        const open = query.candidates.filter((c) => !c.pending).map((c) => c.choice);
        const pending = query.candidates.filter((c) => c.pending).map((c) => c.choice);
        return Promise.resolve({ schema_version: 1, identity: query.identity, choice: actorRule({ choices: open, pending }), source_policy_id: query.policy_id, elapsed_ms: 0, deadline_outcome: "in_budget" });
      },
    };
  }

  it("builds rows at play time, and the stream's examples carry the very rows that were sent", async () => {
    const hash = hashOf();
    const config = withFeatures(hash);
    const setup = featureSetup(engine, config) as NonNullable<ReturnType<typeof featureSetup>>;
    const pool = trainingTeamPool(engine, config.generation.excluded_abilities);
    const client = standIn(setup.statement);
    const result = await playGeneration({ engine, config, generation: 1, policy_id: POLICY, client, pool, features: setup.configs });
    const stream = result.stream as string;
    const examples = stream.split("\n").filter((line) => line !== "").map((line) => readTrainingExample(JSON.parse(line)));
    expect(examples.length).toBeGreaterThan(0);
    expect(examples.every((example) => example.candidate_features?.feature_set_hash === hash)).toBe(true);
    expect(examples.some((example) => (example.candidate_features?.rows.length ?? 0) > 0)).toBe(true);
    // Every example's rows are bytes some query carried (queries of the baseline seat are never sent).
    const sent = new Set(client.queries.map((query) => jcs(query.candidate_features as unknown as Json)));
    for (const example of examples) expect(sent.has(jcs(example.candidate_features as unknown as Json))).toBe(true);
    expect(client.queries.every((query) => query.candidate_features !== undefined)).toBe(true);
  }, 600_000); // real probe over a tiny generation, twice (play, then the loader)

  it("without a set nothing is built or sent, and no example has the field", async () => {
    const pool = trainingTeamPool(engine, plain.generation.excluded_abilities);
    const client = standIn(null);
    const result = await playGeneration({ engine, config: plain, generation: 1, policy_id: POLICY, client, pool });
    expect(client.queries.every((query) => query.candidate_features === undefined)).toBe(true);
    expect((result.stream as string).includes("candidate_features")).toBe(false);
  }, 300_000);
});
