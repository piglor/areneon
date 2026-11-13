// The run config names the candidate feature set (ADR-022 sections 3 and 6, slice D3): `featureSetup` checks the named
// hash against this checkout's records, and a generation given the set builds the rows at play time and recomputes
// them into the stream.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { actorRule } from "../src/actor-rule.ts";
import { type Json, jcs } from "../src/canonical-json.ts";
import { loadEngine } from "../src/engine.ts";
import { BELIEF_THREATS_PER_FOE, FEATURE_SET_3, FEATURE_SET_BELIEF, FEATURE_SET_OUTCOME, featureSetHash, featureSetIdentity } from "../src/candidate-features.ts";
import type { BeliefWorldsFactory } from "../src/damage-aware-worlds.ts";
import { configV2, fakePort } from "./damage-aware-world-support.ts";
import { beliefRowCounts, featureSetup, GenerationError, sheetSetup, playGeneration, trainingTeamPool } from "../src/league-generation.ts";
import type { PolicyClient } from "../src/policy-worker.ts";
import type { PolicyQueryInput } from "../src/policy-protocol.ts";
import { createFeatureMemo } from "../src/candidate-features.ts";
import type { BeliefFeaturesFactory } from "../src/belief-features.ts";
import { pairFeatureSetup } from "../src/progress-cli.ts";
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
const hashOf = (set = "probe-damage/1"): string => {
  try {
    featureSetup(engine, withFeatures("0".repeat(64), set));
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

  it("builds probe-damage/3 through the belief features factory of the composition root, and fails closed without one or without move inclusions", () => {
    const calls: string[] = [];
    const inclusions = () => ({ slots: [] });
    const factory: BeliefFeaturesFactory = () => {
      calls.push("factory");
      return { identity_hash: "e".repeat(64), features_hash: "f".repeat(64), prior_hash: "0".repeat(64), belief_config_hash: "1".repeat(64), prior_format_id: "x", k: { item: 4, ability: 3, nature: 3, moves: 8 }, sheet: () => ({}) as never, moveInclusions: inclusions };
    };
    const attempt = (): string => {
      try {
        featureSetup(engine, withFeatures("0".repeat(64), FEATURE_SET_3), undefined, factory);
      } catch (error) {
        return /computes as ([0-9a-f]{64})/.exec((error as Error).message)?.[1] ?? "";
      }
      return "";
    };
    const hash = attempt();
    expect(hash).toMatch(/^[0-9a-f]{64}$/);
    const setup = featureSetup(engine, withFeatures(hash, FEATURE_SET_3), undefined, factory) as NonNullable<ReturnType<typeof featureSetup>>;
    expect(setup.statement).toEqual({ feature_set: FEATURE_SET_3, feature_set_hash: hash });
    expect(setup.configs.belief_threats?.per_foe).toBe(BELIEF_THREATS_PER_FOE);
    expect(setup.configs.belief_threats?.port.identity_hash).toBe("e".repeat(64));
    expect(hash).not.toBe(hashOf());
    expect(() => featureSetup(engine, withFeatures(hash, FEATURE_SET_3))).toThrow(/belief features factory of the composition root/);
    const bare: BeliefFeaturesFactory = (context) => {
      const { moveInclusions: _gone, ...rest } = factory(context);
      return rest;
    };
    expect(() => featureSetup(engine, withFeatures(hash, FEATURE_SET_3), undefined, bare)).toThrow(/no move inclusions/);
    // the pair-level wrapper passes the factory through and makes the refusal a usage error
    expect(pairFeatureSetup(engine, withFeatures(hash, FEATURE_SET_3), undefined, factory)?.statement.feature_set).toBe(FEATURE_SET_3);
  });

  it("builds probe-outcome/1 from the reviewed outcome record and the move_outcome scope, and the statement names its own hash (E-22.5)", () => {
    const attempt = (): string => {
      try {
        featureSetup(engine, withFeatures("0".repeat(64), FEATURE_SET_OUTCOME));
      } catch (error) {
        return /computes as ([0-9a-f]{64})/.exec((error as Error).message)?.[1] ?? "";
      }
      return "";
    };
    const hash = attempt();
    expect(hash).toMatch(/^[0-9a-f]{64}$/);
    const setup = featureSetup(engine, withFeatures(hash, FEATURE_SET_OUTCOME)) as NonNullable<ReturnType<typeof featureSetup>>;
    expect(setup.statement).toEqual({ feature_set: FEATURE_SET_OUTCOME, feature_set_hash: hash });
    expect(setup.configs.outcome?.config.max_branches).toBe(5);
    expect(setup.configs.mechanics.scope.query_kinds.some((entry) => entry.kind === "move_outcome")).toBe(true);
    expect(featureSetHash(featureSetIdentity(engine, setup.configs, FEATURE_SET_OUTCOME))).toBe(hash);
    expect(hash).not.toBe(hashOf());
    expect(hash).not.toBe(hashOf("probe-damage/2"));
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

  it("a belief pair: the rows are sent, and each game row carries what the belief did (fallbacks by cause, sorted)", async () => {
    const refuse = (): never => {
      throw Object.assign(new Error("no world"), { name: "BeliefError", code: "evidence_has_no_support" });
    };
    const factory: BeliefWorldsFactory = () => fakePort(undefined, { sample: refuse });
    let hash = "";
    try {
      featureSetup(engine, withFeatures("0".repeat(64), FEATURE_SET_BELIEF), factory);
    } catch (error) {
      hash = /computes as ([0-9a-f]{64})/.exec((error as Error).message)?.[1] as string;
    }
    const config = withFeatures(hash, FEATURE_SET_BELIEF);
    const setup = featureSetup(engine, config, factory) as NonNullable<ReturnType<typeof featureSetup>>;
    // The evaluation commands set the pair up with the same factory (and refuse without one as a usage error).
    expect(pairFeatureSetup(engine, config, factory)?.statement).toEqual(setup.statement);
    expect(() => pairFeatureSetup(engine, config)).toThrow(/belief port of the composition root/);
    const pool = trainingTeamPool(engine, config.generation.excluded_abilities);
    const result = await playGeneration({ engine, config, generation: 1, policy_id: POLICY, client: standIn(setup.statement), pool, features: setup.configs });
    const withBelief = result.games.filter((game) => game.feature_belief !== undefined);
    expect(withBelief.length).toBe(result.games.length);
    expect(withBelief.some((game) => (game.feature_belief?.fallbacks.evidence_has_no_support ?? 0) > 0)).toBe(true);
    for (const game of withBelief) {
      const keys = Object.keys(game.feature_belief?.fallbacks ?? {});
      expect(keys).toEqual([...keys].sort());
    }
  }, 600_000); // a tiny generation whose belief is refused at every request (the v1 fallback rows)

  it("a pair without the belief set has no feature_belief on any row", async () => {
    const hash = hashOf();
    const config = withFeatures(hash);
    const setup = featureSetup(engine, config) as NonNullable<ReturnType<typeof featureSetup>>;
    const result = await playGeneration({ engine, config, generation: 1, policy_id: POLICY, client: standIn(setup.statement), pool: trainingTeamPool(engine, config.generation.excluded_abilities), features: setup.configs });
    expect(result.games.every((game) => game.feature_belief === undefined)).toBe(true);
  }, 600_000);
});

describe("beliefRowCounts", () => {
  it("is undefined without belief counts and sums the seats with the causes sorted", () => {
    expect(beliefRowCounts([createFeatureMemo()])).toBeUndefined();
    const a = createFeatureMemo();
    a.belief = { decisions: 2, worlds_sampled: 16, redraws: 1, contradicted_worlds: 3, fallbacks: { no_consistent_world: 1 } };
    const b = createFeatureMemo();
    b.belief = { decisions: 1, worlds_sampled: 8, redraws: 0, contradicted_worlds: 0, fallbacks: { evidence_has_no_support: 1, no_consistent_world: 2 } };
    const sum = beliefRowCounts([a, createFeatureMemo(), b]);
    expect(sum).toEqual({ decisions: 3, worlds_sampled: 24, redraws: 1, contradicted_worlds: 3, fallbacks: { evidence_has_no_support: 1, no_consistent_world: 3 } });
    expect(Object.keys(sum?.fallbacks ?? {})).toEqual(["evidence_has_no_support", "no_consistent_world"]);
  });

  describe("the uninformative belief mode (#1309): the named hash selects the mode", () => {
    const sheetNamed = (hash: string): TrainingRunConfig => {
      const raw = structuredClone(tiny) as { model: { encoder: { options?: unknown } } };
      raw.model.encoder.options = { opponent_sheet: { feature_set: "opponent-sheet/1", feature_set_hash: hash } };
      return readTrainingRunConfig(raw);
    };
    const seen: (string | undefined)[] = [];
    const factory: BeliefFeaturesFactory = (context) => {
      seen.push(context.evidence);
      const none = context.evidence === "none";
      return { identity_hash: (none ? "d" : "e").repeat(64), features_hash: (none ? "2" : "1").repeat(64), prior_hash: "0".repeat(64), belief_config_hash: "1".repeat(64), prior_format_id: "x", k: { item: 4, ability: 3, nature: 3, moves: 8 }, sheet: () => ({}) as never, moveInclusions: () => ({ slots: [] }) };
    };

    it("sheetSetup takes the informative port for its hash, the uninformative one for the other, and refuses a third", () => {
      seen.length = 0;
      expect(sheetSetup(engine, sheetNamed("1".repeat(64)), factory)?.statement.feature_set_hash).toBe("1".repeat(64));
      expect(seen).toEqual([undefined]);
      seen.length = 0;
      expect(sheetSetup(engine, sheetNamed("2".repeat(64)), factory)?.statement.feature_set_hash).toBe("2".repeat(64));
      expect(seen).toEqual([undefined, "none"]);
      expect(() => sheetSetup(engine, sheetNamed("3".repeat(64)), factory)).toThrow(/computes as 1{64} \(informative belief\); the other mode computes 2{64}/);
    });

    it("probe-damage/3 gets its own hash from the mode, and the named hash selects the port", () => {
      // The hash this checkout computes for a port with the given identity, read from the refusal of a wrong one.
      const hashOfIdentity = (identity: string): string => {
        const fixed: BeliefFeaturesFactory = (context) => ({ ...factory(context), identity_hash: identity.repeat(64) });
        try {
          featureSetup(engine, withFeatures("0".repeat(64), FEATURE_SET_3), undefined, fixed);
        } catch (error) {
          return /computes as ([0-9a-f]{64})/.exec((error as Error).message)?.[1] ?? "";
        }
        return "";
      };
      const informative = hashOfIdentity("e");
      const uninformative = hashOfIdentity("d");
      expect(informative).not.toBe(uninformative);
      seen.length = 0;
      const first = featureSetup(engine, withFeatures(informative, FEATURE_SET_3), undefined, factory) as NonNullable<ReturnType<typeof featureSetup>>;
      expect(first.configs.belief_threats?.port.identity_hash).toBe("e".repeat(64));
      expect(seen).toEqual([undefined]);
      seen.length = 0;
      const second = featureSetup(engine, withFeatures(uninformative, FEATURE_SET_3), undefined, factory) as NonNullable<ReturnType<typeof featureSetup>>;
      expect(second.configs.belief_threats?.port.identity_hash).toBe("d".repeat(64));
      expect(second.statement.feature_set_hash).toBe(uninformative);
      expect(seen).toEqual([undefined, "none"]);
      // a wrong hash lists every hash tried
      expect(() => featureSetup(engine, withFeatures("0".repeat(64), FEATURE_SET_3), undefined, factory)).toThrow(new RegExp(`computes as ${informative}; the other mode computes ${uninformative}`));
    });
  });
});
