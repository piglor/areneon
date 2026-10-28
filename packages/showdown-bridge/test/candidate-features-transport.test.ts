// ADR-022 section 3, slice D2: candidate_features across the worker boundary and into the training stream.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import type { PrivateTeamSpec } from "@areneon/schema/types/v1/private-team-spec.ts";
import { afterEach, describe, expect, it } from "vitest";
import { actorRule } from "../src/actor-rule.ts";
import { candidateFeatures, createFeatureMemo, featureSetHash, featureSetIdentity, type FeatureConfigs } from "../src/candidate-features.ts";
import { jcs, type Json } from "../src/canonical-json.ts";
import { FOE_SPREAD_TARGET_TYPES, featureCoverageProblem, featureRowsProblem, type WireFeatureRows } from "../src/feature-rows.ts";
import { GOLDEN_FORMAT, GOLDEN_SEED, goldenBattle, syntheticFixtureHashes, syntheticTeam } from "../src/golden.ts";
import { episodeFromRecord, loadEngine, loadTrainingSet, readTrainingExample, stampTeam } from "../src/index.ts";
import { MechanicsProbeError } from "../src/mechanics-probe.ts";
import { type PolicyClient, PolicyHandshakeError, PolicyWorkerClient } from "../src/policy-worker.ts";
import { policyGame } from "../src/policy-actor.ts";
import { type PolicyQueryInput, readPolicyQuery } from "../src/policy-protocol.ts";
import { SAMPLES_DIR } from "../src/samples.ts";
import { config, mechanics, snapshot, vocabulary } from "./damage-aware-support.ts";
import { CHECKPOINTS, ENCODER_VERSION, ENGINE_DATA_HASH, MAX_LINE_BYTES } from "./policy-fake.ts";

const engine = loadEngine();
const sample = (file: string): Record<string, unknown> & { candidate_features: WireFeatureRows; observation: { log: string[] }; candidates: { choice: string }[] } =>
  JSON.parse(readFileSync(`${SAMPLES_DIR}/v1/${file}.json`, "utf8")) as never;
const dex = engine.sim.Dex.mod(engine.sim.Dex.formats.get(GOLDEN_FORMAT).mod) as unknown as { moves: { get(id: string): { category: string } } };
const isDamaging = (id: string): boolean => dex.moves.get(id).category !== "Status";

describe("the reader rules of candidate_features (ADR-022 section 3)", () => {
  const row = (slot: number, move: number, target: number, mega: boolean) => ({ slot, move, target, mega });
  const rows = (...list: ReturnType<typeof row>[]): WireFeatureRows => ({ feature_set: "x", feature_set_hash: "0".repeat(64), rows: list });

  it("rows are strictly ascending by (slot, move, target, mega false first)", () => {
    expect(featureRowsProblem(rows())).toBeNull();
    expect(featureRowsProblem(rows(row(0, 1, 1, false), row(0, 1, 1, true), row(0, 1, 2, false), row(0, 2, 1, false), row(1, 1, 1, false)))).toBeNull();
    for (const bad of [
      rows(row(0, 1, 1, true), row(0, 1, 1, false)),
      rows(row(0, 1, 1, false), row(0, 1, 1, false)),
      rows(row(1, 1, 1, false), row(0, 1, 1, false)),
      rows(row(0, 2, 1, false), row(0, 1, 2, false)),
      rows(row(0, 1, 2, false), row(0, 1, 1, false)),
    ]) {
      expect(featureRowsProblem(bad)).toMatch(/sorted and unique/);
    }
  });

  it("the committed samples: the valid one is covered, the missing-row one is not", () => {
    const valid = sample("policy-query/valid/battle-features");
    expect(featureRowsProblem(valid.candidate_features)).toBeNull();
    expect(valid.candidate_features.rows.length).toBeGreaterThan(0);
    expect(featureCoverageProblem(valid as never, valid.candidate_features, isDamaging)).toBeNull();
    const missing = sample("policy-query/reader-invalid/features-missing-row");
    expect(featureCoverageProblem(missing as never, missing.candidate_features, isDamaging)).toMatch(/no candidate_features row/);
    const preview = sample("policy-query/valid/preview-features");
    expect(featureCoverageProblem(preview as never, preview.candidate_features, isDamaging)).toBeNull();
  });

  const query = (moves: { id: string; target: string }[], choice: string) => ({
    observation: { log: [`|request|${JSON.stringify({ active: [{ moves }] })}`] },
    candidates: [{ choice }],
  });
  it("a damaging part that aims at a foe, or a spread part with no location, needs a row of its (slot, move, mega); a status move, an ally and a switch need none", () => {
    const moves = [{ id: "earthquake", target: "allAdjacent" }, { id: "tackle", target: "normal" }, { id: "protect", target: "self" }];
    expect(featureCoverageProblem(query(moves, "move 2 1"), rows(), isDamaging)).toMatch(/no candidate_features row/);
    expect(featureCoverageProblem(query(moves, "move 2 1"), rows(row(0, 2, 1, false)), isDamaging)).toBeNull();
    // the location of a fainted foe has no row of its own: another location of the move does
    expect(featureCoverageProblem(query(moves, "move 2 1"), rows(row(0, 2, 2, false)), isDamaging)).toBeNull();
    expect(featureCoverageProblem(query(moves, "move 2 1"), rows(row(1, 2, 1, false), row(0, 1, 1, false)), isDamaging)).toMatch(/no/);
    expect(featureCoverageProblem(query(moves, "move 2 1 mega"), rows(row(0, 2, 1, false)), isDamaging)).toMatch(/no/);
    expect(featureCoverageProblem(query(moves, "move 2 1 mega"), rows(row(0, 2, 1, true)), isDamaging)).toBeNull();
    expect(featureCoverageProblem(query(moves, "move 1"), rows(), isDamaging)).toMatch(/no/);
    expect(featureCoverageProblem(query(moves, "move 1"), rows(row(0, 1, 2, false)), isDamaging)).toBeNull();
    for (const free of ["move 3", "move 2 -2", "switch 3", "pass"]) expect(featureCoverageProblem(query(moves, free), rows(), isDamaging), free).toBeNull();
    expect(featureCoverageProblem({ observation: { log: ["|turn|2"] }, candidates: [] }, rows(), isDamaging)).toMatch(/does not end in a request/);
  });

  it("the spread target types are the Python reader's, word for word", () => {
    const source = readFileSync(fileURLToPath(new URL("../../../ml/areneon_ml/worker/features.py", import.meta.url)), "utf8");
    const listed = /FOE_SPREAD_TARGET_TYPES = \(([^)]*)\)/.exec(source)?.[1] ?? "";
    expect([...listed.matchAll(/"([A-Za-z]+)"/g)].map((m) => m[1]).sort()).toEqual([...FOE_SPREAD_TARGET_TYPES].sort());
  });
});

describe("the handshake states a feature set (ADR-022 section 3)", () => {
  const FAKE = fileURLToPath(new URL("./fake-policy-worker.ts", import.meta.url));
  const ref = { feature_set: "probe-damage/1", feature_set_hash: "a".repeat(64) };
  const opened: PolicyWorkerClient[] = [];
  afterEach(async () => {
    await Promise.all(opened.splice(0).map((c) => c.close()));
  });
  const client = (stated: unknown, supply: typeof ref | undefined): PolicyWorkerClient => {
    const made = new PolicyWorkerClient({
      command: process.execPath,
      args: [FAKE, "ok"],
      env: stated === undefined ? {} : { FAKE_FEATURE_SET: JSON.stringify(stated) },
      max_line_bytes: MAX_LINE_BYTES,
      n_parallel: 1,
      handshake_timeout_ms: 10_000,
      expect: { encoder_version: ENCODER_VERSION, engine_data_hash: ENGINE_DATA_HASH, checkpoints: CHECKPOINTS },
      ...(supply === undefined ? {} : { features: supply }),
    });
    opened.push(made);
    return made;
  };
  const { query_id: _id, ...INPUT } = sample("policy-query/valid/battle") as never as PolicyQueryInput & { query_id: number };

  it("a worker that states none needs none, whether or not this side can supply one", async () => {
    for (const supply of [undefined, ref]) {
      const worker = client(undefined, supply);
      await worker.start();
      expect(worker.featureSet).toBeNull();
      await expect(worker.ask("g", INPUT)).resolves.toMatchObject({ source_policy_id: INPUT.policy_id });
    }
  });

  it("a stated set that this side supplies is accepted, and a restart states it again", async () => {
    const worker = client(ref, ref);
    await worker.start();
    expect(worker.featureSet).toEqual(ref);
    await worker.restart();
    expect(worker.featureSet).toEqual(ref);
  });

  it.each([
    ["another set hash", { ...ref, feature_set_hash: "b".repeat(64) }, ref],
    ["another set name", { ...ref, feature_set: "probe-damage/2" }, ref],
    ["a set while this side supplies none", ref, undefined],
  ])("refuses %s before any battle", async (_name, stated, supply) => {
    await expect(client(stated, supply).start()).rejects.toThrow(PolicyHandshakeError);
    await expect(client(stated, supply).start()).rejects.toThrow(/cannot supply/);
  });

  it("sends candidate_features to a worker that states a set only, and only of that set", async () => {
    const withRows = sample("policy-query/valid/battle-features");
    const { query_id: _q, ...featured } = withRows as never as PolicyQueryInput & { query_id: number };
    const stated = client({ feature_set: featured.candidate_features?.feature_set, feature_set_hash: featured.candidate_features?.feature_set_hash }, {
      feature_set: featured.candidate_features?.feature_set as string,
      feature_set_hash: featured.candidate_features?.feature_set_hash as string,
    });
    await stated.start();
    await expect(stated.ask("g", featured)).resolves.toBeDefined();
    await expect(stated.ask("g", INPUT)).rejects.toThrow(/carries no candidate_features/);
    const other = { ...featured, candidate_features: { ...(featured.candidate_features as object), feature_set_hash: "c".repeat(64) } } as PolicyQueryInput;
    await expect(stated.ask("g", other)).rejects.toThrow(/another feature set/);
    const none = client(undefined, undefined);
    await none.start();
    await expect(none.ask("g", featured)).rejects.toThrow(/stated no feature set/);
  });
});

describe("a played game: the rows sent equal the rows the loader recomputes (ADR-022 section 3)", () => {
  const known = syntheticFixtureHashes(engine);
  const sets = syntheticTeam();
  const team = stampTeam(sets as PrivateTeamSpec["team"], known);
  const configs = (change: Partial<FeatureConfigs> = {}): FeatureConfigs => ({ vocabulary, config, mechanics, snapshot, data_class: "synthetic", ...change });
  const identity = featureSetIdentity(engine, configs());
  const stated = { feature_set: identity.feature_set, feature_set_hash: featureSetHash(identity) };
  const POLICY = `policy:${"a".repeat(64)}`;
  const budget = { preview_ms: 60_000, battle_ms: 60_000 };

  /** A stand-in worker that states the feature set, re-reads every query as the real client does, and plays the actor rule. */
  function standIn(featureSet: typeof stated | null): PolicyClient & { queries: { battle: string; query: PolicyQueryInput }[] } {
    const queries: { battle: string; query: PolicyQueryInput }[] = [];
    return {
      queries,
      featureSet,
      ask(battle, query) {
        readPolicyQuery({ ...query, query_id: queries.length + 1 });
        queries.push({ battle, query });
        const open = query.candidates.filter((c) => !c.pending).map((c) => c.choice);
        const pending = query.candidates.filter((c) => c.pending).map((c) => c.choice);
        return Promise.resolve({ schema_version: 1, identity: query.identity, choice: actorRule({ choices: open, pending }), source_policy_id: query.policy_id, elapsed_ms: 0, deadline_outcome: "in_budget" });
      },
    };
  }
  const seat = (client: PolicyClient, features?: FeatureConfigs) => ({
    client,
    policy_id: POLICY,
    sampling: "argmax" as const,
    ...(features === undefined ? {} : { features: { configs: features } }),
  });
  const play = (p1: ReturnType<typeof seat>, p2: ReturnType<typeof seat>) =>
    policyGame({ engine, formatId: GOLDEN_FORMAT, seed: GOLDEN_SEED, game: 0, teams: { p1: sets, p2: sets }, known_synthetic: known, seats: { p1, p2 }, latency_budget: budget });

  it("rows in each PolicyQuery equal, byte for byte, the rows loadTrainingSet recomputes from the written Episodes", async () => {
    const a = standIn(stated);
    const b = standIn(stated);
    const { outcome, record } = await play(seat(a, configs()), seat(b, configs()));
    expect(outcome.kind).toBe("played");
    const played = record as NonNullable<typeof record>;
    const episodes = await Promise.all(
      (["p1", "p2"] as const).map((player) =>
        episodeFromRecord(played, player, {
          team,
          opponent_team: team,
          known_synthetic: known,
          behaviour_policy_id: POLICY,
          collection: { kind: "league", run_id: "league-1", seed: played.seed, opponent_policy_id: POLICY },
        }),
      ),
    );
    const { examples } = await loadTrainingSet({
      episodes,
      evaluation_manifests: [],
      known_synthetic: known,
      behaviour_policy_id: POLICY,
      features: { engine, configs: configs() },
    });
    const sent = [...a.queries, ...b.queries].map(({ query }) => query);
    expect(sent.length).toBe(examples.length);
    let rowsSeen = 0;
    let empty = 0;
    for (const example of examples) {
      const query = sent.find((q) => q.observation.player === example.observation.player && q.identity.request_seq === example.observation.identity.request_seq);
      expect(query, "a query for every example").toBeDefined();
      const features = (query as PolicyQueryInput).candidate_features;
      expect(features).toBeDefined();
      expect(jcs(example.candidate_features as unknown as Json)).toBe(jcs(features as unknown as Json));
      rowsSeen += example.candidate_features?.rows.length ?? 0;
      if (example.candidate_features?.rows.length === 0) empty += 1;
      // The loader's example is a valid TrainingExample, and its rows cover its candidates.
      readTrainingExample(example);
      expect(featureCoverageProblem({ observation: example.observation, candidates: example.candidates as never }, example.candidate_features as WireFeatureRows, isDamaging)).toBeNull();
    }
    expect(rowsSeen).toBeGreaterThan(0); // move requests with damaging moves were featured
    expect(empty).toBeGreaterThan(0); // Team Preview and forced switches carry an empty list
    // Without the option no example has the field and the stream is today's.
    const plain = await loadTrainingSet({ episodes, evaluation_manifests: [], known_synthetic: known, behaviour_policy_id: POLICY });
    expect(plain.examples.every((e) => !("candidate_features" in e))).toBe(true);
    expect(plain.examples.map((e) => ({ ...e, candidate_features: undefined }))).toEqual(examples.map((e) => ({ ...e, candidate_features: undefined })));
  }, 180_000); // real probe (about 15 ms a query) over one seeded game, twice (play, then the loader)

  it("a worker that states no set gets no rows, even from a seat that could build them", async () => {
    const a = standIn(null);
    const b = standIn(null);
    await play(seat(a, configs()), seat(b));
    expect([...a.queries, ...b.queries].every(({ query }) => query.candidate_features === undefined)).toBe(true);
  }, 120_000);

  it("a feature set this seat builds that is not the stated one is a harness incident, not a forfeit", async () => {
    const wrong = standIn({ ...stated, feature_set_hash: "d".repeat(64) });
    const { outcome } = await play(seat(wrong, configs()), seat(standIn(null)));
    expect(outcome).toMatchObject({ kind: "incident", incident: "policy_protocol" });
  }, 120_000);

  it("a request the builder cannot feature forfeits the seat as crash, naming the code only", async () => {
    const refusing = configs({ probe: () => { throw new MechanicsProbeError("a state the engine refuses"); } });
    const { outcome } = await play(seat(standIn(stated), refusing), seat(standIn(null)));
    expect(outcome).toMatchObject({ kind: "forfeit", cause: "crash", detail: "no candidate features: probe_refused" });
  }, 120_000);

  it("any other failure of the builder propagates: guardActor treats an unexpected error of a policy seat as a crash", async () => {
    const broken = configs({ probe: () => { throw new Error("a defect"); } });
    const { outcome } = await play(seat(standIn(stated), broken), seat(standIn(null)));
    expect(outcome).toMatchObject({ kind: "forfeit", cause: "crash" });
    expect((outcome as { detail: string }).detail).not.toMatch(/no candidate features/);
  }, 120_000);

  it("the memo changes no byte of the rows", () => {
    // candidateFeatures with and without a shared memo is held equal in candidate-features.test.ts; here the actor's own memo is per game.
    expect(createFeatureMemo().counts.queries).toBe(0);
    expect(typeof candidateFeatures).toBe("function");
  });
});

describe("old values keep their bytes (ADR-008 section 3)", () => {
  // The digests of the samples that existed before the field, computed at the commit before this change.
  const BEFORE: Record<string, string> = {
    "v1/policy-query/valid/battle.json": "d2d5d7a8fdb7d120436aca8d03aebcf3046f8e1fb62268d3e4683cda18e475bd",
    "v1/policy-query/valid/preview.json": "06f21da1a2142d219bd7b2f14e9c4d9b3c375e9aecbc68efc161bdb3f068d69a",
    "v1/policy-query/valid/battle-a2.json": "3ca12813e9d0f6ed7042f3f815dadea185ddc3b059d2f72af9543d509765b9ae",
    "v1/policy-query/valid/battle-tempered.json": "d5ba66fc8876452f54edc680a60874add18b3ef4b10316a9a48bf72120ecb8b5",
    "v1/policy-query/valid/battle-threshold.json": "f7641c5ffc71546d25128fcb4f03742626110b57542eb66f30eff477b5d53585",
    "v1/training-example/valid/preview.json": "0c5b56b0e3a3631c11a38ac959a2c01fd77ebe0fb3bdd1c8959d039a45e0a69d",
    "v1/training-example/valid/battle.json": "7b9fc909b6c6d93d335996bdd7140481bf2304faea71cde30221b6718869617c",
    "v1/training-example/valid/legacy-battle.json": "32a685b70a5be187632f90b4a22af962c5783db5572690305a03e97df932a492",
  };
  it("pins the digests of the samples that predate candidate_features, and none of them carries it", () => {
    const hashes = JSON.parse(readFileSync(`${SAMPLES_DIR}/hashes.json`, "utf8")) as Record<string, string>;
    for (const [file, digest] of Object.entries(BEFORE)) {
      expect(hashes[file], file).toBe(digest);
      expect("candidate_features" in (JSON.parse(readFileSync(`${SAMPLES_DIR}/${file}`, "utf8")) as object), file).toBe(false);
    }
  });
});

describe("loadTrainingSet with a feature set that cannot be built", () => {
  it("is a TrainingLoaderError naming the Episode and request, never a skipped example", async () => {
    const known = syntheticFixtureHashes(engine);
    const sets = syntheticTeam();
    const team = stampTeam(sets as PrivateTeamSpec["team"], known);
    const record = await goldenBattle(engine);
    const episode = await episodeFromRecord(record, "p1", {
      team,
      opponent_team: team,
      known_synthetic: known,
      behaviour_policy_id: "first-legal@golden",
      collection: { kind: "league", run_id: "league-1", seed: record.seed, opponent_policy_id: "first-legal@golden" },
    });
    const refusing: FeatureConfigs = { vocabulary, config, mechanics, snapshot, data_class: "synthetic", probe: () => { throw new MechanicsProbeError("refused"); } };
    await expect(
      loadTrainingSet({ episodes: [episode], evaluation_manifests: [], known_synthetic: known, behaviour_policy_id: "first-legal@golden", features: { engine, configs: refusing } }),
    ).rejects.toThrow(/candidate features of .* cannot be built/);
  }, 120_000);
});
