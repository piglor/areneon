// ADR-022 erratum E-22.3.2, #1174: the `probe-damage/2` arrays across the worker boundary and into the training stream.
import { readFileSync } from "node:fs";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import type { PrivateTeamSpec } from "@areneon/schema/types/v1/private-team-spec.ts";
import { afterEach, describe, expect, it } from "vitest";
import { actorRule } from "../src/actor-rule.ts";
import { featureSetHash, featureSetIdentity, type FeatureConfigs } from "../src/candidate-features.ts";
import { jcs, type Json } from "../src/canonical-json.ts";
import { configuredSetHash, featureSetOf } from "../src/feature-build.ts";
import { readFeatureRows, writeFeatureRows } from "../src/feature-recompute.ts";
import { featureCoverageProblem, featureRowsProblem, type WireFeatureRows } from "../src/feature-rows.ts";
import { GOLDEN_FORMAT, GOLDEN_SEED, syntheticFixtureHashes, syntheticTeam } from "../src/golden.ts";
import { episodeFromRecord, loadEngine, loadTrainingSet, readTrainingExample, stampTeam } from "../src/index.ts";
import { policyGame } from "../src/policy-actor.ts";
import { type PolicyQueryInput, readPolicyQuery } from "../src/policy-protocol.ts";
import { type PolicyClient, PolicyWorkerClient } from "../src/policy-worker.ts";
import { SAMPLES_DIR } from "../src/samples.ts";
import { config, mechanics, snapshot, vocabulary } from "./damage-aware-support.ts";
import { CHECKPOINTS, ENCODER_VERSION, ENGINE_DATA_HASH, MAX_LINE_BYTES } from "./policy-fake.ts";

const engine = loadEngine();
type Sample = Record<string, unknown> & { candidate_features: WireFeatureRows & Record<string, unknown[]>; observation: { log: string[] }; candidates: { choice: string }[] };
const sample = (file: string): Sample => JSON.parse(readFileSync(`${SAMPLES_DIR}/v1/${file}.json`, "utf8")) as never;
const dex = engine.sim.Dex.mod(engine.sim.Dex.formats.get(GOLDEN_FORMAT).mod) as unknown as { moves: { get(id: string): { category: string } } };
const isDamaging = (id: string): boolean => dex.moves.get(id).category !== "Status";
const coverage = (query: Sample): string | null => featureCoverageProblem(query as never, query.candidate_features, isDamaging);

describe("the reader rules of the probe-damage/2 arrays", () => {
  it("the committed /2 samples read, and their arrays cover the request", () => {
    const query = sample("policy-query/valid/battle-features-2");
    expect(featureRowsProblem(query.candidate_features)).toBeNull();
    expect(coverage(query)).toBeNull();
    for (const name of ["threats", "speeds", "own_priorities", "foe_priorities"]) expect(query.candidate_features[name]?.length, name).toBeGreaterThan(0);
    // the unsupported reasons the sample shows include the two limits that only the order arrays carry
    const reasons = JSON.stringify([query.candidate_features.speeds, query.candidate_features.own_priorities]);
    expect(reasons).toMatch(/unqueueable/);
    expect(reasons).toMatch(/fractional_priority/);
    const preview = sample("policy-query/valid/preview-features-2");
    expect(featureRowsProblem(preview.candidate_features)).toBeNull();
    expect(coverage(preview)).toBeNull();
    readPolicyQuery({ ...query, query_id: 1 });
    readTrainingExample(sample("training-example/valid/battle-features-2"));
  });

  it("the arrays are present exactly under probe-damage/2", () => {
    const query = sample("policy-query/valid/battle-features-2");
    expect(featureRowsProblem(sample("policy-query/reader-invalid/features-2-array-under-1").candidate_features)).toMatch(/present exactly under probe-damage\/2/);
    expect(featureRowsProblem(sample("policy-query/reader-invalid/features-2-missing-array").candidate_features)).toMatch(/present exactly/);
    for (const name of ["threats", "speeds", "own_priorities", "foe_priorities"]) {
      const { [name]: _gone, ...rest } = query.candidate_features;
      expect(featureRowsProblem(rest as never), name).toMatch(/present exactly/);
    }
    // a /1 value has none and keeps reading
    expect(featureRowsProblem(sample("policy-query/valid/battle-features").candidate_features)).toBeNull();
  });

  it.each([
    ["threats-unsorted", /threats must be sorted and unique/],
    ["speeds-duplicate", /speeds must be sorted and unique/],
    ["priorities-unsorted", /own_priorities must be sorted and unique/],
    ["foe-priorities-unsorted", /foe_priorities must be sorted and unique/],
  ])("%s is refused", (name, message) => {
    expect(featureRowsProblem(sample(`policy-query/reader-invalid/features-2-${name}`).candidate_features)).toMatch(message);
  });

  it.each([
    ["missing-speed-row", /speeds must hold exactly one row/],
    ["missing-own-priority", /own_priorities must hold exactly one row/],
    ["missing-foe-priority", /foe_priorities must hold a row for the foe move of every threat/],
    ["foreign-own", /threats name an own Pokemon/],
  ])("%s passes the structure and is refused against the request", (name, message) => {
    const query = sample(`policy-query/reader-invalid/features-2-${name}`);
    expect(featureRowsProblem(query.candidate_features)).toBeNull();
    expect(coverage(query)).toMatch(message);
  });

  const withLog = (request: object, choices: string[], features: Record<string, unknown>): [Sample, WireFeatureRows] => {
    const base = sample("policy-query/valid/battle-features-2");
    const query = { ...base, observation: { ...base.observation, log: [`|request|${JSON.stringify(request)}`] }, candidates: choices.map((choice) => ({ choice })) } as Sample;
    return [query, { feature_set: "probe-damage/2", feature_set_hash: "0".repeat(64), rows: [], threats: [], speeds: [], own_priorities: [], foe_priorities: [], ...features } as WireFeatureRows];
  };
  const mon = (condition: string) => ({ condition });
  const side = { pokemon: [mon("100/100"), mon("100/100"), mon("100/100"), mon("0 fnt")] };
  const speed = (own: number, mega: boolean, foe: number) => ({ kind: "computed" as const, own, mega, foe, relation: "tie" });
  const prio = (slot: number, move: number, mega: boolean) => ({ kind: "computed" as const, slot, move, mega, priority: 0 });

  it("a Mega variant exists only for a slot where a candidate Mega Evolves; a benched switch-in only where a candidate names it", () => {
    const request = { active: [{ moves: [{ id: "protect" }] }, { moves: [{ id: "protect" }] }], side };
    const speeds = [speed(0, false, 1), speed(1, false, 1), speed(2, false, 1)];
    const ok = withLog(request, ["move 1 1, move 1 1", "switch 3, move 1 1"], { speeds, own_priorities: [prio(0, 1, false), prio(1, 1, false)], threats: [] });
    expect(coverage(Object.assign(ok[0], { candidate_features: ok[1] }) as Sample)).toBeNull();
    // without a candidate that names the bench Pokemon, its speed row is extra
    const extra = withLog(request, ["move 1 1, move 1 1"], { speeds, own_priorities: [prio(0, 1, false), prio(1, 1, false)] });
    expect(coverage(Object.assign(extra[0], { candidate_features: extra[1] }) as Sample)).toMatch(/speeds must hold exactly one row/);
    // a fainted bench Pokemon has no row
    const fainted = withLog(request, ["move 1 1, move 1 1", "switch 4, move 1 1"], { speeds: speeds.slice(0, 2), own_priorities: [prio(0, 1, false), prio(1, 1, false)] });
    expect(coverage(Object.assign(fainted[0], { candidate_features: fainted[1] }) as Sample)).toBeNull();
    // a switch to a Pokemon the request does not list
    const foreign = withLog(request, ["switch 7, move 1 1"], { speeds: [] });
    expect(coverage(Object.assign(foreign[0], { candidate_features: foreign[1] }) as Sample)).toMatch(/do not list|does not list/);
    // the Mega variant: speeds and the own-priority row of the Mega part (a row for a part no candidate has is extra)
    const mega = withLog(request, ["move 1 1 mega, move 1 1"], { speeds: [speed(0, false, 1), speed(0, true, 1), speed(1, false, 1)], own_priorities: [prio(0, 1, false), prio(0, 1, true), prio(1, 1, false)] });
    expect(coverage(Object.assign(mega[0], { candidate_features: mega[1] }) as Sample)).toMatch(/own_priorities must hold exactly one row/);
    const megaOk = withLog(request, ["move 1 1 mega, move 1 1"], { speeds: [speed(0, false, 1), speed(0, true, 1), speed(1, false, 1)], own_priorities: [prio(0, 1, true), prio(1, 1, false)] });
    expect(coverage(Object.assign(megaOk[0], { candidate_features: megaOk[1] }) as Sample)).toBeNull();
    // a Mega variant in speeds with no candidate that Mega Evolves
    const noMega = withLog(request, ["move 1 1, move 1 1"], { speeds: [speed(0, false, 1), speed(0, true, 1), speed(1, false, 1)], own_priorities: [prio(0, 1, false), prio(1, 1, false)] });
    expect(coverage(Object.assign(noMega[0], { candidate_features: noMega[1] }) as Sample)).toMatch(/speeds must hold exactly one row/);
  });

  it("a forced switch has speeds for the switch-ins and no priority row; Team Preview has nothing", () => {
    const forced = { forceSwitch: [true, false], side };
    // the builder emits threats for the switch-ins and no foe priorities on a forced switch: accepted
    const threat = { kind: "computed", own: 2, mega: false, foe: 1, move: "protect", mean_ppm: 1, max_ppm: 1, min_ppm: 1, ohko_rolls: 0 };
    const priced = withLog(forced, ["switch 3, pass"], { speeds: [speed(2, false, 1)], threats: [threat] });
    expect(coverage(Object.assign(priced[0], { candidate_features: priced[1] }) as Sample)).toBeNull();
    const ok = withLog(forced, ["switch 3, pass"], { speeds: [speed(2, false, 1)], threats: [] });
    expect(coverage(Object.assign(ok[0], { candidate_features: ok[1] }) as Sample)).toBeNull();
    const foe = withLog(forced, ["switch 3, pass"], { speeds: [speed(2, false, 1)], foe_priorities: [{ kind: "computed", foe: 1, move: "protect", priority: 4 }] });
    expect(coverage(Object.assign(foe[0], { candidate_features: foe[1] }) as Sample)).toMatch(/foe_priorities are empty/);
    const preview = withLog({ teamPreview: true, side }, ["team 1234"], {});
    expect(coverage(Object.assign(preview[0], { candidate_features: preview[1] }) as Sample)).toBeNull();
    const stray = withLog({ teamPreview: true, side }, ["team 1234"], { speeds: [speed(0, false, 1)] });
    expect(coverage(Object.assign(stray[0], { candidate_features: stray[1] }) as Sample)).toMatch(/speeds must hold/);
    // a foe location shown only by another array still needs its speed rows
    const only = withLog(forced, ["switch 3, pass"], { speeds: [speed(2, false, 1)], threats: [{ kind: "computed", own: 2, mega: false, foe: 2, move: "tackle", mean_ppm: 1, max_ppm: 1, min_ppm: 1, ohko_rolls: 0 }] });
    expect(coverage(Object.assign(only[0], { candidate_features: only[1] }) as Sample)).toMatch(/speeds must hold/);
  });
});

describe("the handshake states probe-damage/2 (ADR-022 erratum E-22.3.2)", () => {
  const FAKE = fileURLToPath(new URL("./fake-policy-worker.ts", import.meta.url));
  const two = { feature_set: "probe-damage/2", feature_set_hash: "a".repeat(64) };
  const one = { feature_set: "probe-damage/1", feature_set_hash: "a".repeat(64) };
  const opened: PolicyWorkerClient[] = [];
  afterEach(async () => {
    await Promise.all(opened.splice(0).map((c) => c.close()));
  });
  const client = (stated: unknown, supply: typeof two | undefined): PolicyWorkerClient => {
    const made = new PolicyWorkerClient({
      command: process.execPath,
      args: [FAKE, "ok"],
      env: { FAKE_FEATURE_SET: JSON.stringify(stated) },
      max_line_bytes: MAX_LINE_BYTES,
      n_parallel: 1,
      handshake_timeout_ms: 10_000,
      expect: { encoder_version: ENCODER_VERSION, engine_data_hash: ENGINE_DATA_HASH, checkpoints: CHECKPOINTS },
      ...(supply === undefined ? {} : { features: supply }),
    });
    opened.push(made);
    return made;
  };

  it("a worker that states /2 to a client that supplies /1 or nothing is refused before any battle", async () => {
    await expect(client(two, one).start()).rejects.toThrow(/cannot supply/);
    await expect(client(two, undefined).start()).rejects.toThrow(/cannot supply/);
  });

  it("a stated /2 that this side supplies is accepted, and its arrays go to that worker only", async () => {
    const worker = client(two, two);
    await worker.start();
    expect(worker.featureSet).toEqual(two);
    const { query_id: _id, ...query } = sample("policy-query/valid/battle-features-2") as never as PolicyQueryInput & { query_id: number };
    const stated = { ...query, candidate_features: { ...(query.candidate_features as object), ...two } } as PolicyQueryInput;
    await expect(worker.ask("g", stated)).resolves.toBeDefined();
    const missing = { ...stated, candidate_features: { ...(stated.candidate_features as object), threats: undefined } } as unknown as PolicyQueryInput;
    await expect(worker.ask("g", JSON.parse(JSON.stringify(missing)) as PolicyQueryInput)).rejects.toThrow(/present exactly under probe-damage\/2/);
  });
});

describe("a played game under probe-damage/2: the arrays sent equal the arrays the loader recomputes", () => {
  const known = syntheticFixtureHashes(engine);
  const sets = syntheticTeam();
  const team = stampTeam(sets as PrivateTeamSpec["team"], known);
  const configs = (change: Partial<FeatureConfigs> = {}): FeatureConfigs => ({ vocabulary, config, mechanics, snapshot, data_class: "synthetic", feature_set: "probe-damage/2", ...change });
  const identity = featureSetIdentity(engine, configs(), "probe-damage/2");
  const stated = { feature_set: identity.feature_set, feature_set_hash: featureSetHash(identity) };
  const POLICY = `policy:${"a".repeat(64)}`;
  const budget = { preview_ms: 120_000, battle_ms: 120_000 };

  /** A stand-in worker that states the set, re-reads every query as the real client does, and plays the actor rule. */
  function standIn(): PolicyClient & { queries: PolicyQueryInput[] } {
    const queries: PolicyQueryInput[] = [];
    return {
      queries,
      featureSet: stated,
      ask(_battle, query) {
        readPolicyQuery({ ...query, query_id: queries.length + 1 });
        queries.push(query);
        const open = query.candidates.filter((c) => !c.pending).map((c) => c.choice);
        const pending = query.candidates.filter((c) => c.pending).map((c) => c.choice);
        return Promise.resolve({ schema_version: 1, identity: query.identity, choice: actorRule({ choices: open, pending }), source_policy_id: query.policy_id, elapsed_ms: 0, deadline_outcome: "in_budget" });
      },
    };
  }

  it("the set is the configs' own; its hash differs from /1's and is the one the configs name", () => {
    expect(featureSetOf(configs())).toBe("probe-damage/2");
    expect(featureSetOf({})).toBe("probe-damage/1");
    expect(configuredSetHash(engine, configs())).toBe(stated.feature_set_hash);
    expect(configuredSetHash(engine, configs({ feature_set: "probe-damage/1" }))).not.toBe(stated.feature_set_hash);
  });

  it("arrays in each PolicyQuery equal, byte for byte, those loadTrainingSet recomputes from the written Episodes; the examples are valid and cover their candidates", async () => {
    const a = standIn();
    const b = standIn();
    const seat = (client: PolicyClient) => ({ client, policy_id: POLICY, sampling: "argmax" as const, features: { configs: configs() } });
    const { outcome, record } = await policyGame({ engine, formatId: GOLDEN_FORMAT, seed: GOLDEN_SEED, game: 0, teams: { p1: sets, p2: sets }, known_synthetic: known, seats: { p1: seat(a), p2: seat(b) }, latency_budget: budget });
    expect(outcome.kind).toBe("played");
    const played = record as NonNullable<typeof record>;
    const episodes = await Promise.all(
      (["p1", "p2"] as const).map((player) =>
        episodeFromRecord(played, player, { team, opponent_team: team, known_synthetic: known, behaviour_policy_id: POLICY, collection: { kind: "league", run_id: "league-1", seed: played.seed, opponent_policy_id: POLICY } }),
      ),
    );
    const { examples } = await loadTrainingSet({ episodes, evaluation_manifests: [], known_synthetic: known, behaviour_policy_id: POLICY, features: { engine, configs: configs() } });
    const sent = [...a.queries, ...b.queries];
    expect(sent.length).toBe(examples.length);
    const seen = { threats: 0, speeds: 0, own_priorities: 0, foe_priorities: 0 };
    for (const example of examples) {
      const query = sent.find((q) => q.observation.player === example.observation.player && q.identity.request_seq === example.observation.identity.request_seq);
      expect(query, "a query for every example").toBeDefined();
      const features = (query as PolicyQueryInput).candidate_features as unknown as WireFeatureRows & Record<string, unknown[]>;
      expect(features.feature_set).toBe("probe-damage/2");
      // the whole object, rows and the four arrays: threats completeness against the revealed moves is held here
      expect(jcs(example.candidate_features as unknown as Json)).toBe(jcs(features as unknown as Json));
      for (const name of Object.keys(seen)) seen[name as keyof typeof seen] += features[name]?.length ?? 0;
      readTrainingExample(example);
      expect(featureCoverageProblem({ observation: example.observation, candidates: example.candidates as never }, example.candidate_features as WireFeatureRows, isDamaging)).toBeNull();
    }
    for (const [name, count] of Object.entries(seen)) expect(count, name).toBeGreaterThan(0);

    // The game processes' recomputation (feature-recompute.ts) writes and reads the same /2 arrays, and a /1 reader refuses them.
    const dir = mkdtempSync(join(tmpdir(), "areneon-rows2-"));
    try {
      const { writeFileSync } = await import("node:fs");
      const { episodeHash } = await import("../src/episode.ts");
      const { nodeSha256 } = await import("../src/player-channel.ts");
      const one = episodes[0] as (typeof episodes)[number];
      const hash = await episodeHash(one, nodeSha256);
      const path = join(dir, `${hash}.episode.json`);
      writeFileSync(path, JSON.stringify(one));
      expect(await writeFeatureRows({ engine, configs: configs(), known_synthetic: known }, [{ hash, path }], join(dir, "rows"))).toBe(1);
      const rows = readFeatureRows(engine, configs(), join(dir, "rows"))(hash);
      expect(rows?.length).toBe(one.decisions.length);
      const own = examples.filter((e) => e.episode_hash === hash).map((e) => e.candidate_features);
      expect(jcs(rows as unknown as Json)).toBe(jcs(own as unknown as Json));
      expect(readFeatureRows(engine, configs({ feature_set: "probe-damage/1" }), join(dir, "rows"))(hash)).toBeUndefined();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }, 600_000); // real probe over one seeded game: /2 asks about 25 queries a decision, at play and again in the loader
});
