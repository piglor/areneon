// ADR-022 erratum E-22.5.3, #1342: `probe-outcome/1` on the wire. The `outcomes` array and the five `/2` arrays across the worker
// boundary, into the training stream and into the play-time sidecar; the reader rules of both.
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import type { PrivateTeamSpec } from "@areneon/schema/types/v1/private-team-spec.ts";
import { afterEach, describe, expect, it } from "vitest";
import { actorRule } from "../src/actor-rule.ts";
import { featureSetHash, featureSetIdentity, type FeatureConfigs, FEATURE_SET_OUTCOME } from "../src/candidate-features.ts";
import { jcs, type Json } from "../src/canonical-json.ts";
import { configuredSetHash, featureSetOf, isFeatureSetId, type AnyCandidateFeatures } from "../src/feature-build.ts";
import { readFeatureRows, writeFeatureRows, writePlayRows } from "../src/feature-recompute.ts";
import { featureCoverageProblem, featureRowsProblem, MAX_OUTCOME_BRANCHES, type WireFeatureRows } from "../src/feature-rows.ts";
import { GOLDEN_FORMAT, GOLDEN_SEED, syntheticFixtureHashes, syntheticTeam } from "../src/golden.ts";
import { episodeFromRecord, loadEngine, loadTrainingSet, readTrainingExample, stampTeam } from "../src/index.ts";
import { decisionObservations, readEpisode } from "../src/episode-io.ts";
import { episodeHash } from "../src/episode.ts";
import { withMoveOutcome } from "../src/mechanics-config.ts";
import { loadOutcomeConfig, MAX_BRANCHES_LIMIT } from "../src/outcome-config.ts";
import { nodeSha256 } from "../src/player-channel.ts";
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
const problem = (name: string): string | null => featureRowsProblem(sample(name).candidate_features);
type Row = Record<string, unknown>;
const withRows = (outcomes: Row[]): WireFeatureRows => ({ ...sample("policy-query/valid/battle-features-outcome").candidate_features, outcomes: outcomes as never });

describe("the reader rules of probe-outcome/1", () => {
  it("the committed samples read, and the outcomes cover the request", () => {
    const query = sample("policy-query/valid/battle-features-outcome");
    expect(isFeatureSetId("probe-outcome/1")).toBe(true);
    expect(featureRowsProblem(query.candidate_features)).toBeNull();
    expect(coverage(query)).toBeNull();
    for (const name of ["threats", "speeds", "own_priorities", "foe_priorities", "outcomes"]) expect(query.candidate_features[name]?.length, name).toBeGreaterThan(0);
    // the sample shows every branch kind and both new kinds of value
    const outcomes = query.candidate_features.outcomes as unknown as Row[];
    expect(new Set(outcomes.map((r) => r.precondition))).toEqual(new Set([undefined, "passes", "chance", "fails"]));
    expect(outcomes.some((r) => r.assumed === true)).toBe(true);
    expect(outcomes.some((r) => r.reason === "history_unknown")).toBe(true);
    const preview = sample("policy-query/valid/preview-features-outcome");
    expect(featureRowsProblem(preview.candidate_features)).toBeNull();
    expect(coverage(preview)).toBeNull();
    readPolicyQuery({ ...query, query_id: 1 });
    readTrainingExample(sample("training-example/valid/battle-features-outcome"));
  });

  it("the readers' branch ceiling is the ceiling a record may allow", () => {
    expect(MAX_OUTCOME_BRANCHES).toBe(MAX_BRANCHES_LIMIT);
  });

  it("the history_unknown reason is this set's alone, on any array", () => {
    const base = sample("policy-query/valid/battle-features-2").candidate_features;
    const unknown = { kind: "unsupported", reason: "history_unknown" };
    expect(featureRowsProblem({ ...base, threats: [{ ...unknown, own: 0, mega: false, foe: 1, move: "protect" }] })).toMatch(/history_unknown is a probe-outcome\/1 value/);
    expect(featureRowsProblem({ ...base, rows: [{ ...unknown, slot: 0, move: 1, target: 1, mega: false }] })).toMatch(/history_unknown is a probe-outcome\/1 value/);
    expect(featureRowsProblem({ ...base, speeds: [{ ...unknown, own: 0, mega: false, foe: 1 }] })).toMatch(/history_unknown/);
    const mine = sample("policy-query/valid/battle-features-outcome").candidate_features;
    expect(featureRowsProblem({ ...mine, threats: [{ ...unknown, own: 0, mega: false, foe: 1, move: "protect" }] })).toBeNull();
    expect(problem("policy-query/reader-invalid/features-outcome-history-unknown-under-2")).toMatch(/history_unknown is a probe-outcome\/1 value/);
  });

  it("the arrays are present exactly under probe-outcome/1, and outcomes under no other set", () => {
    expect(problem("policy-query/reader-invalid/features-outcome-under-2")).toMatch(/present exactly under probe-damage\/2/);
    expect(problem("policy-query/reader-invalid/features-outcome-missing-array")).toMatch(/with outcomes under probe-outcome\/1/);
    expect(problem("policy-query/reader-invalid/features-outcome-missing-speeds")).toMatch(/present exactly/);
    expect(featureRowsProblem(sample("policy-query/valid/battle-features-3").candidate_features)).toBeNull();
  });

  it.each([
    ["unsorted", /outcomes must be sorted and unique/],
    ["duplicate", /outcomes must be sorted and unique/],
    ["missing-switch-branch", /each own move part starts with its switch branch/],
    ["missing-move-branch", /a switch branch and one to 4 move branches/],
    ["assumed-with-revealed", /an assumed branch is the only move branch/],
    ["switch-with-foe-move", /a switch branch has no foe move/],
    ["success-mismatch", /success_ppm must be floor/],
    ["precondition-mismatch", /the precondition does not fit its ppm/],
  ])("%s is refused by the structure rules", (name, message) => {
    expect(problem(`policy-query/reader-invalid/features-outcome-${name}`)).toMatch(message);
  });

  it("the value rules: passes is 10^6, fails is 0, a chance is in between; a move branch names its foe move; at most four move branches", () => {
    const rows = sample("policy-query/valid/battle-features-outcome").candidate_features.outcomes as unknown as Row[];
    const at = (index: number, change: Row): Row[] => rows.map((r, i) => (i === index ? { ...r, ...change } : r));
    // row 1 is part 0's `protect` branch (a chance or a fail); row 2 its `thunderbolt` (passes, 800000)
    expect(featureRowsProblem(withRows(at(2, { precondition_ppm: 500000, success_ppm: 400000 })))).toMatch(/the precondition does not fit its ppm/);
    expect(featureRowsProblem(withRows(at(2, { precondition: "fails" })))).toMatch(/the precondition does not fit its ppm/);
    expect(featureRowsProblem(withRows(at(2, { precondition: "chance", precondition_ppm: 0, success_ppm: 0 })))).toMatch(/the precondition does not fit its ppm/);
    expect(featureRowsProblem(withRows(at(1, { foe_move: null })))).toMatch(/a move branch names its foe move/);
    expect(featureRowsProblem(withRows(at(2, { assumed: true })))).toMatch(/an assumed branch is the only move branch/);
    const extra = ["uturn", "xscissor", "zenheadbutt"].map((foe_move) => ({ ...(rows[2] as Row), foe_move }));
    expect(featureRowsProblem(withRows([...rows.slice(0, 3), ...extra, ...rows.slice(3)]))).toMatch(/one to 4 move branches/);
    expect(featureRowsProblem(withRows([...rows.slice(0, 3), ...extra.slice(0, 2), ...rows.slice(3)]))).toBeNull();
    // a unsupported row has no values to check, and a part that starts with a move branch has no switch branch
    expect(featureRowsProblem(withRows(rows.slice(1)))).toMatch(/starts with its switch branch/);
  });

  it.each([
    ["missing-part", /outcomes must hold the rows of exactly the own move parts/],
    ["foreign-part", /outcomes must hold the rows of exactly the own move parts/],
  ])("%s passes the structure and is refused against the request", (name, message) => {
    const query = sample(`policy-query/reader-invalid/features-outcome-${name}`);
    expect(featureRowsProblem(query.candidate_features)).toBeNull();
    expect(coverage(query)).toMatch(message);
  });

  const withLog = (request: object, choices: string[], features: Record<string, unknown>): Sample => {
    const base = sample("policy-query/valid/battle-features-outcome");
    const query = { ...base, observation: { ...base.observation, log: [`|request|${JSON.stringify(request)}`] }, candidates: choices.map((choice) => ({ choice })) } as Sample;
    query.candidate_features = { feature_set: FEATURE_SET_OUTCOME, feature_set_hash: "0".repeat(64), rows: [], threats: [], speeds: [], own_priorities: [], foe_priorities: [], outcomes: [], ...features } as never;
    return query;
  };
  const mon = (condition: string) => ({ condition });
  const side = { pokemon: [mon("100/100"), mon("100/100"), mon("100/100")] };
  const speed = (own: number, foe: number) => ({ kind: "computed", own, mega: false, foe, relation: "tie" });
  const prio = (slot: number, move: number, mega = false) => ({ kind: "computed", slot, move, mega, priority: 0 });
  const part = (slot: number, move: number, target: number, mega = false): Row[] => {
    const key = { slot, move, target, mega };
    const result = { precondition: "passes", precondition_ppm: 1_000_000, hit_ppm: 1_000_000, success_ppm: 1_000_000 };
    return [
      { kind: "computed", ...key, action: "switch", foe_move: null, assumed: false, ...result },
      { kind: "unsupported", ...key, action: "move", foe_move: "protect", assumed: true, reason: "history_unknown" },
    ];
  };
  const request = { active: [{ moves: [{ id: "protect" }] }, { moves: [{ id: "protect" }] }], side };

  it("the outcomes are one block per own move part of the candidates: untargeted and ally-targeted parts are about the lowest foe; a fainted foe's location has none", () => {
    const base = { speeds: [speed(0, 1), speed(1, 1)], own_priorities: [prio(0, 1), prio(1, 1)] };
    // an untargeted part, and a part with an ally target, each have a block
    const ok = withLog(request, ["move 1, move 1 -1"], { ...base, outcomes: [...part(0, 1, 0), ...part(1, 1, -1)] });
    expect(coverage(ok)).toBeNull();
    // a missing block, and a block for a part no candidate has
    expect(coverage(withLog(request, ["move 1, move 1 -1"], { ...base, outcomes: part(0, 1, 0) }))).toMatch(/outcomes must hold/);
    expect(coverage(withLog(request, ["move 1, move 1"], { ...base, outcomes: [...part(0, 1, 0), ...part(1, 1, 0), ...part(1, 1, 1)] }))).toMatch(/outcomes must hold/);
    // with the second foe down (the arrays show a foe at location 1 only) a part aimed at location 2 has no block; with both foes up it has one
    expect(coverage(withLog(request, ["move 1 1, move 1 2"], { ...base, outcomes: part(0, 1, 1) }))).toBeNull();
    expect(coverage(withLog(request, ["move 1 1, move 1 2"], { ...base, outcomes: [...part(0, 1, 1), ...part(1, 1, 2)] }))).toMatch(/outcomes must hold/);
    const both = { speeds: [speed(0, 1), speed(0, 2), speed(1, 1), speed(1, 2)], own_priorities: base.own_priorities };
    expect(coverage(withLog(request, ["move 1 1, move 1 2"], { ...both, outcomes: [...part(0, 1, 1), ...part(1, 1, 2)] }))).toBeNull();
    expect(coverage(withLog(request, ["move 1 1, move 1 2"], { ...both, outcomes: part(0, 1, 1) }))).toMatch(/outcomes must hold/);
  });

  it("a part of a fainted active, a forced switch and Team Preview have no outcomes; a request with them is refused", () => {
    const fainted = { active: [{ moves: [{ id: "protect" }] }, { moves: [{ id: "protect" }] }], side: { pokemon: [mon("100/100"), mon("0 fnt"), mon("100/100")] } };
    expect(coverage(withLog(fainted, ["move 1 1, move 1 1"], { speeds: [speed(0, 1)], own_priorities: [prio(0, 1), prio(1, 1)], outcomes: part(0, 1, 1) }))).toBeNull(); // slot 1 is fainted: it has no block
    expect(coverage(withLog(fainted, ["move 1 1, pass"], { speeds: [speed(0, 1)], own_priorities: [prio(0, 1)], outcomes: part(0, 1, 1) }))).toBeNull();
    expect(coverage(withLog(fainted, ["move 1 1, pass"], { speeds: [speed(0, 1)], own_priorities: [prio(0, 1)], outcomes: [...part(0, 1, 1), ...part(1, 1, 1)] }))).toMatch(/outcomes must hold/);
    const forced = { forceSwitch: [true, false], side };
    expect(coverage(withLog(forced, ["switch 3, pass"], { speeds: [speed(2, 1)] }))).toBeNull();
    expect(coverage(withLog(forced, ["switch 3, pass"], { speeds: [speed(2, 1)], outcomes: part(0, 1, 1) }))).toMatch(/outcomes must hold/);
    const preview = { teamPreview: true, side };
    expect(coverage(withLog(preview, ["team 1234"], {}))).toBeNull();
    expect(coverage(withLog(preview, ["team 1234"], { outcomes: part(0, 1, 0) }))).toMatch(/outcomes must hold/);
  });

  it("the Mega variant is a part of its own, keyed by the Mega flag", () => {
    const base = { speeds: [speed(0, 1), speed(1, 1)], own_priorities: [prio(0, 1, true), prio(1, 1)] };
    expect(coverage(withLog(request, ["move 1 1 mega, move 1 1"], { ...base, speeds: [{ ...speed(0, 1), mega: false }, { ...speed(0, 1), mega: true }, speed(1, 1)], outcomes: [...part(0, 1, 1, true), ...part(1, 1, 1)] }))).toBeNull();
  });
});

describe("the handshake states probe-outcome/1", () => {
  const FAKE = fileURLToPath(new URL("./fake-policy-worker.ts", import.meta.url));
  const outcome = { feature_set: "probe-outcome/1", feature_set_hash: "a".repeat(64) };
  const two = { feature_set: "probe-damage/2", feature_set_hash: "a".repeat(64) };
  const opened: PolicyWorkerClient[] = [];
  afterEach(async () => {
    await Promise.all(opened.splice(0).map((c) => c.close()));
  });
  const client = (stated: unknown, supply: typeof outcome | undefined): PolicyWorkerClient => {
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

  it("a worker that states it to a client that supplies another set or nothing is refused before any battle", async () => {
    await expect(client(outcome, two).start()).rejects.toThrow(/cannot supply/);
    await expect(client(outcome, undefined).start()).rejects.toThrow(/cannot supply/);
  });

  it("a stated set that this side supplies is accepted; a query with its arrays goes to that worker only, and a missing outcomes array is refused", async () => {
    const worker = client(outcome, outcome);
    await worker.start();
    expect(worker.featureSet).toEqual(outcome);
    const { query_id: _id, ...query } = sample("policy-query/valid/battle-features-outcome") as never as PolicyQueryInput & { query_id: number };
    const stated = { ...query, candidate_features: { ...(query.candidate_features as object), ...outcome } } as PolicyQueryInput;
    await expect(worker.ask("g", stated)).resolves.toBeDefined();
    const missing = { ...stated, candidate_features: { ...(stated.candidate_features as object), outcomes: undefined } } as unknown as PolicyQueryInput;
    await expect(worker.ask("g", JSON.parse(JSON.stringify(missing)) as PolicyQueryInput)).rejects.toThrow(/with outcomes under probe-outcome\/1/);
  });
});

describe("a played game under probe-outcome/1: the features sent equal the features the loader recomputes and the sidecar holds", () => {
  const known = syntheticFixtureHashes(engine);
  const sets = syntheticTeam();
  const team = stampTeam(sets as PrivateTeamSpec["team"], known);
  const withOutcomeMechanics = withMoveOutcome(mechanics);
  const configs = (change: Partial<FeatureConfigs> = {}): FeatureConfigs => ({
    vocabulary,
    config,
    mechanics: withOutcomeMechanics,
    outcome: { config: loadOutcomeConfig(GOLDEN_FORMAT) },
    snapshot,
    data_class: "synthetic",
    feature_set: FEATURE_SET_OUTCOME,
    ...change,
  });
  const identity = featureSetIdentity(engine, configs(), FEATURE_SET_OUTCOME);
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

  it("the set is the configs' own; its hash differs from /2's and is the one the configs name", () => {
    expect(featureSetOf(configs())).toBe(FEATURE_SET_OUTCOME);
    expect(configuredSetHash(engine, configs())).toBe(stated.feature_set_hash);
    expect(configuredSetHash(engine, configs({ feature_set: "probe-damage/2" }))).not.toBe(stated.feature_set_hash);
  });

  it("every query's candidate_features equals, byte for byte, what loadTrainingSet recomputes and what the sidecar holds; the examples are valid and cover their candidates", async () => {
    const a = standIn();
    const b = standIn();
    const played = { p1: new Map<number, AnyCandidateFeatures>(), p2: new Map<number, AnyCandidateFeatures>() };
    const seat = (client: PolicyClient, player: "p1" | "p2") => ({ client, policy_id: POLICY, sampling: "argmax" as const, features: { configs: configs(), record: (seq: number, features: AnyCandidateFeatures) => void played[player].set(seq, features) } });
    const { outcome, record } = await policyGame({ engine, formatId: GOLDEN_FORMAT, seed: GOLDEN_SEED, game: 0, teams: { p1: sets, p2: sets }, known_synthetic: known, seats: { p1: seat(a, "p1"), p2: seat(b, "p2") }, latency_budget: budget });
    expect(outcome.kind).toBe("played");
    const battle = record as NonNullable<typeof record>;
    const episodes = await Promise.all(
      (["p1", "p2"] as const).map((player) =>
        episodeFromRecord(battle, player, { team, opponent_team: team, known_synthetic: known, behaviour_policy_id: POLICY, collection: { kind: "league", run_id: "league-1", seed: battle.seed, opponent_policy_id: POLICY } }),
      ),
    );
    const { examples } = await loadTrainingSet({ episodes, evaluation_manifests: [], known_synthetic: known, behaviour_policy_id: POLICY, features: { engine, configs: configs() } });
    const sent = [...a.queries, ...b.queries];
    expect(sent.length).toBe(examples.length);
    let outcomes = 0;
    let unknowns = 0;
    for (const example of examples) {
      const query = sent.find((q) => q.observation.player === example.observation.player && q.identity.request_seq === example.observation.identity.request_seq);
      expect(query, "a query for every example").toBeDefined();
      const features = (query as PolicyQueryInput).candidate_features as unknown as WireFeatureRows & Record<string, unknown[]>;
      expect(features.feature_set).toBe(FEATURE_SET_OUTCOME);
      // the whole object: rows, the four arrays and the outcomes
      expect(jcs(example.candidate_features as unknown as Json)).toBe(jcs(features as unknown as Json));
      outcomes += features.outcomes?.length ?? 0;
      unknowns += JSON.stringify(features).split("history_unknown").length - 1;
      readTrainingExample(example);
      expect(featureRowsProblem(example.candidate_features as WireFeatureRows)).toBeNull();
      expect(featureCoverageProblem({ observation: example.observation, candidates: example.candidates as never }, example.candidate_features as WireFeatureRows, isDamaging)).toBeNull();
    }
    expect(outcomes).toBeGreaterThan(0);
    expect(unknowns).toBeGreaterThanOrEqual(0);

    // The game processes' recomputation and the play-time sidecar write and read the same bytes.
    const dir = mkdtempSync(join(tmpdir(), "areneon-outcome-rows-"));
    try {
      const { writeFileSync } = await import("node:fs");
      const one = episodes[0] as (typeof episodes)[number];
      const hash = await episodeHash(one, nodeSha256);
      const path = join(dir, `${hash}.episode.json`);
      writeFileSync(path, JSON.stringify(one));
      expect(await writeFeatureRows({ engine, configs: configs(), known_synthetic: known }, [{ hash, path }], join(dir, "rows"))).toBe(1);
      const read = await readEpisode(JSON.parse(JSON.stringify(one)), { known_synthetic: known });
      const seqs = (await decisionObservations(read)).map((o) => o.identity.request_seq);
      const recomputed = readFeatureRows(engine, configs(), join(dir, "rows"))(hash, seqs);
      expect(recomputed?.length).toBe(one.decisions.length);
      const own = examples.filter((e) => e.episode_hash === hash).map((e) => e.candidate_features);
      expect(jcs(recomputed as unknown as Json)).toBe(jcs(own as unknown as Json));
      // the sidecar: the rows the seat attached while playing, written as the Episode's rows file
      expect(await writePlayRows({ engine, configs: configs() }, read, hash, played.p1, join(dir, "play"))).toBe(true);
      const sidecar = readFeatureRows(engine, configs(), join(dir, "play"))(hash, seqs);
      expect(jcs(sidecar as unknown as Json)).toBe(jcs(own as unknown as Json));
      // another set reads neither
      expect(readFeatureRows(engine, configs({ feature_set: "probe-damage/2" }), join(dir, "play"))(hash, seqs)).toBeUndefined();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }, 900_000); // real probe over one seeded game: the outcomes add a move step per own move part and branch, at play and again in the loader
});
