// The private test (ADR-015 erratum E1, item 2) on stand-in teams: the committed
// fixture teams, treated as private and given marker nicknames, in a throwaway
// `private-data` world. The policy worker is an in-process stand-in (the real
// worker is driven by the Python command's tests). Nothing here is a real team.
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { actorRule } from "../src/actor-rule.ts";
import { jcs } from "../src/canonical-json.ts";
import { loadEngine } from "../src/engine.ts";
import type { PolicyClient } from "../src/policy-worker.ts";
import { readTrainingRunConfig, type TrainingRunConfig } from "../src/training-run-config.ts";
import { ProgressError } from "../src/progress-evaluation.ts";
import { PRIVATE_TEST_REFUSALS, PrivateTestRefusal, privateTestTeams, runPrivateTestCli, type PrivateTestSources } from "../src/private-test-cli.ts";
import { opponentName, PrivateTestRecordError, privateTestAllowed, privateTestRecordHash, readPrivateTestRecord } from "../src/private-test-record.ts";
import { checkSalt, privateTestCommitment, privateTestPairs, privateTestStreams } from "../src/private-test-seed.ts";
import { pairsAvailable, playPrivateTest, type PrivateTestOptions } from "../src/private-test.ts";
import { damageAwareSetup, featureSetup } from "../src/league-generation.ts";
import type { FrozenPairClientOptions } from "../src/progress-cli.ts";
import { progressGameStreams } from "../src/progress-seed.ts";
import { leagueGameStreams } from "../src/league-seed.ts";
import { type CanaryWorld, expectNoLeak, findLeaks, makeCanaryWorld, observe } from "./support/private-canary.ts";
import type { BeliefWorldsFactory } from "../src/damage-aware-worlds.ts";
import { fakePort } from "./damage-aware-world-support.ts";

const engine = loadEngine();
const ROOT = new URL("../../..", import.meta.url).pathname;
const FORMAT = "gen9championsvgc2026regmc";
const SALT_A = "ab".repeat(32);
const SALT_B = "cd".repeat(32);
const hex = (c: string): string => c.repeat(64);
const worlds: CanaryWorld[] = [];
const scratch: string[] = [];
// The guard looks at the environment: an agent session sets these, and the stand-in runs are the owner's.
beforeEach(() => {
  vi.stubEnv("CLAUDECODE", "");
  vi.stubEnv("AI_AGENT", "");
});
afterEach(() => {
  vi.unstubAllEnvs();
  while (worlds.length > 0) worlds.pop()?.cleanup();
  while (scratch.length > 0) rmSync(scratch.pop() as string, { recursive: true, force: true });
});
const world = (count = 4): CanaryWorld => {
  const w = makeCanaryWorld(engine, count, { legal: true });
  worlds.push(w);
  return w;
};

const tiny = JSON.parse(readFileSync(join(ROOT, "fixtures/contracts/v1/training-run-config/valid/tiny.json"), "utf8")) as Record<string, unknown>;
const configFile = (): string => {
  const dir = mkdtempSync(join(tmpdir(), "private-test-config-"));
  scratch.push(dir);
  const path = join(dir, "config.json");
  // A turn cap that lets games be decided (the tiny sample's cap of 6 turns ends every game as a tie).
  writeFileSync(path, JSON.stringify({ ...tiny, generation: { ...(tiny.generation as object), max_turns: 100 } }));
  return path;
};
const readPrivateConfig = (): TrainingRunConfig => readTrainingRunConfig(JSON.parse(readFileSync(configFile(), "utf8")));
const ENGINE_DATA = join(ROOT, "fixtures/showdown", FORMAT, "engine-data.json");

type Client = PolicyClient & { beginGame: () => Promise<void>; start: () => Promise<void>; close: () => Promise<void>; asked: number };
/** An in-process worker: the shared actor rule over the query's own candidates; `fail` makes the nth query an error. */
function actorClient(fail?: (n: number) => Error | undefined): Client {
  const client: Client = {
    asked: 0,
    beginGame: () => Promise.resolve(),
    start: () => Promise.resolve(),
    close: () => Promise.resolve(),
    ask(_battle, query) {
      client.asked++;
      const error = fail?.(client.asked);
      if (error !== undefined) return Promise.reject(error);
      const open = query.candidates.filter((c) => !c.pending).map((c) => c.choice);
      const pending = query.candidates.filter((c) => c.pending).map((c) => c.choice);
      return Promise.resolve({
        schema_version: 1 as const,
        identity: query.identity,
        choice: actorRule({ choices: open, pending }),
        source_policy_id: query.policy_id,
        elapsed_ms: 0,
        deadline_outcome: "in_budget" as const,
      });
    },
  };
  return client;
}

const noSources: PrivateTestSources = {
  snapshot: () => Promise.reject(new Error("no snapshot here")),
  dataset: () => {
    throw new Error("no dataset here");
  },
};

interface Run {
  code: number;
  out: string[];
  err: string[];
  record: ReturnType<typeof readPrivateTestRecord> | null;
}

/** Run the CLI in a world; every line it prints is captured. */
async function runCli(w: CanaryWorld, extra: string[] = [], options: { sources?: PrivateTestSources; client?: Client; salt?: string; skip?: string[]; config?: string; onClient?: (asked: FrozenPairClientOptions) => void } = {}): Promise<Run> {
  const saltFile = join(w.root, "salt.key");
  try {
    writeFileSync(saltFile, `${options.salt ?? SALT_A}\n`, { flag: "wx", mode: 0o600 });
  } catch {
    // the same world reused: the salt file is there (a test that changes the salt makes a new world)
  }
  const argv = [
    ["--config", options.config ?? configFile()],
    ["--generation", "2"],
    ["--preview", hex("1")],
    ["--battle", hex("2")],
    ["--policy-id-sample", `policy:${hex("a")}`],
    ["--policy-id-argmax", `policy:${hex("b")}`],
    ["--generation-manifest-sha256", hex("3")],
    ["--candidate-data-class", "synthetic"],
    ["--run-dir", w.outside],
    ["--engine-data", ENGINE_DATA],
    ["--encoder-version", "enc-test"],
    ["--worker-python", "python-not-started"],
    ["--salt-file", saltFile],
    ["--source", "staging"],
    ["--subset", "all"],
    ["--teams-dir", w.teamsDir],
  ].filter(([name]) => !(options.skip ?? []).includes(name as string)).flat();
  const out: string[] = [];
  const err: string[] = [];
  const client = options.client ?? actorClient();
  const code = await runPrivateTestCli([...argv, ...extra], (line) => out.push(line), options.sources ?? noSources, (line) => err.push(line), {
    clientFor: (asked) => {
      options.onClient?.(asked);
      return client;
    },
  });
  const line = out[0];
  return { code, out, err, record: line === undefined ? null : readPrivateTestRecord((JSON.parse(line) as { record: unknown }).record) };
}

describe("the seeds, the pairing and the commitment", () => {
  const teams = ["a", "b", "c", "d", "e"].map(hex);

  it("depends on the salt alone: the same salt replays, another salt changes everything", () => {
    expect(privateTestPairs(SALT_A, teams)).toEqual(privateTestPairs(SALT_A, [...teams].reverse()));
    expect(privateTestPairs(SALT_A, teams)).not.toEqual(privateTestPairs(SALT_B, teams));
    expect(privateTestStreams(SALT_A, 3)).toEqual(privateTestStreams(SALT_A, 3));
    expect(privateTestStreams(SALT_A, 3).seed).not.toBe(privateTestStreams(SALT_B, 3).seed);
    expect(privateTestStreams(SALT_A, 3).seed).not.toBe(privateTestStreams(SALT_A, 4).seed);
    expect(privateTestCommitment(SALT_A, teams)).toBe(privateTestCommitment(SALT_A, [...teams].reverse()));
    expect(privateTestCommitment(SALT_A, teams)).not.toBe(privateTestCommitment(SALT_B, teams));
    expect(privateTestCommitment(SALT_A, teams)).not.toBe(privateTestCommitment(SALT_A, teams.slice(1)));
  });

  it("makes disjoint pairs and leaves an odd team out", () => {
    const pairs = privateTestPairs(SALT_A, teams);
    expect(pairs).toHaveLength(2);
    expect(new Set(pairs.flat()).size).toBe(4);
    expect(pairsAvailable(5)).toBe(2);
  });

  it("shares no preimage with the league or the progress derivation", () => {
    for (let k = 0; k < 4; k++) {
      expect(privateTestStreams(SALT_A, k).seed).not.toBe(progressGameStreams("run", k).seed);
      expect(privateTestStreams(SALT_A, k).seed).not.toBe(leagueGameStreams("run", 1, k).seed);
    }
  });

  it("refuses a salt that is not 256 bits of hex and a team twice, without repeating the value", () => {
    for (const bad of ["", "ab", "AB".repeat(32), `${"ab".repeat(31)}zz`]) {
      let message = "";
      try {
        checkSalt(bad);
      } catch (error) {
        message = (error as Error).message;
      }
      expect(message).not.toBe("");
      if (bad.length > 3) expect(message).not.toContain(bad);
    }
    expect(() => privateTestPairs(SALT_A, [hex("a"), hex("a")])).toThrow(/once/);
    expect(() => privateTestCommitment(SALT_A, [hex("a"), hex("a")])).toThrow(/once/);
    expect(() => privateTestStreams(SALT_A, -1)).toThrow(RangeError);
  });
});

describe("the PrivateTestRecord reader", () => {
  const dir = join(ROOT, "fixtures/contracts/v1/private-test-record");
  const files = (kind: string): string[] => readdirSync(join(dir, kind)).sort();
  const sample = (kind: string, name: string): unknown => JSON.parse(readFileSync(join(dir, kind, name), "utf8"));

  it("reads every valid sample and refuses every reader-invalid one with a fixed rule", () => {
    expect(files("valid").length).toBe(3);
    for (const name of files("valid")) expect(readPrivateTestRecord(sample("valid", name)).data_class).toBe("private");
    expect(files("reader-invalid").length).toBeGreaterThan(15);
    for (const name of files("reader-invalid")) {
      const value = sample("reader-invalid", name);
      expect(() => readPrivateTestRecord(value), name).toThrow(PrivateTestRecordError);
    }
  });

  it("refuses every schema-invalid sample, also with the private label put back", () => {
    // synthetic-class is a sound record with the wrong label, so it is invalid only for that.
    expect(() => readPrivateTestRecord(sample("invalid", "synthetic-class.json"))).toThrow(PrivateTestRecordError);
    expect(readPrivateTestRecord({ ...(sample("invalid", "synthetic-class.json") as object), data_class: "private" }).data_class).toBe("private");
    for (const name of files("invalid").filter((n) => n !== "synthetic-class.json")) {
      const value = sample("invalid", name) as Record<string, unknown>;
      expect(() => readPrivateTestRecord(value), name).toThrow(PrivateTestRecordError);
      expect(() => readPrivateTestRecord({ ...value, data_class: "private" }), `${name} (private)`).toThrow(PrivateTestRecordError);
    }
  });

  it("states a rule and never repeats a value", () => {
    const value = sample("valid", "staging.json") as Record<string, unknown>;
    const canary = "CanaryValue123456";
    let message = "";
    try {
      readPrivateTestRecord({ ...value, run_id: canary, extra: canary });
    } catch (error) {
      message = (error as Error).message;
    }
    expect(message).not.toBe("");
    expect(message).not.toContain(canary);
  });

  it("names the opponent of a match and the sources a private pair may use", () => {
    expect(opponentName({ opponent_policy_id: "actor-rule@v1" })).toBe("actor-rule");
    expect(opponentName({ opponent_policy_id: `policy:${hex("a")}` })).toBe("pair");
    expect(privateTestAllowed("snapshot", "holdout")).toBe(true);
    expect(privateTestAllowed("dataset", "all")).toBe(true);
    expect(privateTestAllowed("snapshot", "all")).toBe(false);
    expect(privateTestAllowed("staging", "all")).toBe(false);
    const record = readPrivateTestRecord(sample("valid", "staging.json"));
    expect(privateTestRecordHash(record)).toMatch(/^[0-9a-f]{64}$/);
  });
});

describe("privateTestTeams", () => {
  it("refuses a paste with a code and a 1-based position and never the paste", () => {
    const w = world(2);
    const [a, b] = w.teams as [CanaryWorld["teams"][number], CanaryWorld["teams"][number]];
    const broken = (w.markers[0] as string) + "\nnot a team";
    const cases: [string[], string, number][] = [
      [[a.paste, broken], "team_import_failed", 2],
      [[a.paste, a.paste], "team_repeated", 2],
      [[a.paste, a.paste.replace(/Level: \d+/, "Level: 1").replace(/EVs: [^\n]*/g, "EVs: 252 HP / 252 Atk")], "team_invalid", 2],
    ];
    for (const [pastes, code, position] of cases) {
      let thrown: unknown;
      try {
        privateTestTeams(engine, FORMAT, pastes);
      } catch (error) {
        thrown = error;
      }
      expect(thrown, code).toBeInstanceOf(PrivateTestRefusal);
      const refusal = thrown as PrivateTestRefusal;
      if (refusal.code === "team_invalid" || refusal.code === "team_import_failed" || refusal.code === "team_repeated") expect(refusal.position).toBe(position);
      expect(refusal.message).toBe(`private-test: ${refusal.code} ${refusal.position}`);
      expect(findLeaks(w, { stdout: "", stderr: "", thrown: refusal.stack ?? "", surfaces: {}, files: [], loosePaths: [] })).toEqual([]);
    }
    expect(privateTestTeams(engine, FORMAT, [a.paste, b.paste]).map((t) => t.canonical_team_hash)).toHaveLength(2);
  });
});

describe("the private test on stand-in teams", () => {
  it("plays every pair from both seats and reports counts only, with no marker anywhere", async () => {
    const w = world(4);
    const client = actorClient();
    const seen = await observe(w, async () => {
      const run = await runCli(w, [], { client });
      expect(run.err).toEqual([]);
      expect(run.code).toBe(0);
      return { surfaces: { stdout: run.out.join("\n"), report: run.err.join("\n") } };
    });
    expectNoLeak(w, seen);
    expect(seen.thrown).toBeNull();
    expect(seen.files).toEqual([]);
    // Negative control: the scan is able to see what it looks for.
    expect(findLeaks(w, { ...seen, surfaces: { stdout: `${seen.surfaces.stdout}\n${w.markers[0]}` } }).length).toBeGreaterThan(0);
    const run = await runCli(w, [], { client: actorClient() });
    const record = run.record as NonNullable<Run["record"]>;
    expect(record.data_class).toBe("private");
    expect(record.test_set).toMatchObject({ team_count: 4, pairs: 2, source_kind: "staging", subset: "all" });
    expect(record.matches.map((m) => m.match_id)).toEqual(["actor-rule-argmax", "actor-rule-sample", "random-legal-argmax", "random-legal-sample"]);
    for (const m of record.matches) {
      expect(m.games).toBe(4);
      expect(m.wins + m.ties + m.losses).toBe(4);
      expect(m.pair_score_counts.reduce((a, b) => a + b, 0)).toBe(2);
    }
    // The commitment opens against the stand-in teams' canonical hashes and the salt, and only so.
    const hashes = w.teams.map((t) => privateTestTeams(engine, FORMAT, [t.paste])[0]?.canonical_team_hash as string);
    expect(record.test_set.commitment).toBe(privateTestCommitment(SALT_A, hashes));
    const text = jcs(record as never);
    for (const hash of hashes) expect(text).not.toContain(hash);
    expect(text).not.toContain(SALT_A);
  }, 120_000);

  it("gives the same record bytes for the same salt and inputs, and another commitment for another salt", async () => {
    const w = world(4);
    const first = await runCli(w, [], { client: actorClient() });
    const second = await runCli(w, [], { client: actorClient() });
    expect(jcs(first.record as never)).toBe(jcs(second.record as never));
    const w2 = world(4);
    const other = await runCli(w2, [], { client: actorClient(), salt: SALT_B });
    expect(other.record?.test_set.commitment).not.toBe(first.record?.test_set.commitment);
  }, 180_000);

  it("scores a policy failure as a loss with its cause and keeps its text out", async () => {
    const w = world(2);
    const marker = w.markers[0] as string;
    const seen = await observe(w, async () => {
      const run = await runCli(w, [], { client: actorClient((n) => (n >= 1 ? new Error(`worker said ${marker}`) : undefined)) });
      expect(run.code).toBe(0);
      const rec = run.record as NonNullable<Run["record"]>;
      for (const m of rec.matches) {
        expect(m.wins).toBe(0);
        expect(m.losses).toBe(2);
        expect(m.forfeits.reduce((a, f) => a + f.count, 0)).toBe(2);
        expect(m.forfeits.every((f) => f.side === "candidate")).toBe(true);
      }
      return { surfaces: { stdout: run.out.join("\n"), report: run.err.join("\n") } };
    });
    expectNoLeak(w, seen);
  }, 120_000);

  it("plays the damage-aware baseline when asked, and another frozen pair", async () => {
    const w = world(2);
    writeFileSync(join(w.outside, "run-config.json"), JSON.stringify(tiny)); // the other run's config: it names its own candidate features, here none
    const run = await runCli(w, ["--damage-aware-budget-ms", "600000", "--max-pairs", "1", "--opponent-pair", `${hex("5")},${hex("6")},policy:${hex("c")},policy:${hex("d")}`, "--opponent-run-dir", w.outside, "--opponent-encoder-version", "enc-test"], { client: actorClient() });
    expect(run.code).toBe(0);
    const ids = (run.record as NonNullable<Run["record"]>).matches.map((m) => m.match_id);
    expect(ids).toEqual(["actor-rule-argmax", "actor-rule-sample", "damage-aware-argmax", "damage-aware-sample", "pair-argmax", "pair-sample", "random-legal-argmax", "random-legal-sample"]);
  }, 240_000);

  it("gives a pair that reads candidate features its statement and its rows, and the opponent pair its own", async () => {
    const w = world(2);
    const named = (feature_set_hash: string): Record<string, unknown> => {
      const model = tiny.model as { encoder: object };
      return { ...tiny, generation: { ...(tiny.generation as object), max_turns: 100 }, model: { ...model, encoder: { ...model.encoder, options: { candidate_features: { feature_set: "probe-damage/1", feature_set_hash } } } } };
    };
    let hash = "";
    try {
      featureSetup(engine, readTrainingRunConfig(named("0".repeat(64))));
    } catch (error) {
      hash = /computes as ([0-9a-f]{64})/.exec((error as Error).message)?.[1] ?? "";
    }
    expect(hash).toMatch(/^[0-9a-f]{64}$/);
    const dir = mkdtempSync(join(tmpdir(), "private-test-features-"));
    scratch.push(dir);
    const own = join(dir, "config.json");
    writeFileSync(own, JSON.stringify(named(hash)));
    writeFileSync(join(w.outside, "run-config.json"), JSON.stringify(tiny)); // the opponent pair reads none
    const asked: FrozenPairClientOptions[] = [];
    const run = await runCli(w, ["--max-pairs", "1", "--opponent-pair", `${hex("5")},${hex("6")},policy:${hex("c")},policy:${hex("d")}`, "--opponent-run-dir", w.outside, "--opponent-encoder-version", "enc-test"], { config: own, onClient: (o) => asked.push(o) });
    expect(run.code).toBe(0);
    expect(asked.map((o) => o.features)).toEqual([{ feature_set: "probe-damage/1", feature_set_hash: hash }, undefined]);
    // A config naming a feature set this checkout does not compute is a fixed refusal, before any worker starts.
    const wrong = join(dir, "wrong.json");
    writeFileSync(wrong, JSON.stringify(named("0".repeat(64))));
    expect((await runCli(w, ["--max-pairs", "1"], { config: wrong })).err).toEqual(["private-test: worker_failed"]);
  }, 240_000);

  it("sets a belief pair up through the composition root's factory, and refuses one without a factory", async () => {
    const w = world(2);
    const named = (feature_set_hash: string): Record<string, unknown> => {
      const model = tiny.model as { encoder: object };
      return { ...tiny, generation: { ...(tiny.generation as object), max_turns: 100 }, model: { ...model, encoder: { ...model.encoder, options: { candidate_features: { feature_set: "probe-damage-belief/1", feature_set_hash } } } } };
    };
    const calls: number[] = [];
    const factory: BeliefWorldsFactory = () => {
      calls.push(1);
      return fakePort();
    };
    let hash = "";
    try {
      featureSetup(engine, readTrainingRunConfig(named("0".repeat(64))), factory);
    } catch (error) {
      hash = /computes as ([0-9a-f]{64})/.exec((error as Error).message)?.[1] ?? "";
    }
    expect(hash).toMatch(/^[0-9a-f]{64}$/);
    calls.length = 0;
    const dir = mkdtempSync(join(tmpdir(), "private-test-belief-"));
    scratch.push(dir);
    const own = join(dir, "config.json");
    writeFileSync(own, JSON.stringify(named(hash)));
    const asked: FrozenPairClientOptions[] = [];
    const run = await runCli(w, ["--max-pairs", "1"], { config: own, onClient: (o) => asked.push(o), sources: { ...noSources, belief_worlds: factory } });
    expect(run.code).toBe(0);
    expect(calls.length).toBeGreaterThan(0);
    expect(asked.map((o) => o.features)).toEqual([{ feature_set: "probe-damage-belief/1", feature_set_hash: hash }]);
    expect((await runCli(w, ["--max-pairs", "1"], { config: own })).err).toEqual(["private-test: worker_failed"]);
  }, 240_000);

  it("refuses an options mistake before it plays", async () => {
    const w = world(2);
    expect((await runCli(w, ["--max-pairs", "2"])).err).toEqual(["private-test: too_many_pairs"]);
    await expect(playPrivateTest({ engine, config: readPrivateConfig(), generation: 0, generation_manifest_sha256: hex("3"), checkpoints: { preview: hex("1"), battle: hex("2") }, policy_ids: { sample: `policy:${hex("a")}`, argmax: `policy:${hex("b")}` }, candidate_data_class: "synthetic", client: actorClient(), teams: [], source: { kind: "staging" }, subset: "all", salt: SALT_A })).rejects.toThrow(/at least two teams/);
  });
});

describe("every refusal", () => {
  it("refuses in an agent session before it reads a team", async () => {
    const w = world(2);
    for (const name of ["CLAUDECODE", "AI_AGENT"]) {
      vi.stubEnv(name, "1");
      const read = vi.fn();
      const run = await runCli(w, [], { sources: { ...noSources, dataset: read } });
      expect(run.code).toBe(1);
      expect(run.err).toEqual(["refusing: an agent session is running this; use your own terminal"]);
      expect(run.out).toEqual([]);
      expect(read).not.toHaveBeenCalled();
    }
  });

  it("names a code only for each bad input", async () => {
    const w = world(2);
    const cases: [string[], string[], string][] = [
      [["--source", "snapshot", "--subset", "holdout"], ["--source", "--subset"], "usage"],
      [["--subset", "holdout"], ["--subset"], "usage"],
      [["--candidate-data-class", "private"], ["--candidate-data-class"], "private_pair_source"],
      [["--candidate-data-class", "other"], ["--candidate-data-class"], "usage"],
      [["--no-such-flag", "x"], [], "usage"],
      [["--max-pairs", "0"], [], "usage"],
      [["--opponent-pair", "x,y"], [], "usage"],
    ];
    for (const [extra, skip, code] of cases) {
      const run = await runCli(w, extra, { skip });
      expect(run.err, extra.join(" ")).toEqual([`private-test: ${code}`]);
      expect(run.code).toBe(code === "usage" ? 2 : 1);
    }
  });

  it("refuses a missing or malformed salt without saying where it looked", async () => {
    const w = world(2);
    const saltFile = join(w.root, "bad.key");
    writeFileSync(saltFile, "not-hex");
    const run = await runCli(w, ["--salt-file", saltFile], { skip: ["--salt-file"] });
    expect(run.err).toEqual(["private-test: salt_unreadable"]);
    const missing = await runCli(w, ["--salt-file", join(w.root, "absent.key")], { skip: ["--salt-file"] });
    expect(missing.err).toEqual(["private-test: salt_unreadable"]);
  });

  it("refuses an invalid team with its position and nothing of the team", async () => {
    const w = world(3);
    const seen = await observe(w, async () => {
      const bad = join(w.teamsDir, "zzz-last.txt");
      writeFileSync(bad, `${w.markers[0]} (Pikachu) @ Light Ball\nAbility: Static\n- Thunderbolt`);
      const run = await runCli(w);
      expect(run.code).toBe(1);
      expect(run.err).toEqual(["private-test: team_import_failed 4"]);
      return { surfaces: { stdout: run.out.join("\n"), report: run.err.join("\n") } };
    });
    expectNoLeak(w, seen);
    expect(Object.keys(PRIVATE_TEST_REFUSALS)).toContain("team_invalid");
  });

  it("refuses an empty or unreadable staging directory and a single team", async () => {
    const w = world(1);
    expect((await runCli(w)).err).toEqual(["private-test: too_few_teams"]);
    const empty = join(w.root, "..", "empty");
    mkdirSync(empty, { mode: 0o700 });
    expect((await runCli(w, ["--teams-dir", empty], { skip: ["--teams-dir"] })).err).toEqual(["private-test: source_unreadable"]);
  });

  it("reads a snapshot and a dataset only through the sources it is given", async () => {
    const w = world(2);
    const teams = privateTestTeams(engine, FORMAT, w.teams.map((t) => t.paste));
    const snapshot = vi.fn(() => Promise.resolve({ teams, source: { kind: "snapshot" as const, snapshot_hash: hex("8") }, subset: "holdout" as const }));
    const run = await runCli(w, ["--source", "snapshot", "--subset", "holdout", "--snapshot-hash", hex("8"), "--snapshots-dir", w.outside, "--candidate-data-class", "private"], { sources: { ...noSources, snapshot }, skip: ["--source", "--subset", "--teams-dir", "--candidate-data-class"] });
    expect(run.code).toBe(0);
    expect(snapshot).toHaveBeenCalledOnce();
    expect(run.record?.test_set).toMatchObject({ source_kind: "snapshot", snapshot_hash: hex("8"), subset: "holdout" });
    expect(run.record?.candidate_data_class).toBe("private");
    const bad = await runCli(w, ["--source", "snapshot", "--snapshot-hash", "xyz", "--snapshots-dir", w.outside], { sources: { ...noSources, snapshot }, skip: ["--source", "--teams-dir"] });
    expect(bad.err).toEqual(["private-test: usage"]);
    const dataset = await runCli(w, ["--source", "dataset"], { skip: ["--source", "--teams-dir"] });
    expect(dataset.err).toEqual(["private-test: source_unreadable"]);
  }, 120_000);

  it("shows an unexpected failure as a bare word", async () => {
    const w = world(2);
    const run = await runCli(w, [], { client: { ...actorClient(), start: () => Promise.reject(new Error(`boom ${w.markers[0]}`)) } });
    expect(run.err).toEqual(["private-test: worker_failed"]);
    expect(run.code).toBe(1);
  });
});

describe("playPrivateTest refusals", () => {
  const options = (w: CanaryWorld, change: Partial<PrivateTestOptions> = {}): PrivateTestOptions => ({
    engine,
    config: readPrivateConfig(),
    generation: 0,
    generation_manifest_sha256: hex("3"),
    checkpoints: { preview: hex("1"), battle: hex("2") },
    policy_ids: { sample: `policy:${hex("a")}`, argmax: `policy:${hex("b")}` },
    candidate_data_class: "synthetic",
    client: actorClient(),
    teams: privateTestTeams(engine, FORMAT, w.teams.map((team) => team.paste)),
    source: { kind: "staging" },
    subset: "all",
    salt: SALT_A,
    ...change,
  });

  it("refuses a team twice, more pairs than the set gives and a damage-aware setup that does not fit", async () => {
    const w = world(2);
    const base = options(w);
    await expect(playPrivateTest({ ...base, teams: [...base.teams, base.teams[0] as never] })).rejects.toThrow(ProgressError);
    await expect(playPrivateTest({ ...base, max_pairs: 2 })).rejects.toThrow(/1 disjoint pairs/);
    await expect(playPrivateTest({ ...base, max_pairs: 0 })).rejects.toThrow(/disjoint pairs/);
    const setup = damageAwareSetup(engine, FORMAT);
    await expect(playPrivateTest({ ...base, damage_aware: { setup, match_budget_ms: 0 } })).rejects.toThrow(/positive whole number/);
    await expect(playPrivateTest({ ...base, damage_aware: { setup: { ...setup, config: { ...setup.config, epsilon_ppm: 5 } }, match_budget_ms: 1000 } })).rejects.toThrow(/epsilon_ppm/);
    await expect(playPrivateTest({ ...base, damage_aware: { setup: { ...setup, config: { ...setup.config, format_id: "gen9other" } }, match_budget_ms: 1000 } })).rejects.toThrow(/another format/);
  });

  it("fails a damage-aware match that passes its wall-clock budget, writing nothing", async () => {
    const w = world(2);
    let now = 0;
    const clock = (): number => (now += 10);
    await expect(playPrivateTest({ ...options(w), max_pairs: 1, damage_aware: { setup: damageAwareSetup(engine, FORMAT), match_budget_ms: 1, clock } })).rejects.toThrow(/wall-clock budget/);
  }, 120_000);

  it("records a public dataset as its source with its hash, and a snapshot with its hash", async () => {
    const w = world(2);
    const dataset = await playPrivateTest({ ...options(w), max_pairs: 1, source: { kind: "dataset", dataset_hash: hex("9") }, candidate_data_class: "private" });
    expect(dataset.record.test_set).toMatchObject({ source_kind: "dataset", dataset_hash: hex("9") });
    expect(dataset.record.matches).toHaveLength(4);
    const snapshot = await playPrivateTest({ ...options(w), max_pairs: 1, source: { kind: "snapshot", snapshot_hash: hex("8") }, subset: "holdout", candidate_data_class: "private" });
    expect(snapshot.record.test_set).toMatchObject({ source_kind: "snapshot", snapshot_hash: hex("8"), subset: "holdout" });
  }, 120_000);
});

describe("the command line, more refusals", () => {
  it("names a usage error for every malformed option", async () => {
    const w = world(2);
    const bad: [string[], string[]][] = [
      [["--subset", "some"], ["--subset"]],
      [["--source", "bucket"], ["--source"]],
      [["--source", "dataset", "--subset", "holdout"], ["--source", "--subset"]],
      [["--source", "snapshot"], ["--source"]],
      [["--source", "staging"], ["--teams-dir"]],
      [["--source", "snapshot", "--snapshot-hash", hex("8")], ["--source", "--teams-dir"]],
      [["--generation", "x"], ["--generation"]],
      [["--opponent-pair", "a,b,c,d", "--opponent-run-dir", w.outside, "--opponent-encoder-version", "e"], []],
      [["--opponent-pair", `${hex("1")},${hex("2")},policy:${hex("c")},policy:${hex("d")}`], []],
      [["--config", join(w.outside, "absent.json")], ["--config"]],
    ];
    for (const [extra, skip] of bad) {
      const run = await runCli(w, extra, { skip });
      expect(run.err, extra.join(" ")).toEqual(["private-test: usage"]);
      expect(run.code).toBe(2);
    }
    expect((await runCli(w, [], { skip: ["--battle"] })).err).toEqual(["private-test: usage"]);
  });

  it("reads a public dataset through its source, with its teams proven synthetic", async () => {
    const w = world(4);
    const teams = privateTestTeams(engine, FORMAT, w.teams.map((team) => team.paste));
    const dataset = vi.fn(() => ({ dataset_hash: hex("9"), teams, known_synthetic: new Set<string>() }));
    const run = await runCli(w, ["--source", "dataset", "--candidate-data-class", "private", "--max-pairs", "1"], { sources: { ...noSources, dataset }, skip: ["--source", "--teams-dir", "--candidate-data-class"] });
    expect(run.code).toBe(0);
    expect(dataset).toHaveBeenCalledOnce();
    expect(run.record?.test_set).toMatchObject({ source_kind: "dataset", dataset_hash: hex("9"), team_count: 4, pairs: 1 });
    expect(run.record?.candidate_data_class).toBe("private");
  }, 120_000);

  it("shows an unexpected failure of a source, a worker or a match as a bare word", async () => {
    const w = world(2);
    const throwing = { ...noSources, snapshot: () => Promise.reject(new Error(`boom ${w.markers[0]}`)) };
    const run = await runCli(w, ["--source", "snapshot", "--snapshot-hash", hex("8"), "--snapshots-dir", w.outside], { sources: throwing, skip: ["--source", "--teams-dir"] });
    expect(run.err).toEqual(["private-test: failed"]);
    expect(run.code).toBe(1);
    // The real worker command is used without the seam: a Python that does not exist is a failed worker.
    const out: string[] = [];
    const err: string[] = [];
    const saltFile = join(w.root, "salt2.key");
    writeFileSync(saltFile, SALT_A, { mode: 0o600 });
    const argv = ["--config", configFile(), "--generation", "0", "--preview", hex("1"), "--battle", hex("2"), "--policy-id-sample", `policy:${hex("a")}`, "--policy-id-argmax", `policy:${hex("b")}`, "--generation-manifest-sha256", hex("3"), "--candidate-data-class", "synthetic", "--run-dir", w.outside, "--engine-data", ENGINE_DATA, "--encoder-version", "e", "--worker-python", join(w.outside, "no-python"), "--salt-file", saltFile, "--source", "staging", "--subset", "all", "--teams-dir", w.teamsDir];
    expect(await runPrivateTestCli(argv, (l) => out.push(l), noSources, (l) => err.push(l))).toBe(1);
    expect(err).toEqual(["private-test: worker_failed"]);
    // A match that passes its wall-clock budget fails the whole test with one word.
    const slow = await runCli(w, ["--damage-aware-budget-ms", "1", "--max-pairs", "1"]);
    expect(slow.err).toEqual(["private-test: play_failed"]);
    expect(slow.out).toEqual([]);
  }, 120_000);

  it("rejects a record that is not an object and one with no match, with a path only", () => {
    for (const bad of ["x", {}, { ...(JSON.parse(readFileSync(join(ROOT, "fixtures/contracts/v1/private-test-record/valid/staging.json"), "utf8")) as object), matches: [] }]) {
      expect(() => readPrivateTestRecord(bad)).toThrow(/invalid PrivateTestRecord at/);
    }
  });
});
