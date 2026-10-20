import { mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { actorRule } from "../src/actor-rule.ts";
import { jcs, type Json } from "../src/canonical-json.ts";
import { readTrainingRunConfig } from "../src/training-run-config.ts";
import {
  GenerationError,
  type GenerationOptions,
  type GenerationResult,
  planGeneration,
  playGeneration,
  trainingTeamPool,
  type TrainingPool,
  writeGeneration,
} from "../src/league-generation.ts";
import { loadEngine } from "../src/index.ts";
import { type PolicyClient, PolicyWorkerError } from "../src/index.ts";
import { UnsupportedCandidateError } from "../src/evaluation-harness.ts";
import { readTrainingSet } from "../src/training-loader.ts";
import { SAMPLES_DIR } from "../src/samples.ts";
import { teamHash } from "../src/team-class.ts";

const engine = loadEngine();
const at = <T>(items: readonly T[], index: number): T => items[index] as T;
const sample = (name: string): unknown => JSON.parse(readFileSync(join(SAMPLES_DIR, `v1/training-run-config/valid/${name}.json`), "utf8"));
const config = readTrainingRunConfig(sample("tiny"));
const POLICY = `policy:${"c".repeat(64)}`;

/** An in-process worker: the shared actor rule over the query's own candidates. */
function actorClient(change: Partial<PolicyClient> & { begun?: () => void } = {}): PolicyClient & { beginGame: () => Promise<void>; asked: number } {
  const client = {
    asked: 0,
    beginGame: () => {
      change.begun?.();
      return Promise.resolve();
    },
    ask(_battle, query) {
      client.asked++;
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
  } satisfies PolicyClient & { beginGame: () => Promise<void>; asked: number };
  return client;
}

let pool: TrainingPool;
beforeAll(() => {
  pool = trainingTeamPool(engine, config.generation.excluded_abilities);
});

describe("the interim team pool", () => {
  it("holds the committed synthetic teams, sorted by hash, each proven synthetic by source", () => {
    expect(pool.teams.length).toBeGreaterThan(60);
    expect(pool.hashes).toEqual([...pool.hashes].sort());
    expect(new Set(pool.hashes).size).toBe(pool.hashes.length);
    expect(pool.teams.map(teamHash)).toEqual([...pool.hashes]);
    for (const hash of pool.hashes) expect(pool.known_synthetic.has(hash)).toBe(true);
    expect(pool.known_synthetic.has("0".repeat(64))).toBe(false);
  });
});

describe("the excluded abilities", () => {
  it("leave out every team with a set that holds one, and only those", () => {
    const everything = trainingTeamPool(engine);
    const without = trainingTeamPool(engine, ["Illusion"]);
    const holders = everything.teams.filter((team) => team.some((set) => set.ability === "Illusion"));
    expect(holders.length).toBeGreaterThan(0);
    expect(without.teams.length).toBe(everything.teams.length - holders.length);
    expect(without.teams.every((team) => team.every((set) => set.ability !== "Illusion"))).toBe(true);
    expect(trainingTeamPool(engine, []).hashes).toEqual(everything.hashes);
  });

  it("are refused when they leave no team", () => {
    const abilities = new Set(trainingTeamPool(engine).teams.flatMap((team) => team.map((set) => set.ability)));
    expect(() => trainingTeamPool(engine, [...abilities])).toThrow(GenerationError);
  });
});

describe("planGeneration", () => {
  it("plans the self-play games, then each baseline's, from the config alone", () => {
    const plan = planGeneration(config, 1, pool.teams.length);
    expect(plan.map((g) => g.game)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(plan.slice(0, 4).every((g) => g.opponent === "self" && g.policy_seat === "both")).toBe(true);
    expect(plan.slice(4).map((g) => g.opponent)).toEqual(Array(2).fill({ actor: "random-legal", policy_id: "random-legal@tiny" }));
    expect(plan.slice(4).map((g) => g.policy_seat)).toEqual(["p1", "p2"]);
    expect(planGeneration(config, 1, pool.teams.length)).toEqual(plan);
    for (const g of plan) {
      expect(g.seed).toMatch(/^sodium,[0-9a-f]{32}$/);
      expect(g.teams.p1).toBeLessThan(pool.teams.length);
      expect(g.teams.p2).toBeLessThan(pool.teams.length);
      expect(Number.isInteger(g.baseline_seed) && g.baseline_seed >= 0 && g.baseline_seed <= 0xffffffff).toBe(true);
    }
    expect(new Set(plan.map((g) => g.seed)).size).toBe(plan.length);
  });

  it("changes with the generation and the run key", () => {
    const first = planGeneration(config, 1, pool.teams.length);
    expect(planGeneration(config, 2, pool.teams.length).map((g) => g.seed)).not.toEqual(first.map((g) => g.seed));
    const other = readTrainingRunConfig({ ...(sample("tiny") as object), seeds: { ...config.seeds, run_key: "another-run" } });
    expect(planGeneration(other, 1, pool.teams.length).map((g) => g.seed)).not.toEqual(first.map((g) => g.seed));
  });

  it("has no baseline games when none are configured, and refuses a bad generation or pool", () => {
    const plain = readTrainingRunConfig(sample("no-baselines"));
    expect(planGeneration(plain, 1, 5)).toHaveLength(4);
    expect(() => planGeneration(config, 0, 5)).toThrow(RangeError);
    expect(() => planGeneration(config, 1.5, 5)).toThrow(RangeError);
    expect(() => planGeneration(config, 1, 0)).toThrow(/pool is empty/);
  });
});

describe("playGeneration", () => {
  let result: GenerationResult;
  let begun = 0;
  const run = (change: Partial<GenerationOptions> = {}) =>
    playGeneration({ engine, config, generation: 1, policy_id: POLICY, client: actorClient({ begun: () => begun++ }), pool, ...change });
  beforeAll(async () => {
    result = await run();
  }, 120_000);

  it("plays every planned game and writes an Episode for each seat", () => {
    expect(result.games).toHaveLength(6);
    expect(result.games.every((g) => g.outcome === "played")).toBe(true);
    expect(result.episodes).toHaveLength(12);
    expect(begun).toBe(3); // three waves of n_parallel = 2
    for (const [index, episode] of result.episodes.entries()) {
      const entry = at(result.plan, Math.floor(index / 2));
      expect(episode.provenance.collection).toMatchObject({ kind: "league", run_id: "tiny-run", seed: entry.seed });
      expect([episode.data_class, episode.rights]).toEqual(["synthetic", "cc0-synthetic"]);
    }
  });

  it("trains only on the frozen pair's own seats: a baseline seat never reaches the stream", () => {
    const own = result.episodes.filter((e) => e.provenance.behaviour_policy_id === POLICY);
    const baseline = result.episodes.filter((e) => e.provenance.behaviour_policy_id === "random-legal@tiny");
    expect(own).toHaveLength(10);
    expect(baseline).toHaveLength(2);
    const ownHashes = result.episode_hashes.filter((_, i) => at(result.episodes, i).provenance.behaviour_policy_id === POLICY);
    expect(result.set.episode_hashes).toEqual([...ownHashes].sort());
    for (const hash of result.episode_hashes.filter((h) => !ownHashes.includes(h))) expect(result.set.episode_hashes).not.toContain(hash);
    expect(result.set).toMatchObject({ learner_ready: true, behaviour_policy_id: POLICY, data_class: "synthetic" });
    expect(readTrainingSet(result.set).examples).toBe(result.stream.split("\n").length - 1);
    // a baseline game names the policy as the baseline's opponent and the baseline as the policy's
    const baselineGame = result.episodes.slice(8);
    const sides = baselineGame.map((e) => e.provenance.collection as { opponent_policy_id: string });
    expect(new Set(sides.map((c) => c.opponent_policy_id))).toEqual(new Set([POLICY, "random-legal@tiny"]));
  });

  it("is reproducible: the same plan, policy and worker give the same stream, hash and teams", async () => {
    const again = await run();
    expect(again.stream).toBe(result.stream);
    expect(again.training_set_hash).toBe(result.training_set_hash);
    expect(again.episode_hashes).toEqual(result.episode_hashes);
    expect(again.team_hashes).toEqual(result.team_hashes);
  }, 120_000);

  it("reports the teams it drew by hash, the version manifest and a hash of the TrainingSet", () => {
    const expected = new Set(result.plan.flatMap((g) => [at(pool.hashes, g.teams.p1), at(pool.hashes, g.teams.p2)]));
    expect(result.team_hashes).toEqual([...expected].sort());
    expect(result.manifest.format_id).toBe(config.format_id);
    expect(result.training_set_hash).toMatch(/^[0-9a-f]{64}$/);
  });

  it("holds nothing the opponent hid: no opponent set, and no move or item it never showed", () => {
    for (const [index, episode] of result.episodes.entries()) {
      if (episode.provenance.behaviour_policy_id !== POLICY) continue;
      const entry = at(result.plan, Math.floor(index / 2));
      const rival = entry.teams[episode.player === "p1" ? "p2" : "p1"];
      const opponent = at(pool.teams, rival);
      const mine = new Set(at(pool.teams, entry.teams[episode.player]).flatMap((set) => [set.item, ...set.moves]));
      const bytes = linesOf(result, at(result.episode_hashes, index));
      expect(bytes.length).toBeGreaterThan(0);
      // Negative control: the check can see a set when it is the seat's own (the other seat of a self game holds its own team).
      const twin = at(result.episodes, index + (episode.player === "p1" ? 1 : -1));
      if (entry.opponent === "self") {
        const twinBytes = linesOf(result, at(result.episode_hashes, index + (episode.player === "p1" ? 1 : -1)));
        expect(twinBytes.includes(jcs(at(at(pool.teams, entry.teams[twin.player]), 0) as unknown as Json))).toBe(true);
      }
      const seen = episode.channel.join("\n");
      for (const set of opponent) {
        expect(bytes.includes(jcs(set as unknown as Json))).toBe(false);
        for (const secret of [set.item, ...set.moves]) {
          if (secret !== "" && !seen.includes(secret) && !mine.has(secret)) expect(bytes.includes(secret), secret).toBe(false);
        }
      }
    }
  });

  it("treats a policy failure as a forfeit, which is not an Episode, so a generation of forfeits is refused", async () => {
    const failing: PolicyClient = { ask: () => Promise.reject(new Error("boom")) };
    await expect(run({ client: failing })).rejects.toThrow(/every game was forfeited/);
    const gone: PolicyClient = { ask: () => Promise.reject(new PolicyWorkerError("crash", "gone")) };
    await expect(run({ client: gone })).rejects.toThrow(GenerationError);
  }, 120_000);

  it("fails the generation on a harness incident: a candidate shape the worker cannot read is not a policy forfeit", async () => {
    const unsupported: PolicyClient = { ask: () => Promise.reject(new UnsupportedCandidateError("a candidate has a shape the worker does not know")) };
    await expect(run({ client: unsupported })).rejects.toThrow(/game 0 \(sodium,[0-9a-f]{32}\): unsupported_candidate/);
  }, 120_000);

  it("counts a forfeited game in the rows and trains on the games that were played", async () => {
    let calls = 0;
    const good = actorClient();
    const flaky: PolicyClient & { beginGame: () => Promise<void> } = {
      beginGame: good.beginGame,
      ask: (battle, query) => (++calls === 3 ? Promise.reject(new PolicyWorkerError("crash", "gone")) : good.ask(battle, query)),
    };
    const partial = await run({ client: flaky, config: readTrainingRunConfig({ ...(sample("tiny") as object), generation: { ...config.generation, n_parallel: 1 } }) });
    expect(partial.games.filter((g) => g.outcome === "forfeit")).toHaveLength(1);
    expect(partial.episodes.length).toBeLessThan(12);
    expect(partial.set.examples).toBeGreaterThan(0);
  }, 120_000);
});

/** The stream lines of one Episode: what the learner sees of that seat's battle. */
const linesOf = (result: GenerationResult, hash: string): string =>
  result.stream
    .split("\n")
    .filter((line) => line.includes(`"episode_hash":"${hash}"`))
    .join("\n");

describe("writeGeneration", () => {
  let dir: string;
  beforeAll(() => {
    dir = mkdtempSync(join(tmpdir(), "areneon-gen-"));
  });
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  it("writes the episodes, stream, TrainingSet, plan and summary as canonical JSON, without a clock", async () => {
    const result = await playGeneration({ engine, config, generation: 2, policy_id: POLICY, client: actorClient(), pool });
    writeGeneration(join(dir, "gen"), result);
    expect(readdirSync(join(dir, "gen", "episodes")).sort()).toEqual(result.episode_hashes.map((h) => `${h}.json`).sort());
    const read = (name: string): string => readFileSync(join(dir, "gen", name), "utf8");
    expect(read("stream.jsonl")).toBe(result.stream);
    expect(read("training-set.json")).toBe(`${jcs(result.set as unknown as Json)}\n`);
    const summary = JSON.parse(read("generation.json")) as Record<string, unknown>;
    expect(summary).toMatchObject({ schema_version: 1, generation: 2, forfeits: 0, training_set_hash: result.training_set_hash, examples: result.set.examples });
    expect(read("generation.json")).toBe(`${jcs(summary as Json)}\n`);
    expect(JSON.parse(read("plan.json"))).toHaveLength(6);
  }, 120_000);
});
