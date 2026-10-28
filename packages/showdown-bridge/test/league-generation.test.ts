import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { actorRule } from "../src/actor-rule.ts";
import { jcs, type Json } from "../src/canonical-json.ts";
import { readTrainingRunConfig } from "../src/training-run-config.ts";
import {
  DAMAGE_AWARE_V2_POLICY_ID,
  damageAwareSetup,
  damageAwareSetupV2,
  DAMAGE_AWARE_V3_POLICY_ID,
  forfeitCauses,
  GenerationError,
  type GenerationOptions,
  type GenerationResult,
  planGeneration,
  playEntries,
  playGeneration,
  trainingTeamPool,
  type TrainingPool,
  writeGeneration,
} from "../src/league-generation.ts";
import { checkLeagueRunRecord, LEAGUE_SEED_DERIVATION, leagueBattleSeed, loadEngine, readLeagueRunRecord } from "../src/index.ts";
import { type PolicyClient, PolicyWorkerError } from "../src/index.ts";
import { UnsupportedCandidateError } from "../src/evaluation-harness.ts";
import { loadTrainingSet, readTrainingSet, TrainingLoaderError } from "../src/training-loader.ts";
import { SAMPLES_DIR } from "../src/samples.ts";
import { startShard } from "../src/generation-cli.ts";
import { teamHash } from "../src/team-class.ts";
import { fakePort } from "./damage-aware-world-support.ts";

const engine = loadEngine();
const at = <T>(items: readonly T[], index: number): T => items[index] as T;
/** The stream of a result played without a `stream_path`. */
const memoryStream = (result: GenerationResult): string => {
  if (result.stream === null) throw new Error("the stream was written to a file");
  return result.stream;
};
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
  });

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
    expect(readTrainingSet(result.set).examples).toBe(memoryStream(result).split("\n").length - 1);
    // a baseline game names the policy as the baseline's opponent and the baseline as the policy's
    const baselineGame = result.episodes.slice(8);
    const sides = baselineGame.map((e) => e.provenance.collection as { opponent_policy_id: string });
    expect(new Set(sides.map((c) => c.opponent_policy_id))).toEqual(new Set([POLICY, "random-legal@tiny"]));
  });

  it("always samples at the frozen pair's own seats, never argmax, and no baseline decision is queried or loaded", async () => {
    const inner = actorClient();
    const asked: { battle: string; kind: string; policy: string }[] = [];
    const spying: PolicyClient & { beginGame: () => Promise<void> } = {
      beginGame: inner.beginGame,
      ask: (battle, query) => {
        asked.push({ battle, kind: query.sampling.kind, policy: query.policy_id });
        return inner.ask(battle, query);
      },
    };
    const run2 = await run({ client: spying });
    expect(asked.length).toBeGreaterThan(0);
    expect(new Set(asked.map((q) => q.kind))).toEqual(new Set(["sample"]));
    expect(new Set(asked.map((q) => q.policy))).toEqual(new Set([POLICY]));
    // The worker is never asked for a seat a baseline plays: game 4 puts the policy at p1, game 5 at p2.
    const seats = new Set(asked.map((q) => q.battle));
    expect(seats.has("game-4-p1") && !seats.has("game-4-p2")).toBe(true);
    expect(seats.has("game-5-p2") && !seats.has("game-5-p1")).toBe(true);
    // Every line of the stream belongs to an Episode of the pair; none to a baseline's.
    const baselineHashes = run2.episode_hashes.filter((_, i) => at(run2.episodes, i).provenance.behaviour_policy_id !== POLICY);
    expect(baselineHashes).toHaveLength(2);
    for (const hash of baselineHashes) expect(memoryStream(run2).includes(`"episode_hash":"${hash}"`)).toBe(false);
    const hashesInStream = new Set(memoryStream(run2).split("\n").filter(Boolean).map((line) => (JSON.parse(line) as { episode_hash: string }).episode_hash));
    expect(hashesInStream).toEqual(new Set(run2.set.episode_hashes));
  });

  it("is reproducible: the same plan, policy and worker give the same stream, hash and teams", async () => {
    const again = await run();
    expect(again.stream).toBe(result.stream);
    expect(again.training_set_hash).toBe(result.training_set_hash);
    expect(again.episode_hashes).toEqual(result.episode_hashes);
    expect(again.team_hashes).toEqual(result.team_hashes);
  });

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
  });

  it("fails the generation on a harness incident: a candidate shape the worker cannot read is not a policy forfeit", async () => {
    const unsupported: PolicyClient = { ask: () => Promise.reject(new UnsupportedCandidateError("a candidate has a shape the worker does not know")) };
    await expect(run({ client: unsupported })).rejects.toThrow(/game 0 \(sodium,[0-9a-f]{32}\): unsupported_candidate/);
  });

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
    const forfeited = partial.games.filter((g) => g.outcome === "forfeit");
    expect(forfeited[0]?.forfeit).toMatchObject({ cause: "crash", detail: expect.stringContaining("gone") });
    expect(forfeitCauses(partial.games)).toEqual({ crash: 1 });
    expect(partial.games.filter((g) => g.outcome === "played").every((g) => !("forfeit" in g))).toBe(true);
    const dir = mkdtempSync(join(tmpdir(), "forfeit-reason-"));
    writeGeneration(dir, partial);
    const summary = JSON.parse(readFileSync(join(dir, "generation.json"), "utf8"));
    const written = readLeagueRunRecord(JSON.parse(readFileSync(join(dir, "league-run.json"), "utf8")));
    rmSync(dir, { recursive: true, force: true });
    expect(summary).toMatchObject({ forfeits: 1, forfeit_causes: { crash: 1 } });
    // The record's totals are the summary's counts, as a sorted list.
    expect(written.forfeit_causes).toEqual(Object.entries(summary.forfeit_causes).map(([cause, count]) => ({ cause, count })));
    expect(partial.run_record.forfeit_causes).toEqual([{ cause: "crash", count: 1 }]);
  });
});

/** The stream lines of one Episode: what the learner sees of that seat's battle. */
const linesOf = (result: GenerationResult, hash: string): string =>
  memoryStream(result)
    .split("\n")
    .filter((line) => line.includes(`"episode_hash":"${hash}"`))
    .join("\n");

describe("league seeds in a generation", () => {
  it("makes every battle seed, and every Episode's recorded seed, from the run key and the game index", async () => {
    const result = await playGeneration({ engine, config, generation: 2, policy_id: POLICY, client: actorClient(), pool });
    for (const entry of result.plan) expect(entry.seed).toBe(leagueBattleSeed(config.seeds.run_key, 2, entry.game));
    const recorded = new Set(result.episodes.map((episode) => (episode.provenance as { collection: { seed: string } }).collection.seed));
    expect(recorded).toEqual(new Set(result.plan.map((entry) => entry.seed)));
  });
});

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
    expect(summary).not.toHaveProperty("forfeit_causes");
    expect(result.run_record).not.toHaveProperty("forfeit_causes");
    expect(summary).toMatchObject({ schema_version: 1, generation: 2, forfeits: 0, training_set_hash: result.training_set_hash, examples: result.set.examples });
    expect(read("generation.json")).toBe(`${jcs(summary as Json)}\n`);
    expect(JSON.parse(read("plan.json"))).toHaveLength(6);
    const written = readLeagueRunRecord(JSON.parse(read("league-run.json")));
    expect(written).toEqual(result.run_record);
    expect(read("league-run.json")).toBe(`${jcs(written as unknown as Json)}\n`);
    expect(written).toMatchObject({ generation: 2, pair_policy_id: POLICY, games: 6, seed_derivation: LEAGUE_SEED_DERIVATION });
    await checkLeagueRunRecord(written, config);
  });
});

describe("a stream over the string limit (#998)", () => {
  let dir: string;
  beforeAll(() => {
    dir = mkdtempSync(join(tmpdir(), "areneon-chunked-"));
  });
  afterAll(() => rmSync(dir, { recursive: true, force: true }));
  const opts = (change: Partial<GenerationOptions> = {}): GenerationOptions => ({ engine, config, generation: 1, policy_id: POLICY, client: actorClient(), pool, ...change });
  it("is written in chunks with no joined string, to the same bytes, TrainingSet and files as the in-memory writer", async () => {
    const reference = await playGeneration(opts());
    const text = memoryStream(reference);
    const longest = Math.max(...text.split("\n").map((line) => line.length + 1));
    expect(text.length).toBeGreaterThan(longest * 3); // the stream is several times the limit below
    const limits = { chunk_chars: longest, max_string_chars: longest * 2 };
    // The in-memory form needs one string for the stream and refuses at the limit ...
    await expect(playGeneration(opts({ stream_limits: limits }))).rejects.toThrow(/write it with writeTrainingStream/);
    // ... the file form never makes one.
    const out = join(dir, "chunked");
    mkdirSync(out, { recursive: true });
    const chunked = await playGeneration(opts({ stream_path: join(out, "stream.jsonl"), stream_limits: limits }));
    expect(chunked.stream).toBeNull();
    expect(chunked.set).toEqual(reference.set);
    expect(chunked.training_set_hash).toBe(reference.training_set_hash);
    writeGeneration(out, chunked);
    writeGeneration(join(dir, "inmemory"), reference);
    const files = (root: string): Record<string, string> =>
      Object.fromEntries(
        ["stream.jsonl", "training-set.json", "plan.json", "league-run.json", "generation.json"].map((name) => [name, readFileSync(join(root, name), "utf8")]),
      );
    expect(files(out)).toEqual(files(join(dir, "inmemory")));
    expect(files(out)["stream.jsonl"]).toBe(text);
    // A stream written elsewhere than where writeGeneration looks is refused.
    expect(() => writeGeneration(join(dir, "elsewhere"), { ...chunked, stream_path: join(out, "other.jsonl") })).toThrow(GenerationError);
  });
});

describe("Episodes kept on disk as their games finish (#1069)", () => {
  let dir: string;
  beforeAll(() => {
    dir = mkdtempSync(join(tmpdir(), "areneon-spill-"));
  });
  afterAll(() => rmSync(dir, { recursive: true, force: true }));
  const opts = (change: Partial<GenerationOptions> = {}): GenerationOptions => ({ engine, config, generation: 1, policy_id: POLICY, client: actorClient(), pool, ...change });
  const names = ["stream.jsonl", "training-set.json", "plan.json", "league-run.json", "generation.json"];
  const tree = (root: string): Record<string, string> =>
    Object.fromEntries([...names.map((n) => `${n}`), ...readdirSync(join(root, "episodes")).map((n) => `episodes/${n}`)].map((n) => [n, readFileSync(join(root, n), "utf8")]));

  it("writes the files of the in-memory path, byte for byte, and holds no Episode", async () => {
    const reference = join(dir, "reference");
    mkdirSync(reference, { recursive: true });
    const inMemory = await playGeneration(opts({ stream_path: join(reference, "stream.jsonl") }));
    writeGeneration(reference, inMemory);
    const out = join(dir, "spilled");
    mkdirSync(join(out, "episodes"), { recursive: true });
    const spilled = await playGeneration(opts({ stream_path: join(out, "stream.jsonl"), episodes_dir: join(out, "episodes") }));
    expect(spilled.episodes).toEqual([]);
    expect(spilled.episodes_dir).toBe(join(out, "episodes"));
    expect(spilled.episode_hashes).toEqual(inMemory.episode_hashes);
    expect(spilled.set).toEqual(inMemory.set);
    expect(spilled.run_record).toEqual(inMemory.run_record);
    writeGeneration(out, spilled);
    expect(tree(out)).toEqual(tree(reference));
  });

  it("gives the same files when game processes each play a slice into one directory", async () => {
    const reference = join(dir, "ref2");
    mkdirSync(reference, { recursive: true });
    writeGeneration(reference, await playGeneration(opts({ stream_path: join(reference, "stream.jsonl") })));
    const out = join(dir, "sliced");
    const episodes = join(out, "episodes");
    mkdirSync(episodes, { recursive: true });
    const of = 2;
    const there = playEntries(opts({ episodes_dir: episodes }), { index: 1, of });
    const result = await playGeneration(opts({ stream_path: join(out, "stream.jsonl"), episodes_dir: episodes, shard: { index: 0, of }, others: there }));
    writeGeneration(out, result);
    expect(tree(out)).toEqual(tree(reference));
    // An entry that went through a shard file keeps its references.
    const entries = await there;
    expect(JSON.parse(jcs(entries as unknown as Json))).toEqual(entries);
    expect(entries.every((entry) => entry.episodes.length === 0 && (entry.stored ?? []).length > 0)).toBe(true);
  });

  it("keeps a forfeited game like an in-memory one: the same files, the forfeit row and cause (#1184)", async () => {
    // Game 1 is forfeited by its policy seat; a game's battle name is `game-<n>-<player>`.
    const forfeiting = (): PolicyClient & { beginGame: () => Promise<void> } => {
      const good = actorClient();
      return { beginGame: good.beginGame, ask: (battle, query) => (battle.startsWith("game-1-") ? Promise.reject(new PolicyWorkerError("crash", "gone")) : good.ask(battle, query)) };
    };
    const reference = join(dir, "forfeit-ref");
    mkdirSync(reference, { recursive: true });
    const inMemory = await playGeneration(opts({ client: forfeiting(), stream_path: join(reference, "stream.jsonl") }));
    writeGeneration(reference, inMemory);
    expect(forfeitCauses(inMemory.games)).toEqual({ crash: 1 });
    const out = join(dir, "forfeit-spilled");
    mkdirSync(join(out, "episodes"), { recursive: true });
    const spilled = await playGeneration(opts({ client: forfeiting(), stream_path: join(out, "stream.jsonl"), episodes_dir: join(out, "episodes") }));
    expect(spilled.games).toEqual(inMemory.games);
    expect(spilled.run_record).toEqual(inMemory.run_record);
    writeGeneration(out, spilled);
    expect(tree(out)).toEqual(tree(reference));
    // The same with a second game process playing the odd games.
    const sliced = join(dir, "forfeit-sliced");
    const episodes = join(sliced, "episodes");
    mkdirSync(episodes, { recursive: true });
    const of = 2;
    const there = playEntries(opts({ client: forfeiting(), episodes_dir: episodes }), { index: 1, of });
    const result = await playGeneration(opts({ client: forfeiting(), stream_path: join(sliced, "stream.jsonl"), episodes_dir: episodes, shard: { index: 0, of }, others: there }));
    writeGeneration(sliced, result);
    expect(tree(sliced)).toEqual(tree(reference));
    const forfeit = (await there).find((entry) => entry.game === 1);
    expect(forfeit).toMatchObject({ episodes: [], stored: [] });
    expect(JSON.parse(jcs((await there) as unknown as Json))).toEqual(await there);
  });

  it("refuses a directory the files are not in, a missing stream path and entries kept the other way", async () => {
    const out = join(dir, "refusals");
    const episodes = join(out, "episodes");
    mkdirSync(episodes, { recursive: true });
    await expect(playGeneration(opts({ episodes_dir: episodes }))).rejects.toThrow(/needs stream_path/);
    const result = await playGeneration(opts({ stream_path: join(out, "stream.jsonl"), episodes_dir: episodes }));
    expect(() => writeGeneration(join(dir, "elsewhere"), result)).toThrow(/not to/);
    const kept = await playEntries(opts());
    await expect(playGeneration(opts({ stream_path: join(out, "s2.jsonl"), episodes_dir: episodes, others: Promise.resolve(kept.filter((e) => e.game % 2 === 1)), shard: { index: 0, of: 2 } }))).rejects.toThrow(/kept the way/);
  });
});

describe("several workers and game processes (#957)", () => {
  const opts = (change: Partial<GenerationOptions> = {}): GenerationOptions => ({ engine, config, generation: 1, policy_id: POLICY, client: actorClient(), pool, ...change });
  let single: GenerationResult;
  beforeAll(async () => {
    single = await playGeneration(opts());
  });
  /** What the files of a result say, in the bytes the loop commits. */
  const bytes = (result: GenerationResult): string[] => [memoryStream(result), jcs(result.set as unknown as Json), jcs(result.games as unknown as Json), ...result.episode_hashes, result.training_set_hash, ...result.team_hashes];

  it("gives the same generation with any number of workers, each game staying on one of them", async () => {
    const workers = [actorClient(), actorClient(), actorClient()];
    const seen = new Map<string, Set<number>>();
    const spy = (worker: ReturnType<typeof actorClient>, id: number): PolicyClient & { beginGame: () => Promise<void> } => ({
      beginGame: worker.beginGame,
      ask: (battle, query) => {
        seen.set(battle, (seen.get(battle) ?? new Set<number>()).add(id));
        return worker.ask(battle, query);
      },
    });
    const many = await playGeneration(opts({ client: spy(workers[0] as never, 0), peers: [spy(workers[1] as never, 1), spy(workers[2] as never, 2)] }));
    expect(bytes(many)).toEqual(bytes(single));
    expect(many.episodes).toEqual(single.episodes);
    expect(workers.map((w) => w.asked > 0)).toEqual([true, true, true]);
    expect(seen.size).toBeGreaterThan(0);
    for (const ids of seen.values()) expect(ids.size).toBe(1);
  });

  it("gives the same generation when game processes each play a slice, in whatever order they finish", async () => {
    const of = 3;
    const slice = (index: number): Promise<Awaited<ReturnType<typeof playEntries>>> => playEntries(opts({ peers: [actorClient()] }), { index, of });
    const slow = slice(2).then(async (entries) => {
      await new Promise((resolve) => setTimeout(resolve, 20));
      return entries;
    });
    const split = await playGeneration(opts({ shard: { index: 0, of }, others: Promise.all([slice(1), slow]).then((all) => all.flat()) }));
    expect(bytes(split)).toEqual(bytes(single));
    expect(split.run_record).toEqual(single.run_record);
    expect(split.episodes).toEqual(single.episodes);
    // A slice holds exactly its games (game mod of = index), and its entries survive a trip through JSON.
    const entries = await playEntries(opts(), { index: 1, of });
    expect(entries.map((e) => e.game)).toEqual([1, 4]);
    expect(JSON.parse(jcs(entries as unknown as Json))).toEqual(entries);
  });

  it("refuses a slice that is not an index below a count, a game that is missing or twice, and another process's failure", async () => {
    await expect(playEntries(opts(), { index: 3, of: 3 })).rejects.toThrow(RangeError);
    await expect(playEntries(opts(), { index: 0, of: 0 })).rejects.toThrow(RangeError);
    const all = await playEntries(opts());
    await expect(playGeneration(opts({ shard: { index: 0, of: 2 }, others: Promise.resolve(all.filter((e) => e.game % 2 === 1).slice(1)) }))).rejects.toThrow(/each once/);
    await expect(playGeneration(opts({ others: Promise.resolve(all.slice(0, 1)) }))).rejects.toThrow(/each once/);
    await expect(playGeneration(opts({ shard: { index: 0, of: 2 }, others: Promise.reject(new GenerationError("game process 1 failed")) }))).rejects.toThrow(/game process 1 failed/);
  });
});

describe("the damage-aware baseline as a training opponent (ADR-018 section 7)", () => {
  const damageConfig = (policyId = "damage-aware@v1") =>
    readTrainingRunConfig({
      ...(sample("tiny") as object),
      generation: { ...config.generation, games: 2, baseline_games: [{ actor: "damage-aware", policy_id: policyId, games: 2 }] },
    });
  const play = (cfg = damageConfig(), client: PolicyClient & { beginGame?: () => Promise<void> } = actorClient()) =>
    playGeneration({ engine, config: cfg, generation: 1, policy_id: POLICY, client, pool });
  let first: GenerationResult;
  const asked: { battle: string; policy: string }[] = [];
  beforeAll(async () => {
    const inner = actorClient();
    first = await play(damageConfig(), {
      beginGame: inner.beginGame,
      ask: (battle, query) => {
        asked.push({ battle, policy: query.policy_id });
        return inner.ask(battle, query);
      },
    });
  }, 120_000);

  it("plans the baseline seat with the config's actor and a seed in bytes 24 to 27 of league-seed/1", () => {
    const plan = planGeneration(damageConfig(), 1, pool.teams.length);
    expect(plan.slice(2).map((g) => g.opponent)).toEqual(Array(2).fill({ actor: "damage-aware", policy_id: "damage-aware@v1" }));
    expect(plan.slice(2).map((g) => g.baseline_seed)).toEqual(planGeneration(config, 1, pool.teams.length).slice(2, 4).map((g) => g.baseline_seed));
  });

  it("plays the games, records the identity hash and the summed counts, and ties them to the opponent entry", async () => {
    expect(first.games.every((g) => g.outcome === "played")).toBe(true);
    const record = readLeagueRunRecord(first.run_record);
    expect(record.opponents).toEqual([
      { opponent_policy_id: "damage-aware@v1", games: 2 },
      { opponent_policy_id: POLICY, games: 2 },
    ]);
    expect(record.baselines).toHaveLength(1);
    const entry = at(record.baselines ?? [], 0);
    expect(entry).toMatchObject({ policy_id: "damage-aware@v1", games: 2 });
    expect(entry.identity_hash).toMatch(/^[0-9a-f]{64}$/);
    expect(entry.counts.decisions).toBeGreaterThan(0);
    expect(entry.counts.preview_decisions).toBe(2);
    expect(entry.counts.epsilon_plays).toBe(0);
    await expect(checkLeagueRunRecord(record, damageConfig())).resolves.toBeUndefined();
  });

  it("never queries the worker for the baseline's seat, and never trains on its Episodes", async () => {
    const seats = new Set(asked.map((q) => q.battle));
    expect(seats.has("game-2-p1") && !seats.has("game-2-p2")).toBe(true);
    expect(seats.has("game-3-p2") && !seats.has("game-3-p1")).toBe(true);
    expect(new Set(asked.map((q) => q.policy))).toEqual(new Set([POLICY]));
    const baseline = first.episodes.filter((e) => e.provenance.behaviour_policy_id === "damage-aware@v1");
    expect(baseline).toHaveLength(2);
    for (const [index, hash] of first.episode_hashes.entries()) {
      if (at(first.episodes, index).provenance.behaviour_policy_id !== "damage-aware@v1") continue;
      expect(first.set.episode_hashes).not.toContain(hash);
      expect(memoryStream(first).includes(`"episode_hash":"${hash}"`)).toBe(false);
    }
    // The loader itself refuses the baseline's Episodes as the pair's training data.
    await expect(
      loadTrainingSet({ episodes: baseline, evaluation_manifests: [], known_synthetic: pool.known_synthetic, behaviour_policy_id: POLICY }),
    ).rejects.toThrow(TrainingLoaderError);
  });

  it("is reproducible: a second run gives the same stream, hashes and run record", async () => {
    const again = await play();
    expect(again.stream).toBe(first.stream);
    expect(again.episode_hashes).toEqual(first.episode_hashes);
    expect(again.run_record).toEqual(first.run_record);
  }, 120_000);

  it("keeps a run without it unchanged: no baselines field, no baseline on an entry", async () => {
    const plain = await playEntries({ engine, config, generation: 1, policy_id: POLICY, client: actorClient(), pool }, { index: 0, of: 3 });
    expect(plain.every((entry) => !("baseline" in entry))).toBe(true);
  });

  it("refuses a config whose policy id is not the one the baseline plays as, and a format without a vocabulary", async () => {
    await expect(play(damageConfig("damage-aware-x@v1"))).rejects.toThrow(/plays as damage-aware@v1/);
    expect(() => damageAwareSetup(engine, "gen9other")).toThrow(/vocabulary for/);
  }, 120_000);
});

describe("damage-aware@v3 as a training opponent (#1183)", () => {
  const v3Config = () =>
    readTrainingRunConfig({
      ...(sample("tiny") as object),
      generation: { ...config.generation, games: 2, baseline_games: [{ actor: "damage-aware", policy_id: DAMAGE_AWARE_V3_POLICY_ID, games: 2 }] },
    });
  const play = (cfg = v3Config(), extra: Partial<Parameters<typeof playGeneration>[0]> = {}) =>
    playGeneration({ engine, config: cfg, generation: 1, policy_id: POLICY, client: actorClient(), pool, ...extra });

  it("plays the v3 record: the opponent entry, the identity and the counts as for v1, byte-identical on a rerun", async () => {
    const first = await play();
    const again = await play();
    expect(again.stream).toBe(first.stream);
    expect(again.run_record).toEqual(first.run_record);
    const record = readLeagueRunRecord(first.run_record);
    expect(record.opponents).toEqual([
      { opponent_policy_id: DAMAGE_AWARE_V3_POLICY_ID, games: 2 },
      { opponent_policy_id: POLICY, games: 2 },
    ]);
    const entry = at(record.baselines ?? [], 0);
    expect(entry).toMatchObject({ policy_id: DAMAGE_AWARE_V3_POLICY_ID, games: 2 });
    expect(entry.counts.decisions).toBeGreaterThan(0);
    await expect(checkLeagueRunRecord(record, v3Config())).resolves.toBeUndefined();
    // Its identity is not v1's: another record (the engine's choice-location convention) gives another baseline identity.
    const v1 = await play(readTrainingRunConfig({ ...(sample("tiny") as object), generation: { ...config.generation, games: 2, baseline_games: [{ actor: "damage-aware", policy_id: "damage-aware@v1", games: 2 }] } }));
    expect(entry.identity_hash).not.toBe(at(readLeagueRunRecord(v1.run_record).baselines ?? [], 0).identity_hash);
  }, 300_000);

  it("never hands a v1 record to a config that names v3", async () => {
    await expect(play(v3Config(), { damage_aware_v3: damageAwareSetup(engine, "gen9championsvgc2026regmc") })).rejects.toThrow(/names another record version/);
  }, 120_000);
});

describe("damage-aware@v2 as a training opponent (ADR-025 section 6)", () => {
  const v2Config = () =>
    readTrainingRunConfig({
      ...(sample("tiny") as object),
      generation: { ...config.generation, games: 2, baseline_games: [{ actor: "damage-aware", policy_id: DAMAGE_AWARE_V2_POLICY_ID, games: 2 }] },
    });
  // the factory is the composition root's; this one builds the port from the record it is given
  const factory = (context: { config: { assumed_set: { prior_hash: string } } }) => fakePort(undefined, { prior_hash: context.config.assumed_set.prior_hash });

  it("builds the record's setup through the factory the composition root supplies, for the committed format only", () => {
    const setup = damageAwareSetupV2(engine, "gen9championsvgc2026regmc", factory as never);
    expect(setup.config.schema_version).toBe(2);
    expect(setup.worlds?.identity_hash).toBe("b".repeat(64));
    expect(() => damageAwareSetupV2(engine, "gen9other", factory as never)).toThrow(/vocabulary for/);
  });

  it("refuses a plan that names it when no belief port is supplied (the bridge does not import the opponent model)", async () => {
    await expect(playEntries({ engine, config: v2Config(), generation: 1, policy_id: POLICY, client: actorClient(), pool })).rejects.toThrow(/belief port of the composition root/);
  });

  it("plays it as the baseline seat: its policy id and identity (with the belief) are recorded, and the pair is never queried for its seat", async () => {
    const asked: string[] = [];
    const inner = actorClient();
    const result = await playGeneration({
      engine,
      config: v2Config(),
      generation: 1,
      policy_id: POLICY,
      client: { beginGame: inner.beginGame, ask: (battle, query) => (asked.push(battle), inner.ask(battle, query)) },
      pool,
      belief_worlds: factory as never,
    });
    const record = readLeagueRunRecord(result.run_record);
    expect(record.opponents).toEqual([
      { opponent_policy_id: DAMAGE_AWARE_V2_POLICY_ID, games: 2 },
      { opponent_policy_id: POLICY, games: 2 },
    ]);
    expect(record.baselines).toHaveLength(1);
    expect(at(record.baselines ?? [], 0)).toMatchObject({ policy_id: DAMAGE_AWARE_V2_POLICY_ID, games: 2 });
    // v1 and v2 are different baselines: the identity hash differs from v1's for the same pin
    expect(at(record.baselines ?? [], 0).identity_hash).not.toBe("d3595137164c7e6bf31c17431fa61e9f7d837cd708ad9daf239f3043c2334596");
    await expect(checkLeagueRunRecord(record, v2Config())).resolves.toBeUndefined();
    // the learned pair's seat is the other one in each baseline game
    const seats = new Set(asked);
    expect(seats.has("game-2-p1") && !seats.has("game-2-p2")).toBe(true);
    expect(seats.has("game-3-p2") && !seats.has("game-3-p1")).toBe(true);
  }, 120_000);

  it("refuses a config that names v2 when the records it is given are v1's: a v1 record never plays as damage-aware@v2", async () => {
    await expect(
      playEntries({ engine, config: v2Config(), generation: 1, policy_id: POLICY, client: actorClient(), pool, damage_aware_v2: damageAwareSetup(engine, "gen9championsvgc2026regmc") }),
    ).rejects.toThrow(/another record version/);
  }, 120_000);
});

describe("a failing or vanished game process (#1100)", () => {
  const serial = readTrainingRunConfig({ ...(sample("tiny") as object), generation: { ...(readTrainingRunConfig(sample("tiny")).generation as object), n_parallel: 1 } });
  const slow = (waited: { count: number }, ms: number): PolicyClient & { beginGame: () => Promise<void> } => ({
    ...actorClient(),
    beginGame: async () => {
      waited.count++;
      await new Promise((resolve) => setTimeout(resolve, ms));
    },
  });
  const opts = (change: Partial<GenerationOptions>): GenerationOptions => ({ engine, config: serial, generation: 1, policy_id: POLICY, client: actorClient(), pool, ...change });
  const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));
  const script = (name: string, body: string): string => {
    const file = join(dir, name);
    writeFileSync(file, body);
    return file;
  };
  let dir: string;
  beforeAll(() => {
    dir = mkdtempSync(join(tmpdir(), "areneon-fail-"));
  });
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  it("stops this process's own games after the current wave and reports the child's message", async () => {
    const waves = Math.ceil(planGeneration(serial, 1, pool.teams.length).length / 2); // this process's share of two
    expect(waves).toBeGreaterThan(2);
    const waited = { count: 0 };
    const child = startShard(script("fails.mjs", `await new Promise((r) => setTimeout(r, 100)); console.error("child said no"); process.exit(1);`), [], join(dir, "none"), 2);
    const started = Date.now();
    await expect(playGeneration(opts({ client: slow(waited, 150), shard: { index: 0, of: 2 }, others: child.entries }))).rejects.toThrow(/game process 2 failed \(exit 1\): child said no/);
    expect(Date.now() - started).toBeLessThan(5000); // bounded; the wave count below is the exact check
    const after = waited.count;
    await sleep(400);
    expect(waited.count).toBe(after); // no wave began after the failure
    expect(after).toBeLessThan(waves);
  });

  it("is told of a child that failed although a grandchild keeps its stderr open", async () => {
    const holder = script("holder.mjs", `import { spawn } from "node:child_process"; spawn(process.execPath, ["-e", "setTimeout(() => {}, 4000)"], { stdio: ["ignore", "ignore", "inherit"], detached: true }).unref(); console.error("left a grandchild"); process.exit(1);`);
    const started = Date.now();
    await expect(startShard(holder, [], join(dir, "none"), 3).entries).rejects.toThrow(/game process 3 failed \(exit 1\): left a grandchild/);
    expect(Date.now() - started).toBeLessThan(2500);
  });

  it("waits normally when its own games finish while a child still runs", async () => {
    const all = await playEntries(opts({}));
    const late = sleep(300).then(() => all.filter((e) => e.game % 2 === 1));
    const result = await playGeneration(opts({ shard: { index: 0, of: 2 }, others: late, live_children: () => 1, idle_grace_ms: 40 }));
    expect(result.games).toHaveLength(all.length);
  });

  it("fails a generation that has no live game process and no game in flight but still awaits one", async () => {
    const started = Date.now();
    await expect(playGeneration(opts({ shard: { index: 0, of: 2 }, others: new Promise(() => undefined), live_children: () => 0, idle_grace_ms: 80 }))).rejects.toThrow(/generation is idle/);
    expect(Date.now() - started).toBeLessThan(5000);
  });

  it("does not trip while this process's own games run", async () => {
    const waited = { count: 0 };
    const all = await playEntries(opts({}));
    const others = sleep(200).then(() => all.filter((e) => e.game % 2 === 1));
    await expect(playGeneration(opts({ client: slow(waited, 60), shard: { index: 0, of: 2 }, others, live_children: () => 0, idle_grace_ms: 150 }))).resolves.toBeDefined();
  });
});
