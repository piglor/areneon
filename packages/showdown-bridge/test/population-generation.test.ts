import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { actorRule } from "../src/actor-rule.ts";
import { checkLeagueRunRecord, loadEngine, readLeagueRunRecordV2 } from "../src/index.ts";
import { GenerationError, trainingTeamPool, type TrainingPool } from "../src/league-generation.ts";
import { checkPopulationConfig, memberDir, membersPlaying, playPopulationEntries, playPopulationGeneration, type PopulationEntry, type PopulationMember, populationWaves, writePopulationGeneration, POPULATION_SHEET_REFUSAL } from "../src/population-generation.ts";
import { readShardFile } from "../src/shard-process.ts";
import type { ShardProcess } from "../src/shard-process.ts";
import { parseMemberFlag, runPopulationCli } from "../src/population-cli.ts";
import { planPopulation } from "../src/population-plan.ts";
import { emptyPastPool } from "../src/past-pool.ts";
import { type PolicyClient, PolicyWorkerError } from "../src/index.ts";
import { UnsupportedCandidateError } from "../src/evaluation-harness.ts";
import { loadTrainingSet, TrainingLoaderError } from "../src/training-loader.ts";
import { readTrainingRunConfig } from "../src/training-run-config.ts";
import { populationConfig } from "./population-support.ts";

const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const scratch = mkdtempSync(join(tmpdir(), "population-play-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));

/** An in-process worker: the shared actor rule over the query's own candidates. */
function actorClient(): PolicyClient & { beginGame: () => Promise<void>; asked: number; begun: number } {
  const client = {
    asked: 0,
    begun: 0,
    beginGame: () => {
      client.begun++;
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
  } satisfies PolicyClient & { beginGame: () => Promise<void>; asked: number; begun: number };
  return client;
}

const config = populationConfig(2, 2);
const POLICIES = [`policy:${"a".repeat(64)}`, `policy:${"b".repeat(64)}`];
const membersOf = (): (PopulationMember & { client: ReturnType<typeof actorClient> })[] =>
  POLICIES.map((policy_id, k) => ({ member: `m${k}`, policy_id, checkpoints: { preview: String(k).repeat(64), battle: String(k + 2).repeat(64) }, client: actorClient() }));

let pool: TrainingPool;
beforeAll(() => {
  pool = trainingTeamPool(engine, config.generation.excluded_abilities);
});

const treeBytes = (dir: string, prefix = ""): Record<string, string> => {
  const out: Record<string, string> = {};
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) Object.assign(out, treeBytes(path, `${prefix}${entry.name}/`));
    else out[`${prefix}${entry.name}`] = readFileSync(path, "utf8");
  }
  return out;
};

describe("a tiny population generation (M = 2, the stand-in worker)", () => {
  const dirs = [join(scratch, "a"), join(scratch, "b")];
  let members: ReturnType<typeof membersOf>;
  let result: Awaited<ReturnType<typeof playPopulationGeneration>>;
  beforeAll(async () => {
    members = membersOf();
    result = await playPopulationGeneration({ engine, config, generation: 1, members, pool, out_dir: dirs[0] as string });
    writePopulationGeneration(dirs[0] as string, result);
    const again = await playPopulationGeneration({ engine, config, generation: 1, members: membersOf(), pool, out_dir: dirs[1] as string });
    writePopulationGeneration(dirs[1] as string, again);
  }, 120_000);

  it("plays the whole plan, each game on its own members' clients", () => {
    const plan = planPopulation(config, 1, pool.teams.length);
    expect(result.plan).toEqual(plan);
    expect(result.games.map((g) => g.game)).toEqual(plan.map((g) => g.game));
    expect(members.every((m) => m.client.asked > 0 && m.client.begun > 0)).toBe(true);
    expect(result.games.every((g) => g.outcome === "played")).toBe(true);
    for (const row of result.games) {
      const entry = plan[row.game];
      expect(row.seats.p1).toBe("member" in (entry?.seats.p1 as object) ? (entry?.seats.p1 as { member: string }).member : (entry?.seats.p1 as { policy_id: string }).policy_id);
      expect(row.pair).toBe(entry?.pair);
    }
    expect(result.games.filter((g) => g.kind === "ladder").every((g) => g.opponent !== g.member && /^m\d$/.test(g.opponent))).toBe(true);
    expect(result.games.filter((g) => g.kind === "self").every((g) => g.opponent === g.member)).toBe(true);
    expect(result.games.filter((g) => g.kind === "anchor").every((g) => g.opponent === "random-legal@tiny")).toBe(true);
  });

  it("writes each member's Episodes, stream and TrainingSet under its own directory, and the top-level files, generation.json last", () => {
    for (const dir of dirs) {
      const files = Object.keys(treeBytes(dir));
      expect(files).toEqual(expect.arrayContaining(["plan.json", "league-run.json", "generation.json", "members/m0/stream.jsonl", "members/m1/training-set.json"]));
      expect(files.filter((f) => f.startsWith("members/m0/episodes/")).length).toBeGreaterThan(0);
    }
    for (const entry of result.members) {
      expect(entry.set.examples).toBeGreaterThan(0);
      const onDisk = readdirSync(join(memberDir(dirs[0] as string, entry.member), "episodes")).map((f) => f.replace(".json", "")).sort();
      expect(onDisk).toEqual(entry.episode_hashes);
    }
  });

  it("writes a version 2 record that passes its reader and ties to the config", async () => {
    const record = readLeagueRunRecordV2(JSON.parse(readFileSync(join(dirs[0] as string, "league-run.json"), "utf8")));
    await checkLeagueRunRecord(record, config);
    expect(record).toEqual(result.run_record);
    expect(record.games).toBe(result.games.length);
    expect(record.members.map((m) => m.pair_policy_id)).toEqual(POLICIES);
    // Counting rule: each member's games are those it holds a seat in; the ladder games are in both entries.
    const ladder = result.games.filter((g) => g.kind === "ladder").length;
    const selfGames = result.games.filter((g) => g.kind === "self").length;
    const anchor = result.games.filter((g) => g.kind === "anchor").length;
    expect(record.games).toBe(ladder + selfGames + anchor);
    expect(record.members.reduce((sum, m) => sum + m.games, 0)).toBe(2 * ladder + selfGames + anchor);
    expect(record.episodes).toBe(2 * record.played);
    expect(record.members.reduce((sum, m) => sum + m.episodes, 0)).toBe(2 * (ladder + selfGames) + anchor);
  });

  it("writes the files the Python readers refuse in fixtures/loop/population-play (the inspector's refusal tests read them)", () => {
    const fixtures = fileURLToPath(new URL("../../../fixtures/loop/population-play", import.meta.url));
    for (const name of ["plan.json", "generation.json", "league-run.json"]) {
      expect(readFileSync(join(dirs[0] as string, name), "utf8")).toBe(readFileSync(join(fixtures, name), "utf8"));
    }
  });

  it("is byte-identical on a rerun, every file of the tree", () => {
    const [a, b] = dirs.map((d) => treeBytes(d));
    expect(Object.keys(a as object)).toEqual(Object.keys(b as object));
    expect(a).toEqual(b);
  });

  it("gives Episodes to the member that played them: the loader accepts them for it and refuses them for another", async () => {
    const stored = (member: string) =>
      readdirSync(join(memberDir(dirs[0] as string, member), "episodes")).map((f) => JSON.parse(readFileSync(join(memberDir(dirs[0] as string, member), "episodes", f), "utf8")));
    for (const [own, other] of [[0, 1], [1, 0]] as const) {
      const episodes = stored(`m${own}`).filter((e) => e.provenance.behaviour_policy_id === POLICIES[own]);
      expect(episodes.length).toBeGreaterThan(0);
      const input = { episodes, evaluation_manifests: [], known_synthetic: pool.known_synthetic };
      const { set } = await loadTrainingSet({ ...input, behaviour_policy_id: POLICIES[own] as string });
      expect(set.behaviour_policy_id).toBe(POLICIES[own]);
      await expect(loadTrainingSet({ ...input, behaviour_policy_id: POLICIES[other] as string })).rejects.toThrow(TrainingLoaderError);
    }
    // Every Episode of a member's directory names a collection of this run and a seed of the plan.
    for (const episode of stored("m0")) {
      expect(episode.provenance.collection.kind).toBe("league");
      expect(episode.provenance.collection.run_id).toBe(config.seeds.run_key);
      expect(result.plan.some((g) => `${g.seed}` === episode.provenance.collection.seed)).toBe(true);
    }
  });

  it("trains each member's stream on its own seats only", async () => {
    // m0 holds a seat in its own pairs and in m1's ladder pairs against it; its set counts only its pair's examples.
    const first = JSON.parse(readFileSync(join(memberDir(dirs[0] as string, "m0"), "training-set.json"), "utf8"));
    expect(first.behaviour_policy_id).toBe(POLICIES[0]);
    expect(first.examples).toBe(result.members[0]?.set.examples);
    expect(result.members[0]?.training_set_hash).not.toBe(result.members[1]?.training_set_hash);
  });
});

describe("game processes and policy workers (#1333)", () => {
  const options = (dir: string, change: object = {}) => ({ engine, config, generation: 1, members: membersOf(), pool, out_dir: dir, ...change });
  const single = async (name: string): Promise<Record<string, string>> => {
    const dir = join(scratch, name);
    writePopulationGeneration(dir, await playPopulationGeneration(options(dir)));
    return treeBytes(dir);
  };

  it("gives the single process's files when a slice is played elsewhere and merged, whichever slices there are", async () => {
    const expected = await single("whole");
    for (const of of [2, 3]) {
      const dir = join(scratch, `sharded-${of}`);
      const others = (
        await Promise.all(Array.from({ length: of - 1 }, (_, k) => playPopulationEntries(options(join(scratch, `elsewhere-${of}-${k}`), { shared_dir: dir }), { index: k + 1, of })))
      ).flat();
      const result = await playPopulationGeneration(options(dir, { shard: { index: 0, of }, others: Promise.resolve(others) }));
      writePopulationGeneration(dir, result);
      expect(treeBytes(dir)).toEqual(expected);
    }
  }, 300_000);

  it("gives the same files with more workers per member, and every worker plays", async () => {
    const expected = await single("whole-again");
    const members = membersOf().map((entry) => ({ ...entry, peers: [actorClient(), actorClient()] }));
    const dir = join(scratch, "peers");
    writePopulationGeneration(dir, await playPopulationGeneration(options(dir, { members })));
    expect(treeBytes(dir)).toEqual(expected);
    for (const entry of members) expect(entry.peers.every((peer) => peer.asked > 0 && peer.begun > 0)).toBe(true);
  }, 300_000);

  it("refuses games that are missing, a failed other process, a slice it cannot place, and goes idle with no process left", async () => {
    await expect(playPopulationGeneration(options(join(scratch, "short"), { shard: { index: 0, of: 2 }, others: Promise.resolve([]) }))).rejects.toThrow(/each once/);
    await expect(playPopulationGeneration(options(join(scratch, "failed"), { shard: { index: 0, of: 2 }, others: Promise.reject(new GenerationError("game process 1 failed")) }))).rejects.toThrow(/game process 1 failed/);
    await expect(playPopulationEntries(options(join(scratch, "slice")), { index: 2, of: 2 })).rejects.toThrow(RangeError);
    const idle = playPopulationGeneration(options(join(scratch, "idle"), { shard: { index: 0, of: 2 }, others: new Promise<PopulationEntry[]>(() => undefined), live_children: () => 0, idle_grace_ms: 20 }));
    await expect(idle).rejects.toThrow(/the generation is idle/);
  }, 300_000);

  it("ends this process's games when the signal it was given aborts while it plays", async () => {
    const stop = new AbortController();
    const members = membersOf();
    const real = (members[0] as ReturnType<typeof membersOf>[number]).client.ask.bind((members[0] as ReturnType<typeof membersOf>[number]).client);
    (members[0] as ReturnType<typeof membersOf>[number]).client.ask = (battle, query) => (stop.abort(), real(battle, query));
    await expect(playPopulationGeneration(options(join(scratch, "aborted"), { members, signal: stop.signal }))).rejects.toThrow(/was stopped/);
  }, 300_000);

  it("names the members that hold a seat in a slice, and gives a member with peers more games a wave", () => {
    const plan = planPopulation(config, 1, pool.teams.length);
    expect([...membersPlaying(plan)].sort()).toEqual(["m0", "m1"]);
    expect(membersPlaying(plan, { index: 99, of: 100 }).size).toBeLessThanOrEqual(2);
    const wide = populationWaves(plan, 1, () => 3);
    const narrow = populationWaves(plan, 1);
    expect(wide.length).toBeLessThan(narrow.length);
    expect(wide.flat().map((g) => g.game).sort((a, b) => a - b)).toEqual(plan.map((g) => g.game));
  });

  const input = (out: string, change: object = {}) => ({
    engine, config, generation: 1, pool, out, run_dir: scratch, worker_python: "unused", engine_data: "unused", engine_data_hash: "e".repeat(64), encoder_version: "enc1:x",
    members: POLICIES.map((policy_id, k) => ({ member: `m${k}`, preview: String(k).repeat(64), battle: String(k + 2).repeat(64), policy_id })),
    make_client: () => Object.assign(actorClient(), { start: () => Promise.resolve(), close: () => Promise.resolve() }),
    ...change,
  });

  it("starts further game processes with the flags they need, merges their slices and prints the single process's line", async () => {
    const lines: string[] = [];
    const out = join(scratch, "cli-whole");
    expect(await runPopulationCli(input(out), (line) => lines.push(line))).toBe(0);
    const started: { args: readonly string[]; stopped: boolean; told: string[] }[] = [];
    const shared = join(scratch, "cli-two");
    const start = (_script: string, args: readonly string[], _out: string, index: number): ShardProcess<PopulationEntry> => {
      const record = { args, stopped: false, told: [] as string[] };
      started.push(record);
      return {
        entries: playPopulationEntries(options(join(scratch, "cli-two-other"), { shared_dir: shared, members: membersOf() }), { index, of: 2 }),
        usage: Promise.resolve({}),
        alive: () => true,
        tell: (line) => void record.told.push(line),
        stop: () => void (record.stopped = true),
      };
    };
    let written: string | undefined;
    const printed: string[] = [];
    const run = input(shared, { pool: undefined, processes: 2, shard: { index: 0, of: 2 }, shard_script: "play.ts", forward: ["--config", "c.json"], start_shard: start, load_pool: (got: { write?: string | undefined; vouch?: ((digest: string) => void) | undefined }) => ((written = got.write), got.vouch?.("f".repeat(64)), Promise.resolve(pool)) });
    expect(await runPopulationCli(run, (line) => printed.push(line))).toBe(0);
    expect(printed).toEqual(lines);
    expect(treeBytes(shared)).toEqual(treeBytes(out));
    expect(started).toHaveLength(1);
    expect(started[0]?.args.slice(0, 2)).toEqual(["--config", "c.json"]);
    expect(started[0]?.args).toEqual(expect.arrayContaining(["--shard", "1/2", "--episodes-dir", shared, "--pool-file"]));
    expect(started[0]?.stopped).toBe(true);
    expect(started[0]?.told).toEqual([`pool ${"f".repeat(64)}\n`]);
    expect(written).toMatch(/pool\.json$/);
    expect(existsSync(written as string)).toBe(false); // the scratch directory is removed
  }, 300_000);

  it("plays a further game process's slice into its starter's directory and leaves the entries for it", async () => {
    const shared = join(scratch, "further-shared");
    const out = join(scratch, "further-out");
    const seen: { file: string | undefined; digest: string | undefined }[] = [];
    const load = async (share: { read?: { file: string; digest: Promise<string> } }): Promise<TrainingPool> => {
      seen.push({ file: share.read?.file, digest: await share.read?.digest });
      return pool;
    };
    const run = input(out, { pool: undefined, further: true, shard: { index: 1, of: 2 }, episodes_dir: shared, pool_file: "pool.json", parent_link: () => Promise.resolve("a".repeat(64)), load_pool: load });
    expect(await runPopulationCli(run, () => undefined)).toBe(0);
    expect(seen).toEqual([{ file: "pool.json", digest: "a".repeat(64) }]);
    const entries = readShardFile<PopulationEntry>(join(out, "shard.jsonl"));
    expect(entries.length).toBeGreaterThan(0);
    expect(entries.every((entry) => entry.game % 2 === 1)).toBe(true);
    const files = readdirSync(join(shared, "members", "m0", "episodes")).length + readdirSync(join(shared, "members", "m1", "episodes")).length;
    expect(files).toBe(entries.flatMap((entry) => entry.stored).length);
    expect(entries).toEqual(await playPopulationEntries(options(join(scratch, "further-ref"), { shared_dir: join(scratch, "further-ref-shared") }), { index: 1, of: 2 }));
  }, 300_000);

  it("starts no worker for a member that holds no seat in its slice", async () => {
    let made = 0;
    const out = join(scratch, "idle-slice");
    const run = input(out, { further: true, shard: { index: 99, of: 100 }, make_client: () => (made++, Object.assign(actorClient(), { start: () => Promise.resolve(), close: () => Promise.resolve() })), parent_link: () => Promise.resolve("a".repeat(64)) });
    const played = membersPlaying(planPopulation(config, 1, pool.teams.length), { index: 99, of: 100 });
    expect(await runPopulationCli(run, () => undefined)).toBe(0);
    expect(made).toBe(played.size);
  }, 300_000);

  it("starts the policy workers of a member as many as it was asked for", async () => {
    let made = 0;
    const lines: string[] = [];
    const run = input(join(scratch, "cli-workers"), { policy_workers: 3, make_client: () => (made++, Object.assign(actorClient(), { start: () => Promise.resolve(), close: () => Promise.resolve() })) });
    expect(await runPopulationCli(run, (line) => lines.push(line))).toBe(0);
    expect(made).toBe(6);
  }, 300_000);
});

describe("populationWaves", () => {
  it("keeps every member's client within n_parallel games of a wave and covers each game once", () => {
    const plan = planPopulation(populationConfig(3, 6), 1, 50);
    const waves = populationWaves(plan, 2);
    expect(waves.flat().map((g) => g.game).sort((a, b) => a - b)).toEqual(plan.map((g) => g.game));
    for (const wave of waves) {
      const load = new Map<string, number>();
      for (const g of wave) for (const m of new Set([g.seats.p1, g.seats.p2].flatMap((s) => ("member" in s ? [s.member] : [])))) load.set(m, (load.get(m) ?? 0) + 1);
      expect(Math.max(...load.values())).toBeLessThanOrEqual(2);
    }
  });
});

describe("refusals", () => {
  const options = (change: object = {}) => ({ engine, config, generation: 1, members: membersOf(), pool, out_dir: join(scratch, "refused"), ...change });

  it("refuses members that do not match the config, or share a frozen pair", async () => {
    await expect(playPopulationGeneration(options({ members: membersOf().slice(0, 1) }))).rejects.toThrow(GenerationError);
    const same = membersOf();
    (same[1] as { policy_id: string }).policy_id = POLICIES[0] as string;
    await expect(playPopulationGeneration(options({ members: same }))).rejects.toThrow(/same frozen pair/);
  });

  it("refuses a past-pool share without its pool, and a pool without a share, before any worker is asked", async () => {
    const members = membersOf();
    await expect(playPopulationGeneration(options({ members, config: populationConfig(2, 2, [500_000, 250_000, 250_000]) }))).rejects.toThrow(/needs the committed past pool/);
    await expect(playPopulationGeneration(options({ members, past: { pool: emptyPastPool(config), open: () => Promise.reject(new Error("never opened")) } }))).rejects.toThrow(/without a past-pool share/);
    expect(members.every((m) => m.client.asked === 0)).toBe(true);
  });

  it("refuses a population run that names an opponent sheet, and a config without a population", async () => {
    const raw = JSON.parse(JSON.stringify(config)) as { model: { encoder: { options?: object } } };
    raw.model.encoder.options = { opponent_sheet: { feature_set: "opponent-sheet/1", feature_set_hash: "d".repeat(64) } };
    await expect(playPopulationGeneration(options({ config: readTrainingRunConfig(raw) }))).rejects.toThrow(POPULATION_SHEET_REFUSAL);
    const { population: _unused, ...single } = JSON.parse(JSON.stringify(config)) as Record<string, unknown>;
    await expect(playPopulationGeneration(options({ config: readTrainingRunConfig(single) }))).rejects.toThrow(/no population section/);
  });

  it("stops between waves when told to", async () => {
    const stop = new AbortController();
    stop.abort();
    await expect(playPopulationGeneration(options({ signal: stop.signal, out_dir: join(scratch, "stopped") }))).rejects.toThrow(/was stopped/);
  });

  it("parses a --member flag and refuses a malformed one", () => {
    const text = `m3,${"1".repeat(64)},${"2".repeat(64)},policy:${"3".repeat(64)}`;
    expect(parseMemberFlag(text)).toEqual({ member: "m3", preview: "1".repeat(64), battle: "2".repeat(64), policy_id: `policy:${"3".repeat(64)}` });
    for (const bad of ["", "m03,a,b,c", `x0,${"1".repeat(64)},${"2".repeat(64)},policy:${"3".repeat(64)}`, `m0,${"1".repeat(64)},${"2".repeat(64)},${"3".repeat(64)}`]) {
      expect(() => parseMemberFlag(bad)).toThrow(GenerationError);
    }
  });
});

describe("forfeits and the population CLI", () => {
  /** A member whose worker fails on its n-th query: that game is forfeited, the rest are played. */
  const failing = (member: ReturnType<typeof membersOf>[number], nth: number) => {
    const real = member.client.ask.bind(member.client);
    let seen = 0;
    member.client.ask = (battle, query) => (++seen === nth ? Promise.reject(new PolicyWorkerError("crash", "gone")) : real(battle, query));
  };

  it("counts a forfeited game for every member that held a seat in it, by cause, in a record that passes its reader", async () => {
    const members = membersOf();
    failing(members[1] as ReturnType<typeof membersOf>[number], 7);
    const dir = join(scratch, "forfeit");
    const result = await playPopulationGeneration({ engine, config, generation: 1, members, pool, out_dir: dir });
    writePopulationGeneration(dir, result);
    const forfeited = result.games.filter((g) => g.outcome === "forfeit");
    expect(forfeited).toHaveLength(1);
    expect(forfeited[0]?.forfeit?.cause).toBe("crash");
    const record = readLeagueRunRecordV2(JSON.parse(readFileSync(join(dir, "league-run.json"), "utf8")));
    await checkLeagueRunRecord(record, config);
    expect(record.forfeits).toBe(1);
    expect(record.forfeit_causes).toEqual([{ cause: "crash", count: 1 }]);
    const generation = JSON.parse(readFileSync(join(dir, "generation.json"), "utf8"));
    expect(generation.forfeits).toBe(1);
    expect(generation.forfeit_causes).toEqual({ crash: 1 });
    expect(generation.schema_version).toBe(2);
    const row = forfeited[0] as (typeof result.games)[number];
    const holders = new Set([row.seats.p1, row.seats.p2].filter((s) => /^m\d$/.test(s)));
    for (const entry of record.members) expect(entry.forfeits).toBe(holders.has(entry.member) ? 1 : 0);
  }, 120_000);

  it("fails the generation on a harness incident, naming the game", async () => {
    const members = membersOf();
    (members[0] as ReturnType<typeof membersOf>[number]).client.ask = () => Promise.reject(new UnsupportedCandidateError("no"));
    await expect(playPopulationGeneration({ engine, config, generation: 1, members, pool, out_dir: join(scratch, "incident") })).rejects.toThrow(/game \d+ \(sodium,[0-9a-f]{32}\): unsupported_candidate/);
  }, 120_000);

  it("plays at the run's declared temperature and records the same shapes", async () => {
    const raw = JSON.parse(JSON.stringify(config)) as { generation: Record<string, unknown> };
    raw.generation.play = { temperature_ppm: 500_000 };
    const tempered = readTrainingRunConfig(raw);
    const asked: (number | undefined)[] = [];
    const members = membersOf();
    for (const member of members) {
      const real = member.client.ask.bind(member.client);
      member.client.ask = (battle, query) => (asked.push((query.sampling as { temperature_ppm?: number }).temperature_ppm), real(battle, query));
    }
    const result = await playPopulationGeneration({ engine, config: tempered, generation: 1, members, pool, out_dir: join(scratch, "tempered") });
    expect(asked.length).toBeGreaterThan(0);
    expect(new Set(asked)).toEqual(new Set([500_000]));
    expect(result.games.every((g) => g.outcome === "played")).toBe(true);
  }, 120_000);

  it("fails the generation when a member has no Episode of its own pair", async () => {
    const members = membersOf();
    (members[1] as ReturnType<typeof membersOf>[number]).client.ask = () => Promise.reject(new PolicyWorkerError("crash", "gone"));
    await expect(playPopulationGeneration({ engine, config, generation: 1, members, pool, out_dir: join(scratch, "empty") })).rejects.toThrow(/no Episode of m1/);
  }, 120_000);

  const input = (change: object = {}) => ({
    engine, config, generation: 1, pool, out: join(scratch, "cli"), run_dir: scratch, worker_python: "unused", engine_data: "unused", engine_data_hash: "e".repeat(64), encoder_version: "enc1:x",
    members: POLICIES.map((policy_id, k) => ({ member: `m${k}`, preview: String(k).repeat(64), battle: String(k + 2).repeat(64), policy_id })),
    make_client: () => Object.assign(actorClient(), { start: () => Promise.resolve(), close: () => Promise.resolve() }),
    ...change,
  });

  it("plays a generation through the CLI port and prints one summary line", async () => {
    const lines: string[] = [];
    expect(await runPopulationCli(input(), (line) => lines.push(line))).toBe(0);
    const summary = JSON.parse(lines[0] as string);
    expect(summary.generation).toBe(1);
    expect(summary.members.map((m: { member: string }) => m.member)).toEqual(["m0", "m1"]);
    expect(readdirSync(join(scratch, "cli"))).toEqual(expect.arrayContaining(["generation.json", "league-run.json", "plan.json", "members"]));
  }, 120_000);

  it("refuses a bad config before it creates any client", async () => {
    let made = 0;
    const make = () => (made++, Object.assign(actorClient(), { start: () => Promise.resolve(), close: () => Promise.resolve() }));
    const error = vi.spyOn(console, "error").mockImplementation(() => undefined);
    // Past-pool share, an odd baseline count, a wrong member count, equal pairs, and the real client path (no injected client).
    const same = input().members.map((m) => ({ ...m, policy_id: POLICIES[0] as string }));
    for (const change of [{ config: populationConfig(2, 2, [500_000, 250_000, 250_000]) }, { config: populationConfig(2, 2, [500_000, 500_000, 0], 3) }, { members: input().members.slice(0, 1) }, { members: same }]) {
      expect(await runPopulationCli(input({ make_client: make, ...change }), () => undefined)).toBe(1);
    }
    error.mockRestore();
    expect(made).toBe(0);
    expect(() => checkPopulationConfig(config, input().members)).not.toThrow();
  });

  it("rethrows an error that is no refusal (a generation below 1) before it creates any client", async () => {
    let made = 0;
    const make = () => (made++, Object.assign(actorClient(), { start: () => Promise.resolve(), close: () => Promise.resolve() }));
    await expect(runPopulationCli(input({ generation: 0, make_client: make }), () => undefined)).rejects.toThrow(RangeError);
    expect(made).toBe(0);
  });

  it("exits 1 when a member's worker cannot start, and closes the clients", async () => {
    const closed: string[] = [];
    const lines: string[] = [];
    const run = input({
      make_client: () => Object.assign(actorClient(), { start: () => Promise.reject(new GenerationError("the worker did not start")), close: () => (closed.push("closed"), Promise.resolve()) }),
    });
    const error = vi.spyOn(console, "error").mockImplementation(() => undefined);
    expect(await runPopulationCli(run, (line) => lines.push(line))).toBe(1);
    expect(error.mock.calls.flat().join(" ")).toMatch(/did not start/);
    error.mockRestore();
    expect(lines).toEqual([]);
    expect(closed).toHaveLength(2);
  });

  it("starts a policy worker per member when no client is injected, and rethrows its failure", async () => {
    await expect(runPopulationCli(input({ make_client: undefined }), () => undefined)).rejects.toThrow();
  });

  it("rethrows anything but a generation error", async () => {
    const run = input({ make_client: () => Object.assign(actorClient(), { start: () => Promise.reject(new TypeError("boom")), close: () => Promise.resolve() }) });
    await expect(runPopulationCli(run, () => undefined)).rejects.toThrow(TypeError);
  });
});
