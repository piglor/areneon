import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { jcs, type Json } from "../src/canonical-json.ts";
import { interimPoolSource, UsageError } from "../src/generation-cli.ts";
import { GOLDEN_FORMAT, firstLegal } from "../src/golden.ts";
import { GenerationError, trainingTeamPool, type TrainingPool } from "../src/league-generation.ts";
import { loadEngine } from "../src/engine.ts";
import { type BattleRecord, runBattle } from "../src/runner.ts";
import { loadTrainingSet, TrainingLoaderError } from "../src/training-loader.ts";
import { interimTeacherPool, runTeacherCli } from "../src/teacher-cli.ts";
import {
  checkTeacherSettings,
  planTeacher,
  playTeacherEntries,
  playTeacherRun,
  teacherContext,
  type TeacherEntry,
  teacherPolicyId,
  teacherReport,
  teacherSeatSeed,
  type TeacherSettings,
  writeTeacherRun,
} from "../src/teacher-generation.ts";

const engine = loadEngine();
let pool: TrainingPool;
let dir: string;
beforeAll(() => {
  pool = trainingTeamPool(engine, ["Illusion"]);
  dir = mkdtempSync(join(tmpdir(), "areneon-teacher-"));
  for (const name of ["a", "b", "c"]) mkdirSync(join(dir, name));
});
afterAll(() => rmSync(dir, { recursive: true, force: true }));

const settings = (change: Partial<TeacherSettings> = {}): TeacherSettings => ({
  run_key: "tiny-teacher",
  format_id: GOLDEN_FORMAT,
  games: 2,
  random_games: 1,
  actor_rule_games: 0,
  max_turns: 100,
  epsilon_ppm: 0,
  latency_budget_ms: { preview_ms: 30000, battle_ms: 30000 },
  ...change,
});

/** SHA-256 of every file under a directory, by relative path. */
function digest(root: string, prefix = ""): Record<string, string> {
  const out: Record<string, string> = {};
  for (const name of readdirSync(join(root, prefix)).sort()) {
    const path = join(prefix, name);
    if (statSync(join(root, path)).isDirectory()) Object.assign(out, digest(root, path));
    else out[path] = createHash("sha256").update(readFileSync(join(root, path))).digest("hex");
  }
  return out;
}

describe("the teacher plan", () => {
  it("lists the mirror games, then each baseline's, with league-seed/1 streams under the teacher's run key", () => {
    const plan = planTeacher(settings({ games: 2, random_games: 2, actor_rule_games: 1 }), 100);
    expect(plan.map((entry) => entry.opponent)).toEqual([
      "self", "self",
      { actor: "random-legal", policy_id: "random-legal@v1" }, { actor: "random-legal", policy_id: "random-legal@v1" },
      { actor: "actor-rule", policy_id: "actor-rule@v1" },
    ]);
    expect(plan.map((entry) => entry.policy_seat)).toEqual(["both", "both", "p1", "p2", "p1"]);
    expect(new Set(plan.map((entry) => entry.seed)).size).toBe(5);
    expect(planTeacher(settings({ games: 2, random_games: 2, actor_rule_games: 1 }), 100)).toEqual(plan);
    expect(planTeacher(settings({ run_key: "another" }), 100)[0]?.seed).not.toBe(plan[0]?.seed);
    // the first game's streams do not depend on the number of games after it
    expect(planTeacher(settings({ games: 7 }), 100)[0]).toEqual(planTeacher(settings({ games: 1, random_games: 0 }), 100)[0]);
  });

  it("names the teacher damage-aware@v1, and another identity when it explores", () => {
    expect(teacherPolicyId(0)).toBe("damage-aware@v1");
    expect(teacherPolicyId(50_000)).toBe("damage-aware@v1-eps50000");
    expect(teacherSeatSeed(7, "p1")).not.toBe(teacherSeatSeed(7, "p2"));
    expect(teacherSeatSeed(7, "p1")).toBe(teacherSeatSeed(7, "p1"));
    expect(Number.isInteger(teacherSeatSeed(7, "p1"))).toBe(true);
  });

  it("refuses settings that play nothing or out of range", () => {
    expect(() => checkTeacherSettings(settings())).not.toThrow();
    for (const change of [
      { games: -1 }, { games: 1.5 }, { games: 0, random_games: 0, actor_rule_games: 0 }, { max_turns: 0 }, { epsilon_ppm: 1_000_001 },
      { run_key: "" }, { run_key: "CHANGE-ME-PER-RUN" }, { latency_budget_ms: { preview_ms: 0, battle_ms: 1 } },
      { latency_budget_ms: { preview_ms: 1, battle_ms: Number.NaN } },
    ]) {
      expect(() => checkTeacherSettings(settings(change)), JSON.stringify(change)).toThrow(RangeError);
    }
  });

  it("plays the golden format only", () => {
    expect(() => teacherContext(engine, settings({ format_id: "gen9other" }))).toThrow(GenerationError);
  });
});

describe("a tiny teacher run", () => {
  const cache: { run?: Awaited<ReturnType<typeof playTeacherRun>>; entries?: TeacherEntry[] } = {};
  const run = async () => (cache.run ??= await playTeacherRun({ engine, settings: settings(), pool, stream_path: join(dir, "a", "stream.jsonl") }));

  it("loads every teacher Episode into a learner-ready, synthetic TrainingSet of the teacher", async () => {
    const result = await run();
    expect(result.games.map((game) => game.outcome)).toEqual(["played", "played", "played"]);
    // two mirror games give two Episodes each, the baseline game one teacher seat and one baseline seat
    expect(result.episodes).toHaveLength(6);
    const ids = result.episodes.map((episode) => episode.provenance.behaviour_policy_id);
    expect(ids.filter((id) => id === "damage-aware@v1")).toHaveLength(5);
    expect(ids.filter((id) => id === "random-legal@v1")).toHaveLength(1);
    expect(result.set).toMatchObject({ data_class: "synthetic", learner_ready: true, behaviour_policy_id: "damage-aware@v1" });
    expect(result.set.episode_hashes).toHaveLength(5);
    expect(result.set.examples).toBeGreaterThan(20);
    expect(result.training_set_hash).toMatch(/^[0-9a-f]{64}$/);
    for (const episode of result.episodes) {
      const collection = episode.provenance.collection;
      expect(collection).toMatchObject({ kind: "league", run_id: "tiny-teacher" });
      expect(episode).toMatchObject({ data_class: "synthetic", rights: "cc0-synthetic" });
    }
    // the other seat's policy is explicit in every collection record
    const pairs = result.episodes.map((e) => [e.provenance.behaviour_policy_id, (e.provenance.collection as { opponent_policy_id: string }).opponent_policy_id]);
    expect(pairs.filter(([own, other]) => own === "damage-aware@v1" && other === "damage-aware@v1")).toHaveLength(4);
    expect(pairs).toContainEqual(["damage-aware@v1", "random-legal@v1"]);
    expect(pairs).toContainEqual(["random-legal@v1", "damage-aware@v1"]);
  }, 120_000);

  it("shows the learner no seed and no draw", async () => {
    const result = await run();
    const stream = readFileSync(join(dir, "a", "stream.jsonl"), "utf8");
    expect(stream.length).toBeGreaterThan(1000);
    for (const entry of result.plan) {
      expect(stream).not.toContain(entry.seed.slice("sodium,".length));
    }
    expect(stream).not.toContain("sodium,");
    expect(stream).not.toContain("baseline_seed");
  });

  it("refuses the run under any other behaviour policy id, and the baseline's own seat under the teacher's", async () => {
    const result = await run();
    const input = { episodes: result.episodes, evaluation_manifests: [], known_synthetic: pool.known_synthetic };
    await expect(loadTrainingSet({ ...input, behaviour_policy_id: "damage-aware@v2" })).rejects.toThrow(TrainingLoaderError);
    await expect(loadTrainingSet({ ...input, behaviour_policy_id: "damage-aware@v1" })).rejects.toThrow(/random-legal@v1, not damage-aware@v1/);
    const mirror = result.episodes.filter((episode) => episode.provenance.behaviour_policy_id === "damage-aware@v1");
    const own = { ...input, episodes: mirror };
    await expect(loadTrainingSet({ ...own, behaviour_policy_id: "actor-rule@v1" })).rejects.toThrow(TrainingLoaderError);
    const loaded = await loadTrainingSet({ ...own, behaviour_policy_id: "damage-aware@v1" });
    expect(loaded.set.stream_sha256).toBe(result.set.stream_sha256);
    // a private team's Episode never loads: the same Episodes with a team the pool does not know are refused
    await expect(loadTrainingSet({ ...own, known_synthetic: new Set(), behaviour_policy_id: "damage-aware@v1" })).rejects.toThrow(TrainingLoaderError);
  }, 120_000);

  it("records the teacher's identity, its counts and its budget", async () => {
    const { teacher } = await run();
    expect(teacher).toMatchObject({ policy_id: "damage-aware@v1", epsilon_ppm: 0, latency_budget_ms: { preview_ms: 30000, battle_ms: 30000 }, seats: 5 });
    expect(teacher.identity_hash).toMatch(/^[0-9a-f]{64}$/);
    expect(teacher.totals.decisions).toBeGreaterThan(20);
    expect(teacher.totals.scored_decisions).toBeGreaterThan(0);
    expect(teacher.totals.epsilon_plays).toBe(0);
    expect(teacher.games).toHaveLength(3);
    expect(teacher.games[2]?.p1 === undefined || teacher.games[2]?.p2 === undefined).toBe(true);
    const sum = teacher.games.flatMap((game) => [game.p1, game.p2]).reduce((total, counts) => total + (counts?.decisions ?? 0), 0);
    expect(sum).toBe(teacher.totals.decisions);
  });

  it("is byte-identical when rerun from the same key, pool and commit, whatever the split into game processes", async () => {
    const result = await run();
    writeTeacherRun(join(dir, "a"), result, null);
    const b = await playTeacherRun({ engine, settings: settings(), pool, stream_path: join(dir, "b", "stream.jsonl") });
    writeTeacherRun(join(dir, "b"), b, null);
    const context = teacherContext(engine, settings());
    const halves = await Promise.all([0, 1].map((index) => playTeacherEntries({ engine, settings: settings(), pool }, context, { index, of: 2 })));
    const c = await playTeacherRun({ engine, settings: settings(), pool, stream_path: join(dir, "c", "stream.jsonl"), shard: { index: 0, of: 2 }, others: Promise.resolve(halves[1] as TeacherEntry[]) });
    writeTeacherRun(join(dir, "c"), c, null);
    const first = digest(join(dir, "a"));
    expect(Object.keys(first)).toEqual(expect.arrayContaining(["run.json", "teacher.json", "plan.json", "stream.jsonl", "training-set.json", "version-manifest.json"]));
    expect(Object.keys(first).filter((name) => name.startsWith("episodes/"))).toHaveLength(6);
    expect(digest(join(dir, "b"))).toEqual(first);
    expect(digest(join(dir, "c"))).toEqual(first);
    const note = JSON.parse(readFileSync(join(dir, "a", "run.json"), "utf8")) as Record<string, unknown>;
    expect(note).toMatchObject({ teacher_policy_id: "damage-aware@v1", seed_derivation: "league-seed/1", dataset_hash: null, training_set_hash: result.training_set_hash });
    expect(jcs(JSON.parse(readFileSync(join(dir, "a", "training-set.json"), "utf8")) as Json)).toBe(jcs(result.set as unknown as Json));
  }, 180_000);

  it("refuses to write a stream that is not where the run says", async () => {
    const result = await run();
    expect(() => writeTeacherRun(join(dir, "elsewhere"), { ...result, stream_path: join(dir, "not-here.jsonl") }, null)).toThrow(/not to/);
    const memory = await playTeacherRun({ engine, settings: settings({ games: 1, random_games: 0 }), pool });
    expect(memory.stream).not.toBeNull();
    writeTeacherRun(join(dir, "memory"), memory, "d".repeat(64));
    expect(readFileSync(join(dir, "memory", "stream.jsonl"), "utf8")).toBe(memory.stream);
    expect(JSON.parse(readFileSync(join(dir, "memory", "run.json"), "utf8"))).toMatchObject({ dataset_hash: "d".repeat(64) });
  }, 120_000);

  it("refuses entries that are not the plan's games, and seats of two baseline identities", async () => {
    const context = teacherContext(engine, settings());
    const entries = await playTeacherEntries({ engine, settings: settings({ games: 1, random_games: 0 }), pool }, context);
    await expect(playTeacherRun({ engine, settings: settings(), pool, shard: { index: 0, of: 2 }, others: Promise.resolve([]) })).rejects.toThrow(/not the plan's games/);
    const forged = structuredClone(entries) as TeacherEntry[];
    (forged[0]?.teacher.p2 as { identity_hash: string }).identity_hash = "f".repeat(64);
    expect(() => teacherReport(forged, context, settings())).toThrow(/2 baseline identities/);
    expect(() => teacherReport([], context, settings())).toThrow(/0 baseline identities/);
    await expect(playTeacherEntries({ engine, settings: settings(), pool }, context, { index: 2, of: 2 })).rejects.toThrow(RangeError);
  }, 120_000);

  it("fails the run, retrying nothing, when the teacher exceeds its per-decision limit", async () => {
    const tight = settings({ games: 1, random_games: 0, latency_budget_ms: { preview_ms: 30000, battle_ms: 0.000001 } });
    await expect(playTeacherRun({ engine, settings: tight, pool })).rejects.toThrow(/baseline_failure/);
  }, 120_000);
});

describe("exploration", () => {
  it("is another baseline identity, drawn from only when epsilon_ppm is above 0", async () => {
    const exploring = settings({ games: 1, random_games: 0, epsilon_ppm: 300_000 });
    const result = await playTeacherRun({ engine, settings: exploring, pool });
    expect(result.set.behaviour_policy_id).toBe("damage-aware@v1-eps300000");
    expect(result.teacher.totals.epsilon_plays).toBeGreaterThan(0);
    const plain = await playTeacherRun({ engine, settings: settings({ games: 1, random_games: 0 }), pool });
    expect(result.teacher.identity_hash).not.toBe(plain.teacher.identity_hash);
    expect(plain.teacher.totals.epsilon_plays).toBe(0);
    const again = await playTeacherRun({ engine, settings: exploring, pool });
    expect(again.training_set_hash).toBe(result.training_set_hash);
  }, 120_000);
});

describe("hidden information (ADR-018 section 3, reused on the teacher)", () => {
  /** The p1 choices a game recorded, by request number, for requests whose p1 channel prefix is `lines`. */
  const choiceAt = (record: BattleRecord, seq: number): string | undefined => record.choices.find((c) => c.player === "p1" && c.request_seq === seq)?.choice;
  const prefix = (record: BattleRecord, seq: number): string => {
    let seen = 0;
    const lines = record.channels.p1;
    for (let i = 0; i < lines.length; i++) if ((lines[i] as string).startsWith("|request|") && ++seen === seq) return jcs(lines.slice(0, i + 1) as unknown as Json);
    return "";
  };

  it("makes the teacher's choices depend on its own channel, not on what the opponent holds unrevealed", async () => {
    const context = teacherContext(engine, settings());
    const mine = pool.teams[0] as TrainingPool["teams"][number];
    const theirs = structuredClone(pool.teams[1] as TrainingPool["teams"][number]);
    const other = theirs.map((set, index) => ({ ...structuredClone(set), nature: index % 2 === 0 ? "Adamant" : "Timid", evs: { ...set.evs, hp: set.evs.spe, spe: set.evs.hp } }));
    const seed = "sodium,00112233445566778899aabbccddeeff";
    const play = (p2: typeof theirs) =>
      runBattle({ engine, formatId: GOLDEN_FORMAT, seed, teams: { p1: structuredClone(mine), p2: structuredClone(p2) }, actors: { p1: context.makeActor("p1", 1), p2: firstLegal }, maxTurns: 6 });
    const a = await play(theirs);
    const b = await play(other);
    let compared = 0;
    for (let seq = 1; seq <= Math.min(a.requests.p1.length, b.requests.p1.length); seq++) {
      const left = prefix(a, seq);
      if (left === "" || left !== prefix(b, seq)) break;
      expect(choiceAt(b, seq)).toBe(choiceAt(a, seq));
      compared++;
    }
    expect(compared).toBeGreaterThanOrEqual(2);
  }, 120_000);
});

describe("teacher-run command", () => {
  const out = (name: string): string => join(dir, `cli-${name}`);
  const flags = (name: string, more: string[] = []): string[] => ["--run-key", "tiny-teacher", "--format", GOLDEN_FORMAT, "--games", "2", "--random-games", "1", "--out", out(name), ...more];
  const quiet = (): void => undefined;

  it("refuses what it was not given, before it starts anything", async () => {
    await expect(runTeacherCli(["--games", "1"], quiet)).rejects.toThrow(/--run-key is required/);
    await expect(runTeacherCli(["--nope"], quiet)).rejects.toThrow(UsageError);
    await expect(runTeacherCli([...flags("x"), "--dataset", "somewhere"], quiet)).rejects.toThrow(/go together/);
    for (const [flag, bad] of [["--games", "x"], ["--epsilon-ppm", "x"], ["--games", "1.5"]] as const) {
      await expect(runTeacherCli([...flags("x"), flag, bad], quiet)).rejects.toThrow(/non-negative integer/);
    }
    await expect(runTeacherCli([...flags("x"), "--epsilon-ppm", "2000000"], quiet)).rejects.toThrow(/at most 1,000,000/);
    await expect(runTeacherCli(["--run-key", "CHANGE-ME-1", "--format", GOLDEN_FORMAT, "--games", "1", "--out", out("x")], quiet)).rejects.toThrow(/run_key/);
    for (const bad of ["0", "65", "x"]) await expect(runTeacherCli([...flags("x"), "--game-processes", bad], quiet, undefined, "s.ts")).rejects.toThrow(/--game-processes must be/);
    await expect(runTeacherCli([...flags("x"), "--game-processes", "2"], quiet)).rejects.toThrow(/needs the script/);
    for (const bad of ["1", "x/2", "2/2", "0/65"]) await expect(runTeacherCli([...flags("x"), "--shard", bad], quiet)).rejects.toThrow(/INDEX\/COUNT/);
    await expect(runTeacherCli([...flags("x"), "--shard", "1/2", "--game-processes", "2"], quiet, undefined, "s.ts")).rejects.toThrow(/INDEX\/COUNT/);
    writeFileSync(join(dir, "occupied"), "x");
    await expect(runTeacherCli(["--run-key", "k", "--format", GOLDEN_FORMAT, "--games", "1", "--out", dir], quiet)).rejects.toThrow(/not empty/);
  });

  it("takes its teams from the pool source it is given, and the interim source refuses a dataset", async () => {
    const refuse = (): never => {
      throw new UsageError("from the port");
    };
    await expect(runTeacherCli(flags("port"), quiet, refuse)).rejects.toThrow("from the port");
    expect(() => interimTeacherPool(engine, { format_id: GOLDEN_FORMAT, generation: { excluded_abilities: [] }, team_dataset: { path: "p", dataset_hash: "a".repeat(64) } })).toThrow(/team-dataset/);
    expect(interimPoolSource).toBeTypeOf("function");
  });

  it("exits 1 when the exclusions leave no team", async () => {
    const abilities = [...new Set(trainingTeamPool(engine).teams.flatMap((team) => team.map((set) => set.ability)))].join(",");
    const error = vi.spyOn(console, "error").mockImplementation(quiet);
    try {
      expect(await runTeacherCli([...flags("none"), "--excluded-abilities", abilities], quiet)).toBe(1);
      expect(error).toHaveBeenCalledWith(expect.stringContaining("the team pool is empty"));
      expect(readdirSync(dir).filter((name) => name.includes("cli-none"))).toEqual([]);
    } finally {
      error.mockRestore();
    }
  });

  it("writes the directory in place, prints one line, and gives the same bytes from two game processes", async () => {
    const lines: string[] = [];
    expect(await runTeacherCli(flags("one"), (line) => lines.push(line))).toBe(0);
    const summary = JSON.parse(lines[0] as string) as Record<string, unknown>;
    expect(summary).toMatchObject({ run_key: "tiny-teacher", teacher_policy_id: "damage-aware@v1", games: 3, episodes: 6 });
    expect(readdirSync(dir).filter((name) => name.endsWith(".partial"))).toEqual([]);
    const one = digest(out("one"));
    expect(JSON.parse(readFileSync(join(out("one"), "training-set.json"), "utf8"))).toMatchObject({ learner_ready: true });
    await expect(runTeacherCli(flags("one"), quiet)).rejects.toThrow(/not empty/);
    const script = new URL("../../team-dataset/scripts/teacher-run.ts", import.meta.url).pathname;
    expect(await runTeacherCli(flags("two", ["--game-processes", "2"]), quiet, undefined, script)).toBe(0);
    expect(digest(out("two"))).toEqual(one);
  }, 240_000);

  it("plays a further game process's slice and leaves it for the process that started it", async () => {
    const handlers: (() => void)[] = [];
    const once = vi.spyOn(process.stdin, "once").mockImplementation(((_event: string, handler: () => void) => {
      handlers.push(handler);
      return process.stdin;
    }) as never);
    const resume = vi.spyOn(process.stdin, "resume").mockImplementation(() => process.stdin);
    try {
      expect(await runTeacherCli(flags("shard", ["--shard", "1/2"]), quiet)).toBe(0);
    } finally {
      once.mockRestore();
      resume.mockRestore();
    }
    // it ends if the process that started it dies (its stdin closes)
    const exit = vi.spyOn(process, "exit").mockImplementation((() => undefined) as never);
    try {
      handlers[0]?.();
      expect(exit).toHaveBeenCalledWith(1);
    } finally {
      exit.mockRestore();
    }
    expect(readdirSync(out("shard"))).toEqual(["shard.jsonl"]);
    expect(readFileSync(join(out("shard"), "shard.jsonl"), "utf8").trim().split("\n")).toHaveLength(1);
  }, 120_000);
});
