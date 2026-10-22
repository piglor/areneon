// The progress evaluation end to end on CPU (ADR-017 slice 4): the Python driver makes a tiny run on the committed
// fixture team dataset (generation 0, the seeded pair, and generation 1, played on the dev teams), then the
// `progress` command plays the frozen pairs against the scripted baselines on the validation teams through Node
// and the real policy worker. It needs the project's Python environment (`make setup`, uv sync); without it the file
// is skipped, unless ARENEON_REQUIRE_PYTHON_WORKER=1 (the typescript CI job sets it).
import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  jcs,
  omniscientLogHash,
  progressGameStreams,
  type ProgressGameRecord,
  type ProgressEvaluationRecordV2,
  readProgressEvaluationRecordV2,
  readProgressGameRecord,
  readTrainingRunConfig,
  replayActor,
  runBattle,
  runProgressCli,
  teamHash,
  UsageError,
  type Json,
} from "@areneon/showdown-bridge";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { verifiedDatasetHash } from "../src/dataset-hash.ts";
import { DATASETS_DIR } from "../src/fixtures.ts";
import { datasetValidationPool } from "../src/progress-pool.ts";
import { engine, FORMAT } from "./support.ts";

const ROOT = fileURLToPath(new URL("../../../", import.meta.url));
const PYTHON = join(ROOT, ".venv/bin/python");
const DATA = join(ROOT, "fixtures/showdown", FORMAT, "engine-data.json");
const SOURCE = join(DATASETS_DIR(FORMAT), "fixtures-v1");
const env = { ...(process.env.SHOWDOWN_DIR ? { SHOWDOWN_DIR: process.env.SHOWDOWN_DIR } : {}) };

const loop = (...args: string[]): string => execFileSync(PYTHON, ["-I", "-m", "areneon_ml.loop", ...args], { encoding: "utf8", env, cwd: ROOT });
const gen = (n: number): string => `g${String(n).padStart(6, "0")}`;
const json = <T>(...path: string[]): T => JSON.parse(readFileSync(join(...path), "utf8")) as T;
const tree = (dir: string, prefix = ""): string[] =>
  readdirSync(dir, { withFileTypes: true }).flatMap((e) => (e.isDirectory() ? tree(join(dir, e.name), `${prefix}${e.name}/`) : [`${prefix}${e.name}`]));

describe.skipIf(!existsSync(PYTHON) && process.env.ARENEON_REQUIRE_PYTHON_WORKER !== "1")("the progress evaluation, tiny and on CPU", () => {
  let dir: string;
  let runDir: string;
  let configFile: string;

  beforeAll(() => {
    dir = mkdtempSync(join(tmpdir(), "areneon-progress-"));
    runDir = join(dir, "run");
    const tiny = json<Record<string, unknown>>(ROOT, "fixtures/contracts/v1/training-run-config/valid/tiny.json");
    configFile = join(dir, "config.json");
    writeFileSync(configFile, JSON.stringify({ ...tiny, team_dataset: { path: SOURCE, dataset_hash: verifiedDatasetHash(engine, SOURCE, FORMAT) } }));
    loop("run", "--config", configFile, "--run-dir", runDir, "--engine-data", DATA, "--generations", "1", "--node", process.execPath, "--worker-python", PYTHON);
  }, 240_000);
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  const progress = (progressDir: string, ...more: string[]): { generations: { generation: number; skipped: boolean; wall_ms?: Record<string, number>; matches: { match_id: string }[] }[]; progress_dir: string } =>
    JSON.parse(loop("progress", "--run-dir", runDir, "--engine-data", DATA, "--games", "4", "--record-games", "2", "--anchor-games", "4", "--progress-dir", progressDir, "--node", process.execPath, "--worker-python", PYTHON, ...more));

  it("evaluates both committed generations, twice, with byte-identical records", () => {
    const a = join(dir, "a.progress");
    const b = join(dir, "b.progress");
    // The default schedule (--all) is every 5th generation from 0; generation 1 is asked for, with the opt-in previous matches.
    const first = progress(a, "--all");
    expect(first.generations.map((g) => [g.generation, g.skipped])).toEqual([[0, false]]);
    expect(progress(a, "--generation", "1", "--previous-matches").generations.map((g) => [g.generation, g.skipped])).toEqual([[1, false]]);
    progress(b, "--all");
    progress(b, "--generation", "1", "--previous-matches");
    for (const g of [0, 1]) {
      const files = tree(join(a, gen(g)));
      expect(files).toContain("progress.json");
      expect(files).toContain("manifest.json");
      expect(files.filter((f) => f.startsWith("games/"))).toHaveLength(g === 0 ? 8 : 12); // 4 matches (6 from generation 1), 2 recorded games each
      expect(tree(join(b, gen(g)))).toEqual(files);
      for (const file of files) expect(readFileSync(join(a, gen(g), file), "utf8"), file).toBe(readFileSync(join(b, gen(g), file), "utf8"));
    }
    // Nothing was written into the run directory, and a second command verifies and keeps what is there.
    expect(readdirSync(runDir).sort()).toEqual([".lock", "checkpoints", "generations", "metrics", "run-config.json"].filter((n) => existsSync(join(runDir, n))).sort());
    expect(progress(a, "--all").generations.map((g) => g.skipped)).toEqual([true]);
    expect(progress(a, "--generation", "1").generations.map((g) => g.skipped)).toEqual([true]);
  }, 600_000);

  it("plays an anchor next to the previous generation, each pair from both seats on one seed", () => {
    const c = join(dir, "c.progress");
    const printed = progress(c, "--generation", "1", "--anchors", "0,1", "--anchor-games", "2", "--previous-matches");
    expect(printed.generations[0]?.matches.map((m) => m.match_id)).toEqual(["actor-rule-argmax", "actor-rule-sample", "anchor-g0-argmax", "anchor-g0-sample", "previous-argmax", "previous-sample", "random-legal-argmax", "random-legal-sample"]);
    const record = readProgressEvaluationRecordV2(json<ProgressEvaluationRecordV2>(c, gen(1), "progress.json"));
    expect(record.plan).toMatchObject({ games_per_match: 4, anchor_games_per_match: 2 });
    expect(record.teams.pairs_used).toBe(2);
    const g0 = json<{ checkpoints: Record<string, string> }>(runDir, "generations", gen(0), "manifest.json").checkpoints;
    for (const m of record.matches.filter((x) => x.opponent.kind === "anchor")) {
      expect(m.opponent).toMatchObject({ kind: "anchor", generation: 0, checkpoints: g0 });
      expect(m.games).toBe(2);
    }
    const rows = record.games.filter((r) => r.match_id === "anchor-g0-sample");
    expect(rows.map((r) => [r.pair, r.candidate_seat])).toEqual([[0, "p1"], [0, "p2"]]);
    // Pair 0 was played twice: the teams stay on their seats, the policies swap, the battle seed is the pair's.
    const [a, b] = [0, 1].map((k) => readProgressGameRecord(json(c, gen(1), "games", `anchor-g0-sample-${k}.json`)));
    const [first, second] = [a as ProgressGameRecord, b as ProgressGameRecord];
    expect(first.teams).toEqual(second.teams);
    expect(first.teams.p1.team_hash).not.toBe(first.teams.p2.team_hash);
    expect(first.seed).toBe(second.seed);
    expect(first.policies.p1).toBe(second.policies.p2); // the candidate's policy sits on p1 then p2
    expect(first.policies.p2).toBe(second.policies.p1);
  }, 600_000);

  it("plays validation teams only, never a team of the run's team-set, on seeds no training game used", () => {
    const a = join(dir, "a.progress");
    const run = readTrainingRunConfig(json(runDir, "run-config.json"));
    const trained = new Set(json<{ teams: string[] }>(runDir, "generations", gen(1), "team-set.json").teams);
    const league = new Set(json<{ seed: string }[]>(runDir, "generations", gen(1), "plan.json").map((p) => p.seed));
    const pool = datasetValidationPool(engine, run);
    const validation = new Set(pool.teams.map((t) => t.sets).map((sets) => jcs(sets as unknown as Json)));
    expect(trained.size).toBeGreaterThan(0);
    const record = json<ProgressEvaluationRecordV2>(a, gen(1), "progress.json");
    expect(readProgressEvaluationRecordV2(record).plan.games_per_match).toBe(4);
    expect(record.generation_manifest_sha256).toMatch(/^[0-9a-f]{64}$/);
    const games = tree(join(a, gen(1), "games")).map((f) => readProgressGameRecord(json(a, gen(1), "games", f)));
    expect(games).toHaveLength(12);
    // Generation 1 also played the previous generation's frozen pair, named by its checkpoint ids; generation 0 played none.
    const g0 = json<{ checkpoints: Record<string, string> }>(runDir, "generations", gen(0), "manifest.json").checkpoints;
    const previous = record.matches.filter((m) => m.opponent.kind === "previous");
    expect(previous.map((m) => m.match_id)).toEqual(["previous-argmax", "previous-sample"]);
    for (const m of previous) expect(m.opponent).toMatchObject({ kind: "previous", checkpoints: g0 });
    expect(json<ProgressEvaluationRecordV2>(a, gen(0), "progress.json").matches.some((m) => m.opponent.kind === "previous")).toBe(false);
    for (const stored of games) {
      for (const seat of ["p1", "p2"] as const) {
        expect(trained.has(stored.teams[seat].team_hash)).toBe(false);
        expect(validation.has(jcs(stored.teams[seat].team as unknown as Json))).toBe(true);
      }
      expect(league.has(stored.seed)).toBe(false);
      expect(stored.seed).toBe(progressGameStreams(run.seeds.run_key, stored.game >> 1).seed); // one seed per pair
    }
  });

  it("stores games that replay on the engine to their omniscient_log_sha256", async () => {
    const a = join(dir, "a.progress");
    const run = readTrainingRunConfig(json(runDir, "run-config.json"));
    let replayed = 0;
    for (const file of tree(join(a, gen(1), "games"))) {
      const stored = json<ProgressGameRecord>(a, gen(1), "games", file);
      if (stored.outcome.kind !== "played") continue;
      const choices = stored.decisions.map((d) => ({ player: d.player, request_seq: d.request_seq, choice: d.chosen }));
      const replay = await runBattle({
        engine,
        formatId: run.format_id,
        seed: stored.seed,
        teams: { p1: stored.teams.p1.team as never, p2: stored.teams.p2.team as never },
        actors: { p1: replayActor(choices, "p1"), p2: replayActor(choices, "p2") },
        maxTurns: run.generation.max_turns,
      });
      expect(omniscientLogHash(replay.omniscient_log)).toBe(stored.omniscient_log_sha256);
      replayed++;
    }
    expect(replayed).toBeGreaterThan(0);
  }, 120_000);

  it("refuses to evaluate a run whose team-set holds a validation team, and writes nothing", () => {
    const bad = join(dir, "bad");
    execFileSync("cp", ["-r", runDir, bad]);
    const pool = datasetValidationPool(engine, readTrainingRunConfig(json(runDir, "run-config.json")));
    const set = join(bad, "generations", gen(1), "team-set.json");
    const teams = json<{ teams: string[] }>(set);
    // A validation team in the run's team set. The file no longer matches its generation's manifest, so the Python
    // side refuses first; the Node side's own refusal of such a team is tested in process below.
    const forged = { ...teams, teams: [...teams.teams, teamHash(pool.teams[0]?.sets as never)] };
    writeFileSync(set, `${jcs(forged as unknown as Json)}\n`);
    const out = join(dir, "bad.progress");
    const run = spawnSync(PYTHON, ["-I", "-m", "areneon_ml.loop", "progress", "--run-dir", bad, "--generation", "1", "--engine-data", DATA, "--games", "2", "--progress-dir", out, "--node", process.execPath, "--worker-python", PYTHON], { encoding: "utf8", env, cwd: ROOT });
    expect(run.status).toBe(3);
    expect(existsSync(join(out, gen(1)))).toBe(false);
    expect(run.stderr).toMatch(/differ from its manifest|team-set/);
  }, 120_000);

  describe("the Node command in process", () => {
    const flags = (out: string, change: Record<string, string> = {}): string[] => {
      const g1 = json<{ checkpoints: Record<string, string> }>(runDir, "generations", gen(1), "manifest.json");
      const base: Record<string, string> = {
        config: join(runDir, "run-config.json"),
        generation: "1",
        preview: g1.checkpoints.preview as string,
        battle: g1.checkpoints.battle as string,
        "policy-id-sample": `policy:${"a".repeat(64)}`,
        "policy-id-argmax": `policy:${"b".repeat(64)}`,
        "generation-manifest-sha256": "c".repeat(64),
        "team-set": join(runDir, "generations", gen(1), "team-set.json"),
        "run-dir": runDir,
        "engine-data": DATA,
        "encoder-version": json<{ encoder_version: string }>(runDir, "generations", gen(1), "records", "battle.json").encoder_version,
        "worker-python": PYTHON,
        games: "2",
        "record-games": "1",
        out,
        ...change,
      };
      return Object.entries(base).flatMap(([k, v]) => [`--${k}`, v]);
    };
    const run = async (out: string, change: Record<string, string> = {}): Promise<{ code: number; lines: string[] }> => {
      const lines: string[] = [];
      const code = await runProgressCli(flags(out, change), (l) => lines.push(l), datasetValidationPool);
      return { code, lines };
    };

    it("plays and prints one canonical line", async () => {
      const out = join(dir, "cli-ok");
      const { code, lines } = await run(out);
      expect(code).toBe(0);
      const printed = JSON.parse(lines[0] as string) as { generation: number; matches: { games: number }[]; wall_ms: Record<string, number> };
      expect(lines[0]).toBe(jcs(printed as unknown as Json));
      expect(printed.generation).toBe(1);
      expect(printed.matches.map((m) => m.games)).toEqual([2, 2, 2, 2]);
      expect(Object.keys(printed.wall_ms)).toHaveLength(4);
      expect(tree(out)).toContain("progress.json");
    }, 120_000);

    it("plays the previous generation's pair when given, and refuses half a set of previous flags", async () => {
      const g0 = json<{ checkpoints: Record<string, string> }>(runDir, "generations", gen(0), "manifest.json").checkpoints;
      const out = join(dir, "cli-previous");
      const { code, lines } = await run(out, { "previous-preview": g0.preview as string, "previous-battle": g0.battle as string, "previous-policy-id-sample": `policy:${"e".repeat(64)}`, "previous-policy-id-argmax": `policy:${"f".repeat(64)}` });
      expect(code).toBe(0);
      const printed = JSON.parse(lines[0] as string) as { matches: { match_id: string }[] };
      expect(printed.matches.map((m) => m.match_id)).toEqual(["actor-rule-argmax", "actor-rule-sample", "previous-argmax", "previous-sample", "random-legal-argmax", "random-legal-sample"]);
      await expect(run(join(dir, "half"), { "previous-preview": g0.preview as string })).rejects.toThrow(/go together/);
    }, 180_000);

    it("plays an anchor when given, and refuses anchors without anchor games and a malformed anchor", async () => {
      const g0 = json<{ checkpoints: Record<string, string> }>(runDir, "generations", gen(0), "manifest.json").checkpoints;
      const spec = `0,${g0.preview},${g0.battle},policy:${"1".repeat(64)},policy:${"2".repeat(64)}`;
      const lines: string[] = [];
      const out = join(dir, "cli-anchor");
      const code = await runProgressCli([...flags(out), "--anchor", spec, "--anchor-games", "2"], (l) => lines.push(l), datasetValidationPool);
      expect(code).toBe(0);
      const printed = JSON.parse(lines[0] as string) as { matches: { match_id: string; games: number }[] };
      expect(printed.matches.map((m) => [m.match_id, m.games]).filter(([id]) => String(id).startsWith("anchor"))).toEqual([["anchor-g0-argmax", 2], ["anchor-g0-sample", 2]]);
      await expect(runProgressCli([...flags(join(dir, "n1")), "--anchor", spec], () => undefined, datasetValidationPool)).rejects.toThrow(/go together/);
      await expect(runProgressCli([...flags(join(dir, "n2")), "--anchor-games", "2"], () => undefined, datasetValidationPool)).rejects.toThrow(/go together/);
      await expect(runProgressCli([...flags(join(dir, "n3")), "--anchor", "0,x", "--anchor-games", "2"], () => undefined, datasetValidationPool)).rejects.toThrow(/--anchor must be/);
      await expect(runProgressCli([...flags(join(dir, "n4")), "--anchor", spec, "--anchor-games", "1"], () => undefined, datasetValidationPool)).rejects.toThrow(/at least 2/);
      const original = console.error;
      const said: string[] = [];
      console.error = (m: unknown): void => void said.push(String(m));
      try {
        // A refusal of the evaluation (not a usage error): exit 1 with the reason.
        expect(await runProgressCli([...flags(join(dir, "n5")), "--anchor", spec, "--anchor-games", "3"], () => undefined, datasetValidationPool)).toBe(1);
      } finally {
        console.error = original;
      }
      expect(said.join("\n")).toMatch(/anchor-games must be an even integer/);
    }, 240_000);

    it("refuses a missing flag, an unknown flag and a bad integer as usage errors", async () => {
      await expect(runProgressCli(["--generation", "1"], () => undefined, datasetValidationPool)).rejects.toThrow(UsageError);
      await expect(runProgressCli(["--nope"], () => undefined, datasetValidationPool)).rejects.toThrow(UsageError);
      await expect(run(join(dir, "x"), { games: "0" })).rejects.toThrow(/--games must be an integer of at least 1/);
      await expect(run(join(dir, "x"), { generation: "-1" })).rejects.toThrow(UsageError);
      await expect(run(join(dir, "x"), { "record-games": "1.5" })).rejects.toThrow(UsageError);
    });

    it("exits 1 on a refusal of the evaluation: more games than pairs, a team of the run's team-set", async () => {
      const refusal: string[] = [];
      const spy = (m: unknown): void => void refusal.push(String(m));
      const original = console.error;
      console.error = spy;
      try {
        expect((await run(join(dir, "too-many"), { games: "100000" })).code).toBe(1);
        expect(refusal.join("\n")).toMatch(/disjoint team pairs/);
        const pool = datasetValidationPool(engine, readTrainingRunConfig(json(runDir, "run-config.json")));
        const trained = join(dir, "team-set-with-validation.json");
        // The team set lists the hashes of the played forms of the teams a run drew.
        writeFileSync(trained, JSON.stringify({ teams: pool.teams.map((t) => teamHash(t.sets)) }));
        expect((await run(join(dir, "trained"), { "team-set": trained })).code).toBe(1);
        expect(refusal.join("\n")).toMatch(/team-set\.json/);
      } finally {
        console.error = original;
      }
    }, 120_000);

    it("lets a failure that is not a refusal through, and still closes the worker", async () => {
      const broken = (): never => {
        throw new TypeError("a bug in the pool source");
      };
      await expect(runProgressCli(flags(join(dir, "bug")), () => undefined, broken)).rejects.toThrow(TypeError);
    });

    it("refuses a config without a team dataset", async () => {
      const plain = join(dir, "plain-config.json");
      writeFileSync(plain, JSON.stringify(json<Record<string, unknown>>(ROOT, "fixtures/contracts/v1/training-run-config/valid/tiny.json")));
      const refusal: string[] = [];
      const original = console.error;
      console.error = (m: unknown): void => void refusal.push(String(m));
      try {
        expect((await run(join(dir, "plain"), { config: plain })).code).toBe(1);
      } finally {
        console.error = original;
      }
      expect(refusal.join("\n")).toMatch(/names no team dataset/);
    });
  });
});
