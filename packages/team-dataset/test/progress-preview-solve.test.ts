// The Team Preview solver candidate of the progress evaluation (ADR-029 section 8, #1260) with the real engine, the real opponent
// model and integer solver, and a stand-in worker (the real worker is driven by progress-preview-solve.python.test.ts): the matches,
// the counts of both sides, the control, the stored games, the Team Preview budget, byte-identical reruns and the refusals.
import { readFileSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { readerSha256 } from "../../public-state/src/index.ts";
import {
  actorRule,
  jcs,
  type Json,
  loadPreviewSolveConfig,
  loadRulesSnapshot,
  type PolicyClient,
  playProgress,
  type PreviewSolveConfig,
  previewSolvePolicy,
  previewSolveSetup,
  type PreviewSolveSetup,
  ProgressError,
  type ProgressOptions,
  readProgressGameRecord,
  readTrainingRunConfig,
  resolveFormat,
  runProgressCli,
} from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it } from "vitest";
import { verifiedDatasetHash } from "../src/dataset-hash.ts";
import { DATASETS_DIR } from "../src/fixtures.ts";
import { createPreviewSolvePorts } from "../src/preview-solver-port.ts";
import { datasetValidationPool } from "../src/progress-pool.ts";
import { goldenTeamVocabulary } from "../../showdown-bridge/src/golden.ts";
import { StandInWorker } from "./search-actor-support.ts";
import { engine, FORMAT } from "./support.ts";

const ROOT = new URL("../../..", import.meta.url).pathname;
const SOURCE = join(DATASETS_DIR(FORMAT), "fixtures-v1");
const tiny = JSON.parse(readFileSync(join(ROOT, "fixtures/contracts/v1/training-run-config/valid/tiny.json"), "utf8")) as Record<string, unknown>;
const config = readTrainingRunConfig({ ...tiny, generation: { ...(tiny.generation as object), max_turns: 2 }, team_dataset: { path: SOURCE, dataset_hash: verifiedDatasetHash(engine, SOURCE, FORMAT) } });
const pool = datasetValidationPool(engine, config);
const checkpoints = { preview: "1".repeat(64), battle: "2".repeat(64) };
const vocabulary = goldenTeamVocabulary();
const manifest = resolveFormat(engine, FORMAT, loadRulesSnapshot(engine.commit)).manifest;

/** The committed solver config at test size: two worlds, two rows, two columns. */
const smallConfig = (change: (c: PreviewSolveConfig) => void = () => {}): PreviewSolveConfig => {
  const c = structuredClone(loadPreviewSolveConfig(FORMAT));
  c.k_own = 2;
  c.k_opp = 2;
  c.worlds.n = 2;
  c.worlds.min_valid = 2;
  c.solver.iterations = 256;
  change(c);
  return c;
};
const setupFor = (change?: (c: PreviewSolveConfig) => void): PreviewSolveSetup => {
  const ports = createPreviewSolvePorts({ engine, vocabulary, reader_sha256: readerSha256(), manifest, arm: "generator-matched" });
  return previewSolveSetup({ engine, format_id: FORMAT, vocabulary, config: smallConfig(change), teams: ports.teams, solver: ports.solver, data_class: "synthetic" });
};

/** The stand-in worker with the full reply `ask` gives a policy seat. */
class Worker extends StandInWorker {
  beginGame(): Promise<void> {
    return Promise.resolve();
  }
  override ask(_battle: string, query: Parameters<PolicyClient["ask"]>[1]): ReturnType<PolicyClient["ask"]> {
    const open = query.candidates.filter((c) => !c.pending).map((c) => c.choice);
    const pending = query.candidates.filter((c) => c.pending).map((c) => c.choice);
    this.queries.push({ kind: "ask", log_lines: 0, candidates: 0, rows: false });
    return Promise.resolve({ schema_version: 1 as const, identity: query.identity, choice: actorRule({ choices: open, pending }), source_policy_id: query.policy_id, elapsed_ms: 0, deadline_outcome: "in_budget" as const });
  }
}

const options = (change: Partial<ProgressOptions> = {}): ProgressOptions => ({
  engine,
  config,
  generation: 1,
  generation_manifest_sha256: "a".repeat(64),
  checkpoints,
  policy_ids: { sample: `policy:${"c".repeat(64)}`, argmax: `policy:${"d".repeat(64)}` },
  client: new Worker(),
  pool,
  games: 2,
  record_games: 2,
  training_teams: new Set(),
  league_seeds: new Set(),
  preview_solve: { setup: setupFor(), mode: "argmax" },
  ...change,
});

describe("the Team Preview solver candidate of the progress evaluation", () => {
  it("plays the baselines and the no-solve control, counts both sides and reruns to the same bytes", async () => {
    const first = await playProgress(options());
    const named = previewSolvePolicy(setupFor(), manifest, checkpoints, "argmax");
    const { record } = first;
    expect(record.matches.map((m) => m.match_id)).toEqual(["psolve-actor-rule-argmax", "psolve-random-legal-argmax", "psolve-self-no-solve-argmax"]);
    expect(record.plan.preview_solve_budget_ms).toBe(smallConfig().budget_ms);
    expect(record.plan.latency_budget_ms).toEqual({ preview_ms: config.generation.latency_budget_ms.preview_ms, battle_ms: config.generation.latency_budget_ms.battle_ms });
    for (const match of record.matches) {
      expect(match.policy_id).toBe(named.policy_id);
      expect(match.mode).toBe("argmax");
      expect(match.solve?.identity_hash).toBe(named.identity_hash);
      expect(match.solve?.counts.decisions).toBeGreaterThan(0);
      expect(match.solve?.counts.previews).toBe(2); // one Team Preview per game
      expect(match.solve?.counts.solved).toBe(2);
    }
    const control = record.matches[2];
    expect(control?.opponent).toEqual({ kind: "self-no-solve", policy_id: `policy:${"d".repeat(64)}`, checkpoints, mode: "argmax" });
    expect(control?.opponent_solve?.identity_hash).toBe(named.identity_hash);
    expect(control?.opponent_solve?.counts.solved).toBe(0);
    expect(control?.opponent_solve?.counts.previews).toBe(2);
    expect(control?.opponent_solve?.counts.decisions).toBeGreaterThan(0);
    expect(record.matches[0]?.opponent_solve).toBeUndefined();
    const stored = [...first.game_records.values()].filter((g) => g.match_id === "psolve-self-no-solve-argmax");
    expect(stored).toHaveLength(2);
    for (const game of stored) {
      expect(game.solve?.seat).toBe(game.policies.p1 === named.policy_id ? "p1" : "p2");
      expect(game.opponent_solve?.seat).not.toBe(game.solve?.seat);
      expect(readProgressGameRecord(JSON.parse(jcs(game as unknown as Json)))).toEqual(game);
    }
    const second = await playProgress(options());
    expect(jcs(second.record as unknown as Json)).toBe(jcs(record as unknown as Json));
  }, 900_000);

  it("runs the games with a Team Preview budget of at least the config's, and leaves a plain record's plan as it was", async () => {
    // A config budget below the run's own leaves the run's preview budget in force; the record still names the solver's.
    const small = await playProgress(options({ preview_solve: { setup: setupFor((c) => (c.budget_ms = 1)), mode: "argmax" }, games: 2, record_games: 0 }));
    expect(small.record.plan.preview_solve_budget_ms).toBe(1);
    const without = options({ games: 2, record_games: 0 });
    delete without.preview_solve;
    const plain = await playProgress(without);
    expect(plain.record.plan.preview_solve_budget_ms).toBeUndefined();
    expect(plain.record.matches.every((m) => m.solve === undefined && m.opponent.kind !== "self-no-solve")).toBe(true);
  }, 900_000);

  it("plays a sampled solver with a temperature: another policy id, the same counts shape, both seats", async () => {
    const play = { temperature_ppm: 500_000 };
    const { record } = await playProgress(options({ preview_solve: { setup: setupFor(), mode: "sample", play }, games: 2, record_games: 0 }));
    const named = previewSolvePolicy(setupFor(), manifest, checkpoints, "sample", play);
    expect(record.matches.map((m) => m.match_id)).toEqual(["psolve-actor-rule-sample", "psolve-random-legal-sample", "psolve-self-no-solve-sample"]);
    expect(record.matches[0]?.policy_id).toBe(named.policy_id);
    expect(record.matches[2]?.opponent_solve?.counts.solved).toBe(0);
    expect(record.matches[0]?.solve?.counts.solved).toBe(2);
  }, 900_000);

  it("is refused beside the ablation, a search, anchors, a previous generation, extra modes, a features or sheet pair, and in an unknown mode", async () => {
    const other = { checkpoints, policy_ids: { sample: `policy:${"e".repeat(64)}`, argmax: `policy:${"f".repeat(64)}` }, client: new Worker() };
    const ablation = { generation: { generation: 0, checkpoint: "3".repeat(64), policy_id: `policy:${"e".repeat(64)}`, client: new Worker() }, random_policy_id: `policy:${"f".repeat(64)}` };
    await expect(playProgress(options({ ablation }))).rejects.toThrow(/evaluated alone/);
    await expect(playProgress(options({ previous: other }))).rejects.toThrow(/evaluated alone/);
    await expect(playProgress(options({ anchors: [{ generation: 0, ...other }], anchor_games: 2 }))).rejects.toThrow(/evaluated alone/);
    await expect(playProgress(options({ search: { setup: {} as never, mode: "argmax" } }))).rejects.toThrow(/evaluated alone/);
    await expect(playProgress(options({ extra_modes: [{ play: { temperature_ppm: 500_000 }, policy_id: `policy:${"9".repeat(64)}` }] }))).rejects.toThrow(/extra play modes are not played/);
    await expect(playProgress(options({ features: {} as never }))).rejects.toThrow(/does not solve a features pair/);
    await expect(playProgress(options({ sheet: {} as never }))).rejects.toThrow(/opponent_sheet is not solved/);
    await expect(playProgress(options({ preview_solve: { setup: setupFor(), mode: "greedy" as never } }))).rejects.toThrow(ProgressError);
  });

  it("refuses a validation team that is a dev or training team of the run's own dataset", async () => {
    const own = pool.teams[0]?.canonical_team_hash as string;
    await expect(playProgress(options({ training_teams: new Set([own]) }))).rejects.toThrow(/dev or training team of the run's own dataset/);
  });
});

describe("the command's solver flag (usage errors come before any worker)", () => {
  const dir = mkdtempSync(join(tmpdir(), "progress-solve-cli-"));
  afterAll(() => rmSync(dir, { recursive: true, force: true }));
  const configFile = join(dir, "config.json");
  const teamSet = join(dir, "team-set.json");
  writeFileSync(configFile, JSON.stringify({ ...tiny, team_dataset: { path: SOURCE, dataset_hash: verifiedDatasetHash(engine, SOURCE, FORMAT) } }));
  writeFileSync(teamSet, JSON.stringify({ teams: [] }));
  const flags = (...more: string[]): string[] => [
    ...["--config", configFile, "--generation", "1", "--preview", "1".repeat(64), "--battle", "2".repeat(64)],
    ...["--policy-id-sample", `policy:${"c".repeat(64)}`, "--policy-id-argmax", `policy:${"d".repeat(64)}`, "--generation-manifest-sha256", "a".repeat(64)],
    ...["--team-set", teamSet, "--run-dir", dir, "--engine-data", join(dir, "none.json"), "--encoder-version", "enc", "--worker-python", "python", "--games", "2", "--record-games", "0", "--out", join(dir, "out")],
    ...more,
  ];
  const run = (more: string[], ports = {}) => runProgressCli(flags(...more), () => undefined, datasetValidationPool, ports);
  const solve = () => setupFor();

  it("refuses an unknown mode, a solver beside anchors, previous flags, the ablation, a search or extra modes, and a command with no solver setup", async () => {
    await expect(run(["--preview-solve", "greedy"], { preview_solve: solve })).rejects.toThrow(/argmax or sample/);
    await expect(run(["--preview-solve", "argmax", "--anchor", "0,a,b,c,d", "--anchor-games", "2"], { preview_solve: solve })).rejects.toThrow(/evaluated alone/);
    await expect(run(["--preview-solve", "argmax", "--ablation-preview", "0,a,c", "--ablation-random-policy-id", "r"], { preview_solve: solve })).rejects.toThrow(/evaluated alone/);
    await expect(run(["--preview-solve", "argmax", "--search-mode", "argmax"], { preview_solve: solve, search: () => ({}) as never })).rejects.toThrow(/evaluated alone/);
    await expect(run(["--preview-solve", "argmax", "--extra-mode", "sample:t500000,policy:x"], { preview_solve: solve })).rejects.toThrow(/extra play modes are not played/);
    await expect(run(["--preview-solve", "argmax"])).rejects.toThrow(/no solver setup/);
  });
});

describe("the composition root", () => {
  it("builds the committed setup for the committed format only", async () => {
    const { progressPreviewSolveSetup } = await import("../src/progress-preview-solve-setup.ts");
    const built = progressPreviewSolveSetup(engine, FORMAT);
    expect(built.config).toEqual(loadPreviewSolveConfig(FORMAT));
    expect(built.data_class).toBe("synthetic");
    expect(() => progressPreviewSolveSetup(engine, "gen9other")).toThrow(/vocabulary for/);
  });
});
