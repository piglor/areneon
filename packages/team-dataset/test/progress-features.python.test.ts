// The progress evaluation of a pair that reads a candidate feature set (ADR-022, #1181), end to end on CPU with the
// real policy workers and the real probe: a features run and a no-features run on the fixture team dataset, the
// features pair evaluated against the baselines, an anchor, the previous generation and the no-features pair (the
// DF2 control, each seat with its own worker and statement). Needs the project's Python environment (skipped without
// it, unless ARENEON_REQUIRE_PYTHON_WORKER=1, which the typescript CI job sets).
import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { featureSetup, policyId, readProgressEvaluationRecordV2, readTrainingRunConfig, runConfigHash, runProgressCli, type ProgressEvaluationRecordV2 } from "@areneon/showdown-bridge";
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
const tree = (dir: string, prefix = ""): string[] => readdirSync(dir, { withFileTypes: true }).flatMap((e) => (e.isDirectory() ? tree(join(dir, e.name), `${prefix}${e.name}/`) : [`${prefix}${e.name}`]));

describe.skipIf(!existsSync(PYTHON) && process.env.ARENEON_REQUIRE_PYTHON_WORKER !== "1")("the progress evaluation of a pair that reads candidate features, tiny and on CPU", () => {
  let dir: string;
  let featuresRun: string;
  let plainRun: string;
  let hash = "";

  beforeAll(() => {
    dir = mkdtempSync(join(tmpdir(), "areneon-progress-features-"));
    const tiny = json<{ seeds: Record<string, unknown>; model: { encoder: { options?: unknown } } }>(ROOT, "fixtures/contracts/v1/training-run-config/valid/tiny.json");
    const dataset = { path: SOURCE, dataset_hash: verifiedDatasetHash(engine, SOURCE, FORMAT) };
    // The hash this checkout computes for the feature set is read from the refusal of a config that names a wrong one.
    const named = (feature_set_hash: string): unknown => ({ ...tiny, team_dataset: dataset, model: { ...tiny.model, encoder: { ...tiny.model.encoder, options: { candidate_features: { feature_set: "probe-damage/1", feature_set_hash } } } } });
    try {
      featureSetup(engine, readTrainingRunConfig(named("0".repeat(64))));
    } catch (error) {
      hash = /computes as ([0-9a-f]{64})/.exec((error as Error).message)?.[1] ?? "";
    }
    expect(hash).toMatch(/^[0-9a-f]{64}$/);
    const make = (name: string, config: unknown): string => {
      const file = join(dir, `${name}.json`);
      const run = join(dir, name);
      writeFileSync(file, JSON.stringify(config));
      loop("run", "--config", file, "--run-dir", run, "--engine-data", DATA, "--generations", "1", "--node", process.execPath, "--worker-python", PYTHON);
      return run;
    };
    featuresRun = make("features-run", named(hash));
    plainRun = make("plain-run", { ...tiny, seeds: { ...tiny.seeds, run_key: "plain-run", init_preview: 31, init_battle: 32 }, team_dataset: dataset });
  }, 900_000); // two tiny runs of one generation of six games each, the first with the real probe
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  const evaluate = (progressDir: string): void => {
    loop("progress", "--run-dir", featuresRun, "--engine-data", DATA, "--games", "4", "--record-games", "4", "--generation", "1", "--anchors", "0", "--anchor-games", "2", "--previous-matches", "--versus-run-dir", plainRun, "--versus-generation", "1", "--run-games", "2", "--progress-dir", progressDir, "--node", process.execPath, "--worker-python", PYTHON);
  };

  it("plays the features pair against baselines, anchor, previous generation and a no-features pair; ids equal training's; byte-identical on rerun", () => {
    const a = join(dir, "a.progress");
    evaluate(a);
    const record = readProgressEvaluationRecordV2(json<ProgressEvaluationRecordV2>(a, gen(1), "progress.json"));
    const pairOf = (run: string, g: number): { preview: string; battle: string } => json<{ checkpoints: { preview: string; battle: string } }>(run, "generations", gen(g), "manifest.json").checkpoints;
    const own = pairOf(featuresRun, 1);
    const plain = pairOf(plainRun, 1);
    expect(record.matches.map((m) => m.match_id)).toEqual(["actor-rule-argmax", "actor-rule-sample", "anchor-g0-argmax", "anchor-g0-sample", "previous-argmax", "previous-sample", "random-legal-argmax", "random-legal-sample", "run-argmax", "run-sample"]);
    for (const m of record.matches) {
      // The features pair's ids are the training ids (the `features` member); the other run's pair has none.
      expect(m.policy_id).toBe(policyId(own, m.mode, {}, hash));
      if (m.opponent.kind === "run") expect(m.opponent.policy_id).toBe(policyId(plain, m.mode));
      if (m.opponent.kind === "previous") expect(m.opponent.policy_id).toBe(policyId(pairOf(featuresRun, 0), m.mode, {}, hash));
      if (m.opponent.kind === "anchor") expect(m.opponent.policy_id).toBe(policyId(pairOf(featuresRun, 0), m.mode, {}, hash));
    }
    expect(policyId(own, "sample", {}, hash)).not.toBe(policyId(own, "sample"));
    // A seat whose rows were missing would have been refused by its worker and lost as a forfeit: none did.
    expect(record.games.filter((r) => r.outcome.kind === "forfeit")).toEqual([]);
    const b = join(dir, "b.progress");
    evaluate(b);
    expect(tree(join(b, gen(1)))).toEqual(tree(join(a, gen(1))));
    for (const file of tree(join(a, gen(1)))) expect(readFileSync(join(a, gen(1), file), "utf8"), file).toBe(readFileSync(join(b, gen(1), file), "utf8"));
  }, 1_800_000); // twice: eight matches of four games, two of two, with the real probe in the features seats

  it("evaluates a no-features pair against a features pair too (the other orientation of the control) and reads the record back", () => {
    const out = join(dir, "reverse.progress");
    loop("progress", "--run-dir", plainRun, "--engine-data", DATA, "--games", "2", "--record-games", "2", "--generation", "1", "--versus-run-dir", featuresRun, "--versus-generation", "1", "--run-games", "2", "--progress-dir", out, "--node", process.execPath, "--worker-python", PYTHON);
    const record = readProgressEvaluationRecordV2(json<ProgressEvaluationRecordV2>(out, gen(1), "progress.json"));
    const theirs = json<{ checkpoints: { preview: string; battle: string } }>(featuresRun, "generations", gen(1), "manifest.json").checkpoints;
    const runs = record.matches.filter((m) => m.opponent.kind === "run");
    expect(runs.map((m) => m.match_id)).toEqual(["run-argmax", "run-sample"]);
    for (const m of runs) expect(m.opponent.kind === "run" && m.opponent.policy_id).toBe(policyId(theirs, m.mode, {}, hash));
    expect(record.games.filter((r) => r.outcome.kind === "forfeit")).toEqual([]);
    // Reading the root back (a second command verifies and keeps it) recomputes what it can.
    loop("progress", "--run-dir", plainRun, "--engine-data", DATA, "--games", "2", "--record-games", "2", "--generation", "1", "--progress-dir", out, "--node", process.execPath, "--worker-python", PYTHON);
  }, 900_000); // one pair twice per match with the features seat running the real probe

  it("runs the Node command in process for both orientations, so the seat wiring of each run is exercised", async () => {
    const cli = async (name: string, own: string, other: string, ownFeatures: boolean, otherFeatures: boolean): Promise<void> => {
      const manifest = (run: string): { preview: string; battle: string } => json<{ checkpoints: { preview: string; battle: string } }>(run, "generations", gen(1), "manifest.json").checkpoints;
      const [mine, theirs] = [manifest(own), manifest(other)];
      const encoder = (run: string): string => json<{ encoder_version: string }>(run, "generations", gen(1), "records", "battle.json").encoder_version;
      const config = readTrainingRunConfig(json(other, "run-config.json"));
      const flags = [
        ["config", join(own, "run-config.json")], ["generation", "1"], ["preview", mine.preview], ["battle", mine.battle],
        ["policy-id-sample", policyId(mine, "sample", {}, ownFeatures ? hash : undefined)], ["policy-id-argmax", policyId(mine, "argmax", {}, ownFeatures ? hash : undefined)],
        ["generation-manifest-sha256", "c".repeat(64)], ["team-set", join(own, "generations", gen(1), "team-set.json")], ["run-dir", own], ["engine-data", DATA],
        ["encoder-version", encoder(own)], ["worker-python", PYTHON], ["games", "2"], ["record-games", "0"], ["out", join(dir, name)],
        ["versus", `${config.seeds.run_key},${await runConfigHash(config)},1,${theirs.preview},${theirs.battle},${policyId(theirs, "sample", {}, otherFeatures ? hash : undefined)},${policyId(theirs, "argmax", {}, otherFeatures ? hash : undefined)}`],
        ["versus-dir", other], ["versus-encoder-version", encoder(other)], ["versus-team-set", join(other, "generations", gen(1), "team-set.json")], ["run-games", "2"],
      ].flatMap(([k, v]) => [`--${k}`, v as string]);
      const lines: string[] = [];
      expect(await runProgressCli(flags, (l) => lines.push(l), datasetValidationPool)).toBe(0);
      const record = readProgressEvaluationRecordV2(json<ProgressEvaluationRecordV2>(join(dir, name), "progress.json"));
      expect(record.games.filter((r) => r.outcome.kind === "forfeit")).toEqual([]);
    };
    await cli("inproc-features", featuresRun, plainRun, true, false);
    await cli("inproc-plain", plainRun, featuresRun, false, true);
  }, 900_000); // four short matches, the features seats running the real probe
});
