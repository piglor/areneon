// The progress evaluation of a pair that reads the opponent sheet (ADR-027 slice 5, ADR-025 erratum E4), end to end on CPU with the
// real policy workers and the real opponent model: a sheet run and a plain run on the fixture team dataset, the sheet pair evaluated
// against the baselines and the plain pair (each seat with its own worker and statement), and the plain pair against the sheet pair.
// Needs the project's Python environment (skipped without it, unless ARENEON_REQUIRE_PYTHON_WORKER=1, which the typescript CI job sets).
import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadRulesSnapshot, readProgressEvaluationRecordV2, resolveFormat, type ProgressEvaluationRecordV2 } from "@areneon/showdown-bridge";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { readerSha256 } from "../../public-state/src/index.ts";
import { goldenTeamVocabulary } from "../../showdown-bridge/src/golden.ts";
import { verifiedDatasetHash } from "../src/dataset-hash.ts";
import { createBeliefFeatures } from "../src/belief-features.ts";
import { DATASETS_DIR } from "../src/fixtures.ts";
import { engine, FORMAT } from "./support.ts";

const ROOT = fileURLToPath(new URL("../../../", import.meta.url));
const PYTHON = join(ROOT, ".venv/bin/python");
const DATA = join(ROOT, "fixtures/showdown", FORMAT, "engine-data.json");
const SOURCE = join(DATASETS_DIR(FORMAT), "fixtures-v1");
const env = { ...(process.env.SHOWDOWN_DIR ? { SHOWDOWN_DIR: process.env.SHOWDOWN_DIR } : {}) };
const loop = (...args: string[]): string => execFileSync(PYTHON, ["-I", "-m", "areneon_ml.loop", ...args], { encoding: "utf8", env, cwd: ROOT });
const gen = (n: number): string => `g${String(n).padStart(6, "0")}`;
const json = <T>(...path: string[]): T => JSON.parse(readFileSync(join(...path), "utf8")) as T;

describe.skipIf(!existsSync(PYTHON) && process.env.ARENEON_REQUIRE_PYTHON_WORKER !== "1")("the progress evaluation of a pair that reads the opponent sheet, tiny and on CPU", () => {
  let dir: string;
  let sheetRun: string;
  let plainRun: string;

  beforeAll(() => {
    dir = mkdtempSync(join(tmpdir(), "areneon-progress-sheet-"));
    const tiny = json<{ seeds: Record<string, unknown>; model: { encoder: { options?: unknown } } }>(ROOT, "fixtures/contracts/v1/training-run-config/valid/tiny.json");
    const dataset = { path: SOURCE, dataset_hash: verifiedDatasetHash(engine, SOURCE, FORMAT) };
    const manifest = resolveFormat(engine, FORMAT, loadRulesSnapshot(engine.commit)).manifest;
    const hash = createBeliefFeatures({ engine, vocabulary: goldenTeamVocabulary(), reader_sha256: readerSha256(), manifest }).features_hash;
    const make = (name: string, config: unknown): string => {
      const file = join(dir, `${name}.json`);
      const run = join(dir, name);
      writeFileSync(file, JSON.stringify(config));
      loop("run", "--config", file, "--run-dir", run, "--engine-data", DATA, "--generations", "1", "--node", process.execPath, "--worker-python", PYTHON);
      return run;
    };
    sheetRun = make("sheet-run", { ...tiny, team_dataset: dataset, model: { ...tiny.model, encoder: { ...tiny.model.encoder, options: { opponent_sheet: { feature_set: "opponent-sheet/1", feature_set_hash: hash } } } } });
    plainRun = make("plain-run", { ...tiny, seeds: { ...tiny.seeds, run_key: "plain-run", init_preview: 31, init_battle: 32 }, team_dataset: dataset });
  }, 900_000); // two tiny runs of one generation of six games each, the first with the real belief
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  const evaluate = (run: string, versus: string, out: string): ProgressEvaluationRecordV2 => {
    loop("progress", "--run-dir", run, "--engine-data", DATA, "--games", "4", "--record-games", "2", "--generation", "1", "--versus-run-dir", versus, "--versus-generation", "1", "--run-games", "2", "--progress-dir", out, "--node", process.execPath, "--worker-python", PYTHON);
    return readProgressEvaluationRecordV2(json<ProgressEvaluationRecordV2>(out, gen(1), "progress.json"));
  };

  it("plays the sheet pair against the baselines and a plain pair: no forfeit, the candidate's counts in every match", () => {
    const record = evaluate(sheetRun, plainRun, join(dir, "sheet.progress"));
    expect(record.matches.length).toBe(6);
    expect(record.games.filter((r) => r.outcome.kind === "forfeit")).toEqual([]); // a seat without its block would have been refused by its worker
    for (const m of record.matches) {
      expect(m.sheet?.requests, m.match_id).toBeGreaterThan(0);
      expect(record.games.filter((g) => g.match_id === m.match_id).every((g) => g.sheet !== undefined)).toBe(true);
    }
  }, 1_800_000);

  it("evaluates a plain pair against the sheet pair too: the plain pair has no counts, the other run's seat still gets its block", () => {
    const record = evaluate(plainRun, sheetRun, join(dir, "plain.progress"));
    expect(record.games.filter((r) => r.outcome.kind === "forfeit")).toEqual([]);
    expect(record.matches.every((m) => m.sheet === undefined)).toBe(true);
    expect(record.matches.filter((m) => m.opponent.kind === "run")).toHaveLength(2);
  }, 1_800_000);
});
