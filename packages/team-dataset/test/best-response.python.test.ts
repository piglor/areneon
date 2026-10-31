// The best-response evaluation end to end on CPU (ADR-029 section 2, #1257): the Python driver makes a tiny run on the committed
// fixture team dataset, then the `best-response` command plays the four arms on the validation pairs through Node and the real
// policy worker, logs a look and commits the record. Two runs give byte-identical records. It needs the project's Python
// environment; without it the file is skipped, unless ARENEON_REQUIRE_PYTHON_WORKER=1 (the typescript CI job sets it).
import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { BEST_RESPONSE_FILE, jcs, loadRulesSnapshot, readBestResponseRecord, resolveFormat, runBestResponseCli, type Json, UsageError } from "@areneon/showdown-bridge";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { readerSha256 } from "../../public-state/src/index.ts";
import { goldenTeamVocabulary } from "../../showdown-bridge/src/golden.ts";
import { verifiedDatasetHash } from "../src/dataset-hash.ts";
import { DATASETS_DIR, loadFixtureInputs, readFixtureRun, recordedPilot, regenerate, writeFixtures } from "../src/fixtures.ts";
import { createOpponentTeams } from "../src/opponent-teams.ts";
import { datasetTrainingHashes, datasetValidationPool } from "../src/progress-pool.ts";
import { engine, FORMAT } from "./support.ts";

const ROOT = fileURLToPath(new URL("../../../", import.meta.url));
const PYTHON = join(ROOT, ".venv/bin/python");
const DATA = join(ROOT, "fixtures/showdown", FORMAT, "engine-data.json");
const SOURCE = join(DATASETS_DIR(FORMAT), "fixtures-v1");
const env = { ...(process.env.SHOWDOWN_DIR ? { SHOWDOWN_DIR: process.env.SHOWDOWN_DIR } : {}) };
const loop = (...args: string[]): string => execFileSync(PYTHON, ["-I", "-m", "areneon_ml.loop", ...args], { encoding: "utf8", env, cwd: ROOT });
const gen = (n: number): string => `g${String(n).padStart(6, "0")}`;
const json = <T>(...path: string[]): T => JSON.parse(readFileSync(join(...path), "utf8")) as T;
const manifest = resolveFormat(engine, FORMAT, loadRulesSnapshot(engine.commit)).manifest;
const ports = {
  run_teams: datasetTrainingHashes,
  teams: () => createOpponentTeams({ engine, vocabulary: goldenTeamVocabulary(), reader_sha256: readerSha256(), manifest, arm: "generator-matched" }),
};

describe.skipIf(!existsSync(PYTHON) && process.env.ARENEON_REQUIRE_PYTHON_WORKER !== "1")("the best-response evaluation, tiny and on CPU", () => {
  let dir: string;
  let runDir: string;
  let configFile: string;

  beforeAll(() => {
    dir = mkdtempSync(join(tmpdir(), "areneon-best-response-"));
    runDir = join(dir, "run");
    const tiny = json<Record<string, unknown>>(ROOT, "fixtures/contracts/v1/training-run-config/valid/tiny.json");
    configFile = join(dir, "config.json");
    writeFileSync(configFile, JSON.stringify({ ...tiny, generation: { ...(tiny.generation as object), max_turns: 30 }, team_dataset: { path: SOURCE, dataset_hash: verifiedDatasetHash(engine, SOURCE, FORMAT) } }));
    loop("run", "--config", configFile, "--run-dir", runDir, "--engine-data", DATA, "--generations", "1", "--node", process.execPath, "--worker-python", PYTHON);
  }, 240_000);
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  const command = (out: string, ...more: string[]): { record_hash: string; summary: { pairs: number; margin: { low_ppm: number; high_ppm: number }; label: string }; looks: number } =>
    JSON.parse(
      loop("best-response", "--run-dir", runDir, "--generation", "1", "--mode", "argmax", "--pairs", "1", "--worlds", "2", "--k-own", "4", "--own-rule", "distinct-leads", "--leaf", "first-request", "--out", out, "--engine-data", DATA, "--node", process.execPath, "--worker-python", PYTHON, "--looks", join(dir, "br.looks.jsonl"), ...more),
    );

  it("is byte-identical on a rerun, reads under both readers and logs each run as a look", () => {
    const one = command(join(dir, "one"));
    const two = command(join(dir, "two"));
    const bytes = (name: string): string => readFileSync(join(dir, name, BEST_RESPONSE_FILE), "utf8");
    expect(bytes("two")).toBe(bytes("one"));
    expect(two.record_hash).toBe(one.record_hash);
    const record = readBestResponseRecord(JSON.parse(bytes("one")));
    expect(record.plan.games).toBe(2);
    expect(record.evaluated.generation).toBe(1);
    expect(one.summary.pairs).toBe(1);
    expect(one.summary.margin.low_ppm).toBeLessThanOrEqual(one.summary.margin.high_ppm);
    expect(one.summary.label).toContain("white-box access");
    expect([one.looks, two.looks]).toEqual([1, 2]);
    expect(readFileSync(join(dir, "br.looks.jsonl"), "utf8").trim().split("\n")).toHaveLength(2);
  }, 300_000);

  it("takes the evaluation key and the rollout leaf and records them", () => {
    command(join(dir, "keyed"), "--evaluation-key", "e0p-key", "--mode", "sample");
    const record = readBestResponseRecord(JSON.parse(readFileSync(join(dir, "keyed", BEST_RESPONSE_FILE), "utf8")));
    expect(record.plan.evaluation_key).toBe("e0p-key");
    expect(record.evaluated.mode).toBe("sample");
  }, 300_000);

  describe("the command in this process", () => {
    const g1 = (): { checkpoints: Record<string, string> } => json(runDir, "generations", gen(1), "manifest.json");
    const flags = (out: string, change: Record<string, string> = {}): string[] =>
      Object.entries({
        config: join(runDir, "run-config.json"),
        generation: "1",
        preview: g1().checkpoints.preview as string,
        battle: g1().checkpoints.battle as string,
        mode: "argmax",
        "policy-id": `policy:${"a".repeat(64)}`,
        "generation-manifest-sha256": "c".repeat(64),
        "team-set": join(runDir, "generations", gen(1), "team-set.json"),
        "run-dir": runDir,
        "engine-data": DATA,
        "encoder-version": json<{ encoder_version: string }>(runDir, "generations", gen(1), "records", "battle.json").encoder_version,
        "worker-python": PYTHON,
        pairs: "1",
        worlds: "2",
        "k-own": "4",
        "own-rule": "distinct-leads",
        leaf: "first-request",
        "preview-budget-ms": "3600000",
        out,
        ...change,
      }).flatMap(([k, v]) => [`--${k}`, v]);

    it("plays and prints one canonical line, and refuses an existing record", async () => {
      const out = join(dir, "cli-ok");
      const lines: string[] = [];
      expect(await runBestResponseCli(flags(out), (l) => lines.push(l), datasetValidationPool, ports)).toBe(0);
      const printed = JSON.parse(lines[0] as string) as { games: number; record_hash: string };
      expect(lines[0]).toBe(jcs(printed as unknown as Json));
      expect(printed.games).toBe(2);
      const record = readBestResponseRecord(JSON.parse(readFileSync(join(out, BEST_RESPONSE_FILE), "utf8")));
      expect(record.plan.responder).toMatchObject({ worlds: 2, k_own: 4, own_rule: "distinct-leads" });
      // the record file is written once
      await expect(runBestResponseCli(flags(out), () => undefined, datasetValidationPool, ports)).rejects.toThrow(/EEXIST/);
    }, 300_000);

    it("plays on the validation split of another verified dataset and names it", async () => {
      // A second tiny dataset: the committed fixture's sizes and pilot under other keys.
      const second = join(dir, "second");
      const inputs = loadFixtureInputs(engine, FORMAT);
      writeFixtures(second, regenerate(inputs, { ...readFixtureRun(SOURCE), dataset_id: "eval-v1", generation_key: "11".repeat(32), assignment_key: "22".repeat(32) }, recordedPilot(inputs, SOURCE)));
      const hash = verifiedDatasetHash(engine, second, FORMAT);
      const out = join(dir, "cli-eval");
      expect(await runBestResponseCli(flags(out, { "eval-dataset-dir": second, "eval-dataset-hash": hash, "evaluation-key": "e0p-key" }), () => undefined, datasetValidationPool, ports)).toBe(0);
      const record = readBestResponseRecord(JSON.parse(readFileSync(join(out, BEST_RESPONSE_FILE), "utf8")));
      expect(record.teams.evaluation_dataset_hash).toBe(hash);
      expect(record.plan.evaluation_key).toBe("e0p-key");
    }, 300_000);

    it("refuses a pair that is not enough validation teams as an evaluation failure, exit 1", async () => {
      const said: string[] = [];
      const original = console.error;
      console.error = (m: unknown): void => void said.push(String(m));
      try {
        expect(await runBestResponseCli(flags(join(dir, "cli-many"), { pairs: "10000" }), () => undefined, datasetValidationPool, ports)).toBe(1);
      } finally {
        console.error = original;
      }
      expect(said.join("\n")).toMatch(/best-response: .*disjoint team pairs/);
    }, 120_000);

    it("refuses an evaluation dataset that does not hold the run's own teams apart, and a missing belief port", async () => {
      await expect(runBestResponseCli(flags(join(dir, "cli-port")), () => undefined, datasetValidationPool, {})).rejects.toThrow(UsageError);
      await expect(runBestResponseCli(flags(join(dir, "cli-eval"), { "eval-dataset-dir": SOURCE, "eval-dataset-hash": verifiedDatasetHash(engine, SOURCE, FORMAT) }), () => undefined, datasetValidationPool, ports)).rejects.toThrow(/own training dataset/);
    }, 120_000);
  });
});
