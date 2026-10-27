// The scripted head-to-head end to end (ADR-025 section 13, #1130): the Python command runs the Node script on the
// committed fixture dataset's validation pairs, logs a look and commits the record. Two runs, one process and two,
// give byte-identical records; the second appends to the same look log. It needs the project's Python environment;
// without it the file is skipped, unless ARENEON_REQUIRE_PYTHON_WORKER=1 (the typescript CI job sets it).
import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { H2H_FILE, readScriptedH2HRecord } from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it } from "vitest";
import { verifiedDatasetHash } from "../src/dataset-hash.ts";
import { DATASETS_DIR } from "../src/fixtures.ts";
import { engine, FORMAT } from "./support.ts";

const ROOT = fileURLToPath(new URL("../../../", import.meta.url));
const PYTHON = join(ROOT, ".venv/bin/python");
const SOURCE = join(DATASETS_DIR(FORMAT), "fixtures-v1");
const env = { ...(process.env.SHOWDOWN_DIR ? { SHOWDOWN_DIR: process.env.SHOWDOWN_DIR } : {}) };

describe.skipIf(!existsSync(PYTHON) && process.env.ARENEON_REQUIRE_PYTHON_WORKER !== "1")("the scripted head-to-head, tiny", () => {
  const dir = mkdtempSync(join(tmpdir(), "areneon-h2h-py-"));
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  it("is byte-identical for one and two game processes and logs each run as a look", () => {
    const tiny = JSON.parse(readFileSync(join(ROOT, "fixtures/contracts/v1/training-run-config/valid/tiny.json"), "utf8")) as Record<string, unknown>;
    const configFile = join(dir, "config.json");
    writeFileSync(configFile, JSON.stringify({ ...tiny, generation: { ...(tiny.generation as object), max_turns: 100 }, team_dataset: { path: SOURCE, dataset_hash: verifiedDatasetHash(engine, SOURCE, FORMAT) } }));
    const looks = join(dir, "h2h.looks.jsonl");
    const run = (name: string, processes: string): { record_hash: string; summary: { a: { low_ppm: number; high_ppm: number }; pairs: number }; looks: number } =>
      JSON.parse(
        execFileSync(
          PYTHON,
          ["-I", "-m", "areneon_ml.loop", "scripted-h2h", "--config", configFile, "--arm-a", "actor-rule@v1", "--arm-b", "damage-aware@v1", "--pairs", "2", "--out", join(dir, name), "--game-processes", processes, "--looks", looks, "--node", process.execPath],
          { encoding: "utf8", env, cwd: ROOT },
        ),
      );
    const one = run("one", "1");
    const two = run("two", "2");
    const bytes = (name: string): string => readFileSync(join(dir, name, H2H_FILE), "utf8");
    expect(bytes("two")).toBe(bytes("one"));
    expect(two.record_hash).toBe(one.record_hash);
    expect(readScriptedH2HRecord(JSON.parse(bytes("one"))).plan.games).toBe(4);
    expect(one.summary.pairs).toBe(2);
    expect(one.summary.a.low_ppm).toBeLessThanOrEqual(one.summary.a.high_ppm);
    expect([one.looks, two.looks]).toEqual([1, 2]);
    expect(readFileSync(looks, "utf8").trim().split("\n")).toHaveLength(2);
  }, 300_000);
});
