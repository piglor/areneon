// The real Python worker (ml/areneon_ml/worker, stub scorer) behind the TS client, A1 battles on top.
// It needs the project's Python environment (`make setup`, uv sync); without it the file is skipped,
// unless ARENEON_REQUIRE_PYTHON_WORKER=1. The typescript CI job installs the worker's environment without torch and requires this file to run.
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { engineDataHash, readEngineData } from "../src/engine-data.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT, GOLDEN_SEED, syntheticFixtureHashes, syntheticTeam } from "../src/golden.ts";
import { type BattleRecord, loadEngine } from "../src/index.ts";
import { policyGame } from "../src/policy-actor.ts";
import { PolicyHandshakeError, PolicyWorkerClient, type PolicyWorkerOptions } from "../src/policy-worker.ts";

const ROOT = fileURLToPath(new URL("../../../", import.meta.url));
const PYTHON = join(ROOT, ".venv/bin/python");
const sha = (data: Buffer | string): string => createHash("sha256").update(data).digest("hex");

// CI sets ARENEON_REQUIRE_PYTHON_WORKER, so a missing environment fails there instead of skipping.
describe.skipIf(!existsSync(PYTHON) && process.env.ARENEON_REQUIRE_PYTHON_WORKER !== "1")("the Python policy worker behind the TS client", () => {
  const engine = loadEngine();
  const known = syntheticFixtureHashes(engine);
  const sets = syntheticTeam();
  const dataFile = join(GOLDEN_DIR, "engine-data.json");
  const expectedHash = engineDataHash(readEngineData(JSON.parse(readFileSync(dataFile, "utf8"))));
  const maxLine = 4_000_000;
  let runDir: string;
  let preview: string;
  let battle: string;

  beforeAll(() => {
    runDir = mkdtempSync(join(tmpdir(), "areneon-run-"));
    mkdirSync(join(runDir, "checkpoints"));
    const ids = ["preview", "battle"].map((role) => {
      const bytes = Buffer.from(`stub weights for the ${role} policy`);
      writeFileSync(join(runDir, "checkpoints", `${sha(bytes)}.safetensors`), bytes);
      return sha(bytes);
    });
    [preview, battle] = ids as [string, string];
  });
  afterAll(() => rmSync(runDir, { recursive: true, force: true }));

  const options = (change: Partial<PolicyWorkerOptions["expect"]> = {}): PolicyWorkerOptions => ({
    command: PYTHON,
    args: [
      "-I", "-m", "areneon_ml.worker", "--run-dir", runDir, "--preview", preview, "--battle", battle,
      "--engine-data", dataFile, "--max-line-bytes", String(maxLine),
    ],
    env: {},
    max_line_bytes: maxLine,
    n_parallel: 2,
    handshake_timeout_ms: 30_000,
    expect: {
      encoder_version: "stub-1",
      engine_data_hash: expectedHash,
      checkpoints: [
        { role: "battle", checkpoint_id: battle },
        { role: "preview", checkpoint_id: preview },
      ],
      ...change,
    },
  });

  const play = async (client: PolicyWorkerClient, game: number) => {
    await client.beginGame();
    const seat = { client, policy_id: `policy:${"a".repeat(64)}`, sampling: "sample" as const };
    return policyGame({
      engine, formatId: GOLDEN_FORMAT, seed: GOLDEN_SEED, game, teams: { p1: sets, p2: sets }, known_synthetic: known,
      seats: { p1: seat, p2: seat }, latency_budget: { preview_ms: 30_000, battle_ms: 30_000 },
    });
  };

  it("plays A1 battles against the stub model, and a game reproduces from its seed and index", async () => {
    const client = new PolicyWorkerClient(options());
    await client.start();
    try {
      const first = await play(client, 1);
      const again = await play(client, 1);
      const other = await play(client, 3);
      expect(first.outcome.kind).toBe("played");
      expect((again.record as BattleRecord).omniscient_log).toEqual((first.record as BattleRecord).omniscient_log);
      expect((other.record as BattleRecord).choices).not.toEqual((first.record as BattleRecord).choices);
    } finally {
      await client.close();
    }
  }, 120_000);

  it("refuses a session whose engine data or checkpoints are not the expected ones", async () => {
    await expect(new PolicyWorkerClient(options({ engine_data_hash: "0".repeat(64) })).start()).rejects.toThrow(PolicyHandshakeError);
    await expect(new PolicyWorkerClient(options({ encoder_version: "stub-2" })).start()).rejects.toThrow(/encoder_version/);
  }, 60_000);
});
