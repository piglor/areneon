// The real Python worker (ml/areneon_ml/worker, real networks) behind the TS client, A1 battles on top.
// It needs the project's Python environment (`make setup`, uv sync); without it the file is skipped,
// unless ARENEON_REQUIRE_PYTHON_WORKER=1. The typescript CI job installs the worker's environment (CPU torch) and requires this file to run.
// The checkpoints are seeded, untrained tiny networks written by `python -m areneon_ml.checkpoint init`.
import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { engineDataHash, readEngineData } from "../src/engine-data.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT, GOLDEN_SEED, syntheticFixtureHashes, syntheticTeam } from "../src/golden.ts";
import { type BattleRecord, buildPayoffTable, importPaste, loadEngine, type PayoffConfig, startHypothetical } from "../src/index.ts";
import { TEAMS_DIR } from "../src/golden.ts";
import { policyGame } from "../src/policy-actor.ts";
import { SAMPLES_DIR } from "../src/samples.ts";
import type { EvaluateQueryInput, PolicyQueryInput } from "../src/policy-protocol.ts";
import { PolicyHandshakeError, PolicyWorkerClient, type PolicyWorkerOptions } from "../src/policy-worker.ts";

const ROOT = fileURLToPath(new URL("../../../", import.meta.url));
const PYTHON = join(ROOT, ".venv/bin/python");
// Tiny sizes keep the test fast; sizes are configuration (ADR-010 section 3).
const TINY_MODEL = {
  encoder: { volatile_buckets: 4, condition_buckets: 4, max_moves: 4, max_team: 6, max_picks: 4, stat_scale: 500, turn_scale: 100 },
  d_model: 16,
  n_layers: 1,
  n_heads: 2,
  d_ff: 32,
};

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
  let encoderVersion: string;

  beforeAll(() => {
    runDir = mkdtempSync(join(tmpdir(), "areneon-run-"));
    const configFile = join(runDir, "model.json");
    writeFileSync(configFile, JSON.stringify(TINY_MODEL));
    const init = (role: string, seed: number): { checkpoint_id: string; encoder_version: string } =>
      JSON.parse(
        execFileSync(PYTHON, ["-I", "-m", "areneon_ml.checkpoint", "init", "--role", role, "--seed", String(seed), "--engine-data", dataFile, "--model-config", configFile, "--run-dir", runDir], {
          encoding: "utf8",
          env: {},
        }),
      );
    const [first, second] = [init("preview", 1), init("battle", 2)];
    preview = first.checkpoint_id;
    battle = second.checkpoint_id;
    encoderVersion = first.encoder_version;
  }, 120_000);
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
      encoder_version: encoderVersion,
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

  it("plays A1 battles against the networks, and a game reproduces from its seed and index", async () => {
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

  // ADR-026 section 5: evaluate answers a prior and a value from a batch-1 CPU pass, the same integers however often and after whatever.
  it("answers evaluate queries with bit-identical integers, alone, repeated and after other queries, for the A1 and A2 forms", async () => {
    const load = (name: string): PolicyQueryInput & { query_id: number } => JSON.parse(readFileSync(`${SAMPLES_DIR}/v1/policy-query/valid/${name}.json`, "utf8"));
    const { query_id: _a, ...a1 } = load("battle-evaluate");
    const { query_id: _b, ...a2 } = load("battle-a2");
    const evaluateA2 = { ...a2, sampling: { kind: "evaluate" as const } };
    const evaluateA1 = a1 as EvaluateQueryInput;
    const client = new PolicyWorkerClient(options());
    await client.start();
    try {
      const alone = await client.evaluate("g1", evaluateA1);
      await client.ask("g1", { ...a1, sampling: { kind: "argmax" } });
      const { query_id: _c, ...preview } = load("preview-evaluate");
      await client.evaluate("g1", preview as EvaluateQueryInput);
      const again = await client.evaluate("g1", evaluateA1);
      const other = await client.evaluate("g1", evaluateA2);
      expect(again).toEqual(alone);
      expect(alone.prior_ppm).toHaveLength(a1.candidates.length);
      expect(other.prior_ppm).toEqual(alone.prior_ppm);
      expect(other.value_ppm).toBe(alone.value_ppm);
    } finally {
      await client.close();
    }
  }, 120_000);

  // ADR-029 section 3 (#1256): the payoff table of a Team Preview decision from the real networks is the same integers on a rerun, for both leaves.
  it("builds bit-identical Team Preview payoff tables on a rerun, first-request and rollout leaves", async () => {
    const opponent = importPaste(engine, readFileSync(join(TEAMS_DIR, "legal-opponent.txt"), "utf8"));
    const channel = JSON.parse(readFileSync(join(GOLDEN_DIR, "channels/p1.json"), "utf8")) as { lines: string[] };
    const own_request = (channel.lines.find((line) => line.startsWith("|request|")) as string).slice("|request|".length);
    const tables = async (leaf: PayoffConfig["leaf"]) => {
      const client = new PolicyWorkerClient(options());
      await client.start();
      try {
        const worlds = [0, 1].map((index) => ({
          index,
          preview: startHypothetical({ engine, format_id: GOLDEN_FORMAT, seat: "p1", own_team: sets, world_team: opponent, stream: new Uint8Array(32).fill(5), world: index, data_class: "synthetic", own_request }),
        }));
        const config: PayoffConfig = { k_own: 2, k_opp: 2, own_rule: "distinct-leads", leaf };
        return await buildPayoffTable({ config, worlds, client, battle: "g1", policy_id: `policy:${"a".repeat(64)}`, own_request, stream: new Uint8Array(32).fill(5) });
      } finally {
        await client.close();
      }
    };
    for (const leaf of [{ kind: "first-request" }, { kind: "rollout", turns: 1 }] as const) {
      const first = await tables(leaf);
      const again = await tables(leaf);
      expect(first.complete).toBe(true);
      expect(first.counts.engine_refusals).toBe(0);
      expect(JSON.stringify(again)).toBe(JSON.stringify(first));
    }
  }, 240_000);

  it("refuses a session whose engine data or checkpoints are not the expected ones", async () => {
    await expect(new PolicyWorkerClient(options({ engine_data_hash: "0".repeat(64) })).start()).rejects.toThrow(PolicyHandshakeError);
    await expect(new PolicyWorkerClient(options({ encoder_version: `${encoderVersion}x` })).start()).rejects.toThrow(/encoder_version/);
  }, 60_000);
});
