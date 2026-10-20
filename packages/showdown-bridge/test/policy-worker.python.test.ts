// The real Python worker (ml/areneon_ml/worker, stub scorer) behind the TS client, A1 battles on top.
// It needs the project's Python environment (`make setup`, uv sync); without it the file is skipped,
// as the typescript CI job has none. The python job covers the worker on its own (tests/ml).
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import type { PrivateTeamSpec } from "@areneon/schema/types/v1/private-team-spec.ts";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { engineDataHash, readEngineData } from "../src/engine-data.ts";
import { guardActor, playGame } from "../src/evaluation-harness.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT, GOLDEN_SEED, syntheticFixtureHashes, syntheticTeam } from "../src/golden.ts";
import { type BattleRecord, loadEngine } from "../src/index.ts";
import { resolveFormat } from "../src/formats.ts";
import { policyActor } from "../src/policy-actor.ts";
import { PolicyHandshakeError, PolicyWorkerClient, type PolicyWorkerOptions } from "../src/policy-worker.ts";
import { battleClass, stampTeam, teamHash } from "../src/team-class.ts";

const ROOT = fileURLToPath(new URL("../../../", import.meta.url));
const PYTHON = join(ROOT, ".venv/bin/python");
const sha = (data: Buffer | string): string => createHash("sha256").update(data).digest("hex");

describe.skipIf(!existsSync(PYTHON))("the Python policy worker behind the TS client", () => {
  const engine = loadEngine();
  const known = syntheticFixtureHashes(engine);
  const sets = syntheticTeam();
  const team = stampTeam(sets as PrivateTeamSpec["team"], known);
  const manifest = resolveFormat(engine, GOLDEN_FORMAT).manifest;
  const channel_class = battleClass({ p1: teamHash(sets), p2: teamHash(sets) }, known);
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

  const play = (client: PolicyWorkerClient, game: number) => {
    const actor = (player: "p1" | "p2") =>
      guardActor(
        policyActor({ client, battle: `g${game}-${player}`, player, team, manifest, channel_class, seed: GOLDEN_SEED, game, policy_id: `policy:${"a".repeat(64)}`, sampling: "sample" }),
        { role: "policy", latency_budget: { preview_ms: 30_000, battle_ms: 30_000 } },
      );
    return playGame({ engine, formatId: GOLDEN_FORMAT, seed: GOLDEN_SEED, teams: { p1: sets, p2: sets }, actors: { p1: actor("p1"), p2: actor("p2") } });
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
