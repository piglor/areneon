// The search actor behind the REAL Python worker (ADR-026 sections 2 and 6): untrained, seeded tiny networks give a prior and a value
// by `evaluate`, batch 1 on CPU, and a seeded A1 game is reproduced from its seed with the same choices and counts. It needs the
// project's Python environment (`make setup`); without it the file is skipped unless ARENEON_REQUIRE_PYTHON_WORKER=1 (the
// typescript CI job sets it). The pair's checkpoints are written by `python -m areneon_ml.checkpoint init`, as in the bridge's
// policy-worker test.
import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  battleClass,
  engineDataHash,
  type PolicyWorkerOptions,
  PolicyWorkerClient,
  readEngineData,
  runBattle,
  searchActor,
  searchPolicy,
  stampTeam,
  syntheticTeamHashes,
  teamHash,
} from "@areneon/showdown-bridge";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { firstLegal, GOLDEN_DIR } from "../../showdown-bridge/src/golden.ts";
import { loadRandomCorpus } from "../../showdown-bridge/src/random-corpus.ts";
import { engine, FORMAT } from "./support.ts";
import { manifest, setupFor } from "./search-actor-support.ts";

const ROOT = fileURLToPath(new URL("../../../", import.meta.url));
const PYTHON = join(ROOT, ".venv/bin/python");
const TINY_MODEL = {
  encoder: { volatile_buckets: 4, condition_buckets: 4, max_moves: 4, max_team: 6, max_picks: 4, stat_scale: 500, turn_scale: 100 },
  d_model: 16,
  n_layers: 1,
  n_heads: 2,
  d_ff: 32,
};

describe.skipIf(!existsSync(PYTHON) && process.env.ARENEON_REQUIRE_PYTHON_WORKER !== "1")("the search actor behind the Python worker", () => {
  const dataFile = join(GOLDEN_DIR, "engine-data.json");
  const expectedHash = engineDataHash(readEngineData(JSON.parse(readFileSync(dataFile, "utf8"))));
  const maxLine = 8_000_000;
  let runDir: string;
  let preview: string;
  let battle: string;
  let encoderVersion: string;

  beforeAll(() => {
    runDir = mkdtempSync(join(tmpdir(), "areneon-search-run-"));
    const configFile = join(runDir, "model.json");
    writeFileSync(configFile, JSON.stringify(TINY_MODEL));
    const init = (role: string, seed: number): { checkpoint_id: string; encoder_version: string } =>
      JSON.parse(execFileSync(PYTHON, ["-I", "-m", "areneon_ml.checkpoint", "init", "--role", role, "--seed", String(seed), "--engine-data", dataFile, "--model-config", configFile, "--run-dir", runDir], { encoding: "utf8", env: {} }));
    const [first, second] = [init("preview", 1), init("battle", 2)];
    preview = first.checkpoint_id;
    battle = second.checkpoint_id;
    encoderVersion = first.encoder_version;
  }, 120_000);
  afterAll(() => rmSync(runDir, { recursive: true, force: true }));

  const options = (): PolicyWorkerOptions => ({
    command: PYTHON,
    args: ["-I", "-m", "areneon_ml.worker", "--run-dir", runDir, "--preview", preview, "--battle", battle, "--engine-data", dataFile, "--max-line-bytes", String(maxLine)],
    env: {},
    max_line_bytes: maxLine,
    n_parallel: 1,
    handshake_timeout_ms: 60_000,
    expect: { encoder_version: encoderVersion, engine_data_hash: expectedHash, checkpoints: [{ role: "battle", checkpoint_id: battle }, { role: "preview", checkpoint_id: preview }] },
  });

  it("searches against the networks' prior and value, and a seeded game reproduces with the same choices and counts", async () => {
    const source = loadRandomCorpus()[1];
    if (source === undefined) throw new Error("no corpus battle");
    const teams = source.teams;
    const known = syntheticTeamHashes([teamHash(teams.p1), teamHash(teams.p2)]);
    const setup = setupFor();
    const { policy_id, identity_hash } = searchPolicy(setup, manifest, { preview, battle }, "argmax");
    const client = new PolicyWorkerClient(options());
    await client.start();
    try {
      const play = async () => {
        await client.beginGame();
        const actor = searchActor({
          setup,
          client,
          battle: "game-0-p1",
          player: "p1",
          team: stampTeam(teams.p1 as never, known),
          manifest,
          channel_class: battleClass({ p1: teamHash(teams.p1), p2: teamHash(teams.p2) }, known),
          seed: source.seed,
          game: 0,
          policy_id,
          identity_hash,
          sampling: "argmax",
        });
        const record = await runBattle({ engine, formatId: FORMAT, seed: source.seed, teams, actors: { p1: actor, p2: firstLegal }, maxTurns: 3 });
        return { record, counts: actor.report().counts };
      };
      const first = await play();
      const again = await play();
      expect(first.counts.searched_decisions).toBeGreaterThan(0);
      expect(first.counts.leaf_queries).toBeGreaterThan(0);
      expect(again.record.choices).toEqual(first.record.choices);
      expect(again.counts).toEqual(first.counts);
    } finally {
      await client.close();
    }
  }, 300_000);
});
