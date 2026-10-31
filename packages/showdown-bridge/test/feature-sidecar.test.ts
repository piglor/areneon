// The play-time sidecar (#1271) is written by the game itself: with the post-game pass of #1185 turned into a no-op,
// the rows directory still holds one file per Episode of the pair, and the stream written from it is the serial path's.
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it, vi } from "vitest";
import { actorRule } from "../src/actor-rule.ts";
import { loadEngine } from "../src/engine.ts";
import { featureSetup, playGeneration, trainingTeamPool } from "../src/league-generation.ts";
import type { PolicyClient } from "../src/policy-worker.ts";
import { SAMPLES_DIR } from "../src/samples.ts";
import { readTrainingRunConfig } from "../src/training-run-config.ts";

vi.mock("../src/feature-recompute.ts", async (original) => ({ ...(await original<typeof import("../src/feature-recompute.ts")>()), writeFeatureRows: () => Promise.resolve(0) }));

const engine = loadEngine();
const tiny = JSON.parse(readFileSync(join(SAMPLES_DIR, "v1/training-run-config/valid/tiny.json"), "utf8")) as Record<string, unknown>;
const POLICY = `policy:${"c".repeat(64)}`;
const withSet = (hash: string) => {
  const raw = structuredClone(tiny) as { model: { encoder: { options?: unknown } } };
  raw.model.encoder.options = { candidate_features: { feature_set: "probe-damage/1", feature_set_hash: hash } };
  return readTrainingRunConfig(raw);
};

describe("the play-time sidecar is written during play", () => {
  const dir = mkdtempSync(join(tmpdir(), "areneon-sidecar-"));
  afterAll(() => rmSync(dir, { recursive: true, force: true }));
  it("leaves a rows file per Episode of the pair with no post-game pass", async () => {
    let hash = "";
    try {
      featureSetup(engine, withSet("0".repeat(64)));
    } catch (error) {
      hash = /computes as ([0-9a-f]{64})/.exec((error as Error).message)?.[1] as string;
    }
    const config = withSet(hash);
    const configs = (featureSetup(engine, config) as NonNullable<ReturnType<typeof featureSetup>>).configs;
    const client: PolicyClient & { beginGame: () => Promise<void>; featureSet: { feature_set: string; feature_set_hash: string } } = {
      beginGame: () => Promise.resolve(),
      featureSet: { feature_set: "probe-damage/1", feature_set_hash: hash },
      ask: (_battle, query) =>
        Promise.resolve({
          schema_version: 1 as const,
          identity: query.identity,
          choice: actorRule({ choices: query.candidates.filter((c) => !c.pending).map((c) => c.choice), pending: query.candidates.filter((c) => c.pending).map((c) => c.choice) }),
          source_policy_id: query.policy_id,
          elapsed_ms: 0,
          deadline_outcome: "in_budget" as const,
        }),
    };
    mkdirSync(join(dir, "episodes"), { recursive: true });
    const rows = join(dir, "rows");
    const result = await playGeneration({ engine, config, generation: 1, policy_id: POLICY, client, pool: trainingTeamPool(engine, config.generation.excluded_abilities), features: configs, stream_path: join(dir, "stream.jsonl"), episodes_dir: join(dir, "episodes"), features_dir: rows });
    expect(result.games.length).toBeGreaterThan(0);
    const own = readdirSync(join(dir, "episodes")).filter((name) => (JSON.parse(readFileSync(join(dir, "episodes", name), "utf8")) as { provenance: { behaviour_policy_id: string } }).provenance.behaviour_policy_id === POLICY);
    expect(own.length).toBeGreaterThan(0);
    expect(readdirSync(rows).sort()).toEqual(own.map((name) => name.replace(".json", ".rows.json")).sort());
  }, 600_000); // a tiny generation with the real probe at play time
});
