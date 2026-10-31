// The play-time rows sidecar (#1271) under `probe-damage-belief/1` with the real opponent model, the committed prior, the pinned
// validator and the real probe: for seeded games on the verified fixtures dataset, the file each game leaves is, byte for byte,
// the rows the stream writer would recompute from the stored Episode, and a stream written from the files with a probe that
// must not run is the stream written without them.
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { actorRule } from "../../showdown-bridge/src/actor-rule.ts";
import { type Json, jcs } from "../../showdown-bridge/src/canonical-json.ts";
import { decisionObservations, readEpisode } from "../../showdown-bridge/src/episode-io.ts";
import { readFeatureRows } from "../../showdown-bridge/src/feature-recompute.ts";
import { featureSetup, playGeneration } from "../../showdown-bridge/src/league-generation.ts";
import type { PolicyClient } from "../../showdown-bridge/src/policy-worker.ts";
import { episodeFeatureRows, writeStoredTrainingStream } from "../../showdown-bridge/src/training-loader.ts";
import { readTrainingRunConfig } from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it } from "vitest";
import { verifiedDatasetHash } from "../src/dataset-hash.ts";
import { beliefWorldsFactory } from "../src/damage-aware-worlds.ts";
import { DATASETS_DIR } from "../src/fixtures.ts";
import { datasetTeamPool } from "../src/loop-pool.ts";
import { engine, FORMAT } from "./support.ts";

const SOURCE = join(DATASETS_DIR(FORMAT), "fixtures-v1");
const tiny = JSON.parse(readFileSync(new URL("../../../fixtures/contracts/v1/training-run-config/valid/tiny.json", import.meta.url), "utf8")) as Record<string, unknown>;
const POLICY = `policy:${"c".repeat(64)}`;
const withHash = (hash: string) => {
  const raw = structuredClone(tiny) as { model: { encoder: { options?: unknown } }; team_dataset?: unknown };
  raw.model.encoder.options = { candidate_features: { feature_set: "probe-damage-belief/1", feature_set_hash: hash } };
  raw.team_dataset = { path: SOURCE, dataset_hash: verifiedDatasetHash(engine, SOURCE, FORMAT) };
  return readTrainingRunConfig(raw);
};

describe("the play-time sidecar under probe-damage-belief/1", () => {
  const dir = mkdtempSync(join(tmpdir(), "areneon-belief-rows-"));
  afterAll(() => rmSync(dir, { recursive: true, force: true }));
  it("equals the recomputation byte for byte, and the stream needs no probe", { timeout: 900_000 }, async () => {
    let hash = "";
    try {
      featureSetup(engine, withHash("0".repeat(64)), beliefWorldsFactory);
    } catch (error) {
      hash = /computes as ([0-9a-f]{64})/.exec((error as Error).message)?.[1] as string;
    }
    const config = withHash(hash);
    const setup = featureSetup(engine, config, beliefWorldsFactory) as NonNullable<ReturnType<typeof featureSetup>>;
    const pool = datasetTeamPool(engine, config);
    const client: PolicyClient & { beginGame: () => Promise<void>; featureSet: { feature_set: string; feature_set_hash: string } } = {
      beginGame: () => Promise.resolve(),
      featureSet: setup.statement,
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
    const result = await playGeneration({ engine, config, generation: 1, policy_id: POLICY, client, pool, features: setup.configs, stream_path: join(dir, "stream.jsonl"), episodes_dir: join(dir, "episodes"), features_dir: rows });
    expect(result.games.some((game) => (game.feature_belief?.worlds_sampled ?? 0) > 0)).toBe(true); // the real belief ran
    const own = readdirSync(join(dir, "episodes")).map((name) => ({ hash: name.replace(".json", ""), path: join(dir, "episodes", name) })).filter(({ path }) => (JSON.parse(readFileSync(path, "utf8")) as { provenance: { behaviour_policy_id: string } }).provenance.behaviour_policy_id === POLICY);
    expect(readdirSync(rows).sort()).toEqual(own.map(({ hash }) => `${hash}.rows.json`).sort());
    for (const { hash, path } of own) {
      const episode = await readEpisode(JSON.parse(readFileSync(path, "utf8")), { known_synthetic: pool.known_synthetic });
      const recomputed = await episodeFeatureRows(episode, hash, { engine, configs: setup.configs });
      const stored = JSON.parse(readFileSync(join(rows, `${hash}.rows.json`), "utf8")) as { rows: unknown; request_seqs: number[] };
      expect(jcs(stored.rows as Json)).toBe(jcs(recomputed as unknown as Json));
      expect(stored.request_seqs).toEqual((await decisionObservations(episode)).map((o) => o.identity.request_seq));
    }
    const noProbe = { ...setup.configs, probe: () => { throw new Error("the probe must not run"); } };
    const again = join(dir, "again.jsonl");
    await writeStoredTrainingStream({ episodes: own, evaluation_manifests: [], known_synthetic: pool.known_synthetic, behaviour_policy_id: POLICY, features: { engine, configs: noProbe, precomputed: readFeatureRows(engine, setup.configs, rows) } }, { path: again });
    expect(readFileSync(again, "utf8")).toBe(readFileSync(join(dir, "stream.jsonl"), "utf8"));
  });
});
