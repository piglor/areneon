// The progress evaluation of a pair that reads the opponent sheet (ADR-025 erratum E4 point 5, ADR-027 slice 5): with the real engine,
// the real opponent model and a stand-in worker that states the source and re-reads every query as the real client does. The
// candidate seat's block counts are in each game row and summed in the match, the record reads back, a rerun gives the same bytes.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { actorRule, jcs, type Json, loadMechanicsConfig, loadRulesSnapshot, type PolicyClient, type PolicyQueryInput, playProgress, type ProgressOptions, readPolicyQuery, readProgressEvaluationRecordV2, readTrainingRunConfig, resolveFormat } from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { readerSha256 } from "../../public-state/src/index.ts";
import { goldenTeamVocabulary } from "../../showdown-bridge/src/golden.ts";
import { checkScenario } from "../../showdown-bridge/src/mechanics-scenario.ts";
import { verifiedDatasetHash } from "../src/dataset-hash.ts";
import { createBeliefFeatures } from "../src/belief-features.ts";
import { DATASETS_DIR } from "../src/fixtures.ts";
import { datasetValidationPool } from "../src/progress-pool.ts";
import { setupFor } from "./search-actor-support.ts";
import { engine, FORMAT } from "./support.ts";

const ROOT = new URL("../../..", import.meta.url).pathname;
const SOURCE = join(DATASETS_DIR(FORMAT), "fixtures-v1");
const tiny = JSON.parse(readFileSync(join(ROOT, "fixtures/contracts/v1/training-run-config/valid/tiny.json"), "utf8")) as Record<string, unknown>;
const config = readTrainingRunConfig({ ...tiny, generation: { ...(tiny.generation as object), max_turns: 2 }, team_dataset: { path: SOURCE, dataset_hash: verifiedDatasetHash(engine, SOURCE, FORMAT) } });
const pool = datasetValidationPool(engine, config);
const checkpoints = { preview: "1".repeat(64), battle: "2".repeat(64) };
const vocabulary = goldenTeamVocabulary();
const manifest = resolveFormat(engine, FORMAT, loadRulesSnapshot(engine.commit)).manifest;
const port = createBeliefFeatures({ engine, vocabulary, reader_sha256: readerSha256(), manifest });
const sheet = { engine, vocabulary, port, scenario_id: checkScenario(engine, loadMechanicsConfig(FORMAT)) };

/** A worker that states the source, re-reads each query and plays the actor rule; it records the blocks it was sent. */
class Worker implements PolicyClient {
  readonly sheetSource = { feature_set: "opponent-sheet/1", feature_set_hash: port.features_hash };
  readonly blocks: unknown[] = [];
  beginGame(): Promise<void> {
    return Promise.resolve();
  }
  ask(_battle: string, query: PolicyQueryInput): ReturnType<PolicyClient["ask"]> {
    readPolicyQuery({ ...query, query_id: 1 });
    this.blocks.push(query.opponent_sheet);
    const open = query.candidates.filter((c) => !c.pending).map((c) => c.choice);
    const pending = query.candidates.filter((c) => c.pending).map((c) => c.choice);
    return Promise.resolve({ schema_version: 1 as const, identity: query.identity, choice: actorRule({ choices: open, pending }), source_policy_id: query.policy_id, elapsed_ms: 0, deadline_outcome: "in_budget" as const });
  }
}

const options = (client: Worker, change: Partial<ProgressOptions> = {}): ProgressOptions => ({
  engine,
  config,
  generation: 1,
  generation_manifest_sha256: "a".repeat(64),
  checkpoints,
  policy_ids: { sample: `policy:${"c".repeat(64)}`, argmax: `policy:${"d".repeat(64)}` },
  client,
  pool,
  games: 2,
  record_games: 2,
  training_teams: new Set(),
  league_seeds: new Set(),
  sheet,
  ...change,
});

describe("a sheet pair in the progress evaluation", () => {
  it("counts the candidate seat's blocks per game and per match, reads back and reruns to the same bytes", async () => {
    const worker = new Worker();
    const first = await playProgress(options(worker));
    const { record } = first;
    expect(record.matches.length).toBe(4); // two baselines, two modes
    for (const match of record.matches) {
      const rows = record.games.filter((g) => g.match_id === match.match_id);
      expect(rows).toHaveLength(2);
      expect(rows.every((g) => g.sheet !== undefined && g.sheet.requests > 0)).toBe(true);
      expect(match.sheet?.requests).toBe(rows.reduce((s, g) => s + (g.sheet?.requests ?? 0), 0));
      expect(match.sheet?.refused).toBe(rows.reduce((s, g) => s + (g.sheet?.refused ?? 0), 0));
    }
    expect(worker.blocks.length).toBeGreaterThan(0);
    expect(worker.blocks.every((b) => b !== undefined)).toBe(true);
    expect(readProgressEvaluationRecordV2(JSON.parse(jcs(record as unknown as Json)))).toEqual(record);
    const second = await playProgress(options(new Worker()));
    expect(jcs(second.record as unknown as Json)).toBe(jcs(record as unknown as Json));
  }, 600_000);

  it("a pair without the option has no counts, and a search candidate is refused beside a sheet pair", async () => {
    const plain = await playProgress(options(new Worker(), { sheet: undefined as never }));
    expect(plain.record.matches.every((m) => m.sheet === undefined)).toBe(true);
    await expect(playProgress(options(new Worker(), { search: { setup: setupFor(), mode: "argmax" } }))).rejects.toThrow(/opponent_sheet is not searched/);
  }, 600_000);
});
