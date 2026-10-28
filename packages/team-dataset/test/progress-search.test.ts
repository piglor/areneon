// The search candidate of the progress evaluation (ADR-026 slice P) with the real engine, the real opponent model and a stand-in
// worker (the real worker is driven by progress-search.python.test.ts): the matches, the counts of both sides, the control, the
// stored games, byte-identical reruns and the refusals.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { actorRule, jcs, type Json, type PolicyClient, playProgress, ProgressError, type ProgressOptions, readProgressGameRecord, readTrainingRunConfig, searchPolicy } from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { verifiedDatasetHash } from "../src/dataset-hash.ts";
import { DATASETS_DIR } from "../src/fixtures.ts";
import { datasetValidationPool } from "../src/progress-pool.ts";
import { manifest, setupFor, StandInWorker } from "./search-actor-support.ts";
import { engine, FORMAT } from "./support.ts";

const ROOT = new URL("../../..", import.meta.url).pathname;
const SOURCE = join(DATASETS_DIR(FORMAT), "fixtures-v1");
const tiny = JSON.parse(readFileSync(join(ROOT, "fixtures/contracts/v1/training-run-config/valid/tiny.json"), "utf8")) as Record<string, unknown>;
// A short game keeps the searched decisions few; the turn cap ends it as a tie.
const config = readTrainingRunConfig({ ...tiny, generation: { ...(tiny.generation as object), max_turns: 2 }, team_dataset: { path: SOURCE, dataset_hash: verifiedDatasetHash(engine, SOURCE, FORMAT) } });
const pool = datasetValidationPool(engine, config);
const checkpoints = { preview: "1".repeat(64), battle: "2".repeat(64) };

/** The stand-in worker of the search tests, with the full reply `ask` gives a policy seat. */
class Worker extends StandInWorker {
  beginGame(): Promise<void> {
    return Promise.resolve();
  }
  override ask(_battle: string, query: Parameters<PolicyClient["ask"]>[1]): ReturnType<PolicyClient["ask"]> {
    const open = query.candidates.filter((c) => !c.pending).map((c) => c.choice);
    const pending = query.candidates.filter((c) => c.pending).map((c) => c.choice);
    this.queries.push({ kind: "ask", log_lines: 0 });
    return Promise.resolve({ schema_version: 1 as const, identity: query.identity, choice: actorRule({ choices: open, pending }), source_policy_id: query.policy_id, elapsed_ms: 0, deadline_outcome: "in_budget" as const });
  }
}

const options = (change: Partial<ProgressOptions> = {}): ProgressOptions => ({
  engine,
  config,
  generation: 1,
  generation_manifest_sha256: "a".repeat(64),
  checkpoints,
  policy_ids: { sample: `policy:${"c".repeat(64)}`, argmax: `policy:${"d".repeat(64)}` },
  client: new Worker(),
  pool,
  games: 2,
  record_games: 2,
  training_teams: new Set(),
  league_seeds: new Set(),
  search: { setup: setupFor(), mode: "argmax" },
  ...change,
});

describe("the search candidate of the progress evaluation", () => {
  it("plays the baselines and the no-search control, counts both sides and reruns to the same bytes", async () => {
    const first = await playProgress(options());
    const named = searchPolicy(setupFor(), manifest, checkpoints, "argmax");
    const { record } = first;
    expect(record.matches.map((m) => m.match_id)).toEqual(["search-actor-rule-argmax", "search-random-legal-argmax", "search-self-no-search-argmax"]);
    for (const match of record.matches) {
      expect(match.policy_id).toBe(named.policy_id);
      expect(match.mode).toBe("argmax");
      expect(match.search?.identity_hash).toBe(named.identity_hash);
      expect(match.search?.counts.decisions).toBeGreaterThan(0);
    }
    const control = record.matches[2];
    expect(control?.opponent).toEqual({ kind: "self-no-search", policy_id: `policy:${"d".repeat(64)}`, checkpoints, mode: "argmax" });
    expect(control?.opponent_search?.identity_hash).toBe(named.identity_hash);
    expect(control?.opponent_search?.counts.searched_decisions).toBe(0);
    expect(control?.opponent_search?.counts.decisions).toBeGreaterThan(0);
    expect(control?.opponent_search?.counts.fallbacks.team_preview).toBe(2); // one Preview per game
    expect(record.matches[0]?.opponent_search).toBeUndefined();
    // The stored games carry both seats' counts and read back.
    const stored = [...first.game_records.values()].filter((g) => g.match_id === "search-self-no-search-argmax");
    expect(stored).toHaveLength(2);
    for (const game of stored) {
      expect(game.search?.seat).toBe(game.policies.p1 === named.policy_id ? "p1" : "p2");
      expect(game.opponent_search?.seat).not.toBe(game.search?.seat);
      expect(readProgressGameRecord(JSON.parse(jcs(game as unknown as Json)))).toEqual(game);
    }
    const second = await playProgress(options());
    expect(jcs(second.record as unknown as Json)).toBe(jcs(record as unknown as Json));
  }, 600_000);

  it("is refused beside the ablation and for a pair that reads candidate features, and in an unknown mode", async () => {
    const worker = new Worker();
    const ablation = { generation: { generation: 0, checkpoint: "3".repeat(64), policy_id: `policy:${"e".repeat(64)}`, client: new Worker() }, random_policy_id: `policy:${"f".repeat(64)}` };
    await expect(playProgress(options({ client: worker, ablation }))).rejects.toThrow(/evaluated alone/);
    await expect(playProgress(options({ features: {} as never }))).rejects.toThrow(/candidate features/);
    await expect(playProgress(options({ search: { setup: setupFor(), mode: "greedy" as never } }))).rejects.toThrow(ProgressError);
  });

  it("refuses a validation team that is a dev or training team of the run's own dataset", async () => {
    const own = pool.teams[0]?.canonical_team_hash as string;
    await expect(playProgress(options({ training_teams: new Set([own]) }))).rejects.toThrow(/dev or training team of the run's own dataset/);
  });
});

describe("the run's own dev and training teams", () => {
  it("are listed by canonical and played-form hash, validation teams are not", async () => {
    const { datasetTrainingHashes } = await import("../src/progress-pool.ts");
    const hashes = datasetTrainingHashes(engine, config);
    expect(hashes.size).toBeGreaterThan(0);
    // The committed layout has no training split: dev teams only, twice each (canonical and played form), never a validation team.
    for (const team of pool.teams) expect(hashes.has(team.canonical_team_hash)).toBe(false);
    expect(() => datasetTrainingHashes(engine, { ...config, team_dataset: undefined } as never)).toThrow(/names no team dataset/);
    expect(() => datasetTrainingHashes(engine, { ...config, team_dataset: { path: SOURCE, dataset_hash: "0".repeat(64) } })).toThrow(/hash is not the one/);
  });
});
