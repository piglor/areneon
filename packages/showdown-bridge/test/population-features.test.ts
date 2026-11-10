// A population generation whose pairs read candidate features (#1333): the rows are built at play for every seat that reads a
// worker (the real probe, stand-in workers that state the set), a member's own seats leave their rows as the play-time sidecar,
// and the files of a generation are the same for one process, for game processes and for more workers. The post-game pass of
// #1185 is turned into a no-op here, so the sidecar is the only source of the rows directory.
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { actorRule } from "../src/actor-rule.ts";
import { loadEngine } from "../src/engine.ts";
import { featureSetup, trainingTeamPool, type TrainingPool } from "../src/league-generation.ts";
import { memberDir, playPopulationEntries, playPopulationGeneration, type PopulationEntry, type PopulationMember, writePopulationGeneration } from "../src/population-generation.ts";
import { readFeatureRows } from "../src/feature-recompute.ts";
import { episodeFeatureRows } from "../src/training-loader.ts";
import { readEpisode } from "../src/episode-io.ts";
import { episodeHash } from "../src/episode.ts";
import { nodeSha256 } from "../src/player-channel.ts";
import type { PolicyClient } from "../src/policy-worker.ts";
import { readTrainingRunConfig, type TrainingRunConfig } from "../src/training-run-config.ts";
import { populationConfig } from "./population-support.ts";

vi.mock("../src/feature-recompute.ts", async (original) => ({ ...(await original<typeof import("../src/feature-recompute.ts")>()), writeFeatureRows: () => Promise.resolve(0) }));

const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const scratch = mkdtempSync(join(tmpdir(), "population-features-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));

const POLICIES = [`policy:${"a".repeat(64)}`, `policy:${"b".repeat(64)}`];

const withSet = (hash: string): TrainingRunConfig => {
  const raw = JSON.parse(JSON.stringify(populationConfig(2, 2))) as { model: { encoder: { options?: unknown } } };
  raw.model.encoder.options = { candidate_features: { feature_set: "probe-damage/1", feature_set_hash: hash } };
  return readTrainingRunConfig(raw);
};

type Client = PolicyClient & { beginGame: () => Promise<void>; featureSet: { feature_set: string; feature_set_hash: string }; asked: number; withRows: number; withoutRows: number };

describe("a population generation with candidate features", () => {
  let config: TrainingRunConfig;
  let hash = "";
  let pool: TrainingPool;
  let configs: NonNullable<ReturnType<typeof featureSetup>>["configs"];

  /** A worker that states the run's set and plays the actor rule; it counts the queries that carry rows of that set. */
  const client = (): Client => {
    const found: Client = {
      asked: 0,
      withRows: 0,
      withoutRows: 0,
      beginGame: () => Promise.resolve(),
      get featureSet() {
        return { feature_set: "probe-damage/1", feature_set_hash: hash }; // read when the game asks: the hash is found in beforeAll
      },
      ask: (_battle, query) => {
        found.asked++;
        if ((query as { candidate_features?: { feature_set_hash: string } }).candidate_features?.feature_set_hash === hash) found.withRows++;
        else found.withoutRows++;
        return Promise.resolve({
          schema_version: 1 as const,
          identity: query.identity,
          choice: actorRule({ choices: query.candidates.filter((c) => !c.pending).map((c) => c.choice), pending: query.candidates.filter((c) => c.pending).map((c) => c.choice) }),
          source_policy_id: query.policy_id,
          elapsed_ms: 0,
          deadline_outcome: "in_budget" as const,
        });
      },
    };
    return found;
  };
  const membersOf = (peers = 0): (PopulationMember & { client: Client; peers?: Client[] })[] =>
    POLICIES.map((policy_id, k) => ({ member: `m${k}`, policy_id, checkpoints: { preview: String(k).repeat(64), battle: String(k + 2).repeat(64) }, client: client(), ...(peers === 0 ? {} : { peers: Array.from({ length: peers }, client) }) }));
  const options = (dir: string, change: object = {}) => ({ engine, config, generation: 1, members: membersOf(), pool, out_dir: dir, features: configs, ...change });
  const tree = (dir: string, prefix = ""): Record<string, string> => {
    const out: Record<string, string> = {};
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      if (entry.isDirectory()) Object.assign(out, tree(join(dir, entry.name), `${prefix}${entry.name}/`));
      else out[`${prefix}${entry.name}`] = readFileSync(join(dir, entry.name), "utf8");
    }
    return out;
  };

  const whole = join(scratch, "whole");
  const sharded = join(scratch, "sharded");
  const rows = join(scratch, "rows");
  const members = membersOf();
  const peered = membersOf(1);
  let result: Awaited<ReturnType<typeof playPopulationGeneration>>;
  let other: PopulationEntry[];

  beforeAll(async () => {
    try {
      featureSetup(engine, withSet("0".repeat(64)));
    } catch (error) {
      hash = /computes as ([0-9a-f]{64})/.exec((error as Error).message)?.[1] as string;
    }
    config = withSet(hash);
    configs = (featureSetup(engine, config) as NonNullable<ReturnType<typeof featureSetup>>).configs;
    pool = trainingTeamPool(engine, config.generation.excluded_abilities);
    // One process, the loader recomputes every row.
    result = await playPopulationGeneration({ ...options(whole), members });
    writePopulationGeneration(whole, result);
    // Two game processes with a rows directory: this process's slice and the other's, which leaves its Episodes in this one's directory.
    other = await playPopulationEntries(options(join(scratch, "elsewhere"), { shared_dir: sharded, features_dir: rows, members: peered }), { index: 1, of: 2 });
    const merged = await playPopulationGeneration(options(sharded, { shard: { index: 0, of: 2 }, others: Promise.resolve(other), features_dir: rows, members: membersOf(1) }));
    writePopulationGeneration(sharded, merged);
  }, 900_000);

  it("gives every seat that reads a worker the rows of the run's set, and a baseline seat none", () => {
    for (const entry of members) {
      expect(entry.client.withRows).toBeGreaterThan(0);
      expect(entry.client.withoutRows).toBe(0);
    }
    // Every Episode of a member's own seat carries a decision that the rows cover (the loader refuses a stream whose rows do not).
    expect(result.members.every((entry) => entry.set.examples > 0)).toBe(true);
    const stream = readFileSync(join(memberDir(whole, "m0"), "stream.jsonl"), "utf8");
    expect(stream).toContain(`"feature_set_hash":"${hash}"`);
  });

  it("writes the same files for one process and for two game processes with workers to spare", () => {
    expect(tree(sharded)).toEqual(tree(whole));
    for (const entry of peered) expect(entry.client.asked + (entry.peers?.[0]?.asked ?? 0)).toBeGreaterThan(0);
  });

  it("leaves one rows file per Episode of a member's own seat (the play-time sidecar), equal to the rows the loader recomputes", async () => {
    const own: { hash: string; path: string }[] = [];
    for (const label of ["m0", "m1"]) {
      const policy = POLICIES[Number(label.slice(1))] as string;
      const dir = join(memberDir(sharded, label), "episodes");
      for (const name of readdirSync(dir)) {
        const episode = JSON.parse(readFileSync(join(dir, name), "utf8")) as { provenance: { behaviour_policy_id: string } };
        if (episode.provenance.behaviour_policy_id === policy) own.push({ hash: name.replace(".json", ""), path: join(dir, name) });
      }
    }
    expect(own.length).toBeGreaterThan(0);
    // This process wrote its slice's sidecars into `rows` too, so every own Episode (of either slice) has its file.
    expect(readdirSync(rows).sort()).toEqual(own.map((entry) => `${entry.hash}.rows.json`).sort());
    const read = readFeatureRows(engine, configs, rows);
    for (const entry of own.slice(0, 6)) {
      const episode = await readEpisode(JSON.parse(readFileSync(entry.path, "utf8")), { known_synthetic: pool.known_synthetic });
      expect(await episodeHash(episode, nodeSha256)).toBe(entry.hash);
      const file = JSON.parse(readFileSync(join(rows, `${entry.hash}.rows.json`), "utf8")) as { rows: unknown };
      expect(file.rows).toEqual(await episodeFeatureRows(episode, entry.hash, { engine, configs }));
      expect(read(entry.hash, JSON.parse(readFileSync(join(rows, `${entry.hash}.rows.json`), "utf8")).request_seqs)).toBeDefined();
    }
    expect(existsSync(rows)).toBe(true);
  }, 600_000);
});
