// The stream's candidate feature rows are recomputed by the game processes (ADR-022 section 4, #1185): the stream,
// the TrainingSet and every hash are the bytes of the serial path whether the rows were precomputed by one process,
// several, or not at all, and a missing, stale or damaged rows file only costs time.
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { actorRule } from "../src/actor-rule.ts";
import { loadEngine } from "../src/engine.ts";
import { featureSetHash, featureSetIdentity, type FeatureConfigs } from "../src/candidate-features.ts";
import { readFeatureRows, writeFeatureRows } from "../src/feature-recompute.ts";
import { type GenerationOptions, featureSetup, playEntries, playGeneration, trainingTeamPool, writeGeneration } from "../src/league-generation.ts";
import type { PolicyClient } from "../src/policy-worker.ts";
import { SAMPLES_DIR } from "../src/samples.ts";
import { writeStoredTrainingStream } from "../src/training-loader.ts";
import { readTrainingRunConfig, type TrainingRunConfig } from "../src/training-run-config.ts";

const engine = loadEngine();
const tiny = JSON.parse(readFileSync(join(SAMPLES_DIR, "v1/training-run-config/valid/tiny.json"), "utf8")) as Record<string, unknown>;
const POLICY = `policy:${"c".repeat(64)}`;

function configWith(hash: string): TrainingRunConfig {
  const raw = structuredClone(tiny) as { model: { encoder: { options?: unknown } } };
  raw.model.encoder.options = { candidate_features: { feature_set: "probe-damage/1", feature_set_hash: hash } };
  return readTrainingRunConfig(raw);
}

/** A stand-in worker that states no set (so no rows are built at play time) and plays the actor rule. */
function client(): PolicyClient & { beginGame: () => Promise<void> } {
  return {
    beginGame: () => Promise.resolve(),
    ask(_battle, query) {
      const open = query.candidates.filter((c) => !c.pending).map((c) => c.choice);
      const pending = query.candidates.filter((c) => c.pending).map((c) => c.choice);
      return Promise.resolve({ schema_version: 1 as const, identity: query.identity, choice: actorRule({ choices: open, pending }), source_policy_id: query.policy_id, elapsed_ms: 0, deadline_outcome: "in_budget" as const });
    },
  };
}

describe("the stream's feature rows recomputed by the game processes (#1185)", () => {
  let dir: string;
  let config: TrainingRunConfig;
  let configs: FeatureConfigs;
  let setHash: string;
  let pool: ReturnType<typeof trainingTeamPool>;
  beforeAll(async () => {
    dir = mkdtempSync(join(tmpdir(), "areneon-recompute-"));
    try {
      featureSetup(engine, configWith("0".repeat(64)));
    } catch (error) {
      setHash = /computes as ([0-9a-f]{64})/.exec((error as Error).message)?.[1] as string;
    }
    config = configWith(setHash);
    configs = (featureSetup(engine, config) as NonNullable<ReturnType<typeof featureSetup>>).configs;
    pool = trainingTeamPool(engine, config.generation.excluded_abilities);
    // The serial path is the reference: no rows directory, every row recomputed by the process that writes the stream.
    const out = join(dir, "serial");
    const result = await playGeneration(opts(out));
    writeGeneration(out, result);
    reference = files(out);
    referenceSet = result.set;
  }, 600_000); // a tiny generation with the real probe recomputing every row serially
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  const opts = (out: string, change: Partial<GenerationOptions> = {}): GenerationOptions => {
    mkdirSync(join(out, "episodes"), { recursive: true });
    return { engine, config, generation: 1, policy_id: POLICY, client: client(), pool, features: configs, stream_path: join(out, "stream.jsonl"), episodes_dir: join(out, "episodes"), ...change };
  };
  const files = (out: string): Record<string, string> =>
    Object.fromEntries(
      ["stream.jsonl", "training-set.json", "plan.json", "league-run.json", "generation.json", ...readdirSync(join(out, "episodes")).map((n) => `episodes/${n}`)].map((n) => [n, readFileSync(join(out, n), "utf8")]),
    );
  let reference: Record<string, string>;
  let referenceSet: unknown;

  it("the serial path has rows in every example", () => {
    expect(reference["stream.jsonl"]).toContain('"candidate_features"');
  });

  it("gives the same files when one process recomputes into a rows directory, and the stream writer uses them", async () => {
    const out = join(dir, "one");
    const rows = join(dir, "one-rows");
    const result = await playGeneration(opts(out, { features_dir: rows }));
    writeGeneration(out, result);
    expect(files(out)).toEqual(reference);
    expect(result.set).toEqual(referenceSet);
    expect(readdirSync(rows).length).toBeGreaterThan(0);
    // The rows files are what the stream used: a loader whose probe throws still writes the same stream.
    const broken: FeatureConfigs = {
      ...configs,
      probe: () => {
        throw new Error("the probe must not run");
      },
    };
    const own = readdirSync(join(out, "episodes")).map((name) => ({ hash: name.replace(".json", ""), path: join(out, "episodes", name) })).filter(({ path }) => (JSON.parse(readFileSync(path, "utf8")) as { provenance: { behaviour_policy_id: string } }).provenance.behaviour_policy_id === POLICY);
    const again = join(dir, "one-again.jsonl");
    await writeStoredTrainingStream(
      { episodes: own, evaluation_manifests: [], known_synthetic: pool.known_synthetic, behaviour_policy_id: POLICY, features: { engine, configs: broken, precomputed: readFeatureRows(engine, configs, rows) } },
      { path: again },
    );
    expect(readFileSync(again, "utf8")).toBe(reference["stream.jsonl"]);
  }, 600_000); // as above, then the stream again with a probe that must not be asked

  for (const of of [2, 3]) {
    it(`gives the same files when ${of} game processes each recompute the rows of their own Episodes`, async () => {
      const out = join(dir, `shards-${of}`);
      const rows = join(dir, `shards-${of}-rows`);
      const episodes = join(out, "episodes");
      mkdirSync(episodes, { recursive: true });
      const others = Promise.all(Array.from({ length: of - 1 }, (_, i) => playEntries(opts(out, { features_dir: rows }), { index: i + 1, of }))).then((all) => all.flat());
      const result = await playGeneration(opts(out, { features_dir: rows, shard: { index: 0, of }, others }));
      writeGeneration(out, result);
      expect(files(out)).toEqual(reference);
      expect(result.set).toEqual(referenceSet);
    }, 600_000); // the same generation split over game processes
  }

  describe("a rows file that is missing, stale or damaged only costs time", () => {
    it("is not used unless it names the Episode and the feature set and holds one entry per decision", async () => {
      const out = join(dir, "damaged");
      const rows = join(dir, "damaged-rows");
      await playGeneration(opts(out, { features_dir: rows }));
      const names = readdirSync(rows).sort();
      expect(names.length).toBeGreaterThan(5);
      const read = (name: string): { episode_hash: string; feature_set_hash: string; rows: unknown[] } => JSON.parse(readFileSync(join(rows, name), "utf8"));
      const put = (index: number, value: unknown): void => writeFileSync(join(rows, names[index] as string), typeof value === "string" ? value : JSON.stringify(value));
      put(0, "{not json");
      put(1, { ...read(names[1] as string), feature_set_hash: "0".repeat(64) });
      put(2, { ...read(names[2] as string), episode_hash: "0".repeat(64) });
      put(3, { ...read(names[3] as string), rows: [] });
      put(4, { ...read(names[4] as string), rows: "not a list" });
      put(5, { ...read(names[5] as string), rows: read(names[5] as string).rows.map((row) => ({ ...(row as object), feature_set_hash: "0".repeat(64) })) });
      rmSync(join(rows, names[6] as string));
      // The stream writer, given the damaged directory, writes the reference's bytes.
      const own = readdirSync(join(out, "episodes")).map((name) => ({ hash: name.replace(".json", ""), path: join(out, "episodes", name) })).filter(({ path }) => (JSON.parse(readFileSync(path, "utf8")) as { provenance: { behaviour_policy_id: string } }).provenance.behaviour_policy_id === POLICY);
      const again = join(dir, "damaged.jsonl");
      await writeStoredTrainingStream(
        { episodes: own, evaluation_manifests: [], known_synthetic: pool.known_synthetic, behaviour_policy_id: POLICY, features: { engine, configs, precomputed: readFeatureRows(engine, configs, rows) } },
        { path: again },
      );
      expect(readFileSync(again, "utf8")).toBe(reference["stream.jsonl"]);
    }, 600_000); // a tiny generation with the real probe, then the stream again

    it("is not written for an Episode that cannot be read back as its name says, and not after a stop", async () => {
      const first = join(dir, "serial");
      const [name] = readdirSync(join(first, "episodes")).sort();
      const hash = (name as string).replace(".json", "");
      const rows = join(dir, "writer-rows");
      const known = pool.known_synthetic;
      const path = join(first, "episodes", name as string);
      const ctx = { engine, configs, known_synthetic: known };
      expect(await writeFeatureRows(ctx, [{ hash, path }], rows)).toBe(1);
      expect(await writeFeatureRows(ctx, [{ hash: "0".repeat(64), path }], join(dir, "w2"))).toBe(0); // another name than its own hash
      expect(await writeFeatureRows(ctx, [{ hash, path: join(first, "episodes", "absent.json") }], join(dir, "w3"))).toBe(0);
      expect(await writeFeatureRows(ctx, [{ hash, path }], join(dir, "w4"), () => true)).toBe(0);
      expect(readFeatureRows(engine, configs, rows)(hash)?.every((row) => row.feature_set_hash === featureSetHash(featureSetIdentity(engine, configs)))).toBe(true);
      expect(readFeatureRows(engine, configs, rows)("1".repeat(64))).toBeUndefined();
    }, 600_000); // one Episode's rows with the real probe
  });
});
