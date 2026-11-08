import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { actorRule } from "../src/actor-rule.ts";
import { jcs, type Json } from "../src/canonical-json.ts";
import type { SheetSetup } from "../src/belief-features.ts";
import { buildLeagueRunRecord, checkLeagueRunRecord, loadEngine, type PolicyClient, readLeagueRunRecord, resolveRegimes, runConfigHash } from "../src/index.ts";
import { type GenerationResult, GenerationError, planGeneration, playGeneration, trainingTeamPool, type TrainingPool, writeGeneration } from "../src/league-generation.ts";
import { SAMPLES_DIR } from "../src/samples.ts";
import { loadTrainingSet, TrainingLoaderError } from "../src/training-loader.ts";
import { readTrainingRunConfig, type TrainingRunConfig } from "../src/training-run-config.ts";

const engine = loadEngine();
const CLOSED = "gen9championsvgc2026regmc";
const OPEN = "gen9championsvgc2026regmcbo3";
const POLICY = `policy:${"c".repeat(64)}`;
const sample = (name: string): Record<string, unknown> => JSON.parse(readFileSync(join(SAMPLES_DIR, `v1/training-run-config/valid/${name}.json`), "utf8")) as Record<string, unknown>;
const plain = readTrainingRunConfig(sample("tiny"));
// 4 self-play games and 2 random-legal baseline games per generation; generation 1 is open only, later ones half closed.
const mixed = readTrainingRunConfig({ ...sample("tiny"), sheet_regimes: { open_format: OPEN, closed_share_ppm: 500_000, open_only_generations: 1, closed_sheet_source: "revealed-only/1" } });

/** An in-process worker: the shared actor rule over the query's own candidates. */
function actorClient(): PolicyClient & { beginGame: () => Promise<void> } {
  return {
    beginGame: () => Promise.resolve(),
    ask(_battle, query) {
      const open = query.candidates.filter((c) => !c.pending).map((c) => c.choice);
      const pending = query.candidates.filter((c) => c.pending).map((c) => c.choice);
      return Promise.resolve({
        schema_version: 1 as const,
        identity: query.identity,
        choice: actorRule({ choices: open, pending }),
        source_policy_id: query.policy_id,
        elapsed_ms: 0,
        deadline_outcome: "in_budget" as const,
      });
    },
  };
}

let pool: TrainingPool;
beforeAll(() => {
  pool = trainingTeamPool(engine, plain.generation.excluded_abilities);
});

const play = (config: TrainingRunConfig, generation: number, extra: object = {}): Promise<GenerationResult> =>
  playGeneration({ engine, config, generation, policy_id: POLICY, client: actorClient(), pool, ...extra });

describe("a closed-only run keeps its bytes (the no-change test)", () => {
  // fixtures/loop/inspector-run is a two-generation run of the tiny loop, played before sheet_regimes existed.
  const run = fileURLToPath(new URL("../../../fixtures/loop/inspector-run", import.meta.url));
  const read = (path: string): { text: string; value: unknown } => {
    const text = readFileSync(join(run, path), "utf8");
    return { text, value: JSON.parse(text) };
  };

  it("plans the committed generations' plan.json byte for byte, with no format on any entry", () => {
    const config = readTrainingRunConfig(read("run-config.json").value);
    const poolSize = (read("pool.json").value as { teams: unknown[] }).teams.length;
    for (const generation of [1, 2]) {
      const committed = read(`generations/g00000${generation}/plan.json`);
      const plan = planGeneration(config, generation, poolSize);
      expect(`${jcs(plan as unknown as Json)}\n`).toBe(committed.text);
      expect(plan.some((entry) => "format_id" in entry)).toBe(false);
    }
  });

  it("builds the committed generations' league-run.json byte for byte, with neither regimes nor game_manifests", async () => {
    const config = readTrainingRunConfig(read("run-config.json").value);
    expect(await runConfigHash(config)).toBe((read("generations/g000001/league-run.json").value as { run_config_hash: string }).run_config_hash);
    for (const generation of [1, 2]) {
      const dir = `generations/g00000${generation}`;
      const committed = read(`${dir}/league-run.json`);
      const games = (read(`${dir}/generation.json`).value as { games: { opponent: string; outcome: "played" | "forfeit"; forfeit?: { cause: never } }[] }).games;
      const record = await buildLeagueRunRecord({
        config,
        generation,
        pair_policy_id: (committed.value as { pair_policy_id: string }).pair_policy_id,
        plan: read(`${dir}/plan.json`).value as unknown[],
        games,
      });
      expect(`${jcs(record as unknown as Json)}\n`).toBe(committed.text);
    }
  });
});

describe("a tiny mixed-format generation", () => {
  let first: GenerationResult;
  let second: GenerationResult;
  let again: GenerationResult;
  beforeAll(async () => {
    [first, second, again] = [await play(mixed, 1), await play(mixed, 2), await play(mixed, 2)];
  }, 600_000);

  it("plays open games in the open format and the closed share and every baseline game in the closed one", () => {
    expect(first.plan.map((g) => g.format_id)).toEqual([OPEN, OPEN, OPEN, OPEN, CLOSED, CLOSED]);
    expect(second.plan.map((g) => g.format_id).filter((f) => f === OPEN)).toHaveLength(2);
    expect(second.plan.slice(4).map((g) => g.format_id)).toEqual([CLOSED, CLOSED]);
    for (const result of [first, second]) {
      expect(result.games.every((g) => g.outcome === "played")).toBe(true);
      // Both seats of a game share its format, and every Episode carries its game's own manifest.
      result.plan.forEach((entry, game) => {
        for (const episode of result.episodes.slice(2 * game, 2 * game + 2)) expect(episode.provenance.manifest.format_id).toBe(entry.format_id);
      });
    }
  });

  it("counts the regimes as the quota says and lists both formats' manifests in the run record", async () => {
    expect(first.run_record.regimes).toEqual({ open: 4, closed: 2 });
    expect(second.run_record.regimes).toEqual({ open: 2, closed: 4 });
    const manifests = resolveRegimes(engine, mixed);
    for (const result of [first, second]) {
      expect(result.game_manifests).toEqual(manifests.sorted);
      expect(result.run_record.game_manifests).toEqual(manifests.sorted);
      expect(result.manifest).toEqual(manifests.closed);
      await checkLeagueRunRecord(result.run_record, mixed);
      expect(readLeagueRunRecord(JSON.parse(JSON.stringify(result.run_record))).regimes).toEqual(result.run_record.regimes);
    }
    // The record is the mixed config's alone.
    await expect(checkLeagueRunRecord(first.run_record, plain)).rejects.toThrow(/run_config_hash|exactly for a config with sheet_regimes/);
  });

  it("is byte-identical on a rerun", () => {
    const bytes = (r: GenerationResult): string => jcs({ plan: r.plan, games: r.games, run: r.run_record, set: r.set, episodes: r.episode_hashes } as unknown as Json);
    expect(bytes(again)).toBe(bytes(second));
    expect(bytes(first)).not.toBe(bytes(second));
  });

  it("writes game_manifests into generation.json and nothing else new, and a closed-only run writes none", async () => {
    const dir = mkdtempSync(join(tmpdir(), "regimes-"));
    try {
      writeGeneration(dir, second);
      const written = JSON.parse(readFileSync(join(dir, "generation.json"), "utf8")) as Record<string, unknown>;
      expect(written.game_manifests).toEqual(second.game_manifests);
      expect(existsSync(join(dir, "plan.json"))).toBe(true);
      const closedOnly = await play(plain, 1);
      writeGeneration(join(dir, "closed-only"), closedOnly);
      const plainWritten = JSON.parse(readFileSync(join(dir, "closed-only", "generation.json"), "utf8")) as Record<string, unknown>;
      expect("game_manifests" in plainWritten).toBe(false);
      expect(closedOnly.game_manifests).toBeUndefined();
      expect(closedOnly.run_record.regimes).toBeUndefined();
      expect(closedOnly.run_record.game_manifests).toBeUndefined();
      expect(closedOnly.plan.some((entry) => "format_id" in entry)).toBe(false);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }, 120_000);

  it("is read by the loader with each Episode's own format, and refused where it does not fit", async () => {
    const own = second.episodes.filter((e) => e.provenance.behaviour_policy_id === POLICY);
    const input = { episodes: own, evaluation_manifests: [], known_synthetic: pool.known_synthetic, behaviour_policy_id: POLICY };
    const both = await loadTrainingSet({ ...input, formats: [CLOSED, OPEN] });
    expect(both.set.examples).toBeGreaterThan(0);
    // The stream is the same bytes with or without the format list (it adds a check, not a field).
    expect((await loadTrainingSet(input)).stream).toBe(both.stream);
    await expect(loadTrainingSet({ ...input, formats: [CLOSED] })).rejects.toThrow(/none of the run's formats/);
    await expect(loadTrainingSet({ ...input, formats: [CLOSED, OPEN], sheet: {} as SheetSetup })).rejects.toThrow(TrainingLoaderError);
    await expect(loadTrainingSet({ ...input, formats: [CLOSED, OPEN], sheet: {} as SheetSetup })).rejects.toThrow(/open-format game/);
  }, 120_000);

  it("refuses an open game for a run that reads opponent sheet blocks, and a doctored plan", async () => {
    await expect(play(mixed, 1, { sheet: {} as SheetSetup })).rejects.toThrow(GenerationError);
    await expect(play(mixed, 1, { sheet: {} as SheetSetup })).rejects.toThrow(/engine-sheet/);
    const doctored = { ...mixed, sheet_regimes: { ...mixed.sheet_regimes, closed_share_ppm: 0 } } as TrainingRunConfig;
    // A config's quota decides the plan, so the plan of one config checked against another's is refused.
    const { checkPlanRegimes } = await import("../src/sheet-regimes.ts");
    expect(() => checkPlanRegimes(second.plan, doctored, 2)).toThrow(/quota/);
  }, 120_000);

  afterAll(() => undefined);
});
