// A training generation of a pair that reads the opponent sheet (ADR-025 erratum E4 point 5): the real engine and opponent model with a
// stand-in worker. The pair's seats send a block per request, the stream's examples carry the very blocks that were sent, and each
// game row of generation.json carries the blocks built and the refusals by code (absent for a pair without the option).
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  actorRule,
  jcs,
  type Json,
  type PolicyClient,
  type PolicyQueryInput,
  writeStoredTrainingStream,
  playGeneration,
  readPolicyQuery,
  readTrainingExample,
  readTrainingRunConfig,
  resolveFormat,
  sheetSetup,
} from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it, vi } from "vitest";
import { readSheetBlocks } from "../../showdown-bridge/src/sheet-recompute.ts";
import { readerSha256 } from "../../public-state/src/index.ts";
import { goldenTeamVocabulary } from "../../showdown-bridge/src/golden.ts";
import { trainingTeamPool } from "../../showdown-bridge/src/league-generation.ts";
import { SAMPLES_DIR } from "../../showdown-bridge/src/samples.ts";
import { createBeliefFeatures } from "../src/belief-features.ts";
import { engine } from "./support.ts";

const tiny = JSON.parse(readFileSync(join(SAMPLES_DIR, "v1/training-run-config/valid/tiny.json"), "utf8")) as { model: { encoder: { options?: unknown } } };
const POLICY = `policy:${"a".repeat(64)}`;
const factory = (context: Parameters<typeof createBeliefFeatures>[0]) => createBeliefFeatures(context);

function configWith(hash: string | null) {
  const raw = structuredClone(tiny);
  if (hash !== null) raw.model.encoder.options = { opponent_sheet: { feature_set: "opponent-sheet/1", feature_set_hash: hash } };
  return readTrainingRunConfig(raw);
}

function standIn(source: { feature_set: string; feature_set_hash: string } | null): PolicyClient & { queries: PolicyQueryInput[] } {
  const queries: PolicyQueryInput[] = [];
  return {
    queries,
    sheetSource: source,
    ask(_battle, query) {
      readPolicyQuery({ ...query, query_id: queries.length + 1 });
      queries.push(query);
      const open = query.candidates.filter((c) => !c.pending).map((c) => c.choice);
      const pending = query.candidates.filter((c) => c.pending).map((c) => c.choice);
      return Promise.resolve({ schema_version: 1, identity: query.identity, choice: actorRule({ choices: open, pending }), source_policy_id: query.policy_id, elapsed_ms: 0, deadline_outcome: "in_budget" });
    },
  };
}

describe("a generation of a sheet pair", () => {
  it("sends a block per request, writes the very blocks into the stream and counts them in every game row", async () => {
    const found = createBeliefFeatures({ engine, vocabulary: goldenTeamVocabulary(), reader_sha256: readerSha256(), manifest: resolveFormat(engine, "gen9championsvgc2026regmc").manifest });
    const config = configWith(found.features_hash);
    const setup = sheetSetup(engine, config, factory);
    expect(setup).toBeDefined();
    const pool = trainingTeamPool(engine, config.generation.excluded_abilities);
    const client = standIn(setup?.statement ?? null);
    const reported: number[] = [];
    const result = await playGeneration({ engine, config, generation: 1, policy_id: POLICY, client, pool, ...(setup === undefined ? {} : { sheet: setup.setup }), on_sheet_counts: (game) => void reported.push(game) });
    expect(result.games.length).toBeGreaterThan(0);
    expect(reported.length).toBe(result.games.length);
    for (const game of result.games) {
      expect(game.sheet?.requests).toBeGreaterThan(0);
      const codes = game.sheet?.refusals.map((r) => r.code) ?? [];
      expect(codes).toEqual([...codes].sort());
    }
    const examples = (result.stream as string).split("\n").filter((line) => line !== "").map((line) => readTrainingExample(JSON.parse(line)));
    expect(examples.length).toBeGreaterThan(0);
    const sent = new Set(client.queries.map((query) => jcs(query.opponent_sheet as unknown as Json)));
    for (const example of examples) expect(sent.has(jcs(example.opponent_sheet as unknown as Json))).toBe(true);
    expect(client.queries.every((query) => query.opponent_sheet !== undefined)).toBe(true);
  }, 600_000);

  it("a pair without the option has no sheet on any row and none in the stream", async () => {
    const config = configWith(null);
    const result = await playGeneration({ engine, config, generation: 1, policy_id: POLICY, client: standIn(null), pool: trainingTeamPool(engine, config.generation.excluded_abilities) });
    expect(result.games.every((game) => game.sheet === undefined)).toBe(true);
    expect((result.stream as string).includes("opponent_sheet")).toBe(false);
  }, 300_000);

  it("under the uninformative mode (#1309) a seat sends one block a game and a rerun is byte-identical", async () => {
    const manifest = resolveFormat(engine, "gen9championsvgc2026regmc").manifest;
    const blind = createBeliefFeatures({ engine, vocabulary: goldenTeamVocabulary(), reader_sha256: readerSha256(), manifest, evidence: "none" });
    const informative = createBeliefFeatures({ engine, vocabulary: goldenTeamVocabulary(), reader_sha256: readerSha256(), manifest });
    expect(blind.features_hash).not.toBe(informative.features_hash);
    const config = configWith(blind.features_hash);
    // the setup finds the mode by the named hash through the same factory the commands use
    const setup = sheetSetup(engine, config, factory);
    expect(setup?.statement.feature_set_hash).toBe(blind.features_hash);
    const pool = trainingTeamPool(engine, config.generation.excluded_abilities);
    const play = async () => {
      const client = standIn(setup?.statement ?? null);
      const result = await playGeneration({ engine, config, generation: 1, policy_id: POLICY, client, pool, ...(setup === undefined ? {} : { sheet: setup.setup }) });
      return { client, result };
    };
    const first = await play();
    const second = await play();
    expect(second.result.stream).toBe(first.result.stream);
    const distinct = new Set(first.client.queries.map((query) => jcs(query.opponent_sheet as unknown as Json)));
    expect(first.client.queries.length).toBeGreaterThan(distinct.size);
    expect(distinct.size).toBeLessThanOrEqual(2 * first.result.games.length);
    for (const query of first.client.queries) expect(query.opponent_sheet?.source.hash).toBe(blind.features_hash);
  }, 900_000);
});

// The play-time sheet sidecar (#1315): the blocks a seat built while the game was played are left beside the Episode, and the stream
// writer takes them instead of building every block a second time. No stream byte depends on whether they exist.
describe("the play-time sheet sidecar", () => {
  const dir = mkdtempSync(join(tmpdir(), "areneon-sheet-sidecar-"));
  afterAll(() => rmSync(dir, { recursive: true, force: true }));
  it("is written during play, equals the recomputation byte for byte and is what the stream writer reads", async () => {
    const found = createBeliefFeatures({ engine, vocabulary: goldenTeamVocabulary(), reader_sha256: readerSha256(), manifest: resolveFormat(engine, "gen9championsvgc2026regmc").manifest });
    const config = configWith(found.features_hash);
    const setup = sheetSetup(engine, config, factory);
    if (setup === undefined) throw new Error("no sheet setup");
    const pool = trainingTeamPool(engine, config.generation.excluded_abilities);
    mkdirSync(join(dir, "episodes"), { recursive: true });
    const sheets = join(dir, "sheets");
    const result = await playGeneration({ engine, config, generation: 1, policy_id: POLICY, client: standIn(setup.statement), pool, sheet: setup.setup, stream_path: join(dir, "stream.jsonl"), episodes_dir: join(dir, "episodes"), features_dir: sheets });
    expect(result.games.length).toBeGreaterThan(0);
    const own = readdirSync(join(dir, "episodes")).filter((name) => (JSON.parse(readFileSync(join(dir, "episodes", name), "utf8")) as { provenance: { behaviour_policy_id: string } }).provenance.behaviour_policy_id === POLICY);
    expect(own.length).toBeGreaterThan(0);
    expect(readdirSync(sheets).sort()).toEqual(own.map((name) => name.replace(".json", ".sheet.json")).sort());
    const stored = own.map((name) => ({ hash: name.replace(".json", ""), path: join(dir, "episodes", name) }));
    const base = { episodes: stored, evaluation_manifests: [], known_synthetic: pool.known_synthetic, behaviour_policy_id: POLICY };
    const write = async (name: string, precomputed?: ReturnType<typeof readSheetBlocks>) => {
      const set = await writeStoredTrainingStream({ ...base, sheet: { ...setup.setup, ...(precomputed === undefined ? {} : { precomputed }) } }, { path: join(dir, name) });
      return { set, text: readFileSync(join(dir, name), "utf8") };
    };
    const recomputed = await write("recomputed.jsonl");
    const sidecar = await write("sidecar.jsonl", readSheetBlocks(found.features_hash, sheets));
    // The independence assertion: the stream from the sidecar, the stream recomputed by the loader and the stream written during play agree.
    expect(sidecar.text).toBe(recomputed.text);
    expect(sidecar.text).toBe(readFileSync(join(dir, "stream.jsonl"), "utf8"));
    expect(sidecar.set.stream_sha256).toBe(recomputed.set.stream_sha256);
    // The sidecar is trusted like the rows sidecar (a valid block in the right place is taken as built); the equality above is the guard.
    // Here it is shown that the writer does read the files (blocks moved to other decisions change the stream) and skips a file of another source.
    const first = join(sheets, readdirSync(sheets)[0] as string);
    const original = readFileSync(first, "utf8");
    const parsed = JSON.parse(original) as { blocks: unknown[] };
    parsed.blocks.reverse();
    writeFileSync(first, JSON.stringify(parsed));
    const tampered = await write("tampered.jsonl", readSheetBlocks(found.features_hash, sheets));
    expect(tampered.text).not.toBe(recomputed.text);
    const otherSource = await write("other.jsonl", readSheetBlocks("f".repeat(64), sheets));
    expect(otherSource.text).toBe(recomputed.text);
    // A stale or damaged file is recomputed: wrong request numbers, a bad block, garbage.
    const read = readSheetBlocks(found.features_hash, sheets);
    const hash = (readdirSync(sheets)[0] as string).replace(".sheet.json", "");
    const real = JSON.parse(original) as { request_seqs: number[]; blocks: unknown[] };
    expect(read(hash, real.request_seqs)).toBeDefined();
    writeFileSync(first, original);
    expect(read(hash, real.request_seqs)?.length).toBe(real.blocks.length);
    expect(read(hash, [...real.request_seqs, 99])).toBeUndefined();
    expect(read("0".repeat(64), real.request_seqs)).toBeUndefined();
    writeFileSync(first, JSON.stringify({ ...real, episode_hash: hash, source_hash: found.features_hash, blocks: [{ source: { kind: "oracle-diagnostic", hash: found.features_hash } }, ...real.blocks.slice(1)] }));
    expect(read(hash, real.request_seqs)).toBeUndefined();
    writeFileSync(first, "{not json");
    expect(read(hash, real.request_seqs)).toBeUndefined();
  }, 900_000);

  it("a failing sidecar write is skipped and logged without content, and the generation still completes", async () => {
    const found = createBeliefFeatures({ engine, vocabulary: goldenTeamVocabulary(), reader_sha256: readerSha256(), manifest: resolveFormat(engine, "gen9championsvgc2026regmc").manifest });
    const config = configWith(found.features_hash);
    const setup = sheetSetup(engine, config, factory);
    if (setup === undefined) throw new Error("no sheet setup");
    const blocked = join(dir, "not-a-directory");
    writeFileSync(blocked, "a file where the sidecar directory should be");
    mkdirSync(join(dir, "episodes-failing"), { recursive: true });
    const err: string[] = [];
    const spy = vi.spyOn(process.stderr, "write").mockImplementation(((chunk: string | Uint8Array) => (err.push(String(chunk)), true)) as never);
    try {
      const result = await playGeneration({ engine, config, generation: 1, policy_id: POLICY, client: standIn(setup.statement), pool: trainingTeamPool(engine, config.generation.excluded_abilities), sheet: setup.setup, stream_path: join(dir, "failing.jsonl"), episodes_dir: join(dir, "episodes-failing"), features_dir: blocked });
      expect(result.games.length).toBeGreaterThan(0);
    } finally {
      spy.mockRestore();
    }
    expect(err.join("")).toMatch(/sheet sidecar skipped for game \d+: Error/);
    expect(readFileSync(join(dir, "failing.jsonl"), "utf8")).toBe(readFileSync(join(dir, "stream.jsonl"), "utf8"));
  }, 900_000);
});
