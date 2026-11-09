// A sheet-only generation played end to end through the command (#1315): two game processes, each leaving the sheet sidecar of its own
// Episodes in the shared scratch directory, write the same stream and generation.json as one process. The stand-in policy worker states
// the sheet source of the run.
import { chmodSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { engineDataHash, readEngineDataAny, resolveFormat, runGenerationCli } from "@areneon/showdown-bridge";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { readerSha256 } from "../../public-state/src/index.ts";
import { GOLDEN_DIR, goldenTeamVocabulary } from "../../showdown-bridge/src/golden.ts";
import { trainingTeamPool } from "../../showdown-bridge/src/league-generation.ts";
import { SAMPLES_DIR } from "../../showdown-bridge/src/samples.ts";
import { createBeliefFeatures } from "../src/belief-features.ts";
import { engine } from "./support.ts";

const FAKE = fileURLToPath(new URL("../../showdown-bridge/test/fake-policy-worker.ts", import.meta.url));
const SHARD = fileURLToPath(new URL("./sheet-shard-script.ts", import.meta.url));
const DATA = join(GOLDEN_DIR, "engine-data.json");

let dir: string;
beforeAll(() => {
  dir = mkdtempSync(join(tmpdir(), "areneon-cli-sheet-"));
});
afterAll(() => rmSync(dir, { recursive: true, force: true }));

const sourceHash = (): string => createBeliefFeatures({ engine, vocabulary: goldenTeamVocabulary(), reader_sha256: readerSha256(), manifest: resolveFormat(engine, "gen9championsvgc2026regmc").manifest }).features_hash;

function configFile(hash: string): string {
  const raw = JSON.parse(readFileSync(join(SAMPLES_DIR, "v1/training-run-config/valid/tiny.json"), "utf8")) as { generation: Record<string, unknown>; seeds: Record<string, unknown>; model: { encoder: { options?: unknown } } };
  raw.seeds.run_key = "cli-sheet";
  raw.generation.games = 4;
  raw.generation.baseline_games = [];
  raw.generation.max_turns = 4;
  raw.model.encoder.options = { opponent_sheet: { feature_set: "opponent-sheet/1", feature_set_hash: hash } };
  const file = join(dir, "config.json");
  writeFileSync(file, JSON.stringify(raw));
  return file;
}

function wrapper(hash: string): string {
  const file = join(dir, "worker.sh");
  const data = engineDataHash(readEngineDataAny(JSON.parse(readFileSync(DATA, "utf8"))));
  const statement = JSON.stringify({ feature_set: "opponent-sheet/1", feature_set_hash: hash });
  writeFileSync(file, `#!/bin/sh\nFAKE_OPPONENT_SHEET='${statement}' FAKE_ENGINE_DATA_HASH=${data} exec ${process.execPath} ${FAKE} ok\n`);
  chmodSync(file, 0o755);
  return file;
}

const flags = (config: string, out: string, hash: string, extra: string[]): string[] => [
  "--config", config, "--generation", "1", "--policy-id", `policy:${"a".repeat(64)}`, "--preview", "d".repeat(64), "--battle", "b".repeat(64),
  "--run-dir", dir, "--engine-data", DATA, "--encoder-version", "fake-1", "--worker-python", wrapper(hash), "--out", out, ...extra,
];

describe("play-generation with a sheet source", () => {
  it("writes the same stream with two game processes (each leaving its sidecar) as with one process, and leaves no scratch behind", async () => {
    const hash = sourceHash();
    const config = configFile(hash);
    const scratch = (): string[] => readdirSync(tmpdir()).filter((name) => name.startsWith("areneon-shards-")).sort();
    const before = scratch();
    const pool = (e: Parameters<typeof trainingTeamPool>[0], c: { generation: { excluded_abilities: readonly string[] } }) => trainingTeamPool(e, c.generation.excluded_abilities);
    const factory = (context: Parameters<typeof createBeliefFeatures>[0]) => createBeliefFeatures(context);
    const lone = join(dir, "one-process");
    expect(await runGenerationCli(flags(config, lone, hash, []), () => undefined, pool as never, undefined, undefined, factory)).toBe(0);
    const two = join(dir, "two-processes");
    expect(await runGenerationCli(flags(config, two, hash, ["--game-processes", "2"]), () => undefined, pool as never, SHARD, undefined, factory)).toBe(0);
    expect(scratch()).toEqual(before);
    for (const name of ["stream.jsonl", "generation.json", "training-set.json"]) expect(readFileSync(join(two, name), "utf8"), name).toBe(readFileSync(join(lone, name), "utf8"));
    expect(readFileSync(join(two, "stream.jsonl"), "utf8")).toContain("opponent_sheet");
  }, 1_200_000);
});
