// A generation that labels, played end to end through the command (ADR-026 erratum E2, #1290): the real script path with a stand-in
// policy worker (the fake worker of the client tests, behind a wrapper script), once with two game processes and once with a lone
// process that has two workers. The label stream, the manifest and the summary are the same files either way, the game processes
// were given the shared label directory (else the writer would have recomputed their Episodes), the lone process's own scratch
// directory is made and removed, and the counts are printed on stderr and stated in generation.json.
import { chmodSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { engineDataHash, jcs, readEngineDataAny, runGenerationCli, searchConfigHash, loadSearchConfig } from "@areneon/showdown-bridge";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { GOLDEN_DIR } from "../../showdown-bridge/src/golden.ts";
import { trainingTeamPool } from "../../showdown-bridge/src/league-generation.ts";
import { SAMPLES_DIR } from "../../showdown-bridge/src/samples.ts";
import { progressSearchSetup } from "../src/progress-search-setup.ts";
import { FORMAT } from "./support.ts";

const FAKE = fileURLToPath(new URL("../../showdown-bridge/test/fake-policy-worker.ts", import.meta.url));
const SHARD = fileURLToPath(new URL("./labels-shard-script.ts", import.meta.url));
const DATA = join(GOLDEN_DIR, "engine-data.json");

let dir: string;
beforeAll(() => {
  dir = mkdtempSync(join(tmpdir(), "areneon-cli-labels-"));
});
afterAll(() => rmSync(dir, { recursive: true, force: true }));

const scratchLeft = (): string[] => readdirSync(tmpdir()).filter((name) => /^areneon-(shards|labels)-/.test(name)).sort();

function configFile(): string {
  const raw = JSON.parse(readFileSync(join(SAMPLES_DIR, "v1/training-run-config/valid/search-distill.json"), "utf8")) as { generation: Record<string, unknown>; seeds: Record<string, unknown> };
  raw.seeds.run_key = "cli-labels";
  raw.generation.games = 4;
  raw.generation.baseline_games = [];
  raw.generation.max_turns = 4;
  const labels = raw.generation.search_labels as Record<string, unknown>;
  labels.search_config_hash = searchConfigHash(loadSearchConfig(FORMAT, "rerank-1-label.v1"));
  labels.label_share_ppm = 500_000;
  labels.start_generation = 3;
  const file = join(dir, "config.json");
  writeFileSync(file, JSON.stringify(raw));
  return file;
}

function wrapper(): string {
  const file = join(dir, "worker.sh");
  const hash = engineDataHash(readEngineDataAny(JSON.parse(readFileSync(DATA, "utf8"))));
  writeFileSync(file, `#!/bin/sh\nFAKE_VALUE_QUERY=1 FAKE_VALUE_VARY=1 FAKE_ENGINE_DATA_HASH=${hash} exec ${process.execPath} ${FAKE} ok\n`);
  chmodSync(file, 0o755);
  return file;
}

const flags = (config: string, out: string, extra: string[]): string[] => [
  "--config", config, "--generation", "3", "--policy-id", `policy:${"a".repeat(64)}`, "--preview", "d".repeat(64), "--battle", "b".repeat(64),
  "--run-dir", dir, "--engine-data", DATA, "--encoder-version", "fake-1", "--worker-python", wrapper(), "--out", out, ...extra,
];

async function run(name: string, extra: string[], shardScript?: string) {
  const out = join(dir, name);
  const lines: string[] = [];
  const err: string[] = [];
  const spy = vi.spyOn(process.stderr, "write").mockImplementation(((chunk: string | Uint8Array) => {
    err.push(String(chunk));
    return true;
  }) as never);
  try {
    const code = await runGenerationCli(flags(configFile(), out, extra), (line) => lines.push(line), (engine, c) => trainingTeamPool(engine, c.generation.excluded_abilities), shardScript, undefined, undefined, progressSearchSetup);
    return { code, out, lines, err: err.join("") };
  } finally {
    spy.mockRestore();
  }
}

const files = ["stream.jsonl", "labels.jsonl", "label-set.json", "generation.json", "training-set.json"] as const;

describe("play-generation with search labels", () => {
  it("writes the same label stream, manifest and counts with two game processes as with one process and two workers, and leaves no scratch behind", async () => {
    const before = scratchLeft();
    const two = await run("two-processes", ["--game-processes", "2"], SHARD);
    expect(two.code).toBe(0);
    const lone = await run("one-process", ["--policy-workers", "2"]);
    expect(lone.code).toBe(0);
    expect(scratchLeft()).toEqual(before);
    for (const name of files) expect(readFileSync(join(lone.out, name), "utf8"), name).toBe(readFileSync(join(two.out, name), "utf8"));
    const summary = JSON.parse(readFileSync(join(two.out, "generation.json"), "utf8")) as { search_labels: { selected: number; labelled: number; recomputed: number; refused: { reason: string; count: number }[] } };
    expect(summary.search_labels.labelled).toBeGreaterThan(0);
    // the game processes labelled their own Episodes into the shared directory: the writer made none again
    expect(summary.search_labels.recomputed).toBe(0);
    expect(summary.search_labels.selected).toBe(summary.search_labels.labelled + summary.search_labels.refused.reduce((a, r) => a + r.count, 0));
    expect(readFileSync(join(two.out, "labels.jsonl"), "utf8").split("\n").filter((l) => l !== "").length).toBe(summary.search_labels.labelled);
    for (const run_ of [two, lone]) expect(run_.err).toContain(`search labels: ${jcs({ selected: summary.search_labels.selected, labelled: summary.search_labels.labelled, refused: summary.search_labels.refused, recomputed: 0 } as never)}`);
  }, 1_200_000);

  it("is not labelled before the start generation: no label file, no block", async () => {
    const out = join(dir, "early");
    const code = await runGenerationCli(flags(configFile(), out, []).map((v, i, all) => (all[i - 1] === "--generation" ? "2" : v)), () => undefined, (engine, c) => trainingTeamPool(engine, c.generation.excluded_abilities), undefined, undefined, undefined, progressSearchSetup);
    expect(code).toBe(0);
    expect(readdirSync(out)).not.toContain("labels.jsonl");
    expect(JSON.parse(readFileSync(join(out, "generation.json"), "utf8"))).not.toHaveProperty("search_labels");
  }, 600_000);
});
