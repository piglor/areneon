// The best-response command's argument and configuration refusals (ADR-029 section 2, #1257), before any worker starts. The
// success path needs the policy worker and is driven by best-response.python.test.ts.
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { BEST_RESPONSE_FILE, parseLeaf, runBestResponseCli, UsageError } from "@areneon/showdown-bridge";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { verifiedDatasetHash } from "../src/dataset-hash.ts";
import { DATASETS_DIR } from "../src/fixtures.ts";
import { datasetValidationPool } from "../src/progress-pool.ts";
import { engine, FORMAT } from "./support.ts";

const ROOT = new URL("../../..", import.meta.url).pathname;
const SOURCE = join(DATASETS_DIR(FORMAT), "fixtures-v1");
const json = <T>(path: string): T => JSON.parse(readFileSync(path, "utf8")) as T;
const sample = (name: string): Record<string, unknown> => json(join(ROOT, "fixtures/contracts/v1/training-run-config/valid", name));

describe("parseLeaf", () => {
  it("reads first-request and rollout:<turns> and nothing else", () => {
    expect(parseLeaf("first-request")).toEqual({ kind: "first-request" });
    expect(parseLeaf("rollout:2")).toEqual({ kind: "rollout", turns: 2 });
    for (const bad of ["rollout:0", "rollout:", "rollout:x", "second", ""]) expect(() => parseLeaf(bad)).toThrow(UsageError);
  });
});

describe("runBestResponseCli", () => {
  let dir: string;
  let withDataset: string;
  let withoutDataset: string;
  let features: string;
  let teamSet: string;
  const ports = { teams: (): never => ({}) as never, run_teams: (): ReadonlySet<string> => new Set<string>() };
  beforeAll(() => {
    dir = mkdtempSync(join(tmpdir(), "best-response-cli-"));
    const dataset = { path: SOURCE, dataset_hash: verifiedDatasetHash(engine, SOURCE, FORMAT) };
    withDataset = join(dir, "with.json");
    withoutDataset = join(dir, "without.json");
    features = join(dir, "features.json");
    teamSet = join(dir, "team-set.json");
    writeFileSync(withDataset, JSON.stringify({ ...sample("tiny.json"), team_dataset: dataset }));
    writeFileSync(withoutDataset, JSON.stringify(sample("tiny.json")));
    writeFileSync(features, JSON.stringify({ ...sample("candidate-features.json"), team_dataset: dataset }));
    writeFileSync(teamSet, JSON.stringify({ teams: [] }));
  });
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  const base = (config: string, change: Record<string, string | undefined> = {}): string[] => {
    const flags: Record<string, string | undefined> = {
      config,
      generation: "1",
      preview: "1".repeat(64),
      battle: "2".repeat(64),
      mode: "argmax",
      "policy-id": `policy:${"c".repeat(64)}`,
      "generation-manifest-sha256": "a".repeat(64),
      "team-set": teamSet,
      "run-dir": dir,
      "engine-data": join(dir, "engine-data.json"),
      "encoder-version": "enc",
      "worker-python": "python",
      pairs: "1",
      worlds: "1",
      "k-own": "4",
      "own-rule": "prior-top-k",
      leaf: "first-request",
      "preview-budget-ms": "1000",
      out: join(dir, "out"),
      ...change,
    };
    return Object.entries(flags).flatMap(([name, value]) => (value === undefined ? [] : [`--${name}`, value]));
  };
  const run = (argv: string[], withPorts: Parameters<typeof runBestResponseCli>[3] = ports): Promise<number> => runBestResponseCli(argv, () => undefined, datasetValidationPool, withPorts);

  it("refuses a missing or unknown flag, a bad value and a missing belief port as usage errors", async () => {
    await expect(run(base(withDataset, { config: undefined }))).rejects.toThrow(/--config is required/);
    await expect(run([...base(withDataset), "--unknown", "x"])).rejects.toThrow(UsageError);
    await expect(run(base(withDataset, { mode: "greedy" }))).rejects.toThrow(/argmax or sample/);
    await expect(run(base(withDataset, { "own-rule": "random" }))).rejects.toThrow(/prior-top-k or distinct-leads/);
    await expect(run(base(withDataset, { leaf: "second" }))).rejects.toThrow(/--leaf/);
    await expect(run(base(withDataset, { pairs: "0" }))).rejects.toThrow(/--pairs must be an integer of at least 1/);
    await expect(run(base(withDataset, { worlds: "1.5" }))).rejects.toThrow(/--worlds/);
    await expect(run(base(withDataset), {})).rejects.toThrow(/no belief port/);
    await expect(run(base(withDataset), { teams: ports.teams })).rejects.toThrow(/no belief port/);
    await expect(run(base(withDataset), { run_teams: ports.run_teams })).rejects.toThrow(/no belief port/);
  });

  it("refuses a run config that is a placeholder, names no team dataset or reads candidate features", async () => {
    const placeholder = join(dir, "placeholder.json");
    writeFileSync(placeholder, readFileSync(withDataset, "utf8").replace("tiny-run", "CHANGE-ME-run"));
    await expect(run(base(placeholder))).rejects.toThrow(/CHANGE-ME/);
    const error = vi.spyOn(console, "error").mockImplementation(() => undefined);
    expect(await run(base(withoutDataset))).toBe(1);
    expect(error).toHaveBeenCalledWith(expect.stringContaining("names no team dataset"));
    error.mockRestore();
    await expect(run(base(features))).rejects.toThrow(/features_pair|candidate feature/);
  });

  it("refuses the run's own training dataset as the evaluation dataset and a malformed team set", async () => {
    const own = json<{ team_dataset: { dataset_hash: string } }>(withDataset).team_dataset.dataset_hash;
    await expect(run(base(withDataset, { "eval-dataset-dir": SOURCE, "eval-dataset-hash": own }))).rejects.toThrow(/run's own training dataset/);
    await expect(run(base(withDataset, { "eval-dataset-dir": SOURCE }))).rejects.toThrow(/go together/);
    writeFileSync(join(dir, "bad-team-set.json"), JSON.stringify({ teams: ["x"] }));
    await expect(run(base(withDataset, { "team-set": join(dir, "bad-team-set.json") }))).rejects.toThrow(/team-set/);
  });

  it("names the record file the command writes", () => {
    expect(BEST_RESPONSE_FILE).toBe("best-response.json");
  });
});
