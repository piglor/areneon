// The command's `--extra-mode` flag (ADR-023 section 1, #1044): usage errors come before any worker.
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { runProgressCli } from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it } from "vitest";
import { verifiedDatasetHash } from "../src/dataset-hash.ts";
import { DATASETS_DIR } from "../src/fixtures.ts";
import { datasetValidationPool } from "../src/progress-pool.ts";
import { engine, FORMAT } from "./support.ts";

const ROOT = new URL("../../..", import.meta.url).pathname;
const SOURCE = join(DATASETS_DIR(FORMAT), "fixtures-v1");
const tiny = JSON.parse(readFileSync(join(ROOT, "fixtures/contracts/v1/training-run-config/valid/tiny.json"), "utf8")) as Record<string, unknown>;

describe("the command's --extra-mode flag", () => {
  const dir = mkdtempSync(join(tmpdir(), "progress-modes-cli-"));
  afterAll(() => rmSync(dir, { recursive: true, force: true }));
  const configFile = join(dir, "config.json");
  const teamSet = join(dir, "team-set.json");
  writeFileSync(configFile, JSON.stringify({ ...tiny, team_dataset: { path: SOURCE, dataset_hash: verifiedDatasetHash(engine, SOURCE, FORMAT) } }));
  writeFileSync(teamSet, JSON.stringify({ teams: [] }));
  const flags = (...more: string[]): string[] => [
    ...["--config", configFile, "--generation", "1", "--preview", "1".repeat(64), "--battle", "2".repeat(64)],
    ...["--policy-id-sample", `policy:${"c".repeat(64)}`, "--policy-id-argmax", `policy:${"d".repeat(64)}`, "--generation-manifest-sha256", "a".repeat(64)],
    ...["--team-set", teamSet, "--run-dir", dir, "--engine-data", join(dir, "none.json"), "--encoder-version", "enc", "--worker-python", "python", "--games", "2", "--record-games", "0", "--out", join(dir, "out")],
    ...more,
  ];
  const run = (...more: string[]) => runProgressCli(flags(...more), () => undefined, datasetValidationPool);
  const mine = `policy:${"7".repeat(64)}`;

  it("refuses a mode outside the grammar, a default mode, a missing policy id and a policy id for a run that is not played", async () => {
    await expect(run("--extra-mode", `sample:t1000000,${mine}`)).rejects.toThrow(/--extra-mode: unknown play mode/);
    await expect(run("--extra-mode", `argmax,${mine}`)).rejects.toThrow(/always played/);
    await expect(run("--extra-mode", `sample,${mine}`)).rejects.toThrow(/always played/);
    await expect(run("--extra-mode", "sample:t500000")).rejects.toThrow(/<mode>,<candidate-policy-id>/);
    await expect(run("--extra-mode", `sample:t500000,${mine},${mine},${mine}`)).rejects.toThrow(/<mode>,<candidate-policy-id>/);
    await expect(run("--extra-mode", `sample:t500000,${mine},policy:other`)).rejects.toThrow(/exactly when --versus/);
  });

  it("is not played with the ablation or the search candidate", async () => {
    await expect(run("--extra-mode", `sample:t500000,${mine}`, "--ablation-preview", "0,a,c", "--ablation-random-policy-id", "r")).rejects.toThrow(/not played with the Preview ablation/);
    await expect(run("--extra-mode", `sample:t500000,${mine}`, "--search-mode", "argmax")).rejects.toThrow();
  });
});
