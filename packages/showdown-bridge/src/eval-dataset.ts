// The eval-dataset option (ADR-020 E-17.1), one implementation for every command that measures on the validation
// split of a team dataset other than the run's own: the progress evaluation and the scripted head-to-head. The
// option is two flags that go together, a directory and the dataset's hash; the dataset is then read exactly as the
// run's own is (through `readDataset`, whose hash must be the one given), by replacing the config's `team_dataset`
// before the pool is built. Nothing here reads the directory.

import { UsageError } from "./generation-cli.ts";
import type { TrainingRunConfig } from "./training-run-config.ts";

/** The two flags, to spread into a command's `parseArgs` options (both optional, both or neither). */
export const EVAL_DATASET_FLAGS = {
  "eval-dataset-dir": { type: "string" },
  "eval-dataset-hash": { type: "string" },
} as const;

export type EvalDataset = NonNullable<TrainingRunConfig["team_dataset"]>;

/** The option as given: both flags or none; the hash is 64 lower-case hex digits. */
export function readEvalDataset(values: { "eval-dataset-dir"?: string | undefined; "eval-dataset-hash"?: string | undefined }): EvalDataset | undefined {
  const path = values["eval-dataset-dir"];
  const hash = values["eval-dataset-hash"];
  if ((path === undefined) !== (hash === undefined)) throw new UsageError("--eval-dataset-dir and --eval-dataset-hash go together");
  if (path === undefined || hash === undefined) return undefined;
  if (!/^[0-9a-f]{64}$/.test(hash)) throw new UsageError("--eval-dataset-hash must be 64 lower-case hex digits");
  return { path, dataset_hash: hash };
}

/** The config the pool is built from: the run's own, with the evaluation dataset in place of its `team_dataset`. */
export function withEvalDataset(config: TrainingRunConfig, dataset: EvalDataset | undefined): TrainingRunConfig {
  return dataset === undefined ? config : { ...config, team_dataset: dataset };
}
