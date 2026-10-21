// The `dataset_hash` of a dataset directory (for a run config's `team_dataset`):
// the directory is read with `readDataset`, which trusts nothing, and the hash
// is computed from the verified records.

import type { Engine } from "@areneon/showdown-bridge";
import { datasetHash, readDataset } from "./dataset-reader.ts";
import { loadFixtureInputs } from "./fixtures.ts";

export const verifiedDatasetHash = (engine: Engine, dir: string, formatId: string): string => datasetHash(readDataset(dir, loadFixtureInputs(engine, formatId)));
