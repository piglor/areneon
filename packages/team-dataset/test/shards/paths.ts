import { join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("../../../..", import.meta.url));
export const FORMAT_ID = "gen9championsvgc2026regmc";
export const CONFIG_DIR = join(ROOT, "config/teams", FORMAT_ID);
export const DATASET_DIR = join(ROOT, "fixtures/teams", FORMAT_ID, "datasets/fixtures-v1");
export const REPO = ROOT;
