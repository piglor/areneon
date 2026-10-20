// What the tests share: the pinned engine and the committed inputs (the golden
// vocabulary of the format and the label set and generator config under
// config/teams/). Loaded once per test file.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  exportEngineData,
  loadEngine,
  readTeamGeneratorConfig,
  readTeamLabelSet,
  readTeamVocabulary,
  resolveFormat,
  type TeamGeneratorConfig,
  type TeamLabelSet,
} from "@areneon/showdown-bridge";
import { Drawer } from "../src/draw.ts";
import { Proposal } from "../src/proposal.ts";

export const FORMAT = "gen9championsvgc2026regmc";
const ROOT = fileURLToPath(new URL("../../..", import.meta.url));
export const CONFIG_DIR = join(ROOT, "config/teams", FORMAT);
export const VOCABULARY_FILE = join(ROOT, "fixtures/showdown", FORMAT, "team-vocabulary.json");

export const engine = loadEngine();
export const format = resolveFormat(engine, FORMAT);
export const table = exportEngineData(engine, FORMAT);
export const vocabulary = readTeamVocabulary(JSON.parse(readFileSync(VOCABULARY_FILE, "utf8")), table);
export const labelSet = readTeamLabelSet(JSON.parse(readFileSync(join(CONFIG_DIR, "labels.json"), "utf8")), table);
export const config = readTeamGeneratorConfig(JSON.parse(readFileSync(join(CONFIG_DIR, "generator.json"), "utf8")), table);

export const KEY = "11".repeat(32);
export const ASSIGNMENT_KEY = "22".repeat(32);

/** The committed config with test-sized numbers (the schema floor of 40,000 pilot draws is for real runs). */
export const small = (change: Partial<TeamGeneratorConfig> = {}): TeamGeneratorConfig => ({
  ...config,
  pilot_draws: 600,
  pilot_min_hits: 3,
  d_max: 6000,
  n_stratum_min: 2,
  ...change,
});

/** Three of the five labels (8 cells, all common): a run that fills its bins in a few hundred draws. */
export const light: TeamLabelSet = { ...labelSet, label_set_id: "coverage-v1-light", labels: labelSet.labels.slice(2) as TeamLabelSet["labels"] };

export const drawerFor = (cfg: TeamGeneratorConfig = small(), key: string = KEY, labels: TeamLabelSet = labelSet): Drawer =>
  new Drawer({ engine, format, proposal: new Proposal(vocabulary, table, cfg), labelSet: labels, generationKey: key });
