// `check-team-labels` (ADR-013 section 3): the checker that validates the
// label-set and generator-config records under config/teams/<format>/ against
// the pinned engine. The labeller code holds no ids, so a typo in a record
// would otherwise pass unnoticed; this is where it is caught. It checks, for
// each format directory:
//   - labels.json is a TeamLabelSet whose id lists are sorted, unique and all
//     in the engine-data table, whose labels and values are unique and whose
//     defaults are none of the listed values (the bridge's reader);
//   - its label space fits a stratum number (at most 65536 cells);
//   - every leaf can hold: at least one of its ids is in the format's
//     vocabulary (a species of the roster, a move of some pool, an ability of
//     some species, an item the format allows), so no value is dead;
//   - generator.json is a TeamGeneratorConfig whose item weights name engine
//     items (the bridge's reader).

import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  type Engine,
  type EngineData,
  exportEngineData,
  exportTeamVocabulary,
  readRoleSetTable,
  readTeamGeneratorConfig,
  readTeamLabelSet,
  ROLE_SETS_FILE,
  roleSetTableHash,
  type TeamGeneratorConfig,
  type TeamLabelSet,
  type TeamVocabulary,
} from "@areneon/showdown-bridge";
import { cellCount } from "./labeller.ts";

export const TEAMS_CONFIG_DIR = fileURLToPath(new URL("../../../config/teams", import.meta.url));

const GOLDEN_FIXTURES_DIR = fileURLToPath(new URL("../../../fixtures/showdown", import.meta.url));

type Predicate = TeamLabelSet["labels"][number]["values"][number]["when"];

/** The ids of a leaf and the vocabulary ids they must meet. */
function reachable(leaf: Exclude<Predicate, { kind: "all_of" }>, vocabulary: TeamVocabulary): boolean {
  const roster = vocabulary.roster;
  switch (leaf.kind) {
    case "has_species":
      return leaf.species.some((id) => roster.some((r) => r.id === id));
    case "has_move":
      return leaf.moves.some((id) => roster.some((r) => r.moves.includes(id)));
    case "has_ability":
      return leaf.abilities.some((id) => roster.some((r) => r.abilities.includes(id)));
    case "has_item":
      return leaf.items.some((id) => vocabulary.items.includes(id));
  }
}

/** Problems of a label set against a vocabulary; empty when it is sound. */
export function labelProblems(set: TeamLabelSet, vocabulary: TeamVocabulary): string[] {
  const problems: string[] = [];
  try {
    cellCount(set);
  } catch (error) {
    problems.push((error as Error).message);
  }
  for (const label of set.labels) {
    for (const { value, when } of label.values) {
      for (const leaf of when.kind === "all_of" ? when.of : [when]) {
        if (!reachable(leaf, vocabulary)) problems.push(`label ${label.label_id}, value ${value}: no id of its ${leaf.kind} is in the vocabulary, so it can never hold`);
      }
    }
  }
  return problems;
}

/** A `role_sets` component must name the golden role-set table, which must read against the vocabulary. */
function roleSetProblems(config: TeamGeneratorConfig, format: string, vocabulary: TeamVocabulary, fixtures: string): string[] {
  const named = config.proposal?.components.find((c) => c.kind === "role_sets");
  if (named === undefined) return [];
  const file = join(fixtures, format, ROLE_SETS_FILE);
  if (!existsSync(file)) return [`the role-set table ${format}/${ROLE_SETS_FILE} is missing`];
  const table = readRoleSetTable(JSON.parse(readFileSync(file, "utf8")), vocabulary);
  return roleSetTableHash(table) === named.role_set_table_hash ? [] : [`role_sets component: role_set_table_hash is not the committed table's (${roleSetTableHash(table)})`];
}

/** Check every format directory under `root`; returns the problems, one line each. */
export function checkTeamConfigs(
  engine: Engine,
  root: string = TEAMS_CONFIG_DIR,
  vocabularyOf: (engine: Engine, format: string, table: EngineData) => TeamVocabulary = exportTeamVocabulary,
  fixtures: string = GOLDEN_FIXTURES_DIR,
): string[] {
  const problems: string[] = [];
  for (const format of readdirSync(root).sort()) {
    const dir = join(root, format);
    const table = exportEngineData(engine, format);
    const vocabulary = vocabularyOf(engine, format, table);
    const read = (name: string): unknown => JSON.parse(readFileSync(join(dir, name), "utf8"));
    // Every generator*.json is a TeamGeneratorConfig; a role_sets component must name the committed role-set table.
    const generators = readdirSync(dir).filter((f) => /^generator(\..+)?\.json$/.test(f)).sort();
    for (const [name, check] of [
      ["labels.json", (v: unknown) => labelProblems(readTeamLabelSet(v, table), vocabulary)],
      ...(generators.includes("generator.json") ? [] : [["generator.json", () => []] as const]),
      ...generators.map((g) => [g, (v: unknown) => roleSetProblems(readTeamGeneratorConfig(v, table), format, vocabulary, fixtures)] as const),
    ] as const) {
      if (!existsSync(join(dir, name))) {
        problems.push(`${format}/${name} is missing`);
        continue;
      }
      try {
        for (const problem of check(read(name))) problems.push(`${format}/${name}: ${problem}`);
      } catch (error) {
        problems.push(`${format}/${name}: ${(error as Error).message}`);
      }
    }
  }
  return problems;
}
