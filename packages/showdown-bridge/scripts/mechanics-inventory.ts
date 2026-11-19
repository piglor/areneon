// Export the mechanics coverage inventory of the pinned engine (ADR-030
// groundwork G5, #1465):
//   node packages/showdown-bridge/scripts/mechanics-inventory.ts
// rewrites the golden inventory and its summary (fixtures/showdown/<format>/),
// the committed weights report (the fixtures-v1 dev split and the latest usage
// table) and docs/mechanics-inventory.md. It reads the golden team vocabulary and
// the pinned engine; run export-team-vocabulary.ts first after a pin bump.
//   --dataset DIR [--out FILE]   weights over the fixtures-v1 dev split PLUS the dev and
//                                training splits of DIR (never validation); written to FILE
//                                (default: a summary on stdout), nothing committed
//   --run-dir R --generations 100,101   counts over the synthetic Episodes of a run's committed
//                                generations (read only); printed, never written
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { exportEngineDataV2 } from "../src/engine-data.ts";
import { loadEngine } from "../src/engine.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "../src/golden.ts";
import { exportMechanicsInventory, INVENTORY_FILE, INVENTORY_SUMMARY, inventoryBytes, inventorySummary, readMechanicsInventory } from "../src/mechanics-inventory.ts";
import { INVENTORY_DOC, renderInventoryDoc } from "../src/mechanics-inventory-doc.ts";
import { readRunEpisodes, runReport } from "../src/mechanics-inventory-run.ts";
import { FIXTURES_V1_DIR, latestUsage, readDatasetTeams } from "../src/mechanics-inventory-teams.ts";
import { weightsBytes, weightsReport, WEIGHTS_FILE } from "../src/mechanics-inventory-weights.ts";
import { readTeamVocabulary, TEAM_VOCABULARY_FILE } from "../src/team-vocabulary.ts";
import { readUsageWeightTable } from "../src/usage-weight-table.ts";


const args = process.argv.slice(2);
const flag = (name: string): string | undefined => (args.includes(name) ? args[args.indexOf(name) + 1] : undefined);

const engine = loadEngine();
const v2 = exportEngineDataV2(engine, GOLDEN_FORMAT);
const vocabulary = readTeamVocabulary(JSON.parse(readFileSync(join(GOLDEN_DIR, TEAM_VOCABULARY_FILE), "utf8")));
const latest = latestUsage();
const usageFile = latest.file;
const usage = readUsageWeightTable(latest.value, vocabulary);

const dataset = flag("--dataset");
const runDir = flag("--run-dir");
if (runDir !== undefined) {
  const inventory = readMechanicsInventory(JSON.parse(readFileSync(join(GOLDEN_DIR, INVENTORY_FILE), "utf8")), vocabulary, v2);
  const generations = (flag("--generations") ?? "").split(",").filter((g) => g !== "").map(Number);
  if (generations.length === 0 || generations.some((g) => !Number.isSafeInteger(g))) throw new Error("--generations needs integers, for example 100,101");
  console.log(JSON.stringify(runReport(readRunEpisodes(runDir, generations), inventory)));
} else if (dataset !== undefined) {
  const inventory = readMechanicsInventory(JSON.parse(readFileSync(join(GOLDEN_DIR, INVENTORY_FILE), "utf8")), vocabulary, v2);
  const base = readDatasetTeams(engine, FIXTURES_V1_DIR);
  const extra = readDatasetTeams(engine, dataset);
  const report = weightsReport(inventory, { teams: [...base.teams, ...extra.teams], dataset: { id: `fixtures-v1+${extra.files.join("+")}`, file_sha256: `${base.sha256}${extra.sha256}` }, usage: { file: usageFile, table: usage }, team_size: vocabulary.rules.max_team_size });
  const out = flag("--out");
  if (out !== undefined) writeFileSync(out, weightsBytes(report));
  console.log(JSON.stringify({ teams: report.dataset.teams, damage_entities: report.damage.length, entities_for_90: report.coverage.teams.covered_ppm.findIndex((c) => c >= 900_000), pairs_seen: report.pairs.pairs_seen }));
} else {
  const inventory = exportMechanicsInventory(engine, GOLDEN_FORMAT, vocabulary, v2);
  writeFileSync(join(GOLDEN_DIR, INVENTORY_FILE), inventoryBytes(inventory));
  const summary = inventorySummary(inventory);
  writeFileSync(join(GOLDEN_DIR, INVENTORY_SUMMARY), `${JSON.stringify(summary, null, 2)}\n`);
  const teams = readDatasetTeams(engine, FIXTURES_V1_DIR);
  const report = weightsReport(inventory, { teams: teams.teams, dataset: { id: "fixtures-v1", file_sha256: teams.sha256 }, usage: { file: usageFile, table: usage }, team_size: vocabulary.rules.max_team_size });
  writeFileSync(join(GOLDEN_DIR, WEIGHTS_FILE), weightsBytes(report));
  writeFileSync(INVENTORY_DOC, renderInventoryDoc(inventory, report));
  console.log(`wrote ${INVENTORY_FILE}, ${INVENTORY_SUMMARY}, ${WEIGHTS_FILE} and ${INVENTORY_DOC}`);
  console.log(`inventory_hash ${summary.inventory_hash} ${JSON.stringify(summary.counts)}`);
}
