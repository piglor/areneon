// Owner-team feature inventory (#775). Run it yourself, so your teams never
// enter a chat or a log:
//   node packages/showdown-bridge/scripts/team-inventory.ts [--teams-dir DIR] [--format ID]
// Reads private-data/teams/*.txt, writes private-data/reports/team-inventory.json,
// and prints only counts and the report path.
import { fileURLToPath } from "node:url";
import { loadEngine } from "../src/engine.ts";
import { InventoryError, parseInventoryArgs, runInventory } from "../src/team-inventory.ts";

const repoRoot = fileURLToPath(new URL("../../../", import.meta.url));

try {
  const options = parseInventoryArgs(process.argv.slice(2), process.env, `${repoRoot}private-data/teams`);
  const { path, counts } = runInventory({ engine: loadEngine(), ...options });
  console.log(JSON.stringify({ report: path, ...counts }));
} catch (error) {
  // Never print an engine or parser message: it can quote a team.
  console.error(error instanceof InventoryError ? error.message : `failed: ${error instanceof Error ? error.name : "error"}`);
  process.exit(error instanceof InventoryError && error.message.startsWith("usage") ? 2 : 1);
}
