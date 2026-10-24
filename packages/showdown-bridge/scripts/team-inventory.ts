// Owner-team feature inventory (#775). Run it yourself, in an interactive
// terminal, so your teams never enter a chat or a log:
//   node packages/showdown-bridge/scripts/team-inventory.ts [--teams-dir DIR] [--format ID]
// Reads private-data/teams/*.txt through a PasteDirectorySource. Stdout gets
// counts only; the detail goes to /dev/tty. It writes no file, and it refuses
// unless stdin, stdout and stderr are terminals and no agent variable is set.
import { fileURLToPath } from "node:url";
import { loadEngine } from "../src/engine.ts";
import { currentSession, openControllingTerminal, OwnerSessionError, terminalRefusal } from "../src/owner-session.ts";
import { PasteDirectorySource } from "../src/private-team-source.ts";
import { InventoryError, parseInventoryArgs, runInventory, type InventoryTerminal } from "../src/team-inventory.ts";

const repoRoot = fileURLToPath(new URL("../../../", import.meta.url));

const terminal: InventoryTerminal = {
  refusal() {
    const { env, isTerminal } = currentSession();
    return terminalRefusal(env, isTerminal);
  },
  openDetail() {
    try {
      return openControllingTerminal();
    } catch (error) {
      // The shared guard's fixed words; the inventory reports them as its own refusal.
      throw new InventoryError(error instanceof OwnerSessionError ? error.message : "refusing: no controlling terminal to show the detail on");
    }
  },
};

try {
  const { teamsDir, formatId } = parseInventoryArgs(process.argv.slice(2), process.env, `${repoRoot}private-data/teams`);
  const source = new PasteDirectorySource(teamsDir);
  const summary = await runInventory(
    { engine: loadEngine(), source, ...(formatId === undefined ? {} : { formatId }) },
    terminal,
  );
  console.log(JSON.stringify({ ...summary.counts, gap_matches: summary.gap_matches }));
} catch (error) {
  // Never print an engine or parser message: it can quote a team.
  console.error(error instanceof InventoryError ? error.message : `failed: ${error instanceof Error ? error.name : "error"}`);
  process.exit(error instanceof InventoryError && error.message.startsWith("usage") ? 2 : 1);
}
