// The mechanics probe on the command line (ADR-016 section 6):
//   node packages/showdown-bridge/scripts/mechanics.ts QUERY.json [--config FILE]
// Reads a MechanicsQuery and prints a MechanicsResult. See src/mechanics-cli.ts.
import { loadEngine } from "../src/engine.ts";
import { nodeMechanicsIo, runMechanicsCli } from "../src/mechanics-cli.ts";

process.exitCode = runMechanicsCli(process.argv.slice(2), nodeMechanicsIo(), loadEngine);
