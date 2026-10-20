// Generate a synthetic team dataset or its pilot (ADR-013 section 2); see
// src/command.ts for the options.
//   node packages/team-dataset/scripts/generate.ts generate --config ... --out DIR
import { runCli } from "../src/command.ts";

process.exitCode = runCli(process.argv.slice(2), { out: (line) => console.log(line), err: (line) => console.error(line) });
