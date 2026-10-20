// Sealed team generation, reveal and verification (ADR-013 slice (g); see
// docs/runbooks/evaluation-store.md section 9). Run as the store user, from a
// reviewed checkout the training user cannot write to:
//   node packages/team-dataset/scripts/sealed.ts <generate|reveal-hash|retire|verify> ...
import { runSealedCli } from "../src/sealed-command.ts";

process.exitCode = runSealedCli(process.argv.slice(2), { out: (line) => console.log(line), err: (line) => console.error(line) });
