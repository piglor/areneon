// `make contracts`: fail on any schema outside its ADR-008 profile, or on
// generated types that differ from what the pinned generator produces.
// `make contracts-update` (--write) regenerates the checked-in types.
import { runContracts } from "../src/contracts.ts";

const write = process.argv.includes("--write");
const findings = await runContracts({ write });
for (const f of findings) console.error(`${f.path}: ${f.message}`);
if (findings.length > 0) {
  if (!write) console.error("Run `make contracts-update` after fixing the schemas.");
  process.exit(1);
}
console.log(write ? "contracts: generated types written" : "contracts: schemas conform, no drift");
