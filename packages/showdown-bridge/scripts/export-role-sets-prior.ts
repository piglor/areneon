// Derive the `role-sets` usage prior (ADR-025 section 2) from the golden role-set table and the committed
// `generator-matched` fill prior, and write it:
//   node packages/showdown-bridge/scripts/export-role-sets-prior.ts [--fill FILE] [--out FILE]
// The default fill is config/opponent-model/<golden format>/generator-matched.<config_id>.json and the
// default output config/opponent-model/<format>/role-sets.fill-<config_id>.json. Review the diff.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { type Json, jcs } from "../src/canonical-json.ts";
import { exportEngineData } from "../src/engine-data.ts";
import { loadEngine } from "../src/engine.ts";
import { megaPreviewSpecies } from "../src/generator-prior.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "../src/golden.ts";
import { deriveRoleSetsPrior } from "../src/role-sets-prior.ts";
import { readRoleSetTable, ROLE_SETS_FILE } from "../src/role-sets.ts";
import { readTeamVocabulary, TEAM_VOCABULARY_FILE } from "../src/team-vocabulary.ts";
import { readUsagePrior, usagePriorHash } from "../src/usage-prior.ts";

const ROOT = join(import.meta.dirname, "..", "..", "..");
const args = process.argv.slice(2);
const flag = (name: string): string | undefined => {
  const k = args.indexOf(name);
  return k < 0 ? undefined : args[k + 1];
};
const engine = loadEngine();
const table = exportEngineData(engine, GOLDEN_FORMAT);
const vocabulary = readTeamVocabulary(JSON.parse(readFileSync(join(GOLDEN_DIR, TEAM_VOCABULARY_FILE), "utf8")), table);
const roleSets = readRoleSetTable(JSON.parse(readFileSync(join(GOLDEN_DIR, ROLE_SETS_FILE), "utf8")), vocabulary);
const dir = join(ROOT, "config", "opponent-model", GOLDEN_FORMAT);
const fillFile = flag("--fill") ?? join(dir, "generator-matched.uniform-v1.json");
const fill = readUsagePrior(JSON.parse(readFileSync(fillFile, "utf8")), vocabulary);
const prior = deriveRoleSetsPrior({ vocabulary, table: roleSets, fill, shows_as: megaPreviewSpecies(engine, vocabulary, table) });
const out = flag("--out") ?? join(dir, "role-sets.fill-uniform-v1.json");
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, `${jcs(prior as unknown as Json)}\n`);
console.log(`wrote ${out}`);
console.log(`prior_hash ${usagePriorHash(prior)}`);
