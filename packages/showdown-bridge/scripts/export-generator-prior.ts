// Derive the `generator-matched` usage prior of a generator config (ADR-025 section 10) and write it:
//   node packages/showdown-bridge/scripts/export-generator-prior.ts --commit <40 hex> [--config FILE] [--out FILE]
// The default config is config/teams/<golden format>/generator.json and the default output
// config/opponent-model/<format>/generator-matched.<config_id>.json. `--commit` is the commit of the
// generator code whose draw rules the prior reproduces (packages/team-dataset/src/proposal.ts; the
// golden draw digest in packages/team-dataset/test/generator-golden.test.ts binds it). Review the diff.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { type Json, jcs } from "../src/canonical-json.ts";
import { exportEngineData } from "../src/engine-data.ts";
import { loadEngine } from "../src/engine.ts";
import { deriveGeneratorPrior, megaPreviewSpecies } from "../src/generator-prior.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "../src/golden.ts";
import { readTeamGeneratorConfig } from "../src/team-records.ts";
import { readTeamVocabulary, TEAM_VOCABULARY_FILE } from "../src/team-vocabulary.ts";
import { usagePriorHash, readUsagePrior } from "../src/usage-prior.ts";

const ROOT = join(import.meta.dirname, "..", "..", "..");
const args = process.argv.slice(2);
const flag = (name: string): string | undefined => {
  const k = args.indexOf(name);
  return k < 0 ? undefined : args[k + 1];
};
const commit = flag("--commit");
if (commit === undefined) {
  console.error("usage: export-generator-prior.ts --commit <40 hex> [--config FILE] [--out FILE]");
  process.exit(2);
}
const engine = loadEngine();
const table = exportEngineData(engine, GOLDEN_FORMAT);
const vocabulary = readTeamVocabulary(JSON.parse(readFileSync(join(GOLDEN_DIR, TEAM_VOCABULARY_FILE), "utf8")), table);
const configFile = flag("--config") ?? join(ROOT, "config", "teams", GOLDEN_FORMAT, "generator.json");
const config = readTeamGeneratorConfig(JSON.parse(readFileSync(configFile, "utf8")), table);
const prior = deriveGeneratorPrior({ vocabulary, config, generator_commit: commit, shows_as: megaPreviewSpecies(engine, vocabulary, table) });
readUsagePrior(prior, vocabulary);
const out = flag("--out") ?? join(ROOT, "config", "opponent-model", GOLDEN_FORMAT, `generator-matched.${config.config_id}.json`);
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, `${jcs(prior as unknown as Json)}\n`);
console.log(`wrote ${out}`);
console.log(`prior_hash ${usagePriorHash(prior)}`);
