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
import { deriveGeneratorPrior, megaHolderAbilities, megaPreviewSpecies } from "../src/generator-prior.ts";
import { deriveMixturePrior } from "../src/mixture-prior.ts";
import { readRoleSetTable, ROLE_SETS_FILE } from "../src/role-sets.ts";
import { readUsageWeightTable, usageTableHash } from "../src/usage-weight-table.ts";
import { readdirSync } from "node:fs";
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
  console.error("usage: export-generator-prior.ts --commit <40 hex> [--config FILE] [--out FILE] [--role-sets FILE] [--usage-table FILE]");
  process.exit(2);
}
const engine = loadEngine();
const table = exportEngineData(engine, GOLDEN_FORMAT);
const vocabulary = readTeamVocabulary(JSON.parse(readFileSync(join(GOLDEN_DIR, TEAM_VOCABULARY_FILE), "utf8")), table);
const configFile = flag("--config") ?? join(ROOT, "config", "teams", GOLDEN_FORMAT, "generator.json");
const config = readTeamGeneratorConfig(JSON.parse(readFileSync(configFile, "utf8")), table);
const shows = megaPreviewSpecies(engine, vocabulary, table);
const holder_abilities = megaHolderAbilities(engine, vocabulary, shows);
// A config with a mixture `proposal` is derived by the mixture builder (a separate record); without one the
// uniform builder runs, byte for byte as before. The tables a mixture names are found by hash.
let prior;
if (config.proposal === undefined) {
  prior = deriveGeneratorPrior({ vocabulary, config, generator_commit: commit, shows_as: shows, holder_abilities });
} else {
  const kinds = new Set(config.proposal.components.map((c) => c.kind));
  const role_sets = kinds.has("role_sets") ? readRoleSetTable(JSON.parse(readFileSync(flag("--role-sets") ?? join(GOLDEN_DIR, ROLE_SETS_FILE), "utf8")), vocabulary) : undefined;
  let usage;
  if (kinds.has("usage")) {
    const dir = join(ROOT, "config", "teams", GOLDEN_FORMAT, "usage");
    const files = flag("--usage-table") !== undefined ? [flag("--usage-table") as string] : readdirSync(dir).filter((f) => /^\d{4}-\d{2}-\d+\.json$/.test(f)).map((f) => join(dir, f));
    const named = config.proposal.components.find((c) => c.kind === "usage");
    usage = files.map((f) => readUsageWeightTable(JSON.parse(readFileSync(f, "utf8")), vocabulary)).find((t) => usageTableHash(t) === (named as { usage_table_hash: string }).usage_table_hash);
    if (usage === undefined) throw new Error("no usage table under config/teams/<format>/usage has the hash the config names");
  }
  prior = deriveMixturePrior({ vocabulary, config, generator_commit: commit, shows_as: shows, holder_abilities, ...(role_sets === undefined ? {} : { role_sets }), ...(usage === undefined ? {} : { usage }) });
}
readUsagePrior(prior, vocabulary);
const out = flag("--out") ?? join(ROOT, "config", "opponent-model", GOLDEN_FORMAT, `generator-matched.${config.config_id}.json`);
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, `${jcs(prior as unknown as Json)}\n`);
console.log(`wrote ${out}`);
console.log(`prior_hash ${usagePriorHash(prior)}`);
