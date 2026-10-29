// Check config/baselines/damage-aware.<version>.json against the pinned engine's team vocabulary
// (ADR-018 section 3). Part of `make ts`.
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { isDeepStrictEqual } from "node:util";
import { beliefConfigHash, readBeliefConfig } from "../src/belief-config.ts";
import { DAMAGE_AWARE_CONFIG_DIR, loadDamageAwareConfig, loadDamageAwareConfigV2 } from "../src/damage-aware-config.ts";
import { loadEngine } from "../src/engine.ts";
import { goldenTeamVocabulary } from "../src/golden.ts";
import { readUsagePrior, usagePriorHash } from "../src/usage-prior.ts";

const engine = loadEngine();
// The committed vocabulary is the pinned engine's: team-vocabulary.test.ts holds it equal to a fresh export.
const vocabulary = goldenTeamVocabulary();
const files = readdirSync(DAMAGE_AWARE_CONFIG_DIR).filter((file) => /^damage-aware\.v\d+\.json$/.test(file));
let failed = files.length === 0 || vocabulary.showdown_commit !== engine.commit;
if (vocabulary.showdown_commit !== engine.commit) console.error("the committed team vocabulary is for another engine commit");
for (const file of files) {
  try {
    const version = file.replace(/^damage-aware\.(v\d+)\.json$/, "$1");
    if (version === "v1") {
      // v1 plays the legacy (reversed) choice-location numbering: it never says `target_convention`.
      if (loadDamageAwareConfig(version, vocabulary).target_convention !== undefined) throw new Error("v1 must not select a target convention (it is kept bug-for-bug)");
    } else if (version === "v3") {
      // v3 is v1's values with the engine's target convention (#1180); nothing else may differ.
      const v1 = loadDamageAwareConfig("v1", vocabulary);
      const v3 = loadDamageAwareConfig(version, vocabulary);
      const { target_convention, ...rest } = v3;
      if (target_convention !== "engine") throw new Error("v3 must select the engine's target convention");
      if (!isDeepStrictEqual(rest, v1)) throw new Error("v3 must equal v1 in every value but the target convention");
    }
    else if (version === "v2") {
      // The belief record names the prior and the belief config by hash: the committed files must be those.
      const record = loadDamageAwareConfigV2(vocabulary);
      const dir = join(DAMAGE_AWARE_CONFIG_DIR, "..", "opponent-model", record.format_id);
      const prior = readUsagePrior(JSON.parse(readFileSync(join(dir, "generator-matched.uniform-v1.json"), "utf8")), vocabulary);
      const belief = readBeliefConfig(JSON.parse(readFileSync(join(dir, "belief.v1.json"), "utf8")), vocabulary);
      if (usagePriorHash(prior) !== record.assumed_set.prior_hash) throw new Error("the committed usage prior is not the one the record names");
      if (beliefConfigHash(belief) !== record.assumed_set.belief_config_hash) throw new Error("the committed belief config is not the one the record names");
      // The belief config with damage evidence (ticket #1132) is read against the vocabulary too; no record names it yet.
      if (readBeliefConfig(JSON.parse(readFileSync(join(dir, "belief.damage-v1.json"), "utf8")), vocabulary).damage_evidence === undefined) throw new Error("belief.damage-v1.json must carry damage_evidence");
    } else throw new Error("no reader for this record version");
  } catch (error) {
    console.error(`${file}: ${(error as Error).message}`);
    failed = true;
  }
}
if (failed) process.exit(1);
console.log("damage-aware configs: ok");
