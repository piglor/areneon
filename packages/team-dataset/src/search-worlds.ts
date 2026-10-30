// The composition root of the search's opponent worlds (ADR-026 section 3, slice B; ADR-025 sections 4 and 10).
// It joins the opponent model (`packages/opponent-model`, engine-free), the pinned engine's set legality (the
// bridge) and the generator's own spread routine (this package) into the bridge's `WorldSamplePort`: the belief
// of the reader's opponent half, then draws BY INDEX through ADR-025's `sample` over `worldStream(stream, index)`
// (the stream `sampleWorlds` gives world `index`). There is one sampler, in the model; this file adds none.
//
// Two prior arms through the same `sample`. Primary: the committed `generator-matched` prior, for opponents the
// `uniform-v1` generator drew (labelled as an upper bound for this evaluation). Secondary: the committed
// `role-sets` prior, whose fill is the primary prior (checked by hash) and whose table is the golden role-set
// table (checked by hash). A prior of another format is refused by `readUsagePrior` against the vocabulary
// (review finding N10: never the M-C prior silently after a pin bump); a prior not derived from the committed generator config
// and a role-sets prior over another fill or table are refused (`SearchWorldsError`).

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { belief, beliefIdentityHash, createSampler, worldStream } from "@areneon/opponent-model";
import {
  beliefConfigHash,
  DAMAGE_CHECK_FORMAT,
  createSetLegality,
  type Engine,
  GENERATOR_MATCHED_LABEL,
  generatorConfigHash,
  type PriorKind,
  readBeliefConfig,
  readEngineData,
  readRoleSetTable,
  readTeamGeneratorConfig,
  readUsagePrior,
  refusalCode,
  roleSetTableHash,
  ROLE_SETS_LABEL,
  teamVocabularyHash,
  type TeamVocabulary,
  usagePriorHash,
  type VersionManifest,
  type WorldSamplePort,
} from "@areneon/showdown-bridge";
import { generatorSpreadSampler } from "./spread-sampler.ts";

const REPO_ROOT = fileURLToPath(new URL("../../..", import.meta.url));

export class SearchWorldsError extends Error {
  override name = "SearchWorldsError";
}

/** The committed files an arm is built from. Tests pass other directories. */
export interface SearchWorldsFiles {
  /** `config/opponent-model/<format>`: the two prior records and `belief.v1.json`. */
  readonly model_dir: string;
  readonly generator_prior_file: string;
  readonly role_sets_prior_file: string;
  /** `config/teams/<format>/generator.json`. */
  readonly generator_file: string;
  /** The golden engine-data table the generator config is read against. */
  readonly engine_data_file: string;
  /** The golden role-set table. */
  readonly role_sets_file: string;
}

export const searchWorldsFiles = (formatId: string): SearchWorldsFiles => {
  const model_dir = join(REPO_ROOT, "config", "opponent-model", formatId);
  return {
    model_dir,
    generator_prior_file: join(model_dir, "generator-matched.uniform-v1.json"),
    role_sets_prior_file: join(model_dir, "role-sets.fill-uniform-v1.json"),
    generator_file: join(REPO_ROOT, "config", "teams", formatId, "generator.json"),
    engine_data_file: join(REPO_ROOT, "fixtures", "showdown", formatId, "engine-data.json"),
    role_sets_file: join(REPO_ROOT, "fixtures", "showdown", formatId, "role-sets.json"),
  };
};

export interface SearchWorldsContext {
  readonly engine: Engine;
  readonly vocabulary: TeamVocabulary;
  /** SHA-256 of the public-state reader's file (a part of the belief identity). */
  readonly reader_sha256: string;
  readonly manifest: VersionManifest;
  readonly arm: PriorKind;
}

const json = (file: string): unknown => JSON.parse(readFileSync(file, "utf8")) as unknown;

export function createSearchWorlds(context: SearchWorldsContext, files: SearchWorldsFiles = searchWorldsFiles(context.vocabulary.format_id)): WorldSamplePort {
  const { engine, vocabulary, reader_sha256, manifest, arm } = context;
  const generator = readTeamGeneratorConfig(json(files.generator_file), readEngineData(json(files.engine_data_file)));
  const matched = readUsagePrior(json(files.generator_prior_file), vocabulary);
  if (matched.source.kind !== "generator-matched" || matched.source.generator_config_hash !== generatorConfigHash(generator)) {
    throw new SearchWorldsError("the generator-matched prior was not derived from the committed generator config");
  }
  let prior = matched;
  if (arm === "role-sets") {
    prior = readUsagePrior(json(files.role_sets_prior_file), vocabulary);
    if (prior.source.kind !== "role-sets") throw new SearchWorldsError("the role-sets prior record is not of kind role-sets");
    if (prior.source.fill_prior_hash !== usagePriorHash(matched)) throw new SearchWorldsError("the role-sets prior was filled from another prior than the committed generator-matched one");
    if (prior.source.role_set_table_hash !== roleSetTableHash(readRoleSetTable(json(files.role_sets_file), vocabulary))) throw new SearchWorldsError("the role-sets prior was not derived from the committed role-set table");
  }
  const beliefConfig = readBeliefConfig(json(join(files.model_dir, "belief.v1.json")), vocabulary);
  const prior_hash = usagePriorHash(prior);
  const belief_config_hash = beliefConfigHash(beliefConfig);
  const vocabulary_hash = teamVocabularyHash(vocabulary);
  const view = { ...vocabulary, vocabulary_hash };
  const sampler = createSampler({ legality: createSetLegality({ engine, vocabulary }), spreads: generatorSpreadSampler(vocabulary, generator) });
  return {
    identity_hash: beliefIdentityHash({ prior_hash, belief_config_hash, vocabulary_hash, reader_sha256, version_manifest: manifest, ...(beliefConfig.damage_evidence === undefined ? {} : { damage_check_format: DAMAGE_CHECK_FORMAT }) }),
    prior_hash,
    belief_config_hash,
    prior_format_id: prior.format_id,
    prior_kind: arm,
    prior_label: arm === "generator-matched" ? GENERATOR_MATCHED_LABEL : ROLE_SETS_LABEL,
    open({ opponent, flags, trapping }) {
      let computed;
      try {
        computed = belief({ prior, vocabulary: view, config: beliefConfig, opponent, flags, trapping });
      } catch (error) {
        const code = refusalCode(error);
        if (code === null) throw error;
        return { kind: "refused", code };
      }
      return { kind: "source", source: { draw: (stream, index) => sampler.sample(computed, worldStream(stream, index)) } };
    },
  };
}
