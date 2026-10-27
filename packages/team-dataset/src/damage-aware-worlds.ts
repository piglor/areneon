// The composition root of `damage-aware@v2` (ADR-025 sections 4 to 6, slice 3): it joins the opponent
// model (`packages/opponent-model`, engine-free), the pinned engine's set legality (the bridge) and the
// generator's own spread routine (this package) into the `BeliefWorldsPort` the baseline asks for. The bridge
// imports none of them (the port is structural), and the model imports neither the bridge nor this package.
//
// It loads the usage prior and the belief config the v2 record names BY HASH from `config/opponent-model/<format>/`
// and refuses any file whose hash differs, a prior of another format (review finding N10: never the M-C prior
// silently after a pin bump), a prior that is not the one derived from the committed generator config, and a
// vocabulary the prior was not derived against. The belief identity hash of ADR-025 section 5 is computed here
// and goes into the baseline's identity.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { belief, beliefIdentityHash, createSampler } from "@areneon/opponent-model";
import {
  type BeliefWorldsContext,
  type BeliefWorldsFactory,
  type BeliefWorldsPort,
  beliefConfigHash,
  createSetLegality,
  generatorConfigHash,
  readBeliefConfig,
  readEngineData,
  readTeamGeneratorConfig,
  readUsagePrior,
  teamVocabularyHash,
  usagePriorHash,
} from "@areneon/showdown-bridge";
import { generatorSpreadSampler } from "./spread-sampler.ts";

const REPO_ROOT = fileURLToPath(new URL("../../..", import.meta.url));

export class BeliefWorldsError extends Error {
  override name = "BeliefWorldsError";
}

/** The committed files a v2 record's belief is built from. Tests pass other directories. */
export interface BeliefWorldsFiles {
  /** `config/opponent-model/<format>`: `generator-matched.uniform-v1.json` and `belief.v1.json`. */
  readonly model_dir: string;
  /** `config/teams/<format>/generator.json`. */
  readonly generator_file: string;
  /** The golden engine-data table the generator config is read against. */
  readonly engine_data_file: string;
}

export const beliefWorldsFiles = (formatId: string): BeliefWorldsFiles => ({
  model_dir: join(REPO_ROOT, "config", "opponent-model", formatId),
  generator_file: join(REPO_ROOT, "config", "teams", formatId, "generator.json"),
  engine_data_file: join(REPO_ROOT, "fixtures", "showdown", formatId, "engine-data.json"),
});

const json = (file: string): unknown => JSON.parse(readFileSync(file, "utf8")) as unknown;

/** The port of one v2 record: belief over the committed prior, N worlds on the caller's stream, legality by the pinned validator. */
export function createBeliefWorlds(context: BeliefWorldsContext, files: BeliefWorldsFiles = beliefWorldsFiles(context.config.format_id)): BeliefWorldsPort {
  const { engine, vocabulary, config, reader_sha256, manifest } = context;
  const record = config.assumed_set;
  const prior = readUsagePrior(json(join(files.model_dir, "generator-matched.uniform-v1.json")), vocabulary);
  if (prior.format_id !== config.format_id) throw new BeliefWorldsError("the usage prior is for another format than the record's: it is never used silently");
  const prior_hash = usagePriorHash(prior);
  if (prior_hash !== record.prior_hash) throw new BeliefWorldsError("the committed usage prior is not the one the record names");
  const beliefConfig = readBeliefConfig(json(join(files.model_dir, "belief.v1.json")), vocabulary);
  const belief_config_hash = beliefConfigHash(beliefConfig);
  if (belief_config_hash !== record.belief_config_hash) throw new BeliefWorldsError("the committed belief config is not the one the record names");
  const generator = readTeamGeneratorConfig(json(files.generator_file), readEngineData(json(files.engine_data_file)));
  if (prior.source.kind !== "generator-matched" || prior.source.generator_config_hash !== generatorConfigHash(generator)) {
    throw new BeliefWorldsError("the usage prior was not derived from the committed generator config");
  }
  const vocabulary_hash = teamVocabularyHash(vocabulary);
  const view = { ...vocabulary, vocabulary_hash };
  const sampler = createSampler({ legality: createSetLegality({ engine, vocabulary }), spreads: generatorSpreadSampler(vocabulary, generator) });
  return {
    identity_hash: beliefIdentityHash({ prior_hash, belief_config_hash, vocabulary_hash, reader_sha256, version_manifest: manifest }),
    prior_hash,
    belief_config_hash,
    prior_format_id: prior.format_id,
    sample: ({ opponent, flags, stream, n }) => {
      const sampled = sampler.sampleWorlds(belief({ prior, vocabulary: view, config: beliefConfig, opponent, flags }), stream, n);
      return { worlds: sampled.worlds.map(({ world }) => ({ world })), redraws: sampled.redraws };
    },
  };
}

/** The factory the team-dataset scripts hand to the bridge's CLIs. */
export const beliefWorldsFactory: BeliefWorldsFactory = (context) => createBeliefWorlds(context);
