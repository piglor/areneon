// The belief filter's reviewed configuration (ADR-025 sections 3 and 5): reviewed data, never code.
// The schema checks the shape; the reader adds sorted and unique ids and that every identity-hiding
// ability is an ability of the team vocabulary. `beliefConfigHash` is the `belief_config_hash` of
// the belief identity.

import { ajvFor } from "@areneon/schema/ajv";
import beliefConfigSchema from "@areneon/schema/contracts/v1/belief-config.schema.json" with { type: "json" };
import type { BeliefConfig } from "@areneon/schema/types/v1/belief-config.ts";
import { type Json, jcs } from "./canonical-json.ts";
import { sha256Hex } from "./rules.ts";
import type { TeamVocabulary } from "./team-vocabulary.ts";

export type { BeliefConfig };

export class BeliefConfigError extends Error {
  override name = "BeliefConfigError";
}

const validConfig = ajvFor("contract").compile(beliefConfigSchema);

/** Read a config: the schema, strictly ascending ids, and each id in the vocabulary (the damage evidence's move targets are checked against the engine's when `createDamageCheck` is made). */
export function readBeliefConfig(value: unknown, vocabulary: TeamVocabulary): BeliefConfig {
  if (!validConfig(value)) throw new BeliefConfigError(`invalid BeliefConfig: ${JSON.stringify(validConfig.errors)}`);
  const config = structuredClone(value) as unknown as BeliefConfig;
  const known = new Set(vocabulary.roster.flatMap((r) => r.abilities));
  config.identity_hiding_abilities.forEach((id, k) => {
    if (k > 0 && !((config.identity_hiding_abilities[k - 1] as string) < id)) throw new BeliefConfigError(`identity_hiding_abilities: ${id} is repeated or out of order`);
    if (!known.has(id)) throw new BeliefConfigError(`ability ${id} is not in the team vocabulary`);
  });
  const damage = config.damage_evidence;
  if (damage !== undefined) {
    const abilities = new Set(vocabulary.roster.flatMap((r) => r.abilities));
    const moves = new Set(vocabulary.roster.flatMap((r) => r.moves));
    const items = new Set(vocabulary.items);
    const check = (what: string, list: readonly string[], known: ReadonlySet<string> | null, noun: string): void =>
      list.forEach((id, k) => {
        if (k > 0 && !((list[k - 1] as string) < id)) throw new BeliefConfigError(`damage_evidence.${what}: ${id} is repeated or out of order`);
        if (known !== null && !known.has(id)) throw new BeliefConfigError(`${noun} ${id} is not in the team vocabulary`);
      });
    check("non_isolating.abilities", damage.non_isolating.abilities, abilities, "ability");
    check("non_isolating.lasting_moves", damage.non_isolating.lasting_moves, moves, "move");
    check("non_isolating.moves", damage.non_isolating.moves, moves, "move");
    check("non_isolating.targets", damage.non_isolating.targets, null, "target");
    check("unmodelled.abilities", damage.unmodelled.abilities, abilities, "ability");
    check("unmodelled.items", damage.unmodelled.items, items, "item");
  }
  const speed = config.speed_evidence;
  if (speed !== undefined) {
    const abilities = new Set(vocabulary.roster.flatMap((r) => r.abilities));
    const moves = new Set(vocabulary.roster.flatMap((r) => r.moves));
    const items = new Set(vocabulary.items);
    const check = (what: string, list: readonly string[], known: ReadonlySet<string>, noun: string): void =>
      list.forEach((id, k) => {
        if (k > 0 && !((list[k - 1] as string) < id)) throw new BeliefConfigError(`speed_evidence.${what}: ${id} is repeated or out of order`);
        if (!known.has(id)) throw new BeliefConfigError(`${noun} ${id} is not in the team vocabulary`);
      });
    check("non_isolating.abilities", speed.non_isolating.abilities, abilities, "ability");
    check("non_isolating.lasting_moves", speed.non_isolating.lasting_moves, moves, "move");
    check("non_isolating.moves", speed.non_isolating.moves, moves, "move");
    check("unmodelled.abilities", speed.unmodelled.abilities, abilities, "ability");
    check("unmodelled.items", speed.unmodelled.items, items, "item");
  }
  return config;
}

/** The hash an identity names its config by: SHA-256 of the config's JCS bytes. */
export const beliefConfigHash = (config: BeliefConfig): string => sha256Hex(jcs(config as unknown as Json));
