/* eslint-disable @typescript-eslint/no-explicit-any */
// The DamageAwareConfig's share of the sample corpus (ADR-008 section 2, ADR-018 section 3): the
// committed config as the valid sample, schema-invalid mutations, and reader-invalid ones that
// pass the schema and that `readDamageAwareConfig` refuses against the team vocabulary.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { DAMAGE_AWARE_CONFIG_DIR } from "./damage-aware-config.ts";

type Cases = Record<string, unknown>;

function mutated(value: unknown, change: (copy: any) => void): Record<string, unknown> {
  const copy = structuredClone(value) as Record<string, unknown>;
  change(copy);
  return copy;
}

export function damageAwareSamples(): { valid: Cases; invalid: Cases; readerInvalid: Cases } {
  const config = JSON.parse(readFileSync(join(DAMAGE_AWARE_CONFIG_DIR, "damage-aware.v1.json"), "utf8")) as unknown;
  const second = JSON.parse(readFileSync(join(DAMAGE_AWARE_CONFIG_DIR, "damage-aware.v2.json"), "utf8")) as unknown;
  return {
    valid: { "damage-aware-config/v1": config, "damage-aware-config-2/v2": second },
    invalid: {
      "damage-aware-config-2/extra-field": mutated(second, (c) => (c.note = "x")),
      "damage-aware-config-2/field-rules-instead-of-belief": mutated(second, (c) => (c.assumed_set = (config as any).assumed_set)),
      "damage-aware-config-2/unknown-source": mutated(second, (c) => (c.assumed_set.source = "rules")),
      "damage-aware-config-2/no-prior-hash": mutated(second, (c) => delete c.assumed_set.prior_hash),
      "damage-aware-config-2/prior-hash-not-hex": mutated(second, (c) => (c.assumed_set.prior_hash = "E636")),
      "damage-aware-config-2/zero-worlds": mutated(second, (c) => (c.assumed_set.worlds = 0)),
      "damage-aware-config-2/fractional-worlds": mutated(second, (c) => (c.assumed_set.worlds = 8.5)),
      "damage-aware-config-2/wrong-version": mutated(second, (c) => (c.schema_version = 1)),
      "damage-aware-config/extra-field": mutated(config, (c) => (c.note = "x")),
      "damage-aware-config/fractional-weight": mutated(config, (c) => (c.scoring.ko_weight_ppm = 0.5)),
      "damage-aware-config/weight-over-one": mutated(config, (c) => (c.scoring.crit_weight_ppm = 1000001)),
      "damage-aware-config/negative-epsilon": mutated(config, (c) => (c.epsilon_ppm = -1)),
      "damage-aware-config/unknown-ability-rule": mutated(config, (c) => (c.assumed_set.ability = "random")),
      "damage-aware-config/unknown-item-kind": mutated(config, (c) => (c.assumed_set.item = { kind: "guess" })),
      "damage-aware-config/fixed-item-without-id": mutated(config, (c) => (c.assumed_set.item = { kind: "fixed" })),
      "damage-aware-config/missing-stat-point": mutated(config, (c) => delete c.assumed_set.stat_points.spe),
      "damage-aware-config/unknown-rounding": mutated(config, (c) => (c.hp_rounding = "banker")),
    },
    readerInvalid: {
      "damage-aware-config-2/other-format": mutated(second, (c) => (c.format_id = "gen9ou")),
      "damage-aware-config/unknown-nature": mutated(config, (c) => (c.assumed_set.nature = "grumpy")),
      "damage-aware-config/unknown-fixed-item": mutated(config, (c) => (c.assumed_set.item = { kind: "fixed", item: "notanitem" })),
      "damage-aware-config/stat-points-over-limit": mutated(config, (c) => (c.assumed_set.stat_points.hp = 255)),
      "damage-aware-config/other-format": mutated(config, (c) => (c.format_id = "gen9ou")),
    },
  };
}
