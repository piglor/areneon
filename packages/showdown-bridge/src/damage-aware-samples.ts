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
  return {
    valid: { "damage-aware-config/v1": config },
    invalid: {
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
      "damage-aware-config/unknown-nature": mutated(config, (c) => (c.assumed_set.nature = "grumpy")),
      "damage-aware-config/unknown-fixed-item": mutated(config, (c) => (c.assumed_set.item = { kind: "fixed", item: "notanitem" })),
      "damage-aware-config/stat-points-over-limit": mutated(config, (c) => (c.assumed_set.stat_points.hp = 255)),
      "damage-aware-config/other-format": mutated(config, (c) => (c.format_id = "gen9ou")),
    },
  };
}
