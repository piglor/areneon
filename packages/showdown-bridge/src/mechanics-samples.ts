/* eslint-disable @typescript-eslint/no-explicit-any */
// The mechanics contracts' share of the sample corpus (ADR-008 section 2):
// valid, invalid (schema) and reader-invalid samples of the MechanicsConfig,
// the MechanicsQuery and the MechanicsResult. The `computed` stats, `refused`
// and `unsupported` results are real probe outputs on synthetic fixture sets;
// the damage query and result are a hand-worked fixture case and the probe's
// output for it; the speed_order result and the damage result with KO counts
// are shape samples (their probes come in later slices), so their numbers are
// not engine outputs.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import type { Engine, PokemonSet } from "./engine.ts";
import { GOLDEN_FORMAT, TEAMS_DIR } from "./golden.ts";
import { loadMechanicsConfig } from "./mechanics-config.ts";
import { type MechanicsContext, probeMechanics } from "./mechanics-probe.ts";
import { readMechanicsQuery } from "./mechanics-query.ts";
import type { MechanicsResult } from "./mechanics-result.ts";
import { exportPaste, importPaste } from "./teams.ts";

const DAMAGE_CASES = fileURLToPath(new URL("../../../fixtures/mechanics/gen9championsvgc2026regmc/damage/cases.json", import.meta.url));

type Cases = Record<string, unknown>;

/** A copy of `value` with `change` applied: one mutation per sample. */
function mutated<T>(value: T, change: (copy: Record<string, any>) => void): Record<string, unknown> {
  const copy = structuredClone(value) as unknown as Record<string, unknown>;
  change(copy);
  return copy;
}

/** One Pokemon of a committed synthetic team, as the engine's canonical PokePaste. */
const setOf = (engine: Engine, file: string, index: number): string => {
  const team = importPaste(engine, readFileSync(join(TEAMS_DIR, file), "utf8"));
  return exportPaste(engine, [team[index] as PokemonSet]);
};

export function mechanicsSamples(engine: Engine): { valid: Cases; invalid: Cases; readerInvalid: Cases } {
  const config = loadMechanicsConfig(GOLDEN_FORMAT);
  const context: MechanicsContext = { engine, config };
  const own = setOf(engine, "legal-synthetic.txt", 0);
  const foe = setOf(engine, "legal-opponent.txt", 0);
  const wrap = (query: unknown) => ({ schema_version: 1, data_class: "synthetic", format_id: GOLDEN_FORMAT, query });
  const stats = wrap({ kind: "stats", set: own });
  const boosts = { atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
  const combatant = (set: string) => ({ set, boosts, status: null, hp: null, mega: false });
  const side = { aurora_veil: false, light_screen: false, reflect: false, tailwind: false };
  const field = { weather: null, terrain: null, trick_room: false, sides: [side, side] };
  const speedOrder = wrap({
    kind: "speed_order",
    sides: [[combatant(own)], [combatant(foe)]],
    actions: [
      { side: 0, position: 0, move: "protect", mega: false },
      { side: 1, position: 0, move: "protect", mega: false },
    ],
    state: field,
  });
  const damageCases = JSON.parse(readFileSync(DAMAGE_CASES, "utf8")) as { cases: { id: string; query: unknown }[] };
  const damage = (damageCases.cases.find((c) => c.id === "single-target") as { query: any }).query;

  const run = (query: unknown): MechanicsResult => probeMechanics(context, readMechanicsQuery(query)).result;
  const computed = run(stats);
  const refusedQuery = wrap({ kind: "stats", set: setOf(engine, "illegal-stat-points-over-32.txt", 0) });
  const unsupported = run(speedOrder);
  const identity = { schema_version: 1, data_class: "synthetic", query_hash: computed.query_hash, manifest: computed.manifest, mechanics_config_hash: computed.mechanics_config_hash, adapter_format: computed.adapter_format, gap_ids: computed.gap_ids };
  const damageComputed = run(damage) as any;
  const rolls = Array.from({ length: 16 }, (_, i) => 80 + i * 2);
  const damageResult = (ko: unknown) => ({ ...damageComputed, answer: { ...damageComputed.answer, ko } });
  const counts = { kind: "counts", ohko_no_crit: 0, ohko_crit: 0, thko_no_crit: 256 };
  const speedResult = { ...identity, kind: "computed", answer: { kind: "speed_order", order_basis: "turn_start", actions: [{ side: 0, position: 0, mega: false, order: 200, priority: 4, speed: 100 }, { side: 1, position: 0, mega: false, order: 200, priority: 4, speed: 100 }], groups: [[0, 1]] } };

  return {
    valid: {
      "mechanics-config/regmc": config,
      "mechanics-query/stats": stats,
      "mechanics-query/speed-order": speedOrder,
      "mechanics-query/damage": damage,
      "mechanics-result/computed-stats": computed,
      "mechanics-result/unsupported-query-kind": unsupported,
      "mechanics-result/refused-invalid-set": run(refusedQuery),
      "mechanics-result/computed-speed-order": speedResult,
      "mechanics-result/computed-damage": damageComputed,
      "mechanics-result/computed-damage-ko-counts": damageResult(counts),
    },
    invalid: {
      "mechanics-config/extra-field": mutated(config, (c) => (c.scenario.note = "x")),
      "mechanics-config/unknown-kind": mutated(config, (c) => (c.scope.query_kinds[0].kind = "heal")),
      "mechanics-query/extra-field": mutated(stats, (q) => (q.battle = {})),
      "mechanics-query/unknown-kind": mutated(stats, (q) => (q.query.kind = "heal")),
      "mechanics-query/boost-out-of-range": mutated(damage, (q) => (q.query.attacker.boosts.atk = 7)),
      "mechanics-query/three-on-a-side": mutated(speedOrder, (q) => (q.query.sides[0] = [combatant(own), combatant(own), combatant(own)])),
      "mechanics-result/unknown-reason": mutated(unsupported, (r) => (r.reason = "because")),
      "mechanics-result/refusal-with-text": mutated(run(refusedQuery), (r) => (r.message = "Venusaur has more than 32 Stat Points in HP.")),
      "mechanics-result/fractional-stat": mutated(computed, (r) => (r.answer.stats.hp = 187.5)),
      "mechanics-result/short-roll-vector": mutated(damageResult(counts), (r) => (r.answer.no_crit = rolls.slice(1))),
    },
    readerInvalid: {
      "mechanics-config/unsorted-removed-rule-ids": mutated(config, (c) => c.scenario.removed_rule_keys.reverse()),
      "mechanics-result/unsorted-gap-ids": mutated(computed, (r) => (r.gap_ids = ["stat-points", "level"])),
    },
  };
}
