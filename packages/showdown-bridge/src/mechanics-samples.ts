/* eslint-disable @typescript-eslint/no-explicit-any */
// The mechanics contracts' share of the sample corpus (ADR-008 section 2):
// valid, invalid (schema) and reader-invalid samples of the MechanicsConfig,
// the MechanicsQuery and the MechanicsResult. The `computed` stats, damage and
// speed_order, `refused` and `unsupported` results are real probe outputs on
// synthetic fixture sets (the unsupported one from a config that lists stats
// alone); the damage result with KO counts is a shape sample, so its numbers
// are not engine outputs.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import type { Engine, PokemonSet } from "./engine.ts";
import { GOLDEN_FORMAT, TEAMS_DIR } from "./golden.ts";
import { loadMechanicsConfig, type MechanicsConfig } from "./mechanics-config.ts";
import { type MechanicsContext, probeMechanics } from "./mechanics-probe.ts";
import { mechanicsQueryHash, type MechanicsQuery, readMechanicsQuery } from "./mechanics-query.ts";
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

/** What the history and `move_outcome` samples are made from; none of it needs the engine, so the samples are rebuilt from the committed ones too. */
export interface HistoryParts {
  own: string;
  foe: string;
  config: MechanicsConfig;
  /** A computed result and an unsupported one, the identity of the shape samples. */
  computed: any;
  unsupported: any;
  /** The `damage` query of the corpus (wrapped, no history). */
  damage: any;
}

/**
 * The samples of the declared history and the `move_outcome` kind (ADR-022 erratum E-22.5.3). The two results are shape samples: their numbers are
 * the shape of an answer (the Protect chain of the third use), not a probe output.
 */
export function historySamples(parts: HistoryParts): { valid: Cases; invalid: Cases; readerInvalid: Cases } {
  const { own, foe, config, computed, unsupported, damage } = parts;
  const wrap = (query: unknown) => ({ schema_version: 1, data_class: "synthetic", format_id: GOLDEN_FORMAT, query });
  const boosts = { atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
  const combatant = (set: string, history?: unknown) => ({ set, boosts, status: null, hp: null, mega: false, ...(history === undefined ? {} : { history }) });
  const side = { aurora_veil: false, light_screen: false, reflect: false, tailwind: false };
  const field = { weather: null, terrain: null, trick_room: false, sides: [side, side] };
  const known = (value: unknown) => ({ kind: "known", value });
  const unknown = (first: unknown, second: unknown) => ({ kind: "unknown", ends: [first, second] });
  const battleHistory = { turn: 3, field_last_move: known("tackle"), sides: [{ total_fainted: known(1) }, { pending_slots: known([{ position: 0, id: "wish", started_turn: 2, after_residual: false }]) }] };
  const ownHistory = {
    move_actions: known(2),
    stall_counter: known(3),
    times_attacked: unknown(0, 4),
    hurt_this_turn: known(true),
    attacked_by: known([{ side: 1, position: 0, damage: 30 }]),
    last_move_failed: known(false),
    stats_raised_this_turn: known(false),
    stats_lowered_this_turn: known(true),
    ate_berry: known(true),
    last_item: known("sitrusberry"),
    moves_used: known(["protect"]),
    volatiles: known([{ id: "stockpile", started_turn: 2, after_residual: false, starts: 2 }]),
  };
  const actions = [
    { side: 0, position: 0, move: "protect", mega: false },
    { side: 1, position: 0, move: "tackle", mega: false },
  ];
  const outcome = (history?: unknown, own_history?: unknown) =>
    wrap({
      kind: "move_outcome",
      sides: [[combatant(own, own_history)], [combatant(foe)]],
      actions,
      asked: 0,
      target: { side: 0, position: 0 },
      state: field,
      ...(history === undefined ? {} : { history }),
    });
  const plain = outcome();
  const withHistory = outcome(battleHistory, ownHistory);
  const speedHistory = wrap({ kind: "speed_order", sides: [[combatant(own, { move_actions: known(1) })], [combatant(foe)]], actions, state: field, history: { sides: [{ total_fainted: known(2) }, {}] } });
  const damageHistory = mutated(damage, (q) => {
    q.query.attacker.history = { times_attacked: known(2), stats_lowered_this_turn: unknown(false, true) };
    q.query.history = { sides: [{ total_fainted: known(2) }, {}] };
  });
  const answer = { kind: "move_outcome", precondition: "chance", precondition_ppm: 333333, hit_ppm: 1000000 };
  const outcomeResult = { ...computed, query_hash: mechanicsQueryHash(plain as unknown as MechanicsQuery), gap_ids: [], answer };
  return {
    valid: {
      "mechanics-config/regmc-move-outcome": mutated(config, (c) => {
        c.scope.query_kinds.splice(1, 0, { kind: "move_outcome", gap_ids: [] });
      }),
      "mechanics-query/move-outcome": plain,
      "mechanics-query/move-outcome-history": withHistory,
      "mechanics-query/speed-order-history": speedHistory,
      "mechanics-query/damage-history": damageHistory,
      "mechanics-result/computed-move-outcome": outcomeResult,
      "mechanics-result/unsupported-history-unknown": { ...unsupported, query_hash: mechanicsQueryHash(withHistory as unknown as MechanicsQuery), reason: "history_unknown", position: null },
    },
    invalid: {
      "mechanics-query/move-outcome-no-asked": mutated(plain, (q) => delete q.query.asked),
      "mechanics-query/history-unknown-one-end": mutated(withHistory, (q) => (q.query.sides[0][0].history.times_attacked.ends = [0])),
      "mechanics-query/history-bad-fact-kind": mutated(withHistory, (q) => (q.query.sides[0][0].history.stall_counter.kind = "guessed")),
      "mechanics-query/history-negative-count": mutated(withHistory, (q) => (q.query.sides[0][0].history.move_actions.value = -1)),
      "mechanics-query/history-extra-field": mutated(withHistory, (q) => (q.query.history.rng = 1)),
      "mechanics-result/move-outcome-unknown-precondition": mutated(outcomeResult, (r) => (r.answer.precondition = "maybe")),
      "mechanics-result/move-outcome-ppm-over-a-million": mutated(outcomeResult, (r) => (r.answer.hit_ppm = 1000001)),
    },
    readerInvalid: {
      "mechanics-query/move-outcome-asked-a-switch": mutated(plain, (q) => (q.query.actions[0].move = null)),
      "mechanics-query/move-outcome-target-undeclared": mutated(plain, (q) => (q.query.target.position = 1)),
      "mechanics-query/history-volatile-without-turn": mutated(withHistory, (q) => delete q.query.history.turn),
      "mechanics-query/history-attacker-undeclared": mutated(withHistory, (q) => (q.query.sides[0][0].history.attacked_by.value[0].position = 1)),
    },
  };
}

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
  const statsOnly: MechanicsConfig = { ...config, scope: { query_kinds: config.scope.query_kinds.filter((entry) => entry.kind === "stats") } };
  const unsupported = probeMechanics({ engine, config: statsOnly }, readMechanicsQuery(speedOrder)).result;
  const speedResult = run(speedOrder);
  const damageComputed = run(damage) as any;
  const rolls = Array.from({ length: 16 }, (_, i) => 80 + i * 2);
  const damageResult = (ko: unknown) => ({ ...damageComputed, answer: { ...damageComputed.answer, ko } });
  const counts = { kind: "counts", ohko_no_crit: 0, ohko_crit: 0, thko_no_crit: 256 };
  const histories = historySamples({ own, foe, config, computed, unsupported, damage });

  return {
    valid: {
      "mechanics-config/regmc": config,
      // a config written before ko_non_blockers existed (the field is optional)
      "mechanics-config/regmc-no-non-blockers": mutated(config, (c) => delete c.ko_non_blockers),
      "mechanics-query/stats": stats,
      "mechanics-query/speed-order": speedOrder,
      "mechanics-query/damage": damage,
      "mechanics-result/computed-stats": computed,
      "mechanics-result/unsupported-query-kind": unsupported,
      "mechanics-result/refused-invalid-set": run(refusedQuery),
      "mechanics-result/computed-speed-order": speedResult,
      "mechanics-result/computed-damage": damageComputed,
      "mechanics-result/computed-damage-ko-counts": damageResult(counts),
      ...histories.valid,
    },
    invalid: {
      "mechanics-config/extra-field": mutated(config, (c) => (c.scenario.note = "x")),
      "mechanics-config/unknown-kind": mutated(config, (c) => (c.scope.query_kinds[0].kind = "heal")),
      "mechanics-config/non-blocker-without-reason": mutated(config, (c) => delete c.ko_non_blockers.items[0].reason),
      "mechanics-query/extra-field": mutated(stats, (q) => (q.battle = {})),
      "mechanics-query/unknown-kind": mutated(stats, (q) => (q.query.kind = "heal")),
      "mechanics-query/boost-out-of-range": mutated(damage, (q) => (q.query.attacker.boosts.atk = 7)),
      "mechanics-query/three-on-a-side": mutated(speedOrder, (q) => (q.query.sides[0] = [combatant(own), combatant(own), combatant(own)])),
      "mechanics-result/unknown-reason": mutated(unsupported, (r) => (r.reason = "because")),
      "mechanics-result/refusal-with-text": mutated(run(refusedQuery), (r) => (r.message = "Venusaur has more than 32 Stat Points in HP.")),
      "mechanics-result/fractional-stat": mutated(computed, (r) => (r.answer.stats.hp = 187.5)),
      "mechanics-result/short-roll-vector": mutated(damageResult(counts), (r) => (r.answer.no_crit = rolls.slice(1))),
      ...histories.invalid,
    },
    readerInvalid: {
      "mechanics-config/unsorted-removed-rule-ids": mutated(config, (c) => c.scenario.removed_rule_keys.reverse()),
      "mechanics-config/unsorted-ko-non-blockers": mutated(config, (c) => c.ko_non_blockers.abilities.reverse()),
      "mechanics-result/unsorted-gap-ids": mutated(computed, (r) => (r.gap_ids = ["stat-points", "level"])),
      ...histories.readerInvalid,
    },
  };
}
