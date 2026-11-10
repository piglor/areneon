/* eslint-disable @typescript-eslint/no-explicit-any */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { jcs, type Json } from "../src/canonical-json.ts";
import type { Engine, PokemonSet } from "../src/engine.ts";
import { GOLDEN_FORMAT, TEAMS_DIR } from "../src/golden.ts";
import { loadEngine } from "../src/index.ts";
import { loadMechanicsConfig, type MechanicsConfig } from "../src/mechanics-config.ts";
import { probeMechanics } from "../src/mechanics-probe.ts";
import { readMechanicsQuery } from "../src/mechanics-query.ts";
import { exportPaste, importPaste } from "../src/teams.ts";

// ADR-022 erratum E-22.5.3, slice H2: the declared history in `damage` and `speed_order` queries. A move whose power reads a history fact
// (Last Respects, Rage Fist, Assurance, Stomping Tantrum, Lash Out) is scaled by the declared fact: the engine's own `basePowerCallback`
// or `onBasePower` reads the engine's own field, and the probe names no move. A query without history keeps its answer, on both damage paths.

let engine: Engine;
let config: MechanicsConfig;
beforeAll(() => {
  engine = loadEngine();
  config = loadMechanicsConfig(GOLDEN_FORMAT);
});

const setOf = (file: string, index: number): string => exportPaste(engine, [importPaste(engine, readFileSync(join(TEAMS_DIR, file), "utf8"))[index] as PokemonSet]);
// Snorlax (a strong physical attacker) against Tyranitar and Garchomp
const SNORLAX = () => setOf("legal-synthetic.txt", 4);
const TYRANITAR = () => setOf("legal-opponent.txt", 0);
const GARCHOMP = () => setOf("legal-opponent.txt", 1);

const ZERO = { atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
const NO_CONDITIONS = { aurora_veil: false, light_screen: false, reflect: false, tailwind: false };
const FIELD = { weather: null, terrain: null, trick_room: false, sides: [NO_CONDITIONS, NO_CONDITIONS] };
const known = (value: unknown) => ({ kind: "known", value });
const combatant = (set: string, history?: unknown) => ({ set, boosts: ZERO, status: null, hp: null, mega: false, ...(history === undefined ? {} : { history }) });

const damageQuery = (move: string, attacker?: unknown, defender?: unknown, history?: unknown, foe = TYRANITAR()) =>
  readMechanicsQuery({ schema_version: 1, data_class: "synthetic", format_id: GOLDEN_FORMAT, query: { kind: "damage", attacker: combatant(SNORLAX(), attacker), defender: combatant(foe, defender), move, state: FIELD, ...(history === undefined ? {} : { history }) } });
const vector = (query: ReturnType<typeof damageQuery>, path: "one-clone" | "per-roll" = "one-clone"): number[] => {
  const { result } = probeMechanics({ engine, config, damage_path: path }, query);
  if (result.kind !== "computed" || result.answer.kind !== "damage") throw new Error(`not computed: ${JSON.stringify(result)}`);
  return result.answer.no_crit;
};
const top = (rolls: number[]): number => rolls[15] as number;

describe("a damage that reads the history is scaled by the declared fact", () => {
  it("Last Respects: power 50 + 50 per fainted Pokemon of the attacker's side", () => {
    const none = top(vector(damageQuery("lastrespects")));
    const two = top(vector(damageQuery("lastrespects", undefined, undefined, { sides: [{ total_fainted: known(2) }, {}] })));
    expect(none).toBeGreaterThan(0);
    // 50 -> 150: three times the power, so more than twice the damage whatever the rounding
    expect(two).toBeGreaterThanOrEqual(2 * none);
  });

  it("Rage Fist: power 50 + 50 per hit the attacker took", () => {
    const none = top(vector(damageQuery("ragefist")));
    const hit = top(vector(damageQuery("ragefist", { times_attacked: known(3) })));
    expect(hit).toBeGreaterThanOrEqual(2 * none);
  });

  it("Assurance: doubled against a defender already hurt this turn", () => {
    const fresh = top(vector(damageQuery("assurance", undefined, { hurt_this_turn: known(false) })));
    const hurt = top(vector(damageQuery("assurance", undefined, { hurt_this_turn: known(true) })));
    expect(hurt).toBeGreaterThanOrEqual(Math.floor(1.5 * fresh));
    expect(hurt).toBeGreaterThan(fresh);
  });

  it("Stomping Tantrum: doubled after a failed move", () => {
    const worked = top(vector(damageQuery("stompingtantrum", { last_move_failed: known(false) })));
    const failed = top(vector(damageQuery("stompingtantrum", { last_move_failed: known(true) })));
    expect(failed).toBeGreaterThanOrEqual(Math.floor(1.5 * worked));
    expect(failed).toBeGreaterThan(worked);
  });

  it("Lash Out: doubled after the attacker's stats were lowered this turn", () => {
    const plain = top(vector(damageQuery("lashout", undefined, undefined, undefined, GARCHOMP())));
    const lowered = top(vector(damageQuery("lashout", { stats_lowered_this_turn: known(true) }, undefined, undefined, GARCHOMP())));
    expect(lowered).toBeGreaterThanOrEqual(Math.floor(1.5 * plain));
    expect(lowered).toBeGreaterThan(plain);
  });
});

describe("the history does not change what it does not name", () => {
  it("a history that declares nothing, or a fact the move does not read, gives the vectors of a query without history", () => {
    const plain = vector(damageQuery("lastrespects"));
    expect(vector(damageQuery("lastrespects", undefined, undefined, {}))).toEqual(plain);
    expect(vector(damageQuery("lastrespects", { move_actions: known(5), ate_berry: known(true) }))).toEqual(plain);
  });

  it("gives the same bytes on the one-clone path and the per-roll path, with and without history", () => {
    for (const query of [damageQuery("ragefist", { times_attacked: known(2) }, { hurt_this_turn: known(true) }), damageQuery("lastrespects", undefined, undefined, { sides: [{ total_fainted: known(1) }, {}] }), damageQuery("assurance")]) {
      const one = probeMechanics({ engine, config, damage_path: "one-clone" }, query).result;
      const per = probeMechanics({ engine, config, damage_path: "per-roll" }, query).result;
      expect(jcs(one as unknown as Json)).toBe(jcs(per as unknown as Json));
    }
  });

  it("asks a draw the declaration makes as unsupported: a volatile with a random length", () => {
    // Confusion lasts a random number of turns, drawn when the engine starts it
    const { result } = probeMechanics({ engine, config }, damageQuery("tackle", { volatiles: known([{ id: "confusion", started_turn: 1, after_residual: false }]) }, undefined, { turn: 2 }));
    expect(result).toMatchObject({ kind: "unsupported", reason: "random_effect", position: null });
  });
});

describe("a speed order declares the history too and keeps its answer", () => {
  const speed = (history?: unknown) =>
    readMechanicsQuery({
      schema_version: 1,
      data_class: "synthetic",
      format_id: GOLDEN_FORMAT,
      query: { kind: "speed_order", sides: [[combatant(SNORLAX(), history)], [combatant(TYRANITAR())]], actions: [{ side: 0, position: 0, move: "protect", mega: false }, { side: 1, position: 0, move: "tackle", mega: false }], state: FIELD },
    });

  it("is the same order with a fact declared, and answers an unknown fact's two ends alike", () => {
    const plain = probeMechanics({ engine, config }, speed()).result;
    for (const history of [{ move_actions: known(3), times_attacked: known(1) }, { times_attacked: { kind: "unknown", ends: [0, 4] } }]) {
      const result = probeMechanics({ engine, config }, speed(history)).result;
      expect(result.kind).toBe("computed");
      expect((result as any).answer).toEqual((plain as any).answer);
    }
  });
});
