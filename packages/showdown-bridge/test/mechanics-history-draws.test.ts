/* eslint-disable @typescript-eslint/no-explicit-any */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import type { Engine, PokemonSet } from "../src/engine.ts";
import { GOLDEN_FORMAT, TEAMS_DIR } from "../src/golden.ts";
import { loadEngine } from "../src/index.ts";
import { loadMechanicsConfig, type MechanicsConfig, withMoveOutcome } from "../src/mechanics-config.ts";
import { withStartDraws } from "../src/mechanics-history.ts";
import { MechanicsProbeError, probeMechanics } from "../src/mechanics-probe.ts";
import { readMechanicsQuery } from "../src/mechanics-query.ts";
import { exportPaste, importPaste } from "../src/teams.ts";

// ADR-022 E-22.5.3: the order a Pokemon's history is declared in (its last move before its volatiles: Encore's start handler reads it)
// and the draws of a volatile's start handler (a confusion's length), answered at both ends.

describe("withStartDraws", () => {
  const battle = (): { prng: any } => ({ prng: { random: () => 99, randomChance: () => false, sample: () => "x", shuffle: () => "refused", extra: 1 } });
  const run = (end: "low" | "high", body: (prng: any) => unknown[]) => {
    const b = battle();
    const original = b.prng;
    let draws = 0;
    let seen: unknown[] = [];
    withStartDraws(b as never, end, () => (draws += 1), () => (seen = body(b.prng)));
    expect(b.prng).toBe(original); // restored
    return { draws, seen };
  };
  it("answers each draw at its smallest value (low) or largest (high), and says it drew", () => {
    const body = (prng: any) => [prng.random(2, 6), prng.random(5), prng.random(), prng.randomChance(1, 2), prng.sample(["a", "b", "c"]), prng.extra];
    expect(run("low", body)).toEqual({ draws: 5, seen: [2, 0, 0, true, "a", 1] });
    expect(run("high", body)).toEqual({ draws: 5, seen: [5, 4, 0, true, "c", 1] });
  });
  it("restores the source when the handler throws, and keeps every other method", () => {
    const b = battle();
    const original = b.prng;
    expect(() => withStartDraws(b as never, "low", () => undefined, () => { throw new Error("x"); })).toThrow("x");
    expect(b.prng).toBe(original);
    expect(run("low", (prng) => [prng.shuffle()]).seen).toEqual(["refused"]);
  });
});

let engine: Engine;
let config: MechanicsConfig;
beforeAll(() => {
  engine = loadEngine();
  config = withMoveOutcome(loadMechanicsConfig(GOLDEN_FORMAT));
});
const setOf = (file: string, index: number): string => exportPaste(engine, [importPaste(engine, readFileSync(join(TEAMS_DIR, file), "utf8"))[index] as PokemonSet]);
const ZERO = { atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
const NO = { aurora_veil: false, light_screen: false, reflect: false, tailwind: false };
const known = (value: unknown) => ({ kind: "known", value });
const combatant = (set: string, history?: unknown) => ({ set, boosts: ZERO, status: null, hp: null, mega: false, ...(history === undefined ? {} : { history }) });
const outcomeQuery = (ownHistory: unknown, move = "tackle") =>
  readMechanicsQuery({
    schema_version: 1,
    data_class: "synthetic",
    format_id: GOLDEN_FORMAT,
    query: {
      kind: "move_outcome",
      sides: [[combatant(setOf("legal-synthetic.txt", 0), ownHistory)], [combatant(setOf("legal-opponent.txt", 0))]],
      actions: [{ side: 0, position: 0, move, mega: false }, { side: 1, position: 0, move: null, mega: false }],
      asked: 0,
      target: { side: 1, position: 0 },
      state: { weather: null, terrain: null, trick_room: false, sides: [NO, NO] },
      history: { turn: 4, sides: [{ total_fainted: known(0) }, { total_fainted: known(0) }] },
    },
  } as never);
const ask = (ownHistory: unknown, move?: string) => probeMechanics({ engine, config }, outcomeQuery(ownHistory, move)).result;
// The move the own Pokemon holds first (Encore needs its last move among the Pokemon's moves).
const ownMove = (): string => /^- (.+?)\s*$/m.exec(setOf("legal-synthetic.txt", 0))?.[1]?.toLowerCase().replace(/[^a-z0-9]/g, "") as string;
const encore = { volatiles: known([{ id: "encore", started_turn: 3, after_residual: false }]) };
const confusion = { volatiles: known([{ id: "confusion", started_turn: 3, after_residual: false }]) };

describe("the declared history against the engine", () => {
  it("declares the last move before the volatiles: Encore starts with its last move and is refused without one", () => {
    expect(ask({ ...encore, last_move: known(ownMove()) }).kind).toBe("computed");
    expect(() => ask(encore)).toThrow(MechanicsProbeError);
    expect(() => ask({ ...encore, last_move: known(null) })).toThrow(/declared volatile/);
    expect(() => ask({ ...encore, last_move: known("nosuchmove") })).toThrow(/last move/);
  });
  it("answers a volatile that draws (a confusion's length) at both ends of its draws, equal answers kept", () => {
    const result = ask(confusion);
    expect(result.kind).toBe("computed");
    expect(result).toMatchObject({ answer: { kind: "move_outcome", precondition: "passes" } });
  });
});
