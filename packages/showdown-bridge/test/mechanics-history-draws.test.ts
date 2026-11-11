/* eslint-disable @typescript-eslint/no-explicit-any */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import type { Engine, PokemonSet } from "../src/engine.ts";
import { GOLDEN_FORMAT, TEAMS_DIR } from "../src/golden.ts";
import { loadEngine } from "../src/index.ts";
import { loadMechanicsConfig, type MechanicsConfig, withMoveOutcome } from "../src/mechanics-config.ts";
import { withStartDraws } from "../src/mechanics-history.ts";
import { type Leaf, combine, probeMoveOutcome } from "../src/mechanics-outcome.ts";
import { checkScenario } from "../src/mechanics-scenario.ts";
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
    // a `randomChance` inside a start handler is answered true at BOTH ends (the documented limit: only one end of a chance is asked)
    expect([run("low", (prng) => [prng.randomChance(33, 100)]).seen, run("high", (prng) => [prng.randomChance(33, 100)]).seen]).toEqual([[true], [true]]);
    expect(run("low", body)).toEqual({ draws: 5, seen: [2, 0, 0, true, "a", 1] }); // random() is a float: 0 at the low end
    expect(run("high", body)).toEqual({ draws: 5, seen: [5, 4, 1 - Number.EPSILON / 2, true, "c", 1] }); // and just below 1 at the high end
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

describe("the before-move event", () => {
  it("is the engine's: a paralysed attacker's move works with 7 in 8 (the Champions mod's randomChance(1, 8)), a blocked path counts against success", () => {
    const parsed = outcomeQuery(undefined) as unknown as { query: { sides: { status: string | null }[][] } };
    const paralysed = structuredClone(parsed);
    ((paralysed.query.sides[0] as { status: string | null }[])[0] as { status: string | null }).status = "par";
    const result = probeMechanics({ engine, config }, readMechanicsQuery(JSON.parse(JSON.stringify(paralysed)))).result;
    expect(result).toMatchObject({ kind: "computed", answer: { precondition: "chance", precondition_ppm: 875_000, hit_ppm: 1_000_000 } });
  });
  it("combines a confused and paralysed attacker's two chances as the engine draws them (the paths of both draws)", () => {
    const parsed = outcomeQuery(confusion) as unknown as { query: { sides: { status: string | null }[][] } };
    const both = structuredClone(parsed);
    ((both.query.sides[0] as { status: string | null }[])[0] as { status: string | null }).status = "par";
    const result = probeMechanics({ engine, config }, readMechanicsQuery(JSON.parse(JSON.stringify(both)))).result;
    // confusion (priority 3) runs first: 67/100 pass, then paralysis 7/8: 0.67 * 0.875 = 0.58625
    expect(result).toMatchObject({ kind: "computed", answer: { precondition: "chance", precondition_ppm: 586_250 } });
  });
});

describe("the explored paths of the before-move event", () => {
  const leaf = (answers: [number, number, boolean][], pass: boolean, tail: Extract<Leaf, { kind: "leaf" }>["tail"]): Leaf => ({ kind: "leaf", chances: answers.map(([numerator, denominator, answer]) => ({ numerator, denominator, answer })), pass, tail });
  const reached = (pre: [number, number][], hit: number) => ({ fails: false as const, pre: pre.map(([numerator, denominator]) => ({ kind: "randomChance" as const, numerator, denominator })), hit });
  it("sums the passing paths exactly, multiplies the chances the move step drew and floors to a ppm (never to 0)", () => {
    // one draw, true blocks (33 in 100): the false path passes with 67 in 100; the move step then draws 1 in 3
    const paths = [leaf([[33, 100, true]], false, null), leaf([[33, 100, false]], true, reached([[1, 3]], 900_000))];
    expect(combine(paths, 0)).toEqual({ kind: "computed", answer: { kind: "move_outcome", precondition: "chance", precondition_ppm: 223_333, hit_ppm: 900_000 } });
    // every path passes: the sum is 1
    expect(combine([leaf([[1, 4, true]], true, reached([], 1_000_000)), leaf([[1, 4, false]], true, reached([], 1_000_000))], 0)).toMatchObject({ answer: { precondition: "passes", precondition_ppm: 1_000_000 } });
    // a chance below one part per million stays 1 ppm
    expect(combine([leaf([], true, reached([[1, 5_000_000]], 1_000_000))], 0)).toMatchObject({ answer: { precondition: "chance", precondition_ppm: 1 } });
  });
  it("leaves a path of probability 0 out of the sum and of the equal-tails check", () => {
    // the true answer of a chance of 0 in 100 can never happen: its tail (a different hit chance) must not make the answer unsupported
    const never = leaf([[0, 100, true]], true, reached([], 500_000));
    const always = leaf([[0, 100, false]], true, reached([], 1_000_000));
    expect(combine([never, always], 0)).toMatchObject({ kind: "computed", answer: { precondition: "passes", precondition_ppm: 1_000_000, hit_ppm: 1_000_000 } });
    expect(combine([never], 0)).toMatchObject({ answer: { precondition: "fails" } });
  });
  it("fails when no path passes or the move step never reaches its hit step, and is unsupported when the passing paths read different tails", () => {
    expect(combine([leaf([[1, 2, true]], false, null)], 0)).toMatchObject({ answer: { precondition: "fails", precondition_ppm: 0, hit_ppm: 0 } });
    expect(combine([leaf([], true, { fails: true })], 0)).toMatchObject({ answer: { precondition: "fails" } });
    expect(combine([leaf([[1, 2, true]], true, reached([], 1_000_000)), leaf([[1, 2, false]], true, reached([], 500_000))], 3)).toEqual({ kind: "unsupported", reason: "random_effect", position: 3 });
  });
  it("is fails for a confused attacker's Protect when no action is left after it, and unsupported for more paths than the limit or a sleeping attacker", () => {
    const parse = (query: ReturnType<typeof outcomeQuery>) => ({ q: query.query as never, sets: [importPaste(engine, setOf("legal-synthetic.txt", 0))[0], importPaste(engine, setOf("legal-opponent.txt", 0))[0]] as PokemonSet[] });
    const scenario = () => checkScenario(engine, config);
    const query = JSON.parse(JSON.stringify(outcomeQuery(confusion, "protect"))) as { query: { actions: { side: number; position: number; move: string | null; mega: boolean }[]; sides: { status: string | null; history?: unknown }[][] } };
    // every foe switches (the helper's branch): no action is left after the confused attacker's Protect, so it fails whichever way the self-hit draw went
    expect(probeMechanics({ engine, config }, readMechanicsQuery(query as never)).result).toMatchObject({ kind: "computed", answer: { precondition: "fails", precondition_ppm: 0 } });
    // the foe tackles after it: Protect works unless the confusion self-hit blocks it (67 in 100)
    (query.query.actions[1] as { move: string | null }).move = "tackle";
    expect(probeMechanics({ engine, config }, readMechanicsQuery(query as never)).result).toMatchObject({ kind: "computed", answer: { precondition: "chance", precondition_ppm: 670_000 } });
    // a paralysed attacker has two paths (blocked, not): over a limit of one path it is unsupported
    ((query.query.sides[0] as { status: string | null }[])[0] as { status: string | null }).status = "par";
    const { q, sets } = parse(readMechanicsQuery(query as never));
    expect(probeMoveOutcome(engine, scenario(), q, sets, null, 1)).toMatchObject({ kind: "unsupported", reason: "random_effect" });
    // sleep draws its length when declared: the engine's draw cannot be answered
    ((query.query.sides[0] as { status: string | null }[])[0] as { status: string | null }).status = "slp";
    expect(probeMechanics({ engine, config }, readMechanicsQuery(query as never)).result).toMatchObject({ kind: "unsupported", reason: "random_effect" });
  });
});

describe("the steps of runMove the probe follows", () => {
  it("runs the move's own beforeMoveCallback: Bide is blocked while the bide volatile is up (data/moves.ts), and counts the action first", () => {
    expect(ask({ last_move: known(null) }, "bide")).toMatchObject({ kind: "computed", answer: { precondition: "passes" } });
    const bide = { volatiles: known([{ id: "bide", started_turn: 3, after_residual: false }]), last_move: known(null) };
    expect(ask(bide, "bide")).toMatchObject({ kind: "computed", answer: { precondition: "fails", precondition_ppm: 0, hit_ppm: 0 } });
  });
});

describe("the declared history against the engine", () => {
  it("declares the last move before the volatiles: Encore starts with its last move and is refused without one", () => {
    expect(ask({ ...encore, last_move: known(ownMove()) }).kind).toBe("computed");
    expect(() => ask(encore)).toThrow(MechanicsProbeError);
    expect(() => ask({ ...encore, last_move: known(null) })).toThrow(/declared volatile/);
    expect(() => ask({ ...encore, last_move: known("nosuchmove") })).toThrow(/last move/);
  });
  it("answers a confused attacker's move with the engine's own before-move chance (hand-worked: 67 in 100) and a volatile's length draw at both ends", () => {
    const result = ask(confusion);
    expect(result.kind).toBe("computed");
    // Hand-worked from the engine's confusion condition (data/conditions.ts): `onBeforeMove` counts `time` down (a length of 2 to 5 leaves it above 0, at both ends of the
    // draw) and then `randomChance(33, 100)` is the self-hit that blocks the move, so the move works with the other 67 of 100: 670,000 ppm, and Tackle's accuracy 100 hits.
    expect(result).toMatchObject({ answer: { kind: "move_outcome", precondition: "chance", precondition_ppm: 670_000, hit_ppm: 1_000_000 } });
    // an attacker that is not confused is not blocked
    expect(ask({ last_move: known(null) })).toMatchObject({ answer: { precondition: "passes", precondition_ppm: 1_000_000, hit_ppm: 1_000_000 } });
  });
});
