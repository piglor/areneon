import { beforeAll, describe, expect, it } from "vitest";
import { readerSha256 } from "@areneon/public-state";
import { jcs, type Json } from "../src/canonical-json.ts";
import { DamageAwareError, type DamageAwareActor, type DamageAwareOptions, damageAwareActor } from "../src/damage-aware-actor.ts";
import { loadDamageAwareConfig } from "../src/damage-aware-config.ts";
import { parseJointChoice } from "../src/damage-aware-choice.ts";
import { damageAwareIdentity, damageAwareIdentityHash, STALL_FAILURE_FORMAT, STALL_TIE_FORMAT, SUCCESS_SCORE_FORMAT } from "../src/damage-aware-identity.ts";
import { avoidFailingStalls } from "../src/damage-aware-success.ts";
import { resolveFormat } from "../src/formats.ts";
import { firstLegal, GOLDEN_FORMAT } from "../src/golden.ts";
import { withMoveOutcome } from "../src/mechanics-config.ts";
import { type MechanicsContext, type ProbeOutcome } from "../src/mechanics-probe.ts";
import { type ParsedMechanicsQuery } from "../src/mechanics-query.ts";
import { loadOutcomeConfig } from "../src/outcome-config.ts";
import { runBattle } from "../src/runner.ts";
import { goldenWorldFactsData } from "../src/world-facts-data.ts";
import { type AssumedSet } from "../src/damage-aware-assumed-set.ts";
import type { PublicMon } from "@areneon/public-state";
import type { PokemonSet } from "../src/engine.ts";
import { exportPaste } from "../src/teams.ts";
import { capture, type Captured, corpus, engine, mechanics, snapshot, stateOf, stubProbe, toId, vocabulary } from "./damage-aware-support.ts";
import { comparable, mutated, withGenders } from "./damage-aware-invariance-support.ts";

// #1392: damage-aware@v7 is v6 plus one rule (ADR-022 erratum E-22.8): in a tie for the best score, and in the all-zero fallback, a candidate with a
// stalling part whose computed success chance is below certain is never chosen while another tied, non-pending candidate without one exists.



const v6 = loadDamageAwareConfig("v6", vocabulary);
const v7 = loadDamageAwareConfig("v7", vocabulary);
const facts_data = goldenWorldFactsData();
const stall = new Set(facts_data.stall_moves);
const mechanics5 = withMoveOutcome(mechanics);
const outcome = { config: loadOutcomeConfig(GOLDEN_FORMAT) };

describe("the tie rule on its own (E-22.8.1)", () => {
  const item = (id: string, failing: boolean, pending = false) => ({ id, failing, pending });
  const read = (x: ReturnType<typeof item>) => x;
  const ids = (xs: readonly ReturnType<typeof item>[]) => xs.map((x) => x.id);

  it("a failing stall is left out of a tie while a tied candidate that is neither failing nor pending remains; the rest keeps its order", () => {
    expect(ids(avoidFailingStalls([item("stall", true), item("a", false), item("b", false)], read))).toEqual(["a", "b"]);
    expect(ids(avoidFailingStalls([item("a", false), item("stall", true)], read))).toEqual(["a"]);
  });

  it("a failing stall that scores best alone (a tie of one) is played", () => {
    expect(ids(avoidFailingStalls([item("stall", true)], read))).toEqual(["stall"]);
  });

  it("every tied candidate failing: the tie is returned whole (nothing else scores as high)", () => {
    expect(ids(avoidFailingStalls([item("s1", true), item("s2", true)], read))).toEqual(["s1", "s2"]);
  });

  it("only a pending candidate beside the failing stall is no alternative", () => {
    expect(ids(avoidFailingStalls([item("stall", true), item("p", false, true)], read))).toEqual(["stall", "p"]);
    expect(ids(avoidFailingStalls([item("stall", true), item("p", false, true), item("ok", false)], read))).toEqual(["p", "ok"]);
  });

  it("no failing stall in the tie: the tie is unchanged (v7 equals v6)", () => {
    const tied = [item("a", false), item("b", false, true)];
    expect(avoidFailingStalls(tied, read)).toBe(tied);
  });
});

const options = (config = v7, change: Partial<DamageAwareOptions> = {}): DamageAwareOptions => ({
  engine,
  vocabulary,
  config,
  mechanics: mechanics5,
  snapshot,
  data_class: "synthetic",
  facts_data,
  outcome,
  ...change,
});

describe("construction and identity", () => {
  it("v7 is the policy id of a record with a stall tie break; another version's record or id is refused both ways", () => {
    expect(damageAwareActor(options()).report().policy_id).toBe("damage-aware@v7");
    expect(() => damageAwareActor(options(v7, { commit: "v6" }))).toThrow(DamageAwareError);
    expect(() => damageAwareActor(options(v6, { commit: "v7" }))).toThrow(DamageAwareError);
    expect(damageAwareActor(options(v6)).report().policy_id).toBe("damage-aware@v6");
  });

  it("a record with the tie break but no stall failure cost is refused by the actor", () => {
    const { stall_failure_cost: _c, ...bare } = v7;
    expect(() => damageAwareActor(options(bare as typeof v7))).toThrow("stall failure cost");
  });

  it("the identity of v7 names the stall tie format; v1 to v6 do not (their identities are unchanged)", () => {
    const manifest = resolveFormat(engine, GOLDEN_FORMAT, snapshot).manifest;
    const id7 = damageAwareIdentity({ config: v7, mechanics: mechanics5, manifest, reader_sha256: readerSha256(), outcome_config: outcome.config });
    const id6 = damageAwareIdentity({ config: v6, mechanics: mechanics5, manifest, reader_sha256: readerSha256(), outcome_config: outcome.config });
    expect(id7.stall_tie_format).toBe(STALL_TIE_FORMAT);
    expect(id7.stall_failure_format).toBe(STALL_FAILURE_FORMAT);
    expect(id7.success_format).toBe(SUCCESS_SCORE_FORMAT);
    expect(id6).not.toHaveProperty("stall_tie_format");
    expect(damageAwareIdentityHash(id7)).not.toBe(damageAwareIdentityHash(id6));
    expect(id7.baseline_config_hash).not.toBe(id6.baseline_config_hash);
    const { stall_tie_format: _f, baseline_config_hash: _h, feature_set_hash: _fs, ...rest7 } = id7;
    const { baseline_config_hash: _h6, feature_set_hash: _fs6, ...rest6 } = id6;
    expect(rest7).toEqual(rest6);
  });
});

// ---- a probe stand-in whose damage, speed order and success chances the tests control

interface Scene {
  /** Roll of each move id (the target has 100 hit points, so a roll of 30 is 300,000 ppm and 100 a knock-out). */
  rolls: Record<string, number>;
  safe: string[];
  foeFirst: boolean;
  /** Success chance (ppm) of each own move id in every branch; absent is certain. */
  success: Record<string, number>;
}

function sceneProbe(scene: Scene): (context: MechanicsContext, query: ParsedMechanicsQuery) => ProbeOutcome {
  return (context, query) => {
    const q = query.query;
    if (q.kind === "damage") {
      const defender = q.defender.set.toLowerCase();
      return stubProbe({ values: scene.safe.some((species) => defender.includes(species)) ? {} : scene.rolls })(context, query);
    }
    if (q.kind === "speed_order") {
      const actions = q.actions.map((action) => ({ ...action, order: 0, priority: 0, speed: 100 }));
      const side = (s: number) => q.actions.flatMap((action, index) => (action.side === s ? [index] : []));
      const groups = (scene.foeFirst ? [side(1), side(0)] : [side(0), side(1)]).filter((group) => group.length > 0);
      return stubProbe({ override: { speed_order: { kind: "computed", answer: { kind: "speed_order", order_basis: "turn_start", actions, groups } } as never } })(context, query);
    }
    if (q.kind === "move_outcome") {
      const s = scene.success[(q.actions[0] as { move: string }).move] ?? 1_000_000;
      const precondition = s === 0 ? "fails" : s === 1_000_000 ? "passes" : "chance";
      return stubProbe({ override: { move_outcome: { kind: "computed", answer: { kind: "move_outcome", precondition, precondition_ppm: s, hit_ppm: 1_000_000 } } as never } })(context, query);
    }
    return stubProbe()(context, query);
  };
}

type Body = { active: { moves: { id: string }[] }[]; side: { pokemon: { condition: string; details: string }[] } };
const bodyOf = (c: Captured): Body => c.request.request as unknown as Body;

let captured: Captured[] = [];
beforeAll(async () => {
  captured = [...(await capture(corpus[1] as never, "p1", 14)), ...(await capture(corpus[1] as never, "p2", 14)), ...(await capture(corpus[3] as never, "p1", 14))];
}, 120_000);

/** Scored move requests with a live foe that has revealed a damaging move, a switch candidate, and own moves that are none of those. */
const scenes = (): { c: Captured; foeMoves: string[] }[] =>
  captured
    .filter((c) => c.request.phase === "move" && (c.request.legal?.choices.length ?? 0) >= 2 && (c.request.legal?.choices ?? []).some((choice) => choice.includes("switch")))
    .flatMap((c) => {
      const state = stateOf(c);
      const foeMoves = Object.values(state.opponent.slots).flatMap((index) => {
        const mon = index === null ? undefined : state.opponent.roster[index];
        return mon === undefined || mon.fainted ? [] : mon.moves_revealed.filter((id) => (engine.sim.Dex.mod("champions").moves.get(id) as unknown as { category: string }).category !== "Status");
      });
      const own = new Set(bodyOf(c).active.flatMap((a) => a.moves.map((m) => m.id)));
      return foeMoves.length > 0 && !foeMoves.some((id) => own.has(id)) ? [{ c, foeMoves }] : [];
    });

const sceneFor = (c: Captured, foeMoves: string[], change: Partial<Scene> = {}): Scene => ({
  rolls: Object.fromEntries(foeMoves.map((id) => [id, 100])),
  safe: [],
  foeFirst: true,
  success: {},
  ...change,
});

const run = (c: Captured, scene: Scene, config = v7): { choice: string; actor: DamageAwareActor } => {
  const actor = damageAwareActor(options(config, { probe: sceneProbe(scene) }));
  return { choice: actor(c.request, c.context) as string, actor };
};



describe("choosing in a tie (a stand-in probe with known answers)", () => {
  const partsOf = (c: Captured, choice: string) => parseJointChoice(choice).map((part, slot) => (part.kind === "move" ? (bodyOf(c).active[slot]?.moves[part.index - 1]?.id ?? null) : null));
  const moveOnly = (c: Captured, choice: string): boolean => parseJointChoice(choice).every((part) => part.kind === "move") && partsOf(c, choice).every((id) => id !== null);
  const withStall = (c: Captured, choice: string): boolean => partsOf(c, choice).some((id) => id !== null && stall.has(id));

  /** A request whose legal set is a stall candidate first and then one without a stalling move, both of moves only. */
  const tied = (c: Captured, order: "stall-first" | "stall-last"): Captured | null => {
    const legal = c.request.legal as NonNullable<Captured["request"]["legal"]>;
    const moves = legal.choices.filter((choice) => moveOnly(c, choice));
    const stalling = moves.find((choice) => withStall(c, choice));
    const plain = moves.find((choice) => !withStall(c, choice));
    if (stalling === undefined || plain === undefined) return null;
    const request = structuredClone(c.request);
    request.legal = { ...legal, choices: order === "stall-first" ? [stalling, plain] : [plain, stalling], pending: [] };
    return { ...c, request };
  };
  /** Zero damage everywhere (every score is 0, a tie of everything) and every own move at 1/3, the engine's repeat chance. */
  const flat = (c: Captured): Scene => ({
    rolls: {},
    safe: [],
    foeFirst: false,
    success: Object.fromEntries(bodyOf(c).active.flatMap((a) => a.moves.map((m) => [m.id, 333_333]))),
  });

  it("the all-zero fallback: v6 plays the failing stall that is listed first, v7 the other candidate, and counts it", () => {
    let checked = 0;
    for (const { c } of scenes()) {
      const request = tied(c, "stall-first");
      if (request === null) continue;
      const scene = flat(c);
      const six = run(request, scene, v6);
      const seven = run(request, scene);
      const [first, second] = (request.request.legal as NonNullable<Captured["request"]["legal"]>).choices as [string, string];
      expect(six.choice, `request ${c.request.request_seq}`).toBe(first);
      expect(seven.choice, `request ${c.request.request_seq}`).toBe(second);
      expect(seven.actor.report().success?.stall_tie_avoided).toBe(1);
      expect(six.actor.report().success).not.toHaveProperty("stall_tie_avoided");
      checked++;
    }
    expect(checked).toBeGreaterThan(0);
  });

  it("the stall listed last never wins either, and v7 then plays what v6 plays", () => {
    let checked = 0;
    for (const { c } of scenes()) {
      const request = tied(c, "stall-last");
      if (request === null) continue;
      const scene = flat(c);
      const seven = run(request, scene);
      expect(seven.choice).toBe(run(request, scene, v6).choice);
      expect(withStall(c, seven.choice)).toBe(false);
      checked++;
    }
    expect(checked).toBeGreaterThan(0);
  });

  it("a failing stall is still played when it is the only candidate (nothing else to play, nothing counted)", () => {
    let checked = 0;
    for (const { c } of scenes()) {
      const request = tied(c, "stall-first");
      if (request === null) continue;
      const legal = request.request.legal as NonNullable<Captured["request"]["legal"]>;
      const only: Captured = { ...request, request: { ...request.request, legal: { ...legal, choices: [legal.choices[0] as string] } } };
      // one candidate has one answer: the actor does not score it at all
      const seven = run(only, flat(c));
      expect(seven.choice).toBe(legal.choices[0]);
      expect(seven.actor.report().success?.stall_tie_avoided).toBe(0);
      checked++;
    }
    expect(checked).toBeGreaterThan(0);
  });

  it("with every chance certain (a first use) v7 plays what v6 plays at every request and avoids nothing", () => {
    let compared = 0;
    for (const { c, foeMoves } of scenes()) {
      const scene = sceneFor(c, foeMoves);
      const six = run(c, scene, v6);
      const seven = run(c, scene);
      expect(seven.choice, `request ${c.request.request_seq}`).toBe(six.choice);
      expect(seven.actor.report().success?.stall_tie_avoided).toBe(0);
      compared++;
    }
    expect(compared).toBeGreaterThan(3);
  });

  it("an outcome that cannot be read is not a failing stall: v7 plays what v6 plays and avoids nothing", () => {
    let checked = 0;
    for (const { c, foeMoves } of scenes()) {
      const scene = sceneFor(c, foeMoves);
      const unread = (context: MechanicsContext, query: ParsedMechanicsQuery): ProbeOutcome => {
        const q = query.query;
        if (q.kind === "move_outcome") return stubProbe({ override: { move_outcome: { kind: "unsupported", reason: "history_unknown", position: null } as never } })(context, query);
        return sceneProbe(scene)(context, query);
      };
      const six = damageAwareActor(options(v6, { probe: unread }));
      const seven = damageAwareActor(options(v7, { probe: unread }));
      expect(seven(c.request, c.context), `request ${c.request.request_seq}`).toBe(six(c.request, c.context));
      expect(seven.report().success?.stall_tie_avoided).toBe(0);
      checked++;
    }
    expect(checked).toBeGreaterThan(0);
  });

  it("the counts a v7 report hands to the closed baseline records carry neither `causes` nor `stall_tie_avoided`", () => {
    const { c } = scenes()[0] as ReturnType<typeof scenes>[number];
    const { actor } = run(tied(c, "stall-first") ?? c, flat(c));
    const report = actor.report();
    expect(Object.keys(report.counts).sort()).toEqual(["assumed_set_fallbacks", "decisions", "epsilon_plays", "preview_decisions", "probe_refusals", "probe_runs", "queries", "reader_fallbacks", "scored_decisions", "unsupported_queries"]);
    expect(report.success).toHaveProperty("stall_tie_avoided");
    expect(report.counts).not.toHaveProperty("stall_tie_avoided");
  });

  it("reports the avoided count beside v6's counts, only for v7", () => {
    const { c, foeMoves } = scenes()[0] as ReturnType<typeof scenes>[number];
    expect(run(c, sceneFor(c, foeMoves)).actor.report().success).toMatchObject({ stall_charged: expect.any(Number), stall_tie_avoided: expect.any(Number) });
    expect(run(c, sceneFor(c, foeMoves), v6).actor.report().success).not.toHaveProperty("stall_tie_avoided");
  });
});

describe("hidden-information invariance and determinism (the real probe)", () => {
  const MAX_TURNS = 5;
  async function play(battle: (typeof corpus)[number], p2: typeof battle.teams.p2, p1 = battle.teams.p1) {
    const actor = damageAwareActor(options(v7));
    const record = await runBattle({ engine, formatId: GOLDEN_FORMAT, seed: battle.seed, teams: { p1, p2 }, actors: { p1: actor, p2: firstLegal }, maxTurns: MAX_TURNS });
    return { record, report: actor.report() };
  }

  it("makes the same choices at every request with an equal player channel, whatever the opponent holds unrevealed, and the same run twice", async () => {
    const battle = corpus[0] as (typeof corpus)[number];
    const [mine, theirs] = [withGenders(battle.teams.p1), withGenders(battle.teams.p2)];
    const other = mutated(theirs);
    expect(JSON.stringify(other)).not.toBe(JSON.stringify(theirs));
    const a = await play(battle, theirs, mine);
    const b = await play(battle, other, mine);
    const pairs = comparable(a.record, b.record);
    for (const { seq, first, second } of pairs) expect(second, `request ${seq}`).toBe(first);
    expect(pairs.length).toBeGreaterThanOrEqual(2);
    expect(a.report.counts.scored_decisions).toBeGreaterThan(0);
    const again = await play(battle, theirs, mine);
    expect(again.record.choices).toEqual(a.record.choices);
    const strip = (report: typeof a.report) => jcs({ ...report, counts: { ...report.counts, probe_runs: 0 } } as unknown as Json);
    expect(strip(again.report)).toBe(strip(a.report));
  }, 240_000);

  it("negative control: an actor handed the opponent's real sets makes different choices at such a request, so the comparison has teeth", async () => {
    let differing = 0;
    let compared = 0;
    for (const battle of corpus.slice(0, 3)) {
      const other = mutated(battle.teams.p2);
      const cheating = (real: PokemonSet[]) => (mon: PublicMon): AssumedSet => {
        const set = structuredClone(real.find((candidate) => [mon.species, mon.base_species].includes(toId(candidate.species)))) as PokemonSet;
        return { set, paste: exportPaste(engine, [structuredClone(set)]), revealed_moves: set.moves.map(toId), filler_moves: [], unplaceable_moves: [], shown_hp_percent: mon.fainted ? 0 : mon.hp_num, mega: mon.mega };
      };
      const go = async (team: PokemonSet[]) => {
        const actor = damageAwareActor(options(v7, { assume: cheating(team) }));
        return runBattle({ engine, formatId: GOLDEN_FORMAT, seed: battle.seed, teams: { p1: battle.teams.p1, p2: team }, actors: { p1: actor, p2: firstLegal }, maxTurns: MAX_TURNS });
      };
      for (const { first, second } of comparable(await go(battle.teams.p2), await go(other))) {
        compared++;
        if (first !== second) differing++;
      }
    }
    expect(compared).toBeGreaterThanOrEqual(3);
    expect(differing).toBeGreaterThan(0);
  }, 240_000);
});
