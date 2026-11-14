import { beforeAll, describe, expect, it } from "vitest";
import { readerSha256 } from "@areneon/public-state";
import { jcs, type Json } from "../src/canonical-json.ts";
import { DamageAwareError, type DamageAwareActor, type DamageAwareOptions, damageAwareActor } from "../src/damage-aware-actor.ts";
import { loadDamageAwareConfig } from "../src/damage-aware-config.ts";
import { parseJointChoice } from "../src/damage-aware-choice.ts";
import { damageAwareIdentity, damageAwareIdentityHash, STALL_FAILURE_FORMAT, SUCCESS_SCORE_FORMAT } from "../src/damage-aware-identity.ts";
import { stallCharge, stallThreat } from "../src/damage-aware-situational.ts";
import { type SituationRows } from "../src/situation.ts";
import { resolveFormat } from "../src/formats.ts";
import { firstLegal, GOLDEN_FORMAT } from "../src/golden.ts";
import { withMoveOutcome } from "../src/mechanics-config.ts";
import { type MechanicsContext, MechanicsProbeError, type ProbeOutcome, probeMechanics } from "../src/mechanics-probe.ts";
import { type ParsedMechanicsQuery } from "../src/mechanics-query.ts";
import { loadOutcomeConfig } from "../src/outcome-config.ts";
import { runBattle } from "../src/runner.ts";
import { goldenWorldFactsData } from "../src/world-facts-data.ts";
import { playerChannel } from "../src/player-channel.ts";
import { sheetRegimeOf } from "../src/team-sheet-binding.ts";
import { candidateFeaturesOutcome } from "../src/candidate-features-outcome.ts";
import { type AssumedSet, buildAssumedSet } from "../src/damage-aware-assumed-set.ts";
import type { PublicMon } from "@areneon/public-state";
import type { PokemonSet } from "../src/engine.ts";
import { exportPaste } from "../src/teams.ts";
import { capture, type Captured, corpus, engine, mechanics, snapshot, stateOf, stubProbe, toId, vocabulary } from "./damage-aware-support.ts";
import { comparable, mutated, withGenders } from "./damage-aware-invariance-support.ts";

// #1380: damage-aware@v6 is v5 with a stalling move by a Threatened slot charged the hit it fails to avoid:
// V = floor(P * s / 10^6) - floor(T * (10^6 - s) / 10^6), T the largest Threatening row's mean share plus the KO weight over the reaching rolls.

const v5 = loadDamageAwareConfig("v5", vocabulary);
const v6 = loadDamageAwareConfig("v6", vocabulary);
const facts_data = goldenWorldFactsData();
const stall = new Set(facts_data.stall_moves);
const mechanics5 = withMoveOutcome(mechanics);
const outcome = { config: loadOutcomeConfig(GOLDEN_FORMAT) };
const P = 600_000;
const KO = 1_000_000;
const value = (t: number, s: number): number => Math.floor((P * s) / 1_000_000) - stallCharge(t, s);

const threat = (change: Record<string, unknown>) => ({ own: 0, mega: false, foe: 1, move: "x", kind: "computed", mean_ppm: 880_000, max_ppm: 1_000_000, min_ppm: 0, ohko_rolls: 6, ...change });
const rowsOf = (threats: unknown[], relation: "foe_first" | "own_first" = "foe_first"): SituationRows =>
  ({ threats, speeds: [{ own: 0, mega: false, foe: 1, kind: "computed", relation }, { own: 0, mega: false, foe: 2, kind: "computed", relation }], foe_priorities: [] }) as never;

describe("the failure charge (hand-worked vectors of E-22.6.2)", () => {
  it("T is the mean share plus the KO weight over the reaching rolls: 880,000 + floor(10^6 * 6 / 16) = 1,255,000", () => {
    expect(stallThreat(rowsOf([threat({})]), 0, KO)).toBe(1_255_000);
  });

  it("P 600,000 and T 1,255,000: s = 10^6 gives 600,000 (v5's value), 1/3 gives -636,668, 1/9 gives -1,048,889", () => {
    expect(value(1_255_000, 1_000_000)).toBe(600_000);
    expect(value(1_255_000, 333_333)).toBe(-636_668);
    expect(value(1_255_000, 111_111)).toBe(-1_048_889);
  });

  it("the other table rows: T 912,500 and 2,000,000", () => {
    expect(value(912_500, 333_333)).toBe(-408_334);
    expect(value(912_500, 111_111)).toBe(-744_445);
    expect(value(2_000_000, 333_333)).toBe(-1_133_335);
    expect(value(2_000_000, 111_111)).toBe(-1_711_112);
  });

  it("v5's value of the same repeats (199,999 and 66,666) is what a 1/3 and 1/9 repeat beat attacks with", () => {
    expect(Math.floor((P * 333_333) / 1_000_000)).toBe(199_999);
    expect(Math.floor((P * 111_111) / 1_000_000)).toBe(66_666);
  });

  it("two Threatening foes: the larger t counts", () => {
    const rows = rowsOf([threat({}), threat({ foe: 2, move: "y", mean_ppm: 950_000, ohko_rolls: 2 })]);
    expect(stallThreat(rows, 0, KO)).toBe(Math.max(1_255_000, 950_000 + 125_000));
    expect(stallThreat(rowsOf([threat({}), threat({ foe: 2, move: "y", mean_ppm: 500_000, ohko_rolls: 16 })]), 0, KO)).toBe(1_500_000);
  });

  it("a fallback row (null KO counts, the max reaching the HP) has no KO term", () => {
    expect(stallThreat(rowsOf([threat({ ohko_rolls: null, mean_ppm: 700_000, max_ppm: 1_000_000 })]), 0, KO)).toBe(700_000);
  });

  it("a foe that does not reach, one that acts after the slot, an unsupported row and a Mega variant are not in T; a safe slot has 0", () => {
    expect(stallThreat(rowsOf([threat({ ohko_rolls: 0, max_ppm: 900_000 })]), 0, KO)).toBe(0);
    expect(stallThreat(rowsOf([threat({})], "own_first"), 0, KO)).toBe(0);
    expect(stallThreat(rowsOf([threat({ kind: "unsupported", reason: "query_kind" })]), 0, KO)).toBe(0);
    expect(stallThreat(rowsOf([threat({ mega: true })]), 0, KO)).toBe(0);
    expect(stallThreat(rowsOf([]), 0, KO)).toBe(0);
    // a foe that is first and reaches, next to one that is slower and bigger: only the first counts
    const mixed = { ...rowsOf([threat({}), threat({ foe: 2, move: "z", mean_ppm: 990_000, ohko_rolls: 16 })]), speeds: [{ own: 0, mega: false, foe: 1, kind: "computed", relation: "foe_first" }, { own: 0, mega: false, foe: 2, kind: "computed", relation: "own_first" }] } as never;
    expect(stallThreat(mixed, 0, KO)).toBe(1_255_000);
  });

  it("the KO weight is the record's: a zero weight leaves the mean share", () => {
    expect(stallThreat(rowsOf([threat({})]), 0, 0)).toBe(880_000);
  });

  it("a charge is 0 for a certain part and never above T", () => {
    expect(stallCharge(1_255_000, 1_000_000)).toBe(0);
    expect(stallCharge(1_255_000, 0)).toBe(1_255_000);
  });
});

const options = (config = v6, change: Partial<DamageAwareOptions> = {}): DamageAwareOptions => ({
  engine,
  vocabulary,
  config,
  mechanics: config.success_scaling === undefined ? mechanics : mechanics5,
  snapshot,
  data_class: "synthetic",
  ...(config.situational === undefined ? {} : { facts_data }),
  ...(config.success_scaling === undefined ? {} : { outcome }),
  ...change,
});

describe("construction and identity", () => {
  it("v6 is the policy id of a record with a stall failure cost; another version's record or id is refused both ways", () => {
    expect(damageAwareActor(options()).report().policy_id).toBe("damage-aware@v6");
    expect(() => damageAwareActor(options(v6, { commit: "v5" }))).toThrow(DamageAwareError);
    expect(() => damageAwareActor(options(v5, { commit: "v6" }))).toThrow(DamageAwareError);
    expect(damageAwareActor(options(v5)).report().policy_id).toBe("damage-aware@v5");
  });

  it("a record with the cost but no success scaling is refused by the actor", () => {
    const { success_scaling: _s, ...bare } = v6;
    expect(() => damageAwareActor(options(bare as typeof v6))).toThrow("success scaling");
  });

  it("the identity of v6 names the stall failure format; v1 to v5 do not (their identities are unchanged)", () => {
    const manifest = resolveFormat(engine, GOLDEN_FORMAT, snapshot).manifest;
    const id6 = damageAwareIdentity({ config: v6, mechanics: mechanics5, manifest, reader_sha256: readerSha256(), outcome_config: outcome.config });
    const id5 = damageAwareIdentity({ config: v5, mechanics: mechanics5, manifest, reader_sha256: readerSha256(), outcome_config: outcome.config });
    expect(id6.stall_failure_format).toBe(STALL_FAILURE_FORMAT);
    expect(id6.success_format).toBe(SUCCESS_SCORE_FORMAT);
    expect(id5).not.toHaveProperty("stall_failure_format");
    expect(damageAwareIdentityHash(id6)).not.toBe(damageAwareIdentityHash(id5));
    expect(id6.baseline_config_hash).not.toBe(id5.baseline_config_hash);
    // v6 differs from v5 in the config hash (which the probe-outcome set names too) and the format only
    const { stall_failure_format: _f, baseline_config_hash: _h, feature_set_hash: _fs, ...rest6 } = id6;
    const { baseline_config_hash: _h5, feature_set_hash: _fs5, ...rest5 } = id5;
    expect(rest6).toEqual(rest5);
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

const run = (c: Captured, scene: Scene, config = v6): { choice: string; actor: DamageAwareActor } => {
  const actor = damageAwareActor(options(config, { probe: sceneProbe(scene) }));
  return { choice: actor(c.request, c.context) as string, actor };
};


describe("choosing with the failure charge (a stand-in probe with known answers)", () => {
  const stallOf = (c: Captured) => bodyOf(c).active.flatMap((a, slot) => a.moves.map((m) => ({ slot, id: m.id }))).find((m) => stall.has(m.id));
  const hitOf = (c: Captured, foeMoves: string[], guardSlot: number | undefined) =>
    bodyOf(c).active.flatMap((a, slot) => a.moves.map((m) => ({ slot, id: m.id }))).find((m) => !stall.has(m.id) && m.slot === guardSlot && !foeMoves.includes(m.id) && (engine.sim.Dex.mod("champions").moves.get(m.id) as unknown as { category: string }).category !== "Status");
  const at = (c: Captured, choice: string, slot: number): string | undefined => {
    const part = parseJointChoice(choice)[slot];
    return part?.kind === "move" ? bodyOf(c).active[slot]?.moves[part.index - 1]?.id : undefined;
  };

  it("with every chance certain (a first use) v6 plays what v5 plays at every request, and charges nothing", () => {
    let compared = 0;
    for (const { c, foeMoves } of scenes()) {
      const scene = sceneFor(c, foeMoves);
      const five = run(c, scene, v5);
      const six = run(c, scene);
      expect(six.choice, `request ${c.request.request_seq}`).toBe(five.choice);
      expect(six.actor.report().success?.stall_charged).toBe(0);
      expect(five.actor.report().success).not.toHaveProperty("stall_charged");
      compared++;
    }
    expect(compared).toBeGreaterThan(3);
  });

  it("a repeat at 1/3 that v5 plays (199,999 beats a discounted attack) loses to that attack under v6, and the charge is counted", () => {
    let checked = 0;
    for (const { c, foeMoves } of scenes()) {
      const guard = stallOf(c);
      const hit = hitOf(c, foeMoves, guard?.slot);
      if (guard === undefined || hit === undefined) continue;
      // the attack is worth about 150,000 after the turn-order discount (a roll of 30 of 100), the stalling move 600,000 times its chance
      const scene = sceneFor(c, foeMoves, { rolls: { ...Object.fromEntries(foeMoves.map((id) => [id, 100])), [hit.id]: 30 }, success: Object.fromEntries([...stall].map((id) => [id, 333_333])) });
      const five = run(c, scene, v5);
      if (at(c, five.choice, guard.slot) !== guard.id) continue;
      const six = run(c, scene);
      expect(at(c, six.choice, guard.slot), `request ${c.request.request_seq}`).not.toBe(guard.id);
      expect(six.actor.report().success?.stall_charged).toBeGreaterThan(0);
      checked++;
    }
    expect(checked).toBeGreaterThan(0);
  });

  it("an outcome that cannot be read scores the stall 0 with no charge, as v5 does", () => {
    let checked = 0;
    for (const { c, foeMoves } of scenes()) {
      const guard = stallOf(c);
      if (guard === undefined) continue;
      const scene = sceneFor(c, foeMoves);
      const unread = (context: MechanicsContext, query: ParsedMechanicsQuery): ProbeOutcome => {
        const q = query.query;
        if (q.kind === "move_outcome" && stall.has((q.actions[0] as { move: string }).move)) return stubProbe({ override: { move_outcome: { kind: "unsupported", reason: "history_unknown", position: null } as never } })(context, query);
        return sceneProbe(scene)(context, query);
      };
      const five = damageAwareActor(options(v5, { probe: unread }));
      const six = damageAwareActor(options(v6, { probe: unread }));
      expect(six(c.request, c.context), `request ${c.request.request_seq}`).toBe(five(c.request, c.context));
      expect(six.report().success?.stall_charged).toBe(0);
      checked++;
    }
    expect(checked).toBeGreaterThan(0);
  });

  it("the counts a v6 report hands to the closed baseline records carry neither `causes` nor `stall_charged` (#1379)", () => {
    const { c, foeMoves } = scenes()[0] as ReturnType<typeof scenes>[number];
    const inner = sceneProbe(sceneFor(c, foeMoves));
    const noted = damageAwareActor(options(v6, { probe: (context, query) => {
      if (query.query.kind === "move_outcome") throw new MechanicsProbeError("declared state refused");
      return inner(context, query);
    } }));
    noted(c.request, c.context);
    const report = noted.report();
    expect(report.counts.probe_refusals).toBeGreaterThan(0);
    expect(Object.keys(report.counts).sort()).toEqual(["assumed_set_fallbacks", "decisions", "epsilon_plays", "preview_decisions", "probe_refusals", "probe_runs", "queries", "reader_fallbacks", "scored_decisions", "unsupported_queries"]);
    // the charged count is a diagnostic of the success report only
    expect(report.success).toHaveProperty("stall_charged");
    expect(report.counts).not.toHaveProperty("stall_charged");
  });

  it("reports the charged count beside v5's counts, only for v6", () => {
    const { c, foeMoves } = scenes()[0] as ReturnType<typeof scenes>[number];
    expect(run(c, sceneFor(c, foeMoves)).actor.report().success).toMatchObject({ stall_charged: expect.any(Number) });
    expect(run(c, sceneFor(c, foeMoves), v5).actor.report().success).not.toHaveProperty("stall_charged");
  });
});

describe("Protect, Protect, Protect on the real probe", () => {
  /** A battle whose p1 lead holds a stalling move, played with that move in slot a every turn; the requests p1 was asked, with their contexts. */
  async function* spam(): AsyncGenerator<{ captured: Captured[]; guard: string }> {
    for (const battle of corpus) {
      // the lead gets a stalling move of its species' pool in its last slot, kept only while the team stays legal
      const team = withGenders(battle.teams.p1);
      const lead = team[0] as PokemonSet;
      const pool = vocabulary.roster.find((entry) => entry.id === toId(lead.species) || entry.id === toId(lead.species.split("-")[0] ?? ""))?.moves ?? [];
      const guard = lead.moves.map(toId).find((id) => stall.has(id)) ?? pool.find((id) => stall.has(id));
      if (guard === undefined) continue;
      if (!lead.moves.map(toId).includes(guard)) {
        lead.moves[lead.moves.length - 1] = (engine.sim.Dex.mod("champions").moves.get(guard) as { name: string }).name;
        if ((engine.sim.TeamValidator.get(GOLDEN_FORMAT).validateTeam(structuredClone(team)) ?? []).length > 0) continue;
      }
      const out: Captured[] = [];
      const play = (request: Parameters<typeof firstLegal>[0], context?: Parameters<typeof firstLegal>[1]): string => {
        if (request.phase === "move") out.push({ request: structuredClone(request), context: { ...(context as NonNullable<typeof context>), channel: [...(context as NonNullable<typeof context>).channel] }, battle });
        const legal = request.legal as NonNullable<typeof request.legal>;
        const body = request.request as unknown as Body;
        const pick = request.phase === "move" && body.active !== undefined ? legal.choices.find((choice) => at(body, choice) === guard) : undefined;
        return pick ?? (firstLegal(request, context) as string);
      };
      const at = (body: Body, choice: string): string | undefined => {
        const part = parseJointChoice(choice)[0];
        return part?.kind === "move" ? body.active[0]?.moves[part.index - 1]?.id : undefined;
      };
      await runBattle({ engine, formatId: GOLDEN_FORMAT, seed: battle.seed, teams: { p1: team, p2: withGenders(battle.teams.p2) }, actors: { p1: play as never, p2: firstLegal }, maxTurns: 4 });
      // the first three move requests in which the lead is still the same Pokemon (it holds the stalling move) and none came between them
      const kept = out.filter((c) => (bodyOf(c).active[0]?.moves ?? []).some((m) => m.id === guard));
      if (kept.length >= 3 && kept.slice(0, 3).every((c, i) => c.request.request_seq === (kept[0] as Captured).request.request_seq + i)) yield { captured: kept, guard };
    }
  }

  it("the three uses are worth 600,000, then negative values by the real engine chances (1, 1/3, 1/9), and v6 plays another move when the damage makes that right", async () => {
    const outcomeRows = (c: Captured) => {
      const channel = playerChannel({ schema_version: 1, data_class: "synthetic", player: c.context.player, lines: c.context.channel }, sheetRegimeOf(engine.commit, GOLDEN_FORMAT));
      return candidateFeaturesOutcome(engine, { vocabulary, config: v6, mechanics: mechanics5, snapshot, data_class: "synthetic", outcome }, channel, c.context.team as never, (c.request.legal as NonNullable<Captured["request"]["legal"]>).choices.map((choice) => ({ choice, pending: false })));
    };
    let found: { played: Captured[]; guard: string; chances: number[] } | null = null;
    for await (const { captured: played, guard } of spam()) {
      const chances = played.slice(0, 3).map((c) => {
        const index = (bodyOf(c).active[0] as Body["active"][number]).moves.findIndex((m) => m.id === guard) + 1;
        const rows = outcomeRows(c).outcomes.filter((r) => r.slot === 0 && r.move === index);
        expect(rows.length).toBeGreaterThan(0);
        return Math.max(...rows.map((r) => (r.kind === "computed" ? r.success_ppm : 0)));
      });
      if (chances[2] === 111_111) {
        found = { played, guard, chances };
        break;
      }
    }
    expect(found).not.toBeNull();
    const { played, guard, chances } = found as NonNullable<typeof found>;
    expect(chances).toEqual([1_000_000, 333_333, 111_111]);
    // against a certain knock-out threat (T 1,000,000 with the stand-in's constant rolls): 600,000 at the first use, then negative values
    expect(chances.map((s) => value(1_000_000, s))).toEqual([600_000, -466_668, -822_223]);

    // the foes of the spam battle have revealed no damaging move yet: show each one (the assumed set's own damaging move, a test seam) so the slot is Threatened
    const showing = (mon: PublicMon): AssumedSet => {
      const set = buildAssumedSet({ engine, vocabulary, config: v6 }, mon);
      const dex = engine.sim.Dex.mod("champions");
      const move = vocabulary.roster.find((entry) => entry.id === (mon.species ?? mon.base_species))?.moves.find((id) => (dex.moves.get(id) as unknown as { category: string; basePower: number }).category !== "Status");
      if (move === undefined) return set;
      return { ...set, set: { ...set.set, moves: [...new Set([...set.set.moves.map(toId), move])].slice(0, 4) } as PokemonSet, revealed_moves: [...new Set([...set.revealed_moves, move])] };
    };
    for (const c of played.slice(1, 3)) {
      const body = bodyOf(c);
      const hit = body.active[0]?.moves.find((m) => !stall.has(m.id) && (engine.sim.Dex.mod("champions").moves.get(m.id) as unknown as { category: string }).category !== "Status");
      if (hit === undefined) continue;
      // the foes' damage knocks out (rolls of 100) and acts first; the own damaging move is worth about 150,000 after the discount
      const probe = (context: MechanicsContext, query: ParsedMechanicsQuery): ProbeOutcome => {
        if (query.query.kind === "move_outcome") return probeMechanics(context, query);
        const q = query.query;
        const rolls = q.kind === "damage" ? { [q.move]: q.move === hit.id ? 30 : 100 } : {};
        return sceneProbe({ rolls, safe: [], foeFirst: true, success: {} })(context, query);
      };
      const actor = damageAwareActor(options(v6, { probe, assume: showing }));
      const part = parseJointChoice(actor(c.request, c.context) as string)[0];
      const used = part?.kind === "move" ? body.active[0]?.moves[part.index - 1]?.id : undefined;
      expect(used, `request ${c.request.request_seq}`).not.toBe(guard);
      expect(actor.report().success?.stall_charged).toBeGreaterThan(0);
      return;
    }
    throw new Error("no request with a revealed foe move and a damaging alternative");
  }, 240_000);
});

describe("hidden-information invariance and determinism (the real probe)", () => {
  const MAX_TURNS = 5;
  async function play(battle: (typeof corpus)[number], p2: typeof battle.teams.p2, p1 = battle.teams.p1) {
    const actor = damageAwareActor(options());
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
        const actor = damageAwareActor(options(v6, { assume: cheating(team) }));
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
