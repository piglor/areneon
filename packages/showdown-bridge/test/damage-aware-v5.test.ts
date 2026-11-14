import { beforeAll, describe, expect, it } from "vitest";
import { readerSha256, HISTORY_FACTS_FORMAT } from "@areneon/public-state";
import { actorRule } from "../src/actor-rule.ts";
import { type OutcomeRow } from "../src/candidate-features-outcome.ts";
import { jcs, type Json } from "../src/canonical-json.ts";
import { DamageAwareError, type DamageAwareActor, type DamageAwareOptions, damageAwareActor } from "../src/damage-aware-actor.ts";
import { loadDamageAwareConfig } from "../src/damage-aware-config.ts";
import { parseJointChoice } from "../src/damage-aware-choice.ts";
import { damageAwareIdentity, damageAwareIdentityHash, SUCCESS_SCORE_FORMAT } from "../src/damage-aware-identity.ts";
import { partSuccess, scaled } from "../src/damage-aware-success.ts";
import { situationalScore } from "../src/damage-aware-situational.ts";
import { resolveFormat } from "../src/formats.ts";
import { firstLegal, GOLDEN_FORMAT } from "../src/golden.ts";
import { withMoveOutcome } from "../src/mechanics-config.ts";
import { OUTCOME_FORMAT } from "../src/mechanics-outcome.ts";
import { type MechanicsContext, MechanicsProbeError, type ProbeOutcome, probeMechanics } from "../src/mechanics-probe.ts";
import { type ParsedMechanicsQuery } from "../src/mechanics-query.ts";
import { loadOutcomeConfig } from "../src/outcome-config.ts";
import { runBattle } from "../src/runner.ts";
import { goldenWorldFactsData } from "../src/world-facts-data.ts";
import { playerChannel } from "../src/player-channel.ts";
import { sheetRegimeOf } from "../src/team-sheet-binding.ts";
import { candidateFeaturesOutcome } from "../src/candidate-features-outcome.ts";
import { type AssumedSet } from "../src/damage-aware-assumed-set.ts";
import type { PublicMon } from "@areneon/public-state";
import type { PokemonSet } from "../src/engine.ts";
import { exportPaste } from "../src/teams.ts";
import { capture, type Captured, corpus, engine, mechanics, snapshot, stateOf, stubProbe, toId, vocabulary } from "./damage-aware-support.ts";
import { comparable, mutated, withGenders } from "./damage-aware-invariance-support.ts";

// #1344: damage-aware@v5 is v4 with every scored part multiplied by the engine's success chance (the `move_outcome` probe, read through
// the `probe-outcome/1` builder), damage values computed with the player's own channel history declared, ties broken by the chance and a
// part that cannot work in any branch never played while another candidate exists.

const v4 = loadDamageAwareConfig("v4", vocabulary);
const v5 = loadDamageAwareConfig("v5", vocabulary);
const facts_data = goldenWorldFactsData();
const stall = new Set(facts_data.stall_moves);
const mechanics5 = withMoveOutcome(mechanics);
const outcome = { config: loadOutcomeConfig(GOLDEN_FORMAT) };
const W = { switch_value_ppm: 1_000_000, protect_value_ppm: 600_000 };

const row = (change: Partial<OutcomeRow> & { success?: number }): OutcomeRow => {
  const { success, ...rest } = change;
  return { slot: 0, move: 1, target: 0, mega: false, action: "switch", foe_move: null, assumed: false, kind: "computed", precondition: "passes", precondition_ppm: 1_000_000, hit_ppm: 1_000_000, success_ppm: success ?? 1_000_000, ...rest } as OutcomeRow;
};
const key = { slot: 0, move: 1, target: null, mega: false };

describe("the success reading of a part", () => {
  it("reads the branch of the foe's first revealed damaging move, else the switch branch", () => {
    const rows = [row({ success: 900_000 }), row({ action: "move", foe_move: "a", success: 300_000 }), row({ action: "move", foe_move: "b", success: 100_000 })];
    expect(partSuccess(rows, key, "b")).toEqual({ ppm: 100_000, dead: false, known: true });
    expect(partSuccess(rows, key, null)).toEqual({ ppm: 900_000, dead: false, known: true });
    // a revealed move whose branch was cut (or an assumed branch) falls back to the switch branch
    expect(partSuccess(rows, key, "zzz").ppm).toBe(900_000);
    expect(partSuccess([row({ success: 900_000 }), row({ action: "move", foe_move: "a", assumed: true, success: 5 })], key, "a").ppm).toBe(900_000);
  });

  it("matches the row by slot, move, target and Mega flag, a missing target being 0", () => {
    const rows = [row({ target: 2, success: 400_000 }), row({ target: -1, success: 200_000 }), row({ mega: true, success: 100_000 })];
    expect(partSuccess(rows, { ...key, target: 2 }, null).ppm).toBe(400_000);
    expect(partSuccess(rows, { ...key, target: -1 }, null).ppm).toBe(200_000);
    expect(partSuccess(rows, { ...key, mega: true }, null).ppm).toBe(100_000);
    expect(partSuccess(rows, { ...key, slot: 1 }, null)).toEqual({ ppm: 1_000_000, dead: false, known: false });
  });

  it("a part is dead only when every branch is computed and gives 0; an unsupported row is certain and unknown", () => {
    const dead = [row({ success: 0 }), row({ action: "move", foe_move: "a", success: 0 })];
    expect(partSuccess(dead, key, "a")).toEqual({ ppm: 0, dead: true, known: true });
    expect(partSuccess([...dead.slice(0, 1), row({ action: "move", foe_move: "a", success: 1 })], key, "a").dead).toBe(false);
    const unsupported = [row({ kind: "unsupported", reason: "query_kind" } as never), row({ action: "move", foe_move: "a", success: 0 })];
    expect(partSuccess(unsupported, key, null)).toEqual({ ppm: 1_000_000, dead: false, known: false });
    expect(partSuccess(unsupported, key, "a")).toEqual({ ppm: 0, dead: false, known: true });
  });

  it("scales a value by a chance with integer flooring", () => {
    expect(scaled(600_000, 333_333)).toBe(199_999);
    expect(scaled(600_000, 1_000_000)).toBe(600_000);
    expect(scaled(600_000, 0)).toBe(0);
  });

  it("a stalling move is worth its weight with no streak rule (the chance scales it); the streak list stays v4's rule", () => {
    const base = { threats: [{ own: 0, mega: false, foe: 1, move: "x", kind: "computed", mean_ppm: 0, max_ppm: 0, min_ppm: 0, ohko_rolls: 2 }], speeds: [{ own: 0, mega: false, foe: 1, kind: "computed", relation: "foe_first" }], foe_priorities: [] } as never;
    const input = (streaks: (number | null)[] | null) => ({ rows: base, streaks, stall, moveId: () => "protect", weights: W });
    const part = { kind: "move" as const, index: 1, target: null, mega: false };
    expect(situationalScore(input(null), 0, part)).toBe(600_000);
    expect(situationalScore(input([1]), 0, part)).toBe(0);
    expect(situationalScore(input([0]), 0, part)).toBe(600_000);
  });
});

const options = (config = v5, change: Partial<DamageAwareOptions> = {}): DamageAwareOptions => ({
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
  it("v5 is the policy id of a record with success scaling; another version's record or id is refused both ways", () => {
    expect(damageAwareActor(options()).report().policy_id).toBe("damage-aware@v5");
    expect(() => damageAwareActor(options(v5, { commit: "v4" }))).toThrow(DamageAwareError);
    expect(() => damageAwareActor(options(v4, { commit: "v5" }))).toThrow(DamageAwareError);
    expect(damageAwareActor(options(v4)).report().policy_id).toBe("damage-aware@v4");
  });

  it("needs the outcome record and a mechanics config that lists move_outcome exactly when it scales", () => {
    const { outcome: _gone, ...without } = options();
    expect(() => damageAwareActor(without)).toThrow("needs the probe-outcome record");
    expect(() => damageAwareActor(options(v4, { outcome }))).toThrow("only a record with success scaling");
    expect(() => damageAwareActor(options(v5, { mechanics }))).toThrow("move_outcome");
    const { situational: _s, ...bare } = v5;
    expect(() => damageAwareActor(options(bare as typeof v5))).toThrow("situational");
  });

  it("the identity of v5 names the success, history and outcome formats and the probe-outcome set; v4's keys are unchanged", () => {
    const manifest = resolveFormat(engine, GOLDEN_FORMAT, snapshot).manifest;
    const id5 = damageAwareIdentity({ config: v5, mechanics: mechanics5, manifest, reader_sha256: readerSha256(), outcome_config: outcome.config });
    const id4 = damageAwareIdentity({ config: v4, mechanics, manifest, reader_sha256: readerSha256() });
    expect(id5.success_format).toBe(SUCCESS_SCORE_FORMAT);
    expect(id5.history_facts_format).toBe(HISTORY_FACTS_FORMAT);
    expect(id5.outcome_format).toBe(OUTCOME_FORMAT);
    expect(id5.feature_set_hash).toMatch(/^[0-9a-f]{64}$/);
    expect(id5).not.toHaveProperty("world_facts_format");
    expect(id4).not.toHaveProperty("success_format");
    expect(id4).not.toHaveProperty("outcome_format");
    expect(id4).toHaveProperty("world_facts_format");
    expect(damageAwareIdentityHash(id5)).not.toBe(damageAwareIdentityHash(id4));
    expect(id5.baseline_config_hash).not.toBe(id4.baseline_config_hash);
    // the outcome record goes with success scaling and only with it
    expect(() => damageAwareIdentity({ config: v5, mechanics: mechanics5, manifest, reader_sha256: readerSha256() })).toThrow(RangeError);
    expect(() => damageAwareIdentity({ config: v4, mechanics, manifest, reader_sha256: readerSha256(), outcome_config: outcome.config })).toThrow(RangeError);
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

const run = (c: Captured, scene: Scene, config = v5): { choice: string; actor: DamageAwareActor } => {
  const actor = damageAwareActor(options(config, { probe: sceneProbe(scene) }));
  return { choice: actor(c.request, c.context) as string, actor };
};
const legalOf = (c: Captured): string[] => (c.request.legal as NonNullable<Captured["request"]["legal"]>).choices;
const moveIds = (c: Captured, choice: string): string[] =>
  parseJointChoice(choice).flatMap((part, slot) => (part.kind === "move" ? [(bodyOf(c).active[slot] as Body["active"][number]).moves[part.index - 1]?.id ?? ""] : []));

describe("choosing with the engine's success chance (a stand-in probe with known answers)", () => {
  it("with every part certain it plays what v4 plays (v5 reads no Protect streak, so requests after a Protect are left out)", () => {
    let compared = 0;
    for (const { c, foeMoves } of scenes()) {
      const scene = sceneFor(c, foeMoves);
      const asked = run(c, scene, v4).choice;
      const v5choice = run(c, scene).choice;
      const used = moveIds(c, asked);
      // v4 would score a repeated stalling move 0: compare only where no candidate's stalling move can have a streak (turn 1 requests)
      if (c.request.request_seq <= 2 || used.every((id) => !stall.has(id))) {
        expect(v5choice, `request ${c.request.request_seq}`).toBe(asked);
        compared++;
      }
    }
    expect(compared).toBeGreaterThan(3);
  });

  it("never plays a candidate with a part that cannot work in any branch while another candidate exists", () => {
    let excluded = 0;
    for (const { c, foeMoves } of scenes()) {
      const normal = run(c, sceneFor(c, foeMoves)).choice;
      const used = moveIds(c, normal);
      if (used.length === 0) continue;
      const scene = sceneFor(c, foeMoves, { success: Object.fromEntries(used.map((id) => [id, 0])) });
      const { choice, actor } = run(c, scene);
      expect(moveIds(c, choice).some((id) => used.includes(id)), `request ${c.request.request_seq}`).toBe(legalOf(c).every((candidate) => moveIds(c, candidate).some((id) => used.includes(id))));
      if (!moveIds(c, choice).some((id) => used.includes(id))) {
        excluded++;
        expect(actor.report().success?.dead_candidates).toBeGreaterThan(0);
      }
    }
    expect(excluded).toBeGreaterThan(2);
  });

  it("a second consecutive Protect scores a third of its value and loses to a damaging move it beat at full chance", () => {
    let checked = 0;
    for (const { c, foeMoves } of scenes()) {
      const body = bodyOf(c);
      const own = body.active.flatMap((a, slot) => a.moves.map((m) => ({ slot, id: m.id })));
      const guard = own.find((m) => stall.has(m.id));
      const hit = own.find((m) => !stall.has(m.id) && m.slot === guard?.slot && !foeMoves.includes(m.id) && (engine.sim.Dex.mod("champions").moves.get(m.id) as unknown as { category: string }).category !== "Status");
      if (guard === undefined || hit === undefined) continue;
      // Protect is worth 600,000 at a certain chance; the damaging move 300,000 (a roll of 60 of 100, halved by the turn-order discount: a foe acts first); no switch is safe
      const rolls = { ...Object.fromEntries(foeMoves.map((id) => [id, 100])), [hit.id]: 60 };
      const certain = run(c, sceneFor(c, foeMoves, { rolls })).choice;
      const third = run(c, sceneFor(c, foeMoves, { rolls, success: Object.fromEntries([...stall].map((id) => [id, 333_333])) })).choice;
      const at = (choice: string): string | undefined => {
        const part = parseJointChoice(choice)[guard.slot];
        return part?.kind === "move" ? body.active[guard.slot]?.moves[part.index - 1]?.id : undefined;
      };
      if (at(certain) !== guard.id) continue;
      expect(at(third), `request ${c.request.request_seq}`).toBe(hit.id);
      expect(third).not.toBe(certain);
      checked++;
    }
    expect(checked).toBeGreaterThan(0);
  });

  it("a request where every total is 0 plays the candidate with the highest summed success chance, then the first in canonical order", () => {
    let differs = 0;
    for (const { c } of scenes()) {
      // nothing deals damage and nothing threatens: every candidate totals 0
      const success = Object.fromEntries(bodyOf(c).active.flatMap((a) => a.moves.map((m) => [m.id, 100_000 + ((m.id.length * 137_003 + m.id.charCodeAt(0) * 7_919) % 800_000)])));
      const scene: Scene = { rolls: {}, safe: [], foeFirst: false, success };
      const chance = (choice: string): number => parseJointChoice(choice).reduce((sum, part, slot) => sum + (part.kind === "move" ? (success[(bodyOf(c).active[slot] as Body["active"][number]).moves[part.index - 1]?.id ?? ""] ?? 1_000_000) : 1_000_000), 0);
      const legal = legalOf(c);
      const top = Math.max(...legal.map(chance));
      const expected = legal.find((choice) => chance(choice) === top) as string;
      const { choice, actor } = run(c, scene);
      expect(choice, `request ${c.request.request_seq}`).toBe(expected);
      if (expected !== actorRule(c.request.legal as never)) {
        differs++;
        expect(actor.report().success?.chance_ties).toBe(1);
      }
    }
    expect(differs).toBeGreaterThan(0);
  });

  it("keeps the view's `causes` table out of the counts a report hands to the closed baseline records (#1379)", () => {
    const { c } = scenes()[0] as ReturnType<typeof scenes>[number];
    // A refused outcome query is noted by cause in the view's counts (history is declared for v5) ...
    const inner = sceneProbe(sceneFor(c, []));
    const noted = damageAwareActor(options(v5, { probe: (context, query) => {
      if (query.query.kind === "move_outcome") throw new MechanicsProbeError("declared state refused");
      return inner(context, query);
    } }));
    noted(c.request, c.context);
    expect(noted.report().counts.probe_refusals).toBeGreaterThan(0);
    // ... and the report carries the nine counts (and probe_refusals) only.
    expect(Object.keys(noted.report().counts).sort()).toEqual(["assumed_set_fallbacks", "decisions", "epsilon_plays", "preview_decisions", "probe_refusals", "probe_runs", "queries", "reader_fallbacks", "scored_decisions", "unsupported_queries"]);
  });

  it("answers a request the outcome reading cannot be built for by the actor-rule candidate and counts it by cause", () => {
    const { c } = scenes()[0] as ReturnType<typeof scenes>[number];
    const refused = damageAwareActor(options(v5, { probe: stubProbe({ throwOn: { stats: new MechanicsProbeError("declared state refused") } }) }));
    expect(refused(c.request, c.context)).toBe(actorRule(c.request.legal as never));
    expect(refused.report().counts).toMatchObject({ probe_refusals: 1, scored_decisions: 0 });
  });

  it("reports what the scaling read, beside the nine counts, and only for v5", () => {
    const { c, foeMoves } = scenes()[0] as ReturnType<typeof scenes>[number];
    const { actor } = run(c, sceneFor(c, foeMoves));
    expect(actor.report().success).toMatchObject({ dead_candidates: 0 });
    expect((actor.report().success?.parts_scaled ?? 0) + (actor.report().success?.parts_unread ?? 0)).toBeGreaterThan(0);
    expect(actor.report().situational).toEqual({ protect_no_facts: 0 });
    expect(run(c, sceneFor(c, foeMoves), v4).actor.report()).not.toHaveProperty("success");
  });
});

describe("an outcome that cannot be read never raises a stalling part's value", () => {
  /** A scene where a Threatened slot has a stalling move and a damaging move worth less than the Protect weight at a certain chance. */
  const guardScenes = () =>
    scenes().flatMap(({ c, foeMoves }) => {
      const body = bodyOf(c);
      const own = body.active.flatMap((a, slot) => a.moves.map((m) => ({ slot, id: m.id })));
      const guard = own.find((m) => stall.has(m.id));
      const hit = own.find((m) => !stall.has(m.id) && m.slot === guard?.slot && !foeMoves.includes(m.id) && (engine.sim.Dex.mod("champions").moves.get(m.id) as unknown as { category: string }).category !== "Status");
      return guard === undefined || hit === undefined ? [] : [{ c, foeMoves, guard, hit }];
    });
  const at = (c: Captured, choice: string, slot: number): string | undefined => {
    const part = parseJointChoice(choice)[slot];
    return part?.kind === "move" ? bodyOf(c).active[slot]?.moves[part.index - 1]?.id : undefined;
  };
  /** The probe of `sceneProbe`, except that a stalling move's outcome is the given one. */
  const withStallOutcome = (scene: Scene, unread: boolean) => (context: MechanicsContext, query: ParsedMechanicsQuery): ProbeOutcome => {
    const q = query.query;
    if (unread && q.kind === "move_outcome" && stall.has((q.actions[0] as { move: string }).move)) {
      return stubProbe({ override: { move_outcome: { kind: "unsupported", reason: "history_unknown", position: null } as never } })(context, query);
    }
    return sceneProbe(scene)(context, query);
  };

  it("an unsupported (history_unknown) outcome for the stalling move gives the Protect situational value 0, counted; a computed one gives the weight", () => {
    let checked = 0;
    for (const { c, foeMoves, guard, hit } of guardScenes()) {
      const scene = sceneFor(c, foeMoves, { rolls: { ...Object.fromEntries(foeMoves.map((id) => [id, 100])), [hit.id]: 60 } });
      const known = damageAwareActor(options(v5, { probe: withStallOutcome(scene, false) }));
      const chosen = known(c.request, c.context) as string;
      if (at(c, chosen, guard.slot) !== guard.id) continue;
      const unread = damageAwareActor(options(v5, { probe: withStallOutcome(scene, true) }));
      const other = unread(c.request, c.context) as string;
      expect(at(c, other, guard.slot), `request ${c.request.request_seq}`).not.toBe(guard.id);
      expect(unread.report().success?.stall_unread).toBeGreaterThan(0);
      expect(known.report().success?.stall_unread).toBe(0);
      checked++;
    }
    expect(checked).toBeGreaterThan(0);
  });

  it("the same when the history module refuses a line of the channel: no stalling move is chosen on the situational value", () => {
    let checked = 0;
    for (const { c, foeMoves, guard, hit } of guardScenes()) {
      const scene = sceneFor(c, foeMoves, { rolls: { ...Object.fromEntries(foeMoves.map((id) => [id, 100])), [hit.id]: 60 } });
      const known = damageAwareActor(options(v5, { probe: sceneProbe(scene) }));
      if (at(c, known(c.request, c.context) as string, guard.slot) !== guard.id) continue;
      // a turn number that goes back: the public reader takes it, the history facts refuse it (every fact is then unknown for every Pokemon)
      const lines = [...c.context.channel];
      lines.splice(lines.length - 1, 0, "|turn|0");
      // the stand-in answers `history_unknown` for a stalling move whose declared counter is unknown, as the probe's two ends rule does
      const realOutcome = (context: MechanicsContext, query: ParsedMechanicsQuery): ProbeOutcome => {
        const q = query.query;
        if (q.kind === "move_outcome" && stall.has((q.actions[0] as { move: string }).move)) {
          const actor = (q.sides[0] as { history?: { stall_counter?: { kind: string } } }[])[(q.actions[0] as { position: number }).position];
          if (actor?.history?.stall_counter?.kind === "unknown") return stubProbe({ override: { move_outcome: { kind: "unsupported", reason: "history_unknown", position: null } as never } })(context, query);
        }
        return sceneProbe(scene)(context, query);
      };
      const refused = damageAwareActor(options(v5, { probe: realOutcome }));
      const choice = refused(c.request, { ...c.context, channel: lines }) as string;
      if (refused.report().counts.scored_decisions === 0) continue; // the public reader flagged it first: not this case
      expect(at(c, choice, guard.slot)).not.toBe(guard.id);
      // an unsettled Pokemon makes every answer of its queries history_unknown: nothing threatens, so no Protect value is read either
      expect(refused.report().success?.stall_unread).toBeGreaterThanOrEqual(0);
      checked++;
    }
    expect(checked).toBeGreaterThan(0);
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

  it("the second and third use are computed from the declared stall counter (1/3, 1/9), and v5 plays another move when the damage makes that right", async () => {
    const outcomeConfigs = (c: Captured) => {
      const channel = playerChannel({ schema_version: 1, data_class: "synthetic", player: c.context.player, lines: c.context.channel }, sheetRegimeOf(engine.commit, GOLDEN_FORMAT));
      return candidateFeaturesOutcome(engine, { vocabulary, config: v5, mechanics: mechanics5, snapshot, data_class: "synthetic", outcome }, channel, c.context.team as never, (c.request.legal as NonNullable<Captured["request"]["legal"]>).choices.map((choice) => ({ choice, pending: false })));
    };
    // a battle in which the lead's stalling move worked twice in a row (a failed use would reset the counter)
    let found: { played: Captured[]; guard: string; chances: number[] } | null = null;
    for await (const { captured: played, guard } of spam()) {
      const chances = played.slice(0, 3).map((c) => {
        const index = (bodyOf(c).active[0] as Body["active"][number]).moves.findIndex((m) => m.id === guard) + 1;
        const rows = outcomeConfigs(c).outcomes.filter((r) => r.slot === 0 && r.move === index);
        expect(rows.length).toBeGreaterThan(0);
        expect(rows.every((r) => r.kind === "computed")).toBe(true);
        return Math.max(...rows.map((r) => (r.kind === "computed" ? r.success_ppm : 0)));
      });
      if (chances[2] === 111_111) {
        found = { played, guard, chances };
        break;
      }
    }
    expect(found).not.toBeNull();
    const { played, guard, chances } = found as NonNullable<typeof found>;
    // 1, then 1/3, then 1/9 (from the facts' stall counter, declared in the query)
    expect(chances).toEqual([1_000_000, 333_333, 111_111]);

    // damage and turn order are the scene's (the foe's revealed moves reach the slot and act first; an own damaging move is worth more than a third of Protect), the outcome is the real probe's
    for (const c of played.slice(1, 3)) {
      const state = stateOf(c);
      const foeMoves = Object.values(state.opponent.slots).flatMap((i) => (i === null ? [] : ((state.opponent.roster[i] as { moves_revealed: string[] }).moves_revealed ?? [])));
      const body = bodyOf(c);
      const hit = body.active[0]?.moves.find((m) => !stall.has(m.id) && !foeMoves.includes(m.id) && (engine.sim.Dex.mod("champions").moves.get(m.id) as unknown as { category: string }).category !== "Status");
      if (hit === undefined || foeMoves.length === 0) continue;
      const scene = sceneFor(c, foeMoves, { rolls: { ...Object.fromEntries(foeMoves.map((id) => [id, 100])), [hit.id]: 60 } });
      const probe = (context: MechanicsContext, query: ParsedMechanicsQuery): ProbeOutcome => (query.query.kind === "move_outcome" ? probeMechanics(context, query) : sceneProbe(scene)(context, query));
      const actor = damageAwareActor(options(v5, { probe }));
      const choice = actor(c.request, c.context) as string;
      const part = parseJointChoice(choice)[0];
      const used = part?.kind === "move" ? body.active[0]?.moves[part.index - 1]?.id : undefined;
      expect(used, `request ${c.request.request_seq}`).not.toBe(guard);
      expect(actor.report().success?.parts_scaled).toBeGreaterThan(0);
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
        const actor = damageAwareActor(options(v5, { assume: cheating(team) }));
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
