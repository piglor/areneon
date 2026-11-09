import { beforeAll, describe, expect, it } from "vitest";
import { ownChannel, worldFactsAtRequests } from "@areneon/public-state";
import { actorRule } from "../src/actor-rule.ts";
import { jcs, type Json } from "../src/canonical-json.ts";
import { DamageAwareError, type DamageAwareActor, type DamageAwareOptions, damageAwareActor } from "../src/damage-aware-actor.ts";
import { loadDamageAwareConfig } from "../src/damage-aware-config.ts";
import { parseJointChoice } from "../src/damage-aware-choice.ts";
import { damageAwareIdentity, damageAwareIdentityHash, SITUATIONAL_SCORE_FORMAT } from "../src/damage-aware-identity.ts";
import { situationalScore, type SituationalInput } from "../src/damage-aware-situational.ts";
import { firstLegal, GOLDEN_FORMAT } from "../src/golden.ts";
import { type MechanicsContext, MechanicsProbeError, type ProbeOutcome } from "../src/mechanics-probe.ts";
import { AssumedSetError } from "../src/damage-aware-assumed-set.ts";
import { type ParsedMechanicsQuery } from "../src/mechanics-query.ts";
import { readerSha256 } from "@areneon/public-state";
import { resolveFormat } from "../src/formats.ts";
import { classifySlot, foeFirst, reaches, threatShare, type SituationRows } from "../src/situation.ts";
import { runBattle } from "../src/runner.ts";
import { goldenWorldFactsData } from "../src/world-facts-data.ts";
import { capture, type Captured, corpus, engine, mechanics, snapshot, stateOf, stubProbe, vocabulary } from "./damage-aware-support.ts";
import { comparable, mutated, withGenders } from "./damage-aware-invariance-support.ts";

// #1329: damage-aware@v4 is v3 plus two scored actions read from the probe and the engine's stall list: a switch out of a Threatened
// own active to a bench Pokemon the same foes' revealed moves do not threaten, and a stalling move by a Threatened own active.

const v3 = loadDamageAwareConfig("v3", vocabulary);
const v4 = loadDamageAwareConfig("v4", vocabulary);
const facts_data = goldenWorldFactsData();
const stall = new Set(facts_data.stall_moves);
const W = { switch_value_ppm: 1_000_000, protect_value_ppm: 600_000 };

const base = { own: 0, mega: false, foe: 1, move: "x", kind: "computed" as const, mean_ppm: 0, max_ppm: 0, min_ppm: 0, ohko_rolls: 0 };
const threat = (own: number, foe: number, move: string, extra: Partial<typeof base> = {}) => ({ ...base, own, foe, move, ...extra });
const rowsOf = (parts: Partial<SituationRows> = {}): SituationRows => ({ threats: [], speeds: [], foe_priorities: [], ...parts }) as SituationRows;
const speed = (own: number, foe: number, relation: "own_first" | "tie" | "foe_first") => ({ own, mega: false, foe, kind: "computed" as const, relation });
const priority = (foe: number, move: string, value: number) => ({ foe, move, kind: "computed" as const, priority: value });

describe("the situation reading (shared with the situational diagnostic)", () => {
  it("a slot is Threatened only by a revealed move that reaches its HP and acts first", () => {
    const reach = [threat(0, 1, "x", { ohko_rolls: 3 })];
    expect(classifySlot(rowsOf(), 0).situation).toBe("safe");
    expect(classifySlot(rowsOf({ threats: reach, speeds: [speed(0, 1, "foe_first")] }), 0).situation).toBe("threatened");
    expect(classifySlot(rowsOf({ threats: reach, speeds: [speed(0, 1, "own_first")] }), 0).situation).toBe("exposed_slower");
    expect(classifySlot(rowsOf({ threats: reach, speeds: [speed(0, 1, "tie")] }), 0).situation).toBe("exposed_slower");
    // a move that does not reach (no roll at the HP) does not threaten
    expect(classifySlot(rowsOf({ threats: [threat(0, 1, "x", { ohko_rolls: 0 })], speeds: [speed(0, 1, "foe_first")] }), 0).situation).toBe("safe");
  });

  it("a priority move acts first whatever the speed, a negative one does not, and a missing row is reported", () => {
    const reach = [threat(0, 1, "x", { ohko_rolls: 1 })];
    const slower = [speed(0, 1, "own_first")];
    expect(classifySlot(rowsOf({ threats: reach, speeds: slower, foe_priorities: [priority(1, "x", 1)] }), 0).situation).toBe("threatened");
    expect(classifySlot(rowsOf({ threats: reach, speeds: [speed(0, 1, "foe_first")], foe_priorities: [priority(1, "x", -1)] }), 0).situation).toBe("exposed_slower");
    const seen: string[] = [];
    expect(classifySlot(rowsOf({ threats: reach }), 0, (kind) => seen.push(kind)).situation).toBe("exposed_slower");
    expect(seen).toEqual(["foe_priority", "speed_relation"]);
    expect(foeFirst(rowsOf({ speeds: [{ ...speed(0, 1, "foe_first"), kind: "unsupported", reason: "query_kind" } as never] }), 0, 1, "x")).toBe(false);
  });

  it("an unsupported row never threatens; with no KO counts the largest roll decides", () => {
    expect(reaches(rowsOf({ threats: [{ ...threat(0, 1, "x"), kind: "unsupported", reason: "random_effect" } as never] }), 0)).toEqual([]);
    expect(reaches(rowsOf({ threats: [threat(0, 1, "x", { ohko_rolls: null as never, max_ppm: 999_999 })] }), 0)).toEqual([]);
    expect(reaches(rowsOf({ threats: [threat(0, 1, "x", { ohko_rolls: null as never, max_ppm: 1_000_000 })] }), 0)).toEqual([{ foe: 1, move: "x", fallback: true }]);
  });

  it("the threat share is the largest mean share a revealed move takes, unsupported rows counting 0", () => {
    const rows = rowsOf({ threats: [threat(2, 1, "x", { mean_ppm: 250_000 }), threat(2, 2, "y", { mean_ppm: 400_000 }), { ...threat(2, 2, "z"), kind: "unsupported", reason: "query_kind" } as never, threat(3, 1, "x", { mean_ppm: 900_000 })] });
    expect(threatShare(rows, 2)).toBe(400_000);
    expect(threatShare(rows, 5)).toBe(0);
  });
});

describe("the situational score of one slot part", () => {
  const moves = ["tackle", "protect"];
  const input = (rows: SituationRows, streaks: (number | null)[] = [0, 0]): SituationalInput => ({ rows, streaks, stall, moveId: (_slot, index) => moves[index - 1] ?? null, weights: W });
  const threatened = (slot: number) => [threat(slot, 1, "x", { ohko_rolls: 2, mean_ppm: 700_000 })];
  const slower = (slot: number) => [speed(slot, 1, "foe_first")];

  it("a switch of a Threatened slot to a safe bench Pokemon is worth the weight times one minus the bench Pokemon's threat share", () => {
    const rows = rowsOf({ threats: [...threatened(0), threat(2, 1, "x", { ohko_rolls: 0, mean_ppm: 250_000 })], speeds: slower(0) });
    expect(situationalScore(input(rows), 0, { kind: "switch", to: 3 })).toBe(750_000);
    // no row at all for the bench Pokemon: share 0
    expect(situationalScore(input(rowsOf({ threats: threatened(0), speeds: slower(0) })), 0, { kind: "switch", to: 4 })).toBe(1_000_000);
  });

  it("a switch to a Pokemon the same foes' moves reach, and a switch out of a slot that is not Threatened, score 0", () => {
    const reached = rowsOf({ threats: [...threatened(0), threat(2, 1, "x", { ohko_rolls: 1 })], speeds: slower(0) });
    expect(situationalScore(input(reached), 0, { kind: "switch", to: 3 })).toBe(0);
    const exposed = rowsOf({ threats: threatened(0), speeds: [speed(0, 1, "own_first")] });
    expect(situationalScore(input(exposed), 0, { kind: "switch", to: 3 })).toBe(0);
    expect(situationalScore(input(rowsOf()), 0, { kind: "switch", to: 3 })).toBe(0);
  });

  it("a stalling move of a Threatened slot is worth the weight, 0 after a successful use last turn or with no facts, and any other move 0", () => {
    const rows = rowsOf({ threats: threatened(0), speeds: slower(0) });
    const part = (index: number) => ({ kind: "move" as const, index, target: null, mega: false });
    expect(situationalScore(input(rows), 0, part(2))).toBe(600_000);
    expect(situationalScore(input(rows, [1, 0]), 0, part(2))).toBe(0);
    expect(situationalScore(input(rows, [2, 0]), 0, part(2))).toBe(0);
    expect(situationalScore(input(rows, [null, 0]), 0, part(2))).toBe(0);
    expect(situationalScore(input(rows), 0, part(1))).toBe(0);
    expect(situationalScore(input(rows), 0, part(3))).toBe(0);
    expect(situationalScore(input(rowsOf()), 0, part(2))).toBe(0);
    expect(situationalScore(input(rows), 0, { kind: "pass" })).toBe(0);
  });
});

const options = (config = v4, change: Partial<DamageAwareOptions> = {}): DamageAwareOptions => ({
  engine,
  vocabulary,
  config,
  mechanics,
  snapshot,
  data_class: "synthetic",
  ...(config.situational === undefined ? {} : { facts_data }),
  ...change,
});

describe("construction and identity", () => {
  it("v4 is the policy id of a record with situational weights; a record and a policy id of another version are refused", () => {
    const actor = damageAwareActor(options());
    expect(actor.report().policy_id).toBe("damage-aware@v4");
    expect(() => damageAwareActor(options(v4, { commit: "v3" }))).toThrow(DamageAwareError);
    expect(() => damageAwareActor(options(v3, { commit: "v4" }))).toThrow(DamageAwareError);
    expect(damageAwareActor(options(v3)).report().policy_id).toBe("damage-aware@v3");
  });

  it("needs the world facts' data exactly when it has weights, and the engine's target convention", () => {
    const { facts_data: _gone, ...without } = options();
    expect(() => damageAwareActor(without)).toThrow("needs the world facts");
    expect(() => damageAwareActor(options(v3, { facts_data }))).toThrow("only a record with situational weights");
    const legacy = { ...v4 } as typeof v4;
    delete (legacy as { target_convention?: string }).target_convention;
    expect(() => damageAwareActor(options(legacy))).toThrow("engine's target convention");
  });

  it("the identity names the situational format for v4 only; v3's identity keys are unchanged", () => {
    const manifest = resolveFormat(engine, GOLDEN_FORMAT, snapshot).manifest;
    const identity = (config: typeof v4) => damageAwareIdentity({ config, mechanics, manifest, reader_sha256: readerSha256() });
    expect(identity(v4).situational_format).toBe(SITUATIONAL_SCORE_FORMAT);
    expect(Object.keys(identity(v3)).sort()).toEqual(["adapter_format", "baseline_config_hash", "mechanics_config_hash", "reader_sha256", "version_manifest"]);
    expect(damageAwareIdentityHash(identity(v4))).not.toBe(damageAwareIdentityHash(identity(v3)));
    expect(identity(v4).baseline_config_hash).not.toBe(identity(v3).baseline_config_hash);
  });
});

// ---- a probe stand-in whose threats, bench safety and speed order the tests control

interface Scene {
  /** Roll of each foe move (the target has 100 hit points, so 100 or more is a knock-out). */
  rolls: Record<string, number>;
  /** Species (lower case) that no move reaches (a safe bench Pokemon). */
  safe: string[];
  foeFirst: boolean;
  log?: ParsedMechanicsQuery[];
}

function sceneProbe(scene: Scene): (context: MechanicsContext, query: ParsedMechanicsQuery) => ProbeOutcome {
  return (context, query) => {
    const q = query.query;
    scene.log?.push(query);
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
    return stubProbe()(context, query);
  };
}

type Body = { active: { moves: { id: string }[] }[]; side: { pokemon: { condition: string; details: string }[] } };
const bodyOf = (c: Captured): Body => c.request.request as unknown as Body;
const speciesOf = (details: string): string => (details.split(",")[0] as string).toLowerCase();

let captured: Captured[] = [];
beforeAll(async () => {
  captured = [...(await capture(corpus[1] as never, "p1", 14)), ...(await capture(corpus[1] as never, "p2", 14)), ...(await capture(corpus[3] as never, "p1", 14))];
}, 120_000);

/** A scored move request with a live foe that has revealed a damaging move, a switch candidate, and own moves that are none of those. */
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

const factsAt = (c: Captured) => {
  const lines = c.context.channel;
  const all = worldFactsAtRequests({ channel: ownChannel(c.context.player, lines), data: facts_data });
  return all.find((f) => f.line === lines.length - 1)?.facts ?? null;
};

/** The test's own reading of a candidate's situational score under a `Scene` (every slot Threatened or not, switches and stalling moves). */
function expectedScore(c: Captured, choice: string, scene: Scene): number {
  const body = bodyOf(c);
  const facts = factsAt(c);
  return parseJointChoice(choice).reduce((sum, part, slot) => {
    const alive = !/ fnt$/.test((body.side.pokemon[slot] as { condition: string }).condition);
    const threatened = alive && scene.foeFirst;
    if (!threatened) return sum;
    if (part.kind === "switch") {
      const bench = body.side.pokemon[part.to - 1] as { details: string };
      return sum + (scene.safe.includes(speciesOf(bench.details)) ? W.switch_value_ppm : 0);
    }
    if (part.kind !== "move") return sum;
    const id = (body.active[slot] as Body["active"][number]).moves[part.index - 1]?.id ?? "";
    const streak = facts?.own.active[String.fromCharCode(97 + slot)]?.protect_streak ?? null;
    return sum + (stall.has(id) && streak !== null && streak < 1 ? W.protect_value_ppm : 0);
  }, 0);
}

const sceneFor = (c: Captured, foeMoves: string[], foeFirstNow: boolean, safeBench: boolean): Scene => ({
  rolls: Object.fromEntries(foeMoves.map((id) => [id, 100])),
  safe: safeBench ? bodyOf(c).side.pokemon.slice(bodyOf(c).active.length).map((p) => speciesOf(p.details)) : [],
  foeFirst: foeFirstNow,
});

const run = (c: Captured, scene: Scene, config = v4): { choice: string; actor: DamageAwareActor } => {
  const actor = damageAwareActor(options(config, { probe: sceneProbe(scene) }));
  return { choice: actor(c.request, c.context) as string, actor };
};

describe("choosing with the situational values (a stand-in probe with known answers)", () => {
  it("plays the candidate with the highest situational sum, the first of equal sums, and differs from v3 when that sum is positive", () => {
    let positive = 0;
    let withSwitch = 0;
    let withProtect = 0;
    for (const { c, foeMoves } of scenes()) {
      for (const safeBench of [true, false]) {
        const scene = sceneFor(c, foeMoves, true, safeBench);
        const legal = (c.request.legal as NonNullable<Captured["request"]["legal"]>).choices;
        const scores = legal.map((choice) => expectedScore(c, choice, scene));
        const top = Math.max(...scores);
        const { choice } = run(c, scene);
        expect(choice, `request ${c.request.request_seq}`).toBe(legal[scores.indexOf(top)]);
        const v3choice = run(c, scene, v3).choice;
        expect(v3choice).toBe(actorRule(c.request.legal as never));
        if (top > 0) {
          positive++;
          if (scores.indexOf(top) !== 0) expect(choice).not.toBe(v3choice);
        }
        if (top >= W.switch_value_ppm && safeBench) withSwitch++;
        if (!safeBench && top === W.protect_value_ppm) withProtect++;
      }
    }
    expect(positive).toBeGreaterThan(3);
    // both new actions were exercised, not only one
    expect(withSwitch).toBeGreaterThan(0);
    expect(withProtect).toBeGreaterThan(0);
  });

  it("plays v3's choice when no slot is Threatened (the foe acts after the actor), or when the weights are 0", () => {
    const zero = { ...v4, situational: { switch_value_ppm: 0, protect_value_ppm: 0 } };
    for (const { c, foeMoves } of scenes().slice(0, 12)) {
      const slow = sceneFor(c, foeMoves, false, true);
      expect(run(c, slow).choice).toBe(run(c, slow, v3).choice);
      const fast = sceneFor(c, foeMoves, true, true);
      expect(run(c, fast, zero).choice).toBe(run(c, fast, v3).choice);
    }
  });

  it("counts the situational reading's probe queries and keeps the nine v3 count keys", () => {
    const { c, foeMoves } = scenes()[0] as ReturnType<typeof scenes>[number];
    const { actor } = run(c, sceneFor(c, foeMoves, true, true));
    const counts = actor.report().counts;
    expect(Object.keys(counts).sort()).toEqual(Object.keys(run(c, sceneFor(c, foeMoves, true, true), v3).actor.report().counts).sort());
    expect(counts.queries).toBeGreaterThan(run(c, sceneFor(c, foeMoves, true, true), v3).actor.report().counts.queries);
    expect(counts.scored_decisions).toBe(1);
  });

  it("answers a request the probe-damage/2 reading cannot be built for by the actor-rule candidate and counts it by cause", () => {
    const { c } = scenes()[0] as ReturnType<typeof scenes>[number];
    const refused = damageAwareActor(options(v4, { probe: stubProbe({ throwOn: { stats: new MechanicsProbeError("declared state refused") } }) }));
    expect(refused(c.request, c.context)).toBe(actorRule(c.request.legal as never));
    expect(refused.report().counts).toMatchObject({ probe_refusals: 1, scored_decisions: 0 });
    const noSet = damageAwareActor(options(v4, { probe: stubProbe(), assume: () => { throw new AssumedSetError("no legal assumed set"); } }));
    expect(noSet(c.request, c.context)).toBe(actorRule(c.request.legal as never));
    expect(noSet.report().counts).toMatchObject({ assumed_set_fallbacks: 1, scored_decisions: 0 });
  });
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
});
