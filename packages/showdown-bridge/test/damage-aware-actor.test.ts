import { readFileSync } from "node:fs";
import { beforeAll, describe, expect, it } from "vitest";
import { actorRule, NoCandidateError } from "../src/actor-rule.ts";
import { parseJointChoice } from "../src/damage-aware-choice.ts";
import { AssumedSetError } from "../src/damage-aware-assumed-set.ts";
import { DamageAwareError, damageAwareActor } from "../src/damage-aware-actor.ts";
import { damageAwareIdentity, damageAwareIdentityHash } from "../src/damage-aware-identity.ts";
import { PPM, rollsValue } from "../src/damage-aware-score.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { MechanicsProbeError } from "../src/mechanics-probe.ts";
import { type ParsedMechanicsQuery } from "../src/mechanics-query.ts";
import { randomBelow, randomLegal } from "../src/random-legal.ts";
import { type PublicMon, readerSha256 } from "@areneon/public-state";
import { capture, type Captured, config, corpus, engine, mechanics, newActor, options, stateOf, stubProbe, toId } from "./damage-aware-support.ts";

let captured: Captured[] = [];
beforeAll(async () => {
  captured = [...(await capture(corpus[0] as never)), ...(await capture(corpus[1] as never)), ...(await capture(corpus[3] as never))];
}, 120_000);

const moveRequests = (): Captured[] => captured.filter((c) => c.request.phase === "move" && (c.request.legal?.choices.length ?? 0) >= 2);
const run = (c: Captured, actor = newActor({ probe: stubProbe() })) => actor(c.request, c.context) as string;

type Active = { moves: { id: string; target?: string }[] }[];
const activeOf = (c: Captured): Active => (c.request.request as unknown as { active: Active }).active;
const category = (id: string): string => (engine.sim.Dex.mod("champions").moves.get(id) as unknown as { category: string }).category;
const rollOf = (id: string): number => 1 + ([...id].reduce((sum, ch) => sum + ch.charCodeAt(0), 0) % 90);

/** Opponent actives as the reader shows them, by choice location (2 for the first slot, 1 for the second). */
const foesAt = (c: Captured) => {
  const state = stateOf(c);
  return Object.keys(state.opponent.slots).sort().map((letter, position, all) => ({ loc: all.length - position, mon: state.opponent.roster[state.opponent.slots[letter] as number] as PublicMon }));
};

/** The test's own reading of a candidate's score under `stubProbe`: constant per move, spread = sum over the foes, ally and status = 0. */
function expectedScore(c: Captured, choice: string, discount: (slot: number, loc: number) => boolean = () => false): number {
  const foes = foesAt(c).filter((foe) => !foe.mon.fainted);
  return parseJointChoice(choice).reduce((sum, part, slot) => {
    if (part.kind !== "move") return sum;
    const move = (activeOf(c)[slot] as Active[number]).moves[part.index - 1] as Active[number]["moves"][number];
    if (category(move.id) === "Status" || move.target === undefined) return sum;
    const targets = move.target === "allAdjacentFoes" || move.target === "allAdjacent" ? foes : move.target === "randomNormal" ? foes : foes.filter((foe) => foe.loc === part.target);
    const value = rollsValue(Array(16).fill(rollOf(move.id)), 100, 1_000_000);
    return sum + targets.reduce((total, foe) => total + (discount(slot, foe.loc) ? Math.floor(value / 2) : value), 0);
  }, 0);
}

describe("choosing by damage (a stand-in probe with known answers)", () => {
  it("plays the highest-scoring non-pending candidate, the first of equal scores", () => {
    const values = Object.fromEntries(moveRequests().flatMap((c) => activeOf(c).flatMap((a) => a.moves.map((m) => [m.id, rollOf(m.id)] as const))));
    let checked = 0;
    for (const c of moveRequests().slice(0, 14)) {
      const choice = run(c, newActor({ probe: stubProbe({ values }) }));
      const legal = c.request.legal as NonNullable<typeof c.request.legal>;
      expect(legal.choices).toContain(choice);
      const scores = legal.choices.map((candidate) => expectedScore(c, candidate));
      const top = Math.max(...scores);
      expect(scores[legal.choices.indexOf(choice)], `request ${c.request.request_seq}`).toBe(top);
      // the first candidate in canonical order among the equal tops
      expect(legal.choices.indexOf(choice)).toBe(scores.indexOf(top));
      checked++;
    }
    expect(checked).toBeGreaterThan(8);
  });

  it("plays the actor-rule candidate when every score is 0", () => {
    for (const c of moveRequests().slice(0, 6)) expect(run(c)).toBe(actorRule(c.request.legal as never));
  });

  it("plays the actor-rule candidate at Team Preview and counts it", () => {
    const preview = captured.find((c) => c.request.phase === "preview") as Captured;
    const actor = newActor({ probe: stubProbe() });
    expect(actor(preview.request, preview.context)).toBe(actorRule(preview.request.legal as never));
    expect(actor.report().counts).toMatchObject({ decisions: 1, preview_decisions: 1, scored_decisions: 0, queries: 0 });
  });

  it("discounts an action whose target acts first and can knock the actor out with a revealed move", () => {
    // a request where a foe on the field has revealed a damaging move
    const threatened = moveRequests().find((c) => foesAt(c).some((foe) => foe.mon.moves_revealed.some((m) => category(m) !== "Status") && !foe.mon.fainted)) as Captured;
    expect(threatened).toBeDefined();
    const foes = foesAt(threatened);
    const threatMove = (foes.flatMap((foe) => foe.mon.moves_revealed).find((m) => category(m) !== "Status")) as string;
    const holders = new Set(foes.filter((foe) => foe.mon.moves_revealed.includes(threatMove)).map((foe) => foe.loc));
    // the threat knocks out (roll 100 into 100 hp); every own move is far below that and not equal to the threat's id
    const own = new Set(activeOf(threatened).flatMap((a) => a.moves.map((m) => m.id)));
    const values: Record<string, number> = Object.fromEntries([...own].map((id) => [id, rollOf(id)]));
    values[threatMove] = 100;
    // when the threat move is also one of the actor's own, the stand-in cannot tell the two queries apart: skip that ambiguity
    expect(own.has(threatMove)).toBe(false);
    const legal = threatened.request.legal as NonNullable<typeof threatened.request.legal>;
    const best = (discount: (slot: number, loc: number) => boolean): string => {
      const scores = legal.choices.map((choice) => expectedScore(threatened, choice, discount));
      return legal.choices[scores.indexOf(Math.max(...scores))] as string;
    };
    const log: ParsedMechanicsQuery[] = [];
    const slow = run(threatened, newActor({ probe: stubProbe({ values, foeFirst: false, log }) }));
    expect(slow).toBe(best(() => false));
    const fast = run(threatened, newActor({ probe: stubProbe({ values, foeFirst: true }) }));
    expect(fast).toBe(best((_, loc) => holders.has(loc)));
    // no speed query without a threat; one with it
    expect(log.some((q) => q.query.kind === "speed_order")).toBe(true);
  });

  it("asks no speed order when nothing threatens, and applies no discount for a tie", () => {
    const c = moveRequests()[0] as Captured;
    const log: ParsedMechanicsQuery[] = [];
    run(c, newActor({ probe: stubProbe({ values: { [activeOf(c)[0]?.moves[0]?.id as string]: 50 }, foeFirst: true, log }) }));
    expect(log.filter((q) => q.query.kind === "speed_order")).toEqual([]);
  });

  it("counts unsupported answers, scores them 0 and never approximates", () => {
    const c = moveRequests()[0] as Captured;
    const actor = newActor({ probe: stubProbe({ override: { damage: { kind: "unsupported", reason: "random_effect", position: null } } }) });
    expect(run(c, actor)).toBe(actorRule(c.request.legal as never));
    const counts = actor.report().counts;
    expect(counts.unsupported_queries).toBeGreaterThan(0);
    expect(counts).toMatchObject({ scored_decisions: 1, reader_fallbacks: 0, assumed_set_fallbacks: 0 });
    // an unsupported order leaves the value undiscounted
    const threatened = moveRequests().find((r) => foesAt(r).some((foe) => foe.mon.moves_revealed.some((m) => category(m) !== "Status"))) as Captured;
    const own = activeOf(threatened).flatMap((a) => a.moves.map((m) => m.id));
    const values = Object.fromEntries(own.map((id) => [id, rollOf(id)]));
    const threat = foesAt(threatened).flatMap((foe) => foe.mon.moves_revealed).find((m) => category(m) !== "Status") as string;
    const plain = run(threatened, newActor({ probe: stubProbe({ values: { ...values, [threat]: 100 }, foeFirst: false }) }));
    const unsupportedOrder = newActor({ probe: stubProbe({ values: { ...values, [threat]: 100 }, foeFirst: true, override: { speed_order: { kind: "unsupported", reason: "random_effect", position: null } } }) });
    expect(run(threatened, unsupportedOrder)).toBe(plain);
    expect(unsupportedOrder.report().counts.unsupported_queries).toBeGreaterThan(0);
  });

  it("counts a speed order the probe cannot queue as unsupported, and falls back on any other probe refusal, and lets every other failure through", () => {
    const threatened = moveRequests().find((r) => foesAt(r).some((foe) => foe.mon.moves_revealed.some((m) => category(m) !== "Status"))) as Captured;
    const own = activeOf(threatened).flatMap((a) => a.moves.map((m) => m.id));
    const threat = foesAt(threatened).flatMap((foe) => foe.mon.moves_revealed).find((m) => category(m) !== "Status") as string;
    const values = { ...Object.fromEntries(own.map((id) => [id, rollOf(id)])), [threat]: 100 };
    const plain = run(threatened, newActor({ probe: stubProbe({ values }) }));
    const limit = newActor({ probe: stubProbe({ values, foeFirst: true, throwOn: { speed_order: new MechanicsProbeError("the action needs more than a move and a Mega Evolution") } }) });
    expect(run(threatened, limit)).toBe(plain);
    expect(limit.report().counts.unsupported_queries).toBeGreaterThan(0);
    // a fractional priority (a Quick Claw holder) is the probe's other known limit
    const fractional = newActor({ probe: stubProbe({ values, foeFirst: true, throwOn: { speed_order: new MechanicsProbeError("the action's priority is not a whole number") } }) });
    expect(run(threatened, fractional)).toBe(plain);
    expect(fractional.report().counts.unsupported_queries).toBeGreaterThan(0);
    // Any other refusal of the probe no longer leaves the actor (it used to): the request falls back to actor-rule and is counted (#965 review of public-main-004).
    const other = newActor({ probe: stubProbe({ values, foeFirst: true, throwOn: { speed_order: new MechanicsProbeError("the Pokemon cannot Mega Evolve") } }) });
    expect(run(threatened, other)).toBe(actorRule(threatened.request.legal as never));
    expect(other.report().counts.probe_refusals).toBe(1);
    const damage = newActor({ probe: stubProbe({ values, throwOn: { damage: new MechanicsProbeError("the action needs more than a move and a Mega Evolution") } }) });
    expect(run(threatened, damage)).toBe(actorRule(threatened.request.legal as never));
    expect(damage.report().counts.probe_refusals).toBe(1);
    expect(() => run(threatened, newActor({ probe: stubProbe({ values, foeFirst: true, throwOn: { speed_order: new Error("boom") } }) }))).toThrow("boom");
  });

  it("throws if the probe refuses a set the baseline built (a defect, never hidden)", () => {
    const c = moveRequests()[0] as Captured;
    const actor = newActor({ probe: stubProbe({ override: { damage: { kind: "refused", code: "invalid_set", position: 0 } } }) });
    expect(() => run(c, actor)).toThrow(DamageAwareError);
  });

  it("memoises a repeated query within a game without changing the answer", () => {
    const c = moveRequests()[0] as Captured;
    const log: ParsedMechanicsQuery[] = [];
    const actor = newActor({ probe: stubProbe({ log }) });
    const first = run(c, actor);
    // the same request again, as a retry would: nothing is asked of the probe a second time
    const asked = log.length;
    const again = actor(c.request, { ...c.context, channel: c.context.channel });
    expect(again).toBe(first);
    expect(log.length).toBe(asked);
    const counts = actor.report().counts;
    expect(counts.queries).toBeGreaterThan(counts.probe_runs);
  });
});

describe("what the baseline refuses and where it falls back", () => {
  it("plays the actor-rule candidate and counts it when the reader flagged a line", () => {
    const c = moveRequests()[0] as Captured;
    const actor = newActor({ probe: stubProbe({ values: { [activeOf(c)[0]?.moves[0]?.id as string]: 50 } }) });
    const flagged = { ...c.context, channel: [...c.context.channel, "|nonsense|x|y"] };
    expect(actor(c.request, flagged)).toBe(actorRule(c.request.legal as never));
    expect(actor.report().counts).toMatchObject({ reader_fallbacks: 1, scored_decisions: 0, queries: 0 });
  });

  it("falls back and counts when no legal assumed opponent set can be built", () => {
    const c = moveRequests()[0] as Captured;
    const actor = newActor({ probe: stubProbe(), assume: () => { throw new AssumedSetError("no legal set"); } });
    expect(run(c, actor)).toBe(actorRule(c.request.legal as never));
    expect(actor.report().counts).toMatchObject({ assumed_set_fallbacks: 1, scored_decisions: 0 });
  });

  it("falls back and counts when an own Pokemon is not in the reader's state", () => {
    const c = moveRequests()[0] as Captured;
    // the request and the team agree on a name the player's channel never showed
    const request = structuredClone(c.request);
    const body = request.request as unknown as { side: { pokemon: { ident: string }[] } };
    body.side.pokemon[0] = { ...(body.side.pokemon[0] as { ident: string }), ident: "p1: Ghost" };
    const team = (c.context.team as readonly { name: string; species: string }[]).map((set, i) => (i === 0 ? { ...set, name: "Ghost" } : set));
    const actor = newActor({ probe: stubProbe() });
    expect(actor(request, { ...c.context, team: team as never })).toBe(actorRule(c.request.legal as never));
    expect(actor.report().counts).toMatchObject({ reader_fallbacks: 1, scored_decisions: 0 });
  });

  it("fails closed on a missing context, a missing team and a missing legal set", () => {
    const c = moveRequests()[0] as Captured;
    const actor = newActor({ probe: stubProbe() });
    expect(() => actor(c.request)).toThrow(DamageAwareError);
    expect(() => actor(c.request, { player: c.context.player, channel: c.context.channel })).toThrow(DamageAwareError);
    const { legal: _legal, ...noLegal } = c.request;
    void _legal;
    expect(() => actor(noLegal, c.context)).toThrow(NoCandidateError);
  });

  it("is one actor for one seat of one game", () => {
    const c = moveRequests()[0] as Captured;
    const actor = newActor({ probe: stubProbe() });
    actor(c.request, c.context);
    expect(() => actor(c.request, { ...c.context, channel: c.context.channel.slice(1) })).toThrow(DamageAwareError);
    const other = c.context.player === "p1" ? "p2" : "p1";
    expect(() => actor(c.request, { ...c.context, player: other })).toThrow(DamageAwareError);
    expect(() => newActor({ probe: stubProbe() })(c.request, { ...c.context, player: "p3" as never })).toThrow(DamageAwareError);
  });

  it("refuses a team that does not hold the request's own Pokemon", () => {
    const c = moveRequests()[0] as Captured;
    const team = (c.context.team as readonly { species: string; name: string }[]).map((set) => ({ ...set, name: "Nobody", species: "Pikachu" }));
    expect(() => run({ ...c, context: { ...c.context, team: team as never } })).toThrow(DamageAwareError);
  });

  it("refuses request shapes it does not know", () => {
    const c = moveRequests().find((r) => activeOf(r).some((a) => a.moves.some((m) => m.target === "normal"))) as Captured;
    const edit = (change: (request: Record<string, unknown>) => void) => {
      const request = structuredClone(c.request);
      change(request.request as unknown as Record<string, unknown>);
      return () => newActor({ probe: stubProbe({ values: Object.fromEntries(activeOf(c).flatMap((a) => a.moves.map((m) => [m.id, 40] as const))) }) })(request, c.context);
    };
    expect(edit((r) => ((r.active as Active)[0] as Active[number]).moves.forEach((m) => (m.target = "weirdTarget")))).toThrow(DamageAwareError);
    expect(edit((r) => delete r.active)).toThrow(DamageAwareError);
    expect(edit((r) => (r.side = [1]))).toThrow(DamageAwareError);
    expect(edit((r) => ((r.side as { pokemon: { condition: string }[] }).pokemon[0] as { condition: string }).condition = "unknown")).toThrow(DamageAwareError);
    expect(edit((r) => ((r.side as { pokemon: { ident: unknown }[] }).pokemon[0] as { ident: unknown }).ident = 5)).toThrow(DamageAwareError);
    // a joint choice with the wrong number of parts
    const bad = { ...c.request, legal: { ...(c.request.legal as object), choices: ["move 1 1", "move 2 1"] } } as typeof c.request;
    expect(() => newActor({ probe: stubProbe() })(bad, c.context)).toThrow(DamageAwareError);
    // a single-target move chosen without a target
    const noTarget = { ...c.request, legal: { ...(c.request.legal as object), choices: ["move 1, move 1", "move 2, move 2"] } } as typeof c.request;
    expect(() => newActor({ probe: stubProbe() })(noTarget, c.context)).toThrow(DamageAwareError);
  });

  it("scores the other target types as the request names them", () => {
    const c = moveRequests().find((r) => activeOf(r).some((a) => a.moves.some((m) => m.target === "normal"))) as Captured;
    const kinds = ["allAdjacentFoes", "allAdjacent", "randomNormal", "self", "adjacentAlly", "allySide", "foeSide", "all", "scripted"];
    for (const target of kinds) {
      const request = structuredClone(c.request);
      for (const a of (request.request as unknown as { active: Active }).active) for (const m of a.moves) m.target = target;
      const legal = { ...(c.request.legal as object), choices: ["move 1, move 1", "move 2, move 2", "move 3, move 3"] } as unknown as NonNullable<typeof c.request.legal>;
      const choice = newActor({ probe: stubProbe({ values: Object.fromEntries(activeOf(c).flatMap((a) => a.moves.map((m) => [m.id, rollOf(m.id)] as const))) }) })({ ...request, legal }, c.context) as string;
      expect(legal.choices).toContain(choice);
    }
  });

  it("answers a forced switch, a single candidate and a pending-only set without scoring", () => {
    const c = moveRequests()[0] as Captured;
    const actor = newActor({ probe: stubProbe() });
    const one = { ...c.request, legal: { ...(c.request.legal as object), choices: ["move 1 1, move 1 1"] } } as typeof c.request;
    expect(actor(one, c.context)).toBe("move 1 1, move 1 1");
    expect(actor.report().counts.queries).toBe(0);
    const forced = captured.find((r) => r.request.phase === "forceSwitch");
    if (forced !== undefined) {
      const f = newActor({ probe: stubProbe() });
      expect(f(forced.request, forced.context)).toBe(actorRule(forced.request.legal as never));
      expect(f.report().counts.queries).toBe(0);
    }
    const pending = { ...c.request, legal: { ...(c.request.legal as object), choices: [], pending: ["move 2 1, move 2 1", "move 3 1, move 3 1"] } } as unknown as typeof c.request;
    expect(newActor({ probe: stubProbe() })(pending, c.context)).toBe("move 2 1, move 2 1");
  });
});

describe("epsilon_ppm (ADR-018 section 5)", () => {
  const spy = () => {
    const calls = { below: 0, legal: 0 };
    return {
      calls,
      random: {
        below: ((n: number, state: number) => { calls.below++; return randomBelow(n, state); }) as typeof randomBelow,
        legal: ((legal: never, state: number) => { calls.legal++; return randomLegal(legal, state); }) as typeof randomLegal,
      },
    };
  };

  it("never draws from the stream with epsilon_ppm 0, whatever the seed", () => {
    const { calls, random } = spy();
    expect(config.epsilon_ppm).toBe(0);
    const actor = newActor({ probe: stubProbe(), random, seed: 7 });
    for (const c of captured.filter((r) => r.battle === corpus[0]).slice(0, 8)) actor(c.request, c.context);
    expect(calls).toEqual({ below: 0, legal: 0 });
    expect(actor.report().counts.epsilon_plays).toBe(0);
  });

  it("draws once per decision request and plays a random legal candidate with epsilon_ppm above 0", () => {
    const { calls, random } = spy();
    const sure = { ...config, epsilon_ppm: PPM };
    const actor = newActor({ probe: stubProbe(), random, seed: 99, config: sure });
    const mine = captured.filter((r) => r.battle === corpus[0]);
    const picks = mine.map((c) => actor(c.request, c.context) as string);
    const decisions = mine.filter((c) => c.request.phase !== "preview").length;
    expect(calls).toEqual({ below: decisions, legal: decisions });
    expect(actor.report().counts.epsilon_plays).toBe(decisions);
    mine.forEach((c, i) => expect([...(c.request.legal?.choices ?? []), ...(c.request.legal?.pending ?? [])]).toContain(picks[i]));
    // the same seed gives the same picks; another seed another
    const again = newActor({ probe: stubProbe(), seed: 99, config: sure });
    expect(mine.map((c) => again(c.request, c.context))).toEqual(picks);
    const other = newActor({ probe: stubProbe(), seed: 100, config: sure });
    expect(mine.map((c) => other(c.request, c.context))).not.toEqual(picks);
  });

  it("plays the scored candidate when the draw does not fall under epsilon", () => {
    const rare = { ...config, epsilon_ppm: 1 };
    const actor = newActor({ probe: stubProbe(), seed: 5, config: rare });
    const c = moveRequests()[0] as Captured;
    expect(run(c, actor)).toBe(actorRule(c.request.legal as never));
    expect(actor.report().counts.epsilon_plays).toBe(0);
  });

  it("needs the seat's seed when epsilon_ppm is above 0", () => {
    expect(() => newActor({ config: { ...config, epsilon_ppm: 1 } })).toThrow(DamageAwareError);
  });
});

describe("coupling to the probe and the engine", () => {
  it("names the probe's known speed-order limit as the probe words it", () => {
    expect(readFileSync(new URL("../src/mechanics-speed.ts", import.meta.url), "utf8")).toContain("the action needs more than a move and a Mega Evolution");
    expect(readFileSync(new URL("../src/mechanics-speed.ts", import.meta.url), "utf8")).toContain("the action's priority is not a whole number");
  });
});

describe("identity and report", () => {
  it("reports the policy id, the identity and its hash, and the integer counts", () => {
    const actor = newActor({ probe: stubProbe(), commit: "abc123" });
    const report = actor.report();
    expect(report.policy_id).toBe("damage-aware@abc123");
    expect(report.identity).toEqual(damageAwareIdentity({ config, mechanics, manifest: report.identity.version_manifest, reader_sha256: readerSha256() }));
    expect(report.identity.version_manifest.showdown_commit).toBe(engine.commit);
    expect(report.identity.version_manifest.format_id).toBe(GOLDEN_FORMAT);
    expect(report.identity_hash).toBe(damageAwareIdentityHash(report.identity));
    expect(Object.values(report.counts).every((n) => Number.isInteger(n) && n >= 0)).toBe(true);
    expect(newActor({ probe: stubProbe() }).report().policy_id).toBe("damage-aware@v1");
    // a copy: reading the report does not let a caller change the counts
    report.counts.decisions = 99;
    expect(actor.report().counts.decisions).toBe(0);
  });

  it("reads a Pokemon's name the way the request states it", () => {
    expect(toId("Mr. Mime")).toBe("mrmime");
    expect(options().data_class).toBe("synthetic");
    expect(damageAwareActor).toBeTypeOf("function");
  });
});
