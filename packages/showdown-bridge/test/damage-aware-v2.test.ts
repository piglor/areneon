import { beforeAll, describe, expect, it } from "vitest";
import type { PublicMon } from "@areneon/public-state";
import { readerSha256 } from "@areneon/public-state";
import { actorRule } from "../src/actor-rule.ts";
import { DamageAwareError, type DamageAwareReport } from "../src/damage-aware-actor.ts";
import { parseJointChoice } from "../src/damage-aware-choice.ts";
import { damageAwareIdentity, damageAwareIdentityHash } from "../src/damage-aware-identity.ts";
import { rollsValue } from "../src/damage-aware-score.ts";
import { worldsStream } from "../src/damage-aware-worlds.ts";
import { MechanicsProbeError } from "../src/mechanics-probe.ts";
import type { ParsedMechanicsQuery } from "../src/mechanics-query.ts";
import { capture, type Captured, config, corpus, mechanics, newActor, stateOf, stubProbe } from "./damage-aware-support.ts";
import { configV2, engine, fakePort, newActorV2, optionsV2, slotOf, vocabulary, worldOf } from "./damage-aware-world-support.ts";
import { damageAwareActor } from "../src/damage-aware-actor.ts";

let captured: Captured[] = [];
beforeAll(async () => {
  captured = [...(await capture(corpus[0] as never)), ...(await capture(corpus[1] as never))];
}, 120_000);
const moveRequests = (): Captured[] => captured.filter((c) => c.request.phase === "move" && (c.request.legal?.choices.length ?? 0) >= 2);
const refusal = (code: string): Error => Object.assign(new Error("refused"), { name: "BeliefError", code });
const legalOf = (c: Captured) => c.request.legal as NonNullable<Captured["request"]["legal"]>;

describe("construction (ADR-025 section 6, review finding N10)", () => {
  it("is named damage-aware@v2, takes its identity from the belief and lists its count of worlds", () => {
    const actor = newActorV2();
    const report = actor.report();
    expect(report.policy_id).toBe("damage-aware@v2");
    expect(report.identity.belief_identity_hash).toBe("b".repeat(64));
    expect(report.identity_hash).toBe(damageAwareIdentityHash(damageAwareIdentity({ config: configV2, mechanics, manifest: report.identity.version_manifest, reader_sha256: readerSha256(), belief_identity_hash: "b".repeat(64) })));
    expect(report.belief).toEqual({ worlds: 8, decisions: 0, worlds_sampled: 0, redraws: 0, contradicted_worlds: 0, no_consistent_world: 0, refusals: {} });
    expect(newActorV2({ commit: "v2-x" }).report().policy_id).toBe("damage-aware@v2-x");
    // a policy id names the version of the record it plays
    expect(() => newActorV2({ commit: "v1" })).toThrow("another record version");
    expect(() => newActorV2({ commit: "abc123" })).toThrow("another record version");
    expect(() => newActor({ commit: "v2" })).toThrow("another record version");
    expect(() => newActor({ commit: "v3-x" })).toThrow("another record version");
    expect(newActor({ commit: "v1-eps5000" }).report().policy_id).toBe("damage-aware@v1-eps5000");
  });

  it("gives v1 a report with no belief key and its own policy id and identity", () => {
    const report = newActor().report();
    expect(report.policy_id).toBe("damage-aware@v1");
    expect("belief" in report).toBe(false);
    expect("belief_identity_hash" in report.identity).toBe(false);
  });

  it("refuses a v2 record without a port, a v1 record with one, and an assume seam beside the belief", () => {
    expect(() => damageAwareActor({ ...optionsV2(), worlds: undefined as never })).toThrow(DamageAwareError);
    expect(() => damageAwareActor({ ...optionsV2(), config, worlds: fakePort() })).toThrow("names no belief");
    expect(() => newActorV2({ assume: (() => null) as never })).toThrow("not from an assume seam");
  });

  it("refuses a prior of another format, whatever else matches: it is never used silently", () => {
    expect(() => newActorV2({ worlds: fakePort(undefined, { prior_format_id: "gen9championsvgc2026regmd" }) })).toThrow("another format");
  });

  it("refuses a port that holds another usage prior or another belief config than the record names", () => {
    expect(() => newActorV2({ worlds: fakePort(undefined, { prior_hash: "0".repeat(64) }) })).toThrow("another usage prior");
    expect(() => newActorV2({ worlds: fakePort(undefined, { belief_config_hash: "0".repeat(64) }) })).toThrow("another belief config");
  });
});

describe("scoring over sampled worlds", () => {
  it("asks the belief once per scored request with the record's worlds, on the stream of the player's own request", () => {
    const port = fakePort();
    const actor = newActorV2({ worlds: port, probe: stubProbe() });
    const c = moveRequests()[0] as Captured;
    actor(c.request, c.context);
    expect(port.calls).toHaveLength(1);
    expect(port.calls[0]?.n).toBe(8);
    expect(Buffer.from(port.calls[0]?.stream as Uint8Array).toString("hex")).toBe(Buffer.from(worldsStream(c.request)).toString("hex"));
    const report = actor.report();
    expect(report.belief).toMatchObject({ decisions: 1, worlds_sampled: 8, redraws: 0 });
    expect(report.counts).toMatchObject({ decisions: 1, scored_decisions: 1 });
    expect(legalOf(c).choices).toContain(actor(c.request, c.context) as string);
  });

  it("answers Team Preview, one candidate and non-move requests without asking the belief", () => {
    const port = fakePort();
    const quiet = captured.filter((x) => x.request.phase !== "move" || legalOf(x).choices.length < 2).slice(0, 6);
    expect(quiet.some((c) => c.request.phase === "preview")).toBe(true);
    for (const c of quiet) expect(newActorV2({ worlds: port, probe: stubProbe() })(c.request, c.context)).toBe(actorRule(legalOf(c)));
    expect(port.calls).toHaveLength(0);
  });

  /** A request with a damaging move in each of two move slots of its first active Pokemon, for the mean test below. */
  const rollFor = (marker: "A" | "B", move: string, hot: [string, string]): number => (marker === "A" ? (move === hot[0] ? 90 : 1) : move === hot[1] ? 90 : 1);

  it("plays the candidate with the best mean over the worlds, not the first world's", () => {
    const category = (id: string): string => (engine.sim.Dex.mod("champions").moves.get(id) as unknown as { category: string }).category;
    type Active = { moves: { id: string; target?: string }[] }[];
    let decisive = 0;
    for (const c of moveRequests().slice(0, 30)) {
      const active = (c.request.request as unknown as { active: Active }).active;
      const damaging = (active[0]?.moves ?? []).filter((m) => m.target !== undefined && category(m.id) !== "Status").map((m) => m.id);
      if (damaging.length < 2) continue;
      const hot: [string, string] = [damaging[0] as string, damaging[1] as string];
      // 3 worlds of nature adamant (marker A: the first move is hot) then 5 of hardy (marker B: the second is hot)
      const markerOf = (index: number): "A" | "B" => (index < 3 ? "A" : "B");
      const port = fakePort((index, opponent) => worldOf(opponent, { nature: index < 3 ? "adamant" : "hardy", variant: 0 }));
      // the defender's nature is the only thing that tells the worlds apart; the attacker is the player's own Pokemon
      const probe = stubProbe({ log: [] });
      const base = probe;
      const value = (query: ParsedMechanicsQuery) => {
        const q = query.query;
        if (q.kind !== "damage") return base(null as never, query);
        const marker = q.defender.set.includes("Adamant") ? "A" : "B";
        const roll = rollFor(marker, q.move, hot);
        return stubProbe({ values: { [q.move]: roll } })(null as never, query);
      };
      const actor = newActorV2({ worlds: port, probe: (_context, query) => value(query) });
      const choice = actor(c.request, c.context) as string;
      const state = stateOf(c);
      const foes = Object.keys(state.opponent.slots).sort().map((letter, position, all) => ({ loc: all.length - position, mon: state.opponent.roster[state.opponent.slots[letter] as number] as PublicMon })).filter((f) => !f.mon.fainted);
      const score = (candidate: string, world: number): number =>
        parseJointChoice(candidate).reduce((sum, part, slot) => {
          if (part.kind !== "move") return sum;
          const move = (active[slot]?.moves[part.index - 1] ?? { id: "" }) as { id: string; target?: string };
          if (move.target === undefined || category(move.id) === "Status") return sum;
          const targets = ["allAdjacentFoes", "allAdjacent", "randomNormal"].includes(move.target) ? foes : foes.filter((f) => f.loc === part.target);
          return sum + targets.length * rollsValue(Array(16).fill(rollFor(markerOf(world), move.id, hot)), 100, 1_000_000);
        }, 0);
      const choices = legalOf(c).choices;
      const mean = choices.map((candidate) => Array.from({ length: 8 }, (_, w) => score(candidate, w)).reduce((a, b) => a + b, 0));
      const first = choices.map((candidate) => score(candidate, 0));
      expect(choices.indexOf(choice), `request ${c.request.request_seq}`).toBe(mean.indexOf(Math.max(...mean)));
      if (mean.indexOf(Math.max(...mean)) !== first.indexOf(Math.max(...first))) decisive++;
    }
    // at least one request where the mean over the worlds and the first world alone disagree: the test has teeth
    expect(decisive).toBeGreaterThan(0);
  });

  it("scores a v2 request in one world exactly as v1 does on that world's sets", () => {
    // one world of the player's public facts; v1 is given the same sets through its assume seam
    const values = Object.fromEntries(moveRequests().flatMap((c) => (c.request.request as unknown as { active: { moves: { id: string }[] }[] }).active.flatMap((a) => a.moves.map((m, i) => [m.id, 5 + ((i * 17 + m.id.length) % 80)] as const))));
    let compared = 0;
    for (const c of moveRequests().slice(0, 10)) {
      const second = newActorV2({ config: { ...configV2, assumed_set: { ...configV2.assumed_set, worlds: 1 } }, worlds: fakePort((_i, opponent) => worldOf(opponent), { prior_hash: configV2.assumed_set.prior_hash }), probe: stubProbe({ values }) });
      const first = newActor({
        probe: stubProbe({ values }),
        assume: (mon) => {
          const set = slotOf(mon, 0).set as NonNullable<ReturnType<typeof slotOf>["set"]>;
          return buildFrom(mon, set);
        },
      });
      expect(second(c.request, c.context)).toBe(first(c.request, c.context));
      compared++;
    }
    expect(compared).toBeGreaterThan(5);
  });
});

import { buildWorldAssumedSet, type WorldSetLike } from "../src/damage-aware-assumed-set.ts";
const buildFrom = (mon: PublicMon, set: WorldSetLike) => buildWorldAssumedSet({ engine, vocabulary, format_id: config.format_id }, mon, set);

describe("refusals and fallbacks are counted and play the actor-rule candidate", () => {
  const request = (): Captured => moveRequests()[0] as Captured;

  it.each([
    ["reader_flagged"],
    ["disguise_assignments_exceeded"],
    ["legality_redraws_exhausted"],
    ["species_not_in_prior"],
  ])("a belief refusal %s", (code) => {
    const c = request();
    const actor = newActorV2({ worlds: fakePort(undefined, { sample: () => { throw refusal(code); } }), probe: stubProbe() });
    expect(actor(c.request, c.context)).toBe(actorRule(legalOf(c)));
    expect(actor(c.request, c.context)).toBe(actorRule(legalOf(c)));
    const report: DamageAwareReport = actor.report();
    expect(report.belief?.refusals).toEqual({ [code]: 2 });
    expect(report.belief).toMatchObject({ decisions: 2, worlds_sampled: 0 });
    expect(report.counts).toMatchObject({ decisions: 2, scored_decisions: 0 });
  });

  it.each([["format_mismatch"], ["item_clause_unsupported"], ["spread_sampler_required"]])("a defect of the run (%s) is thrown, never counted", (code) => {
    const c = request();
    const actor = newActorV2({ worlds: fakePort(undefined, { sample: () => { throw refusal(code); } }), probe: stubProbe() });
    expect(() => actor(c.request, c.context)).toThrow("refused");
  });

  it("a failure that is no belief refusal is thrown", () => {
    const c = request();
    const actor = newActorV2({ worlds: fakePort(undefined, { sample: () => { throw new TypeError("boom"); } }), probe: stubProbe() });
    expect(() => actor(c.request, c.context)).toThrow("boom");
  });

  it("counts the worlds' redraws", () => {
    const c = request();
    const inner = fakePort();
    const actor = newActorV2({ worlds: fakePort(undefined, { sample: (input) => ({ ...inner.sample(input), redraws: 3 }) }), probe: stubProbe() });
    actor(c.request, c.context);
    expect(actor.report().belief?.redraws).toBe(3);
  });

  it("drops a world that is another species than the Pokemon on the field and counts it; the others decide", () => {
    const c = request();
    const bad = (opponent: Parameters<typeof worldOf>[0]) => ({ slots: worldOf(opponent).slots.map((slot) => ({ ...slot, set: slot.set === null ? null : { ...slot.set, species: "notaspecies" } })) });
    const actor = newActorV2({ worlds: fakePort((index, opponent) => (index === 0 ? bad(opponent) : worldOf(opponent))), probe: stubProbe() });
    actor(c.request, c.context);
    expect(actor.report().belief).toMatchObject({ contradicted_worlds: 1, no_consistent_world: 0 });
    expect(actor.report().counts.scored_decisions).toBe(1);
  });

  it("plays the actor-rule candidate when no world is consistent with the public facts, and says so", () => {
    const c = request();
    const unbrought = (opponent: Parameters<typeof worldOf>[0]) => ({ slots: worldOf(opponent).slots.map((slot) => ({ ...slot, brought: false, set: null })) });
    const actor = newActorV2({ worlds: fakePort((_i, opponent) => unbrought(opponent)), probe: stubProbe() });
    expect(actor(c.request, c.context)).toBe(actorRule(legalOf(c)));
    expect(actor.report().belief).toMatchObject({ contradicted_worlds: 8, no_consistent_world: 1 });
    expect(actor.report().counts).toMatchObject({ scored_decisions: 0, decisions: 1 });
  });

  it("falls back on a probe refusal, as v1 does", () => {
    const c = request();
    const actor = newActorV2({ probe: stubProbe({ throwOn: { damage: new MechanicsProbeError("a state the engine will not take") } }) });
    expect(actor(c.request, c.context)).toBe(actorRule(legalOf(c)));
    expect(actor.report().counts.probe_refusals).toBe(1);
  });

  it("falls back, as v1 does, when a world's set is not a legal one", () => {
    const c = request();
    const illegal = (opponent: Parameters<typeof worldOf>[0]) => ({ slots: worldOf(opponent).slots.map((slot) => ({ ...slot, set: slot.set === null ? null : { ...slot.set, stat_points: { ...slot.set.stat_points, hp: 200 } } })) });
    const actor = newActorV2({ worlds: fakePort((_i, opponent) => illegal(opponent)), probe: stubProbe() });
    expect(actor(c.request, c.context)).toBe(actorRule(legalOf(c)));
    expect(actor.report().counts.assumed_set_fallbacks).toBe(1);
  });
});

describe("a foe with a status", () => {
  /** A request whose channel shows the first opposing active Pokemon paralysed. */
  const statused = (): { c: Captured; context: Captured["context"] } => {
    const c = moveRequests().find((r) => Object.values(stateOf(r).opponent.slots).some((i) => i !== null)) as Captured;
    const state = stateOf(c);
    const letter = Object.keys(state.opponent.slots).sort().find((l) => state.opponent.slots[l] !== null) as string;
    const mon = state.opponent.roster[state.opponent.slots[letter] as number] as PublicMon;
    const line = `|-status|p2${letter.slice(-1)}: ${mon.name ?? mon.base_species}|par`;
    return { c, context: { ...c.context, channel: [...c.context.channel.slice(0, -1), line, c.context.channel[c.context.channel.length - 1] as string] } };
  };

  it("drops a world whose ability blocks the status it is shown with", () => {
    const { c, context } = statused();
    expect(stateOf({ ...c, context }).opponent.roster.some((m) => m.status === "par")).toBe(true);
    // the stand-in engine refuses the status on worlds of nature adamant
    const probe = (_ctx: unknown, query: ParsedMechanicsQuery) => {
      if (query.query.kind === "damage" && query.query.defender.status === "par" && query.query.defender.set.includes("Adamant")) throw new MechanicsProbeError("the engine does not take that status");
      return stubProbe()(null as never, query);
    };
    const actor = newActorV2({ worlds: fakePort((index, opponent) => worldOf(opponent, { nature: index % 2 === 0 ? "adamant" : "hardy" })), probe: probe as never });
    actor(c.request, context);
    expect(actor.report().belief).toMatchObject({ contradicted_worlds: 4, no_consistent_world: 0 });
    expect(actor.report().counts.scored_decisions).toBe(1);
  });
});

describe("the reader's flags reach the belief", () => {
  it("passes every flagged line seen so far, so a later request is refused by the belief too", () => {
    const [first, second] = moveRequests().filter((c) => c.battle.id === (moveRequests()[0] as Captured).battle.id && c.request.phase === "move");
    const port = fakePort();
    const actor = newActorV2({ worlds: port, probe: stubProbe() });
    const a = first as Captured;
    const b = second as Captured;
    const bad = "|zzz-not-a-line|x";
    // the first request carries a line the reader cannot take: v1's own rule answers it without asking the belief
    expect(actor(a.request, { ...a.context, channel: [...a.context.channel, bad] })).toBe(actorRule(legalOf(a)));
    expect(port.calls).toHaveLength(0);
    expect(actor.report().counts.reader_fallbacks).toBe(1);
    // the next request has no new flagged line, and the belief hears of the old one
    actor(b.request, { ...b.context, channel: [...a.context.channel, bad, ...b.context.channel.slice(a.context.channel.length)] });
    expect(port.calls).toHaveLength(1);
    expect(port.calls[0]?.flags).toBe(1);
  });
});

describe("determinism", () => {
  it("gives the same choices, counts and identity from two runs", () => {
    const run = () => {
      const actor = newActorV2({ probe: stubProbe({ values: { tackle: 3 } }) });
      const choices = moveRequests().slice(0, 8).map((c) => actor(c.request, c.context));
      const report = actor.report();
      return { choices, report: { ...report, counts: { ...report.counts, probe_runs: 0 } } };
    };
    expect(run()).toEqual(run());
  });

  it("keeps the epsilon stream separate: it draws nothing from the world stream and needs the seat's seed", () => {
    expect(() => newActorV2({ config: { ...configV2, epsilon_ppm: 1000 } })).toThrow("seed");
    const c = moveRequests()[0] as Captured;
    const port = fakePort();
    const actor = newActorV2({ config: { ...configV2, epsilon_ppm: 1_000_000 }, seed: 7, worlds: port, probe: stubProbe() });
    actor(c.request, c.context);
    expect(actor.report().counts.epsilon_plays).toBe(1);
    expect(port.calls).toHaveLength(0);
  });
});
