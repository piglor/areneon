// The speed-order evidence with a stand-in port, on numbers worked out on paper (ticket #1133, ADR-025 section 7). The port's
// verdicts are scripted here; the real probe is in `speed-engine.test.ts`. The evidence shares its implementation with the
// damage evidence (`applyEvidence`), whose marginals `damage.test.ts` works out; this file holds what is the speed evidence's own.

import type { BeliefConfig } from "@areneon/schema/types/v1/belief-config.ts";
import { describe, expect, it } from "vitest";
import { type DamageCandidate, type SpeedCheck, type SpeedEvent, applySpeed, belief } from "../src/index.ts";
import { alpha, config, input, prior, text } from "./support.ts";

const evidence = (change: Partial<NonNullable<BeliefConfig["speed_evidence"]>> = {}): BeliefConfig => ({
  ...config,
  speed_evidence: { max_checked: 100, non_isolating: { abilities: [], lasting_moves: [], moves: [] }, unmodelled: { abilities: [], items: [] }, ...change },
});

const spread = (nature: string, weight: number) => ({ nature, stat_points: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 }, weight });
/** Alpha with two named spreads (6 and 4 of 10), abilities a1 3/4 and a2 1/4, items ix 5, iy 2 and none 3 of 10. */
const twoSpreads = () => prior((p) => {
  p.species[0] = { id: "alpha", denominator: 1, formes: [alpha((f) => { f.spreads = { denominator: 10, entries: [spread("hardy", 6), spread("bold", 4)], residual: 0, unmapped: 0 }; })] };
});

interface Scripted extends SpeedCheck {
  readonly asked: DamageCandidate[];
}
function port(events: readonly SpeedEvent[], verdict: (c: DamageCandidate, e: SpeedEvent) => boolean | null): Scripted {
  const asked: DamageCandidate[] = [];
  return { events, asked, check: (e, c) => { asked.push(c); return verdict(c, e); } };
}
const order = (id = "s1", slot = 0, role: SpeedEvent["role"] = "first"): SpeedEvent => ({ id, slot, role });
const run = (extra: Partial<Parameters<typeof input>[1]>, cfg: BeliefConfig = evidence()) => belief({ ...input([], { prior: twoSpreads() }), config: cfg, ...extra });
const core = (b: ReturnType<typeof belief>): string => text(b.slots.map(({ constraints: _, ...rest }) => rest));
const zero = { enabled: false, events: 0, applied: 0, checked: 0, rejected: 0, unsupported: 0, unchecked: 0, contradictions: 0 };

describe("an observed order reweighs the fields it passes through", () => {
  // The slow nature (bold) fails whatever the ability and item: the order says the Pokemon outspeeds the player's.
  const fast = (c: DamageCandidate): boolean => c.nature !== "bold";

  it("drops the failing spread, keeps the rest as they were, and names the evidence in the notes", () => {
    const b = run({ speed: port([order()], fast) });
    expect(b.slots[0]?.spreads.map((m) => m.id)).toEqual(["hardy:0-0-0-0-0-0"]);
    // abilities and items are untouched: a failing candidate is failing for every ability and item alike
    const plain = run({});
    expect(text(b.slots[0]?.abilities.map((m) => m.id))).toBe(text(plain.slots[0]?.abilities.map((m) => m.id)));
    expect(b.speed).toEqual({ enabled: true, events: 1, applied: 1, checked: 12, rejected: 6, unsupported: 0, unchecked: 0, contradictions: 0 });
    expect(b.slots[0]?.constraints).toContain("speed_applied:first");
    expect(b.damage).toEqual(zero);
  });

  it("a candidate the probe cannot answer, or that the config calls unmodelled, constrains nothing", () => {
    const without = run({});
    expect(core(run({ speed: port([order()], () => null) }))).toBe(core(without));
    const unmodelled = port([order()], () => false);
    const b = run({ speed: unmodelled }, evidence({ unmodelled: { abilities: ["a1", "a2"], items: [] } }));
    expect(unmodelled.asked).toHaveLength(0);
    expect(core(b)).toBe(core(without));
    expect(b.speed.unsupported).toBeGreaterThan(0);
  });

  it("only the most likely candidates are asked, most likely first", () => {
    const p = port([order()], () => true);
    run({ speed: p }, evidence({ max_checked: 2 }));
    expect(p.asked.map((c) => [c.ability, c.item, c.nature])).toEqual([["a1", "ix", "hardy"], ["a1", "ix", "bold"]]);
  });

  it("an order that rules out every checked candidate is dropped and counted, never emptying the belief", () => {
    const b = run({ speed: port([order("s1", 0, "second")], () => false) });
    expect(core(b)).toBe(core(run({})));
    expect(b.speed.contradictions).toBe(1);
    expect(b.slots[0]?.constraints).toContain("speed_contradiction:second");
  });

  it("an order with nothing to ask (no named spread) is counted as unchecked and changes nothing", () => {
    const other = prior((p) => {
      p.species[0] = { id: "alpha", denominator: 1, formes: [alpha((f) => { f.spreads = { denominator: 10, entries: [], residual: 10, unmapped: 0 }; })] };
    });
    const base = input([], { prior: other });
    const b = belief({ ...base, config: evidence(), speed: port([order()], () => false) });
    expect(core(b)).toBe(core(belief({ ...base, config: evidence() })));
    expect(b.speed.unchecked).toBe(1);
    expect(b.slots[0]?.constraints).toContain("speed_unchecked:first");
  });
});

describe("the damage and the speed evidence together", () => {
  const both = (): BeliefConfig => ({ ...evidence(), damage_evidence: { max_checked: 100, shown_hp_rounding: "floor", non_isolating: { abilities: [], lasting_moves: [], moves: [], targets: [] }, unmodelled: { abilities: [], items: [] } } });

  it("each reads its own port and counts alone, the damage first", () => {
    const b = run({ damage: { events: [{ id: "h1", slot: 0, role: "defender" }], check: (_, c) => c.item !== "ix" }, speed: port([order()], (c) => c.nature !== "bold") }, both());
    expect(b.damage).toMatchObject({ enabled: true, events: 1, applied: 1 });
    expect(b.speed).toMatchObject({ enabled: true, events: 1, applied: 1 });
    expect(b.slots[0]?.spreads.map((m) => m.id)).toEqual(["hardy:0-0-0-0-0-0"]);
    expect(b.slots[0]?.items.map((m) => m.id)).not.toContain("ix");
    const notes = b.slots[0]?.constraints ?? [];
    expect(notes.indexOf("damage_applied:defender")).toBeLessThan(notes.indexOf("speed_applied:first"));
  });
});

describe("what the evidence is off for", () => {
  it("is off without the config field, without a port, and under a disguise mixture", () => {
    const base = input([], { prior: twoSpreads() });
    const p = port([order()], () => false);
    expect(belief({ ...base, speed: p }).speed).toEqual(zero);
    expect(p.asked).toHaveLength(0);
    expect(belief({ ...base, config: evidence() }).speed.enabled).toBe(false);
    expect(belief({ ...base, config: evidence(), speed: null }).speed.enabled).toBe(false);
    const hidden = belief({ ...base, config: { ...evidence(), identity_hiding_abilities: ["ill"] }, speed: p, opponent: input(["|switch|p2a: Alpha|Alpha, L50, M|100/100"], { prior: twoSpreads() }).opponent });
    expect(hidden.speed.enabled).toBe(false);
    expect(p.asked).toHaveLength(0);
  });

  it("events about other slots are not asked of this one", () => {
    const p = port([order("s", 2)], () => false);
    const b = run({ speed: p });
    expect(p.asked.every((c) => c.species !== "alpha")).toBe(true);
    expect(b.speed.events).toBe(1);
  });

  it("applySpeed never hands back a slot with no forme (a defect of the caller, not a belief)", () => {
    expect(() => applySpeed([], "alpha", [], port([], () => true), { max_checked: 1, unmodelled: { abilities: [], items: [] } }, { ...zero })).toThrow(RangeError);
  });
});
