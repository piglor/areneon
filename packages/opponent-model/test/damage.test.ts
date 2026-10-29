// The damage evidence with a stand-in port, on numbers worked out on paper (ticket #1132, ADR-025 section 7). The
// port's verdicts are scripted here; the real probe is in `damage-engine.test.ts`.

import type { BeliefConfig } from "@areneon/schema/types/v1/belief-config.ts";
import { describe, expect, it } from "vitest";
import { type DamageCandidate, type DamageCheck, type DamageEvent, belief } from "../src/index.ts";
import { type Ratio } from "../src/ratio.ts";
import { alpha, config, input, prior, text } from "./support.ts";

const evidence = (change: Partial<NonNullable<BeliefConfig["damage_evidence"]>> = {}): BeliefConfig => ({
  ...config,
  damage_evidence: { max_checked: 100, shown_hp_rounding: "floor", non_isolating: { abilities: [], lasting_moves: [], moves: [], targets: [] }, unmodelled: { abilities: [], items: [] }, ...change },
});

const spread = (nature: string, weight: number) => ({ nature, stat_points: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 }, weight });
/** Alpha with two named spreads (6 and 4 of 10), abilities a1 3/4 and a2 1/4, items ix 5, iy 2 and none 3 of 10. */
const twoSpreads = () => prior((p) => {
  p.species[0] = { id: "alpha", denominator: 1, formes: [alpha((f) => { f.spreads = { denominator: 10, entries: [spread("hardy", 6), spread("bold", 4)], residual: 0, unmapped: 0 }; })] };
});

interface Scripted extends DamageCheck {
  readonly asked: DamageCandidate[];
}
/** A port with the given events whose verdict is `verdict(candidate)`. */
function port(events: readonly DamageEvent[], verdict: (c: DamageCandidate, e: DamageEvent) => boolean | null): Scripted {
  const asked: DamageCandidate[] = [];
  return { events, asked, check: (e, c) => { asked.push(c); return verdict(c, e); } };
}
const hit = (id = "h1", slot = 0, role: DamageEvent["role"] = "defender"): DamageEvent => ({ id, slot, role });
const fraction = (r: Ratio): string => `${r.num}/${r.den}`;
const run = (extra: Partial<Parameters<typeof input>[1]>, cfg: BeliefConfig = evidence()) => belief({ ...input([], { prior: twoSpreads() }), config: cfg, ...extra });
/** The slots without their fixed-word notes (the notes name the evidence that was applied or left out). */
const core = (b: ReturnType<typeof belief>): string => text(b.slots.map(({ constraints: _, ...rest }) => rest));

describe("an isolated hit reweighs the fields it passes through", () => {
  // Only (a1, ix, bold) fails: a mass of 3/4 * 1/2 * 2/5 = 3/20 of the forme.
  const failing = (c: DamageCandidate): boolean => !(c.ability === "a1" && c.item === "ix" && c.nature === "bold");

  it("each marginal is the share of the other fields' mass that passes with it", () => {
    const b = run({ damage: port([hit()], failing) });
    const forme = b.assignments[0]?.slots[0]?.formes[0];
    // spreads: hardy 6/10 * 1 and bold 4/10 * (1 - 3/4 * 1/2) = 4/10 * 5/8 = 1/4, so 12 : 5
    const sp = new Map(forme?.spreads.options.map((o) => [o.id, o.weight]));
    expect((sp.get("hardy:0-0-0-0-0-0") as bigint) * 5n).toBe((sp.get("bold:0-0-0-0-0-0") as bigint) * 12n);
    // abilities: a1 3/4 * (1 - 1/2 * 2/5) = 3/5 and a2 1/4, so 12 : 5
    const ab = new Map(forme?.abilities.options.map((o) => [o.id, o.weight]));
    expect((ab.get("a1") as bigint) * 5n).toBe((ab.get("a2") as bigint) * 12n);
    // items: ix 1/2 * (1 - 3/4 * 2/5) = 7/20, iy 1/5, none 3/10, so 7 : 4 : 6 in the cells
    const w = new Map((b.assignments[0]?.slots[0]?.cells ?? []).map((c) => [c.item, c.weight]));
    expect((w.get("ix") as bigint) * 4n).toBe((w.get("iy") as bigint) * 7n);
    expect((w.get(null) as bigint) * 4n).toBe((w.get("iy") as bigint) * 6n);
    expect(b.damage).toEqual({ enabled: true, events: 1, applied: 1, checked: 12, rejected: 1, unsupported: 0, unchecked: 0, contradictions: 0 });
    expect(b.slots[0]?.constraints).toContain("damage_applied:defender");
  });

  it("an option whose every combination fails disappears, and the others keep a positive weight", () => {
    // Every candidate with the item ix fails.
    const b = run({ damage: port([hit()], (c) => c.item !== "ix") });
    expect(b.slots[0]?.items.map((m) => m.id)).toEqual([null, "iy"]);
    expect(b.slots[0]?.abilities.map((m) => m.id)).toEqual(["a1", "a2"]);
    expect(b.slots[0]?.spreads).toHaveLength(2);
  });

  it("a candidate the probe cannot answer, or that the config calls unmodelled, constrains nothing", () => {
    const without = run({});
    const unsupported = port([hit()], () => null);
    expect(core(run({ damage: unsupported }))).toBe(core(without));
    const unmodelled = port([hit()], () => false);
    const b = run({ damage: unmodelled }, evidence({ unmodelled: { abilities: ["a1", "a2"], items: [] } }));
    expect(unmodelled.asked).toHaveLength(0);
    expect(core(b)).toBe(core(without));
    expect(b.damage.unsupported).toBeGreaterThan(0);
  });

  it("an unmodelled item is not asked either", () => {
    const p = port([hit()], () => true);
    run({ damage: p }, evidence({ unmodelled: { abilities: [], items: ["ix"] } }));
    expect(p.asked.some((c) => c.item === "ix")).toBe(false);
    expect(p.asked.some((c) => c.item === "iy")).toBe(true);
  });

  it("only the most likely candidates are asked, most likely first", () => {
    const p = port([hit()], () => true);
    const b = run({ damage: p }, evidence({ max_checked: 2 }));
    // the two most likely of the forme: (a1, ix, hardy) = 3 * 5 * 6 = 90 and (a1, ix, bold) = 3 * 5 * 4 = 60
    expect(p.asked.map((c) => [c.ability, c.item, c.nature])).toEqual([["a1", "ix", "hardy"], ["a1", "ix", "bold"]]);
    expect(b.damage.checked).toBe(2);
  });
});

describe("what the evidence cannot do", () => {
  it("an event whose checked candidates are all ruled out is dropped and counted, never emptying the belief", () => {
    const without = run({});
    const b = run({ damage: port([hit()], () => false) });
    expect(core(b)).toBe(core(without));
    expect(b.damage.contradictions).toBe(1);
    expect(b.damage.applied).toBe(0);
    expect(b.slots[0]?.constraints).toContain("damage_contradiction:defender");
  });

  it("an event with nothing to ask (no named spread) is counted as unchecked and changes nothing", () => {
    const other = prior((p) => {
      p.species[0] = { id: "alpha", denominator: 1, formes: [alpha((f) => { f.spreads = { denominator: 10, entries: [], residual: 10, unmapped: 0 }; })] };
    });
    const base = input([], { prior: other });
    const without = belief({ ...base, config: evidence() });
    const b = belief({ ...base, config: evidence(), damage: port([hit()], () => false) });
    expect(core(b)).toBe(core(without));
    expect(b.damage.unchecked).toBe(1);
  });

  it("is off without the config field, without a port, and under a disguise mixture", () => {
    const base = input([], { prior: twoSpreads() });
    const p = port([hit()], () => false);
    expect(belief({ ...base, damage: p }).damage).toEqual({ enabled: false, events: 0, applied: 0, checked: 0, rejected: 0, unsupported: 0, unchecked: 0, contradictions: 0 });
    expect(p.asked).toHaveLength(0);
    expect(belief({ ...base, config: evidence() }).damage.enabled).toBe(false);
    expect(belief({ ...base, config: evidence(), damage: null }).damage.enabled).toBe(false);
    const hidden = belief({ ...base, config: { ...evidence(), identity_hiding_abilities: ["ill"] }, damage: p, opponent: input(["|switch|p2a: Alpha|Alpha, L50, M|100/100"], { prior: twoSpreads() }).opponent });
    expect(hidden.damage.enabled).toBe(false);
    expect(p.asked).toHaveLength(0);
  });

  it("events about other slots are not asked of this one", () => {
    const p = port([hit("h", 2)], () => false);
    const b = run({ damage: p });
    expect(p.asked.every((c) => c.species !== "alpha")).toBe(true);
    expect(b.damage.events).toBe(1);
  });

  it("a forme that none of the failures touches keeps its numbers", () => {
    // beta (slot 1) has the formes beta and betamega; only beta's candidates with the ability b1 fail.
    const without = run({});
    const b = run({ damage: port([hit("h1", 1)], (c) => !(c.forme === "beta" && c.ability === "b1")) });
    const slot = (x: typeof b) => x.assignments[0]?.slots[1];
    expect(slot(b)?.formes.map((f) => f.id)).toEqual(["beta", "betamega"]);
    expect(text(slot(b)?.formes.find((f) => f.id === "betamega")?.abilities)).toBe(text(slot(without)?.formes.find((f) => f.id === "betamega")?.abilities));
    expect(slot(b)?.abilities.map((m) => m.id)).toContain("b2");
    expect(b.damage.applied).toBe(1);
  });

  it("two events on a slot apply one after the other", () => {
    const first = run({ damage: port([hit("h1")], (c) => c.item !== "ix") });
    const both = run({ damage: port([hit("h1"), hit("h2")], (c, e) => (e.id === "h1" ? c.item !== "ix" : c.nature !== "bold")) });
    expect(both.damage.applied).toBe(2);
    expect(both.slots[0]?.spreads.map((m) => m.id)).toEqual(["hardy:0-0-0-0-0-0"]);
    expect(both.slots[0]?.items.map((m) => m.id)).toEqual(first.slots[0]?.items.map((m) => m.id));
  });

  it("is a pure function of the port's verdicts: the same answers give the same bytes", () => {
    const verdict = (c: DamageCandidate): boolean => c.item !== "iy";
    const a = run({ damage: port([hit()], verdict) });
    const b = run({ damage: port([hit()], verdict) });
    expect(text(a)).toBe(text(b));
    expect(fraction(a.slots[0]?.items[0]?.p as Ratio)).toBe(fraction(b.slots[0]?.items[0]?.p as Ratio));
  });
});
