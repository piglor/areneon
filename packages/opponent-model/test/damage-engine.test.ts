// The damage evidence with the bridge's real probe over the committed battles (ticket #1132). The bridge is imported
// by path only for this test: the package itself is engine-free. The oracle reads the opponent's real team, never the belief.

import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { interpretChannel } from "@areneon/public-state";
import { beforeAll, describe, expect, it } from "vitest";
import { loadDamageAwareConfig } from "../../showdown-bridge/src/damage-aware-config.ts";
import { readBeliefConfig } from "../../showdown-bridge/src/belief-config.ts";
import { createDamageCheck } from "../../showdown-bridge/src/damage-check.ts";
import { GOLDEN_FORMAT } from "../../showdown-bridge/src/golden.ts";
import { loadEngine } from "../../showdown-bridge/src/index.ts";
import type { Engine, PokemonSet } from "../../showdown-bridge/src/engine.ts";
import { loadMechanicsConfig } from "../../showdown-bridge/src/mechanics-config.ts";
import type { UsagePrior } from "@areneon/schema/types/v1/usage-prior.ts";
import { type Game, type Point, type Set6, games, points, sound } from "./corpus.ts";
import { belief } from "../src/index.ts";
import { DIR, ROOT, realVocabulary, uniformPrior } from "./real.ts";
import { text } from "./support.ts";

const toId = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");
const beliefConfig = readBeliefConfig(JSON.parse(readFileSync(join(ROOT, "config", "opponent-model", GOLDEN_FORMAT, "belief.damage-v1.json"), "utf8")), realVocabulary);
const rules = beliefConfig.damage_evidence as NonNullable<typeof beliefConfig.damage_evidence>;

let engine: Engine;
beforeAll(() => {
  engine = loadEngine();
});

function context(rounding?: "floor" | "ceil" | "nearest") {
  const belief = rounding === undefined ? beliefConfig : { ...beliefConfig, damage_evidence: { ...rules, shown_hp_rounding: rounding } };
  return { engine, vocabulary: realVocabulary, config: loadDamageAwareConfig("v1", realVocabulary), mechanics: loadMechanicsConfig(GOLDEN_FORMAT), belief, data_class: "synthetic" as const };
}

describe("the real check over the committed battles", () => {
  /** Every isolated event of the committed battles, judged against the opponent's real set (the oracle). */
  function sweep(list: readonly Game[], rounding?: "floor" | "ceil" | "nearest") {
    let events = 0;
    let passed = 0;
    let unsupported = 0;
    const reasons: Record<string, number> = {};
    const failures: string[] = [];
    for (const game of list) {
      for (const seat of ["p1", "p2"] as const) {
        const lines = game.channels[seat] as string[];
        const run = interpretChannel(seat, lines);
        const last = points(game).filter((p) => p.seat === seat && p.truth !== null).at(-1);
        if (last === undefined || last.truth === null) continue;
        const port = createDamageCheck(context(rounding), { player: seat, lines, team: game.teams[seat] as unknown as PokemonSet[], run });
        if (port === null) continue;
        for (const [k, v] of Object.entries(port.counts().not_isolated)) reasons[k] = (reasons[k] ?? 0) + v;
        for (const event of port.events) {
          const set = last.truth.team[event.slot];
          if (set === undefined) continue;
          const nature = toId(set.nature);
          // What the belief does before it asks: a candidate with an unmodelled ability or item is not asked.
          if (rules.unmodelled.abilities.includes(toId(set.ability)) || rules.unmodelled.items.includes(toId(set.item))) continue;
          const shown = port.hits.find((x) => `hit${x.frame}` === event.id)?.state.opponent.roster[event.slot]?.species ?? toId(set.species);
          // The belief asks about a stone holder as its Mega forme (`mega: true`), which the port does not state.
          const stone = realVocabulary.roster.some((r) => r.required_item === toId(set.item));
          const verdict = port.check(event, { species: toId(set.species), forme: realVocabulary.roster.some((r) => r.id === shown) ? shown : toId(set.species), mega: stone, ability: toId(set.ability), item: set.item === "" ? null : toId(set.item), nature, stat_points: set.evs as never });
          events += 1;
          if (verdict === null) unsupported += 1;
          else if (verdict) passed += 1;
          else failures.push(`${game.id}/${seat}/${event.id}/${event.role}/${set.species}`);
        }
      }
    }
    return { events, passed, unsupported, reasons, failures };
  }

  it("the real set of the opponent is never ruled out by an isolated hit it took or dealt (the random-play, unavailable and single-battle corpora)", () => {
    const found = sweep(games());
    expect(found.failures).toEqual([]);
    expect(found.events).toBeGreaterThan(100);
    expect(found.passed).toBeGreaterThan(100);
  }, 600_000);

  it("... and over the 200 battles of the targeted fidelity corpus (1,436 events at this pin, 426 of them unsupported)", () => {
    const found = sweep(fidelity());
    expect(found.failures).toEqual([]);
    expect(found.events).toBeGreaterThan(1000);
    expect(found.passed).toBeGreaterThan(900);
    // each kind of non-isolated hit is found in the corpus
    expect(Object.keys(found.reasons)).toEqual(expect.arrayContaining(["listed_move", "lasting_move", "listed_ability", "listed_target", "possible_disguise", "volatile", "changed_mon"]));
  }, 900_000);

  it("negative control: a wrong display rule rules the real set out somewhere", () => {
    expect(sweep(games(), "ceil").failures.length + sweep(games(), "nearest").failures.length).toBeGreaterThan(0);
  }, 600_000);
});

/** The targeted fidelity corpus as games. */
function fidelity(): Game[] {
  return readdirSync(join(DIR, "search-fidelity")).filter((f) => /^b\d+\.json$/.test(f)).sort().map((f) => {
    const file = JSON.parse(readFileSync(join(DIR, "search-fidelity", f), "utf8")) as { teams: Game["teams"]; channels: Record<"p1" | "p2", { lines: string[] }> };
    return { id: `fidelity/${f.slice(0, -5)}`, teams: file.teams, channels: { p1: file.channels.p1.lines, p2: file.channels.p2.lines } };
  });
}

// ---- the belief ----

const FIELDS = ["hp", "atk", "def", "spa", "spd", "spe"] as const;
const spreadKey = (set: Set6): string => `${toId(set.nature)}:${FIELDS.map((k) => set.evs[k] ?? 0).join("-")}`;
const decoySpreads = [["modest", [0, 0, 0, 32, 0, 2]], ["bold", [32, 0, 32, 0, 0, 2]], ["calm", [32, 0, 0, 0, 32, 2]], ["adamant", [0, 32, 0, 0, 0, 32]], ["timid", [0, 0, 0, 32, 0, 32]], ["hardy", [32, 0, 0, 0, 0, 0]]] as const;
const decoyItems = ["lifeorb", "expertbelt", "leftovers", "sitrusberry", "focussash", "charcoal"];

/**
 * A prior for a point that names the real spread and item of each preview slot beside decoys, with no residual on spreads or
 * items, so the evidence has candidates to narrow. It is built from the oracle's real team, which only a test may do: the
 * production belief is never given such a prior, it needs one that is not derived from the opponent.
 */
function decoyPrior(p: Point & { truth: NonNullable<Point["truth"]> }): UsagePrior {
  const base = uniformPrior(p.state.opponent.roster.map((m) => m.base_species));
  const species = base.species.map((entry) => {
    // the prior's species are sorted by id, the roster and the real team are in preview order
    const set = p.truth.team[p.state.opponent.roster.findIndex((m) => m.base_species === entry.id)] as Set6;
    const spreads = [{ nature: toId(set.nature), stat_points: Object.fromEntries(FIELDS.map((k) => [k, set.evs[k] ?? 0])) as never, weight: 1 }, ...decoySpreads.filter(([n]) => n !== toId(set.nature)).map(([nature, v]) => ({ nature, stat_points: Object.fromEntries(FIELDS.map((k, i) => [k, v[i]])) as never, weight: 1 }))];
    const items = [...new Set([toId(set.item), ...decoyItems].filter((i) => i !== ""))];
    const hasNone = toId(set.item) === "";
    return {
      ...entry,
      formes: entry.formes.map((f) => ({ ...f, spreads: { denominator: spreads.length, entries: spreads, residual: 0, unmapped: 0 }, items: { denominator: items.length + 1, entries: items.map((id) => ({ id, weight: 1 })), no_item: hasNone ? 2 : 1, residual: 0, unmapped: 0 } })),
    };
  });
  return { ...base, species: species as unknown as UsagePrior["species"] };
}

/** The lines and reader frames of a seat's channel up to and including its n-th request, as the port reads them. */
function upTo(game: Game, seat: "p1" | "p2", ordinal: number) {
  const lines = game.channels[seat] as string[];
  const run = interpretChannel(seat, lines);
  const frame = run.frames.filter((f) => f.kind === "request")[ordinal];
  if (frame === undefined) throw new Error("no such request");
  const cut = { frames: run.frames.slice(0, frame.index + 1), flags: run.flags.filter((f) => f.index <= frame.index), requests: run.requests.slice(0, frame.request) };
  return { lines: lines.slice(0, frame.index + 1), run: cut };
}

describe("the belief with the real check", () => {
  const wide = { ...beliefConfig, damage_evidence: { ...rules, max_checked: 4096 } };

  it("narrows the items and spreads of the slots that took or dealt isolated hits in the golden battle, and keeps the real set", () => {
    const game = games({ random: 0 })[0] as Game;
    const seat = "p2" as const;
    const point = points(game).filter((p) => p.seat === seat && p.truth !== null).at(-1) as Point & { truth: NonNullable<Point["truth"]> };
    const ordinal = point.request;
    const { lines, run } = upTo(game, seat, ordinal);
    const port = createDamageCheck(context(), { player: seat, lines, team: game.teams[seat] as unknown as PokemonSet[], run });
    expect(port?.events.length).toBeGreaterThan(5);
    const prior = decoyPrior(point);
    const base = { prior, vocabulary: realVocabulary, config: wide, opponent: point.state.opponent, flags: point.flags };
    const without = belief(base);
    const withDamage = belief({ ...base, damage: port });
    expect(without.damage.enabled).toBe(false);
    expect(withDamage.damage).toMatchObject({ enabled: true, contradictions: 0 });
    expect(withDamage.damage.applied).toBeGreaterThan(3);
    expect(withDamage.damage.rejected).toBeGreaterThan(30);
    // narrowed: some named item or spread is gone from a slot, and many weights moved
    const gone = withDamage.slots.flatMap((s, i) => [...without.slots[i]!.items.filter((m) => !s.items.some((n) => n.id === m.id)), ...without.slots[i]!.spreads.filter((m) => !s.spreads.some((n) => n.id === m.id))]);
    expect(gone.length).toBeGreaterThan(0);
    const moved = withDamage.slots.filter((s, i) => text(s.spreads) !== text(without.slots[i]!.spreads)).length;
    expect(moved).toBeGreaterThan(2);
    // and the real set is still possible in every slot: forme, ability, item, moves (the oracle) and the spread
    expect(sound(withDamage, point).length).toBeGreaterThan(0);
    withDamage.slots.forEach((s, i) => expect(s.spreads.map((m) => m.id), `slot ${i}`).toContain(spreadKey(point.truth.team[i] as Set6)));
  }, 300_000);

  it("keeps the real set possible at sampled requests of the committed battles (apart from dropped contradictions, which are counted)", () => {
    let beliefs = 0;
    const total = { applied: 0, rejected: 0, contradictions: 0, unsupported: 0 };
    for (const game of games({ random: 6 })) {
      for (const seat of ["p1", "p2"] as const) {
        const pts = points(game).filter((p) => p.seat === seat);
        for (const point of pts.filter((p, k) => p.truth !== null && k % 3 === 2)) {
          const { lines, run } = upTo(game, seat, point.request);
          const port = createDamageCheck(context(), { player: seat, lines, team: game.teams[seat] as unknown as PokemonSet[], run });
          if (port === null) continue;
          const truthPoint = point as Point & { truth: NonNullable<Point["truth"]> };
          const b = belief({ prior: decoyPrior(truthPoint), vocabulary: realVocabulary, config: beliefConfig, opponent: point.state.opponent, flags: point.flags, damage: port });
          const plain = belief({ prior: decoyPrior(truthPoint), vocabulary: realVocabulary, config: beliefConfig, opponent: point.state.opponent, flags: point.flags });
          expect(sound(plain, truthPoint).length, `plain ${game.id}/${seat}/${point.request}`).toBeGreaterThan(0);
          beliefs += 1;
          expect(sound(b, truthPoint).length, `${game.id}/${seat}/${point.request}`).toBeGreaterThan(0);
          b.slots.forEach((s, i) => expect(s.spreads.map((m) => m.id), `${game.id}/${seat}/${point.request} slot ${i}`).toContain(spreadKey(truthPoint.truth.team[i] as Set6)));
          for (const k of Object.keys(total) as (keyof typeof total)[]) total[k] += b.damage[k];
        }
      }
    }
    expect(beliefs).toBeGreaterThan(30);
    expect(total.applied).toBeGreaterThan(10);
    expect(total.rejected).toBeGreaterThan(100);
  }, 900_000);
});

describe("a non-isolated hit adds nothing (a real turn with the modifier put in)", () => {
  it("a Helping Hand before the golden battle's Flamethrower makes the hit non-isolated and removes its event", () => {
    const game = games({ random: 0 })[0] as Game;
    const lines = [...(game.channels.p2 as string[])];
    const at = lines.indexOf("|move|p1b: Charizard|Flamethrower|p2a: Blastoise");
    const make = (extra: string[]) => createDamageCheck(context(), { player: "p2", lines: [...lines.slice(0, at), ...extra, ...lines.slice(at)], team: game.teams.p2 as unknown as PokemonSet[] }) as NonNullable<ReturnType<typeof createDamageCheck>>;
    const plain = make([]);
    const helped = make(["|move|p2b: Charizard|Helping Hand|p2a: Blastoise"]);
    const flame = (port: typeof plain) => port.hits.find((h) => h.move === "flamethrower" && h.frame >= at);
    expect(flame(plain)?.reason).toBeNull();
    expect(flame(helped)?.reason).toBe("listed_move");
    expect(helped.events.length).toBe(plain.events.length - 1);
  }, 120_000);
});
