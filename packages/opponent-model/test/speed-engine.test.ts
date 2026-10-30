// The speed-order evidence with the bridge's real `speed_order` probe over the committed battles (ticket #1133). The bridge is
// imported by path only for this test: the package itself is engine-free. The oracle reads the opponent's real team, never the belief.

import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { interpretChannel } from "@areneon/public-state";
import { beforeAll, describe, expect, it } from "vitest";
import { readBeliefConfig } from "../../showdown-bridge/src/belief-config.ts";
import { loadDamageAwareConfig } from "../../showdown-bridge/src/damage-aware-config.ts";
import type { Engine, PokemonSet } from "../../showdown-bridge/src/engine.ts";
import { GOLDEN_FORMAT } from "../../showdown-bridge/src/golden.ts";
import { loadEngine } from "../../showdown-bridge/src/index.ts";
import { loadMechanicsConfig } from "../../showdown-bridge/src/mechanics-config.ts";
import { type SpeedCheckContext, createSpeedCheck } from "../../showdown-bridge/src/speed-check.ts";
import { type Game, games, points, sound } from "./corpus.ts";
import { belief } from "../src/index.ts";
import { DIR, ROOT, realVocabulary, uniformPrior } from "./real.ts";
import type { UsagePrior } from "@areneon/schema/types/v1/usage-prior.ts";

const toId = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");
const beliefConfig = readBeliefConfig(JSON.parse(readFileSync(join(ROOT, "config", "opponent-model", GOLDEN_FORMAT, "belief.speed-v1.json"), "utf8")), realVocabulary);
const rules = beliefConfig.speed_evidence as NonNullable<typeof beliefConfig.speed_evidence>;

let engine: Engine;
beforeAll(() => {
  engine = loadEngine();
});

const context = (): SpeedCheckContext => ({ engine, vocabulary: realVocabulary, config: loadDamageAwareConfig("v1", realVocabulary), mechanics: loadMechanicsConfig(GOLDEN_FORMAT), belief: beliefConfig, data_class: "synthetic" });

/** The targeted fidelity corpus as games. */
function fidelity(): Game[] {
  return readdirSync(join(DIR, "search-fidelity")).filter((f) => /^b\d+\.json$/.test(f)).sort().map((f) => {
    const file = JSON.parse(readFileSync(join(DIR, "search-fidelity", f), "utf8")) as { teams: Game["teams"]; channels: Record<"p1" | "p2", { lines: string[] }> };
    return { id: `fidelity/${f.slice(0, -5)}`, teams: file.teams, channels: { p1: file.channels.p1.lines, p2: file.channels.p2.lines } };
  });
}

/** Every isolated order of the games, judged against the opponent's real set (the oracle). */
function sweep(list: readonly Game[]) {
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
      const port = createSpeedCheck(context(), { player: seat, lines, team: game.teams[seat] as unknown as PokemonSet[], run });
      if (port === null) continue;
      for (const [k, v] of Object.entries(port.counts().not_isolated)) reasons[k] = (reasons[k] ?? 0) + v;
      for (const event of port.events) {
        const set = last.truth.team[event.slot];
        if (set === undefined) continue;
        // What the belief does before it asks: a candidate with an unmodelled ability or item is not asked.
        if (rules.unmodelled.abilities.includes(toId(set.ability)) || rules.unmodelled.items.includes(toId(set.item))) continue;
        const pair = port.pairs.find((x) => `speed${x.own.frame}-${x.foe.frame}` === event.id);
        const shown = pair?.state.opponent.roster[event.slot]?.species ?? toId(set.species);
        // The belief asks about a stone holder as its Mega forme (`mega: true`), which the port does not state.
        const stone = realVocabulary.roster.some((r) => r.required_item === toId(set.item));
        const verdict = port.check(event, { species: toId(set.species), forme: realVocabulary.roster.some((r) => r.id === shown) ? shown : toId(set.species), mega: stone, ability: toId(set.ability), item: set.item === "" ? null : toId(set.item), nature: toId(set.nature), stat_points: set.evs as never });
        events += 1;
        if (verdict === null) unsupported += 1;
        else if (verdict) passed += 1;
        else failures.push(`${game.id}/${seat}/${event.id}/${event.role}/${set.species}`);
      }
    }
  }
  return { events, passed, unsupported, reasons, failures };
}

describe("the real check over the committed battles", () => {
  it("the real set of the opponent is never ruled out by an isolated order (the random-play, unavailable and single-battle corpora)", () => {
    const found = sweep(games());
    expect(found.failures).toEqual([]);
    expect(found.events).toBeGreaterThan(20);
    expect(found.passed).toBeGreaterThan(10);
  }, 600_000);

  it("... and over the 200 battles of the targeted fidelity corpus", () => {
    const found = sweep(fidelity());
    expect(found.failures).toEqual([]);
    expect(found.events).toBeGreaterThan(100);
    expect(found.passed).toBeGreaterThan(50);
    // each kind of non-isolated order that the reviewed rules name is found in the corpus
    expect(Object.keys(found.reasons)).toEqual(expect.arrayContaining(["priority", "changed_state"]));
  }, 900_000);
});

// ---- the belief ----

const FIELDS = ["hp", "atk", "def", "spa", "spd", "spe"] as const;
type Real = NonNullable<ReturnType<typeof points>[number]["truth"]>["team"][number];
const spreadKey = (set: Real): string => `${toId(set.nature)}:${FIELDS.map((k) => set.evs[k] ?? 0).join("-")}`;
/** The nature and Speed Stat Points of the real set beside decoys: a slow and a fast spread. */
const decoys = [["relaxed", [0, 0, 0, 0, 0, 0]], ["timid", [0, 0, 0, 0, 0, 32]], ["brave", [0, 0, 0, 0, 0, 0]], ["jolly", [0, 0, 0, 0, 0, 32]]] as const;

/** A prior that names the real spread of each preview slot beside decoys with no residual on spreads, built from the oracle's real team, which only a test may do. */
function decoyPrior(point: ReturnType<typeof points>[number] & { truth: NonNullable<ReturnType<typeof points>[number]["truth"]> }): UsagePrior {
  const base = uniformPrior(point.state.opponent.roster.map((m) => m.base_species));
  const species = base.species.map((entry) => {
    const set = point.truth.team[point.state.opponent.roster.findIndex((m) => m.base_species === entry.id)] as Real;
    const spreads = [{ nature: toId(set.nature), stat_points: Object.fromEntries(FIELDS.map((k) => [k, set.evs[k] ?? 0])) as never, weight: 1 }, ...decoys.filter(([n]) => n !== toId(set.nature)).map(([nature, v]) => ({ nature, stat_points: Object.fromEntries(FIELDS.map((k, i) => [k, v[i]])) as never, weight: 1 }))];
    return { ...entry, formes: entry.formes.map((f) => ({ ...f, spreads: { denominator: spreads.length, entries: spreads, residual: 0, unmapped: 0 } })) };
  });
  return { ...base, species: species as unknown as UsagePrior["species"] };
}

describe("the belief with the real check", () => {
  it("narrows the spreads of the slots whose order was seen and keeps the real set possible, at sampled requests of the committed battles", () => {
    let beliefs = 0;
    const total = { applied: 0, rejected: 0, contradictions: 0, unsupported: 0, events: 0 };
    for (const game of games({ random: 6 })) {
      for (const seat of ["p1", "p2"] as const) {
        const lines = game.channels[seat] as string[];
        const run = interpretChannel(seat, lines);
        const frames = run.frames.filter((f) => f.kind === "request");
        for (const point of points(game).filter((p, k) => p.seat === seat && p.truth !== null && k % 3 === 2)) {
          const frame = frames[point.request];
          if (frame === undefined) continue;
          const cut = { frames: run.frames.slice(0, frame.index + 1), flags: run.flags.filter((f) => f.index <= frame.index), requests: run.requests.slice(0, frame.request) };
          const port = createSpeedCheck(context(), { player: seat, lines: lines.slice(0, frame.index + 1), team: game.teams[seat] as unknown as PokemonSet[], run: cut });
          if (port === null) continue;
          const truthPoint = point as typeof point & { truth: NonNullable<typeof point.truth> };
          const input = { prior: decoyPrior(truthPoint), vocabulary: realVocabulary, config: beliefConfig, opponent: point.state.opponent, flags: point.flags };
          const plain = belief(input);
          const b = belief({ ...input, speed: port });
          expect(sound(plain, truthPoint).length, `plain ${game.id}/${seat}/${point.request}`).toBeGreaterThan(0);
          expect(sound(b, truthPoint).length, `${game.id}/${seat}/${point.request}`).toBeGreaterThan(0);
          b.slots.forEach((s, i) => expect(s.spreads.map((m) => m.id), `${game.id}/${seat}/${point.request} slot ${i}`).toContain(spreadKey(truthPoint.truth.team[i] as Real)));
          beliefs += 1;
          for (const k of Object.keys(total) as (keyof typeof total)[]) total[k] += b.speed[k];
        }
      }
    }
    expect(beliefs).toBeGreaterThan(30);
    expect(total.events).toBeGreaterThan(0);
    expect(total.rejected).toBeGreaterThan(0);
  }, 900_000);
});
