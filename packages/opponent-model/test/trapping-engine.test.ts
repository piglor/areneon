// The trapping constraint with the bridge's real engine check (ticket #1127): over the committed battles the flags
// the requests showed never exclude what the opponent really held, and a species outside the format's roster that
// lists a trapping ability shows how the flags tell its abilities apart. The bridge is imported by path only for this
// test: the package itself is engine-free.

import { interpretChannel } from "@areneon/public-state";
import { beforeAll, describe, expect, it } from "vitest";
import { GOLDEN_FORMAT } from "../../showdown-bridge/src/golden.ts";
import { loadEngine } from "../../showdown-bridge/src/index.ts";
import type { Engine, PokemonSet } from "../../showdown-bridge/src/engine.ts";
import { loadMechanicsConfig } from "../../showdown-bridge/src/mechanics-config.ts";
import { checkScenario } from "../../showdown-bridge/src/mechanics-scenario.ts";
import { createTrappingCheck } from "../../showdown-bridge/src/trapping-check.ts";
import { cases } from "../../showdown-bridge/test/world-builder-support.ts";
import { BeliefError, belief } from "../src/index.ts";
import { games, sound, points } from "./corpus.ts";
import { realVocabulary, uniformPrior } from "./real.ts";
import { config, text } from "./support.ts";

let engine: Engine;
let scenario_id: string;
beforeAll(() => {
  engine = loadEngine();
  scenario_id = checkScenario(engine, loadMechanicsConfig(GOLDEN_FORMAT));
});

describe("the real check over the committed battles", () => {
  it("the flags of every request leave the real sets possible and exclude nothing the engine does not exclude", () => {
    let checked = 0;
    let informative = 0;
    for (const game of games({ random: 12 })) {
      for (const seat of ["p1", "p2"] as const) {
        const run = interpretChannel(seat, game.channels[seat] as string[]);
        const pts = points({ ...game, channels: { ...game.channels } }).filter((p) => p.seat === seat);
        let n = 0;
        for (const frame of run.frames) {
          if (frame.kind !== "request") continue;
          const point = pts[n];
          n += 1;
          const request = run.requests[frame.request - 1] as Parameters<typeof createTrappingCheck>[1]["request"];
          if (point === undefined || point.truth === null || request.active === undefined || request.forceSwitch !== undefined) continue;
          const trapping = createTrappingCheck({ engine, vocabulary: realVocabulary, scenario_id }, { request, team: game.teams[seat] as unknown as PokemonSet[], state: frame.state });
          if (trapping === null) continue;
          checked += 1;
          informative += trapping.observed.length;
          const ids = frame.state.opponent.roster.map((m) => m.base_species);
          const base = { prior: uniformPrior(ids), vocabulary: realVocabulary, config, opponent: frame.state.opponent, flags: run.flags };
          const without = belief(base);
          const withTrap = belief({ ...base, trapping });
          // At this pin the roster has no species that lists a trapping ability beside the one it holds, so nothing is excluded.
          expect(text(withTrap.slots), `${game.id}/${seat}/${n}`).toBe(text(without.slots));
          expect(sound(withTrap, { ...point, truth: point.truth }).length).toBeGreaterThan(0);
        }
      }
    }
    expect(checked).toBeGreaterThan(300);
    expect(informative).toBeGreaterThan(checked);
  }, 300_000);
});

describe("a species that lists a trapping ability (outside the format's roster, added to a copy of the vocabulary)", () => {
  const diglett = { id: "diglett", abilities: ["sandveil", "arenatrap", "sandforce"] as [string, ...string[]], moves: ["earthquake", "rockslide", "protect", "substitute", "shadowclaw", "sandtomb"] as [string, ...string[]] };
  const vocabulary = { ...realVocabulary, roster: [...realVocabulary.roster, diglett] };

  function setup() {
    const c = cases().find((x) => x.id === "b00/p1/33");
    if (c === undefined) throw new Error("the case is missing");
    const roster = c.input.state.opponent.roster.map((mon, index) => (index === c.input.state.opponent.roster.findIndex((m) => m.active_slot !== null) ? { ...mon, species: "diglett", base_species: "diglett", temp_species: null, name: null } : mon));
    const opponent = { ...c.input.state.opponent, roster };
    const ids = roster.map((m) => m.base_species);
    const others = uniformPrior(ids.filter((id) => id !== "diglett"));
    const forme = { id: "diglett", weight: 1, abilities: { denominator: 1, entries: [], residual: 1, unmapped: 0 }, items: { denominator: 2, entries: [], no_item: 1, residual: 1, unmapped: 0 }, spreads: { denominator: 1, entries: [], residual: 1, unmapped: 0 }, moves: { denominator: 2, slots: 4, entries: [], empty_slots: 0, residual: 8, unmapped: 0 } };
    const prior = { ...others, species: [...others.species, { id: "diglett", denominator: 1, formes: [forme] }].sort((a, b) => (a.id < b.id ? -1 : 1)) } as typeof others;
    const request = (maybe: boolean) => ({ ...c.input.request, active: c.input.request.active?.map((slot, k) => (k === 1 && maybe ? { ...slot, maybeTrapped: true } : slot)) }) as typeof c.input.request;
    const check = (maybe: boolean) => createTrappingCheck({ engine, vocabulary, scenario_id }, { request: request(maybe), team: c.input.team as unknown as PokemonSet[], state: c.input.state });
    return { opponent, prior, check };
  }

  it("a maybeTrapped on the last active rules out the held trapping ability, which also traps the earlier active", () => {
    const { opponent, prior, check } = setup();
    const trapping = check(true);
    expect(trapping?.observed).toEqual([0, 1]);
    const b = belief({ prior, vocabulary, config, opponent, flags: [], trapping });
    const slot = b.slots.find((s) => s.species === "diglett");
    expect(slot?.abilities.map((m) => m.id)).toEqual(["sandforce", "sandveil"]);
    expect(slot?.constraints).toContain("trapping_excluded:arenatrap");
    expect(b.constraints).toContain("trapping:upper");
  });

  it("no flag at all rules out every ability of a species that would have shown one: the evidence has no support", () => {
    const { opponent, prior, check } = setup();
    expect(() => belief({ prior, vocabulary, config, opponent, flags: [], trapping: check(false) })).toThrow(BeliefError);
  });
});
