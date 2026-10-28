import type { PublicMon } from "@areneon/public-state";
import { beforeAll, describe, expect, it } from "vitest";
import { candidateFeatures, createFeatureMemo, type FeatureConfigs } from "../src/candidate-features.ts";
import { buildAssumedSet } from "../src/damage-aware-assumed-set.ts";
import { damageView } from "../src/damage-aware-view.ts";
import type { PokemonSet } from "../src/engine.ts";
import { damageAwareActor } from "../src/damage-aware-actor.ts";
import { parseJointChoice } from "../src/damage-aware-choice.ts";
import { loadDamageAwareConfig } from "../src/damage-aware-config.ts";
import { scenarioFormatId, startScenario } from "../src/mechanics-scenario.ts";
import { playerChannel } from "../src/player-channel.ts";
import { importPaste } from "../src/teams.ts";
import { capture, type Captured, config, corpus, engine, mechanics, snapshot, stateOf, stubProbe, vocabulary } from "./damage-aware-support.ts";

// #1180: the pinned engine's choice-location convention. `Pokemon.getAtLoc(L)` returns `foeSide.active[L - 1]`, so
// `move N 1` hits the foe in active slot 0 (`p2a`, the reader's slot `a`) and `move N 2` the foe in slot 1.
// The legacy numbering (`damage-aware@v1` and `@v2`, kept bug-for-bug) is the reverse.

interface LocPokemon { getAtLoc(loc: number): { name: string; hp: number; maxhp: number } }
interface LocBattle { sides: { active: (LocPokemon & { name: string })[] }[]; choose(slot: string, input: string): boolean }

const pokemon = (paste: string): PokemonSet[] => importPaste(engine, paste);
const GARCHOMP = "Garchomp\nAbility: Rough Skin\nLevel: 50\nEVs: 32 HP / 32 Atk / 2 Spe\nAdamant Nature\n- Dragon Claw\n- Rock Slide\n\n";
const ARCANINE = "Arcanine\nAbility: Intimidate\nLevel: 50\nEVs: 32 HP / 32 SpA\nModest Nature\n- Heat Wave\n- Protect\n\n";

describe("the engine's convention (the reference the module is held to)", () => {
  let battle: LocBattle;
  beforeAll(() => {
    // p1: Garchomp, Arcanine. p2: Arcanine (slot a), Garchomp (slot b): the two foes are distinct species.
    battle = startScenario(engine, scenarioFormatId(mechanics), { p1: [...pokemon(GARCHOMP), ...pokemon(ARCANINE)], p2: [...pokemon(ARCANINE), ...pokemon(GARCHOMP)] }) as unknown as LocBattle;
  });

  it("getAtLoc(L) of an own Pokemon is the foe in active slot L - 1", () => {
    const mine = (battle.sides[0] as LocBattle["sides"][number]).active[0] as LocPokemon;
    const foes = (battle.sides[1] as LocBattle["sides"][number]).active;
    expect(foes.map((foe) => foe.name)).toEqual(["Arcanine", "Garchomp"]);
    expect(mine.getAtLoc(1).name).toBe("Arcanine");
    expect(mine.getAtLoc(2).name).toBe("Garchomp");
  });

  it("a single-target `move N 2` played on the engine damages the foe in slot b, not the one in slot a", () => {
    const foes = (battle.sides[1] as LocBattle["sides"][number]).active as unknown as { hp: number; maxhp: number }[];
    // p1 slot a (Garchomp) uses Dragon Claw at location 2; slot b (Arcanine) protects itself; p2's moves (Heat Wave, Rock Slide) hit only p1's side, so they cannot damage the foes.
    expect(battle.choose("p1", "move 1 2, move 2")).toBe(true);
    expect(battle.choose("p2", "move 1, move 2")).toBe(true);
    const [a, b] = foes as [{ hp: number; maxhp: number }, { hp: number; maxhp: number }];
    expect(b.hp).toBeLessThan(b.maxhp); // location 2: Garchomp in slot b took the hit
    expect(a.hp).toBe(a.maxhp); // Arcanine in slot a: untouched (Heat Wave / Rock Slide hit the other side only)
  });
});

describe("the module's mapping equals the engine's", () => {
  let two: Captured[] = [];
  beforeAll(async () => {
    two = (await capture(corpus[0] as never)).concat(await capture(corpus[1] as never)).filter((c) => Object.keys(stateOf(c).opponent.slots).length === 2);
  }, 120_000); // seeded real battles on the pinned engine

  const viewOf = (engineTargets: boolean) =>
    damageView({ engine, vocabulary, config, v1: config, mechanics, snapshot, data_class: "synthetic", probe: stubProbe(), counts: { queries: 0, probe_runs: 0, unsupported_queries: 0 }, memo: new Map(), engine_targets: engineTargets });

  it("foesOf numbers location L as active slot L - 1 under the engine convention, and the reverse under the legacy one", () => {
    let both = 0;
    for (const c of two) {
      const state = stateOf(c);
      const letters = Object.keys(state.opponent.slots).sort();
      for (const [engineTargets, locOf] of [[true, (position: number) => position + 1], [false, (position: number) => letters.length - position]] as const) {
        const foes = viewOf(engineTargets).foesOf(state, (mon) => buildAssumedSet({ engine, vocabulary, config }, mon));
        for (const foe of foes) {
          const position = letters.findIndex((letter) => state.opponent.roster[state.opponent.slots[letter] as number] === foe.mon);
          expect(foe.loc).toBe(locOf(position));
        }
        if (foes.length === 2) both += 1;
      }
    }
    expect(both).toBeGreaterThan(0);
  });

  it("a feature row keyed `target: L` holds the damage against the foe in active slot L - 1", () => {
    let checked = 0;
    for (const c of two) {
      const state = stateOf(c);
      const letters = Object.keys(state.opponent.slots).sort();
      const slotOfPaste = new Map<string, number>();
      const configs: FeatureConfigs = {
        vocabulary, config, mechanics, snapshot, data_class: "synthetic",
        assume: (mon) => {
          const set = buildAssumedSet({ engine, vocabulary, config }, mon);
          slotOfPaste.set(set.paste, letters.findIndex((letter) => (state.opponent.roster[state.opponent.slots[letter] as number] as { name: string }).name === mon.name));
          return set;
        },
        // the roll is 10 + the defender's active slot: the row's value says which foe the builder asked about
        probe: (context, query) => {
          const q = query.query;
          const slot = q.kind === "damage" ? (slotOfPaste.get(q.defender.set) as number) : 0;
          return stubProbe({ values: q.kind === "damage" ? { [q.move]: 10 + slot } : {} })(context, query);
        },
      };
      const legal = c.request.legal;
      if (c.request.phase !== "move" || legal === undefined) continue;
      const candidates = [...legal.choices, ...legal.pending].map((choice) => ({ choice, pending: legal.pending.includes(choice) }));
      const channel = playerChannel({ schema_version: 1, data_class: "synthetic", player: c.context.player, lines: c.context.channel });
      const { rows } = candidateFeatures(engine, configs, channel, c.context.team as never, candidates, createFeatureMemo());
      for (const row of rows) {
        if (row.kind !== "computed") continue;
        expect(row.mean_ppm, `target ${row.target}`).toBe((10 + row.target - 1) * 10_000);
        checked += 1;
      }
    }
    expect(checked).toBeGreaterThan(0);
  }, 120_000);
});

describe("the v3 record", () => {
  it("is v1's values plus the engine's convention", () => {
    const v1 = loadDamageAwareConfig("v1", vocabulary);
    const v3 = loadDamageAwareConfig("v3", vocabulary);
    expect(v1.target_convention).toBeUndefined();
    const { target_convention, ...rest } = v3;
    expect(target_convention).toBe("engine");
    expect(rest).toEqual(v1);
  });
});

describe("damage-aware@v3 against v1 on the same requests", () => {
  let two: Captured[] = [];
  beforeAll(async () => {
    two = (await capture(corpus[0] as never)).concat(await capture(corpus[1] as never), await capture(corpus[3] as never));
  }, 120_000); // seeded real battles on the pinned engine

  type Active = { moves: { id: string; target?: string }[] }[];
  const v3 = loadDamageAwareConfig("v3", vocabulary);
  const category = (id: string): string => (engine.sim.Dex.mod("champions").moves.get(id) as unknown as { category: string }).category;

  /** The assumed set and a probe whose damage is 10 + the defender's active slot (a roll of 10 or 11 of 100), so the foe a part is scored against decides the score. */
  const standIns = (c: Captured, cfg: typeof v3 | typeof config) => {
    const state = stateOf(c);
    const letters = Object.keys(state.opponent.slots).sort();
    const slotOfPaste = new Map<string, number>();
    const assume = (mon: PublicMon) => {
      const set = buildAssumedSet({ engine, vocabulary, config: cfg }, mon);
      slotOfPaste.set(set.paste, letters.findIndex((letter) => (state.opponent.roster[state.opponent.slots[letter] as number] as { name: string }).name === mon.name));
      return set;
    };
    const probe = (context: Parameters<ReturnType<typeof stubProbe>>[0], query: Parameters<ReturnType<typeof stubProbe>>[1]) => {
      const q = query.query;
      if (q.kind !== "damage") return stubProbe()(context, query);
      return stubProbe({ values: { [q.move]: 10 + (slotOfPaste.get(q.defender.set) ?? 0) } })(context, query);
    };
    return { assume, probe };
  };

  it("v3 targets by the engine's numbering and v1 by the legacy one: with two live foes they pick different single-target parts", () => {
    let differ = 0;
    for (const c of two) {
      const state = stateOf(c);
      const letters = Object.keys(state.opponent.slots).sort();
      const live = letters.filter((letter) => {
        const mon = state.opponent.roster[state.opponent.slots[letter] as number];
        return mon !== undefined && !mon.fainted;
      });
      const legal = c.request.legal;
      if (c.request.phase !== "move" || legal === undefined || live.length !== 2 || legal.choices.length < 2) continue;
      const active = (c.request.request as unknown as { active: Active }).active;
      // the expected score of a candidate, by the test's own reading of the numbering: part value = 10 + the slot the location names
      const expected = (locSlot: (loc: number) => number): string => {
        const scores = legal.choices.map((choice) =>
          parseJointChoice(choice).reduce((sum, part, slot) => {
            if (part.kind !== "move") return sum;
            const move = (active[slot] as Active[number]).moves[part.index - 1] as Active[number]["moves"][number];
            if (move.target === undefined || category(move.id) === "Status") return sum;
            if (["allAdjacentFoes", "allAdjacent", "randomNormal"].includes(move.target)) return sum + 21;
            if (!["normal", "any"].includes(move.target) || part.target === null || part.target < 1) return sum;
            return sum + 10 + locSlot(part.target);
          }, 0),
        );
        return legal.choices[scores.indexOf(Math.max(...scores))] as string;
      };
      const play = (cfg: typeof v3 | typeof config, commit: string): string =>
        damageAwareActor({ engine, vocabulary, config: cfg, mechanics, snapshot, data_class: "synthetic", commit, ...standIns(c, cfg) })(c.request, c.context) as string;
      // each `play` builds a new actor and answers this request from its own channel
      const pick3 = play(v3, "v3");
      const pick1 = play(config, "v1");
      expect(pick3).toBe(expected((loc) => loc - 1));
      expect(pick1).toBe(expected((loc) => 2 - loc));
      if (pick1 !== pick3) differ += 1;
    }
    expect(differ).toBeGreaterThan(0);
  }, 300_000);

  it("a record names its own version: v3 plays only as damage-aware@v3 and a v1 record only as v1", () => {
    const base = { engine, vocabulary, mechanics, snapshot, data_class: "synthetic" as const };
    expect(damageAwareActor({ ...base, config: v3 }).report().policy_id).toBe("damage-aware@v3");
    expect(damageAwareActor({ ...base, config }).report().policy_id).toBe("damage-aware@v1");
    expect(() => damageAwareActor({ ...base, config: v3, commit: "v1" })).toThrow(/another record version/);
    expect(() => damageAwareActor({ ...base, config, commit: "v3" })).toThrow(/another record version/);
    const a = damageAwareActor({ ...base, config: v3 }).report().identity_hash;
    expect(a).not.toBe(damageAwareActor({ ...base, config }).report().identity_hash);
  });
});
