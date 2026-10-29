import { readFileSync } from "node:fs";
import { beforeAll, describe, expect, it } from "vitest";
import { jcs, type Json } from "../src/canonical-json.ts";
import type { Engine, PokemonSet } from "../src/engine.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { loadEngine } from "../src/index.ts";
import { loadMechanicsConfig, type MechanicsConfig } from "../src/mechanics-config.ts";
import { declaredDamageScenario } from "../src/mechanics-damage.ts";
import { MechanicsProbeError, probeMechanics } from "../src/mechanics-probe.ts";
import { type MechanicsQuery, readMechanicsQuery } from "../src/mechanics-query.ts";
import { readMechanicsResult } from "../src/mechanics-result.ts";
import { checkScenario } from "../src/mechanics-scenario.ts";
import { exportPaste, importPaste } from "../src/teams.ts";

// ADR-016 slice 3: the damage probe. The fixtures are worked by hand outside
// the probe; the real-battle test holds a result to what the engine deals.

interface Case {
  id: string;
  description: string;
  query: unknown;
  expected: Record<string, unknown>;
}
const fixture = JSON.parse(readFileSync(new URL("../../../fixtures/mechanics/gen9championsvgc2026regmc/damage/cases.json", import.meta.url), "utf8")) as { data_class: string; cases: Case[] };

let engine: Engine;
let config: MechanicsConfig;
beforeAll(() => {
  engine = loadEngine();
  config = loadMechanicsConfig(GOLDEN_FORMAT);
});

const probe = (query: unknown) => probeMechanics({ engine, config }, readMechanicsQuery(query));
const answerOf = (query: unknown) => {
  const { result } = probe(query);
  if (result.kind !== "computed" || result.answer.kind !== "damage") throw new Error(`not computed: ${JSON.stringify(result)}`);
  return result.answer;
};

describe("hand-worked fixtures (computed outside the probe from the modifyDamage steps)", () => {
  it("is synthetic and covers every case the ticket names", () => {
    expect(fixture.data_class).toBe("synthetic");
    expect(fixture.cases.map((c) => c.id)).toEqual([
      "single-target", "spread", "reflect", "light-screen-spread", "rain", "sun", "mega", "burn", "always-crit", "crit-immune", "high-crit-ratio", "boost-and-hp",
    ]);
  });

  it.each(fixture.cases.map((c) => [c.id, c] as const))("%s", (_id, c) => {
    const { result, diagnostics } = probe(c.query);
    expect(diagnostics).toEqual([]);
    expect(result).toMatchObject({ kind: "computed", gap_ids: ["damage-formula", "level", "mega-evolution", "stat-points"] });
    expect(readMechanicsResult(result)).toEqual(result);
    if (result.kind === "computed") expect(result.answer).toEqual({ kind: "damage", ...c.expected });
  });

  it("gives each crit outcome the engine's own chance: a pair, always, or never", () => {
    const chances = fixture.cases.map((c) => (c.expected.crit_chance as { kind: string }).kind);
    expect(new Set(chances)).toEqual(new Set(["chance", "always", "never"]));
    const slash = fixture.cases.find((c) => c.id === "high-crit-ratio");
    expect(slash?.expected.crit_chance).toEqual({ kind: "chance", numerator: 1, denominator: 8 });
  });

  it("is byte-identical on a rerun", () => {
    const first = fixture.cases[0] as Case;
    const run = (): string => jcs(probe(first.query).result as unknown as Json);
    expect(run()).toBe(run());
  });

  it("names the data class of the query", () => {
    const first = fixture.cases[0] as Case;
    const asPrivate = { ...(first.query as object), data_class: "private" };
    expect(probe(asPrivate).result).toMatchObject({ kind: "computed", data_class: "private" });
  });
});

// A query built from the first fixture case, with the move or the defender state changed.
interface Declared {
  set: string;
  status: string | null;
  hp: number | null;
  mega: boolean;
}
interface DamageDoc {
  kind: "damage";
  attacker: Declared;
  defender: Declared;
  move: string;
  state: { weather: string | null; terrain: string | null; trick_room: boolean; sides: { aurora_veil: boolean; tailwind: boolean }[] };
}
const base = (fixture.cases[0] as Case).query as MechanicsQuery & { query: DamageDoc };
const variant = (change: (query: typeof base.query) => void): unknown => {
  const copy = structuredClone(base);
  change(copy.query);
  return copy;
};

describe("a declared status is an observed fact", () => {
  const LIMBER = "Hawlucha (F)  \nAbility: Limber  \nLevel: 50  \nEVs: 11 HP / 11 Atk / 11 Def / 11 SpA / 11 SpD / 11 Spe  \nHardy Nature  \n- Brick Break  \n\n";
  it("is taken whatever the type: a Steel Pokemon declared poisoned (Corrosion does it in play) is computed, not refused", () => {
    const steel = exportPaste(engine, [importPaste(engine, "Empoleon\nAbility: Torrent\nLevel: 50\nEVs: 32 HP / 32 SpA / 2 Spe\nModest Nature\n- Icy Wind\n- Protect\n")[0] as PokemonSet]);
    for (const status of ["psn", "tox", "brn"]) {
      expect(probe(variant((q) => Object.assign(q.defender, { set: steel, status }))).result, status).toMatchObject({ kind: "computed" });
    }
  });

  it("is still refused when the Pokemon's ability blocks it (the assumed ability then contradicts the fact)", () => {
    expect(() => probe(variant((q) => Object.assign(q.defender, { set: LIMBER, status: "par" })))).toThrow("the engine does not take that status");
    expect(probe(variant((q) => Object.assign(q.defender, { set: LIMBER, status: "brn" }))).result).toMatchObject({ kind: "computed" });
  });
});

describe("moves and states outside what the probe answers exactly", () => {
  it("a move whose damage has a random base power is unsupported (random_effect): nothing is answered for the engine", () => {
    for (const move of ["magnitude", "present"]) {
      expect(probe(variant((q) => (q.move = move))).result, move).toMatchObject({ kind: "unsupported", reason: "random_effect", position: null });
    }
  });

  it("a move that deals no damage of its own, and a multi-hit move, are outside the v1 scope (query_kind)", () => {
    for (const move of ["protect", "willowisp", "doublekick", "bulletseed"]) {
      expect(probe(variant((q) => (q.move = move))).result, move).toMatchObject({ kind: "unsupported", reason: "query_kind", position: null });
    }
  });

  it("a move that never reaches the damage step for another reason (a charge turn, a failed precondition) is unsupported, not zero", () => {
    for (const move of ["solarbeam", "fly", "suckerpunch", "counter"]) {
      expect(probe(variant((q) => (q.move = move))).result, move).toMatchObject({ kind: "unsupported", reason: "query_kind", position: null });
    }
  });

  it("an immunity deals nothing: sixteen zeros and no crit", () => {
    // Normal into a Ghost: the Gengar set is the attacker of the Mega fixture.
    const ghost = ((fixture.cases.find((c) => c.id === "mega") as Case).query as typeof base).query.attacker.set;
    const answer = answerOf(variant((q) => ((q.move = "bodyslam"), (q.defender.set = ghost))));
    expect(answer.no_crit).toEqual(new Array(16).fill(0));
    expect(answer.crit).toEqual(new Array(16).fill(0));
    expect(answer.crit_chance).toEqual({ kind: "never" });
  });

  it("fixed and HP-based damage (Seismic Toss, Super Fang, a one-hit KO move) is outside the v1 scope: query_kind", () => {
    for (const move of ["seismictoss", "superfang", "fissure"]) {
      expect(probe(variant((q) => (q.move = move))).result, move).toMatchObject({ kind: "unsupported", reason: "query_kind", position: null });
    }
  });

  it("a move the engine makes multi-hit (Parental Bond) is unsupported, not answered as one hit", () => {
    const kanga = "Kangaskhan @ Kangaskhanite\nAbility: Early Bird\nLevel: 50\nEVs: 32 HP / 32 Atk\nAdamant Nature\n- Body Slam\n- Protect\n";
    const set = exportPaste(engine, importPaste(engine, kanga));
    expect(probe(variant((q) => ((q.move = "bodyslam"), (q.attacker.set = set), (q.attacker.mega = true)))).result).toMatchObject({ kind: "unsupported", reason: "query_kind", position: null });
  });

  it("no ability or item of the pinned engine draws on the damage path, so the random_effect refusal is shown with random-base-power moves and the source's own tests", () => {
    expect(probe(variant((q) => (q.move = "magnitude"))).result).toMatchObject({ kind: "unsupported", reason: "random_effect" });
  });

  it("the KO field holds the counts (the full KO fixtures are in mechanics-ko.test.ts)", () => {
    expect(answerOf(base).ko).toMatchObject({ kind: "counts" });
  });
});

describe("states the engine will not take throw, with no result", () => {
  it.each([
    ["an unknown move", (q: typeof base.query) => (q.move = "notamove")],
    ["a move id that is only an alias", (q: typeof base.query) => (q.move = "dragonclaw2")],
    ["HP above the maximum", (q: typeof base.query) => (q.defender.hp = 9999)],
    ["an unknown status", (q: typeof base.query) => (q.attacker.status = "zzz")],
    ["an unknown weather", (q: typeof base.query) => (q.state.weather = "zzz")],
    ["an unknown terrain", (q: typeof base.query) => (q.state.terrain = "zzz")],
    ["Mega Evolution of a Pokemon with no Mega Stone", (q: typeof base.query) => (q.attacker.mega = true)],
  ])("%s", (_name, change) => {
    expect(() => probe(variant(change))).toThrow(MechanicsProbeError);
  });

  it("takes the declared terrain, Trick Room, Tailwind and Aurora Veil", () => {
    const answer = answerOf(variant((q) => {
      q.state.terrain = "electricterrain";
      q.state.trick_room = true;
      q.state.sides[0].tailwind = true;
    }));
    // None of them changes a damage roll: the vector is the plain case's.
    expect(answer.no_crit).toEqual((fixture.cases[0] as Case).expected.no_crit);
    // Aurora Veil on the defender's side does: the engine halves the roll to 2732/4096 in doubles.
    const veil = answerOf(variant((q) => (q.state.sides[1].aurora_veil = true)));
    expect((veil.no_crit[0] as number) < answer.no_crit[0]).toBe(true);
  });
});

describe("a declared status is a fact of the state, not a new infliction", () => {
  const paste = (species: string, ability: string, moves: string): string => exportPaste(engine, importPaste(engine, `${species}\nAbility: ${ability}\nLevel: 50\nEVs: 32 HP / 32 Atk / 2 Spe\nAdamant Nature\n${moves}`));

  it("takes a burn on a Fire-type attacker, which the engine would never inflict, and halves its physical damage", () => {
    // Libero users take the type of their move, so a public burned Pokemon can be of a type that is immune to the burn
    // when the state is read with its assumed typing and ability.
    const cinderace = paste("Cinderace", "Blaze", "- Pyro Ball\n- Protect\n");
    const plain = answerOf(variant((q) => ((q.attacker.set = cinderace), (q.move = "pyroball"))));
    const burned = answerOf(variant((q) => ((q.attacker.set = cinderace), (q.move = "pyroball"), (q.attacker.status = "brn"))));
    expect(burned.no_crit[0]).toBeLessThan(plain.no_crit[0] as number);
    // An unknown status is still refused.
    expect(() => probe(variant((q) => ((q.attacker.set = cinderace), (q.attacker.status = "zzz"))))).toThrow(MechanicsProbeError);
  });
});

describe("declared state replaces what the switch-in effects set", () => {
  it("a weather set by a Pokemon's ability at battle start is cleared unless declared", () => {
    // Abomasnow's Snow Warning would otherwise set snow; the declared field has no weather.
    const fixtureCase = fixture.cases.find((c) => c.id === "always-crit") as Case;
    expect(answerOf(fixtureCase.query).no_crit).toEqual((fixtureCase.expected as { no_crit: number[] }).no_crit);
  });
});

describe("a declared Mega Evolution leaves the other combatant's declared stages alone (#1243)", () => {
  const MANECTRIC = "Manectric @ Manectite  \nAbility: Static  \nLevel: 50  \nEVs: 32 HP / 32 SpA / 2 Spe  \nTimid Nature  \n- Thunderbolt  \n- Protect  \n\n";
  const stagesOf = (attackerAtk: number): { atk: number; ability: string } => {
    const query = variant((q) => {
      q.attacker.boosts = { atk: attackerAtk, def: 0, spa: 0, spd: 0, spe: 0 };
      q.defender.set = MANECTRIC;
      q.defender.mega = true;
    });
    const parsed = readMechanicsQuery(query).query;
    if (parsed.kind !== "damage") throw new Error("not a damage query");
    const scenarioId = checkScenario(engine, config);
    const battle = declaredDamageScenario(engine, scenarioId, parsed, importPaste(engine, parsed.attacker.set)[0] as PokemonSet, importPaste(engine, parsed.defender.set)[0] as PokemonSet) as unknown as { sides: { pokemon: { boosts: { atk: number }; ability: string }[] }[] };
    return { atk: (battle.sides[0]?.pokemon[0] as { boosts: { atk: number } }).boosts.atk, ability: (battle.sides[1]?.pokemon[0] as { ability: string }).ability };
  };
  it("the attacker keeps the declared Attack stage although the evolved defender's ability is Intimidate", () => {
    expect(stagesOf(0)).toEqual({ atk: 0, ability: "intimidate" });
    expect(stagesOf(2).atk).toBe(2);
  });
});

describe("damage equals HP lost (ADR-016 section 7, slice 3)", () => {
  // The same declared state played as a real, seeded battle: the attacker uses
  // the move, everything else protects or harms only the attacker; the HP the
  // defender loses is the vector entry for the roll and crit outcome the battle drew.
  const garchomp = (): PokemonSet => importPaste(engine, "Garchomp\nAbility: Rough Skin\nLevel: 50\nEVs: 32 HP / 32 Atk / 2 Spe\nAdamant Nature\n- Dragon Claw\n- Rock Slide\n- Protect\n")[0] as PokemonSet;
  const empoleon = (): PokemonSet => importPaste(engine, "Empoleon\nAbility: Torrent\nLevel: 50\nEVs: 32 HP / 32 SpA / 2 Spe\nModest Nature\n- Icy Wind\n- Protect\n")[0] as PokemonSet;
  const zero = { atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
  const side = { aurora_veil: false, light_screen: false, reflect: false, tailwind: false };
  const queryFor = (move: string, extra: { reflect?: boolean; weather?: string | null; burn?: boolean }): unknown => ({
    schema_version: 1,
    data_class: "synthetic",
    format_id: GOLDEN_FORMAT,
    query: {
      kind: "damage",
      attacker: { set: exportPaste(engine, [garchomp()]), boosts: zero, status: extra.burn ? "brn" : null, hp: null, mega: false },
      defender: { set: exportPaste(engine, [empoleon()]), boosts: zero, status: null, hp: null, mega: false },
      move,
      state: { weather: extra.weather ?? null, terrain: null, trick_room: false, sides: [side, { ...side, reflect: extra.reflect ?? false }] },
    },
  });

  interface Spy {
    rolls: { move: string | undefined; roll: number }[];
  }
  interface PlayedBattle {
    sides: { pokemon: { hp: number }[] }[];
    log: string[];
    prng: unknown;
    activeMove?: { id: string };
    resetRNG(seed: string): void;
    choose(slot: string, input: string): boolean;
  }

  /** Play one real turn from the declared state; returns the defender's HP loss, the roll the attacker's move drew and whether it crit. */
  function playTurn(query: unknown, move: string, choice: string, seed: number): { loss: number; roll: number | null; crit: boolean } {
    const parsed = readMechanicsQuery(query).query;
    if (parsed.kind !== "damage") throw new Error("not a damage query");
    const scenarioId = checkScenario(engine, config);
    const battle = declaredDamageScenario(engine, scenarioId, parsed, importPaste(engine, parsed.attacker.set)[0] as PokemonSet, importPaste(engine, parsed.defender.set)[0] as PokemonSet) as unknown as PlayedBattle;
    battle.resetRNG(`sodium,${seed.toString(16).padStart(32, "0")}`);
    const spy: Spy = { rolls: [] };
    const real = battle.prng as Record<string, (...args: unknown[]) => unknown>;
    battle.prng = new Proxy(real, {
      get(target, name: string) {
        const member = target[name] as (...args: unknown[]) => unknown;
        if (name !== "random") return member.bind(target);
        return (...args: unknown[]) => {
          const value = member.apply(target, args) as number;
          if (args[0] === 16) spy.rolls.push({ move: battle.activeMove?.id, roll: value });
          return value;
        };
      },
    });
    const before = (battle.sides[1] as PlayedBattle["sides"][number]).pokemon[0]?.hp as number;
    battle.choose("p1", `${choice}, move 3`);
    battle.choose("p2", "move 1, move 2");
    const after = (battle.sides[1] as PlayedBattle["sides"][number]).pokemon[0]?.hp as number;
    const drawn = spy.rolls.filter((entry) => entry.move === move);
    return { loss: before - after, roll: drawn.length > 0 ? (drawn[0] as { roll: number }).roll : null, crit: battle.log.some((line) => line.startsWith("|-crit|p2a")) };
  }

  it.each([
    ["dragonclaw", "move 1 1", {}],
    ["dragonclaw", "move 1 1", { reflect: true }],
    ["dragonclaw", "move 1 1", { burn: true }],
    ["rockslide", "move 2", {}],
    ["rockslide", "move 2", { weather: "raindance" }],
  ] as const)("%s %j: the HP lost is the vector entry for the drawn roll and crit", (move, choice, extra) => {
    const query = queryFor(move, extra);
    const answer = answerOf(query);
    let compared = 0;
    let crits = 0;
    for (let seed = 1; seed <= 400 && (compared < 40 || crits === 0); seed += 1) {
      const turn = playTurn(query, move, choice, seed);
      if (turn.roll === null) continue; // the move missed or never reached the damage step
      const expected = (turn.crit ? answer.crit : answer.no_crit)[turn.roll];
      expect(turn.loss, `seed ${seed} roll ${turn.roll} crit ${turn.crit}`).toBe(expected);
      compared += 1;
      if (turn.crit) crits += 1;
    }
    expect(compared).toBeGreaterThanOrEqual(40);
    expect(crits).toBeGreaterThan(0);
  });
});
