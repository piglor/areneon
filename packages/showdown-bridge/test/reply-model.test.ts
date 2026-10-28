import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, expectTypeOf, it } from "vitest";
import { enumerateCandidates } from "../src/choices.ts";
import { type DamageAnswer, DAMAGE_VALUE_FORMAT, damageValue } from "../src/damage-aware-score.ts";
import type { Engine, PokemonSet } from "../src/engine.ts";
import { GOLDEN_FORMAT, TEAMS_DIR } from "../src/golden.ts";
import { type BattleRecord, type HiddenVerdicts, type OmniscientLog, loadEngine } from "../src/index.ts";
import { loadMechanicsConfig } from "../src/mechanics-config.ts";
import { MechanicsProbeError } from "../src/mechanics-error.ts";
import { checkScenario, startScenario } from "../src/mechanics-scenario.ts";
import { decodeRequest } from "../src/request.ts";
import { REPLY_MODEL_FORMAT, ReplyModelError, createReplyModel, newWorldMemo } from "../src/reply-model.ts";
import { importPaste } from "../src/teams.ts";
import { createWorldBuilder, type ScenarioWorld } from "../src/world-builder.ts";
import { config, mechanics, snapshot, stubProbe, vocabulary } from "./damage-aware-support.ts";
import { cases } from "./world-builder-support.ts";

let engine: Engine;
let scenarioId: string;
let ownSets: PokemonSet[];
let foeSets: PokemonSet[];
const paste = (file: string): PokemonSet[] => importPaste(engine, readFileSync(join(TEAMS_DIR, file), "utf8"));
beforeAll(() => {
  engine = loadEngine();
  scenarioId = checkScenario(engine, loadMechanicsConfig(GOLDEN_FORMAT));
  // The player's side: Venusaur and Ember (Charizard) are the actives, Blastoise and one more the bench.
  ownSets = paste("legal-synthetic.txt").slice(0, 4);
  // The replying opponent: Tyranitar (Rock Slide spread, Crunch, Low Kick, Stone Edge) and Rex (Garchomp: Earthquake spread, Dragon Claw, Fire Fang, Swords Dance).
  foeSets = paste("legal-opponent.txt").slice(0, 4);
});

/** A scenario world of the declared sets as the engine starts them (no state written): the player is p1, the replying opponent p2. */
const worldOf = (own = ownSets, foe = foeSets): ScenarioWorld => ({
  battle: startScenario(engine, scenarioId, { p1: structuredClone(own), p2: structuredClone(foe) }),
  seat: "p1",
  own_order: [],
  opponent_order: [],
  resampled: [],
  defaulted_sources: 0,
});

const model = (probe = stubProbe()) => createReplyModel({ engine, vocabulary, config, mechanics, snapshot, data_class: "synthetic", probe });

/** A probe whose damage roll (the target has 100 hit points, so `roll * 10,000` ppm) is the test's function of the move and the defender's species. */
function rollsBy(roll: (move: string, defender: string) => number | "unsupported") {
  return (context: Parameters<ReturnType<typeof stubProbe>>[0], query: Parameters<ReturnType<typeof stubProbe>>[1]) => {
    if (query.query.kind !== "damage") return stubProbe()(context, query);
    const defender = /^(?:.*\()?([A-Za-z-]+)\)?\s*(?:\(|@|\n)/.exec(query.query.defender.set)?.[1] ?? "";
    const value = roll(query.query.move, defender);
    return value === "unsupported"
      ? stubProbe({ override: { damage: { kind: "unsupported", reason: "random_effect", position: null } } })(context, query)
      : stubProbe({ values: { [query.query.move]: value } })(context, query);
  };
}

const PPM_PER_ROLL = 10_000;

describe("hand-checked worlds (stub probe: every query's value is the test's roll times 10,000 ppm)", () => {
  it("takes the spread move as the sum over its foes and beats a single-target move of a lower value", () => {
    // Rock Slide 30 into each of 2 foes = 60 > Stone Edge 40; Earthquake 50 into each of 2 foes = 100 > Dragon Claw 20.
    const { reply, counts } = model(stubProbe({ values: { rockslide: 30, stoneedge: 40, crunch: 10, lowkick: 5, earthquake: 50, dragonclaw: 20, firefang: 20 } }));
    const result = reply(worldOf());
    expect(result).toEqual({ choice: "move 1, move 1", kind: "max_damage", value: (60 + 100) * PPM_PER_ROLL });
    expect(counts()).toMatchObject({ replies: 1, max_damage: 1, all_zero: 0, forced_slots: 0 });
    expect(counts().queries).toBeGreaterThan(0);
    expect(counts().probe_runs).toBeLessThanOrEqual(counts().queries);
  });

  it("takes a single-target move when it is worth more than the spread, at the target that takes the most", () => {
    // Stone Edge: 80 into Venusaur (the foe at location 1, the engine's active slot 0) and 10 into Ember (location 2); Rock Slide would be 30 + 30.
    const w = worldOf();
    const tyranitar = (w.battle.sides[1] as unknown as { active: { getAtLoc(loc: number): { name: string } }[] }).active[0];
    expect(tyranitar?.getAtLoc(1).name).toBe("Venusaur"); // the engine's own location convention, which the model must follow
    expect(tyranitar?.getAtLoc(2).name).toBe("Ember");
    const probe = rollsBy((id, defender) => (id === "stoneedge" ? (defender === "Venusaur" ? 80 : 10) : id === "rockslide" ? 30 : 0));
    expect(model(probe).reply(worldOf()).choice.startsWith("move 4 1, ")).toBe(true);
    // The other way round: location 2.
    const flipped = rollsBy((id, defender) => (id === "stoneedge" ? (defender === "Venusaur" ? 10 : 80) : id === "rockslide" ? 30 : 0));
    expect(model(flipped).reply(worldOf()).choice.startsWith("move 4 2, ")).toBe(true);
  });

  it("breaks equal values by the canonical candidate order", () => {
    // Crunch and Low Kick are worth 40 into either foe; every other move 0. The canonical first of them is move 2 at location 1.
    const { reply } = model(stubProbe({ values: { crunch: 40, lowkick: 40 } }));
    const result = reply(worldOf());
    expect(result.kind).toBe("max_damage");
    expect(result.choice.startsWith("move 2 1, ")).toBe(true);
    // The other slot is worth 0 everywhere: the first of the tied candidates in canonical order is played, not a later one.
    const request = decodeRequest(JSON.stringify((worldOf().battle.sides[1] as { activeRequest: object }).activeRequest));
    const canonical = enumerateCandidates({ ...request, request_seq: 1 } as never).candidates.filter((c) => !c.pending).map((c) => c.choice);
    expect(result.choice).toBe(canonical.find((c) => c.startsWith("move 2 1, ")));
  });

  it("plays the actor-rule choice when every value is 0, and counts it", () => {
    const { reply, counts } = model(stubProbe());
    const world = worldOf();
    const result = reply(world);
    const request = decodeRequest(JSON.stringify((worldOf().battle.sides[1] as { activeRequest: object }).activeRequest));
    const first = enumerateCandidates({ ...request, request_seq: 1 } as never).candidates.find((c) => !c.pending);
    expect(result).toEqual({ choice: first?.choice, kind: "all_zero", value: 0 });
    expect(counts()).toMatchObject({ replies: 1, all_zero: 1, max_damage: 0 });
  });

  it("scores an unsupported query 0 and never approximates it", () => {
    const all = model(stubProbe({ override: { damage: { kind: "unsupported", reason: "random_effect", position: null } } }));
    const result = all.reply(worldOf());
    expect(result.kind).toBe("all_zero");
    expect(all.counts().unsupported_queries).toBeGreaterThan(0);
    // Only Earthquake is unsupported: the other slot then takes the next best, Dragon Claw 20 over Fire Fang 15.
    const some = model(rollsBy((id) => (id === "earthquake" ? "unsupported" : ({ dragonclaw: 20, firefang: 15, rockslide: 30 } as Record<string, number>)[id] ?? 0)));
    const mixed = some.reply(worldOf());
    expect(mixed).toEqual({ choice: "move 1, move 2 1", kind: "max_damage", value: (60 + 20) * PPM_PER_ROLL });
    expect(some.counts().unsupported_queries).toBeGreaterThan(0);
  });

  it("counts a query the engine refuses to state as 0 and goes on; any other failure is a defect", () => {
    const refusing = model(stubProbe({ throwOn: { damage: new MechanicsProbeError("the engine does not take that status") } }));
    expect(refusing.reply(worldOf()).kind).toBe("all_zero");
    expect(refusing.counts().probe_refusals).toBeGreaterThan(0);
    expect(() => model(stubProbe({ throwOn: { damage: new Error("boom") } })).reply(worldOf())).toThrow("boom");
    expect(() => model(stubProbe({ throwOn: { damage: new MechanicsProbeError("the mechanics config is for another format") } })).reply(worldOf())).toThrow(MechanicsProbeError);
  });

  it("never switches and never uses a status move: a slot with only status moves is held to its first legal part", () => {
    // Rex is replaced by a Garchomp with status moves only.
    const [tyranitar, , gardevoir, arcanine] = foeSets as [PokemonSet, PokemonSet, PokemonSet, PokemonSet];
    const passive: PokemonSet = { ...structuredClone(foeSets[1] as PokemonSet), moves: ["Swords Dance", "Protect", "Dragon Dance", "Rest"] };
    const { reply, counts } = model(stubProbe({ values: { rockslide: 30, stoneedge: 5 } }));
    const result = reply(worldOf(ownSets, [tyranitar, passive, gardevoir, arcanine]));
    expect(result.kind).toBe("max_damage");
    expect(result.choice.startsWith("move 1, move 1")).toBe(true); // Rock Slide, then Swords Dance (the first legal part)
    expect(result.choice).not.toContain("switch");
    expect(counts().forced_slots).toBe(1);
    // Both slots status-only: nothing to value, the actor-rule choice.
    const both: PokemonSet = { ...structuredClone(tyranitar), moves: ["Protect", "Taunt", "Rest", "Sleep Talk"] };
    const none = model(stubProbe({ values: { rockslide: 30 } }));
    expect(none.reply(worldOf(ownSets, [both, passive, gardevoir, arcanine])).kind).toBe("all_zero");
    expect(none.counts().forced_slots).toBe(2);
  });

  it("takes a random-target move as the mean over the foes it may hit", () => {
    // Outrage (target randomNormal) is worth 30 into either foe: the mean is 30, not the spread's 60.
    const rampage: PokemonSet = { ...structuredClone(foeSets[0] as PokemonSet), moves: ["Outrage", "Protect", "Rest", "Taunt"] };
    const { reply } = model(stubProbe({ values: { outrage: 30 } }));
    const result = reply(worldOf(ownSets, [rampage, foeSets[1] as PokemonSet, foeSets[2] as PokemonSet, foeSets[3] as PokemonSet]));
    expect(result).toMatchObject({ kind: "max_damage", value: 30 * PPM_PER_ROLL });
    expect(result.choice.startsWith("move 1, ")).toBe(true);
  });

  it("names ids the pinned engine has for the field it states", () => {
    const dex = engine.sim.Dex.mod(engine.sim.Dex.formats.get(GOLDEN_FORMAT).mod) as unknown as { conditions: { get(id: string): { exists: boolean } } };
    for (const id of ["auroraveil", "lightscreen", "reflect", "tailwind", "trickroom"]) expect(dex.conditions.get(id).exists, id).toBe(true);
  });

  it("reads the field from the world", () => {
    const log: string[] = [];
    const world = worldOf();
    const battle = world.battle as unknown as { field: { setWeather(id: string, source: unknown): unknown; setTerrain(id: string, source: unknown): unknown; addPseudoWeather(id: string, source: unknown): unknown }; sides: { pokemon: unknown[]; addSideCondition(id: string, source: unknown): unknown }[] };
    battle.field.setWeather("raindance", battle.sides[0]?.pokemon[0]);
    battle.field.setTerrain("electricterrain", battle.sides[0]?.pokemon[0]);
    battle.field.addPseudoWeather("trickroom", battle.sides[0]?.pokemon[0]);
    battle.sides[0]?.addSideCondition("reflect", battle.sides[0].pokemon[0]);
    battle.sides[1]?.addSideCondition("tailwind", battle.sides[1].pokemon[0]);
    const probe = (context: Parameters<ReturnType<typeof stubProbe>>[0], query: Parameters<ReturnType<typeof stubProbe>>[1]) => {
      if (query.query.kind === "damage") log.push(JSON.stringify(query.query.state));
      return stubProbe({ values: { rockslide: 1 } })(context, query);
    };
    model(probe).reply(world);
    const state = JSON.parse(log[0] as string) as { weather: string; terrain: string; trick_room: boolean; sides: { reflect: boolean; tailwind: boolean }[] };
    // The attacker's side (the replying opponent's) first: its Tailwind, then the player's Reflect.
    expect(state).toMatchObject({ weather: "raindance", terrain: "electricterrain", trick_room: true });
    expect(state.sides[0]).toMatchObject({ tailwind: true, reflect: false });
    expect(state.sides[1]).toMatchObject({ tailwind: false, reflect: true });
  });

  it("does not change the world, and a shared memo answers a repeated reply without a probe run", () => {
    const world = worldOf();
    const before = JSON.stringify(world.battle.toJSON());
    const { reply, counts } = model(stubProbe({ values: { rockslide: 30 } }));
    const memo = newWorldMemo();
    const first = reply(world, memo);
    expect(JSON.stringify(world.battle.toJSON())).toBe(before);
    const runs = counts().probe_runs;
    expect(runs).toBeGreaterThan(0);
    const second = reply(world, memo);
    expect(second).toEqual(first);
    expect(counts().probe_runs).toBe(runs); // every query came from the world's memo
    expect(counts().queries).toBeGreaterThan(runs);
    // A fresh memo asks the probe again.
    reply(world);
    expect(counts().probe_runs).toBe(2 * runs);
  });

  it("refuses a world with no request or no seat as a defect of the caller", () => {
    const world = worldOf();
    (world.battle as unknown as { clearRequest(): void }).clearRequest();
    expect(() => model().reply(world)).toThrow(ReplyModelError);
    expect(() => model().reply({ ...worldOf(), seat: "p3" as "p1" })).toThrow(ReplyModelError);
    // A battle asking for switches has only a wait request for a side that owes none.
    const waiting = worldOf();
    (waiting.battle as unknown as { requestState: string }).requestState = "switch";
    expect(() => model().reply(waiting)).toThrow("not a move request");
  });
});

describe("worlds built from the committed channels, with the real probe", () => {
  it("replies with a legal choice that never switches and never uses a status move, and measures the cost", () => {
    const builder = createWorldBuilder({ engine, format_id: GOLDEN_FORMAT, scenario_id: scenarioId, hp_rounding: "ceil" });
    const worlds: { world: ScenarioWorld; id: string }[] = [];
    for (const { id, input } of cases()) {
      const built = builder.build(input);
      if (built.kind === "world" && id.endsWith("/1") === false) worlds.push({ world: built.world, id });
      if (worlds.length >= 40) break;
    }
    expect(worlds.length).toBeGreaterThanOrEqual(20);
    const real = createReplyModel({ engine, vocabulary, config, mechanics, data_class: "synthetic" }); // the default snapshot and probe
    const dex = engine.sim.Dex.mod(engine.sim.Dex.formats.get(GOLDEN_FORMAT).mod) as unknown as { moves: { get(id: string): { category: string } } };
    const times: number[] = [];
    for (const { world, id } of worlds) {
      const before = performance.now();
      const forcedBefore = real.counts().forced_slots;
      const result = real.reply(world);
      const forced = real.counts().forced_slots > forcedBefore;
      times.push(performance.now() - before);
      expect(result.choice, id).not.toContain("switch");
      // every named move is damaging (the request in the clone is the world's as written)
      const clone = engine.sim.Battle.fromJSON(world.battle.toJSON());
      const request = JSON.parse(JSON.stringify((clone.sides[world.seat === "p1" ? 1 : 0] as { activeRequest: object }).activeRequest)) as { active: { moves: { id: string }[] }[] };
      const legal = enumerateCandidates({ ...decodeRequest(JSON.stringify(request)), request_seq: 1 } as never).candidates.filter((c) => !c.pending).map((c) => c.choice);
      expect(legal, id).toContain(result.choice);
      for (const [slot, part] of result.choice.split(", ").entries()) {
        const words = part.split(" ");
        if (words[0] !== "move") continue;
        const id2 = request.active[slot]?.moves[Number(words[1]) - 1]?.id as string;
        // a status move is played only in an all-zero reply (the actor-rule choice) or in a slot with no damaging option
        if (dex.moves.get(id2).category === "Status" && result.kind === "max_damage") expect(forced, id).toBe(true);
      }
    }
    const counts = real.counts();
    expect(counts.replies).toBe(worlds.length);
    expect(counts.max_damage + counts.all_zero).toBe(counts.replies);
    const sorted = [...times].sort((a, b) => a - b);
    const mean = times.reduce((a, b) => a + b, 0) / times.length;
    const queries = counts.queries;
    console.log(
      `reply model: ${worlds.length} worlds, ${JSON.stringify(counts)}, ${(queries / worlds.length).toFixed(1)} queries and ${(counts.probe_runs / worlds.length).toFixed(1)} probe runs per world, ` +
        `mean ${mean.toFixed(0)} ms, median ${(sorted[Math.floor(sorted.length / 2)] as number).toFixed(0)} ms, p95 ${(sorted[Math.floor(sorted.length * 0.95)] as number).toFixed(0)} ms, max ${(sorted[sorted.length - 1] as number).toFixed(0)} ms per world`,
    );
  }, 240_000); // 40 worlds with the real probe, a few hundred ms each on a shared machine
});

describe("the formats the search identity names", () => {
  it("pins the reply model format and the shared value function's format with vectors", () => {
    // If one of these changes, bump the constant (ADR-026 section 5): the vectors are worked by hand from the formula.
    expect(REPLY_MODEL_FORMAT).toBe(1);
    expect(DAMAGE_VALUE_FORMAT).toBe(1);
    const rolls = (value: number): DamageAnswer["no_crit"] => Array.from({ length: 16 }, () => value) as DamageAnswer["no_crit"];
    const answer = (change: Partial<DamageAnswer> = {}): DamageAnswer => ({
      kind: "damage",
      no_crit: rolls(25),
      crit: rolls(50),
      crit_chance: { kind: "never" },
      hp_current: 100,
      hp_max: 100,
      ko: { kind: "unsupported", reason: "ko_state_change" },
      ...change,
    });
    const scoring = { ko_weight_ppm: 1_000_000, turn_order_discount_ppm: 500_000, crit_weight_ppm: 1_000_000 };
    expect(damageValue(answer(), scoring)).toBe(250_000); // 25 / 100
    expect(damageValue(answer({ no_crit: rolls(100) }), scoring)).toBe(2_000_000); // 100 / 100 plus the KO share, 16 / 16, times the weight
    expect(damageValue(answer({ crit_chance: { kind: "chance", numerator: 1, denominator: 4 } }), scoring)).toBe(312_500); // 3/4 * 25% + 1/4 * 50%
  });
});

describe("the type rule", () => {
  it("takes a world and nothing of the real game", () => {
    type Reply = ReturnType<typeof createReplyModel>["reply"];
    expectTypeOf<Parameters<Reply>[0]>().toEqualTypeOf<ScenarioWorld>();
    // @ts-expect-error a battle record is not a world
    expectTypeOf<BattleRecord>().toMatchTypeOf<Parameters<Reply>[0]>();
    // @ts-expect-error the omniscient log is not a world
    expectTypeOf<OmniscientLog>().toMatchTypeOf<Parameters<Reply>[0]>();
    // @ts-expect-error hidden verdicts are not a world
    expectTypeOf<HiddenVerdicts>().toMatchTypeOf<Parameters<Reply>[0]>();
  });
  it("is a code path of its own, not the damage-aware actor", () => {
    const source = readFileSync(new URL("../src/reply-model.ts", import.meta.url), "utf8")
      .split("\n")
      .filter((line) => !line.trimStart().startsWith("//"))
      .join("\n");
    expect(source).not.toMatch(/from "\.\/(damage-aware-actor|runner|player-channel|observation)\.ts"/);
    expect(source).not.toMatch(/PublicState|interpretChannel|OmniscientLog|BattleRecord/);
  });
});
