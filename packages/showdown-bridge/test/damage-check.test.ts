// The engine-backed damage check on the golden battle, with the probe scripted where the real one cannot be made to
// answer a given way (ticket #1132). The sweeps against the real probe are in `packages/opponent-model/test/damage-engine.test.ts`.

import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { interpretChannel } from "@areneon/public-state";
import type { EngineRequest } from "@areneon/schema/types/v1/engine-request.ts";
import { describe, expect, it } from "vitest";
import { readBeliefConfig } from "../src/belief-config.ts";
import { loadDamageAwareConfig } from "../src/damage-aware-config.ts";
import { DamageAwareError, ProbeRefusal } from "../src/damage-aware-view.ts";
import { type DamageCandidatePort, type DamageCheckContext, type DamageCheckInput, createDamageCheck, newDamageCheckCache } from "../src/damage-check.ts";
import type { PokemonSet } from "../src/engine.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { loadEngine } from "../src/index.ts";
import { loadMechanicsConfig } from "../src/mechanics-config.ts";
import { MechanicsProbeError } from "../src/mechanics-error.ts";
import { type MechanicsContext, probeMechanics } from "../src/mechanics-probe.ts";
import type { ParsedMechanicsQuery } from "../src/mechanics-query.ts";
import type { TeamVocabulary } from "../src/team-vocabulary.ts";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
const FIXTURES = join(ROOT, "fixtures", "showdown", GOLDEN_FORMAT);
const vocabulary = JSON.parse(readFileSync(join(FIXTURES, "team-vocabulary.json"), "utf8")) as TeamVocabulary;
const belief = readBeliefConfig(JSON.parse(readFileSync(join(ROOT, "config", "opponent-model", GOLDEN_FORMAT, "belief.damage-v1.json"), "utf8")), vocabulary);
const lines = (JSON.parse(readFileSync(join(FIXTURES, "channels", "p2.json"), "utf8")) as { channel: { lines: string[] } }).channel.lines;
const team = JSON.parse(readFileSync(join(FIXTURES, "synthetic-team.json"), "utf8")) as PokemonSet[];

const engine = loadEngine();
const context = (change: Partial<DamageCheckContext> = {}): DamageCheckContext => ({ engine, vocabulary, config: loadDamageAwareConfig("v1", vocabulary), mechanics: loadMechanicsConfig(GOLDEN_FORMAT), belief, data_class: "synthetic", ...change });
const make = (change: Partial<DamageCheckContext> = {}, extra: string[] = []) => createDamageCheck(context(change), { player: "p2", lines: [...lines, ...extra], team });

const NATURE = "serious";
const spread = { hp: 4, atk: 0, def: 0, spa: 0, spd: 0, spe: 4 };
const candidate = (change: Partial<DamageCandidatePort> = {}): DamageCandidatePort => ({ species: "charizard", forme: "charizard", mega: false, ability: "blaze", item: "charcoal", nature: NATURE, stat_points: spread, ...change });

describe("createDamageCheck", () => {
  it("finds the golden battle's isolated hits as events about opponent slots", () => {
    const port = make();
    expect(port?.events.length).toBeGreaterThan(5);
    expect(port?.events.every((e) => e.role === "attacker" || e.role === "defender")).toBe(true);
    const counts = port?.counts();
    expect(counts?.hits).toBeGreaterThanOrEqual(counts?.events ?? 0);
    expect(Object.keys(counts?.not_isolated ?? {}).length).toBeGreaterThan(0);
  });

  it("is null for a config without damage evidence and for a channel the reader flagged", () => {
    const { damage_evidence: _, ...without } = belief;
    expect(make({ belief: without })).toBeNull();
    expect(make({}, ["|-damage|p9z: Nobody|50/100"])).toBeNull();
  });

  it("refuses a reviewed non-isolating target that is no move target of the engine", () => {
    const rules = belief.damage_evidence as NonNullable<typeof belief.damage_evidence>;
    const bad = { ...belief, damage_evidence: { ...rules, non_isolating: { ...rules.non_isolating, targets: ["notATarget"] } } };
    expect(() => make({ belief: bad })).toThrow(DamageAwareError);
  });
});

describe("check", () => {
  const first = (port: ReturnType<typeof make>) => (port?.events.find((e) => e.role === "attacker") ?? port?.events[0]) as NonNullable<ReturnType<typeof make>>["events"][number];

  it("answers a real candidate with a verdict, remembers it, and counts every call", () => {
    const cache = newDamageCheckCache();
    const port = make({ cache }) as NonNullable<ReturnType<typeof make>>;
    const event = port.events.find((e) => e.role === "attacker" && e.slot === 1) as (typeof port.events)[number];
    const verdict = port.check(event, candidate());
    expect(typeof verdict).toBe("boolean");
    const runs = port.counts().probe_runs;
    expect(port.check(event, candidate())).toBe(verdict);
    expect(port.counts()).toMatchObject({ calls: 2, probe_runs: runs });
    // a second port of the same game and seat shares the verdicts
    const again = make({ cache }) as NonNullable<ReturnType<typeof make>>;
    expect(again.check(event, candidate())).toBe(verdict);
    expect(again.counts().probe_runs).toBe(0);
  });

  it("is null for what it does not state: a Mega forme, an unknown forme or nature, an attacker's move outside the forme's pool", () => {
    const port = make() as NonNullable<ReturnType<typeof make>>;
    const event = first(port);
    expect(port.check(event, candidate({ mega: true }))).toBeNull();
    expect(port.check(event, candidate({ forme: "notaspecies" }))).toBeNull();
    expect(port.check(event, candidate({ nature: "notanature" }))).toBeNull();
    const attacker = port.events.find((e) => e.role === "attacker") as (typeof port.events)[number];
    expect(port.check(attacker, candidate({ forme: "pikachu", species: "pikachu" }))).toBeNull();
  });

  it("is null when the probe cannot answer or the engine refuses the declared state, and propagates any other failure", () => {
    const failing = (error: unknown) => {
      const port = make({ probe: () => { throw error; } }) as NonNullable<ReturnType<typeof make>>;
      return () => port.check(first(port), candidate());
    };
    expect(failing(new ProbeRefusal("refused"))()).toBeNull();
    expect(failing(new MechanicsProbeError("the engine refused"))()).toBeNull();
    expect(failing(new RangeError("a defect"))).toThrow(RangeError);
    const unsupported = (query: ParsedMechanicsQuery, context: MechanicsContext) => {
      const outcome = probeMechanics(context, query);
      return query.query.kind === "damage" ? { ...outcome, result: { ...outcome.result, kind: "unsupported" as const, reason: "query_kind" as const, position: null } } : outcome;
    };
    const port = make({ probe: (c, q) => unsupported(q, c) as never }) as NonNullable<ReturnType<typeof make>>;
    expect(port.check(first(port), candidate())).toBeNull();
  });

  it("an attacker whose damage depends on where in its shown hit-point range it stands gets no constraint, whatever the move", () => {
    // A stand-in probe that reads the attacker's declared hit points, as Eruption, Water Spout or Reversal do. The golden battle's
    // Flamethrower took 36 of the second player's Blastoise's hit points from a Charizard shown at 33%.
    const rolling = (rolls: (hp: number | null) => number) => (c: MechanicsContext, q: ParsedMechanicsQuery) => {
      const outcome = probeMechanics(c, q);
      if (q.query.kind !== "damage" || outcome.result.kind !== "computed" || outcome.result.answer.kind !== "damage") return outcome;
      const fill = Array.from({ length: 16 }, () => rolls(q.query.kind === "damage" ? q.query.attacker.hp : null));
      return { ...outcome, result: { ...outcome.result, answer: { ...outcome.result.answer, no_crit: fill, crit: fill } } } as never;
    };
    const verdict = (rolls: (hp: number | null) => number) => {
      const port = make({ probe: rolling(rolls) }) as NonNullable<ReturnType<typeof make>>;
      const hit = port.hits.find((h) => h.move === "flamethrower" && h.reason === null && h.after.num === 88) as (typeof port.hits)[number];
      return port.check({ id: `hit${hit.frame}`, slot: hit.attacker.index, role: "attacker" }, candidate());
    };
    expect(verdict((hp) => (hp === null || hp % 2 === 0 ? 36 : 10))).toBeNull();
    // negative controls: the same answers at both ends are used (true when they match the 36, false when they do not)
    expect(verdict(() => 36)).toBe(true);
    expect(verdict(() => 10)).toBe(false);
  });

  it("judges a defender-role event: some roll of the real probe takes the foe from its shown percentage to the next", () => {
    const port = make() as NonNullable<ReturnType<typeof make>>;
    const event = port.events.find((e) => e.role === "defender") as (typeof port.events)[number];
    expect(typeof port.check(event, candidate())).toBe("boolean");
  });

  it("is null when the probe cannot say a candidate's maximum hit points or its damage (defender role too)", () => {
    const unsupported = (kinds: string[]) => (c: MechanicsContext, q: ParsedMechanicsQuery) => {
      const outcome = probeMechanics(c, q);
      return kinds.includes(q.query.kind) ? ({ ...outcome, result: { ...outcome.result, kind: "unsupported", reason: "query_kind", position: null } } as never) : outcome;
    };
    for (const kinds of [["stats"], ["damage"]]) {
      const port = make({ probe: unsupported(kinds) }) as NonNullable<ReturnType<typeof make>>;
      for (const role of ["attacker", "defender"] as const) {
        const event = port.events.find((e) => e.role === role) as (typeof port.events)[number];
        expect(port.check(event, candidate({ forme: role === "attacker" ? "charizard" : "charizard" }))).toBeNull();
      }
    }
  });

  it("has nothing to say about a channel with no request", () => {
    expect(createDamageCheck(context(), { player: "p2", lines: lines.slice(0, 12), team })).toBeNull();
  });

  it("tells the observer the rolls and the two ranges it judged", () => {
    const seen: unknown[] = [];
    const port = make({ observe: (_e, rolls, before, after) => seen.push([rolls.length, before.lo <= before.hi, after.lo <= after.hi]) }) as NonNullable<ReturnType<typeof make>>;
    port.check(port.events.find((e) => e.role === "attacker" && e.slot === 1) as (typeof port.events)[number], candidate());
    expect(seen).toEqual([[16, true, true]]);
  });

  it("gives no constraint, and counts it, for an own Pokemon the team does not name once or names twice", () => {
    for (const given of [[], [...team, ...team]]) {
      const other = createDamageCheck(context(), { player: "p2", lines, team: given }) as NonNullable<ReturnType<typeof make>>;
      const event = other.events.find((e) => e.role === "attacker" && e.slot === 1) as (typeof other.events)[number];
      expect(other.check(event, candidate())).toBeNull();
      expect(other.counts().own_unstated).toBe(1);
    }
  });

  describe("an own Pokemon that has Mega Evolved", () => {
    // The golden battle has no Mega, so the reader's run is doctored: the own defender of one hit is marked evolved in every frame
    // and the requests show the forme and item the case names. The probe is the real one; `observe` tells whether the check judged.
    const dex = engine.sim.Dex.mod(engine.sim.Dex.formats.get(GOLDEN_FORMAT).mod);
    const stones = () => (dex.items as unknown as { all(): { id: string; megaStone?: Record<string, string> }[] }).all();
    const stoneOf = (species: string): string => stones().find((i) => i.megaStone?.[species] !== undefined)?.id as string;
    const setup = () => {
      const plain = make() as NonNullable<ReturnType<typeof make>>;
      const event = plain.events.find((e) => e.role === "attacker" && e.slot === 1) as (typeof plain.events)[number];
      const hit = plain.hits.find((h) => `hit${h.frame}` === event.id) as (typeof plain.hits)[number];
      const name = hit.state.own.roster[hit.defender.index]?.name as string;
      const set = team.find((t) => t.name === name || t.species === name) as PokemonSet;
      return { event, name, set };
    };
    const evolved = (name: string, details: string, item: string, sets: PokemonSet[] = team) => {
      const run = structuredClone(interpretChannel("p2", lines));
      for (const frame of run.frames) for (const m of frame.state.own.roster) if (m.name === name) m.mega = true;
      for (const request of run.requests as unknown as EngineRequest[]) for (const entry of request.side.pokemon) if (entry.ident.endsWith(`: ${name}`)) Object.assign(entry, { details, item });
      const seen: unknown[] = [];
      const port = createDamageCheck(context({ observe: () => seen.push(1) }), { player: "p2", lines, team: sets, run }) as NonNullable<ReturnType<typeof make>>;
      return { port, seen };
    };
    const ask = (port: NonNullable<ReturnType<typeof make>>, event: ReturnType<typeof setup>["event"]) => port.check(event, candidate());

    it("is stated when its shown forme is the one its stone makes of the set's species, on the turn it evolves too (the request still shows the base forme)", () => {
      const { event, name, set } = setup();
      const stone = stoneOf(set.species);
      const mega = `${set.species}-Mega`;
      for (const shown of [mega, set.species]) {
        const { port, seen } = evolved(name, `${shown}, L50`, stone);
        expect(typeof ask(port, event)).toBe("boolean");
        expect(seen).toHaveLength(1);
        expect(port.counts().own_unstated).toBe(0);
      }
    });

    it("is declined, and counted in own_unstated, for a stone whose Mega forme is not the shown one or is not the species'", () => {
      const { event, name, set } = setup();
      const other = stones().find((i) => i.megaStone !== undefined && i.megaStone[set.species] === undefined)?.id as string;
      for (const [details, item] of [[`${set.species}-Mega, L50`, other], [`Pikachu-Mega, L50`, stoneOf(set.species)]] as const) {
        const { port, seen } = evolved(name, details, item);
        expect(ask(port, event)).toBeNull();
        expect(seen).toHaveLength(0);
        expect(port.counts().own_unstated).toBe(1);
      }
    });

    it("is declined, and counted in own_unstated, when the team's set already is the Mega forme", () => {
      const { event, name, set } = setup();
      const sets = team.map((t) => (t === set ? { ...t, species: `${t.species}-Mega` } : t));
      const { port, seen } = evolved(name, `${set.species}-Mega, L50`, stoneOf(set.species), sets);
      expect(ask(port, event)).toBeNull();
      expect(seen).toHaveLength(0);
      expect(port.counts().own_unstated).toBe(1);
    });
  });

  it("a cache belongs to one seat", () => {
    const cache = newDamageCheckCache();
    make({ cache });
    expect(() => createDamageCheck(context({ cache }), { player: "p1", lines, team })).toThrow(DamageAwareError);
  });

  it("takes nothing of the opponent's team, an omniscient log, hidden verdicts or the seed (type test)", () => {
    const ok = { player: "p2", lines, team };
    // @ts-expect-error: the opponent's real team is not an input
    const a: DamageCheckInput = { ...ok, opponent_team: [] };
    // @ts-expect-error: nor the omniscient log
    const b: DamageCheckInput = { ...ok, omniscient_log: [] };
    // @ts-expect-error: nor the hidden verdicts of the clone judge
    const c: DamageCheckInput = { ...ok, hidden_verdicts: [] };
    // @ts-expect-error: nor the seed
    const d: DamageCheckInput = { ...ok, seed: "sodium,0" };
    // @ts-expect-error: nor a battle record
    const e: DamageCheckInput = { ...ok, record: {} };
    expect([a, b, c, d, e]).toHaveLength(5);
    // a smuggled field is never read: the port is the same
    expect(createDamageCheck(context(), { ...ok, opponent_team: team, seed: "sodium,0" } as DamageCheckInput)?.events).toEqual(make()?.events);
    const keys: (keyof DamageCheckInput)[] = ["player", "lines", "team", "run"];
    expect(keys).toHaveLength(4);
  });
});
