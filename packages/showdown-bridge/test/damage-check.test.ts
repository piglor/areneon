// The engine-backed damage check on the golden battle, with the probe scripted where the real one cannot be made to
// answer a given way (ticket #1132). The sweeps against the real probe are in `packages/opponent-model/test/damage-engine.test.ts`.

import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { readBeliefConfig } from "../src/belief-config.ts";
import { loadDamageAwareConfig } from "../src/damage-aware-config.ts";
import { DamageAwareError, ProbeRefusal } from "../src/damage-aware-view.ts";
import { type DamageCandidatePort, type DamageCheckContext, createDamageCheck, newDamageCheckCache } from "../src/damage-check.ts";
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

  it("tells the observer the rolls and the two ranges it judged", () => {
    const seen: unknown[] = [];
    const port = make({ observe: (_e, rolls, before, after) => seen.push([rolls.length, before.lo <= before.hi, after.lo <= after.hi]) }) as NonNullable<ReturnType<typeof make>>;
    port.check(port.events.find((e) => e.role === "attacker" && e.slot === 1) as (typeof port.events)[number], candidate());
    expect(seen).toEqual([[16, true, true]]);
  });

  it("fails closed on an own Pokemon the team does not hold", () => {
    const other = createDamageCheck(context(), { player: "p2", lines, team: [] }) as NonNullable<ReturnType<typeof make>>;
    const event = other.events.find((e) => e.role === "attacker" && e.slot === 1) as (typeof other.events)[number];
    expect(() => other.check(event, candidate())).toThrow(DamageAwareError);
  });
});
