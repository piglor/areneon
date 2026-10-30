// The engine-backed speed-order check on the golden battle, with the probe's answer scripted where the real one cannot be
// made to answer a given way (ticket #1133). The sweeps against the real probe are in `packages/opponent-model/test/speed-engine.test.ts`.

import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { readBeliefConfig } from "../src/belief-config.ts";
import { loadDamageAwareConfig } from "../src/damage-aware-config.ts";
import { DamageAwareError, ProbeRefusal } from "../src/damage-aware-view.ts";
import { newDamageCheckCache } from "../src/damage-check.ts";
import type { PokemonSet } from "../src/engine.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { loadEngine } from "../src/index.ts";
import { loadMechanicsConfig } from "../src/mechanics-config.ts";
import { MechanicsProbeError } from "../src/mechanics-error.ts";
import { type MechanicsContext, probeMechanics } from "../src/mechanics-probe.ts";
import type { ParsedMechanicsQuery } from "../src/mechanics-query.ts";
import { type SpeedCheckContext, createSpeedCheck } from "../src/speed-check.ts";
import type { DamageCandidatePort } from "../src/damage-check.ts";
import type { TeamVocabulary } from "../src/team-vocabulary.ts";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
const FIXTURES = join(ROOT, "fixtures", "showdown", GOLDEN_FORMAT);
const vocabulary = JSON.parse(readFileSync(join(FIXTURES, "team-vocabulary.json"), "utf8")) as TeamVocabulary;
const belief = readBeliefConfig(JSON.parse(readFileSync(join(ROOT, "config", "opponent-model", GOLDEN_FORMAT, "belief.speed-v1.json"), "utf8")), vocabulary);
const lines = (JSON.parse(readFileSync(join(FIXTURES, "channels", "p2.json"), "utf8")) as { channel: { lines: string[] } }).channel.lines;
const team = JSON.parse(readFileSync(join(FIXTURES, "synthetic-team.json"), "utf8")) as PokemonSet[];

const engine = loadEngine();
const context = (change: Partial<SpeedCheckContext> = {}): SpeedCheckContext => ({ engine, vocabulary, config: loadDamageAwareConfig("v1", vocabulary), mechanics: loadMechanicsConfig(GOLDEN_FORMAT), belief, data_class: "synthetic", ...change });
const make = (change: Partial<SpeedCheckContext> = {}, extra: string[] = []) => createSpeedCheck(context(change), { player: "p2", lines: [...lines, ...extra], team }) as NonNullable<ReturnType<typeof createSpeedCheck>>;

const candidate = (change: Partial<DamageCandidatePort> = {}): DamageCandidatePort => ({ species: "charizard", forme: "charizard", mega: false, ability: "blaze", item: "charcoal", nature: "serious", stat_points: { hp: 4, atk: 0, def: 0, spa: 0, spd: 0, spe: 4 }, ...change });

/** The real probe with the groups of every speed-order answer replaced. */
const groups = (made: number[][]) => (c: MechanicsContext, q: ParsedMechanicsQuery) => {
  const outcome = probeMechanics(c, q);
  if (q.query.kind !== "speed_order" || outcome.result.kind !== "computed" || outcome.result.answer.kind !== "speed_order") return outcome;
  return { ...outcome, result: { ...outcome.result, answer: { ...outcome.result.answer, groups: made } } } as never;
};

describe("createSpeedCheck", () => {
  it("finds the golden battle's isolated orders as events about opponent slots", () => {
    const port = make();
    expect(port.events.length).toBeGreaterThan(1);
    expect(port.events.every((e) => e.role === "first" || e.role === "second")).toBe(true);
    expect(port.events.some((e) => e.role === "first")).toBe(true);
    expect(port.events.some((e) => e.role === "second")).toBe(true);
    const counts = port.counts();
    expect(counts.pairs).toBeGreaterThanOrEqual(counts.events);
    expect(counts.events).toBe(port.events.length);
  });

  it("is null for a config without speed evidence and for a channel the reader flagged", () => {
    const { speed_evidence: _, ...without } = belief;
    expect(createSpeedCheck(context({ belief: without }), { player: "p2", lines, team })).toBeNull();
    expect(createSpeedCheck(context(), { player: "p2", lines: [...lines, "|-damage|p9z: Nobody|50/100"], team })).toBeNull();
  });

  it("a cache belongs to one seat of one game", () => {
    const cache = newDamageCheckCache();
    make({ cache });
    expect(() => createSpeedCheck(context({ cache }), { player: "p1", lines, team })).toThrow(DamageAwareError);
  });
});

describe("check", () => {
  const eventOf = (port: ReturnType<typeof make>, role: "first" | "second") => port.events.find((e) => e.role === role && e.slot === 1) as (typeof port.events)[number];

  it("passes the candidates whose probe order is the observed one, an exact tie among them, and rules out the others", () => {
    // own faster (the own Pokemon's group is earlier) / the foe faster / an exact tie
    const verdicts = (made: number[][], role: "first" | "second") => {
      const port = make({ probe: groups(made) });
      return port.check(eventOf(port, role), candidate());
    };
    expect([verdicts([[0], [1]], "second"), verdicts([[1], [0]], "second"), verdicts([[0, 1]], "second")]).toEqual([true, false, true]);
    expect([verdicts([[0], [1]], "first"), verdicts([[1], [0]], "first"), verdicts([[0, 1]], "first")]).toEqual([false, true, true]);
  });

  it("asks the probe about the Speed-relevant fields alone: candidates that differ elsewhere share one answer", () => {
    const port = make();
    const event = eventOf(port, "second");
    const a = port.check(event, candidate());
    const runs = port.counts().probe_runs;
    expect(port.check(event, candidate({ stat_points: { hp: 0, atk: 0, def: 8, spa: 0, spd: 0, spe: 4 } }))).toBe(a);
    expect(port.counts().probe_runs).toBe(runs);
    // a different Speed is another question
    port.check(event, candidate({ stat_points: { hp: 4, atk: 0, def: 0, spa: 0, spd: 0, spe: 32 } }));
    expect(port.counts().probe_runs).toBeGreaterThan(runs);
  });

  it("remembers a verdict, counts every call, and a second port of the game shares them", () => {
    const cache = newDamageCheckCache();
    const port = make({ cache });
    const event = eventOf(port, "first");
    const verdict = port.check(event, candidate());
    expect(typeof verdict).toBe("boolean");
    const runs = port.counts().probe_runs;
    expect(port.check(event, candidate())).toBe(verdict);
    expect(port.counts()).toMatchObject({ calls: 2, probe_runs: runs });
    const again = make({ cache });
    expect(again.check(eventOf(again, "first"), candidate())).toBe(verdict);
    expect(again.counts().probe_runs).toBe(0);
  });

  it("a faster candidate is more likely first: the verdicts of an extreme slow and an extreme fast Speed are not both the same on every event", () => {
    const port = make();
    const slow = candidate({ nature: "relaxed", item: null, stat_points: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 } });
    const fast = candidate({ nature: "timid", item: "choicescarf", stat_points: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 32 } });
    const pairs = port.events.map((e) => [port.check(e, slow), port.check(e, fast)]);
    // the foe acted first in some events and second in others, so some candidate is ruled out unless it ties
    expect(pairs.flat().every((v) => v === null || typeof v === "boolean")).toBe(true);
    expect(pairs.some(([s, f]) => s !== f)).toBe(true);
  });

  it("is null for what it does not state: a Mega forme, an unknown forme or nature, a move outside the forme's pool", () => {
    const port = make();
    const event = eventOf(port, "second");
    expect(port.check(event, candidate({ mega: true }))).toBeNull();
    expect(port.check(event, candidate({ forme: "notaspecies" }))).toBeNull();
    expect(port.check(event, candidate({ nature: "notanature" }))).toBeNull();
    expect(port.check(event, candidate({ forme: "pikachu", species: "pikachu" }))).toBeNull();
  });

  it("is null when the probe cannot answer or the engine refuses the declared state, and propagates any other failure", () => {
    const failing = (error: unknown) => {
      const port = make({ probe: () => { throw error; } });
      return () => port.check(eventOf(port, "second"), candidate());
    };
    expect(failing(new ProbeRefusal("refused"))()).toBeNull();
    expect(failing(new MechanicsProbeError("the engine refused"))()).toBeNull();
    expect(failing(new RangeError("a defect"))).toThrow(RangeError);
    // a speed order the probe cannot queue, or one that is unsupported (a random draw)
    expect(failing(new MechanicsProbeError("the action's priority is not a whole number"))()).toBeNull();
    const unsupported = (c: MechanicsContext, q: ParsedMechanicsQuery) => {
      const outcome = probeMechanics(c, q);
      return q.query.kind === "speed_order" ? ({ ...outcome, result: { ...outcome.result, kind: "unsupported", reason: "random_effect", position: null } } as never) : outcome;
    };
    const port = make({ probe: unsupported });
    expect(port.check(eventOf(port, "second"), candidate())).toBeNull();
  });
});
