// The search's world sampler (ADR-026 slice B) with a stand-in port: the mapping of a world onto the builder's
// input, the attempt/index rule, the minimum valid worlds, every count, and the privacy boundary. The real
// opponent model behind the port is tested in packages/team-dataset (search-worlds.test.ts).
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, expectTypeOf, it } from "vitest";
import type { PublicFlag, PublicSide } from "@areneon/public-state";
import { assumedHp, type WorldSetLike } from "../src/damage-aware-assumed-set.ts";
import type { WorldLike } from "../src/damage-aware-worlds.ts";
import type { Engine, PokemonSet } from "../src/engine.ts";
import { GOLDEN_FORMAT, goldenTeamVocabulary } from "../src/golden.ts";
import { type BattleRecord, type HiddenVerdicts, loadEngine, type OmniscientLog, type PlayerChannel } from "../src/index.ts";
import { loadMechanicsConfig } from "../src/mechanics-config.ts";
import type { TrappingCheckPort } from "../src/trapping-check.ts";
import { checkScenario } from "../src/mechanics-scenario.ts";
import { createWorldBuilder, type WorldBuilder, type WorldContext, type WorldInput, type WorldRefusal } from "../src/world-builder.ts";
import {
  createWorldSampler,
  EVIDENCE_CLASSES,
  evidenceClass,
  mapWorld,
  SEARCH_HP_ROUNDING,
  searchStream,
  type WorldRequest,
  type WorldSamplePort,
  type WorldSamplerSettings,
  type WorldSource,
  type WorldSourceResult,
  WorldMappingError,
} from "../src/world-sampler.ts";
import { loadWorldWriters } from "../src/world-writers.ts";
import { type Case, cases, toId } from "./world-builder-support.ts";

let engine: Engine;
let context: WorldContext;
let builder: WorldBuilder;
const vocabulary = goldenTeamVocabulary();
beforeAll(() => {
  engine = loadEngine();
  context = { engine, format_id: GOLDEN_FORMAT, scenario_id: checkScenario(engine, loadMechanicsConfig(GOLDEN_FORMAT)), hp_rounding: SEARCH_HP_ROUNDING, writers: loadWorldWriters(engine, GOLDEN_FORMAT) };
});

const STREAM = Uint8Array.from({ length: 32 }, (_, i) => i + 1);
const settings: WorldSamplerSettings = { worlds: 8, attempts: 4, min_valid: 4 };

const request = ({ input }: Case): WorldRequest => {
  const { opponent: _opponent, ...rest } = input;
  return rest;
};

/** The world of a case's TRUE opponent team, as a sampled world would be (the test's stand-in for the model's draw). */
function trueWorld(c: Case): WorldLike {
  const members = new Map(c.input.opponent.members.map((m) => [m.roster_index, m.set]));
  return {
    slots: c.input.state.opponent.roster.map((_, index) => {
      const set = members.get(index) as PokemonSet | undefined;
      if (set === undefined) return { index, brought: false, set: null };
      const like: WorldSetLike = {
        species: toId(set.species),
        forme: toId(set.species),
        item: set.item === "" ? null : toId(set.item),
        ability: toId(set.ability),
        nature: toId(set.nature),
        stat_points: set.evs as WorldSetLike["stat_points"],
        moves: set.moves.map(toId),
        level: set.level,
        gender: set.gender === "" ? null : set.gender,
      };
      return { index, brought: true, set: like };
    }),
  };
}

interface Plan {
  /** What the draw of index j does: a world, a refusal code thrown like the model's, or an error. */
  draw: (j: number) => { world: WorldLike; redraws: number } | { throws: unknown };
}
const port = (plan: Plan, extra: Partial<WorldSamplePort> = {}, opened: (input: { opponent: PublicSide; flags: readonly PublicFlag[]; trapping?: TrappingCheckPort | null }) => WorldSourceResult | null = () => null): WorldSamplePort & { drawn: number[] } => {
  const drawn: number[] = [];
  const source: WorldSource = {
    draw(_stream, index) {
      drawn.push(index);
      const out = plan.draw(index);
      if ("throws" in out) throw out.throws;
      return out;
    },
  };
  return {
    drawn,
    identity_hash: "ab".repeat(32),
    prior_hash: "cd".repeat(32),
    belief_config_hash: "ef".repeat(32),
    prior_format_id: GOLDEN_FORMAT,
    prior_kind: "generator-matched",
    prior_label: "test label",
    open: (input) => opened(input) ?? { kind: "source", source },
    ...extra,
  };
};
const beliefError = (code: string): Error => Object.assign(new Error(code), { name: "BeliefError", code });

let good: Case;
beforeAll(() => {
  const probe = createWorldBuilder(context);
  good = cases().find((c) => probe.build(c.input).kind === "world") as Case;
});
const make = (p: WorldSamplePort, s: WorldSamplerSettings = settings, b?: WorldBuilder) => {
  builder = b ?? createWorldBuilder(context);
  return createWorldSampler({ engine, vocabulary, format_id: GOLDEN_FORMAT, port: p, builder, settings: s });
};
const refusal = (redraw: boolean): WorldRefusal => ({ code: redraw ? "v2_request" : "v5_mismatch", detail: "d", check: redraw ? "V2" : "V5", state_class: redraw ? null : "W", item: null, redraw });
/** A builder that answers from a script (a refusal or a pass), so the attempt rule is tested apart from the real checks. */
const scripted = (answer: (call: number) => WorldRefusal | null): WorldBuilder => {
  const real = createWorldBuilder(context);
  const passing = real.build(good.input);
  let call = 0;
  return {
    build() {
      const out = answer(call++);
      return out === null ? passing : { kind: "refused", refusal: out };
    },
    counts: () => real.counts(),
  };
};

describe("the mapping of a world onto the builder's input", () => {
  it("builds the true team's world exactly as the true team's members do, on every committed move request", () => {
    const direct = createWorldBuilder(context);
    const mapped = createWorldBuilder(context);
    let same = 0;
    for (const c of cases()) {
      const a = direct.build(c.input);
      const b = mapped.build({ ...request(c), opponent: { members: mapWorld({ engine, vocabulary }, GOLDEN_FORMAT, trueWorld(c)) } });
      // A world is valid or refused the same way whichever way the opponent arrives (an evolved Mega's ability is the one place the names differ, and V1 judges it again).
      if (a.kind === "world") {
        expect(b.kind, c.id).toBe("world");
        same += 1;
      } else if (a.refusal.code === "state_unwritten" || a.refusal.code === "reader_flagged") expect(b.kind === "refused" && b.refusal.code, c.id).toBe(a.refusal.code);
    }
    expect(same).toBeGreaterThanOrEqual(700);
  });

  it("refuses a brought slot with no set and a set outside the vocabulary as a defect", () => {
    const world = trueWorld(good);
    const broken: WorldLike = { slots: world.slots.map((s, i) => (i === good.input.opponent.members[0]?.roster_index ? { ...s, set: null } : s)) };
    expect(() => mapWorld({ engine, vocabulary }, GOLDEN_FORMAT, broken)).toThrow(WorldMappingError);
    const other: WorldLike = { slots: world.slots.map((s) => (s.set === null ? s : { ...s, set: { ...s.set, nature: "notanature" } })) };
    expect(() => mapWorld({ engine, vocabulary }, GOLDEN_FORMAT, other)).toThrow(WorldMappingError);
  });

  it("takes the first listed ability for an ability the preview species cannot hold (an evolved Mega's), and a Mega forme as its base holding the stone", () => {
    const entry = vocabulary.roster.find((r) => r.required_item !== undefined) as (typeof vocabulary.roster)[number];
    const baseId = vocabulary.roster.find((r) => r.id !== entry.id && entry.id.startsWith(r.id) && r.required_item === undefined)?.id as string;
    const base = vocabulary.roster.find((r) => r.id === baseId) as (typeof vocabulary.roster)[number];
    const set: WorldSetLike = { species: baseId, forme: entry.id, item: toId(entry.required_item as string), ability: entry.abilities[0] === base.abilities[0] ? "zzz" : (entry.abilities[0] as string), nature: vocabulary.natures[0]?.id as string, stat_points: { hp: 1, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 }, moves: [base.moves[0] as string], level: 50, gender: null };
    const [member] = mapWorld({ engine, vocabulary }, GOLDEN_FORMAT, { slots: [{ index: 3, brought: true, set }] });
    expect(member?.roster_index).toBe(3);
    expect(toId(member?.set.species ?? "")).toBe(baseId);
    expect(base.abilities.map((a) => engine.sim.Dex.mod(engine.sim.Dex.formats.get(GOLDEN_FORMAT).mod).abilities.get(a).name)).toContain(member?.set.ability);
  });
});

describe("hit points of a searched world", () => {
  it("ceil inverts the Champions display floor(100 * hp / max) for every maximum of 100 or more, so V3 can hold", () => {
    expect(SEARCH_HP_ROUNDING).toBe("ceil");
    for (let max = 100; max <= 800; max += 1) {
      for (let shown = 1; shown <= 100; shown += 1) expect(Math.floor((100 * assumedHp(shown, max, SEARCH_HP_ROUNDING)) / max)).toBe(shown);
    }
  });
});

describe("the attempt rule: sampler index i * R + r", () => {
  it("takes the first attempt that passes as the world, never reuses an index, and counts a clean run", () => {
    const p = port({ draw: () => ({ world: trueWorld(good), redraws: 2 }) });
    const sampler = make(p);
    const out = sampler.sample(request(good), STREAM);
    expect(out).toMatchObject({ kind: "worlds", searched: true, prior_label: "test label" });
    if (out.kind !== "worlds") throw new Error("unreachable");
    expect(out.valid.map((w) => [w.world, w.attempt, w.index])).toEqual(Array.from({ length: 8 }, (_, i) => [i, 0, i * 4]));
    expect(p.drawn).toEqual(Array.from({ length: 8 }, (_, i) => i * 4));
    const c = sampler.counts();
    expect(c).toMatchObject({ decisions: 1, searched: 1, below_min_valid: 0, worlds_asked: 8, worlds_valid: 8, worlds_refused: 0, attempts: 8, redraws: 0, legality_redraws: 16, identity_hash: p.identity_hash, prior_kind: "generator-matched" });
    expect(c.valid_histogram[8]).toBe(1);
    expect(Object.values(c.by_evidence).reduce((a, r) => a + r.decisions, 0)).toBe(1);
  });

  it("redraws a world after a refusal a redraw can fix, at the next index of that world, and takes the first attempt that passes", () => {
    // Every world's attempts 0 and 1 are refused (a V2 trapping refusal); attempt 2 passes.
    const p = port({ draw: () => ({ world: trueWorld(good), redraws: 0 }) });
    const sampler = make(p, settings, scripted((call) => (call % 3 < 2 ? refusal(true) : null)));
    const out = sampler.sample(request(good), STREAM);
    if (out.kind !== "worlds") throw new Error("unreachable");
    expect(out.valid.map((w) => [w.world, w.attempt, w.index])).toEqual(Array.from({ length: 8 }, (_, i) => [i, 2, i * 4 + 2]));
    expect(p.drawn).toEqual(Array.from({ length: 8 }, (_, i) => [i * 4, i * 4 + 1, i * 4 + 2]).flat());
    expect(new Set(p.drawn).size).toBe(p.drawn.length);
    const c = sampler.counts();
    expect(c).toMatchObject({ attempts: 24, redraws: 16, worlds_valid: 8, worlds_refused: 0, redrawn: { "V2:-": 16 } });
  });

  it("refuses a world no attempt makes valid, counting it by code, check and class, and plays below W_min", () => {
    const p = port({ draw: () => ({ world: trueWorld(good), redraws: 0 }) });
    const sampler = make(p, settings, scripted(() => refusal(true)));
    const out = sampler.sample(request(good), STREAM);
    if (out.kind !== "worlds") throw new Error("unreachable");
    expect(out.searched).toBe(false);
    expect(out.valid).toHaveLength(0);
    expect(out.refused).toHaveLength(8);
    expect(out.refused[0]).toMatchObject({ world: 0, attempts: 4, draw_code: null });
    expect(p.drawn).toEqual(Array.from({ length: 32 }, (_, j) => j));
    const c = sampler.counts();
    expect(c).toMatchObject({ below_min_valid: 1, searched: 0, worlds_refused: 8, worlds_valid: 0, attempts: 32, redraws: 24 });
    expect(c.refusals.v2_request).toBe(8);
    expect(c.refusal_check_class["V2:-"]).toBe(8);
    expect(c.redrawn["V2:-"]).toBe(24);
    expect(c.valid_histogram[0]).toBe(1);
  });

  it("searches a decision with exactly W_min valid worlds and not one with fewer", () => {
    // World i passes only for i < 4 (the first attempt of the others is a final refusal).
    const answer = (call: number): WorldRefusal | null => (call < 4 ? null : refusal(false));
    const four = make(port({ draw: () => ({ world: trueWorld(good), redraws: 0 }) }), settings, scripted(answer));
    expect(four.sample(request(good), STREAM)).toMatchObject({ kind: "worlds", searched: true });
    expect(four.counts()).toMatchObject({ searched: 1, worlds_valid: 4, worlds_refused: 4, refusals: { v5_mismatch: 4 }, refusal_check_class: { "V5:W": 4 } });
    const three = make(port({ draw: () => ({ world: trueWorld(good), redraws: 0 }) }), settings, scripted((call) => (call < 3 ? null : refusal(false))));
    expect(three.sample(request(good), STREAM)).toMatchObject({ kind: "worlds", searched: false });
    expect(three.counts().valid_histogram[3]).toBe(1);
  });

  it("does not redraw a refusal a redraw cannot fix", () => {
    const sampler = make(port({ draw: () => ({ world: trueWorld(good), redraws: 0 }) }));
    const flagged: WorldRequest = { ...request(good), flags: [{ index: 1, line: "|x|", reason: "r" }] };
    const out = sampler.sample(flagged, STREAM);
    if (out.kind !== "worlds") throw new Error("unreachable");
    expect(out.refused.every((r) => r.attempts === 1 && r.refusal?.code === "reader_flagged")).toBe(true);
    expect(sampler.counts()).toMatchObject({ attempts: 8, redraws: 0, worlds_refused: 8 });
  });

  it("counts a draw the model refuses (its legality bound) as a failed attempt and goes on to the next index", () => {
    const p = port({ draw: (j) => (j % 4 === 0 ? { throws: beliefError("legality_redraws_exhausted") } : { world: trueWorld(good), redraws: 1 }) });
    const sampler = make(p);
    const out = sampler.sample(request(good), STREAM);
    if (out.kind !== "worlds") throw new Error("unreachable");
    expect(out.valid.map((w) => w.attempt)).toEqual(Array.from({ length: 8 }, () => 1));
    expect(sampler.counts()).toMatchObject({ redraws: 8, draw_refusals: { legality_redraws_exhausted: 8 }, legality_redraws: 8 });
    const every = make(port({ draw: () => ({ throws: beliefError("legality_redraws_exhausted") }) }));
    const none = every.sample(request(good), STREAM);
    if (none.kind !== "worlds") throw new Error("unreachable");
    expect(none.refused[0]).toMatchObject({ refusal: null, draw_code: "legality_redraws_exhausted" });
    expect(every.counts().refusals["draw:legality_redraws_exhausted"]).toBe(8);
  });

  it("propagates a draw error that is not the model's, and the model's setup defects", () => {
    expect(() => make(port({ draw: () => ({ throws: new Error("boom") }) })).sample(request(good), STREAM)).toThrow("boom");
    expect(() => make(port({ draw: () => ({ throws: beliefError("spread_sampler_required") }) })).sample(request(good), STREAM)).toThrow(/spread_sampler_required/);
  });

  it("is a pure function of the port, the request and the stream", () => {
    const run = () => {
      const p = port({ draw: (j) => ({ world: trueWorld(good), redraws: j % 3 }) });
      const sampler = make(p);
      const out = sampler.sample(request(good), STREAM);
      return JSON.stringify({ out: out.kind === "worlds" ? out.valid.map((w) => [w.world, w.attempt, w.index, w.built.opponent_order]) : out, counts: sampler.counts(), drawn: p.drawn });
    };
    expect(run()).toBe(run());
  });
});

describe("the trapping evidence of the own request (ticket #1127)", () => {
  it("hands the port a check built from the request when asked, counts it, and hands none otherwise", () => {
    const seen: (TrappingCheckPort | null | undefined)[] = [];
    const opened = (input: { trapping?: TrappingCheckPort | null }): WorldSourceResult | null => {
      seen.push(input.trapping);
      return null;
    };
    const p = port({ draw: () => ({ world: trueWorld(good), redraws: 0 }) }, {}, opened);
    const asked = createWorldSampler({ engine, vocabulary, format_id: GOLDEN_FORMAT, port: p, builder: createWorldBuilder(context), settings: { worlds: 1, attempts: 1, min_valid: 0 }, trapping: { scenario_id: context.scenario_id } });
    asked.sample(request(good), STREAM);
    expect(asked.counts()).toMatchObject({ trapping_checks: 1, trapping_seen: 0 });
    expect(seen[0]?.observed.length).toBeGreaterThan(0);
    const plain = make(p, { worlds: 1, attempts: 1, min_valid: 0 });
    plain.sample(request(good), STREAM);
    expect(seen[1]).toBeNull();
    expect(plain.counts()).toMatchObject({ trapping_checks: 0, trapping_seen: 0 });
  });

  it("counts a decision whose request showed maybeTrapped", () => {
    const flagged = { ...request(good), request: { ...good.input.request, active: good.input.request.active?.map((slot, k) => (k === 0 ? { ...slot, maybeTrapped: true } : slot)) } } as WorldRequest;
    const p = port({ draw: () => ({ world: trueWorld(good), redraws: 0 }) });
    const asked = createWorldSampler({ engine, vocabulary, format_id: GOLDEN_FORMAT, port: p, builder: createWorldBuilder(context), settings: { worlds: 1, attempts: 1, min_valid: 0 }, trapping: { scenario_id: context.scenario_id } });
    asked.sample(flagged, STREAM);
    expect(asked.counts().trapping_seen).toBe(1);
  });
});

describe("a decision the belief refuses, and the settings", () => {
  it("is counted by the model's code and is not searched; a setup defect is thrown", () => {
    const refused = port({ draw: () => ({ throws: new Error("never") }) }, {}, () => ({ kind: "refused", code: "evidence_has_no_support" }));
    const sampler = make(refused);
    expect(sampler.sample(request(good), STREAM)).toMatchObject({ kind: "refused", code: "evidence_has_no_support" });
    const c = sampler.counts();
    expect(c.belief_refusals).toEqual({ evidence_has_no_support: 1 });
    expect(Object.values(c.by_evidence).reduce((a, r) => a + r.belief_refused, 0)).toBe(1);
    expect(c.worlds_asked).toBe(0);
    expect(() => make(port({ draw: () => ({ throws: 0 }) }, {}, () => ({ kind: "refused", code: "format_mismatch" }))).sample(request(good), STREAM)).toThrow(/format_mismatch/);
  });

  it("checks the settings and the stream", () => {
    const p = port({ draw: () => ({ throws: 0 }) });
    for (const bad of [{ ...settings, worlds: 0 }, { ...settings, attempts: 0 }, { ...settings, min_valid: -1 }, { ...settings, min_valid: 9 }, { ...settings, worlds: 1.5 }]) expect(() => make(p, bad)).toThrow(RangeError);
    expect(() => make(p).sample(request(good), new Uint8Array(31))).toThrow(RangeError);
  });
});

describe("the evidence class of a decision", () => {
  const find = (want: (c: Case) => boolean): Case | undefined => cases().find(want);
  it("is the first of a trapping flag, a volatile, a timed condition, an unseen brought Pokemon, else the rest", () => {
    const all = cases();
    const seen = new Set<string>();
    for (const c of all) seen.add(evidenceClass(request(c)));
    for (const name of EVIDENCE_CLASSES) expect(seen, name).toContain(name);
    const trapped = find((c) => (c.input.request.active ?? []).some((a) => a.maybeTrapped === true || a.trapped === true));
    expect(evidenceClass(request(trapped as Case))).toBe("maybe_trapped");
    // Precedence: a trapped request with a volatile and a weather is still maybe_trapped.
    const crowded = structuredClone(request(trapped as Case));
    crowded.state.weather = "rain";
    expect(evidenceClass(crowded)).toBe("maybe_trapped");
  });
});

describe("the search stream", () => {
  it("is SHA-256 of the JCS of the six fields (a vector computed with Python hashlib)", () => {
    const hex = Buffer.from(searchStream({ seed: "sodium,00112233445566778899aabbccddeeff", game: 3, player: "p1", policy_id: "policy:abc", request_seq: 5 })).toString("hex");
    expect(hex).toBe("9d83b26dfbc40f7c8b22479490197eef7e225f02e39c2daa42fde2b5937a9cda");
    expect(searchStream({ seed: "s", game: 0, player: "p1", policy_id: "a", request_seq: 1 })).not.toEqual(searchStream({ seed: "s", game: 0, player: "p1", policy_id: "a", request_seq: 2 }));
  });
});

describe("the privacy boundary of the sampler path", () => {
  it("has no input for a seed, a live battle, a BattleRecord, an omniscient log, hidden verdicts or the opponent's team: each is a compile error", () => {
    const neverRun = (sampler: ReturnType<typeof make>, record: BattleRecord, log: OmniscientLog, verdicts: HiddenVerdicts[], channel: PlayerChannel, input: WorldInput): void => {
      // @ts-expect-error: a BattleRecord is not a request
      void sampler.sample(record, STREAM);
      // @ts-expect-error: nor the omniscient log
      void sampler.sample(log, STREAM);
      // @ts-expect-error: nor hidden verdicts
      void sampler.sample(verdicts, STREAM);
      // @ts-expect-error: nor a channel
      void sampler.sample(channel, STREAM);
      // @ts-expect-error: the stream is bytes, not a battle seed string
      void sampler.sample(input, "sodium,00000000000000000000000000000000");
      // @ts-expect-error: the request has no opponent team (the sampler fills it from the model)
      void sampler.sample({ request: input.request, team: input.team, state: input.state, flags: input.flags, facts: input.facts, opponent: input.opponent }, STREAM);
      // @ts-expect-error: nor a field for a seed
      void sampler.sample({ ...input, seed: "sodium,00" }, STREAM);
    };
    expect(neverRun).toBeTypeOf("function");
    expectTypeOf<keyof WorldRequest>().toEqualTypeOf<"request" | "team" | "state" | "flags" | "facts">();
    expectTypeOf<Parameters<WorldSamplePort["open"]>[0]>().toEqualTypeOf<{ readonly opponent: PublicSide; readonly flags: readonly PublicFlag[]; readonly trapping?: TrappingCheckPort | null }>();
    expectTypeOf<Parameters<WorldSource["draw"]>>().toEqualTypeOf<[stream: Uint8Array, index: number]>();
  });

  it("imports no runner, omniscient log, record, opponent-model or seed module", () => {
    const source = readFileSync(join(import.meta.dirname, "..", "src", "world-sampler.ts"), "utf8");
    const imports = [...source.matchAll(/^import[^;]*from "([^"]+)"/gm)].map((m) => m[1] as string);
    expect(imports.filter((i) => /runner|omniscient|record|opponent-model|session|league|golden/.test(i))).toEqual([]);
  });
});
