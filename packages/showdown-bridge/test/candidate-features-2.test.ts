import { readFileSync } from "node:fs";
import { beforeAll, describe, expect, it } from "vitest";
import { readerSha256 } from "@areneon/public-state";
import { candidateFeatures, createFeatureMemo, FeatureBuildError, type FeatureConfigs, featureSetHash, featureSetIdentity, refusing } from "../src/candidate-features.ts";
import { type CandidateFeatures2, candidateFeatures2, FEATURE_SET_2 } from "../src/candidate-features-2.ts";
import { parseJointChoice } from "../src/damage-aware-choice.ts";
import { buildAssumedSet } from "../src/damage-aware-assumed-set.ts";
import { DamageAwareError, damageView, foeTargetKind, ReaderGap } from "../src/damage-aware-view.ts";
import { exportPaste } from "../src/teams.ts";
import { mechanicsConfigHash } from "../src/mechanics-config.ts";
import { MechanicsProbeError, type MechanicsContext, probeMechanics } from "../src/mechanics-probe.ts";
import { mechanicsQueryHash, type ParsedMechanicsQuery } from "../src/mechanics-query.ts";
import { buildResult, MECHANICS_ADAPTER_FORMAT, type Outcome } from "../src/mechanics-result.ts";
import { resolveFormat } from "../src/formats.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { type PlayerChannel } from "../src/index.ts";
import { playerChannel } from "../src/player-channel.ts";
import type { PokemonSet } from "../src/engine.ts";
import { capture, type Captured, config, corpus, engine, mechanics, snapshot, stateOf, stubProbe, vocabulary } from "./damage-aware-support.ts";

type Team = PokemonSet[];
type Probe = NonNullable<FeatureConfigs["probe"]>;
type SpeedQuery = Extract<ParsedMechanicsQuery["query"], { kind: "speed_order" }>;
const manifest = resolveFormat(engine, GOLDEN_FORMAT, snapshot).manifest;
const dex = engine.sim.Dex.mod("champions") as unknown as { species: { get(id: string): { name: string } }; moves: { get(id: string): { exists: boolean; id: string; name: string; category: string; target: string; priority: number } } };

// ---------------------------------------------------------------- a probe stand-in whose speed orders the test controls

interface Stub {
  /** Action speed by combatant (higher acts first); default: own side 100, foe side 50 + position. */
  speed?: ((side: number, position: number) => number) | undefined;
  priority?: Record<string, number>;
  /** Make a speed-order query fail: random_effect (an `unsupported` result) or one of the probe's two refusals. */
  fail?: (query: SpeedQuery) => "random" | "unqueueable" | "fractional" | null;
  /** Answer a speed-order query with this instead of an order. */
  garble?: "no-groups" | "stats";
  log?: ParsedMechanicsQuery[];
  values?: Record<string, number>;
  /** Passed to the damage stand-in. */
  damage?: Parameters<typeof stubProbe>[0];
}
function orderProbe(stub: Stub = {}): Probe {
  const base = stubProbe({ ...stub.damage, ...(stub.values === undefined ? {} : { values: stub.values }), ...(stub.log === undefined ? {} : { log: stub.log }) });
  return (context: MechanicsContext, query: ParsedMechanicsQuery) => {
    if (query.query.kind !== "speed_order") return base(context, query);
    stub.log?.push(query);
    const q = query.query as SpeedQuery;
    const failure = stub.fail?.(q) ?? null;
    if (failure === "unqueueable") throw new MechanicsProbeError("the action needs more than a move and a Mega Evolution");
    if (failure === "fractional") throw new MechanicsProbeError("the action's priority is not a whole number");
    const identity = { data_class: query.data_class, query_hash: mechanicsQueryHash(query), manifest, mechanics_config_hash: mechanicsConfigHash(mechanics), gap_ids: [] };
    const speed = stub.speed ?? ((side, position) => (side === 0 ? 100 : 50 + position));
    const actions = q.actions.map((a) => ({ ...a, order: a.move === null ? 103 : 200, priority: a.move === null ? 0 : (stub.priority?.[a.move] ?? 0), speed: speed(a.side, a.position) }));
    const rank = (i: number) => actions[i] as (typeof actions)[number];
    const order = actions.map((_, i) => i).sort((x, y) => rank(x).order - rank(y).order || rank(y).priority - rank(x).priority || rank(y).speed - rank(x).speed);
    const groups: number[][] = [];
    for (const i of order) {
      const last = groups[groups.length - 1];
      const head = last === undefined ? undefined : rank(last[0] as number);
      if (head !== undefined && head.order === rank(i).order && head.priority === rank(i).priority && head.speed === rank(i).speed) last?.push(i);
      else groups.push([i]);
    }
    let outcome: Outcome = { kind: "computed", answer: { kind: "speed_order", order_basis: "turn_start", actions, groups: stub.garble === "no-groups" ? [] : groups } as never };
    if (failure === "random") outcome = { kind: "unsupported", reason: "random_effect", position: 0 };
    if (stub.garble === "stats") outcome = { kind: "computed", answer: { kind: "stats", stats: { hp: 1, atk: 1, def: 1, spa: 1, spd: 1, spe: 1 } } };
    return { result: buildResult(identity, outcome), diagnostics: [] };
  };
}

const configs = (change: Partial<FeatureConfigs> = {}): FeatureConfigs => ({ vocabulary, config, mechanics, snapshot, data_class: "synthetic", probe: orderProbe(), ...change });

let captured: Captured[] = [];
beforeAll(async () => {
  captured = [...(await capture(corpus[0] as never)), ...(await capture(corpus[1] as never)), ...(await capture(corpus[3] as never))];
}, 120_000); // seeded real battles on the pinned engine: seconds on a loaded machine

// ---------------------------------------------------------------- helpers

const teamOf = (c: Captured): Team => c.context.team as Team;
const channelOf = (c: Captured): PlayerChannel => playerChannel({ schema_version: 1, data_class: "synthetic", player: c.context.player, lines: c.context.channel });
const candidatesOf = (c: Captured) => {
  const legal = c.request.legal as NonNullable<typeof c.request.legal>;
  return [...legal.choices, ...legal.pending].map((choice) => ({ choice, pending: legal.pending.includes(choice) }));
};
type Body = { side: { pokemon: { ident: string; condition: string; item: string }[] }; active?: { moves: { move: string; id: string; target?: string }[]; canMegaEvo?: boolean }[]; forceSwitch?: boolean[]; wait?: boolean };
interface Change {
  /** Lines put before the final request. */
  before?: string[];
  request?: (body: Body) => void;
}
/** The channel of `c` with lines added before its request and its request edited. */
function channelWith(c: Captured, change: Change): PlayerChannel {
  const lines = [...c.context.channel];
  const body = JSON.parse((lines[lines.length - 1] as string).slice("|request|".length)) as Body;
  change.request?.(body);
  return playerChannel({ schema_version: 1, data_class: "synthetic", player: c.context.player, lines: [...lines.slice(0, -1), ...(change.before ?? []), `|request|${JSON.stringify(body)}`] });
}
const build = (c: Captured, change: Partial<FeatureConfigs> = {}, memo?: ReturnType<typeof createFeatureMemo>, candidates = candidatesOf(c), channel = channelOf(c), team: readonly PokemonSet[] = teamOf(c)): CandidateFeatures2 =>
  candidateFeatures2(engine, configs(change), channel, team as never, candidates, memo);
const bytes = (features: unknown): string => JSON.stringify(features);
const moveRequests = (): Captured[] => captured.filter((c) => c.request.phase === "move" && (c.request.legal?.choices.length ?? 0) >= 2);
const forcedSwitches = (): Captured[] => captured.filter((c) => c.request.phase === "forceSwitch");
const ownTag = (c: Captured): string => c.context.player;
const foeTag = (c: Captured): string => (c.context.player === "p1" ? "p2" : "p1");

/** The living opposing actives of a request: letter, name, species, choice location (the engine's numbering). */
type Foe = { letter: string; name: string; species: string; loc: number; revealed: string[] };
function foesOf(c: Captured): Foe[] {
  const state = stateOf(c);
  return Object.keys(state.opponent.slots)
    .sort()
    .flatMap((letter, position) => {
      const mon = state.opponent.roster[state.opponent.slots[letter] as number];
      return mon === undefined || mon.fainted || mon.name === null ? [] : [{ letter, name: mon.name, species: mon.species ?? "", loc: position + 1, revealed: [...mon.moves_revealed] }];
    });
}
/** A request with a living foe that can still reveal `room` more moves (an assumed set holds four), and the foe. */
function foeCase(room: number, accept: (c: Captured) => boolean = () => true, foes = 1, pick: (foe: Foe) => boolean = () => true): { c: Captured; foe: Foe } {
  for (const c of moveRequests()) {
    const list = foesOf(c);
    const foe = list.find((f) => f.revealed.length <= 4 - room && pick(f));
    if (list.length >= foes && foe !== undefined && accept(c)) return { c, foe };
  }
  throw new Error("no recorded request fits");
}
const hasMega = (c: Captured): boolean => candidatesOf(c).some((x) => x.choice.includes("mega"));
const hasSwitch = (c: Captured): boolean => candidatesOf(c).some((x) => x.choice.includes("switch"));
/** A move of the species' roster pool that satisfies `pick`, as a move id, not one of `not`. */
function poolMove(species: string, pick: (data: { id: string; category: string; target: string; priority: number }) => boolean, not: string[] = []): string {
  const entry = vocabulary.roster.find((r) => r.id === species) ?? vocabulary.roster.find((r) => species.startsWith(r.id));
  const found = (entry?.moves ?? []).find((id) => !not.includes(id) && pick(dex.moves.get(id)));
  if (found === undefined) throw new Error(`no pool move of ${species}`);
  return found;
}
const damaging = (target: string) => (m: { category: string; target: string }) => m.category !== "Status" && m.target === target;
/** The line that makes the foe at `letter` use `move`. */
const used = (c: Captured, letter: string, name: string, move: string): string => `|move|${foeTag(c)}${letter}: ${name}|${dex.moves.get(move).name}|${ownTag(c)}a: x`;
/** A request with a foe that has used each of `moves` (the same foe, in order). */
const revealing = (c: Captured, foe: { letter: string; name: string }, moves: string[]): string[] => moves.map((move) => used(c, foe.letter, foe.name, move));

// ---------------------------------------------------------------- tests

describe("probe-damage/2 holds probe-damage/1's rows byte for byte", () => {
  it("gives the same rows as candidateFeatures on every move request, stand-in and real probe", () => {
    expect(moveRequests().length).toBeGreaterThan(10);
    for (const c of moveRequests()) {
      const one = candidateFeatures(engine, configs(), channelOf(c), teamOf(c) as never, candidatesOf(c));
      expect(bytes(build(c).rows)).toBe(bytes(one.rows));
    }
    const c = moveRequests()[3] as Captured;
    const real = configs({ probe: probeMechanics });
    const two = candidateFeatures2(engine, real, channelOf(c), teamOf(c) as never, candidatesOf(c));
    expect(bytes(two.rows)).toBe(bytes(candidateFeatures(engine, real, channelOf(c), teamOf(c) as never, candidatesOf(c)).rows));
  }, 120_000); // the real probe on one request: damage queries run 33 clones, speed orders one scenario battle
  it("has the new id and a hash that differs from /1's", () => {
    const c = moveRequests()[0] as Captured;
    const features = build(c);
    expect(features.feature_set).toBe(FEATURE_SET_2);
    expect(features.feature_set_hash).toBe(featureSetHash(featureSetIdentity(engine, configs(), FEATURE_SET_2)));
    expect(features.feature_set_hash).not.toBe(candidateFeatures(engine, configs(), channelOf(c), teamOf(c) as never, candidatesOf(c)).feature_set_hash);
  });
});

describe("speeds: exactly one row per (own position, own Mega flag, foe location)", () => {
  it("covers every living own active and every switch-in a candidate names against every living foe", () => {
    let checked = 0;
    let megas = 0;
    for (const c of [...moveRequests(), ...forcedSwitches()]) {
      const features = build(c);
      const body = c.request.request as unknown as Body;
      const parts = candidatesOf(c).flatMap((candidate) => parseJointChoice(candidate.choice));
      const own = new Set<number>(parts.flatMap((p) => (p.kind === "switch" ? [p.to - 1] : [])));
      if (body.active !== undefined) body.active.forEach((_, slot) => void (!/ fnt$/.test(body.side.pokemon[slot]?.condition ?? "") && own.add(slot)));
      const megaSlots = new Set(candidatesOf(c).flatMap((candidate) => parseJointChoice(candidate.choice).flatMap((p, slot) => (p.kind === "move" && p.mega ? [slot] : []))));
      const expected = [...own].sort((a, b) => a - b).flatMap((position) => [false, ...(megaSlots.has(position) && position < (body.active?.length ?? 0) ? [true] : [])].flatMap((mega) => foesOf(c).map((foe) => `${position}/${mega}/${foe.loc}`)));
      expect(features.speeds.map((r) => `${r.own}/${r.mega}/${r.foe}`), `request ${c.request.request_seq}`).toEqual(expected.sort());
      expect(new Set(features.speeds.map((r) => `${r.own}/${r.mega}/${r.foe}`)).size).toBe(features.speeds.length);
      megas += features.speeds.filter((r) => r.mega).length;
      checked += 1;
    }
    expect(checked).toBeGreaterThan(10);
    expect(megas).toBeGreaterThan(0); // the recorded battles hold a Mega candidate
  });

  it("reads the relation from the engine's tie groups: own_first, tie, foe_first", () => {
    const c = moveRequests()[0] as Captured;
    const relation = (speed: Stub["speed"]) => new Set(build(c, { probe: orderProbe({ speed }) }).speeds.map((r) => (r.kind === "computed" ? r.relation : r.reason)));
    expect(relation((side) => (side === 0 ? 200 : 100))).toEqual(new Set(["own_first"]));
    expect(relation((side) => (side === 0 ? 100 : 200))).toEqual(new Set(["foe_first"]));
    expect(relation(() => 100)).toEqual(new Set(["tie"]));
  });

  it("has no speed row when no foe is left, and no threat row", () => {
    const c = moveRequests().find((candidate) => foesOf(candidate).length === 2) as Captured;
    const lines = foesOf(c).map((f) => `|faint|${foeTag(c)}${f.letter}: ${f.name}`);
    const features = build(c, {}, undefined, candidatesOf(c), channelWith(c, { before: lines }));
    expect([features.speeds, features.threats, features.foe_priorities]).toEqual([[], [], []]);
    expect(features.rows).toEqual([]);
    expect(features.own_priorities.length).toBeGreaterThan(0); // a priority needs no foe: the order is asked with the own Pokemon on both sides
  });

  it("asks a pair of own Pokemon in one query and each alone when the pair is not answered", () => {
    const c = moveRequests().find((candidate) => foesOf(candidate).length === 2 && (candidate.request.request as unknown as Body).side.pokemon.length >= 2) as Captured;
    const log: ParsedMechanicsQuery[] = [];
    const whole = build(c, { probe: orderProbe({ log }) });
    const pairs = log.filter((q) => q.query.kind === "speed_order" && q.query.actions.every((a) => a.move === null) && q.query.sides[0].length === 2);
    expect(pairs.length).toBeGreaterThan(0);
    const alone = build(c, { probe: orderProbe({ fail: (q) => (q.sides[0].length === 2 && q.actions.every((a) => a.move === null) ? "random" : null) }) });
    expect(bytes(alone)).toBe(bytes(whole));
  });

  it("marks every row of a query the probe will not answer, with the probe's reason", () => {
    const c = moveRequests()[0] as Captured;
    const features = build(c, { probe: orderProbe({ fail: (q) => (q.actions.every((a) => a.move === null) ? "random" : null) }) });
    expect(features.speeds.length).toBeGreaterThan(0);
    for (const row of features.speeds) expect(row).toMatchObject({ kind: "unsupported", reason: "random_effect" });
    expect(features.speeds.every((row) => !("relation" in row))).toBe(true);
    for (const row of features.own_priorities) expect(row.kind).toBe("computed");
  });

  it("fails closed on a speed order that does not place every action or is not a speed order", () => {
    const c = moveRequests()[0] as Captured;
    expect(() => build(c, { probe: orderProbe({ garble: "no-groups" }) })).toThrow(/names every action in some group/);
    expect(() => build(c, { probe: orderProbe({ garble: "stats" }) })).toThrow(DamageAwareError);
  });
});

describe("priorities", () => {
  /** The request with the first active's moves replaced by `ids` (the request is the test's input; the candidates then name them). */
  const withMoves = (ids: string[]): Change => ({
    request: (body) => {
      (body.active as NonNullable<Body["active"]>)[0] = { ...(body.active as NonNullable<Body["active"]>)[0], moves: ids.map((id) => ({ move: dex.moves.get(id).name, id, target: dex.moves.get(id).target })) } as never;
    },
  });
  /** Candidates that play each of the replaced moves from the first slot at the first foe. */
  const playing = (ids: string[]) => ids.map((_, k) => ({ choice: `move ${k + 1} 1, pass`, pending: false }));

  it("has one own row per move part of the candidates, status moves included, per Mega variant, and one foe row per revealed move", () => {
    const { c, foe } = foeCase(2, hasMega);
    const reveal = [poolMove(foe.species, (m) => m.category === "Status" && m.target === "self", foe.revealed), poolMove(foe.species, damaging("normal"), foe.revealed)];
    const features = build(c, {}, undefined, candidatesOf(c), channelWith(c, { before: revealing(c, foe, reveal) }));
    const keys = new Set(candidatesOf(c).flatMap((candidate) => parseJointChoice(candidate.choice).flatMap((p, slot) => (p.kind === "move" ? [`${slot}/${p.index}/${p.mega}`] : []))));
    expect(features.own_priorities.map((r) => `${r.slot}/${r.move}/${r.mega}`).sort()).toEqual([...keys].sort());
    expect(features.own_priorities.some((r) => r.mega)).toBe(true);
    expect(features.foe_priorities.filter((r) => r.foe === foe.loc).map((r) => r.move).sort()).toEqual([...foe.revealed, ...reveal].sort());
  });

  it("equals the probe's own key, for Protect, a terrain-dependent move and plain moves, on a real probe", () => {
    const { c } = foeCase(0);
    const ids = ["protect", "grassyglide", "tackle", "extremespeed"];
    const log: ParsedMechanicsQuery[] = [];
    const logging: Probe = (context, query) => {
      const outcome = probeMechanics(context, query);
      if (query.query.kind === "speed_order") log.push(query);
      return outcome;
    };
    const channel = channelWith(c, { ...withMoves(ids), before: ["|-fieldstart|move: Grassy Terrain"] });
    const features = build(c, { probe: logging }, undefined, playing(ids), channel);
    const slot0 = features.own_priorities.filter((r) => r.slot === 0 && !r.mega);
    expect(slot0.map((r) => (r as { priority: number }).priority)).toEqual([4, 1, 0, 2]);
    // each row equals a direct probe query of that single action
    for (const row of slot0) {
      const batch = log.find((q) => q.query.kind === "speed_order" && q.query.actions.some((a) => a.side === 0 && a.position === 0 && a.move === ids[row.move - 1]));
      const query = batch?.query as SpeedQuery;
      const alone = { ...batch, query: { ...query, actions: query.actions.filter((a) => a.side === 0 && a.position === 0) } } as ParsedMechanicsQuery;
      const direct = probeMechanics({ engine, config: mechanics, snapshot }, alone).result;
      expect(direct.kind === "computed" && direct.answer.kind === "speed_order" ? direct.answer.actions[0]?.priority : null).toBe((row as { priority: number }).priority);
    }
    // without the terrain the move is an ordinary one
    const plain = build(c, { probe: probeMechanics }, undefined, playing(ids), channelWith(c, withMoves(ids)));
    expect((plain.own_priorities.find((r) => r.slot === 0 && r.move === 2) as { priority: number }).priority).toBe(0);
  }, 120_000); // the real probe: speed orders run one scenario battle each

  it("equals the probe's key for a foe's revealed Protect and quick move, on a real probe", () => {
    const quickOf = (f: Foe) => (vocabulary.roster.find((r) => r.id === f.species)?.moves ?? []).some((id) => dex.moves.get(id).priority > 0 && dex.moves.get(id).category !== "Status" && !f.revealed.includes(id));
    const { c, foe } = foeCase(2, () => true, 1, (f) => quickOf(f) && ((vocabulary.roster.find((r) => r.id === f.species)?.moves ?? []) as readonly string[]).includes("protect") && !f.revealed.includes("protect"));
    const quick = poolMove(foe.species, (m) => m.priority > 0 && m.category !== "Status", foe.revealed);
    const features = build(c, { probe: probeMechanics }, undefined, candidatesOf(c), channelWith(c, { before: revealing(c, foe, ["protect", quick]) }));
    const ours = features.foe_priorities.filter((r) => r.foe === foe.loc && !foe.revealed.includes(r.move));
    expect(ours).toEqual([
      { foe: foe.loc, move: "protect", kind: "computed", priority: 4 },
      { foe: foe.loc, move: quick, kind: "computed", priority: dex.moves.get(quick).priority },
    ].sort((a, b) => (a.move < b.move ? -1 : 1)));
  }, 120_000); // the real probe

  it("gives a Prankster user's status move its priority from the declared ability", () => {
    // a foe set with Prankster: the assumed-set seam hands the builder a legal Prankster set for a species that may have it
    const { c, foe } = foeCase(1);
    const species = vocabulary.roster.find((r) => r.abilities.includes("prankster") && r.moves.includes("thunderwave") && r.required_item === undefined);
    expect(species).toBeDefined();
    const assume = (mon: Parameters<NonNullable<FeatureConfigs["assume"]>>[0]) => {
      const base = buildAssumedSet({ engine, vocabulary, config }, mon);
      const set: PokemonSet = { ...base.set, species: dex.species.get((species as { id: string }).id).name, item: "", ability: "Prankster", moves: ["Thunder Wave"], gender: "" };
      return { ...base, set, paste: exportPaste(engine, [set]), revealed_moves: mon.name === foe.name ? ["thunderwave"] : [] };
    };
    const features = build(c, { probe: probeMechanics, assume }, undefined, candidatesOf(c), channelWith(c, {}));
    expect(features.foe_priorities.find((r) => r.foe === foe.loc && r.move === "thunderwave")).toEqual({ foe: foe.loc, move: "thunderwave", kind: "computed", priority: 1 });
  }, 120_000); // the real probe

  it("leaves only the offending row unsupported when a batch is not answered, and gives its reason", () => {
    const { c } = foeCase(0);
    const ids = ["tackle", "focuspunch", "quickattack", "protect"];
    const channel = channelWith(c, withMoves(ids));
    const priority = { quickattack: 1, protect: 4 };
    const plain = build(c, { probe: orderProbe({ priority }) }, undefined, playing(ids), channel);
    const mine = (f: CandidateFeatures2) => f.own_priorities.filter((r) => r.slot === 0 && !r.mega);
    expect(mine(plain).map((r) => (r as { priority: number }).priority)).toEqual([0, 0, 1, 4]);
    for (const [how, reason] of [["unqueueable", "unqueueable"], ["fractional", "fractional_priority"], ["random", "random_effect"]] as const) {
      const failing = build(c, { probe: orderProbe({ priority, fail: (q) => (q.actions.some((a) => a.move === "focuspunch") ? how : null) }) }, undefined, playing(ids), channel);
      const slot = mine(failing);
      expect(slot.map((r) => r.kind)).toEqual(["computed", "unsupported", "computed", "computed"]);
      expect(slot[1]).toEqual({ slot: 0, move: 2, mega: false, kind: "unsupported", reason });
      expect(slot.filter((r) => r.kind === "computed").map((r) => (r as { priority: number }).priority)).toEqual([0, 1, 4]); // the others are what they were
      expect(failing.foe_priorities).toEqual(plain.foe_priorities);
    }
  });

  it("gives a pseudo-move the engine does not have (the recharge turn) an unsupported row and asks nobody about it", () => {
    const { c } = foeCase(0);
    const request = withMoves(["tackle", "tackle"]);
    const channel = channelWith(c, {
      request: (body) => {
        request.request?.(body);
        ((body.active as NonNullable<Body["active"]>)[0] as { moves: { move: string; id: string }[] }).moves[0] = { move: "Recharge", id: "recharge" };
      },
    });
    const log: ParsedMechanicsQuery[] = [];
    const features = build(c, { probe: orderProbe({ log }) }, undefined, playing(["recharge", "tackle"]), channel);
    expect(features.own_priorities.filter((r) => r.slot === 0)).toEqual([
      { slot: 0, move: 1, mega: false, kind: "unsupported", reason: "query_kind" },
      { slot: 0, move: 2, mega: false, kind: "computed", priority: 0 },
    ]);
    expect(log.some((q) => q.query.kind === "speed_order" && q.query.actions.some((a) => a.move === "recharge"))).toBe(false);
  });

  it("does not depend on how the queries are batched", () => {
    const { c, foe } = foeCase(2, hasMega, 2);
    const channel = channelWith(c, { before: revealing(c, foe, [poolMove(foe.species, damaging("normal"), foe.revealed), poolMove(foe.species, (m) => m.category === "Status", foe.revealed)]) });
    const priority = { protect: 4, quickattack: 1 };
    const batched = build(c, { probe: orderProbe({ priority }) }, undefined, candidatesOf(c), channel);
    // a probe that answers one-action priority queries only: every batch falls back to single asks
    const single = build(c, { probe: orderProbe({ priority, fail: (q) => (q.actions.length > 1 && q.actions.every((a) => a.move !== null) ? "random" : null) }) }, undefined, candidatesOf(c), channel);
    expect(bytes(single)).toBe(bytes(batched));
    const log: ParsedMechanicsQuery[] = [];
    build(c, { probe: orderProbe({ priority, log }) }, undefined, candidatesOf(c), channel);
    expect(log.some((q) => q.query.kind === "speed_order" && q.query.actions.length > 1 && q.query.actions.every((a) => a.move !== null))).toBe(true);
  });

  it("asks nothing of a move request with every own active fainted, and no priority on a forced switch", () => {
    const c = moveRequests()[0] as Captured;
    const channel = channelWith(c, { request: (body) => body.side.pokemon.slice(0, 2).forEach((p) => void (p.condition = "0 fnt")) });
    const features = build(c, {}, undefined, [{ choice: "pass, pass", pending: false }], channel);
    expect([features.threats, features.speeds, features.own_priorities, features.foe_priorities]).toEqual([[], [], [], []]);
    expect(forcedSwitches().length).toBeGreaterThan(0);
    for (const forced of forcedSwitches()) {
      const f = build(forced);
      expect([f.rows, f.own_priorities, f.foe_priorities]).toEqual([[], [], []]);
    }
  });
});

describe("threats", () => {
  it("has a row per revealed damaging move that reaches an opposing Pokemon, for single, spread and random targets", () => {
    const { c, foe } = foeCase(3, () => true, 2);
    const single = poolMove(foe.species, damaging("normal"), foe.revealed);
    const spread = poolMove(foe.species, damaging("allAdjacentFoes"), foe.revealed);
    const random = poolMove(foe.species, damaging("randomNormal"), foe.revealed);
    const features = build(c, {}, undefined, candidatesOf(c), channelWith(c, { before: revealing(c, foe, [single, spread, random]) }));
    const own = new Set(features.threats.map((r) => r.own));
    expect(own.size).toBeGreaterThanOrEqual(2);
    for (const move of [single, spread, random]) expect(features.threats.filter((r) => r.move === move && r.foe === foe.loc).length).toBe(own.size);
    expect(features.foe_priorities.filter((r) => [single, spread, random].includes(r.move)).length).toBe(3);
    // The readers (and the encoder) require a foe priority row for the move of every threat in a move request.
    const priced = new Set(features.foe_priorities.map((r) => `${r.foe}/${r.move}`));
    for (const row of features.threats) expect(priced.has(`${row.foe}/${row.move}`)).toBe(true);
  });

  it("has none for a foe with only status moves or no revealed move", () => {
    const { c, foe } = foeCase(1, (candidate) => foesOf(candidate).some((f) => f.revealed.length === 0));
    const quiet = foesOf(c).find((f) => f.revealed.length === 0) as Foe;
    const status = poolMove(quiet.species, (m) => m.category === "Status" && m.priority === 0, []);
    void foe;
    const features = build(c, {}, undefined, candidatesOf(c), channelWith(c, { before: revealing(c, quiet, [status]) }));
    expect(features.threats.filter((r) => r.foe === quiet.loc)).toEqual([]);
    expect(features.foe_priorities.filter((r) => r.foe === quiet.loc)).toEqual([{ foe: quiet.loc, move: status, kind: "computed", priority: 0 }]);
    expect(build(c).foe_priorities.filter((r) => r.foe === quiet.loc)).toEqual([]);
  });

  it("covers the actives, and a Mega variant where a candidate Mega Evolves", () => {
    const { c, foe } = foeCase(1, hasMega);
    const move = poolMove(foe.species, damaging("normal"), foe.revealed);
    const features = build(c, {}, undefined, candidatesOf(c), channelWith(c, { before: revealing(c, foe, [move]) }));
    const megaSlots = [...new Set(candidatesOf(c).flatMap((x) => parseJointChoice(x.choice).flatMap((p, slot) => (p.kind === "move" && p.mega ? [slot] : []))))];
    const keys = features.threats.filter((r) => r.move === move).map((r) => `${r.own}/${r.mega}`);
    expect(megaSlots.length).toBeGreaterThan(0);
    for (const slot of megaSlots) expect(keys).toEqual(expect.arrayContaining([`${slot}/false`, `${slot}/true`]));
    expect(features.threats.filter((r) => r.mega).every((r) => megaSlots.includes(r.own))).toBe(true);
  });

  it("covers the benched Pokemon a switch candidate names, as advisory rows", () => {
    const { c, foe } = foeCase(1, hasSwitch);
    const move = poolMove(foe.species, damaging("normal"), foe.revealed);
    const features = build(c, {}, undefined, candidatesOf(c), channelWith(c, { before: revealing(c, foe, [move]) }));
    const switchIns = [...new Set(candidatesOf(c).flatMap((x) => parseJointChoice(x.choice).flatMap((p) => (p.kind === "switch" ? [p.to - 1] : []))))];
    const keys = features.threats.filter((r) => r.move === move).map((r) => `${r.own}/${r.mega}`);
    for (const position of switchIns) expect(keys).toContain(`${position}/false`);
    expect(features.threats.every((r) => r.own < 2 || switchIns.includes(r.own))).toBe(true);
  });

  it.each(["normal", "allAdjacentFoes", "randomNormal"])("equals a direct probeMechanics damage query on the same sets, with the own Pokemon at its exact hit points: %s target", (target) => {
    const { c, foe } = foeCase(1, hasMega, 1, (f) => vocabulary.roster.some((r) => r.id === f.species && r.moves.some((id) => damaging(target)(dex.moves.get(id)))));
    const move = poolMove(foe.species, damaging(target), foe.revealed);
    const channel = channelWith(c, { before: revealing(c, foe, [move]), request: (body) => void ((body.side.pokemon[0] as { condition: string }).condition = "37/100 par") });
    const rowQueries: ParsedMechanicsQuery[] = [];
    candidateFeatures(engine, configs({ probe: stubProbe({ log: rowQueries }) }), channel, teamOf(c) as never, candidatesOf(c));
    const seen = new Set(rowQueries.map(mechanicsQueryHash));
    const log: ParsedMechanicsQuery[] = [];
    const stand = build(c, { probe: orderProbe({ log }) }, undefined, candidatesOf(c), channel);
    const threatQueries = log.filter((q) => q.query.kind === "damage" && !seen.has(mechanicsQueryHash(q)));
    expect(threatQueries.length).toBe(stand.threats.length); // one damage query per row
    expect(stand.threats.length).toBeGreaterThan(0);
    const real = candidateFeatures2(engine, configs({ probe: probeMechanics }), channel, teamOf(c) as never, candidatesOf(c));
    expect(real.threats.length).toBe(threatQueries.length);
    // the first active carries the edited condition: its queries declare 37 hit points and paralysis; a Mega variant declares the defender Mega Evolved
    const defenders = threatQueries.map((q) => (q.query as { defender: { hp: number | null; status: string | null; mega: boolean } }).defender);
    const edited = defenders.filter((d) => d.status === "par");
    expect(edited.length).toBeGreaterThan(0);
    for (const d of edited) expect(d.hp).toBe(37);
    expect(defenders.filter((d) => d.mega).length).toBe(stand.threats.filter((r) => r.mega).length);
    expect(stand.threats.some((r) => r.mega)).toBe(true);
    // the rows are the real probe's own numbers: as multisets, the order of the queries being the order the builder asked in
    const direct = threatQueries.map((query) => probeMechanics({ engine, config: mechanics, snapshot }, query).result);
    const values = (row: (typeof real.threats)[number]) => (row.kind === "computed" ? [row.mean_ppm, row.max_ppm, row.min_ppm, row.ohko_rolls ?? -1].join("/") : row.reason);
    const expected = direct.map((result) => {
      if (result.kind !== "computed" || result.answer.kind !== "damage") return result.kind === "unsupported" ? result.reason : "?";
      const { no_crit: rolls, hp_current: hp, ko } = result.answer;
      return [Math.floor((rolls.reduce((sum, roll) => sum + Math.min(roll, hp), 0) * 1_000_000) / (16 * hp)), Math.floor((Math.min(Math.max(...rolls), hp) * 1_000_000) / hp), Math.floor((Math.min(Math.min(...rolls), hp) * 1_000_000) / hp), ko.kind === "counts" ? ko.ohko_no_crit : -1].join("/");
    });
    expect(real.threats.map(values).sort()).toEqual(expected.sort());
    expect(real.threats.some((row) => row.own === 0 && row.kind === "computed")).toBe(true);
  }, 240_000); // the real probe: 33 clones per damage query

  it("gives a forced switch threats for exactly the Pokemon its switch candidates name", () => {
    const c = forcedSwitches().find((candidate) => foesOf(candidate).some((f) => f.revealed.length <= 3)) as Captured;
    const foe = foesOf(c).find((f) => f.revealed.length <= 3) as Foe;
    const move = poolMove(foe.species, damaging("normal"), foe.revealed);
    const features = build(c, {}, undefined, candidatesOf(c), channelWith(c, { before: revealing(c, foe, [move]) }));
    const named = [...new Set(candidatesOf(c).flatMap((x) => parseJointChoice(x.choice).flatMap((p) => (p.kind === "switch" ? [p.to - 1] : []))))].sort((x, y) => x - y);
    expect(named.length).toBeGreaterThan(0);
    expect([...new Set(features.threats.filter((r) => r.move === move).map((r) => r.own))]).toEqual(named);
    expect([...new Set(features.speeds.map((r) => r.own))]).toEqual(named);
  });

  it("turns an unsupported damage answer into an unsupported row with the probe's reason", () => {
    const { c, foe } = foeCase(1);
    const move = poolMove(foe.species, damaging("normal"), foe.revealed);
    const probe = orderProbe({ damage: { override: { damage: { kind: "unsupported", reason: "random_effect", position: null } } } });
    const features = build(c, { probe }, undefined, candidatesOf(c), channelWith(c, { before: revealing(c, foe, [move]) }));
    expect(features.threats.length).toBeGreaterThan(0);
    for (const row of features.threats) expect(row).toMatchObject({ kind: "unsupported", reason: "random_effect" });
  });

  it("reads a benched Pokemon from the request and team alone, and skips a fainted one", () => {
    const c = moveRequests().find(hasSwitch) as Captured;
    const parts = candidatesOf(c).flatMap((x) => parseJointChoice(x.choice));
    const bench = (parts.find((p) => p.kind === "switch") as { to: number }).to - 1;
    expect(build(c).speeds.some((r) => r.own === bench)).toBe(true);
    // a fainted bench Pokemon has no row even if a candidate names it; a Pokemon the request does not list is a defect
    const down = channelWith(c, { request: (b) => void ((b.side.pokemon[bench] as { condition: string }).condition = "0 fnt") });
    expect(build(c, {}, undefined, candidatesOf(c), down).speeds.some((r) => r.own === bench)).toBe(false);
    expect(() => build(c, {}, undefined, [{ choice: "switch 9, pass", pending: false }])).toThrow(/does not list/);
  });

  it("declares a bench Pokemon at stages 0 and, unseen by the reader, not Mega", () => {
    const c = moveRequests().find(hasSwitch) as Captured;
    const state = stateOf(c);
    const view = damageView({ engine, vocabulary, config, v1: config, mechanics, snapshot, data_class: "synthetic", probe: stubProbe(), counts: { queries: 0, probe_runs: 0, unsupported_queries: 0 }, memo: new Map() });
    const parts = candidatesOf(c).flatMap((x) => parseJointChoice(x.choice));
    const bench = (parts.find((p) => p.kind === "switch") as { to: number }).to - 1;
    const body = c.request.request;
    const named = view.ownBench(body, state, teamOf(c), bench);
    expect(named.fighter.boosts).toEqual({ atk: 0, def: 0, spa: 0, spd: 0, spe: 0 });
    const unseen = { ...state, own: { ...state.own, roster: [] } };
    expect(view.ownBench(body, unseen, teamOf(c), bench).fighter.mega).toBe(false);
    expect(() => view.ownSlots(body, unseen, teamOf(c))).toThrow(ReaderGap); // an active the reader does not show is still a gap
  });
});

describe("foeTargetKind", () => {
  it("classes a move's target type by what it reaches among the foes, and fails closed on an unknown one", () => {
    expect([undefined, "normal", "randomNormal", "allAdjacentFoes", "allAdjacent", "self", "adjacentAlly", "allySide"].map((type) => foeTargetKind(type))).toEqual(["none", "single", "random", "every", "every", "none", "none", "none"]);
    expect(() => foeTargetKind("mystery")).toThrow(/does not know/);
  });
});

describe("requests with nothing to feature", () => {
  it("has empty arrays for Team Preview and for a wait", () => {
    const preview = captured.filter((c) => c.request.phase === "preview");
    expect(preview.length).toBeGreaterThan(0);
    const empty = { rows: [], threats: [], speeds: [], own_priorities: [], foe_priorities: [] };
    for (const c of preview) expect(build(c)).toMatchObject(empty);
    const c = moveRequests()[0] as Captured;
    const wait = channelWith(c, { request: (body) => void (delete body.active, (body.wait = true)) });
    expect(build(c, {}, undefined, [], wait)).toMatchObject(empty);
  });
});

describe("the declared state: Trick Room, Tailwind, paralysis, a Mega's Speed and an exact tie equal a move-against-move query", () => {
  /** Every computed speed row of the builder equals the real probe's order of the same two Pokemon with a priority-0 move each. */
  function crossCheck(c: Captured, channel: PlayerChannel, team: readonly PokemonSet[] = teamOf(c)): { rows: CandidateFeatures2["speeds"] } {
    const log: ParsedMechanicsQuery[] = [];
    const logging: Probe = (context, query) => {
      if (query.query.kind === "speed_order" && query.query.actions.every((a) => a.move === null)) log.push(query);
      return probeMechanics(context, query);
    };
    const features = candidateFeatures2(engine, configs({ probe: logging }), channel, team as never, candidatesOf(c));
    expect(features.speeds.length).toBeGreaterThan(0);
    // The queries come in the builder's order: the non-Mega own Pokemon two at a time, in position order, then each Mega variant alone.
    const positions = [...new Set(features.speeds.filter((r) => !r.mega).map((r) => r.own))].sort((x, y) => x - y);
    const megaPositions = [...new Set(features.speeds.filter((r) => r.mega).map((r) => r.own))].sort((x, y) => x - y);
    const chunks = Math.ceil(positions.length / 2);
    expect(log.length).toBe(chunks + megaPositions.length);
    const foeLocs = foesOf(c).map((f) => f.loc);
    let compared = 0;
    log.forEach((q, k) => {
      const sq = q.query as SpeedQuery;
      const owns = k < chunks ? positions.slice(2 * k, 2 * k + 2) : [megaPositions[k - chunks] as number];
      expect(sq.sides[0].length).toBe(owns.length);
      owns.forEach((position, i) => {
        sq.sides[1].forEach((_, j) => {
          const direct = probeMechanics({ engine, config: mechanics, snapshot }, { ...q, query: { ...sq, actions: [{ side: 0, position: i, move: "tackle", mega: false }, { side: 1, position: j, move: "tackle", mega: false }] } } as ParsedMechanicsQuery).result;
          if (direct.kind !== "computed" || direct.answer.kind !== "speed_order") throw new Error("the probe gave no order");
          const answer = direct.answer;
          const group = (action: number) => answer.groups.findIndex((g) => g.includes(action));
          const sign = Math.sign(group(1) - group(0));
          const row = features.speeds.find((r) => r.own === position && r.mega === (k >= chunks) && r.foe === foeLocs[j]);
          expect(row, `own ${position} against foe ${foeLocs[j]}`).toMatchObject({ kind: "computed", relation: sign > 0 ? "own_first" : sign === 0 ? "tie" : "foe_first" });
          compared += 1;
        });
      });
    });
    expect(compared).toBe(features.speeds.length);
    return { rows: features.speeds };
  }
  const relations = (rows: CandidateFeatures2["speeds"]) => rows.map((r) => (r.kind === "computed" ? r.relation : r.reason));

  it("holds for the declared field and statuses of recorded requests, Mega variants included", () => {
    const c = moveRequests().find((candidate) => foesOf(candidate).length >= 1 && candidatesOf(candidate).some((x) => x.choice.includes("mega"))) as Captured;
    const plain = crossCheck(c, channelOf(c));
    expect(plain.rows.some((r) => r.mega)).toBe(true);
    const foe = foesOf(c)[0] as ReturnType<typeof foesOf>[number];
    const room = crossCheck(c, channelWith(c, { before: ["|-fieldstart|move: Trick Room"] }));
    const wind = crossCheck(c, channelWith(c, { before: [`|-sidestart|${ownTag(c)}: ${ownTag(c)}|move: Tailwind`] }));
    const par = crossCheck(c, channelWith(c, { before: [`|-status|${foeTag(c)}${foe.letter}: ${foe.name}|par`] }));
    // Trick Room reverses who is faster
    const flip = (r: string) => (r === "own_first" ? "foe_first" : r === "foe_first" ? "own_first" : r);
    expect(relations(room.rows)).toEqual(relations(plain.rows).map(flip));
    // Tailwind on the own side and paralysis on a foe can only make the own Pokemon faster against it
    const firsts = (rows: CandidateFeatures2["speeds"]) => relations(rows).filter((r) => r === "own_first").length;
    expect(firsts(wind.rows)).toBeGreaterThanOrEqual(firsts(plain.rows));
    expect(firsts(par.rows)).toBeGreaterThanOrEqual(firsts(plain.rows));
    expect(relations(wind.rows).length).toBe(plain.rows.length);
    expect(relations(par.rows).length).toBe(plain.rows.length);
  }, 180_000); // the real probe: a scenario battle per query

  it("reports a tie when the own Pokemon is the foe's assumed set", () => {
    const plainMon = (mon: { status: string | null; boosts: Record<string, number> } | undefined): boolean => mon !== undefined && mon.status === null && Object.values(mon.boosts).every((v) => v === 0);
    const { c, foe } = foeCase(0, (candidate) => {
      const state = stateOf(candidate);
      const own = state.own.roster[state.own.slots.a as number];
      const foes = foesOf(candidate);
      return foes.length === 2 && plainMon(own) && !(own?.mega ?? true) && foes.every((f) => vocabulary.roster.find((r) => r.id === f.species)?.required_item === undefined && plainMon(state.opponent.roster[state.opponent.slots[f.letter] as number]));
    });
    const log: ParsedMechanicsQuery[] = [];
    candidateFeatures2(engine, configs({ probe: orderProbe({ log }) }), channelOf(c), teamOf(c) as never, candidatesOf(c));
    const query = log.find((q) => q.query.kind === "speed_order" && q.query.actions.every((a) => a.move === null)) as ParsedMechanicsQuery;
    const imported = (engine.sim.Teams.import((query.query as SpeedQuery).sides[1][foe.loc - 1]?.set as string) ?? [])[0] as PokemonSet;
    const first = (c.request.request as unknown as Body).side.pokemon[0] as { ident: string };
    const name = first.ident.slice(first.ident.indexOf(": ") + 2);
    const at = teamOf(c).findIndex((set) => set.name === name || (set.name === "" && set.species === name));
    // the validator refuses a nickname that is another species' name, so the mirror is nicknamed
    const team = teamOf(c).map((set, k) => (k === at ? { ...imported, name: "Mirror" } : set));
    const channel = channelWith(c, {
      request: (body) => {
        (body.side.pokemon[0] as { item: string; ident: string }).item = imported.item;
        (body.side.pokemon[0] as { ident: string }).ident = `${ownTag(c)}: Mirror`;
      },
    });
    const features = candidateFeatures2(engine, configs({ probe: probeMechanics }), channel, team as never, candidatesOf(c));
    const rows = features.speeds.filter((r) => r.own === 0 && !r.mega && r.foe === foe.loc);
    expect(rows).toEqual([{ own: 0, mega: false, foe: foe.loc, kind: "computed", relation: "tie" }]);
  }, 120_000); // the real probe
});

describe("determinism", () => {
  it("gives the same bytes twice, with and without the memo, and in shuffled candidate order", () => {
    for (const c of moveRequests().slice(0, 12)) {
      const plain = bytes(build(c));
      expect(bytes(build(c))).toBe(plain);
      const memo = createFeatureMemo();
      expect(bytes(build(c, {}, memo))).toBe(plain);
      expect(bytes(build(c, {}, memo))).toBe(plain); // a warm memo
      const candidates = candidatesOf(c);
      expect(bytes(build(c, {}, undefined, [...candidates].reverse()))).toBe(plain);
      expect(bytes(build(c, {}, undefined, [...candidates.slice(3), ...candidates.slice(0, 3)]))).toBe(plain);
    }
    for (const c of forcedSwitches()) expect(bytes(build(c, {}, undefined, [...candidatesOf(c)].reverse()))).toBe(bytes(build(c)));
  });

  it("answers a repeated query from the memo, speed orders included", () => {
    const c = moveRequests()[0] as Captured;
    const memo = createFeatureMemo();
    build(c, {}, memo);
    const first = { ...memo.counts };
    build(c, {}, memo);
    expect(memo.counts.queries).toBe(first.queries * 2);
    expect(memo.counts.probe_runs).toBe(first.probe_runs);
  });

  it("remembers a refused state in the memo and refuses again without asking the probe", () => {
    const c = moveRequests()[0] as Captured;
    const memo = createFeatureMemo();
    const probe = orderProbe({ fail: () => null });
    const refusing: Probe = (context, query) => {
      if (query.query.kind === "speed_order") throw new MechanicsProbeError("the declared HP exceeds the maximum");
      return probe(context, query);
    };
    const first = build(c, { probe: refusing }, memo);
    const runs = memo.counts.probe_runs;
    expect(bytes(build(c, { probe: refusing }, memo))).toBe(bytes(first));
    expect(memo.counts.probe_runs).toBe(runs); // the refusal was remembered
    expect(first.speeds.length).toBeGreaterThan(0);
    expect(first.speeds.every((row) => row.kind === "unsupported" && row.reason === "query_kind")).toBe(true); // #1246: rows, not a forfeit
  });

  it("remembers a speed-order limit in the memo", () => {
    const c = moveRequests()[0] as Captured;
    const memo = createFeatureMemo();
    const probe = orderProbe({ fail: (q) => (q.actions.some((a) => a.move === "focuspunch") ? "unqueueable" : null) });
    const ids = ["focuspunch", "tackle"];
    const change: Change = { request: (body) => void ((body.active as NonNullable<Body["active"]>)[0] = { moves: ids.map((id) => ({ move: id, id, target: "normal" })) } as never) };
    const channel = channelWith(c, change);
    const choices = [{ choice: "move 1 1, pass", pending: false }, { choice: "move 2 1, pass", pending: false }];
    const first = build(c, { probe }, memo, choices, channel);
    const runs = memo.counts.probe_runs;
    expect(bytes(build(c, { probe }, memo, choices, channel))).toBe(bytes(first));
    expect(memo.counts.probe_runs).toBe(runs);
    expect(first.own_priorities.find((r) => r.move === 1)).toMatchObject({ kind: "unsupported", reason: "unqueueable" });
  });
});

describe("feature_set_hash", () => {
  const identity = () => featureSetIdentity(engine, configs(), FEATURE_SET_2);
  it("is /1's preimage with the new id and changes with each of its six inputs", () => {
    const id = identity();
    expect(Object.keys(id).sort()).toEqual(["adapter_format", "assumed_set_config_hash", "feature_set", "mechanics_config_hash", "reader_sha256", "version_manifest"]);
    expect(id).toMatchObject({ feature_set: FEATURE_SET_2, adapter_format: MECHANICS_ADAPTER_FORMAT, reader_sha256: readerSha256() });
    const base = featureSetHash(id);
    expect(base).not.toBe(featureSetHash(featureSetIdentity(engine, configs())));
    const variants = [
      { ...id, feature_set: "probe-damage/1" as const },
      featureSetIdentity(engine, configs({ mechanics: { ...mechanics, ko_blockers: { ...mechanics.ko_blockers, items: [] } } }), FEATURE_SET_2),
      featureSetIdentity(engine, configs({ config: { ...config, epsilon_ppm: config.epsilon_ppm + 1 } }), FEATURE_SET_2),
      { ...id, adapter_format: MECHANICS_ADAPTER_FORMAT + 1 },
      { ...id, reader_sha256: "0".repeat(64) },
      { ...id, version_manifest: { ...id.version_manifest, showdown_commit: "1".repeat(40) } },
    ];
    const hashes = variants.map((v, k) => {
      expect(featureSetHash(v), `input ${k}`).not.toBe(base);
      return featureSetHash(v);
    });
    expect(new Set(hashes).size).toBe(hashes.length);
  });

  it("is refused for a mechanics config that does not list speed_order", () => {
    const narrow = { ...mechanics, scope: { ...mechanics.scope, query_kinds: mechanics.scope.query_kinds.filter((entry) => entry.kind !== "speed_order") } };
    expect(() => featureSetIdentity(engine, configs({ mechanics: narrow }), FEATURE_SET_2)).toThrow(/speed_order/);
    expect(() => featureSetIdentity(engine, configs({ mechanics: narrow }))).not.toThrow(); // /1 does not ask speed orders
  });
});

describe("refusals: the five /1 codes", () => {
  const first = (): Captured => moveRequests()[0] as Captured;
  const code = (run: () => unknown): string | undefined => {
    try {
      run();
    } catch (error) {
      return error instanceof FeatureBuildError ? error.code : `other:${String(error)}`;
    }
    return undefined;
  };
  it("refuses with no_request, reader_flagged, reader_gap, assumed_set and probe_refused", () => {
    const c = first();
    const cut = playerChannel({ schema_version: 1, data_class: "synthetic", player: c.context.player, lines: c.context.channel.slice(0, -1) });
    expect(code(() => build(c, {}, undefined, candidatesOf(c), cut))).toBe("no_request");
    expect(code(() => build(c, {}, undefined, candidatesOf(c), channelWith(c, { before: ["|-notarealline|p1a: X|y"] })))).toBe("reader_flagged");
    expect(code(() => refusing(() => void (() => { throw new ReaderGap("the reader does not show an own Pokemon the request lists"); })()))).toBe("reader_gap");
    expect(code(() => build(c, { vocabulary: { ...vocabulary, roster: [] } as never }))).toBe("assumed_set");
    expect(code(() => build(c, { probe: stubProbe({ throwOn: { stats: new MechanicsProbeError("the engine does not take that boost") } }) }))).toBe("probe_refused");
    // A refused damage or speed-order query is a row, not a refusal of the request (#1246).
    expect(code(() => build(c, { probe: stubProbe({ throwOn: { damage: new MechanicsProbeError("the engine does not take that boost") } }) }))).toBeUndefined();
    expect(code(() => build(c, { probe: stubProbe({ throwOn: { speed_order: new MechanicsProbeError("the declared HP exceeds the maximum") } }) }))).toBeUndefined();
  });
  it("lets a defect propagate: a choice of the wrong shape", () => {
    const c = first();
    expect(code(() => build(c, {}, undefined, [{ choice: "pass", pending: false }]))).toMatch(/^other:.*different number of parts/);
    const forced = forcedSwitches()[0] as Captured;
    expect(code(() => build(forced, {}, undefined, [{ choice: "pass", pending: false }, { choice: "pass", pending: false }, { choice: "pass", pending: false }]))).toMatch(/^other:.*different number of parts/);
  });
});

describe("the type rule and the imports", () => {
  it("takes no opponent team, omniscient log, hidden verdicts, battle record or seed", () => {
    const neverRun = (log: import("../src/index.ts").OmniscientLog, seed: string): void => {
      const team = [] as never;
      // @ts-expect-error: the omniscient log is not a PlayerChannel
      void candidateFeatures2(engine, configs(), log, team, []);
      // @ts-expect-error: nor a seed
      void candidateFeatures2(engine, configs(), seed, team, []);
      // @ts-expect-error: the configs hold no opponent team
      void candidateFeatures2(engine, { vocabulary, config, mechanics, data_class: "synthetic", opponent_team: [] }, undefined as never, team, []);
    };
    expect(neverRun).toBeTypeOf("function");
  });
  it("imports no runner, omniscient or seed module at run time", () => {
    const source = readFileSync(new URL("../src/candidate-features-2.ts", import.meta.url), "utf8");
    const imports = [...source.matchAll(/^import .* from "(.*)";$/gm)].map((m) => m[1] as string);
    expect(imports.filter((path) => /runner|omniscient|hidden|league-seed|progress-seed/.test(path))).toEqual([]);
  });
});
