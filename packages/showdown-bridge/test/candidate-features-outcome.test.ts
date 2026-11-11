import { beforeAll, describe, expect, it } from "vitest";
import { readerSha256 } from "@areneon/public-state";
import { HISTORY_FACTS_FORMAT, historyFactsSha256 } from "@areneon/public-state";
import { type AssumedSet, buildAssumedSet } from "../src/damage-aware-assumed-set.ts";
import { candidateFeatures2 } from "../src/candidate-features-2.ts";
import { goldenHistoryFactsData } from "../src/history-facts-data.ts";
import { assumedMoveOf, branchesOf, candidateFeaturesOutcome, factsOf, FEATURE_SET_OUTCOME, historyDataOf, otherFoeMove, type OutcomeRow } from "../src/candidate-features-outcome.ts";
import { createFeatureMemo, type FeatureConfigs, featureSetHash, featureSetIdentity } from "../src/candidate-features.ts";
import { buildFeatures } from "../src/feature-build.ts";
import { parseJointChoice } from "../src/damage-aware-choice.ts";
import { mechanicsConfigHash, withMoveOutcome } from "../src/mechanics-config.ts";
import { MechanicsProbeError, type MechanicsContext } from "../src/mechanics-probe.ts";
import { mechanicsQueryHash, type ParsedMechanicsQuery } from "../src/mechanics-query.ts";
import { OUTCOME_FORMAT } from "../src/mechanics-outcome.ts";
import { buildResult, MECHANICS_ADAPTER_FORMAT, type Outcome } from "../src/mechanics-result.ts";
import { loadOutcomeConfig, outcomeConfigHash } from "../src/outcome-config.ts";
import { resolveFormat } from "../src/formats.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { type PlayerChannel } from "../src/index.ts";
import { playerChannel } from "../src/player-channel.ts";
import type { PokemonSet } from "../src/engine.ts";
import { capture, type Captured, config, corpus, engine, mechanics, snapshot, stateOf, stubProbe, vocabulary } from "./damage-aware-support.ts";

type Query = ParsedMechanicsQuery["query"];
type OutcomeQuery = Extract<Query, { kind: "move_outcome" }>;
const manifest = resolveFormat(engine, GOLDEN_FORMAT, snapshot).manifest;
const record = loadOutcomeConfig(GOLDEN_FORMAT);
const withOutcome = withMoveOutcome(mechanics);
const dex = engine.sim.Dex.mod("champions") as unknown as { moves: { get(id: string): { exists: boolean; id: string; name: string; category: string; target: string } } };

/** A probe stand-in: the damage and speed answers of `stubProbe`, and an outcome per asked move that the test controls. */
function probe(log: ParsedMechanicsQuery[] = [], outcome: (asked: string, branch: (string | null)[], query: OutcomeQuery) => Outcome | "refuse" = () => ({ kind: "computed", answer: { kind: "move_outcome", precondition: "passes", precondition_ppm: 1_000_000, hit_ppm: 950_000 } })) {
  const base = stubProbe({ log: [] });
  return (context: MechanicsContext, query: ParsedMechanicsQuery) => {
    if (query.query.kind === "speed_order") {
      // every action in a group of its own, in the order given (the own side first)
      log.push(query);
      const actions = query.query.actions.map((a) => ({ ...a, order: a.move === null ? 103 : 200, priority: 0, speed: 100 - 10 * a.side }));
      const identity = { data_class: query.data_class, query_hash: mechanicsQueryHash(query), manifest, mechanics_config_hash: mechanicsConfigHash(withOutcome), gap_ids: [] };
      return { result: buildResult(identity, { kind: "computed", answer: { kind: "speed_order", order_basis: "turn_start", actions, groups: actions.map((_, i) => [i]) } as never }), diagnostics: [] };
    }
    if (query.query.kind !== "move_outcome") {
      log.push(query);
      return base(context, query);
    }
    log.push(query);
    const q = query.query as OutcomeQuery;
    const answer = outcome(q.actions[q.asked]?.move as string, q.actions.filter((_, i) => i !== q.asked && q.actions[i]?.side === 1).map((a) => a.move), q);
    if (answer === "refuse") throw new MechanicsProbeError("the engine refused the declared state");
    const identity = { data_class: query.data_class, query_hash: mechanicsQueryHash(query), manifest, mechanics_config_hash: mechanicsConfigHash(withOutcome), gap_ids: [] };
    return { result: buildResult(identity, answer), diagnostics: [] };
  };
}

const configs = (change: Partial<FeatureConfigs> = {}): FeatureConfigs => ({ vocabulary, config, mechanics: withOutcome, outcome: { config: record }, snapshot, data_class: "synthetic", probe: probe(), ...change });

let captured: Captured[] = [];
beforeAll(async () => {
  captured = [...(await capture(corpus[0] as never)), ...(await capture(corpus[1] as never)), ...(await capture(corpus[3] as never))];
}, 120_000);

const teamOf = (c: Captured): PokemonSet[] => c.context.team as PokemonSet[];
const channelOf = (c: Captured): PlayerChannel => playerChannel({ schema_version: 1, data_class: "synthetic", player: c.context.player, lines: c.context.channel });
const candidatesOf = (c: Captured) => {
  const legal = c.request.legal as NonNullable<typeof c.request.legal>;
  return [...legal.choices, ...legal.pending].map((choice) => ({ choice, pending: legal.pending.includes(choice) }));
};
const build = (c: Captured, change: Partial<FeatureConfigs> = {}, memo?: ReturnType<typeof createFeatureMemo>, candidates = candidatesOf(c), channel = channelOf(c)) => candidateFeaturesOutcome(engine, configs(change), channel, teamOf(c), candidates, memo);
const moveRequests = (): Captured[] => captured.filter((c) => c.request.phase === "move" && (c.request.legal?.choices.length ?? 0) >= 2);
const twoFoes = (): Captured => moveRequests().find((c) => Object.values(stateOf(c).opponent.slots).filter((i) => i !== null).length === 2 && Object.values(stateOf(c).opponent.roster).filter((m) => !m.fainted).length >= 2) as Captured;

/** An `assume` seam that gives every foe the revealed moves (and filler) the test says, over ADR-018's assumed set. */
const assuming = (revealed: Record<string, string[]>, filler: string[] = []) => (mon: Parameters<NonNullable<FeatureConfigs["assume"]>>[0]): AssumedSet => ({
  ...buildAssumedSet({ engine, vocabulary, config }, mon),
  revealed_moves: revealed[mon.name ?? ""] ?? revealed["*"] ?? [],
  filler_moves: filler,
});

describe("identity and the earlier sets' bytes", () => {
  it("names the history facts and the outcome record in the preimage, and nothing else changes", () => {
    const identity = featureSetIdentity(engine, configs(), FEATURE_SET_OUTCOME);
    expect(identity).toEqual({
      feature_set: "probe-outcome/1",
      mechanics_config_hash: mechanicsConfigHash(withOutcome),
      assumed_set_config_hash: expect.any(String),
      adapter_format: MECHANICS_ADAPTER_FORMAT,
      reader_sha256: readerSha256(),
      version_manifest: manifest,
      outcome_format: OUTCOME_FORMAT,
      history_facts_format: HISTORY_FACTS_FORMAT,
      history_facts_sha256: historyFactsSha256(),
      outcome_config_hash: outcomeConfigHash(record),
    });
    const hash = featureSetHash(identity);
    for (const change of [{ outcome_format: OUTCOME_FORMAT + 1 }, { history_facts_format: "history-facts/2" }, { history_facts_sha256: "0".repeat(64) }, { outcome_config_hash: "1".repeat(64) }, { mechanics_config_hash: "2".repeat(64) }]) {
      expect(featureSetHash({ ...identity, ...change })).not.toBe(hash);
    }
    expect(featureSetHash(featureSetIdentity(engine, configs({ outcome: { config: { ...record, max_branches: 4 } } }), FEATURE_SET_OUTCOME))).not.toBe(hash);
  });
  it("leaves the preimages of /1 and /2 as they were (no key of this set)", () => {
    for (const set of ["probe-damage/1", "probe-damage/2"] as const) {
      expect(Object.keys(featureSetIdentity(engine, configs({ mechanics }), set)).sort()).toEqual(["adapter_format", "assumed_set_config_hash", "feature_set", "mechanics_config_hash", "reader_sha256", "version_manifest"]);
    }
  });
  it("refuses a missing record and a mechanics config without move_outcome", () => {
    const { outcome: _drop, ...without } = configs();
    expect(() => featureSetIdentity(engine, without, FEATURE_SET_OUTCOME)).toThrow(/reviewed record/);
    expect(() => featureSetIdentity(engine, configs({ mechanics }), FEATURE_SET_OUTCOME)).toThrow(/move_outcome/);
    const c = moveRequests()[0] as Captured;
    expect(() => candidateFeaturesOutcome(engine, without, channelOf(c), teamOf(c), candidatesOf(c))).toThrow(/reviewed record/);
  });
  it("is what buildFeatures builds for the id", () => {
    const c = moveRequests()[0] as Captured;
    const direct = build(c);
    const through = buildFeatures(engine, configs(), channelOf(c), teamOf(c), candidatesOf(c), createFeatureMemo(), FEATURE_SET_OUTCOME);
    expect(JSON.stringify(through)).toBe(JSON.stringify(direct));
  });
});

describe("branches (engine-free)", () => {
  it("lists switch, then the revealed moves in id order, cut to the limit; none revealed is one assumed branch", () => {
    expect(branchesOf(["protect", "fakeout", "tackle"], "x", 5)).toEqual([
      { action: "switch", foe_move: null, assumed: false },
      { action: "move", foe_move: "fakeout", assumed: false },
      { action: "move", foe_move: "protect", assumed: false },
      { action: "move", foe_move: "tackle", assumed: false },
    ]);
    expect(branchesOf(["d", "c", "b", "a", "a"], "x", 3).map((b) => b.foe_move)).toEqual([null, "a", "b"]);
    expect(branchesOf([], "tackle", 5)).toEqual([
      { action: "switch", foe_move: null, assumed: false },
      { action: "move", foe_move: "tackle", assumed: true },
    ]);
  });
  it("takes the assumed move from the filler, else the first move, and the other foe's move from its first revealed damaging move", () => {
    const foe = (revealed: string[], filler: string[], moves: string[]) => ({ assumed: { revealed_moves: revealed, filler_moves: filler, set: { moves } } }) as never;
    expect(assumedMoveOf(foe([], ["Tackle"], ["Protect"]), dex as never)).toBe("tackle");
    expect(assumedMoveOf(foe([], [], ["Protect"]), dex as never)).toBe("protect");
    expect(() => assumedMoveOf(foe([], [], []), dex as never)).toThrow(/no move/);
    expect(otherFoeMove(foe(["protect", "thunderbolt", "tackle"], [], ["x"]), dex as never)).toBe("thunderbolt");
    expect(otherFoeMove(foe(["protect"], ["Tackle"], ["x"]), dex as never)).toBe("tackle");
  });
});

describe("outcomes", () => {
  it("has exactly one row per (own move part, branch), sorted, with the success chance floored from the two chances", () => {
    const c = twoFoes();
    const log: ParsedMechanicsQuery[] = [];
    const revealed = { "*": ["thunderbolt", "protect", "fakeout"] };
    const outcome = (asked: string): Outcome => (asked === "protect" ? { kind: "computed", answer: { kind: "move_outcome", precondition: "chance", precondition_ppm: 333_333, hit_ppm: 999_999 } } : { kind: "computed", answer: { kind: "move_outcome", precondition: "passes", precondition_ppm: 1_000_000, hit_ppm: 950_000 } });
    const features = build(c, { probe: probe(log, outcome), assume: assuming(revealed) });
    const state = stateOf(c);
    const foes = Object.keys(state.opponent.slots).sort().map((letter, position) => ({ letter, loc: position + 1, mon: state.opponent.roster[state.opponent.slots[letter] as number] })).filter((f) => f.mon !== undefined && !f.mon.fainted);
    const body = c.request.request as unknown as { active: { moves: { id: string }[] }[]; side: { pokemon: { condition: string }[] } };
    const parts = new Set<string>();
    for (const { choice } of candidatesOf(c)) parseJointChoice(choice).forEach((part, slot) => void (part.kind === "move" && !/ fnt$/.test(body.side.pokemon[slot]?.condition ?? "") && parts.add(`${slot}/${part.index}/${part.target}/${part.mega}`)));
    const aimed = (target: number | null) => (target !== null && target > 0 ? foes.find((f) => f.loc === target) : foes[0]);
    const expected = [...parts].flatMap((key) => {
      const [slot, move, target, mega] = key.split("/");
      if (aimed(target === "null" ? null : Number(target)) === undefined) return [];
      return ["switch|", "move|fakeout", "move|protect", "move|thunderbolt"].map((b) => `${slot}/${move}/${target === "null" ? 0 : target}/${mega}/${b}`);
    });
    const got = features.outcomes.map((r) => `${r.slot}/${r.move}/${r.target}/${r.mega}/${r.action}|${r.foe_move ?? ""}`);
    expect([...got].sort()).toEqual([...expected].sort());
    // the canonical order: slot, move, target, Mega (false first), switch first, then the foe move by id
    const key = (r: OutcomeRow) => [r.slot, r.move, r.target, Number(r.mega), r.action === "switch" ? 0 : 1, r.foe_move ?? ""] as const;
    for (let i = 1; i < features.outcomes.length; i++) {
      const [a, b] = [key(features.outcomes[i - 1] as OutcomeRow), key(features.outcomes[i] as OutcomeRow)];
      const order = a.findIndex((x, k) => x !== b[k]);
      expect(order < 0 || (a[order] as never) < (b[order] as never), `row ${i}`).toBe(true);
    }
    for (const row of features.outcomes) {
      expect(row.assumed).toBe(false);
      if (row.kind !== "computed") continue;
      const slotMove = body.active[row.slot]?.moves[row.move - 1]?.id;
      expect(row.success_ppm).toBe(slotMove === "protect" ? 333_332 : 950_000);
      expect([row.precondition, row.precondition_ppm, row.hit_ppm]).toEqual(slotMove === "protect" ? ["chance", 333_333, 999_999] : ["passes", 1_000_000, 950_000]);
    }
    expect(features.outcomes.length).toBeGreaterThan(0);
  });

  it("asks one move_outcome query per row: the asked action first, every foe acting, the ally declared with no action, the foe at the target", () => {
    const c = twoFoes();
    const log: ParsedMechanicsQuery[] = [];
    const features = build(c, { probe: probe(log), assume: assuming({ "*": ["protect", "thunderbolt", "fakeout"] }) });
    const queries = log.filter((q) => q.query.kind === "move_outcome").map((q) => q.query as OutcomeQuery);
    expect(queries.length).toBeGreaterThan(0);
    const actives = (c.request.request as unknown as { active: unknown[] }).active.length;
    for (const q of queries) {
      expect(q.asked).toBe(0);
      expect(q.actions[0]?.side).toBe(0);
      expect(q.actions.filter((a) => a.side === 0)).toHaveLength(1); // the ally acts not
      expect(q.sides[0].length).toBeLessThanOrEqual(actives);
      expect(q.target.side).toBe(1);
      const foeActions = q.actions.filter((a) => a.side === 1);
      expect(foeActions).toHaveLength(q.sides[1].length);
      if (foeActions.every((a) => a.move === null)) continue; // the switch branch
      // the other foe acts with its first revealed damaging move: fakeout, protect, thunderbolt are revealed in that order of the list: protect is no damage, thunderbolt is
      const foeMoves = foeActions.map((a) => a.move);
      expect(foeMoves.every((m) => m !== null)).toBe(true);
    }
    expect(features.outcomes.some((r) => r.action === "switch")).toBe(true);
  });

  it("gives a foe with no revealed move one assumed branch from the filler, flagged, beside the switch branch", () => {
    const c = twoFoes();
    const features = build(c, { assume: assuming({ "*": [] }, ["tackle"]) });
    const parts = new Set(features.outcomes.map((r) => `${r.slot}/${r.move}/${r.target}/${r.mega}`));
    expect(parts.size).toBeGreaterThan(0);
    for (const key of parts) {
      const rows = features.outcomes.filter((r) => `${r.slot}/${r.move}/${r.target}/${r.mega}` === key);
      expect(rows.map((r) => [r.action, r.foe_move, r.assumed])).toEqual([["switch", null, false], ["move", "tackle", true]]);
    }
  });

  it("cuts the revealed moves to the record's limit and lists every branch even when the answers are equal", () => {
    const c = twoFoes();
    const features = build(c, { assume: assuming({ "*": ["aerialace", "bite", "crunch", "dig"] }), outcome: { config: { ...record, max_branches: 3 } } });
    const rows = features.outcomes.filter((r) => r.slot === features.outcomes[0]?.slot && r.move === features.outcomes[0]?.move && r.target === features.outcomes[0]?.target && !r.mega);
    expect(rows.map((r) => r.foe_move)).toEqual([null, "aerialace", "bite"]);
    expect(new Set(rows.map((r) => JSON.stringify([r.kind, (r as { success_ppm?: number }).success_ppm]))).size).toBe(1); // equal answers, still three rows
  });

  it("says unsupported with the probe's reason (history_unknown included) and turns a refused declared state into query_kind", () => {
    const c = twoFoes();
    let n = 0;
    const answers: (Outcome | "refuse")[] = [{ kind: "unsupported", reason: "history_unknown", position: null }, { kind: "unsupported", reason: "random_effect", position: 0 }, "refuse"];
    const memo = createFeatureMemo();
    const features = build(c, { probe: probe([], () => answers[n++ % 3] as Outcome | "refuse") }, memo);
    const kinds = new Set(features.outcomes.map((r) => (r.kind === "unsupported" ? r.reason : "computed")));
    expect(kinds).toEqual(new Set(["history_unknown", "random_effect", "query_kind"]));
    expect(memo.counts.probe_refusals).toBeGreaterThan(0);
  });

  it("is a function of the channel, team and candidates: the same bytes with and without the memo and in shuffled candidate order", () => {
    const c = twoFoes();
    const candidates = candidatesOf(c);
    const reversed = [...candidates].reverse();
    const memo = createFeatureMemo();
    const a = JSON.stringify(build(c));
    expect(JSON.stringify(build(c, {}, memo))).toBe(a);
    expect(JSON.stringify(build(c, {}, memo))).toBe(a); // answered from the memo
    expect(JSON.stringify(build(c, {}, undefined, reversed))).toBe(a);
  });

  it("has no outcomes at Team Preview or in a forced switch, and no row for a part aimed at a location without a living foe", () => {
    const forced = captured.find((c) => c.request.phase === "forceSwitch");
    if (forced !== undefined) expect(build(forced).outcomes).toEqual([]);
    const preview = captured.find((c) => c.request.phase === "preview");
    if (preview !== undefined) {
      const f = build(preview);
      expect([f.rows, f.outcomes]).toEqual([[], []]);
    }
    const c = twoFoes();
    const state = stateOf(c);
    const lines = Object.keys(state.opponent.slots).sort().map((letter) => `|faint|${c.context.player === "p1" ? "p2" : "p1"}${letter}: ${state.opponent.roster[state.opponent.slots[letter] as number]?.name}`);
    const lines0 = [...c.context.channel];
    const body = lines0[lines0.length - 1] as string;
    const empty = playerChannel({ schema_version: 1, data_class: "synthetic", player: c.context.player, lines: [...lines0.slice(0, -1), ...lines, body] });
    expect(build(c, {}, undefined, candidatesOf(c), empty).outcomes).toEqual([]);
  });

  it("matches /2 in its row shapes: the same keys and the same arrays are there", () => {
    const c = twoFoes();
    const outcome = build(c);
    const two = candidateFeatures2(engine, { ...configs(), mechanics }, channelOf(c), teamOf(c), candidatesOf(c));
    for (const key of ["rows", "threats", "speeds", "own_priorities", "foe_priorities"] as const) {
      expect(outcome[key].map((r) => [(r as { slot?: number }).slot, (r as { own?: number }).own, (r as { foe?: number }).foe, (r as { move?: number | string }).move, (r as { mega?: boolean }).mega]), key).toEqual(two[key].map((r) => [(r as { slot?: number }).slot, (r as { own?: number }).own, (r as { foe?: number }).foe, (r as { move?: number | string }).move, (r as { mega?: boolean }).mega]));
    }
    expect(outcome.feature_set_hash).not.toBe(two.feature_set_hash);
  });
});

describe("history declared as the channel shows it", () => {
  /** Every query the probe saw, for the request of `c` with `lines` put before the request. */
  function queriesWith(c: Captured, lines: string[]): ParsedMechanicsQuery["query"][] {
    const own = [...c.context.channel];
    const channel = playerChannel({ schema_version: 1, data_class: "synthetic", player: c.context.player, lines: [...own.slice(0, -1), ...lines, own[own.length - 1] as string] });
    const log: ParsedMechanicsQuery[] = [];
    build(c, { probe: probe(log) }, undefined, candidatesOf(c), channel);
    return log.map((q) => q.query);
  }
  const foeSide = (c: Captured) => (c.context.player === "p1" ? "p2" : "p1");

  it("declares a foe's eaten berry as known, and the same request without the line has it not", () => {
    const c = twoFoes();
    const state = stateOf(c);
    const letter = Object.keys(state.opponent.slots).sort()[0] as string;
    const name = state.opponent.roster[state.opponent.slots[letter] as number]?.name as string;
    const eaten = queriesWith(c, [`|-enditem|${foeSide(c)}${letter}: ${name}|Oran Berry|[eat]`]);
    const plain = queriesWith(c, []);
    const flags = (queries: typeof eaten) => queries.flatMap((q) => (q.kind === "move_outcome" || q.kind === "speed_order" ? q.sides.flat() : q.kind === "damage" ? [q.attacker, q.defender] : [])).map((x) => (x as { history?: { ate_berry?: unknown } }).history?.ate_berry);
    expect(flags(eaten)).toContainEqual({ kind: "known", value: true });
    expect(flags(plain)).not.toContainEqual({ kind: "known", value: true });
  });

  it("declares the faint count of a side from its faint lines", () => {
    const c = twoFoes();
    const state = stateOf(c);
    const letter = Object.keys(state.opponent.slots).sort()[0] as string;
    const name = state.opponent.roster[state.opponent.slots[letter] as number]?.name as string;
    // a foe faints and another takes its place in the open request's channel: the facts count one faint on the foe side
    const queries = queriesWith(c, [`|faint|${foeSide(c)}${letter}: ${name}`]);
    const totals = queries.flatMap((q) => (q.history?.sides ?? []).map((s) => (s.total_fainted as { value?: number } | undefined)?.value));
    expect(totals).toContain(1);
  });

  it("never declares an opponent's used moves (its PP): only an own Pokemon's moves_used appears", () => {
    const c = moveRequests()[moveRequests().length - 1] as Captured;
    const log: ParsedMechanicsQuery[] = [];
    build(c, { probe: probe(log) });
    let foeSides = 0;
    for (const { query } of log) {
      if (query.kind !== "move_outcome" && query.kind !== "speed_order") continue;
      for (const combatant of query.sides[1]) {
        foeSides += 1;
        expect((combatant as { history?: { moves_used?: unknown } }).history?.moves_used).toBeUndefined();
      }
    }
    expect(foeSides).toBeGreaterThan(0);
  });

  it("asks a refused channel's facts at both ends: every count unknown", () => {
    const c = twoFoes();
    const state = stateOf(c);
    const letter = Object.keys(state.opponent.slots).sort()[0] as string;
    const name = state.opponent.roster[state.opponent.slots[letter] as number]?.name as string;
    // the facts module refuses a stalling move that is neither succeeded nor failed before the request; the reader takes the line
    const data = { ...goldenHistoryFactsData(), stall_moves: [...goldenHistoryFactsData().stall_moves, "tackle"] };
    const own = [...c.context.channel];
    const channel = playerChannel({ schema_version: 1, data_class: "synthetic", player: c.context.player, lines: [...own.slice(0, -1), `|move|${foeSide(c)}${letter}: ${name}|Tackle|${c.context.player}a: x`, own[own.length - 1] as string] });
    expect(factsOf(channel, data)).toBeNull();
    const log: ParsedMechanicsQuery[] = [];
    build(c, { probe: probe(log), outcome: { config: record, history_data: data } }, undefined, candidatesOf(c), channel);
    const queries = log.map((q) => q.query);
    const counts = queries.flatMap((q) => (q.kind === "move_outcome" ? q.sides.flat() : [])).map((x) => (x as { history?: { move_actions?: unknown } }).history?.move_actions);
    expect(counts.length).toBeGreaterThan(0);
    for (const fact of counts) expect(fact).toEqual({ kind: "unknown", ends: [0, record.unknown_count_end] });
    const lastMoves = queries.flatMap((q) => (q.kind === "move_outcome" ? [q.history?.field_last_move] : []));
    expect(lastMoves[0]).toEqual({ kind: "unknown", ends: [null, record.unknown_move] });
  });
});

describe("the real probe", () => {
  /** The request of `c` with slot 0's first move replaced by `move` (in the request and in the own team), `lines` put before the request. */
  function withOwnMove(c: Captured, move: { id: string; name: string; target: string }, lines: (ident: string, turn: number) => string[]): { channel: PlayerChannel; team: PokemonSet[]; candidates: { choice: string; pending: boolean }[] } {
    const own = [...c.context.channel];
    const body = JSON.parse((own[own.length - 1] as string).slice("|request|".length)) as { side: { pokemon: { ident: string }[] }; active: { moves: { move: string; id: string; target: string; disabled?: boolean }[] }[] };
    body.active[0] = { ...body.active[0], moves: [{ move: move.name, id: move.id, target: move.target }, ...(body.active[0]?.moves.slice(1) ?? [])] } as never;
    const ident = (body.side.pokemon[0] as { ident: string }).ident;
    const turn = own.reduce((last, line) => (line.startsWith("|turn|") ? Number(line.slice(6)) : last), 0);
    const team = structuredClone(teamOf(c));
    const name = ident.slice(ident.indexOf(": ") + 2);
    const set = team.find((s) => [s.name, s.name === "" ? s.species : ""].includes(name) || s.species.startsWith(name.split("-")[0] as string));
    (set as PokemonSet).moves[0] = move.name;
    return {
      channel: playerChannel({ schema_version: 1, data_class: "synthetic", player: c.context.player, lines: [...own.slice(0, -1), ...lines(ident, turn + 1), `|request|${JSON.stringify(body)}`] }),
      team,
      candidates: candidatesOf(c).filter((x) => parseJointChoice(x.choice).some((p) => p.kind === "move" && p.index === 1)),
    };
  }
  const PROTECT = { id: "protect", name: "Protect", target: "self" };
  const used = (request: string, turn: number) => {
    const ident = request.replace(/^(p[12]):/, "$1a:"); // the protocol names a Pokemon in play by its position
    return [`|move|${ident}|Protect|${ident}`, `|-singleturn|${ident}|Protect`, "|upkeep", `|turn|${turn}`];
  };

  it("answers a second consecutive Protect with the engine's 1/3 and a first one with success (hand-worked: the stall counter is declared)", () => {
    let checked = 0;
    for (const c of moveRequests()) {
      for (const lines of [used, () => []]) {
        const built = withOwnMove(c, PROTECT, lines);
        let rows: OutcomeRow[];
        try {
          rows = candidateFeaturesOutcome(engine, { ...configs(), probe: undefined as never }, built.channel, built.team, built.candidates, createFeatureMemo()).outcomes.filter((r) => r.slot === 0 && r.move === 1);
        } catch (error) {
          if (String(error).includes("probe refused a set")) continue;
          throw error;
        }
        if (rows.length === 0) continue;
        const second = lines === used;
        for (const row of rows) {
          expect(row.kind, JSON.stringify(row)).toBe("computed");
          if (row.kind !== "computed") continue;
          if (row.action === "switch") {
            // every foe switches: no foe action is left in the queue, so Protect fails (the engine: no action remains after it)
            expect(row.precondition).toBe("fails");
          } else if (second) expect([row.precondition, row.precondition_ppm, row.success_ppm]).toEqual(["chance", 333_333, 333_333]);
          else expect([row.precondition, row.precondition_ppm, row.success_ppm]).toEqual(["passes", 1_000_000, 1_000_000]);
        }
        checked += 1;
        break;
      }
      if (checked >= 2) break;
    }
    expect(checked).toBeGreaterThanOrEqual(1);
  }, 170_000);
});

describe("failure paths", () => {
  it("lets a defect of the probe or the facts module through, refuses a wrong answer kind and a format without history data", () => {
    const c = twoFoes();
    expect(() => build(c, { probe: probe([], () => { throw new Error("a defect"); }) })).toThrow("a defect");
    const wrong = { kind: "computed", answer: { kind: "stats", stats: { hp: 1, atk: 1, def: 1, spa: 1, spd: 1, spe: 1 } } } as Outcome;
    expect(() => build(c, { probe: probe([], () => wrong) })).toThrow(/no outcome/);
    expect(() => factsOf(channelOf(c), { ...goldenHistoryFactsData(), eat_tags: null, action_sources: null, stall_moves: null, slot_conditions: null } as never)).toThrow(TypeError);
    expect(() => historyDataOf(engine, { config: { format_id: "other" }, outcome: { config: record } } as never)).toThrow(/history facts data/);
    expect(historyDataOf(engine, configs()).stall_moves.length).toBeGreaterThan(0);
  });
  it("gives no rows for a part of a fainted own Pokemon and an unsupported row for a move the engine lacks", () => {
    const c = twoFoes();
    const own = [...c.context.channel];
    const body = JSON.parse((own[own.length - 1] as string).slice("|request|".length)) as { side: { pokemon: { condition: string }[] }; active: { moves: { id: string; move: string; target: string }[] }[] };
    const living = body.active.length;
    (body.side.pokemon[0] as { condition: string }).condition = "0 fnt";
    (body.active[1] as { moves: unknown[] }).moves[0] = { move: "Recharge", id: "recharge", target: "self" };
    const channel = playerChannel({ schema_version: 1, data_class: "synthetic", player: c.context.player, lines: [...own.slice(0, -1), `|request|${JSON.stringify(body)}`] });
    const candidates = [{ choice: "move 1 1, move 1", pending: false }];
    const features = build(c, {}, undefined, candidates, channel);
    expect(living).toBe(2);
    expect(new Set(features.outcomes.map((r) => r.slot))).toEqual(new Set([1]));
    expect(features.outcomes.every((r) => r.kind === "unsupported" && r.reason === "query_kind")).toBe(true);
  });
});

describe("the type rule", () => {
  it("takes no opponent team, omniscient log, hidden verdicts, battle record or seed", () => {
    const neverRun = (log: never, record: never, seed: string, raw: { player: "p1"; lines: string[] }, channel: PlayerChannel): void => {
      const team = [] as never;
      // @ts-expect-error: the omniscient log is not a PlayerChannel
      void candidateFeaturesOutcome(engine, configs(), log, team, []);
      // @ts-expect-error: nor a seed
      void candidateFeaturesOutcome(engine, configs(), seed, team, []);
      // @ts-expect-error: nor an unchecked log
      void candidateFeaturesOutcome(engine, configs(), raw, team, []);
      // @ts-expect-error: the configs hold no opponent team
      void candidateFeaturesOutcome(engine, { ...configs(), opponent_team: [] }, channel, team, []);
      // @ts-expect-error: the team argument is the player's own team, not a record
      void candidateFeaturesOutcome(engine, configs(), channel, record, []);
    };
    expect(neverRun).toBeTypeOf("function");
  });
});
