// The S0 measurement tools (ADR-026 section 8 S0 measures 3, 4, 5 and 8; #1308) on a committed fixture battle: the oracle rollouts
// from the test-only fork against the real opponent policy, the exhaustive re-rank and its recall, the split halves, the HP/KO
// variant, and the per-component cost sink. A stand-in worker (no Python) stands for the pair; the tool code is the real one.
import { configuredSetHash, COST_COMPONENTS, type CostComponent, damageAwareSetup, loadDamageAwareConfig, loadMechanicsConfig, loadSearchConfig, randomLegalActor, syntheticTeamHashes, teamHash, type Actor, type FeatureConfigs } from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { captureDecisions, type OracleSpec } from "../../showdown-bridge/test/fidelity/oracle.ts";
import { loadRandomCorpus } from "../../showdown-bridge/src/random-corpus.ts";
import { type CostSink } from "../../showdown-bridge/src/search-cost.ts";
import { foeChannelAt, oracleRollout, oracleValues, rolloutSeed } from "./s0/rollout.ts";
import { type BattleMeasure, MAX_CAPPED_PPM, measureBattle, summarise } from "./s0/measure.ts";
import { CAPPED_SCORE_PPM } from "./s0/rollout.ts";
import { pairGameSpec } from "./s0/games.ts";
import { engine, FORMAT } from "./support.ts";
import { manifest, playSearch, portFor, setupFor, snapshot, ValueStandInWorker, vocabulary } from "./search-actor-support.ts";

const battle = loadRandomCorpus()[0] as OracleSpec & ReturnType<typeof loadRandomCorpus>[number];

describe("rollout helpers", () => {
  it("cuts the opponent's channel at its first request after the turn line", () => {
    const channel = ["|start", "|turn|1", "|request|{}", "|move|x", "|turn|2", "|request|{\"a\":1}", "|request|{\"b\":2}"];
    expect(foeChannelAt(channel, 1)).toEqual(["|start", "|turn|1", "|request|{}"]);
    expect(foeChannelAt(channel, 2)).toEqual(channel.slice(0, 6));
    expect(() => foeChannelAt(channel, 3)).toThrow(/no turn 3/);
    expect(() => foeChannelAt(["|turn|1"], 1)).toThrow(/no request/);
  });

  it("derives a seed from the battle, seat, turn and rollout only", () => {
    const a = rolloutSeed("b", "p1", 3, 0);
    expect(a).toMatch(/^sodium,[0-9a-f]{32}$/);
    expect(rolloutSeed("b", "p1", 3, 0)).toBe(a);
    for (const other of [rolloutSeed("c", "p1", 3, 0), rolloutSeed("b", "p2", 3, 0), rolloutSeed("b", "p1", 4, 0), rolloutSeed("b", "p1", 3, 1)]) expect(other).not.toBe(a);
  });
});

describe("oracle rollouts from the fork", () => {
  it("play a candidate to a score, leave the fork untouched, ask the actors with real channels, and repeat exactly", async () => {
    const { decisions, record } = await captureDecisions(engine, battle);
    const d = decisions.find((x) => x.turn >= 2 && x.recorded !== null) ?? (decisions.find((x) => x.recorded !== null) as (typeof decisions)[number]);
    const before = JSON.stringify(d.fork);
    const foe = d.player === "p1" ? "p2" : "p1";
    const asked: { seat: string; last: string; seq: number; count: number }[] = [];
    const spy = (seat: string, seed: number): ((rollout: number) => Actor) => (rollout) => {
      const inner = randomLegalActor(seed + rollout);
      return (request, ctx) => {
        const channel = ctx?.channel ?? [];
        asked.push({ seat, last: channel.at(-1)?.slice(0, 9) ?? "", seq: request.request_seq, count: channel.filter((l) => l.startsWith("|request|")).length });
        return inner(request, ctx);
      };
    };
    const spec = { engine, decision: d, foe_channel: foeChannelAt(record.channels[foe], d.turn), teams: battle.teams, policies: { own: spy(d.player, 11), foe: spy(foe, 22) }, max_turns: 3 };
    const candidates = d.recorded === null ? [] : [d.recorded.own];
    expect(candidates).toHaveLength(1);
    const first = await oracleValues(spec, candidates, 2);
    expect(JSON.stringify(d.fork)).toBe(before);
    expect(first.scores).toHaveLength(1);
    expect(first.scores[0]).toHaveLength(2);
    for (const s of first.scores[0] as number[]) {
      expect(s).toBeGreaterThanOrEqual(0);
      expect(s).toBeLessThanOrEqual(1_000_000);
    }
    // every actor call saw a channel that ends in the request it answers, numbered by the requests in it
    expect(asked.length).toBeGreaterThan(0);
    for (const a of asked) {
      expect(a.last).toBe("|request|");
      expect(a.seq).toBe(a.count);
    }
    // the same rollout again gives the same score; a cap of zero extra turns is reported as capped
    expect(await oracleValues(spec, candidates, 2)).toEqual(first);
    const capped = await oracleRollout({ ...spec, max_turns: 0 }, candidates[0] as string, 0);
    expect(capped.capped || capped.score_ppm !== null).toBe(true);
  }, 300_000);
});

describe("the S0 pipeline on a fixture battle", () => {
  it("measures decisions end to end and summarises them", async () => {
    const base = loadSearchConfig(FORMAT);
    const small = { ...base, worlds: { ...base.worlds, n: 4 }, min_valid_worlds: 2 } as typeof base;
    const hp = { ...small, leaf: { kind: "hp-ko-differential" } } as typeof base;
    const port = portFor();
    const options = {
      engine,
      manifest,
      snapshot,
      setup: setupFor(port, { config: small }),
      setup_hp: setupFor(port, { config: hp }),
      worker: new ValueStandInWorker(),
      pair: { preview: "a".repeat(64), battle: "b".repeat(64) },
      sampling: "argmax" as const,
      rollouts: 2,
      rollout_max_turns: 2,
      seats: ["p1", "p2"] as const,
      max_decisions: 1,
    };
    const measured = await measureBattle(options, battle, 0);
    expect(measured.seen).toBeGreaterThan(0);
    expect(measured.measures.length).toBeGreaterThan(0);
    for (const m of measured.measures) {
      expect(m.exhaustive_sums).toHaveLength(m.candidates);
      expect(m.ranked).toHaveLength(m.candidates);
      expect(m.top.length).toBeLessThanOrEqual(base.top_k);
      expect(m.hp_q_ppm).toHaveLength(m.top.length);
      expect(m.oracle).toHaveLength(m.top.length);
      for (const row of m.oracle) expect(row).toHaveLength(2);
      expect(m.top.every((i) => m.ranked.includes(i))).toBe(true);
    }
    const summary = summarise([measured]);
    expect(summary.decisions).toBe(measured.measures.length);
    expect(summary.recall.map((r) => `${r.rule}:${r.k}`)).toEqual(["prior-top-k:4", "prior-top-k:6", "prior-top-k:8", "distinct-slot:4", "distinct-slot:6", "distinct-slot:8", "slot-cover:4", "slot-cover:6", "slot-cover:8"]);
    expect(summary.recall[2]?.hits).toBeGreaterThanOrEqual(summary.recall[0]?.hits ?? 0);
    expect(summary.split_half.decisions).toBe(summary.decisions);
    expect(summary.leaf_value_head.decisions).toBe(summary.decisions);
    expect(summary.calibration_hp_ko).toHaveLength(5);
    // reported: the paired means with the decisions dropped, the capped share, the alias, the clustered AUC
    expect(summary.leaf_value_head.dropped).toBe(summary.decisions - summary.leaf_value_head.paired);
    expect(summary.split_half.dropped).toBe(summary.decisions - summary.split_half.defined);
    expect(summary.rollouts).toBe(measured.measures.reduce((n, m) => n + m.oracle.length * 2, 0));
    expect(summary.capped_share_ppm).toBe(Math.floor((summary.capped_rollouts * 1_000_000) / summary.rollouts));
    expect(summary.recall.filter((r) => r.alias_of === "prior-top-k").map((r) => r.rule)).toEqual(["distinct-slot", "distinct-slot", "distinct-slot"]);
    expect(summary.recall.slice(3, 6).map((r) => r.hits)).toEqual(summary.recall.slice(0, 3).map((r) => r.hits));
    expect(summary.value_head_auc.games).toBeLessThanOrEqual(2);
    expect(summary.notes.join(" ")).toContain("alias of prior-top-k");
  }, 600_000);
});

const measure = (capped: number, rollouts = 2): BattleMeasure => ({
  battle_id: "b",
  measures: [{ battle_id: "b", player: "p1", turn: 1, candidates: 2, candidate_list: [{ choice: "move 1", pending: false }, { choice: "move 2", pending: false }], top: [0, 1], prior_ppm: [600_000, 400_000], ranked: [0, 1], exhaustive_sums: [2, 1], exhaustive_q_ppm: [2, 1], hp_q_ppm: [2, 1], oracle: [Array(rollouts).fill(CAPPED_SCORE_PPM), Array(rollouts).fill(0)], capped_rollouts: capped, value_ppm: 500_000, outcome_ppm: 1_000_000, worlds_scored: 4 }],
  seen: 1,
  refused: 0,
});

describe("capped rollouts", () => {
  it("score a tie, and a run with more than the limit is flagged", () => {
    expect(CAPPED_SCORE_PPM).toBe(500_000);
    expect(MAX_CAPPED_PPM).toBe(200_000);
    const fine = summarise([measure(0)]);
    expect(fine).toMatchObject({ rollouts: 4, capped_rollouts: 0, capped_share_ppm: 0, capped_within_limit: true });
    // 1 of 4 is 250,000 ppm: above the limit
    expect(summarise([measure(1)])).toMatchObject({ capped_share_ppm: 250_000, capped_within_limit: false });
    expect(summarise([])).toMatchObject({ rollouts: 0, capped_share_ppm: null, capped_within_limit: true });
  });
});

describe("a features pair against damage-aware@v1 (the S0 pair)", () => {
  it("plays the game, searches it and rolls out with the pair's features and a fresh damage-aware opponent", async () => {
    const configs: FeatureConfigs = { vocabulary, config: loadDamageAwareConfig("v1", vocabulary), mechanics: loadMechanicsConfig(FORMAT), snapshot, data_class: "synthetic" };
    const worker = Object.assign(new ValueStandInWorker(), { featureSet: { feature_set: "probe-damage/1", feature_set_hash: configuredSetHash(engine, configs) } });
    const base = loadSearchConfig(FORMAT);
    const small = { ...base, worlds: { ...base.worlds, n: 4 }, min_valid_worlds: 2 } as typeof base;
    const hp = { ...small, leaf: { kind: "hp-ko-differential" } } as typeof base;
    const port = portFor();
    const pair = { preview: "a".repeat(64), battle: "b".repeat(64) };
    const known = syntheticTeamHashes([teamHash(battle.teams.p1), teamHash(battle.teams.p2)]);
    const spec = pairGameSpec({ engine, manifest, client: worker, pair, sampling: "argmax", features: configs, damage: damageAwareSetup(engine, FORMAT), id: "f0", seed: battle.seed, teams: battle.teams, seat: "p1", game: 0, known });
    expect(Object.keys(spec.make_actor ?? {}).sort()).toEqual(["p1", "p2"]);
    const measured = await measureBattle({ engine, manifest, snapshot, setup: setupFor(port, { config: small }), setup_hp: setupFor(port, { config: hp }), worker, pair, sampling: "argmax", features: configs, rollouts: 2, rollout_max_turns: 2, max_decisions: 1 }, spec, 0, "p1");
    expect(measured.measures.length).toBeGreaterThan(0);
    expect(measured.measures.every((m) => m.player === "p1")).toBe(true);
    // the pair's play (the game and the rollouts) and every root query carried rows; the leaves carried none
    const asked = worker.queries.filter((q) => q.kind === "ask");
    expect(asked.length).toBeGreaterThan(0);
    expect(asked.every((q) => q.rows)).toBe(true);
    const roots = worker.queries.filter((q) => q.kind === "evaluate" && q.candidates > 0);
    expect(roots.length).toBeGreaterThan(0);
    expect(roots.every((q) => q.rows)).toBe(true);
    expect(worker.queries.filter((q) => q.kind === "value").every((q) => q.candidates === 0 && !q.rows)).toBe(true);
    // a features worker with no configs (or the reverse) is refused
    await expect(measureBattle({ engine, manifest, snapshot, setup: setupFor(port, { config: small }), setup_hp: setupFor(port, { config: hp }), worker, pair, sampling: "argmax", rollouts: 2, rollout_max_turns: 2, max_decisions: 1 }, spec, 0, "p1")).rejects.toThrow(/go together/);
  }, 600_000);
});

describe("per-component cost lines", () => {
  it("time every component without changing a choice or a count", async () => {
    const totals = new Map<CostComponent, number>();
    const sink: CostSink = { add: (component, ms) => void totals.set(component, (totals.get(component) ?? 0) + ms) };
    const plain = await playSearch(battle, { worker: new ValueStandInWorker() });
    const timed = await playSearch(battle, { worker: new ValueStandInWorker(), options: { cost: sink } });
    expect(timed.record.choices).toEqual(plain.record.choices);
    expect(timed.actor.report().counts).toEqual(plain.actor.report().counts);
    expect(timed.actor.report().counts.searched_decisions).toBeGreaterThan(0);
    for (const component of COST_COMPONENTS) expect(totals.get(component), component).toBeGreaterThan(0);
  }, 300_000);
});

