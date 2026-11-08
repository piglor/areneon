// The S0 measurement pipeline on seeded A1 battles (ADR-026 section 8 S0 measures 3, 4 and 8; #1147; #1308). TEST TOOLING: it
// reads the test-only fork oracle (`showdown-bridge/test/fidelity/oracle.ts`) and the real opponent policy through `rollout.ts`, so
// like them it lives under test/ and `fidelity-isolation.test.ts` keeps its identifiers out of shipped code.
//
// For each sampled own move decision of a replayed battle it computes, from the player's own view only (the search side never
// sees the fork): the pair's prior, the EXHAUSTIVE re-rank (every non-pending candidate searched with the value-head leaf), the
// HP/KO-leaf scores of the configured top-k, and the oracle rollout values of that top-k. `summarise` turns the decisions into the
// integers and ppm of the S0 tables with the arithmetic of `s0-measures.ts`.

import {
  battleClass,
  decodeRequest,
  enumerateCandidates,
  type Engine,
  type EvaluationClient,
  type PokemonSet,
  type PolicyClient,
  type RulesSnapshot,
  policyActor,
  observe,
  playerChannel,
  policyId,
  randomLegalActor,
  pendingFirstActor,
  searchActor,
  searchedIndices,
  searchPolicy,
  type SearchSetup,
  searchStream,
  sheetRegimeOf,
  stampTeam,
  syntheticTeamHashes,
  teamHash,
  worldRequestOf,
  type VersionManifest,
  type Actor,
  aucPpm,
  leafQuality,
  rolloutSums,
  siblingCalibration,
  splitHalf,
  topKRecall,
} from "@areneon/showdown-bridge";
import type { Candidate } from "../../../showdown-bridge/src/choices.ts";
import { type Decision, type OracleSpec, captureDecisions } from "../../../showdown-bridge/test/fidelity/oracle.ts";
import { foeChannelAt, oracleValues, type RolloutPolicies } from "./rollout.ts";

type Player = "p1" | "p2";
const other = (p: Player): Player => (p === "p1" ? "p2" : "p1");

export interface MeasureOptions {
  engine: Engine;
  manifest: VersionManifest;
  snapshot?: RulesSnapshot;
  /** The search under test (the value-head leaf) and the same search with the HP/KO leaf. */
  setup: SearchSetup;
  setup_hp: SearchSetup;
  /** The frozen pair's worker: the prior, the leaves and the pair's continuation play in the rollouts. */
  worker: PolicyClient & EvaluationClient;
  pair: { preview: string; battle: string };
  sampling: "argmax" | "sample";
  /** Oracle rollouts per candidate (even: the split halves), and turns a rollout may play before it is cut. */
  rollouts: number;
  rollout_max_turns: number;
  /** The seats measured, and the most decisions taken from each battle (every `stride`-th searchable one). */
  seats?: readonly Player[];
  max_decisions: number;
  stride?: number;
}

export interface DecisionMeasure {
  battle_id: string;
  player: Player;
  turn: number;
  /** Non-pending candidates, and the searched top-k (indices into the request's candidates, prior rank). */
  candidates: number;
  top: number[];
  /** Every candidate of the request in canonical order (choice text and pending flag). */
  candidate_list: { choice: string; pending: boolean }[];
  /** All candidates' prior (ppm, canonical order) and the exhaustive re-rank: sums and the mean q (ppm) in prior rank, `ranked` the indices. */
  prior_ppm: number[];
  ranked: number[];
  exhaustive_sums: number[];
  exhaustive_q_ppm: number[];
  /** The same top-k under the HP/KO leaf. */
  hp_q_ppm: number[];
  /** Oracle rollout scores of the top-k: [candidate of `top`][rollout]. */
  oracle: number[][];
  capped_rollouts: number;
  /** The value head on the real state (ppm) and the seat's result of the game (ppm). */
  value_ppm: number;
  outcome_ppm: number;
  worlds_scored: number;
}

export interface BattleMeasure {
  battle_id: string;
  measures: DecisionMeasure[];
  /** Searchable decisions seen, decisions the search refused (belief or too few valid worlds). */
  seen: number;
  refused: number;
}

const foeActors = (spec: OracleSpec, foe: Player): ((rollout: number) => Actor) => {
  const kind = spec.actors?.[foe] ?? "random-legal";
  return (rollout) => {
    const seed = ((spec.actor_seeds[foe] as number) ^ Math.imul(rollout + 1, 0x9e3779b1)) >>> 0;
    return kind === "pending-first" ? pendingFirstActor(seed) : randomLegalActor(seed);
  };
};

/** Measure one replayed battle. */
export async function measureBattle(options: MeasureOptions, spec: OracleSpec, game: number): Promise<BattleMeasure> {
  const { engine, manifest, setup, setup_hp, worker, pair, sampling } = options;
  const { decisions, record } = await captureDecisions(engine, spec);
  const known = syntheticTeamHashes([teamHash(spec.teams.p1), teamHash(spec.teams.p2)]);
  const channel_class = battleClass({ p1: teamHash(spec.teams.p1), p2: teamHash(spec.teams.p2) }, known);
  const out: BattleMeasure = { battle_id: spec.id, measures: [], seen: 0, refused: 0 };
  const taken: Record<string, number> = {};
  const regime = sheetRegimeOf(manifest.showdown_commit, manifest.format_id);
  const actors = new Map<string, { value: ReturnType<typeof searchActor>; hp: ReturnType<typeof searchActor> }>();
  const policy = searchPolicy(setup, manifest, pair, sampling);
  const policyHp = searchPolicy(setup_hp, manifest, pair, sampling);

  for (const d of decisions) {
    const seat = d.player;
    if (options.seats !== undefined && !options.seats.includes(seat)) continue;
    if ((taken[seat] ?? 0) >= options.max_decisions) continue;
    const text = ([...d.channel].reverse().find((line) => line.startsWith("|request|")) as string).slice("|request|".length);
    const candidates = enumerateCandidates(decodeRequest(text)).candidates;
    const open = candidates.flatMap((c, i) => (c.pending ? [] : [i]));
    if (open.length < 2) continue;
    out.seen += 1;
    if ((out.seen - 1) % (options.stride ?? 1) !== 0) continue;

    const team = stampTeam(spec.teams[seat] as never, known);
    const make = (s: SearchSetup, p: typeof policy) =>
      searchActor({ setup: s, client: worker, battle: `s0-${spec.id}-${seat}`, player: seat, team, manifest, channel_class, seed: spec.seed, game, policy_id: p.policy_id, identity_hash: p.identity_hash, sampling });
    const key = `${seat}`;
    if (!actors.has(key)) actors.set(key, { value: make(setup, policy), hp: make(setup_hp, policyHp) });
    const { value: valueActor, hp: hpActor } = actors.get(key) as NonNullable<ReturnType<typeof actors.get>>;

    const observation = await observe({ channel: playerChannel({ schema_version: 1, data_class: channel_class, player: seat, lines: d.channel }, regime), team, manifest });
    const root = await worker.evaluate(`s0-${spec.id}-${seat}`, {
      schema_version: 1,
      identity: observation.identity,
      observation,
      candidates: candidates.map(({ choice, pending }) => ({ choice, pending })),
      sampling: { kind: "evaluate" },
      policy_id: policy.policy_id,
    } as Parameters<EvaluationClient["evaluate"]>[1]);
    const ranked = searchedIndices({ top_k: open.length, top_k_rule: "prior-top-k" }, candidates, root.prior_ppm);
    const top = searchedIndices(setup.config, candidates, root.prior_ppm);
    const world = worldRequestOf(setup.facts_data, seat, { channel: d.channel, team: spec.teams[seat] as PokemonSet[] });
    const stream = searchStream({ seed: spec.seed, game, player: seat, policy_id: policy.policy_id, request_seq: d.request_seq });
    const exhaustive = await valueActor.decide({ world, channel: d.channel, top: ranked.map((i) => candidates[i] as Candidate), stream });
    if (exhaustive.kind !== "choice") {
      out.refused += 1;
      continue;
    }
    // The HP/KO leaf draws its own worlds from the same stream, so it is the same search with another leaf.
    const hp = await hpActor.decide({ world, channel: d.channel, top: top.map((i) => candidates[i] as Candidate), stream });
    if (hp.kind !== "choice") {
      out.refused += 1;
      continue;
    }
    const foe = other(seat);
    const policies: RolloutPolicies = {
      own: (rollout) => rolloutPair(options, spec, known, channel_class, seat, game, rollout),
      foe: foeActors(spec, foe),
    };
    const oracle = await oracleValues(
      { engine, decision: d, foe_channel: foeChannelAt(record.channels[foe], d.turn), teams: spec.teams, policies, max_turns: options.rollout_max_turns },
      top.map((i) => (candidates[i] as { choice: string }).choice),
      options.rollouts,
    );
    out.measures.push({
      battle_id: spec.id,
      player: seat,
      turn: d.turn,
      candidates: open.length,
      candidate_list: candidates.map(({ choice, pending }) => ({ choice, pending })),
      top,
      prior_ppm: root.prior_ppm,
      ranked,
      exhaustive_sums: exhaustive.scores.map((s) => s.sum),
      exhaustive_q_ppm: exhaustive.scores.map((s) => s.q_ppm),
      hp_q_ppm: hp.scores.map((s) => s.q_ppm),
      oracle: oracle.scores,
      capped_rollouts: oracle.capped,
      value_ppm: root.value_ppm,
      outcome_ppm: record.winner === null ? 500_000 : record.winner === seat ? 1_000_000 : 0,
      worlds_scored: exhaustive.worlds_scored,
    });
    taken[seat] = (taken[seat] ?? 0) + 1;
  }
  return out;
}

/** The pair's continuation play in a rollout: the same worker, the seat's own team, a stream of its own per rollout. */
function rolloutPair(options: MeasureOptions, spec: OracleSpec, known: ReturnType<typeof syntheticTeamHashes>, channel_class: ReturnType<typeof battleClass>, seat: Player, game: number, rollout: number): Actor {
  return policyActor({
    client: options.worker,
    battle: `s0-${spec.id}-${seat}-rollout`,
    player: seat,
    team: stampTeam(spec.teams[seat] as never, known),
    manifest: options.manifest,
    channel_class,
    seed: `${spec.seed}:rollout:${rollout}`,
    game: game * 1000 + rollout,
    policy_id: policyId(options.pair, options.sampling, {}),
    sampling: options.sampling,
    engine: options.engine,
  });
}

/** The ranked-prefix of the exhaustive scores that the configured top-k covers, in the order of `m.top`. */
const searchedSums = (m: DecisionMeasure): number[] => m.top.map((i) => m.exhaustive_q_ppm[m.ranked.indexOf(i)] as number);

export interface S0Summary {
  decisions: number;
  battles: number;
  refused_decisions: number;
  capped_rollouts: number;
  recall: ReturnType<typeof topKRecall>[];
  leaf_value_head: ReturnType<typeof leafQuality>;
  leaf_hp_ko: ReturnType<typeof leafQuality>;
  calibration_value_head: ReturnType<typeof siblingCalibration>;
  calibration_hp_ko: ReturnType<typeof siblingCalibration>;
  split_half: ReturnType<typeof splitHalf>;
  value_head_auc_ppm: number | null;
  value_head_auc_decisions: number;
}

const EDGES = [0, 50_000, 100_000, 200_000, 400_000];

/** Turn decisions into the S0 tables: every measure carries its own indices and ppm. */
export function summarise(battles: readonly BattleMeasure[]): S0Summary {
  const all = battles.flatMap((b) => b.measures);
  const oracleMeans = (m: DecisionMeasure): number[] => rolloutSums(m.oracle).map((s) => Math.floor(s / (m.oracle[0] as number[]).length));
  const search = (m: DecisionMeasure): number[] => searchedSums(m);
  const prior = (m: DecisionMeasure): number[] => m.top.map((i) => m.prior_ppm[i] as number);
  const set = (pick: (m: DecisionMeasure) => number[]) => all.map((m) => ({ search: pick(m), prior: prior(m), oracle: oracleMeans(m) }));
  const exhaustive = all.map((m) => {
    // exhaustive scores are in prior rank; the recall reads them by candidate index
    return { candidates: m.candidate_list as Candidate[], prior_ppm: m.prior_ppm, scores: m.ranked.map((candidate, j) => ({ candidate, sum: m.exhaustive_sums[j] as number })) };
  });
  const recall: ReturnType<typeof topKRecall>[] = [];
  for (const k of [4, 6, 8]) recall.push(topKRecall(exhaustive, "prior-top-k", k));
  for (const k of [4, 6, 8]) recall.push(topKRecall(exhaustive, "distinct-slot", k));
  for (const k of [4, 6, 8]) recall.push(topKRecall(exhaustive, "slot-cover", k));
  const decided = all.filter((m) => m.outcome_ppm !== 500_000);
  return {
    decisions: all.length,
    battles: battles.length,
    refused_decisions: battles.reduce((n, b) => n + b.refused, 0),
    capped_rollouts: all.reduce((n, m) => n + m.capped_rollouts, 0),
    recall,
    leaf_value_head: leafQuality(set(search), "s0-leaf-value-head"),
    leaf_hp_ko: leafQuality(set((m) => m.hp_q_ppm), "s0-leaf-hp-ko"),
    calibration_value_head: siblingCalibration(all.map((m) => ({ predicted: search(m), oracle: oracleMeans(m) })), EDGES),
    calibration_hp_ko: siblingCalibration(all.map((m) => ({ predicted: m.hp_q_ppm, oracle: oracleMeans(m) })), EDGES),
    split_half: splitHalf(all.map((m) => m.oracle)),
    value_head_auc_ppm: aucPpm(decided.map((m) => m.value_ppm), decided.map((m) => m.outcome_ppm === 1_000_000)),
    value_head_auc_decisions: decided.length,
  };
}

export type { Decision };
