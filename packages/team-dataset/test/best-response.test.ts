// The best-response evaluation (ADR-029 section 2, #1257) with the real engine, the real opponent model and a stand-in worker
// (the real worker is driven by best-response.python.test.ts). The stand-in has a known ground truth so the test has teeth: the
// evaluated policy always plays ONE fixed strategy, team B wins a game exactly when its first lead is the counter of team A's first
// lead (the side that is wrong forfeits at its first Battle request), and the stand-in's value for a hypothetical first request
// is that same truth. So the responder, which knows the policy, counters it, and the blind control, which averages over every
// lead pair of A, does not: a positive exploitation margin. A uniform evaluated policy has the blind control's own weights, so
// the two choose the same strategy, play the same games, and the margin is exactly zero.
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { readerSha256 } from "../../public-state/src/index.ts";
import {
  type BestResponseClient,
  type BestResponseOptions,
  BestResponseError,
  type BestResponseRecord,
  jcs,
  type Json,
  loadRulesSnapshot,
  PolicyProtocolError,
  PolicyReplyError,
  type ResponderTeamPort,
  type PolicyClient,
  type EvaluationClient,
  playBestResponse,
  progressPlan,
  readBestResponseRecord,
  readTrainingRunConfig,
  withEvalDataset,
  resolveFormat,
} from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it } from "vitest";
import { verifiedDatasetHash } from "../src/dataset-hash.ts";
import { DATASETS_DIR, loadFixtureInputs, readFixtureRun, recordedPilot, regenerate, writeFixtures } from "../src/fixtures.ts";
import { createOpponentTeams } from "../src/opponent-teams.ts";
import { datasetValidationPool } from "../src/progress-pool.ts";
import { goldenTeamVocabulary } from "../../showdown-bridge/src/golden.ts";
import { engine, FORMAT } from "./support.ts";

const ROOT = new URL("../../..", import.meta.url).pathname;
const SOURCE = join(DATASETS_DIR(FORMAT), "fixtures-v1");
const tiny = JSON.parse(readFileSync(join(ROOT, "fixtures/contracts/v1/training-run-config/valid/tiny.json"), "utf8")) as Record<string, unknown>;
// A short game: the stand-in's forfeit decides every game at its first Battle request.
const config = readTrainingRunConfig({ ...tiny, generation: { ...(tiny.generation as object), max_turns: 2 }, team_dataset: { path: SOURCE, dataset_hash: verifiedDatasetHash(engine, SOURCE, FORMAT) } });
const pool = datasetValidationPool(engine, config);
const checkpoints = { preview: "1".repeat(64), battle: "2".repeat(64) };
// A second tiny dataset (the committed fixture's sizes and pilot under other keys): the evaluation-dataset option's teams.
const scratch = mkdtempSync(join(tmpdir(), "best-response-eval-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));
const inputs = loadFixtureInputs(engine, FORMAT);
const SECOND = join(scratch, "second");
writeFixtures(SECOND, regenerate(inputs, { ...readFixtureRun(SOURCE), dataset_id: "eval-v1", generation_key: "11".repeat(32), assignment_key: "22".repeat(32) }, recordedPilot(inputs, SOURCE)));
const secondHash = verifiedDatasetHash(engine, SECOND, FORMAT);
const manifest = resolveFormat(engine, FORMAT, loadRulesSnapshot(engine.commit)).manifest;

// The first validation pair: team A (the evaluated side's) and team B (the responder's).
const [teamA, teamB] = progressPlan({ config, pool, games: 2, training_teams: new Set(), league_seeds: new Set() }).pairs[0] as ReturnType<typeof progressPlan>["pairs"][number];
const speciesA = teamA.sets.map((set) => set.species);
const speciesB = teamB.sets.map((set) => set.species);
const signature = (names: readonly string[]): string => names.join("|");

/** B's counter to a lead at (1-based) position `a` of A: positions 1 to 4 of B. */
const counter = (a: number): number => ((a + 2) % 4) + 1;

interface Seen {
  own: string;
  opp: string;
}
/** The first lead species of each side in a log (the `|switch|` lines of the first slot). */
function leads(player: string, log: readonly string[]): Seen | null {
  const found: Record<string, string> = {};
  for (const line of log) {
    const m = /^\|switch\|(p[12])a: [^|]*\|([^,|]*)/.exec(line);
    if (m !== null && found[m[1] as string] === undefined) found[m[1] as string] = m[2] as string;
  }
  const own = found[player];
  const opp = found[player === "p1" ? "p2" : "p1"];
  return own === undefined || opp === undefined ? null : { own, opp };
}

type Query = Parameters<PolicyClient["ask"]>[1];
type Observation = { player: string; log: readonly string[]; team: { team: { species: string }[] }; kind: string };

/** Whether B's lead is the counter of A's, from the first leads in a log; null while a side's lead is unseen. */
function bCorrect(observation: Observation): { b: boolean; truth: boolean } | null {
  const mine = signature(observation.team.team.map((s) => s.species));
  const b = mine === signature(speciesB);
  const seen = leads(observation.player, observation.log);
  if (seen === null) return null;
  const aLead = (b ? seen.opp : seen.own).replace(/\s+/g, "");
  const bLead = (b ? seen.own : seen.opp).replace(/\s+/g, "");
  const strip = (names: string[]): string[] => names.map((n) => n.replace(/\s+/g, ""));
  const a = strip(speciesA).indexOf(aLead) + 1;
  const bPos = strip(speciesB).indexOf(bLead) + 1;
  return { b, truth: a > 0 && bPos === counter(a) };
}

/** The stand-in worker: a fixed (decreasing) Preview prior or a uniform one, and the ground truth as battle behaviour and value. */
class Standin implements PolicyClient, EvaluationClient {
  readonly featureSet = null;
  queries = 0;
  readonly uniform: boolean;
  readonly fault: "none" | "zero-prior" | "evaluate-throws" | "ask-throws";
  constructor(uniform: boolean, fault: Standin["fault"] = "none") {
    this.uniform = uniform;
    this.fault = fault;
  }
  beginGame(): Promise<void> {
    return Promise.resolve();
  }
  ask(_battle: string, query: Query): ReturnType<PolicyClient["ask"]> {
    if (this.fault === "ask-throws") throw new PolicyProtocolError("the stand-in's worker refuses");
    const reply = (choice: string): Awaited<ReturnType<PolicyClient["ask"]>> => ({ schema_version: 1 as const, identity: query.identity, choice, source_policy_id: query.policy_id, elapsed_ms: 0, deadline_outcome: "in_budget" as const }) as never;
    const observation = query.observation as unknown as Observation;
    if (observation.kind === "preview") {
      const n = query.candidates.length;
      const sampling = query.sampling as { kind: string; u?: number };
      const index = this.uniform && sampling.kind === "sample" ? Math.floor(((sampling.u as number) * n) / 2 ** 53) : 0;
      return Promise.resolve(reply((query.candidates[index] as { choice: string }).choice));
    }
    const verdict = bCorrect(observation);
    // The wrong side forfeits at its first Battle request: B when its lead is not the counter, A when it is.
    if (verdict !== null && verdict.b !== verdict.truth) throw new PolicyReplyError("the stand-in's truth: this side's lead is the wrong one");
    const open = query.candidates.filter((c) => !c.pending);
    return Promise.resolve(reply(((open[0] ?? query.candidates[0]) as { choice: string }).choice));
  }
  evaluate(_battle: string, query: Parameters<EvaluationClient["evaluate"]>[1]): ReturnType<EvaluationClient["evaluate"]> {
    this.queries += 1;
    if (this.fault === "evaluate-throws") return Promise.reject(new Error("the stand-in's worker died"));
    const observation = query.observation as unknown as Observation;
    let value_ppm = 500_000;
    let prior: number[];
    if (observation.kind === "preview") prior = query.candidates.map((_c, i) => (this.fault === "zero-prior" ? 0 : this.uniform ? 1000 : Math.floor(1_000_000_000 / (i + 2))));
    else {
      prior = query.candidates.map(() => 1);
      const verdict = bCorrect(observation);
      if (verdict !== null) value_ppm = verdict.b === verdict.truth ? 1_000_000 : 0;
    }
    return Promise.resolve({ identity: query.identity, prior_ppm: prior, value_ppm } as never);
  }
}

const teams = () => createOpponentTeams({ engine, vocabulary: goldenTeamVocabulary(), reader_sha256: readerSha256(), manifest, arm: "generator-matched" });

const options = (client: BestResponseClient, mode: "argmax" | "sample", change: Partial<BestResponseOptions> = {}): BestResponseOptions => ({
  engine,
  config,
  pool,
  pairs: 1,
  evaluated: { generation: 1, generation_manifest_sha256: "a".repeat(64), checkpoints, mode, policy_id: `policy:${"c".repeat(64)}`, client },
  responder: { settings: { worlds: 1, k_own: 16, own_rule: "distinct-leads", leaf: { kind: "first-request" } }, teams: teams(), vocabulary: goldenTeamVocabulary(), preview_budget_ms: 3_600_000 },
  training_teams: new Set(),
  league_seeds: new Set(),
  ...change,
});

const score = (record: BestResponseRecord, arm: "mirror" | "responder" | "blind" | "room"): number[] => {
  const halves: number[] = record.games.map((g) => {
    const o = g[arm];
    return o.kind === "forfeit" ? (o.loser === "focal" ? 0 : 2) : o.winner === "focal" ? 2 : o.winner === "tie" ? 1 : 0;
  });
  return [halves.slice(0, 2).reduce((a, b) => a + b, 0)];
};

describe("the best-response evaluation", () => {
  const fixed: { record?: BestResponseRecord; again?: BestResponseRecord } = {};

  it("a policy that always plays one fixed strategy is countered by the responder and not by the blind control", { timeout: 900_000 }, async () => {
    const client = new Standin(false);
    fixed.record = await playBestResponse(options(client, "argmax"));
    const { record } = fixed;
    expect(readBestResponseRecord(JSON.parse(jcs(record as unknown as Json)))).toEqual(record);
    // Ground truth: the mirror's B plays the fixed strategy (lead 1), whose counter-position (4 for A's lead 1) it is not: it forfeits both games.
    expect(score(record, "mirror")).toEqual([0]);
    // The responder found the counter from the table alone, and the blind control (best against the average of every lead) did not.
    const b = record.games[0]?.b_decision;
    if (b?.kind !== "table") throw new Error(`expected a table, got ${JSON.stringify(b)}`);
    const [responder, blind] = b.choices;
    expect(responder?.arm).toBe("responder");
    expect(responder?.key.startsWith("4")).toBe(true);
    expect(blind?.key.startsWith("1")).toBe(true);
    expect(score(record, "responder")).toEqual([4]);
    expect(score(record, "blind")).toEqual([0]);
    // BRG = 4 - 0, BRG_blind = 0 - 0 and the margin is positive; the room arm, from A's side, wins too (B forfeits against the fixed lead).
    expect(score(record, "responder")[0] as number - (score(record, "blind")[0] as number)).toBeGreaterThan(0);
    expect(score(record, "room")).toEqual([4]);
    expect(record.plan.responder).toMatchObject({ worlds: 1, k_own: 16, own_rule: "distinct-leads", columns: "all-lead-pairs" });
    expect(record.evaluated.mode).toBe("argmax");
    expect(client.queries).toBeGreaterThan(0);
  });

  it("reruns to the same bytes", { timeout: 900_000 }, async () => {
    fixed.again = await playBestResponse(options(new Standin(false), "argmax"));
    expect(jcs(fixed.again as unknown as Json)).toEqual(jcs(fixed.record as unknown as Json));
  });

  it("a uniform policy gives the responder the blind control's weights: the same choice, the same games, a margin of exactly zero", { timeout: 900_000 }, async () => {
    const record = await playBestResponse(options(new Standin(true), "sample"));
    for (const row of record.games) {
      const b = row.b_decision;
      if (b.kind !== "table") throw new Error("expected a table");
      expect(b.choices[0]?.key).toBe(b.choices[1]?.key);
      expect(row.responder).toEqual(row.blind);
    }
    expect(record.result.responder).toEqual(record.result.blind);
  });

  it("plays the evaluated pair's own Preview, and says why, when the responder has no table to choose from", { timeout: 900_000 }, async () => {
    const real = teams();
    const decisionsOf = (record: BestResponseRecord): string[] => record.games.flatMap((g) => [g.b_decision, g.a_decision]).map((d) => (d.kind === "fallback" ? d.code : "table"));
    // A belief that refuses, a draw that refuses and a team the validator refuses as a whole each leave no world to build a table from.
    const refusing: ResponderTeamPort = { ...real, open: () => ({ kind: "refused", code: "species_not_in_prior" }) };
    const drawRefusing: ResponderTeamPort = { ...real, open: () => ({ kind: "source", source: { draw: () => ({ kind: "refused", code: "legality_redraws_exhausted" }) } }) };
    const unstartable: ResponderTeamPort = {
      ...real,
      open: (input) => {
        const opened = real.open(input);
        if (opened.kind !== "source") return opened;
        // One item on all six: a team the Item Clause refuses.
        return { kind: "source", source: { draw: (stream, index) => { const drawn = opened.source.draw(stream, index); return drawn.kind !== "team" ? drawn : { ...drawn, team: { sets: drawn.team.sets.map((set) => ({ ...set, item: "leftovers" })) } }; } } };
      },
    };
    const results: Record<string, string[]> = {};
    for (const [name, port] of Object.entries({ refusing, drawRefusing, unstartable })) {
      const record = await playBestResponse(options(new Standin(false), "argmax", { responder: { ...options(new Standin(false), "argmax").responder, teams: port } }));
      results[name] = decisionsOf(record);
      // Every responder arm played the evaluated pair's own Preview: the mirror's games, outcome for outcome.
      for (const row of record.games) {
        expect(row.responder).toEqual(row.mirror);
        expect(row.blind).toEqual(row.mirror);
      }
    }
    expect(results.refusing).toEqual(Array(4).fill("species_not_in_prior"));
    expect(results.drawRefusing).toEqual(Array(4).fill("no_world"));
    expect(results.unstartable).toEqual(Array(4).fill("no_world"));
    // A network whose Preview prior is zero everywhere gives the sample model no weight on any column.
    const none = await playBestResponse(options(new Standin(false, "zero-prior"), "sample"));
    expect(decisionsOf(none)).toEqual(Array(4).fill("no_complete_column"));
  });

  it("fails the run, and returns no record, when the worker fails inside a decision or refuses a query", { timeout: 900_000 }, async () => {
    await expect(playBestResponse(options(new Standin(false, "evaluate-throws"), "argmax"))).rejects.toThrow(/the responder failed: the stand-in's worker died/);
    await expect(playBestResponse(options(new Standin(false, "ask-throws"), "argmax"))).rejects.toThrow(/policy_protocol/);
  });

  it("plays another dataset's validation split under an evaluation key, and a game both sides survive is a tie", { timeout: 900_000 }, async () => {
    const evalPool = datasetValidationPool(engine, withEvalDataset(config, { path: SECOND, dataset_hash: secondHash }));
    const record = await playBestResponse(options(new Standin(false), "argmax", { pool: evalPool, evaluation_key: "e0p-key" }));
    expect(readBestResponseRecord(JSON.parse(jcs(record as unknown as Json)))).toEqual(record);
    expect(record.teams).toEqual({ dataset_hash: config.team_dataset?.dataset_hash, evaluation_dataset_hash: secondHash, split: "validation", pairs_used: 1 });
    expect(record.plan.evaluation_key).toBe("e0p-key");
    // The stand-in's truth is about the first dataset's pair, so here no side is wrong: every game runs to the turn cap.
    expect(record.result.mirror).toMatchObject({ ties: 2, wins: 0, losses: 0 });
    // Without the key the plan is the run key's: the field is absent.
    const own = await playBestResponse(options(new Standin(false), "argmax", { pool: evalPool }));
    expect(own.plan.evaluation_key).toBeUndefined();
  });

  it("fails the run when a sampled set names something the vocabulary does not know", { timeout: 900_000 }, async () => {
    const real = teams();
    const bad: ResponderTeamPort = {
      ...real,
      open: (input) => {
        const opened = real.open(input);
        if (opened.kind !== "source") return opened;
        return { kind: "source", source: { draw: (stream, index) => { const drawn = opened.source.draw(stream, index); return drawn.kind !== "team" ? drawn : { ...drawn, team: { sets: drawn.team.sets.map((set) => ({ ...set, species: "notaspecies", forme: "notaspecies" })) } }; } } };
      },
    };
    await expect(playBestResponse(options(new Standin(false), "argmax", { responder: { ...options(new Standin(false), "argmax").responder, teams: bad } }))).rejects.toThrow(/the responder failed/);
  });

  it("refuses what it cannot play: bad numbers, a key equal to the run's own, a pair that reads features", async () => {
    const client = new Standin(false);
    await expect(playBestResponse(options(client, "argmax", { pairs: 0 }))).rejects.toThrow(BestResponseError);
    await expect(playBestResponse(options(client, "argmax", { evaluation_key: config.seeds.run_key }))).rejects.toThrow(/run's own key/);
    await expect(playBestResponse(options(client, "argmax", { evaluation_key: "bad key!" }))).rejects.toThrow(/id token/);
    const features = Object.assign(new Standin(false), { featureSet: { feature_set: "probe-damage/1", feature_set_hash: "f".repeat(64) } });
    await expect(playBestResponse(options(features as unknown as BestResponseClient, "argmax"))).rejects.toThrow(/features_pair/);
    const bad = options(client, "argmax");
    await expect(playBestResponse({ ...bad, responder: { ...bad.responder, settings: { ...bad.responder.settings, worlds: 0 } } })).rejects.toThrow(/worlds/);
    await expect(playBestResponse({ ...bad, responder: { ...bad.responder, preview_budget_ms: 0 } })).rejects.toThrow(/budget/);
    await expect(playBestResponse({ ...bad, pairs: 10_000 })).rejects.toThrow(BestResponseError);
  });
});
