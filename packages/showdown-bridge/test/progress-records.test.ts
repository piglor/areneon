import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { jcs, type Json } from "../src/canonical-json.ts";
import { EpisodeError, readEpisode } from "../src/episode-io.ts";
import { omniscientLogHash, outcomeWinner, progressRecordHash, ProgressRecordError, readProgressEvaluationRecord,
  readProgressEvaluationRecordV2, readProgressGameRecord } from "../src/progress-records.ts";
import { SAMPLES_DIR } from "../src/samples.ts";
import { loadTrainingSet, TrainingLoaderError } from "../src/training-loader.ts";

const sample = (name: string): Record<string, any> => JSON.parse(readFileSync(join(SAMPLES_DIR, `v1/${name}.json`), "utf8")) as Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any
const EVALUATION = "progress-evaluation-record";
const GAME = "progress-game-record";

describe("ProgressEvaluationRecord", () => {
  it.each(["baseline-match", "previous-match"])("reads the valid sample %s and hashes its JCS bytes", (name) => {
    const record = readProgressEvaluationRecord(sample(`${EVALUATION}/valid/${name}`));
    expect(record.data_class).toBe("synthetic");
    expect(progressRecordHash(record)).toMatch(/^[0-9a-f]{64}$/);
    expect(JSON.parse(jcs(record as unknown as Json))).toEqual(record);
  });

  it.each([
    ["counts-disagree", "wins + ties + losses must equal games"],
    ["forfeits-disagree", "forfeits must equal the forfeited game rows"],
    ["hash-count-wrong", "game_record_hashes must number recorded_games_per_match"],
    ["hashes-unsorted", "game_record_hashes must be sorted, each once"],
    ["pairs-used-wrong", "teams.pairs_used must equal plan.games_per_match"],
    ["rows-incomplete", "there must be one game row per game"],
    ["seat-wrong", "the candidate sits on p1 for an even game"],
    ["turn-cap-ties-wrong", "turn_cap_ties must equal"],
    ["unsorted-matches", "matches must be sorted by match_id, each once"],
  ])("refuses the reader-invalid sample %s", (name, message) => {
    expect(() => readProgressEvaluationRecord(sample(`${EVALUATION}/reader-invalid/${name}`))).toThrow(ProgressRecordError);
    expect(() => readProgressEvaluationRecord(sample(`${EVALUATION}/reader-invalid/${name}`))).toThrow(message);
  });

  it.each(["public-class", "extra-field", "fractional-wins", "next-seed-derivation", "test-split", "unknown-outcome", "no-manifest-hash"])("refuses the schema-invalid sample %s", (name) => {
    expect(() => readProgressEvaluationRecord(sample(`${EVALUATION}/invalid/${name}`))).toThrow(/invalid ProgressEvaluationRecord/);
  });

  it("refuses a private class, which no committed sample may carry", () => {
    const raw = sample(`${EVALUATION}/valid/baseline-match`);
    raw.data_class = "private";
    expect(() => readProgressEvaluationRecord(raw)).toThrow(/invalid ProgressEvaluationRecord/);
    const game = sample(`${GAME}/valid/played`);
    game.data_class = "private";
    expect(() => readProgressGameRecord(game)).toThrow(/invalid ProgressGameRecord/);
  });

  it("refuses the rules the samples do not reach", () => {
    const edit = (change: (r: Record<string, any>) => void): unknown => { // eslint-disable-line @typescript-eslint/no-explicit-any
      const raw = sample(`${EVALUATION}/valid/baseline-match`);
      change(raw);
      return raw;
    };
    expect(() => readProgressEvaluationRecord(edit((r) => (r.plan.recorded_games_per_match = 3)))).toThrow(/must not exceed/);
    expect(() => readProgressEvaluationRecord(edit((r) => (r.games[0].match_id = "zzz")))).toThrow(/unknown match/);
    expect(() => readProgressEvaluationRecord(edit((r) => (r.matches[0].games = 3)))).toThrow(/games must equal plan.games_per_match/);
    expect(() => readProgressEvaluationRecord(edit((r) => (r.matches[0].turns_total = 1)))).toThrow(/turns_total/);
    expect(() => readProgressEvaluationRecord(edit((r) => (r.games[1].pair = 0)))).toThrow(/pair equal to the game/);
    expect(() =>
      readProgressEvaluationRecord(
        edit((r) =>
          (r.matches[0].forfeits = [
            { side: "candidate", cause: "x", count: 1 },
            { side: "candidate", cause: "x", count: 1 },
          ]),
        ),
      ),
    ).toThrow(/forfeits must be sorted/);
  });

  it("counts a forfeit by the opponent as a win, and a tie past max_turns as a turn-cap tie", () => {
    const raw = sample(`${EVALUATION}/valid/baseline-match`);
    const match = raw.matches[0];
    Object.assign(match, { wins: 1, ties: 1, losses: 0, turn_cap_ties: 1, turns_total: 127, forfeits: [{ side: "opponent", cause: "crash", count: 1 }] });
    raw.games[0].outcome = { kind: "played", winner: "tie", turns: 120 };
    raw.games[1].outcome = { kind: "forfeit", loser: "opponent", cause: "crash", request_seq: 2, turns: 7 };
    expect(readProgressEvaluationRecord(raw).matches[0]?.turn_cap_ties).toBe(1);
  });
});

describe("ProgressEvaluationRecordV2 (both seats, ADR-017 erratum E2)", () => {
  const V2 = "progress-evaluation-record-2";
  it.each(["baseline-match", "previous-match", "anchor-match", "ablation-match"])("reads the valid sample %s and hashes its JCS bytes", (name) => {
    const record = readProgressEvaluationRecordV2(sample(`${V2}/valid/${name}`));
    expect(record.schema_version).toBe(2);
    expect(progressRecordHash(record)).toMatch(/^[0-9a-f]{64}$/);
    expect(JSON.parse(jcs(record as unknown as Json))).toEqual(record);
  });

  it.each([
    ["counts-disagree", "wins + ties + losses must equal games"],
    ["pair-is-game", "pair equal to the game divided by two"],
    ["seat-wrong", "the candidate sits on p1 for an even game"],
    ["odd-games", "must be even"],
    ["pairs-used-wrong", "teams.pairs_used must be half the largest"],
    ["rows-incomplete", "there must be one game row per game"],
    ["anchor-games-missing", "present exactly when the record has anchor matches"],
    ["anchor-plan-without-match", "present exactly when the record has anchor matches"],
    ["anchor-games-odd", "must be even"],
    ["anchor-not-earlier", "plays an earlier generation"],
    ["anchor-misnamed", "is named anchor-g<generation>-<mode>"],
    ["ablation-vs-anchor", "an ablation match plays a scripted baseline"],
    ["ablation-misnamed", "named preview-<variant>-<baseline>-<mode>"],
    ["ablation-not-earlier", "comes from an earlier generation"],
  ])("refuses the reader-invalid sample %s", (name, message) => {
    expect(() => readProgressEvaluationRecordV2(sample(`${V2}/reader-invalid/${name}`))).toThrow(ProgressRecordError);
    expect(() => readProgressEvaluationRecordV2(sample(`${V2}/reader-invalid/${name}`))).toThrow(message);
  });

  it.each(["public-class", "schema-version-1", "fractional-wins", "anchor-no-generation", "ablation-no-checkpoint"])("refuses the schema-invalid sample %s", (name) => {
    expect(() => readProgressEvaluationRecordV2(sample(`${V2}/invalid/${name}`))).toThrow(/invalid ProgressEvaluationRecordV2/);
  });

  it("plays an ablation in argmax only and never beside ordinary matches", () => {
    const sampleMode = sample(`${V2}/valid/ablation-match`);
    sampleMode.matches[0].mode = "sample";
    expect(() => readProgressEvaluationRecordV2(sampleMode)).toThrow(/played in argmax/);
    const mixed = sample(`${V2}/valid/ablation-match`);
    const ordinary = sample(`${V2}/valid/baseline-match`);
    mixed.matches.unshift(ordinary.matches[0]);
    mixed.games.unshift(...ordinary.games);
    expect(() => readProgressEvaluationRecordV2(mixed)).toThrow(/ablation matches only/);
  });

  it("is a different version from v1: each reader refuses the other's record", () => {
    expect(() => readProgressEvaluationRecord(sample(`${V2}/valid/baseline-match`))).toThrow(/invalid ProgressEvaluationRecord/);
    expect(() => readProgressEvaluationRecordV2(sample(`${EVALUATION}/valid/baseline-match`))).toThrow(/invalid ProgressEvaluationRecordV2/);
  });

  it("refuses unknown matches, unordered rows and a recorded count above the games", () => {
    const stray = sample(`${V2}/valid/baseline-match`);
    stray.games[0].match_id = "zzz";
    expect(() => readProgressEvaluationRecordV2(stray)).toThrow(/unknown match/);
    const unordered = sample(`${V2}/valid/anchor-match`);
    unordered.matches.push({ ...unordered.matches[0], match_id: "anchor-g0-argmax", mode: "argmax" });
    unordered.games.push(...unordered.games.map((g: Record<string, unknown>) => ({ ...g, match_id: "anchor-g0-argmax" })));
    unordered.games.reverse();
    expect(() => readProgressEvaluationRecordV2(unordered)).toThrow();
    const recorded = sample(`${V2}/valid/baseline-match`);
    recorded.plan.recorded_games_per_match = 5;
    expect(() => readProgressEvaluationRecordV2(recorded)).toThrow(/must not exceed/);
    const sorted = sample(`${V2}/valid/baseline-match`);
    sorted.matches.push({ ...sorted.matches[0], match_id: "a-first" });
    expect(() => readProgressEvaluationRecordV2(sorted)).toThrow(/sorted by match_id/);
    const hashes = sample(`${V2}/valid/baseline-match`);
    hashes.matches[0].game_record_hashes = [];
    expect(() => readProgressEvaluationRecordV2(hashes)).toThrow(/must number recorded_games_per_match/);
    const unsorted = sample(`${V2}/valid/baseline-match`);
    unsorted.plan.recorded_games_per_match = 2;
    unsorted.matches[0].game_record_hashes = ["f".repeat(64), "1".repeat(64)];
    expect(() => readProgressEvaluationRecordV2(unsorted)).toThrow(/sorted, each once/);
  });
});

describe("ProgressGameRecord", () => {
  it.each(["played", "forfeit", "played-pair"])("reads the valid sample %s", (name) => {
    const record = readProgressGameRecord(sample(`${GAME}/valid/${name}`));
    expect(record.data_class).toBe("synthetic");
    expect(record.omniscient_log_sha256).toBe(omniscientLogHash(record.omniscient_log));
  });

  it.each([
    ["chosen-not-candidate", "is not among its candidates"],
    ["log-hash-wrong", "omniscient_log_sha256 does not recompute"],
    ["seed-wrong", "progress-seed/1"],
    ["pair-seed-wrong", "progress-seed/1"],
    ["pair-wrong", "pair must be the game index divided by two"],
    ["team-hash-wrong", "team hash of p1"],
  ])("refuses the reader-invalid sample %s", (name, message) => {
    expect(() => readProgressGameRecord(sample(`${GAME}/reader-invalid/${name}`))).toThrow(message);
  });

  it.each(["public-class", "extra-field", "bad-seed-form", "unknown-outcome", "no-log-hash"])("refuses the schema-invalid sample %s", (name) => {
    expect(() => readProgressGameRecord(sample(`${GAME}/invalid/${name}`))).toThrow(/invalid ProgressGameRecord/);
  });

  it("gives the engine's winner of an outcome by seat", () => {
    expect(outcomeWinner({ kind: "played", winner: "p2", turns: 1 })).toBe("p2");
    expect(outcomeWinner({ kind: "played", winner: "tie", turns: 1 })).toBeNull();
    expect(outcomeWinner({ kind: "forfeit", loser: "p1", cause: "crash", request_seq: 1, turns: 1 })).toBe("p2");
    expect(outcomeWinner({ kind: "forfeit", loser: "p2", cause: "crash", request_seq: 1, turns: 1 })).toBe("p1");
  });

  it("is not an Episode: the Episode reader and the training loader refuse it", async () => {
    const game = readProgressGameRecord(sample(`${GAME}/valid/played`));
    await expect(readEpisode(game, { known_synthetic: new Set() })).rejects.toThrow(EpisodeError);
    await expect(loadTrainingSet({ episodes: [game as never], evaluation_manifests: [], known_synthetic: new Set(), behaviour_policy_id: "policy:x" })).rejects.toThrow(TrainingLoaderError);
  });
});

describe("the damage-aware baseline's counts (ADR-018 sections 4 and 5)", () => {
  const V2 = "progress-evaluation-record-2";
  it("reads the valid samples and hashes them", () => {
    const record = readProgressEvaluationRecordV2(sample(`${V2}/valid/damage-aware-match`));
    expect(record.matches[0]?.baseline?.identity_hash).toMatch(/^[0-9a-f]{64}$/);
    expect(readProgressGameRecord(sample(`${GAME}/valid/damage-aware`)).baseline?.seat).toBe("p2");
  });

  it.each([
    ["baseline-counts-disagree", "baseline.counts must be the sum"],
    ["baseline-row-missing", "carries baseline counts in its summary and in every game row"],
    ["baseline-on-other-match", "only a damage-aware match carries baseline counts"],
    ["baseline-probe-runs-exceed-queries", "probe runs, unsupported queries and probe refusals cannot exceed queries"],
    ["baseline-identities-differ", "one baseline identity hash"],
  ])("refuses the evaluation sample %s", (name, message) => {
    expect(() => readProgressEvaluationRecordV2(sample(`${V2}/reader-invalid/${name}`))).toThrow(message);
  });

  it.each([
    ["baseline-seat-wrong", "baseline.seat must be the one seat"],
    ["baseline-missing", "must carry the baseline counts"],
    ["baseline-decisions-differ", "baseline.counts.decisions must be the baseline seat's decisions"],
  ])("refuses the game sample %s", (name, message) => {
    expect(() => readProgressGameRecord(sample(`${GAME}/reader-invalid/${name}`))).toThrow(message);
  });

  it("refuses preview and scored decisions above the decisions, and unsupported queries above the queries", () => {
    const game = sample(`${GAME}/valid/damage-aware`);
    game.baseline.counts.preview_decisions = game.baseline.counts.decisions;
    expect(() => readProgressGameRecord(game)).toThrow("preview and scored decisions cannot exceed decisions");
    const other = sample(`${GAME}/valid/damage-aware`);
    other.baseline.counts.unsupported_queries = other.baseline.counts.queries + 1;
    expect(() => readProgressGameRecord(other)).toThrow("unsupported queries and probe refusals cannot exceed queries");
  });

  it("allows one unanswered request when the game ended in a forfeit", () => {
    const game = sample(`${GAME}/valid/damage-aware`);
    game.outcome = { kind: "forfeit", loser: "p1", cause: "crash", request_seq: 9, turns: 3 };
    game.baseline.counts.decisions += 1;
    expect(() => readProgressGameRecord(game)).not.toThrow();
  });
});

describe("probe_refusals of the damage-aware counts (optional, written only above 0)", () => {
  it("is summed from the rows into the match, and may not exceed the queries", () => {
    const record = sample("progress-evaluation-record-2/valid/damage-aware-match");
    for (const row of record.games) row.baseline_counts.probe_refusals = 1;
    record.matches[0].baseline.counts.probe_refusals = record.games.length;
    expect(() => readProgressEvaluationRecordV2(record)).not.toThrow();
    record.matches[0].baseline.counts.probe_refusals += 1;
    expect(() => readProgressEvaluationRecordV2(record)).toThrow("baseline.counts must be the sum");
    const over = sample("progress-evaluation-record-2/valid/damage-aware-match");
    over.games[0].baseline_counts.probe_refusals = over.games[0].baseline_counts.queries + 1;
    over.matches[0].baseline.counts.probe_refusals = over.games[0].baseline_counts.probe_refusals;
    expect(() => readProgressEvaluationRecordV2(over)).toThrow("probe refusals cannot exceed queries");
  });
});
