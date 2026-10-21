import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { jcs, type Json } from "../src/canonical-json.ts";
import { EpisodeError, readEpisode } from "../src/episode-io.ts";
import { omniscientLogHash, outcomeWinner, progressRecordHash, ProgressRecordError, readProgressEvaluationRecord, readProgressGameRecord } from "../src/progress-records.ts";
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

describe("ProgressGameRecord", () => {
  it.each(["played", "forfeit"])("reads the valid sample %s", (name) => {
    const record = readProgressGameRecord(sample(`${GAME}/valid/${name}`));
    expect(record.data_class).toBe("synthetic");
    expect(record.omniscient_log_sha256).toBe(omniscientLogHash(record.omniscient_log));
  });

  it.each([
    ["chosen-not-candidate", "is not among its candidates"],
    ["log-hash-wrong", "omniscient_log_sha256 does not recompute"],
    ["seed-wrong", "progress-seed/1"],
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
