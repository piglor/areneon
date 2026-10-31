// The BestResponseRecord reader (ADR-029 section 2, #1257) against the shared sample corpus.
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { BEST_RESPONSE_LABEL, BestResponseRecordError, bestResponseRecordHash, readBestResponseRecord, tallyArm } from "../src/best-response-record.ts";
import { SAMPLES_DIR } from "../src/samples.ts";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Rec = Record<string, any>;
const DIR = join(SAMPLES_DIR, "v1", "best-response-record");
const files = (kind: string): string[] => readdirSync(join(DIR, kind)).sort();
const sample = (kind: string, name: string): unknown => JSON.parse(readFileSync(join(DIR, kind, name), "utf8")) as unknown;

// A rule each reader-invalid sample breaks, by the words of the reader.
const RULES: Record<string, string> = {
  "a-seat-not-parity": "team A sits on p1 for an even game",
  "choices-wrong-arms": "the choices must be exactly",
  "clamped-not-sum": "clamped must be the sum",
  "columns-used-over-columns": "columns used cannot exceed the table's columns",
  "complete-with-missing": "complete must hold exactly when no entry is missing",
  "evaluation-dataset-is-own": "cannot be the run's own dataset",
  "forfeits-unsorted": "forfeits must be sorted",
  "game-rows-short": "one game row per game",
  "games-odd": "plan.games must be even",
  "pairs-not-half": "half of plan.games",
  "rank-over-own": "a chosen rank must be below",
  "tally-not-rows": "the tally must equal the game rows",
  "worlds-not-plan": "add up to the plan's worlds",
};

describe("BestResponseRecord", () => {
  it("reads every valid sample and hashes its JCS bytes", () => {
    expect(files("valid")).toEqual(["evaluation-dataset-rollout.json", "two-pairs.json"]);
    for (const name of files("valid")) {
      const record = readBestResponseRecord(sample("valid", name));
      expect(record.data_class).toBe("synthetic");
      expect(bestResponseRecordHash(record)).toMatch(/^[0-9a-f]{64}$/);
    }
  });

  it("refuses every invalid sample by the schema and every reader-invalid one by its rule", () => {
    for (const name of files("invalid")) expect(() => readBestResponseRecord(sample("invalid", name)), name).toThrow(/invalid BestResponseRecord/);
    expect(files("reader-invalid").map((f) => f.replace(/\.json$/, ""))).toEqual(Object.keys(RULES));
    for (const name of files("reader-invalid")) {
      const value = sample("reader-invalid", name);
      const rule = RULES[name.replace(/\.json$/, "")] as string;
      expect(() => readBestResponseRecord(value), name).toThrow(BestResponseRecordError);
      expect(() => readBestResponseRecord(value), name).toThrow(rule);
      expect(() => readBestResponseRecord(value), name).not.toThrow(/invalid BestResponseRecord/); // the schema passes it
    }
  });

  it("tallies from the focal side and counts a forfeit as a scored loss for its side", () => {
    const record = readBestResponseRecord(sample("valid", "two-pairs.json"));
    expect(record.result.responder).toMatchObject({ wins: 4, losses: 0, ties: 0 });
    expect(record.result.responder.forfeits).toEqual([{ side: "other", cause: "latency_overrun", count: 1 }]);
    expect(record.result.blind).toMatchObject({ wins: 1, losses: 2, ties: 1 });
    expect(tallyArm(record.games.map((g) => g.mirror), record.plan.max_turns)).toEqual(record.result.mirror);
    expect(tallyArm([{ kind: "forfeit", loser: "focal", cause: "crash", request_seq: 1, turns: 3 }], 100)).toMatchObject({ wins: 0, losses: 1, turns_total: 3, forfeits: [{ side: "focal", cause: "crash", count: 1 }] });
    expect(tallyArm([{ kind: "played", winner: "tie", turns: 101 }], 100)).toMatchObject({ ties: 1, turn_cap_ties: 1 });
  });

  // The counts and tallies the committed samples cannot show: each mutation breaks one rule of a valid sample.
  it.each([
    ["worlds used over asked", (r: Rec) => (r.games[0].b_decision.counts.worlds_used = 5), "worlds used"],
    ["worlds dropped over used", (r: Rec) => (r.games[0].b_decision.counts.worlds_dropped = 5), "worlds used"],
    ["entries over attempts", (r: Rec) => (r.games[0].b_decision.counts.entries = 99), "entries cannot"],
    ["own over k_own", (r: Rec) => (r.games[0].b_decision.own = 5), "exceed k_own"],
    [
      "missing over cells",
      (r: Rec) => {
        r.games[0].b_decision.complete = false;
        r.games[0].b_decision.counts.missing_entries = 13;
      },
      "missing entries cannot",
    ],
    ["wins over games", (r: Rec) => (r.result.mirror.wins = 9), "must equal plan.games"],
    ["game index", (r: Rec) => (r.games[1].game = 5), "game rows must run"],
  ])("refuses %s", (_name, mutate, message) => {
    const record = sample("valid", "two-pairs.json") as Rec;
    mutate(record);
    expect(() => readBestResponseRecord(record)).toThrow(message);
  });

  it("carries the white-box label the displays show", () => {
    expect(BEST_RESPONSE_LABEL).toContain("white-box access to the evaluated policy's weights");
    expect(BEST_RESPONSE_LABEL).toContain("never held-out evidence");
  });
});
