// The ScriptedHeadToHeadRecord reader (ADR-025 section 13, #1130) against the shared sample corpus.
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { readScriptedH2HRecord, ScriptedH2HRecordError, scriptedH2HRecordHash, sumBeliefCounts, tallyScripted } from "../src/scripted-h2h-record.ts";
import { SAMPLES_DIR } from "../src/samples.ts";

const DIR = join(SAMPLES_DIR, "v1", "scripted-head-to-head-record");
const files = (kind: string): string[] => readdirSync(join(DIR, kind)).sort();
const sample = (kind: string, name: string): unknown => JSON.parse(readFileSync(join(DIR, kind, name), "utf8")) as unknown;

describe("ScriptedHeadToHeadRecord", () => {
  it("reads every valid sample and hashes its JCS bytes", () => {
    expect(files("valid")).toEqual(["actor-rule-against-random-legal.json", "v1-against-v2.json"]);
    for (const name of files("valid")) {
      const record = readScriptedH2HRecord(sample("valid", name));
      expect(record.data_class).toBe("synthetic");
      expect(scriptedH2HRecordHash(record)).toMatch(/^[0-9a-f]{64}$/);
    }
  });

  it("refuses every invalid sample by the schema and every reader-invalid one by a rule", () => {
    for (const name of files("invalid")) expect(() => readScriptedH2HRecord(sample("invalid", name)), name).toThrow(/invalid ScriptedHeadToHeadRecord/);
    for (const name of files("reader-invalid")) {
      const value = sample("reader-invalid", name);
      expect(() => readScriptedH2HRecord(value), name).toThrow(ScriptedH2HRecordError);
      expect(() => readScriptedH2HRecord(value), name).not.toThrow(/invalid ScriptedHeadToHeadRecord/); // the schema passes it
    }
  });

  it("recounts the result from the rows, forfeits as scored losses", () => {
    const record = readScriptedH2HRecord(sample("valid", "v1-against-v2.json"));
    expect(tallyScripted(record.games, record.plan.max_turns)).toEqual(record.result);
    expect(record.result).toMatchObject({ wins_a: 1, wins_b: 3, ties: 0 });
    expect(record.result.forfeits).toEqual([{ side: "a", cause: "latency_overrun", count: 1 }]);
    expect(readScriptedH2HRecord(sample("valid", "actor-rule-against-random-legal.json")).result).toMatchObject({ ties: 1, turn_cap_ties: 1 });
  });

  it("scores a forfeit by arm b as a win for arm a", () => {
    const rows = [
      { outcome: { kind: "forfeit" as const, loser: "b" as const, cause: "crash", request_seq: 1, turns: 3 } },
      { outcome: { kind: "played" as const, winner: "b" as const, turns: 5 } },
    ];
    expect(tallyScripted(rows, 100)).toMatchObject({ wins_a: 1, wins_b: 1, ties: 0, turns_total: 8, forfeits: [{ side: "b", cause: "crash", count: 1 }] });
  });

  it("sums belief counts with refusals sorted by code", () => {
    const record = readScriptedH2HRecord(sample("valid", "v1-against-v2.json"));
    const totals = sumBeliefCounts(record.games.map((g) => g.b_belief as never));
    expect(totals.refusals).toEqual([{ code: "legality_redraws_exhausted", count: 1 }, { code: "reader_flagged", count: 3 }]);
  });
});
