/* eslint-disable @typescript-eslint/no-explicit-any */
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { checkLadderRecord, LADDER_REFERENCE_ANCHOR, LADDER_REFERENCE_STRENGTH, LadderRecordError, ladderRecordHash, readLadderRecord, readTrainingRunConfig, runConfigHash } from "../src/index.ts";
import { SAMPLES_DIR } from "../src/samples.ts";

const dir = (kind: string): string => join(SAMPLES_DIR, `v1/ladder-record/${kind}`);
const load = (kind: string, name: string): Record<string, any> => JSON.parse(readFileSync(join(dir(kind), `${name}.json`), "utf8")) as Record<string, any>;
const names = (kind: string): string[] => readdirSync(dir(kind)).map((f) => f.slice(0, -5)).sort();

// The first refusal each reader-invalid sample draws: the same words in the Python reader (tests/ml/loop/test_ladder.py).
const MESSAGES: Record<string, string> = {
  "anchors-unsorted": "anchors must be sorted by anchor_id, each once",
  "cell-pairs-not-sum": "a cell's pairs must equal the sum of its scores",
  "cell-plays-itself": "a member does not play itself in the table",
  "cell-score-sum-off-step": "score_sum_ppm must be 250000 times",
  "cell-score-sum-wrong": "score_sum_ppm must be 250000 times",
  "cell-unknown-member": "a cell names a member the record does not have",
  "cell-unknown-opponent": "a cell's opponent is not in the record",
  "cells-unsorted": "cells must be sorted by member, opponent kind and opponent id, each once",
  "copy-after-health": "at most one copy",
  "copy-anchor-minimum-wrong": "a copy's anchor minima must be the least anchor pair scores",
  "copy-donor-anchor-below-loser": "a copy's donor cannot have a lower anchor minimum than its loser",
  "copy-donor-weaker": "a copy's donor cannot be weaker than its loser",
  "copy-loser-is-donor": "a copy's loser must differ from its donor",
  "copy-lower-bound-too-low": "a copy needs a lower bound above 500000",
  "copy-pairs-not-cell": "a copy's pairs must equal the donor-against-loser cell's pairs",
  "copy-roots-after-wrong": "a copy's roots must be counted from the members' roots",
  "copy-roots-before-wrong": "a copy's roots must be counted from the members' roots",
  "copy-unknown-member": "a copy names a member the record does not have",
  "copy-without-anchor-cells": "a copy needs anchor cells for its donor and its loser",
  "cyclic-more-than-triples": "cyclic_triples cannot exceed the number of member triples",
  "duplicate-cell": "cells must be sorted by member, opponent kind and opponent id, each once",
  "first-selection-health": "the first selection must be the copy test's outcome",
  "health-duplicate": "health replacements must be sorted by member, each once",
  "health-names-loser": "a health replacement cannot name a copy's loser or donor",
  "health-unknown-member": "a health replacement names a member the record does not have",
  "health-unsorted": "health replacements must be sorted by member, each once",
  "lower-above-upper": "a strength's lower bound cannot exceed its upper bound",
  "members-unsorted": "members must be sorted by member, each once",
  "reference-anchor-missing": "the reference anchor must be present at exactly 1000000000",
  "reference-anchor-moved": "the reference anchor must be present at exactly 1000000000",
  "shares-do-not-sum": "a member's opponent shares must add up to 1000000",
  "snapshot-id-mismatch": "a snapshot's id must be its member and generation",
  "snapshot-lower-above-upper": "a strength's lower bound cannot exceed its upper bound",
  "snapshot-unknown-member": "a snapshot names a member the record does not have",
  "snapshots-unsorted": "snapshots must be sorted by snapshot_id, each once",
  "two-copies": "at most one copy",
  "window-end-not-generation": "the window must end at the record's generation",
  "window-start-wrong": "the window must start size generations before its end",
};

describe("the LadderRecord reader (ADR-021 §4, §6, §8)", () => {
  it("reads every valid sample and hands back a copy", () => {
    for (const name of names("valid")) {
      const raw = load("valid", name);
      const record = readLadderRecord(raw);
      expect(record).toEqual(raw);
      expect(record).not.toBe(raw);
    }
  });

  it.each(names("invalid"))("refuses the schema-invalid sample %s", (name) => {
    expect(() => readLadderRecord(load("invalid", name))).toThrow(LadderRecordError);
    expect(() => readLadderRecord(load("invalid", name))).toThrow("invalid LadderRecord");
  });

  it("holds exactly one refusal message per reader-invalid sample", () => {
    expect(names("reader-invalid")).toEqual(Object.keys(MESSAGES).sort());
  });

  it.each(Object.entries(MESSAGES))("refuses the reader-invalid sample %s", (name, message) => {
    expect(() => readLadderRecord(load("reader-invalid", name))).toThrow(LadderRecordError);
    expect(() => readLadderRecord(load("reader-invalid", name))).toThrow(message);
  });

  it("names the reference anchor and its value", () => {
    expect([LADDER_REFERENCE_ANCHOR, LADDER_REFERENCE_STRENGTH]).toEqual(["actor-rule@v1", 1_000_000_000]);
  });

  it("hashes the JCS bytes: key order does not matter, a value does", async () => {
    const record = readLadderRecord(load("valid", "none"));
    const reordered = readLadderRecord(Object.fromEntries(Object.entries(load("valid", "none")).reverse()));
    expect(await ladderRecordHash(record)).toMatch(/^[0-9a-f]{64}$/);
    expect(await ladderRecordHash(reordered)).toBe(await ladderRecordHash(record));
    expect(await ladderRecordHash({ ...record, cyclic_triples: 1 })).not.toBe(await ladderRecordHash(record));
  });

  describe("against the previous record", () => {
    const previous = () => {
      const p = load("valid", "none");
      p.generation = 4;
      p.window = { size: 3, first_generation: 2, last_generation: 4 };
      return readLadderRecord(p);
    };

    it("accepts the generation after, with the same anchors and snapshot strengths", () => {
      expect(readLadderRecord(load("valid", "none"), previous()).generation).toBe(5);
    });

    it("refuses another run, another config and a generation that is not the next", () => {
      const other = (changes: Partial<ReturnType<typeof previous>>): ReturnType<typeof previous> => ({ ...previous(), ...changes });
      expect(() => readLadderRecord(load("valid", "none"), other({ run_id: "other-run" }))).toThrow("another run or config");
      expect(() => readLadderRecord(load("valid", "none"), other({ run_config_hash: "b".repeat(64) }))).toThrow("another run or config");
      expect(() => readLadderRecord(load("valid", "none"), other({ generation: 3 }))).toThrow("the generation before");
    });

    it("refuses moved anchors and a moved frozen snapshot", () => {
      const moved = structuredClone(previous());
      (moved.anchors[1] as { strength: number }).strength += 1;
      expect(() => readLadderRecord(load("valid", "none"), moved)).toThrow("anchor strengths are frozen");
      const snap = structuredClone(previous());
      (snap.snapshots[0] as { strength: { strength: number } }).strength.strength += 1;
      expect(() => readLadderRecord(load("valid", "none"), snap)).toThrow("frozen at entry");
    });

    it("lets a snapshot that left the pool go and a new one arrive", () => {
      const record = load("valid", "none");
      const earlier = structuredClone(previous());
      earlier.snapshots = [];
      expect(readLadderRecord(record, earlier).snapshots).toHaveLength(1);
    });
  });

  it("ties a record to its config by run key and hash", async () => {
    const config = readTrainingRunConfig(JSON.parse(readFileSync(join(SAMPLES_DIR, "v1/training-run-config/valid/tiny.json"), "utf8")));
    const raw = load("valid", "none");
    raw.run_id = config.seeds.run_key;
    raw.run_config_hash = await runConfigHash(config);
    await expect(checkLadderRecord(readLadderRecord(raw), config)).resolves.toBeUndefined();
    await expect(checkLadderRecord(readLadderRecord({ ...raw, run_id: "other-run" }), config)).rejects.toThrow("run key");
    await expect(checkLadderRecord(readLadderRecord({ ...raw, run_config_hash: "c".repeat(64) }), config)).rejects.toThrow("config's hash");
  });
});
