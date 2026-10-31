// The opponent_sheet reader rules, one rule at a time (ADR-027 section 2). The corpus samples hold the rules the TS and Python
// readers share (`samples.test.ts`, `tests/contracts/test_opponent_sheet.py`); this table holds the rest, on copies of the committed samples.
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { SAMPLES_DIR } from "../src/samples.ts";
import { opponentSheetProblem, readOpponentSheet } from "../src/index.ts";

type Json = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any
const sample = (name: string): Json => JSON.parse(readFileSync(`${SAMPLES_DIR}/v1/opponent-sheet/valid/${name}.json`, "utf8")) as Json;
const lead = (b: Json): Json => b.slots[0];

describe("every rule has its message", () => {
  it.each<[string, string, (b: Json) => void, string]>([
    ["belief", "an unknown field with an other_ppm", (b) => (lead(b).nature = { source: "unknown", entries: [], other_ppm: 5 }), "slot 0 nature: a field of source unknown has other_ppm 0"],
    ["belief", "an unknown move field with slots_known", (b) => (lead(b).moves = { source: "unknown", entries: [], slots_known: 2 }), "slot 0 moves: a field of source unknown has slots_known 0"],
    ["belief", "a predicted field with no entries", (b) => (lead(b).item = { source: "predicted", entries: [], other_ppm: 0 }), "slot 0 item: a field with a source other than unknown has entries"],
    ["belief", "entries over a million", (b) => (lead(b).ability = { source: "predicted", entries: [{ id: "a1", ppm: 600000 }, { id: "a2", ppm: 600000 }], other_ppm: 0 }), "slot 0 ability: the entries sum to more than 1000000 plus one per entry"],
    ["belief", "a null item id sorts last among equals", (b) => (lead(b).item = { source: "predicted", entries: [{ id: null, ppm: 500000 }, { id: "ix", ppm: 500000 }], other_ppm: 0 }), "slot 0 item: the entries must be sorted by ppm descending then id, and unique"],
    ["belief", "a revealed move field with three moves", (b) => (lead(b).moves = { source: "revealed", entries: ["a", "b", "c"].map((id) => ({ id, inclusion_ppm: 1000000 })), slots_known: 3 }), "slot 0 moves: a revealed move field has all four moves known"],
    ["belief", "unsorted moves", (b) => (lead(b).moves.entries = lead(b).moves.entries.slice().reverse()), "slot 0 moves: the entries must be sorted by ppm descending then id, and unique"],
    ["engine-sheet", "an engine-sheet brought_ppm that is revealed", (b) => ((lead(b).brought_source = "revealed"), (lead(b).brought_ppm = 1000000)), "slot 0: an engine-sheet block takes brought_ppm from the belief or leaves it unknown"],
    ["belief", "revealed with a ppm below a million", (b) => ((lead(b).brought_source = "revealed"), (lead(b).brought_ppm = 999999)), "slot 0: brought_ppm does not fit brought_source"],
  ])("%s block, %s", (name, _what, change, message) => {
    const block = sample(name);
    expect(opponentSheetProblem(block as never)).toBeNull();
    change(block);
    expect(opponentSheetProblem(block as never)).toBe(message);
  });

  it("accepts a sum a few ppm above a million (each entry is rounded alone)", () => {
    const block = sample("belief");
    lead(block).ability = { source: "predicted", entries: [{ id: "a1", ppm: 500001 }, { id: "a2", ppm: 500001 }], other_ppm: 0 };
    expect(opponentSheetProblem(block as never)).toBeNull();
  });

  it("readOpponentSheet returns a copy, not the caller's object", () => {
    const block = sample("belief");
    const read = readOpponentSheet(block);
    expect(read).toEqual(block);
    expect(read).not.toBe(block);
  });
});

describe("old values keep their bytes (ADR-008 section 3)", () => {
  // The digests of every PolicyQuery and TrainingExample sample that existed before the field, as hashes.json had them on main.
  const BEFORE: Record<string, string> = {
    "v1/training-example/valid/preview.json": "0c5b56b0e3a3631c11a38ac959a2c01fd77ebe0fb3bdd1c8959d039a45e0a69d",
    "v1/training-example/valid/battle.json": "7b9fc909b6c6d93d335996bdd7140481bf2304faea71cde30221b6718869617c",
    "v1/training-example/valid/legacy-battle.json": "32a685b70a5be187632f90b4a22af962c5783db5572690305a03e97df932a492",
    "v1/policy-query/valid/battle.json": "d2d5d7a8fdb7d120436aca8d03aebcf3046f8e1fb62268d3e4683cda18e475bd",
    "v1/policy-query/valid/preview.json": "06f21da1a2142d219bd7b2f14e9c4d9b3c375e9aecbc68efc161bdb3f068d69a",
    "v1/policy-query/valid/battle-a2.json": "3ca12813e9d0f6ed7042f3f815dadea185ddc3b059d2f72af9543d509765b9ae",
    "v1/policy-query/valid/battle-tempered.json": "d5ba66fc8876452f54edc680a60874add18b3ef4b10316a9a48bf72120ecb8b5",
    "v1/policy-query/valid/battle-threshold.json": "f7641c5ffc71546d25128fcb4f03742626110b57542eb66f30eff477b5d53585",
    "v1/policy-query/valid/battle-features.json": "bd29b19a6cf4c14a66b9b87f3db5535575a8d54bbaf99013589c06b5d9021852",
    "v1/policy-query/valid/preview-features.json": "2595470c768f98d96e03ba0ffd25208396aa89c2389df133d5db5a582f45df3d",
    "v1/policy-query/valid/battle-features-2.json": "6854281ede4c83a6765994ef128c179597a23e5728e6df790a43e25d660e7303",
    "v1/policy-query/valid/preview-features-2.json": "4d73a648c7a103f97ee18e50bda69c2e4f76741e58613e072c76510bf81762fe",
    "v1/training-example/valid/battle-features-2.json": "aa6b199540ea6e280d6a8585d0d4afc098768d89a731d18a6ea4933f8158af8b",
    "v1/training-example/valid/battle-features.json": "946eaaa30e084ce40d26746745f076e10f23516f0666cef7fe0f2a0b1a63ced0",
    "v1/policy-query/valid/battle-evaluate.json": "574861c4ca6794f01199cd7b7e344e7a08b4f868226e2785a2bf838fe1916ad5",
    "v1/policy-query/valid/preview-evaluate.json": "4f8aad1c58c03b8afe8a96ba1de3abdd6513e18ea625319fb08604f73d40d8c1",
    "v1/policy-query/valid/battle-features-evaluate.json": "b2f1c3b4831fb1c429af3bf86d88a780c6f94981c7098b793d919ef3582f6a12",
  };
  it("pins the digests of the samples that predate opponent_sheet, and none of them carries it", () => {
    const hashes = JSON.parse(readFileSync(`${SAMPLES_DIR}/hashes.json`, "utf8")) as Record<string, string>;
    for (const [file, digest] of Object.entries(BEFORE)) {
      expect(hashes[file], file).toBe(digest);
      expect("opponent_sheet" in (JSON.parse(readFileSync(`${SAMPLES_DIR}/${file}`, "utf8")) as object), file).toBe(false);
    }
  });
});
