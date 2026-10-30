/* eslint-disable @typescript-eslint/no-explicit-any */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { readTeamVocabulary, readUsageWeightTable, teamVocabularyHash, UsageWeightTableError, usageTableHash } from "../src/index.ts";
import { SAMPLES_DIR } from "../src/samples.ts";
import { exportEngineData } from "../src/engine-data.ts";
import { loadEngine } from "../src/engine.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { sha256Hex } from "../src/rules.ts";
import { jcs, type Json } from "../src/canonical-json.ts";
import { integerRoot, rootedWeight } from "../src/usage-weight-table.ts";

// The corpus's slim vocabulary (two species, two items, three natures) is what the samples were built against.
const committed = (file: string): any => JSON.parse(readFileSync(join(SAMPLES_DIR, file), "utf8"));
const full = exportEngineData(loadEngine(), GOLDEN_FORMAT);
const slim = () => {
  const vocabulary = committed("v1/team-vocabulary/valid/golden.json");
  return readTeamVocabulary(vocabulary, {
    ...full,
    species: full.species.filter((x) => vocabulary.roster.some((r: { id: string }) => r.id === x.id)),
    moves: full.moves.filter((m) => vocabulary.roster.some((r: { moves: string[] }) => r.moves.includes(m.id))),
    items: full.items.slice(0, 2),
    abilities: full.abilities.filter((a) => full.species.some((x) => vocabulary.roster.some((r: { id: string }) => r.id === x.id) && x.abilities.includes(a.id))),
  });
};

describe("readUsageWeightTable (ADR-020 section 4)", () => {
  const vocabulary = slim();
  const valid = (name: string) => committed(`v1/usage-weight-table/valid/${name}.json`);

  it("reads the valid samples and hashes them by their JCS bytes", () => {
    for (const name of ["v1", "root-1", "no-teammates"]) {
      const table = readUsageWeightTable(valid(name), vocabulary);
      expect(usageTableHash(table)).toBe(sha256Hex(jcs(valid(name) as Json)));
    }
    expect(usageTableHash(valid("v1"))).not.toBe(usageTableHash(valid("root-1")));
    expect(vocabulary.format_id).toBe(valid("v1").format_id);
    expect(teamVocabularyHash(vocabulary)).toBe(valid("v1").vocabulary_hash);
  });

  it("does not change the value it is given", () => {
    const value = valid("v1");
    readUsageWeightTable(value, vocabulary);
    expect(value).toEqual(valid("v1"));
  });

  it.each([
    ["unsorted-species", /species: .* repeated or out of order/],
    ["unsorted-sources", /sources: .* repeated or out of order/],
    ["species-sum", /usage shares do not sum/],
    ["species-root", /w_species is not the integer root/],
    ["unknown-species", /is not in the vocabulary roster/],
    ["duplicate-item", /items: .* repeated or out of order/],
    ["unsorted-moves", /moves: .* repeated or out of order/],
    ["unsorted-spreads", /spread 1 is repeated or out of order/],
    ["unsorted-teammates", /teammates: .* repeated or out of order/],
    ["unknown-item", /item aaaaaaaaa is not in the vocabulary/],
    ["unknown-move", /move aaaaaaaaa is not in the roster move pool/],
    ["unknown-ability", /ability aaaaaaaaa is not one the roster lists/],
    ["unknown-nature", /nature aaaaaaaaa is not in the vocabulary/],
    ["unknown-teammate", /teammate aaaaaaaaa is not in the roster/],
    ["abilities-sum", /abilities: the masses do not sum/],
    ["items-sum", /items: the masses do not sum/],
    ["moves-sum", /moves: the masses do not sum/],
    ["spreads-sum", /spreads: the masses do not sum/],
    ["teammates-sum", /teammates: the masses do not sum/],
    ["teammate-root", /teammate weight is not the root/],
    ["spread-over-limit", /more Stat Points than the format's limit/],
    ["too-many-items", /more entries than k_items/],
    ["too-many-moves", /more entries than k_moves/],
    ["too-many-spreads", /more entries than k_spreads/],
    ["too-many-teammates", /more entries than k_teammates/],
    ["stale-vocabulary-hash", /vocabulary_hash is not the hash/],
    ["other-format", /format_id differs/],
    ["other-engine-data", /engine_data_hash differs/],
  ])("refuses the reader-invalid sample %s", (name, message) => {
    const value = committed(`v1/usage-weight-table/reader-invalid/${name}.json`);
    expect(() => readUsageWeightTable(value, vocabulary)).toThrow(UsageWeightTableError);
    expect(() => readUsageWeightTable(value, vocabulary)).toThrow(message);
  });

  it("orders spreads of one nature by their Stat Points and refuses a repeat", () => {
    const table = valid("v1");
    const [a, b] = table.species[0].spreads.entries;
    b.nature = a.nature;
    expect(() => readUsageWeightTable(table, vocabulary)).not.toThrow();
    table.species[0].spreads.entries = [b, a];
    expect(() => readUsageWeightTable(table, vocabulary)).toThrow(/spread 1 is repeated or out of order/);
    table.species[0].spreads.entries = [a, { ...a }];
    expect(() => readUsageWeightTable(table, vocabulary)).toThrow(/spread 1 is repeated or out of order/);
  });

  it("refuses every schema-invalid sample and an evs, paste or team-shaped intruder", () => {
    for (const name of ["extra-field", "fractional-ppm", "zero-ppm", "ppm-over-total", "missing-truncated", "missing-no-item", "missing-empty", "abilities-truncated", "missing-spread-point", "missing-teammate-weight", "bad-month", "no-sources", "other-basis", "no-attribution", "no-species", "root-zero", "next-version"]) {
      expect(() => readUsageWeightTable(committed(`v1/usage-weight-table/invalid/${name}.json`), vocabulary), name).toThrow(UsageWeightTableError);
    }
    const intruders: [string, (t: any) => void][] = [
      ["evs", (t) => (t.species[0].spreads.entries[0].evs = {})],
      ["paste", (t) => (t.paste = "Pikachu @ Light Ball")],
      ["team", (t) => (t.species[0].sets = [{ species: "x", moves: [], evs: {} }])],
    ];
    for (const [what, change] of intruders) {
      const table = valid("v1");
      change(table);
      expect(() => readUsageWeightTable(table, vocabulary), what).toThrow(/invalid usage weight table/);
    }
  });
});

describe("integer roots", () => {
  it("are exact", () => {
    expect(integerRoot(15n, 2n)).toBe(3n);
    expect(integerRoot(16n, 2n)).toBe(4n);
    expect(integerRoot(10n ** 12n - 1n, 2n)).toBe(999_999n);
    expect(integerRoot(7n * 10n ** 18n, 3n)).toBe(1_912_931n);
    expect(integerRoot(123456789n, 1n)).toBe(123456789n);
    expect(integerRoot(0n, 2n)).toBe(0n);
    expect(rootedWeight(600_000, 2)).toBe(774_596);
    expect(rootedWeight(1_000_000, 3)).toBe(1_000_000);
    expect(rootedWeight(1, 2)).toBe(1000);
  });

  it("refuse a negative value and a zero root", () => {
    expect(() => integerRoot(-1n, 2n)).toThrow(UsageWeightTableError);
    expect(() => integerRoot(4n, 0n)).toThrow(UsageWeightTableError);
  });
});
