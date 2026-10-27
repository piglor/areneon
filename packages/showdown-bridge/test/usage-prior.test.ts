/* eslint-disable @typescript-eslint/no-explicit-any */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { BeliefConfigError, beliefConfigHash, readBeliefConfig, readTeamVocabulary, readUsagePrior, teamVocabularyHash, UsagePriorError, usagePriorHash } from "../src/index.ts";
import { SAMPLES_DIR } from "../src/samples.ts";
import { exportEngineData } from "../src/engine-data.ts";
import { loadEngine } from "../src/engine.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { sha256Hex } from "../src/rules.ts";
import { jcs, type Json } from "../src/canonical-json.ts";

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

describe("readUsagePrior (ADR-025 section 2)", () => {
  const vocabulary = slim();
  const valid = (name: string) => committed(`v1/usage-prior/valid/${name}.json`);
  const forme = (prior: any) => prior.species[0].formes[0];

  it("reads the three source kinds and hashes them by their JCS bytes", () => {
    for (const name of ["smogon-chaos", "role-sets", "generator-matched"]) {
      const prior = readUsagePrior(valid(name), vocabulary);
      expect(prior.source.kind).toBe(name);
      expect(usagePriorHash(prior)).toBe(sha256Hex(jcs(valid(name) as Json)));
    }
    expect(usagePriorHash(valid("smogon-chaos"))).not.toBe(usagePriorHash(valid("role-sets")));
  });

  it("does not change the value it is given", () => {
    const value = valid("smogon-chaos");
    readUsagePrior(value, vocabulary);
    expect(value).toEqual(valid("smogon-chaos"));
  });

  it.each([
    ["unsorted-species", /species: .* repeated or out of order/],
    ["duplicate-item", /items: .* repeated or out of order/],
    ["unsorted-moves", /moves: .* repeated or out of order/],
    ["unsorted-spreads", /spread 1 is repeated or out of order/],
    ["unsorted-inputs", /source inputs: .* repeated or out of order/],
    ["unknown-species", /is not in the vocabulary roster/],
    ["unknown-forme", /forme aaaaaaaaa is not in the vocabulary roster/],
    ["unknown-item", /item aaaaaaaaa is not in the vocabulary/],
    ["unknown-move", /move aaaaaaaaa is not in the roster move pool/],
    ["unknown-ability", /ability aaaaaaaaa is not one the vocabulary lists/],
    ["unknown-nature", /nature aaaaaaaaa is not in the vocabulary/],
    ["forme-weights-sum", /forme weights do not sum to the denominator/],
    ["abilities-sum", /abilities: the weights do not sum/],
    ["items-sum", /items: the weights do not sum/],
    ["spreads-sum", /spreads: the weights do not sum/],
    ["moves-sum", /moves: the masses do not sum to slots times the denominator/],
    ["inclusion-over-denominator", /included more often than there are sets/],
    ["spread-over-limit", /more Stat Points than the format's limit/],
    ["too-many-items", /more entries than k_items/],
    ["too-many-spreads", /more entries than k_spreads/],
    ["too-many-moves", /more entries than k_moves/],
    ["stale-vocabulary-hash", /vocabulary_hash is not the hash/],
    ["other-format", /format_id differs/],
    ["other-engine-data", /engine_data_hash differs/],
  ])("refuses the reader-invalid sample %s", (name, message) => {
    const value = committed(`v1/usage-prior/reader-invalid/${name}.json`);
    expect(() => readUsagePrior(value, vocabulary)).toThrow(UsagePriorError);
    expect(() => readUsagePrior(value, vocabulary)).toThrow(message);
  });

  it("orders spreads of one nature by their Stat Points and refuses a repeat", () => {
    const prior = valid("smogon-chaos");
    const [a, b] = forme(prior).spreads.entries;
    b.nature = a.nature;
    expect(() => readUsagePrior(prior, vocabulary)).not.toThrow();
    forme(prior).spreads.entries = [b, a];
    expect(() => readUsagePrior(prior, vocabulary)).toThrow(/spread 1 is repeated or out of order/);
    forme(prior).spreads.entries = [a, { ...a }];
    expect(() => readUsagePrior(prior, vocabulary)).toThrow(/spread 1 is repeated or out of order/);
  });

  it("refuses an evs key, a paste or a team-shaped object anywhere (the schema is closed; the guard keeps such samples out of the corpus)", () => {
    const intruders: [string, (p: any) => void][] = [
      ["evs", (p) => (forme(p).spreads.entries[0].evs = {})],
      ["paste", (p) => (p.paste = "Pikachu @ Light Ball")],
      ["team", (p) => (forme(p).sets = [{ species: "x", moves: [], evs: {} }])],
    ];
    for (const [what, change] of intruders) {
      const prior = valid("smogon-chaos");
      change(prior);
      expect(() => readUsagePrior(prior, vocabulary), what).toThrow(/invalid usage prior/);
    }
  });

  it("refuses every schema-invalid sample", () => {
    for (const name of ["fractional-weight", "negative-residual", "zero-denominator", "zero-weight", "missing-unmapped", "missing-no-item", "ppm-share", "unknown-source-kind", "chaos-no-inputs", "bad-month", "missing-spread-point", "no-species", "no-formes", "next-version", "extra-field"]) {
      expect(() => readUsagePrior(committed(`v1/usage-prior/invalid/${name}.json`), vocabulary), name).toThrow(UsagePriorError);
    }
  });

  it("accepts a Mega forme's post-Evolution ability from any roster entry and refuses one no entry lists", () => {
    const mega = { ...vocabulary, roster: vocabulary.roster.map((r, k) => (k === 0 ? { ...r, required_item: vocabulary.items[0] as string } : r)) };
    const prior = { ...valid("smogon-chaos"), vocabulary_hash: teamVocabularyHash(mega) };
    // An ability of the second roster entry, which the first (the Mega forme) does not list; the entry still sums to its denominator.
    const foreign = vocabulary.roster[1]?.abilities.find((a) => !vocabulary.roster[0]?.abilities.includes(a)) ?? (vocabulary.roster[1]?.abilities[0] as string);
    const abilities = (id: string) => ({ denominator: 450, entries: [{ id, weight: 300 }], residual: 100, unmapped: 50 });
    forme(prior).abilities = abilities(foreign);
    expect(() => readUsagePrior(prior, mega)).not.toThrow();
    forme(prior).abilities = abilities("aaaaaaaaa");
    expect(() => readUsagePrior(prior, mega)).toThrow(/ability aaaaaaaaa/);
  });
});

describe("readBeliefConfig (ADR-025 section 5)", () => {
  const vocabulary = slim();
  const config = () => committed("v1/belief-config/valid/v1.json");

  it("reads the sample and hashes its JCS bytes", () => {
    expect(readBeliefConfig(config(), vocabulary).max_legality_redraws).toBe(32);
    expect(beliefConfigHash(config())).toBe(sha256Hex(jcs(config() as Json)));
  });

  it.each([
    ["unknown-ability", /is not in the team vocabulary/],
    ["unsorted-abilities", /repeated or out of order/],
    ["duplicate-ability", /repeated or out of order/],
  ])("refuses the reader-invalid sample %s", (name, message) => {
    const value = committed(`v1/belief-config/reader-invalid/${name}.json`);
    expect(() => readBeliefConfig(value, vocabulary)).toThrow(BeliefConfigError);
    expect(() => readBeliefConfig(value, vocabulary)).toThrow(message);
  });

  it("refuses every schema-invalid sample", () => {
    for (const name of ["extra-field", "fractional-bound", "zero-assignments", "missing-abilities"]) {
      expect(() => readBeliefConfig(committed(`v1/belief-config/invalid/${name}.json`), vocabulary), name).toThrow(BeliefConfigError);
    }
  });
});
