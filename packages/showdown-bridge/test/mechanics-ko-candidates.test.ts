import { beforeAll, describe, expect, it } from "vitest";
import type { Engine } from "../src/engine.ts";
import { GOLDEN_FORMAT, goldenTeamVocabulary } from "../src/golden.ts";
import { loadEngine } from "../src/index.ts";
import { loadMechanicsConfig, type MechanicsConfig } from "../src/mechanics-config.ts";
import { checkKoBlockers, KO_HOOKS, koCandidates } from "../src/mechanics-ko-candidates.ts";

// #1166: the KO blocker list is tied to the roster through the engine's handler
// names. Decisions are the config's; these tests hold the mechanism.

let engine: Engine;
let config: MechanicsConfig;
const vocabulary = goldenTeamVocabulary();
beforeAll(() => {
  engine = loadEngine();
  config = loadMechanicsConfig(GOLDEN_FORMAT);
});
const edited = (change: (copy: MechanicsConfig) => void): MechanicsConfig => {
  const copy = structuredClone(config);
  change(copy);
  return copy;
};
const nonBlockers = (c: MechanicsConfig) => c.ko_non_blockers as NonNullable<MechanicsConfig["ko_non_blockers"]>;

describe("the candidates are read from the engine data", () => {
  it("every handler name is used by some roster item or ability (a mistyped name would find nothing)", () => {
    const found = new Set<string>();
    const candidates = koCandidates(engine, GOLDEN_FORMAT, vocabulary);
    for (const hooks of [...Object.values(candidates.abilities), ...Object.values(candidates.items)]) for (const hook of hooks) found.add(hook);
    expect([...found].sort()).toEqual([...KO_HOOKS].sort());
  });

  it("finds every reviewed blocker, a plain item or ability not at all, and only roster-legal ids", () => {
    const candidates = koCandidates(engine, GOLDEN_FORMAT, vocabulary);
    for (const id of config.ko_blockers.abilities) expect(candidates.abilities, id).toHaveProperty(id);
    for (const id of config.ko_blockers.items) expect(candidates.items, id).toHaveProperty(id);
    expect(candidates.items).not.toHaveProperty("leftovers");
    expect(candidates.abilities).not.toHaveProperty("intimidate");
    const roster = new Set(vocabulary.roster.flatMap((species) => species.abilities));
    for (const id of Object.keys(candidates.abilities)) expect(roster.has(id), id).toBe(true);
    for (const id of Object.keys(candidates.items)) expect(vocabulary.items.includes(id), id).toBe(true);
  });
});

describe("checkKoBlockers", () => {
  it("passes for the committed config", () => {
    expect(checkKoBlockers(engine, config, vocabulary)).toEqual([]);
  });

  it("fails on a candidate that is in neither list, naming its handlers", () => {
    const withoutAbility = edited((c) => void (nonBlockers(c).abilities = nonBlockers(c).abilities.filter((entry) => entry.id !== "filter")));
    expect(checkKoBlockers(engine, withoutAbility, vocabulary)).toEqual(["ko_blockers: ability filter has onSourceModifyDamage and is in neither ko_blockers nor ko_non_blockers"]);
    const withoutItem = edited((c) => void (nonBlockers(c).items = nonBlockers(c).items.filter((entry) => entry.id !== "rockyhelmet")));
    expect(checkKoBlockers(engine, withoutItem, vocabulary)).toEqual(["ko_blockers: item rockyhelmet has onDamagingHit and is in neither ko_blockers nor ko_non_blockers"]);
  });

  it("fails on every candidate when the config has no reviewed list (a config written before it existed)", () => {
    const legacy = edited((c) => void delete c.ko_non_blockers);
    const problems = checkKoBlockers(engine, legacy, vocabulary);
    const candidates = koCandidates(engine, GOLDEN_FORMAT, vocabulary);
    const expected = Object.keys(candidates.abilities).length + Object.keys(candidates.items).length - config.ko_blockers.abilities.length - config.ko_blockers.items.length;
    expect(problems).toHaveLength(expected);
  });

  it("fails on an id in both lists, and on a stale non-blocker that is no candidate", () => {
    const both = edited((c) => void nonBlockers(c).abilities.push({ id: "sturdy", reason: "x" }));
    expect(checkKoBlockers(engine, both, vocabulary)).toEqual(["ko_non_blockers: ability sturdy is also in ko_blockers"]);
    const stale = edited((c) => {
      nonBlockers(c).items.push({ id: "leftovers", reason: "x" });
      nonBlockers(c).abilities.push({ id: "intimidate", reason: "x" });
    });
    expect(checkKoBlockers(engine, stale, vocabulary)).toEqual([
      "ko_non_blockers: ability intimidate is not a roster-legal ability with a KO handler",
      "ko_non_blockers: item leftovers is not a roster-legal item with a KO handler",
    ]);
  });

  it("flags an ability a regulation bump adds to the roster: Ice Face and Shadow Shield (Tera Shell has no handler to find: its effect is in the engine's simulation code)", () => {
    const grown = structuredClone(vocabulary);
    (grown.roster[0] as { abilities: string[] }).abilities = ["iceface", "shadowshield"];
    const problems = checkKoBlockers(engine, config, grown);
    for (const id of ["iceface", "shadowshield"]) expect(problems.some((p) => p.startsWith(`ko_blockers: ability ${id} has `)), id).toBe(true);
  });

  it("flags an item the format starts to allow", () => {
    const grown = structuredClone(vocabulary);
    grown.items = [...grown.items, "assaultvest", "protectivepads", "weaknesspolicy"].sort();
    expect(checkKoBlockers(engine, config, grown).map((p) => p.split(" has ")[0])).toEqual(["ko_blockers: item weaknesspolicy"]);
  });

  it("refuses a vocabulary of another commit or format", () => {
    expect(checkKoBlockers(engine, config, { ...vocabulary, showdown_commit: "0".repeat(40) })).toEqual(["ko_blockers: the team vocabulary is for another engine commit"]);
    expect(checkKoBlockers(engine, { ...config, format_id: "gen9other" }, vocabulary)).toEqual(["ko_blockers: the team vocabulary is for another format"]);
  });
});
