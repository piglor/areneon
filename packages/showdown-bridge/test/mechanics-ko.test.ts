/* eslint-disable @typescript-eslint/no-explicit-any */
import { readFileSync } from "node:fs";
import { beforeAll, describe, expect, it } from "vitest";
import type { Engine } from "../src/engine.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { loadEngine } from "../src/index.ts";
import { checkMechanicsConfig, loadMechanicsConfig, type MechanicsConfig } from "../src/mechanics-config.ts";
import { countKos, countPairKos, koBlocked, koCounts } from "../src/mechanics-ko.ts";
import { probeMechanics } from "../src/mechanics-probe.ts";
import { readMechanicsQuery } from "../src/mechanics-query.ts";
import { readMechanicsResult } from "../src/mechanics-result.ts";
import { exportPaste, importPaste } from "../src/teams.ts";

// ADR-016 slice 5: KO counts. The expected values were counted with Python
// outside the probe (see the fixture's `source`); the probe's vectors are the
// hand-worked ones of the damage fixture.

const fixtureUrl = (path: string): URL => new URL(`../../../fixtures/mechanics/gen9championsvgc2026regmc/${path}`, import.meta.url);
interface Counts {
  kind: "counts";
  ohko_no_crit: number;
  ohko_crit: number;
  thko_no_crit: number;
}
const ko = JSON.parse(readFileSync(fixtureUrl("ko/cases.json"), "utf8")) as {
  data_class: string;
  vector_cases: { id: string; hp: number; no_crit: number[]; crit: number[]; expected: Counts }[];
  enumeration: { no_crit: number[]; hp: number; sums: number[][]; expected_thko_no_crit: number };
  probe_cases: { id: string; description: string; base_case: string; defender_hp: number; expected: Counts }[];
};
const damage = JSON.parse(readFileSync(fixtureUrl("damage/cases.json"), "utf8")) as { cases: { id: string; query: any; expected: { no_crit: number[]; crit: number[]; ko: unknown } }[] };

let engine: Engine;
let config: MechanicsConfig;
beforeAll(() => {
  engine = loadEngine();
  config = loadMechanicsConfig(GOLDEN_FORMAT);
});

const caseOf = (id: string) => damage.cases.find((c) => c.id === id) as (typeof damage.cases)[number];
const answerOf = (query: unknown, withConfig: MechanicsConfig = config) => {
  const { result } = probeMechanics({ engine, config: withConfig }, readMechanicsQuery(query));
  if (result.kind !== "computed" || result.answer.kind !== "damage") throw new Error(`not computed: ${JSON.stringify(result)}`);
  return result.answer;
};
const withDefender = (id: string, change: (defender: any) => void): unknown => {
  const query = structuredClone(caseOf(id).query);
  change(query.query.defender);
  return query;
};
const canonical = (paste: string): string => exportPaste(engine, importPaste(engine, paste));

describe("the counting functions on hand-counted vectors", () => {
  it("is synthetic", () => {
    expect(ko.data_class).toBe("synthetic");
  });

  it.each(ko.vector_cases.map((c) => [c.id, c] as const))("%s", (_id, c) => {
    expect(koCounts(c.no_crit, c.crit, c.hp, false)).toEqual(c.expected);
  });

  it("covers the boundary where the lowest roll equals the HP", () => {
    const boundary = ko.vector_cases.find((c) => c.id === "boundary-lowest-roll-equals-hp") as (typeof ko.vector_cases)[number];
    expect(Math.min(...boundary.no_crit)).toBe(boundary.hp);
    expect(countKos(boundary.no_crit, boundary.hp)).toBe(16);
    expect(countKos(boundary.no_crit, boundary.hp + 1)).toBe(15);
  });

  it("counts the two-hit pairs against an explicit enumeration of all 256 sums", () => {
    const { no_crit, hp, sums, expected_thko_no_crit: expected } = ko.enumeration;
    expect(sums).toHaveLength(16);
    // the written-out table is the sums of the vector, row = first roll, column = second roll
    sums.forEach((row, first) => row.forEach((sum, second) => expect(sum).toBe((no_crit[first] as number) + (no_crit[second] as number))));
    expect(sums.flat()).toHaveLength(256);
    expect(sums.flat().filter((sum) => sum >= hp)).toHaveLength(expected);
    expect(countPairKos(no_crit, hp)).toBe(expected);
    // the count is not a count of rolls: 241 pairs, not 16 or 15 rolls
    expect(expected).toBeGreaterThan(16);
    expect(expected).toBeLessThan(256);
  });

  it("uses the same vector for both hits and never the crit vector in the two-hit count", () => {
    expect(koCounts([50, 50, ...new Array(14).fill(0)], new Array(16).fill(1000), 100, false)).toMatchObject({ ohko_crit: 16, thko_no_crit: 4 });
  });

  it("withholds every count when blocked", () => {
    expect(koCounts(new Array(16).fill(500), new Array(16).fill(500), 1, true)).toEqual({ kind: "unsupported", reason: "ko_state_change" });
  });
});

describe("the damage probe's counts", () => {
  it.each(ko.probe_cases.map((c) => [c.id, c] as const))("%s", (_id, c) => {
    const answer = answerOf(withDefender(c.base_case, (defender) => (defender.hp = c.defender_hp)));
    expect(answer.hp_current).toBe(c.defender_hp);
    expect(answer.ko).toEqual(c.expected);
  });

  it("every hand-worked damage case carries the counts counted outside the module under test", () => {
    // Plain loops written here, not countKos/countPairKos: the expected values must not come from the code they check.
    // tests/scripts/test_mechanics_ko_fixture.py recounts the same fixture in Python.
    for (const c of damage.cases) {
      const hp = (c.expected as unknown as { hp_current: number }).hp_current;
      let ohkoNoCrit = 0;
      let ohkoCrit = 0;
      let thko = 0;
      for (const first of c.expected.no_crit) {
        if (first >= hp) ohkoNoCrit += 1;
        for (const second of c.expected.no_crit) if (first + second >= hp) thko += 1;
      }
      for (const roll of c.expected.crit) if (roll >= hp) ohkoCrit += 1;
      expect(c.expected.ko, c.id).toEqual({ kind: "counts", ohko_no_crit: ohkoNoCrit, ohko_crit: ohkoCrit, thko_no_crit: thko });
    }
  });

  it("counts are before the HP clamp: a defender at 1 HP counts the same rolls as the damage dealt would", () => {
    const answer = answerOf(withDefender("single-target", (defender) => (defender.hp = 1)));
    expect(answer.ko).toEqual({ kind: "counts", ohko_no_crit: 16, ohko_crit: 16, thko_no_crit: 256 });
    expect(answer.no_crit).toEqual(caseOf("single-target").expected.no_crit);
  });

  it("an immunity has sixteen zero rolls and no KO", () => {
    const ghost = (caseOf("mega").query.query.attacker as { set: string }).set;
    const query = structuredClone(caseOf("single-target").query);
    query.query.move = "bodyslam";
    query.query.defender.set = ghost;
    expect(answerOf(query).ko).toEqual({ kind: "counts", ohko_no_crit: 0, ohko_crit: 0, thko_no_crit: 0 });
  });
});

describe("ko_blockers", () => {
  let SASH = "";
  let MULTISCALE = "";
  let RESTING = "";
  beforeAll(() => {
    SASH = canonical("Arcanine @ Focus Sash\nAbility: Intimidate\nLevel: 50\nEVs: 32 HP / 32 SpA\nModest Nature\n- Heat Wave\n- Protect\n");
    MULTISCALE = canonical("Dragonite\nAbility: Multiscale\nLevel: 50\nEVs: 32 HP / 32 Atk\nAdamant Nature\n- Extreme Speed\n- Protect\n");
    RESTING = canonical("Arcanine @ Leftovers\nAbility: Intimidate\nLevel: 50\nEVs: 32 HP / 32 SpA\nModest Nature\n- Heat Wave\n- Protect\n");
  });

  it("the committed list is checked against the pinned engine and names the ADR's classes", () => {
    expect(checkMechanicsConfig(engine, config)).toEqual([]);
    expect(config.ko_blockers.items).toEqual(expect.arrayContaining(["focussash", "sitrusberry", "oranberry", "occaberry", "yacheberry"]));
    expect(config.ko_blockers.abilities).toEqual(expect.arrayContaining(["sturdy", "disguise", "multiscale"]));
  });

  it("a blocker item returns the same vectors with the counts unsupported", () => {
    const plain = answerOf(withDefender("single-target", () => undefined));
    const blocked = answerOf(withDefender("single-target", (defender) => (defender.set = SASH)));
    expect(blocked.ko).toEqual({ kind: "unsupported", reason: "ko_state_change" });
    expect(blocked.no_crit).toEqual(plain.no_crit);
    expect(blocked.crit).toEqual(plain.crit);
    expect(blocked.hp_current).toBe(plain.hp_current);
    expect(plain.ko).toMatchObject({ kind: "counts" });
    const { result } = probeMechanics({ engine, config }, readMechanicsQuery(withDefender("single-target", (defender) => (defender.set = SASH))));
    expect(readMechanicsResult(result)).toEqual(result);
  });

  it("a blocker ability does the same; a held item outside the list does not", () => {
    const blocked = answerOf(withDefender("single-target", (defender) => (defender.set = MULTISCALE)));
    expect(blocked.ko).toEqual({ kind: "unsupported", reason: "ko_state_change" });
    expect(blocked.no_crit).toHaveLength(16);
    expect(answerOf(withDefender("single-target", (defender) => (defender.set = RESTING))).ko).toMatchObject({ kind: "counts" });
  });

  it("is the config that decides: an empty list gives counts for the same defender, an added id blocks", () => {
    const none = { ...config, ko_blockers: { abilities: [], items: [] } };
    expect(answerOf(withDefender("single-target", (defender) => (defender.set = SASH)), none).ko).toMatchObject({ kind: "counts" });
    const leftovers = { ...config, ko_blockers: { abilities: [], items: ["leftovers"] } };
    expect(answerOf(withDefender("single-target", (defender) => (defender.set = RESTING)), leftovers).ko).toEqual({ kind: "unsupported", reason: "ko_state_change" });
  });

  it("an id that is not in the engine's vocabulary is a config problem", () => {
    const bad = { ...config, ko_blockers: { abilities: ["notanability"], items: ["notanitem"] } };
    expect(checkMechanicsConfig(engine, bad).join("\n")).toMatch(/notanitem is not an item[\s\S]*notanability is not an ability/);
  });

  it("koBlocked matches an item or an ability id", () => {
    const blockers = { abilities: ["sturdy"], items: ["focussash"] };
    expect(koBlocked(blockers, { item: "focussash", ability: "x" })).toBe(true);
    expect(koBlocked(blockers, { item: "", ability: "sturdy" })).toBe(true);
    expect(koBlocked(blockers, { item: "", ability: "x" })).toBe(false);
  });
});
