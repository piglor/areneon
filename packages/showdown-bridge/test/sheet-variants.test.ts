import { describe, expect, it } from "vitest";
import {
  checkSheetVariants,
  type Engine,
  loadEngine,
  loadRulesSnapshot,
  SHEET_VARIANT_KEYS,
  SheetVariantError,
  sheetVariantPairs,
} from "../src/index.ts";
import type { SimFormat, SimRuleTable } from "../src/engine.ts";

const CLOSED = "gen9championsvgc2026regmc";
const OPEN = "gen9championsvgc2026regmcbo3";
const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const snapshot = loadRulesSnapshot(engine.commit);

describe("checkSheetVariants on the pinned engine (ADR-027 section 4)", () => {
  it("passes for the committed pair", () => {
    expect(() => checkSheetVariants(engine, CLOSED, OPEN)).not.toThrow();
    expect(sheetVariantPairs(snapshot)).toEqual([{ closed: CLOSED, open: OPEN }]);
  });

  // The ticket's rule set, recomputed from the engine's tables with no help from the module.
  it("matches the engine's actual rule-table difference", () => {
    const formats = engine.sim.Dex.formats;
    const a = formats.getRuleTable(formats.get(CLOSED));
    const b = formats.getRuleTable(formats.get(OPEN));
    const onlyOne = [...new Set([...a.keys(), ...b.keys()])].filter((key) => a.has(key) !== b.has(key)).sort();
    expect(onlyOne).toEqual(["bestof", "forceopenteamsheets", "openteamsheets"]);
    expect([...SHEET_VARIANT_KEYS]).toEqual(onlyOne);
    expect(b.valueRules.get("bestof")).toBe("3");
    expect(formats.get(CLOSED).mod).toBe(formats.get(OPEN).mod);
  });

  it("refuses a pair in the wrong order, the same format twice and an unknown format", () => {
    expect(() => checkSheetVariants(engine, OPEN, CLOSED)).toThrow(SheetVariantError);
    expect(() => checkSheetVariants(engine, CLOSED, CLOSED)).toThrow(SheetVariantError);
    expect(() => checkSheetVariants(engine, CLOSED, "gen9nonsense")).toThrow(/does not exist/);
    expect(() => checkSheetVariants(engine, "gen9nonsense", OPEN)).toThrow(/does not exist/);
    // Another doubles format of the same mod differs in far more than the sheet rules.
    expect(() => checkSheetVariants(engine, CLOSED, "gen9championsbssregmc")).toThrow(SheetVariantError);
  });

  it("refuses an open format with no closed counterpart of its regulation", () => {
    const only = { ...snapshot, formats: { [OPEN]: snapshot.formats[OPEN] } } as typeof snapshot;
    expect(() => sheetVariantPairs(only as typeof snapshot)).toThrow(/no closed format/);
  });
});

// Stand-in rule tables, so doctored differences can be tried without a doctored engine.
interface Spec {
  mod?: string;
  gameType?: string;
  battle?: Record<string, unknown>;
  rules: Record<string, string>;
  values?: Record<string, string>;
  fields?: Partial<SimRuleTable>;
}
function table(spec: Spec): SimRuleTable {
  const t = new Map(Object.entries(spec.rules)) as SimRuleTable;
  t.valueRules = new Map(Object.entries(spec.values ?? {}));
  Object.assign(t, { minTeamSize: 6, maxTeamSize: 6, pickedTeamSize: 4, adjustLevel: 50, evLimit: 510, ...spec.fields });
  return t;
}
function fakeEngine(closed: Spec, open: Spec): Engine {
  const entries = new Map<string, { format: SimFormat; table: SimRuleTable }>();
  for (const [id, spec] of [["c", closed], ["o", open]] as const) {
    entries.set(id, {
      format: { exists: true, id, name: id, mod: spec.mod ?? "champions", gameType: spec.gameType ?? "doubles", ...(spec.battle ? { battle: spec.battle } : {}) },
      table: table(spec),
    });
  }
  const byFormat = new Map([...entries.values()].map((e) => [e.format, e.table]));
  return {
    sim: {
      Dex: {
        formats: {
          get: (id: string) => entries.get(id)?.format ?? ({ exists: false } as SimFormat),
          getRuleTable: (format: SimFormat) => {
            if (format.id === "boom") throw new Error("boom");
            return byFormat.get(format) as SimRuleTable;
          },
        },
      },
    },
  } as unknown as Engine;
}

const base: Spec = { rules: { flatrules: "x", vgctimer: "y", itemclause: "z" }, values: { itemclause: "1", adjustlevel: "50" } };
const closedSpec: Spec = { ...base, rules: { ...base.rules, openteamsheets: "fmt" } };
const openSpec: Spec = {
  ...base,
  rules: { ...base.rules, forceopenteamsheets: "fmt", bestof: "fmt" },
  values: { ...base.values, bestof: "3" },
};

describe("checkSheetVariants on doctored rule tables", () => {
  const refuses = (open: Spec, pattern: RegExp, closed: Spec = closedSpec) =>
    expect(() => checkSheetVariants(fakeEngine(closed, open), "c", "o")).toThrow(pattern);

  it("passes for exactly the sheet and series rules", () => {
    expect(() => checkSheetVariants(fakeEngine(closedSpec, openSpec), "c", "o")).not.toThrow();
  });

  it("refuses a fourth differing rule, in either direction", () => {
    refuses({ ...openSpec, rules: { ...openSpec.rules, sleepclause: "x" } }, /present in one format only/);
    refuses(openSpec, /present in one format only/, { ...closedSpec, rules: { ...closedSpec.rules, sleepclause: "x" } });
  });

  it("refuses a missing sheet or series rule", () => {
    const { bestof: _b, ...noBestOf } = openSpec.rules;
    refuses({ ...openSpec, rules: noBestOf }, /present in one format only/);
    refuses({ ...openSpec, rules: { ...openSpec.rules, openteamsheets: "x" } }, /must have forceopenteamsheets and not openteamsheets/);
  });

  it("refuses another mod, game type, battle options, rule source, value and field", () => {
    refuses({ ...openSpec, mod: "gen9" }, /mod gen9 is not champions/);
    refuses({ ...openSpec, gameType: "singles" }, /game type singles is not doubles/);
    refuses({ ...openSpec, battle: { trunc: 1 } }, /battle options differ/);
    refuses({ ...openSpec, rules: { ...openSpec.rules, vgctimer: "other" } }, /rule vgctimer comes from another source/);
    refuses({ ...openSpec, values: { ...openSpec.values, itemclause: "2" } }, /value rule itemclause differs/);
    refuses({ ...openSpec, values: { ...openSpec.values, extra: "1" } }, /value rule extra differs/);
    refuses({ ...openSpec, fields: { adjustLevel: 100 } }, /adjustLevel differs/);
    refuses({ ...openSpec, fields: { pickedTeamSize: 6 } }, /pickedTeamSize differs/);
  });

  it("names the engine's own refusal", () => {
    const e = fakeEngine(closedSpec, openSpec);
    expect(() => checkSheetVariants(e, "c", "boom")).toThrow(SheetVariantError);
    (e.sim.Dex.formats as unknown as { get: (id: string) => SimFormat }).get = (id) => {
      throw new Error(`no ${id}`);
    };
    expect(() => checkSheetVariants(e, "c", "o")).toThrow(/the engine refuses a format: no c/);
  });
});
