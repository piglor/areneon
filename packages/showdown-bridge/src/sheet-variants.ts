// The sheet-variant check (ADR-027 section 4, ADR-003 erratum E-3.1; the
// ADR-016 scenario-check pattern). Areneon trains and evaluates on the
// engine's open-sheet format as well as on the closed-sheet CTS target, which
// is only sound if the two differ in exactly the sheet and series rules. The
// check asks the pinned engine's rule tables and refuses the pair otherwise.

import type { Engine, SimRuleTable } from "./engine.ts";
import { battleOptions, SHARED_FIELDS } from "./mechanics-scenario.ts";
import type { RulesSnapshot } from "./rules.ts";
import { teamSheetsOf } from "./rules.ts";

/**
 * The rule keys by which the open-sheet variant may differ from the closed
 * one: the closed format opts into `openteamsheets` and declines, the open
 * one forces the sheets and plays a best-of series. Checked against the
 * engine's rule tables at the pin, never assumed.
 */
export const SHEET_VARIANT_KEYS = ["bestof", "forceopenteamsheets", "openteamsheets"] as const;

export class SheetVariantError extends Error {
  override name = "SheetVariantError";
  /** Every difference found, in words. */
  readonly problems: string[];

  constructor(problems: string[]) {
    super(`the open-sheet format is not the closed one plus the sheet rules: ${problems.join("; ")}`);
    this.problems = problems;
  }
}

const TEAM_FIELDS = ["minTeamSize", "maxTeamSize", "pickedTeamSize"] as const;

function differences(
  engine: Engine,
  closedId: string,
  openId: string,
): string[] {
  const formats = engine.sim.Dex.formats;
  const closed = formats.get(closedId);
  const open = formats.get(openId);
  if (!closed.exists) return [`${closedId} does not exist`];
  if (!open.exists) return [`${openId} does not exist`];
  const tc: SimRuleTable = formats.getRuleTable(closed);
  const to: SimRuleTable = formats.getRuleTable(open);
  const problems: string[] = [];
  if (open.mod !== closed.mod) problems.push(`mod ${open.mod} is not ${closed.mod}`);
  if (open.gameType !== closed.gameType) problems.push(`game type ${open.gameType} is not ${closed.gameType}`);
  if (battleOptions(open) !== battleOptions(closed)) problems.push("the battle options differ");
  const keys = new Set([...tc.keys(), ...to.keys()]);
  const differing = [...keys].filter((key) => tc.has(key) !== to.has(key)).sort();
  if (differing.join() !== [...SHEET_VARIANT_KEYS].join()) {
    problems.push(`the rule keys present in one format only are [${differing.join(", ")}], not [${SHEET_VARIANT_KEYS.join(", ")}]`);
  }
  if (!tc.has("openteamsheets") || tc.has("forceopenteamsheets")) problems.push("the closed format must have openteamsheets and not forceopenteamsheets");
  if (!to.has("forceopenteamsheets") || to.has("openteamsheets")) problems.push("the open format must have forceopenteamsheets and not openteamsheets");
  if (tc.has("bestof") || !to.has("bestof")) problems.push("only the open format may have bestof");
  for (const [key, source] of to) if (tc.has(key) && tc.get(key) !== source) problems.push(`rule ${key} comes from another source`);
  const valueKeys = new Set([...tc.valueRules.keys(), ...to.valueRules.keys()]);
  for (const key of valueKeys) {
    if ((SHEET_VARIANT_KEYS as readonly string[]).includes(key)) continue;
    if (tc.valueRules.get(key) !== to.valueRules.get(key)) problems.push(`value rule ${key} differs`);
  }
  for (const key of [...SHARED_FIELDS, ...TEAM_FIELDS]) if (to[key] !== tc[key]) problems.push(`${key} differs`);
  return problems;
}

/**
 * Returns when the engine's rule tables for `openId` equal `closedId`'s up to
 * exactly `{openteamsheets, forceopenteamsheets, bestof}` (same mod, game
 * type, battle options, rule sources, rule values, levels and team sizes);
 * throws a `SheetVariantError` naming every difference otherwise.
 */
export function checkSheetVariants(engine: Engine, closedId: string, openId: string): void {
  let problems: string[];
  try {
    problems = differences(engine, closedId, openId);
  } catch (error) {
    problems = [`the engine refuses a format: ${(error as Error).message}`];
  }
  if (problems.length > 0) throw new SheetVariantError(problems);
}

/**
 * The (closed, open) pairs the snapshot declares: each open format with every
 * closed format of the same regulation. `check-rules` runs `checkSheetVariants`
 * on each, so an open format with no closed counterpart is refused too.
 */
export function sheetVariantPairs(snapshot: RulesSnapshot): { closed: string; open: string }[] {
  const entries = Object.entries(snapshot.formats);
  return entries
    .filter(([, spec]) => teamSheetsOf(spec) === "open")
    .flatMap(([open, spec]) => {
      const closed = entries.filter(([, other]) => teamSheetsOf(other) === "closed" && other.regulation === spec.regulation);
      if (closed.length === 0) throw new SheetVariantError([`${open} has no closed format of regulation ${spec.regulation}`]);
      return closed.map(([id]) => ({ closed: id, open }));
    });
}
