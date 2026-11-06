// The engine-sent team sheet of an open-sheet format (ADR-027 section 5, #1150), read beside the
// interpreter, not in it: `interpreter.cjs` is inlined into the replay pages and its hash is part of
// ADR-018's baseline identity, so it stays byte-identical. `index.ts` takes the `|showteam|` lines out
// of the stream before the interpreter sees them and attaches the sheets to the states afterwards.
//
// The packed team is parsed with a parser written from the engine's sim/TEAMS.md and the committed
// open-regime channels; no client code was used. One Pokemon is twelve `|` fields, `]` between them:
//   NAME|SPECIES|ITEM|ABILITY|MOVES|NATURE|EVS|GENDER|IVS|SHINY|LEVEL|EXTRAS
// The engine's sheet has no nickname (it prints the species name as NAME and leaves SPECIES blank), no
// Stat Points, IVs, shiny flag or extras: any of those present is refused. This reader holds no
// vocabulary, so ids come from the names (the Python reducer checks them against the engine data).

import type { PublicSheetMon } from "./index.ts";

const FIELDS = 12;
const MAX_POKEMON = 24; // a sanity bound on the list; the preview count is checked separately
const MAX_LEVEL = 100;
const NATURE = /^[A-Za-z]{1,16}$/;
const LEVEL = /^\d{1,3}$/;

export class SheetError extends Error {
  override name = "SheetError";
}

const need = (ok: boolean, reason: string): void => {
  if (!ok) throw new SheetError(reason);
};

export const toId = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");

/** The Pokemon of a packed team sheet, or a `SheetError` naming the rule. */
export function parseSheet(packed: string): PublicSheetMon[] {
  const entries = packed.split("]");
  need(entries.length <= MAX_POKEMON, "a sheet lists between one and 24 Pokemon");
  return entries.map((entry) => {
    const f = entry.split("|");
    need(f.length === FIELDS, "a sheet Pokemon has 12 fields");
    const [name = "", nickname, item = "", ability = "", moves = "", nature = "", points, gender = "", ivs, shiny, level = "", extras] = f;
    need(!nickname, "a sheet shows no nickname");
    need(!points && !ivs, "a sheet shows no Stat Points and no IVs");
    need(!shiny && !extras, "a sheet carries no shiny flag and no extra fields");
    need(name !== "" && ability !== "" && moves !== "", "a sheet Pokemon lacks its species, ability or moves");
    need(gender === "" || gender === "M" || gender === "F", `unknown sheet gender '${gender}'`);
    need(nature === "" || NATURE.test(nature), "a sheet nature is a name");
    need(level === "" || (LEVEL.test(level) && Number(level) >= 1 && Number(level) <= MAX_LEVEL), "a sheet level is a number from 1 to 100");
    const names = moves.split(",");
    need(!names.includes(""), "a sheet move list has an empty move");
    need(![name, ability, ...names].some((text) => toId(text) === ""), "a sheet name has no letters");
    return {
      species: toId(name),
      item: toId(item) || null,
      ability: toId(ability),
      moves: names.map(toId),
      nature: toId(nature) || null,
      gender: gender || null,
      level: level === "" ? MAX_LEVEL : Number(level),
    };
  });
}
