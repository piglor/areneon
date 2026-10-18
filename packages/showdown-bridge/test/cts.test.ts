import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { GOLDEN_FORMAT, GOLDEN_SEED, syntheticTeam, TEAMS_DIR } from "../src/golden.ts";
import { importPaste, loadEngine, type PokemonSet } from "../src/index.ts";

// CTS invariant (ADR-003, #676): with Open Team Sheets declined, Team
// Preview shows the opponent's species, level and gender only. In-process,
// the sheets stay closed unless someone writes `>show-openteamsheets`;
// runBattle never does, and an actor's choice cannot (each of its lines is
// prefixed with its own `>p1`/`>p2`).

const engine = loadEngine();
const own = syntheticTeam();
const opponent = importPaste(engine, readFileSync(join(TEAMS_DIR, "legal-opponent.txt"), "utf8"));

/** Every spelling the engine uses for a name: as written, as an ID, and packed. */
const spellings = (name: string): string[] => [
  name,
  name.toLowerCase().replace(/[^a-z0-9]/g, ""),
  name.replace(/[^A-Za-z0-9]/g, ""),
];

/** The sheet entries of `team` a preview must not show: nicknames, items, abilities, moves, natures. */
const sheet = (team: PokemonSet[]): string[] =>
  team.flatMap((set) =>
    [set.name === set.species ? "" : set.name, set.item, set.ability, set.nature ?? "", ...set.moves].filter(Boolean),
  );

/** What p1's stream receives up to and including its Team Preview request. */
async function p1Preview(showSheets = false): Promise<string[]> {
  const streams = engine.sim.getPlayerStreams(new engine.sim.BattleStream());
  streams.omniscient.write(
    [
      `>start ${JSON.stringify({ formatid: GOLDEN_FORMAT, seed: GOLDEN_SEED })}`,
      `>player p1 ${JSON.stringify({ name: "p1", team: engine.sim.Teams.pack(own) })}`,
      `>player p2 ${JSON.stringify({ name: "p2", team: engine.sim.Teams.pack(opponent) })}`,
      ...(showSheets ? [">show-openteamsheets"] : []),
    ].join("\n"),
  );
  const lines: string[] = [];
  for await (const chunk of streams.p1) {
    lines.push(...chunk.split("\n"));
    if (lines.some((line) => line.startsWith("|request|") && line.includes('"teamPreview":true'))) break;
  }
  streams.omniscient.writeEnd();
  return lines;
}

const leaked = (view: string, entries: string[]): string[] =>
  entries.filter((entry) => spellings(entry).some((spelling) => view.includes(spelling)));

// Everything p1 may see before choosing leads. Anything else (a sheet, the
// opponent's request, stats or Stat Points) fails the allowlist.
const ALLOWED: readonly RegExp[] = [
  /^$/,
  /^\|t:\|\d+$/,
  /^\|gametype\|doubles$/,
  /^\|player\|p[12]\|p[12]\|\|$/,
  /^\|gen\|9$/,
  /^\|tier\|[^|]+$/,
  /^\|rule\|[^|]+$/,
  /^\|clearpoke$/,
  /^\|poke\|p1\|[^|]+\|$/,
  /^\|poke\|p2\|[^,|]+, L50(, [MF])?\|$/,
  /^\|teampreview\|4$/,
  /^\|uhtml\|otsrequest\|/,
  /^\|request\|/,
];

describe("Team Preview with Open Team Sheets declined", () => {
  it("uses an opponent team that shares no sheet entry or species with p1's", () => {
    const ours = new Set(sheet(own).flatMap(spellings));
    expect(sheet(opponent).flatMap(spellings).filter((spelling) => ours.has(spelling))).toEqual([]);
    expect(opponent.filter((set) => own.some((mine) => mine.species === set.species))).toEqual([]);
    expect(opponent.some((set) => set.name !== set.species)).toBe(true);
  });

  it("shows p1 only the allowed preview lines", async () => {
    const lines = await p1Preview();
    expect(lines.filter((line) => !ALLOWED.some((allowed) => allowed.test(line)))).toEqual([]);
    const pokes = lines.filter((line) => line.startsWith("|poke|p2|"));
    expect(pokes.map((line) => line.split("|")[3]?.split(",")[0])).toEqual(opponent.map((set) => set.species));
  });

  it("gives p1 a request about its own side only", async () => {
    const request = (await p1Preview()).find((line) => line.startsWith("|request|")) ?? "";
    const { side } = JSON.parse(request.slice("|request|".length)) as { side: { id: string } };
    expect(side.id).toBe("p1");
    expect(request).not.toMatch(/p2[a-z]?:/);
  });

  it("leaks no nickname, item, ability, move or nature of the opponent", async () => {
    expect(leaked((await p1Preview()).join("\n"), sheet(opponent))).toEqual([]);
  });

  // Negative control: once a sheet is shown, the same check finds every
  // item, ability, move and nature on it. (Showdown never sends nicknames or
  // stats in a sheet, so the allowlist above is what excludes those.)
  it("would see the whole sheet if Open Team Sheets were accepted", async () => {
    const view = (await p1Preview(true)).join("\n");
    expect(view).toContain("|showteam|p2|");
    const shown = sheet(opponent.map((set) => ({ ...set, name: set.species })));
    expect(leaked(view, shown)).toEqual(shown);
  });
});
