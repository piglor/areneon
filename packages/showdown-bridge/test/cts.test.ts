import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { GOLDEN_FORMAT, GOLDEN_SEED, syntheticTeam, TEAMS_DIR } from "../src/golden.ts";
import { importPaste, loadEngine, type PokemonSet } from "../src/index.ts";

// CTS invariant (ADR-003, #676): with Open Team Sheets declined, Team
// Preview shows the opponent's species and details only. In-process, the
// sheets stay closed unless someone writes `>show-openteamsheets`; runBattle
// never does, and an actor's choice cannot (each of its lines is prefixed
// with its own `>p1`/`>p2`).

const engine = loadEngine();
const own = syntheticTeam();
const opponent = importPaste(engine, readFileSync(join(TEAMS_DIR, "legal-opponent.txt"), "utf8"));

const id = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");

/** Item, ability, move and nature names of `team`, as written and as IDs. */
function sheet(team: PokemonSet[]): Set<string> {
  const names = team.flatMap((set) => [set.item, set.ability, set.nature ?? "", ...set.moves]).filter(Boolean);
  return new Set(names.flatMap((name) => [name, id(name)]));
}

/** Everything p1's stream receives up to and including its Team Preview request. */
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

// Names only the opponent's sheet would reveal: shared ones prove nothing.
const ownSheet = sheet(own);
const secret = [...sheet(opponent)].filter((name) => !ownSheet.has(name));

describe("Team Preview with Open Team Sheets declined", () => {
  it("uses an opponent team that shares no sheet entry with p1's", () => {
    expect(secret.length).toBeGreaterThan(20);
    expect(opponent.some((set) => own.some((mine) => mine.species === set.species))).toBe(false);
  });

  it("shows the opponent's species and details only", async () => {
    const lines = await p1Preview();
    const pokes = lines.filter((line) => line.startsWith("|poke|p2|"));
    expect(pokes.map((line) => line.split("|")[3]?.split(",")[0])).toEqual(opponent.map((set) => set.species));
    for (const line of pokes) expect(line).toMatch(/^\|poke\|p2\|[^,|]+, L50(, [MF])?\|(item)?$/);
    expect(lines.some((line) => line.startsWith("|showteam|"))).toBe(false);
  });

  it("leaks no item, ability, move or nature of the opponent", async () => {
    const view = (await p1Preview()).join("\n");
    expect(secret.filter((name) => view.includes(name))).toEqual([]);
  });

  // Negative control: the check above does see a sheet once one is shown.
  it("would see the opponent's sheet if Open Team Sheets were accepted", async () => {
    const view = (await p1Preview(true)).join("\n");
    expect(view).toContain("|showteam|p2|");
    expect(secret.filter((name) => view.includes(name)).length).toBeGreaterThan(0);
  });
});
