import { describe, expect, it } from "vitest";
import { fixtures, fullDex, loadInterpreter } from "./support.ts";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { applyVolatileCopy, interpretChannel as wrapped, onlyVolatileCopy, volatileCopyDex } from "../src/index.ts";

// The volatiles a Baton Pass or Shed Tail hands over (#1102): the engine copies every volatile except
// the `noCopy` ones (a Shed Tail only Substitute) and prints no line for the copy.
const api = loadInterpreter();
const dex = fullDex();

const SPECIES = ["Venusaur", "Charizard", "Blastoise", "Pikachu", "Snorlax", "Gengar"];
const request = (names: string[]) =>
  "|request|" +
  JSON.stringify({
    teamPreview: true,
    maxChosenTeamSize: 4,
    side: {
      name: "Alice",
      id: "p1",
      pokemon: names.map((n, i) => ({
        ident: `p1: ${n}`,
        details: `${n}, L50, F`,
        condition: "150/150",
        active: i < 2,
        stats: { atk: 1, def: 2, spa: 3, spd: 4, spe: 5 },
        moves: ["tackle", "protect"],
        baseAbility: "overgrow",
        item: "leftovers",
        ability: "overgrow",
      })),
    },
  });
const HEADER = [
  "|gametype|doubles",
  "|player|p1|Alice||",
  "|player|p2|Bob||",
  "|tier|[Gen 9 Champions] VGC 2026 Reg M-C",
  "|clearpoke",
  ...SPECIES.map((n) => `|poke|p1|${n}, L50, F|`),
  ...SPECIES.map((n) => `|poke|p2|${n}, L50, M|`),
  "|teampreview|4",
  request(SPECIES),
  "|teamsize|p1|4",
  "|teamsize|p2|4",
  "|start",
  "|switch|p1a: Venusaur|Venusaur, L50, F|150/150",
  "|switch|p1b: Charizard|Charizard, L50, F|150/150",
  "|switch|p2a: Venusaur|Venusaur, L50, M|100/100",
  "|switch|p2b: Charizard|Charizard, L50, F|100/100",
  "|turn|1",
];
const SETUP = [
  "|-boost|p2a: Venusaur|atk|2",
  "|-start|p2a: Venusaur|Substitute",
  "|-start|p2a: Venusaur|move: Leech Seed",
  "|-start|p2a: Venusaur|Encore",
  "|-endability|p2a: Venusaur",
];
const incoming = "|switch|p2a: Blastoise|Blastoise, L50, M|100/100";

function incomingMon(cause: string | null, dexArg: unknown = dex, setup: string[] = SETUP) {
  const tag = cause === null ? "" : `|[from] ${cause}`;
  const run = api.interpretChannel("p1", [...HEADER, ...setup, incoming + tag], dexArg);
  const state = run.frames.at(-1)!.state;
  return {
    run,
    state,
    receiver: state.opponent.roster.find((m: any) => m.species === "blastoise"),
    sender: state.opponent.roster.find((m: any) => m.name === "Venusaur"),
  };
}

describe("a Baton Pass or Shed Tail switch", () => {
  it("Baton Pass keeps the volatiles the engine does not block, the stages and the suppressed ability", () => {
    const { receiver, sender, run } = incomingMon("Baton Pass");
    expect(run.flags).toEqual([]);
    expect(receiver.volatiles).toEqual(["leechseed", "substitute"]);
    expect(receiver.boosts.atk).toBe(2);
    expect(receiver.ability_suppressed).toBe(true);
    expect(sender.volatiles).toEqual([]);
    expect(sender.ability_suppressed).toBe(false);
  });

  it("Shed Tail passes the Substitute alone, no stages and no suppression", () => {
    const { receiver, run } = incomingMon("Shed Tail");
    expect(run.flags).toEqual([]);
    expect(receiver.volatiles).toEqual(["substitute"]);
    expect(receiver.boosts.atk).toBe(0);
    expect(receiver.ability_suppressed).toBe(false);
  });

  it("a plain switch passes nothing", () => {
    const { receiver } = incomingMon(null);
    expect(receiver.volatiles).toEqual([]);
    expect(receiver.ability_suppressed).toBe(false);
  });

  it("a pass from a Pokemon with none passes none, and a blocked suppression is not carried", () => {
    const { receiver } = incomingMon("Baton Pass", dex, []);
    expect(receiver.volatiles).toEqual([]);
    const blocked = { ...dex, volatile_copy: { ...dex.volatile_copy, not_copied: [...dex.volatile_copy.not_copied, "gastroacid"] } };
    expect(incomingMon("Baton Pass", blocked).receiver.ability_suppressed).toBe(false);
  });

  it("without a table the pass line is kept and flagged, not guessed", () => {
    const { run } = incomingMon("Baton Pass", { species: dex.species, terrains: dex.terrains });
    expect(run.flags.map((f: { reason: string }) => f.reason)).toEqual([
      "no volatile-copy table for a Baton Pass or Shed Tail switch",
    ]);
  });

  it("the package wrapper supplies the committed table when the caller gives none", () => {
    const run = wrapped("p1", [...HEADER, ...SETUP, incoming + "|[from] Baton Pass"], { species: dex.species });
    expect(run.flags).toEqual([]);
    const receiver = run.frames.at(-1)!.state.opponent.roster.find((m) => m.species === "blastoise")!;
    expect(receiver.volatiles).toEqual(["leechseed", "substitute"]);
    const own = wrapped("p1", [...HEADER], { volatile_copy: { not_copied: [], shed_tail_copies: [] } });
    expect(own.flags).toEqual([]);
  });

  it("the default table is the only committed one, and a dex is completed only when it has none", () => {
    const dir = mkdtempSync(join(tmpdir(), "volatile-copy-dir-"));
    expect(onlyVolatileCopy(dir)).toBeNull();
    const table = { not_copied: ["a"], shed_tail_copies: ["b"] };
    writeFileSync(join(dir, "volatile-copy.one.json"), JSON.stringify(table));
    writeFileSync(join(dir, "notes.txt"), "x");
    expect(onlyVolatileCopy(dir)).toEqual(table);
    writeFileSync(join(dir, "volatile-copy.two.json"), JSON.stringify(table));
    expect(onlyVolatileCopy(dir)).toBeNull();
    expect(applyVolatileCopy(undefined, table)).toEqual({ volatile_copy: table });
    expect(applyVolatileCopy({ terrains: [] }, null)).toEqual({ terrains: [] });
    expect(applyVolatileCopy(undefined, null)).toBeUndefined();
    const own = { volatile_copy: { not_copied: [], shed_tail_copies: [] } };
    expect(applyVolatileCopy(own, table)).toBe(own);
  });

  it("the table of an unknown format is null", () => {
    expect(volatileCopyDex("gen9nothing")).toBeNull();
    expect(volatileCopyDex("gen9championsvgc2026regmc")?.shed_tail_copies).toEqual(["substitute"]);
  });

  it("the committed baton-pass battle passes a Substitute on", () => {
    const fixture = fixtures().find((f) => f.id === "baton-pass-volatiles/battle/p1")!;
    const run = api.interpretChannel(fixture.player, fixture.lines, dex);
    expect(run.flags).toEqual([]);
    const index = fixture.lines.findIndex((l) => l.startsWith("|switch|") && l.includes("Baton Pass"));
    const state = run.frames.find((f) => f.line === fixture.lines[index])!.state;
    const mon = state.own.roster.concat(state.opponent.roster).find((m: any) => m.active_slot === "b" && m.name === "Charizard" && m.volatiles.length > 0);
    expect(mon.volatiles).toEqual(["substitute"]);
  });
});
