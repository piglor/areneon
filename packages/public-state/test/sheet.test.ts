import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { createInterpreter, interpretChannel, parseSheet, readerSha256, SheetError } from "../src/index.ts";
import { FORMAT_DIR, ROOT, fixtures, fullDex, openFixtures } from "./support.ts";

const dex = fullDex();
const python = join(ROOT, ".venv", "bin", "python");
const here = dirname(fileURLToPath(import.meta.url));
const open = openFixtures();

const SHEET = "|showteam|p2|Altaria||Altarianite|NaturalCure|Substitute,FeatherDance,HyperBeam,FireSpin|Serious||F|||50|]Delphox||Pidgeotite|Blaze|PsychUp,ThunderPunch,SunnyDay,BlastBurn|Serious||M|||50|";
const PREVIEW = [
  "|gametype|doubles",
  "|player|p1|p1||",
  "|player|p2|p2||",
  "|clearpoke",
  "|poke|p2|Altaria, L50, F|",
  "|poke|p2|Delphox, L50, M|",
  "|teampreview|2",
];

const OWN = SHEET.replace("|p2|", "|p1|");
const mon = (fields: string[]): string => fields.join("|");
const run = (lines: string[], player = "p1"): { frames: any[]; flags: { index: number; line: string; reason: string }[] } =>
  interpretChannel(player, lines, dex);
const reasons = (lines: string[]) => run(lines).flags.map((f) => f.reason);
const withSheet = (sheet: string): string[] => [...PREVIEW, sheet];

/** Replace the field at `index` of the first Pokemon of the sheet. */
const editField = (index: number, value: string): string => {
  const [head, ...others] = SHEET.split("]");
  const parts = head!.split("|");
  parts[3 + index] = value;
  return [parts.join("|"), ...others].join("]");
};

describe("the interpreter reads |showteam| into a per-side sheet", () => {
  it("keeps the sheet apart from what the battle reveals", () => {
    const state = run(withSheet(SHEET)).frames.at(-1)!.state;
    expect(state.opponent.sheet).toEqual([
      { species: "altaria", item: "altarianite", ability: "naturalcure", moves: ["substitute", "featherdance", "hyperbeam", "firespin"], nature: "serious", gender: "F", level: 50 },
      { species: "delphox", item: "pidgeotite", ability: "blaze", moves: ["psychup", "thunderpunch", "sunnyday", "blastburn"], nature: "serious", gender: "M", level: 50 },
    ]);
    expect(state.own.sheet).toBeUndefined();
    // the roster's revealed fields are untouched by the sheet
    expect(state.opponent.roster.map((m: any) => [m.item, m.ability, m.moves_revealed])).toEqual([[null, null, []], [null, null, []]]);
  });

  it("reads the own side's sheet too, and a closed channel has no sheet key", () => {
    const own = run([...PREVIEW, OWN]).frames.at(-1)!.state;
    expect(own.own.sheet).toHaveLength(2);
    for (const fixture of fixtures().slice(0, 4)) {
      const last = run(fixture.lines, fixture.player).frames.at(-1)!.state;
      expect("sheet" in last.own).toBe(false);
      expect("sheet" in last.opponent).toBe(false);
    }
  });

  it("parses a missing gender, item, nature and level", () => {
    const sheet = `|showteam|p2|${mon(["Altaria", "", "", "NaturalCure", "Substitute", "", "", "", "", "", "", ""])}]${mon(["Delphox", "", "Leftovers", "Blaze", "Psychup", "", "", "M", "", "", "", ""])}`;
    const [a, b] = run(withSheet(sheet)).frames.at(-1)!.state.opponent.sheet;
    expect(a).toEqual({ species: "altaria", item: null, ability: "naturalcure", moves: ["substitute"], nature: null, gender: null, level: 100 });
    expect(b.level).toBe(100);
  });

  it("keeps line indices, request counts and the frame of the sheet line aligned with the channel", () => {
    const lines = withSheet(SHEET);
    const result = run(lines);
    expect(result.frames).toHaveLength(lines.length + 1);
    const frame = result.frames.at(-1)!;
    expect(frame).toMatchObject({ index: lines.length - 1, line: SHEET, kind: "showteam", caption: null, flagged: null });
    expect(result.frames.at(-2)!.state.opponent.sheet).toBeUndefined();
    const flagged = run([...lines, SHEET]);
    expect(flagged.flags).toEqual([{ index: lines.length, line: SHEET, reason: "a second team sheet for one side" }]);
    expect(flagged.frames.at(-1)!.flagged).toBe("a second team sheet for one side");
  });

  it("reads the same through the incremental reader", () => {
    const reader = createInterpreter({ player: "p1", dex });
    for (const line of PREVIEW) expect(reader.feed(line).flagged).toBeNull();
    expect(reader.snapshot().opponent.sheet).toBeUndefined();
    expect(reader.feed(SHEET)).toEqual({ kind: "showteam", caption: null, flagged: null });
    expect(reader.snapshot().opponent.sheet).toEqual(run(withSheet(SHEET)).frames.at(-1)!.state.opponent.sheet);
    expect(reader.feed(SHEET).flagged).toBe("a second team sheet for one side");
    expect(reader.feed("|start").flagged).toBeNull();
    expect(reader.feed(OWN).flagged).toBe("a team sheet after |start");
    expect(reader.requestCount()).toBe(0);
    expect(reader.requests()).toEqual([]);
  });

  it("parses a bare packed team and refuses with a SheetError", () => {
    expect(parseSheet(SHEET.split("|").slice(3).join("|"))).toHaveLength(2);
    expect(() => parseSheet("a")).toThrow(SheetError);
  });

  it("leaves the interpreter file byte-identical to main's: the replay pages and ADR-018's baseline identity hash it", () => {
    expect(readerSha256()).toBe("f6687af131b7d355d5a317b6ec8ac629575898479eff4b39aad76b4496655ea8");
  });
});

describe("the interpreter fails closed on a sheet it cannot take", () => {
  it("a second sheet for a side, a sheet after |start, a side it does not know", () => {
    expect(reasons([...withSheet(SHEET), SHEET])).toEqual(["a second team sheet for one side"]);
    expect(reasons([...PREVIEW, "|start", SHEET])).toEqual(["a team sheet after |start"]);
    expect(reasons([...PREVIEW, SHEET.replace("|p2|", "|p3|")])).toEqual(["unsupported side 'p3' in showteam"]);
    expect(reasons([...PREVIEW, "|showteam|p2"])).toEqual(["showteam needs at least 2 fields"]);
  });

  it("a sheet that is not the preview's, in count or in species", () => {
    expect(reasons(withSheet(SHEET.split("]")[0]!))).toEqual(["a team sheet that is not the team preview's"]);
    const swapped = SHEET.replace("Altaria||", "Arbok||").replace("Delphox||", "Altaria||");
    expect(reasons(withSheet(swapped))).toEqual(["a team sheet that is not the team preview's"]);
    expect(reasons([...PREVIEW.slice(0, 3), SHEET])).toEqual([]);
  });

  it("an unknown field: nickname, Stat Points, IVs, shiny, extras, gender", () => {
    expect(reasons(withSheet(editField(1, "Birdy")))).toEqual(["a sheet shows no nickname"]);
    expect(reasons(withSheet(editField(6, "252,,,,,4")))).toEqual(["a sheet shows no Stat Points and no IVs"]);
    expect(reasons(withSheet(editField(8, ",,,30,30,")))).toEqual(["a sheet shows no Stat Points and no IVs"]);
    expect(reasons(withSheet(editField(9, "S")))).toEqual(["a sheet carries no shiny flag and no extra fields"]);
    expect(reasons(withSheet(editField(11, ",,,,,tera")))).toEqual(["a sheet carries no shiny flag and no extra fields"]);
    expect(reasons(withSheet(editField(7, "X")))).toEqual(["unknown sheet gender 'X'"]);
  });

  it("a malformed Pokemon", () => {
    expect(reasons(withSheet(`${SHEET}|extra`))).toEqual(["a sheet Pokemon has 12 fields"]);
    expect(reasons(withSheet(`${SHEET}]`))).toEqual(["a sheet Pokemon has 12 fields"]);
    expect(reasons(withSheet(editField(4, "")))).toEqual(["a sheet Pokemon lacks its species, ability or moves"]);
    expect(reasons(withSheet(editField(0, "")))).toEqual(["a sheet Pokemon lacks its species, ability or moves"]);
    expect(reasons(withSheet(editField(4, "Substitute,,Protect")))).toEqual(["a sheet move list has an empty move"]);
    expect(reasons(withSheet(editField(4, "Substitute,!!")))).toEqual(["a sheet name has no letters"]);
    expect(reasons(withSheet(editField(5, "Ser1ous")))).toEqual(["a sheet nature is a name"]);
    expect(reasons(withSheet(editField(10, "0")))).toEqual(["a sheet level is a number from 1 to 100"]);
    expect(reasons(withSheet(editField(10, "101")))).toEqual(["a sheet level is a number from 1 to 100"]);
    expect(reasons(withSheet(editField(10, "5x")))).toEqual(["a sheet level is a number from 1 to 100"]);
  });

  it("a list of more than 24 Pokemon", () => {
    const many = `|showteam|p1|${Array.from({ length: 25 }, () => mon(["Altaria", "", "", "NaturalCure", "Substitute", "", "", "", "", "", "", ""])).join("]")}`;
    expect(reasons([...PREVIEW, many])).toEqual(["a sheet lists between one and 24 Pokemon"]);
  });

  it("a flagged sheet leaves the state without one", () => {
    const state = run(withSheet(editField(1, "Birdy"))).frames.at(-1)!.state;
    expect(state.opponent.sheet).toBeUndefined();
  });
});

describe("the interpreter against the Python reducer on the open-regime channels", () => {
  it("has the committed goldens", () => {
    expect(open.length).toBeGreaterThanOrEqual(16);
  });

  it("reads one sheet per side, no flags, in team order", () => {
    for (const fixture of open) {
      const run_ = run(fixture.lines, fixture.player);
      expect(run_.flags, fixture.id).toEqual([]);
      const state = run_.frames.at(-1)!.state;
      expect(state.own.sheet, fixture.id).toHaveLength(6);
      expect(state.opponent.sheet, fixture.id).toHaveLength(6);
    }
  });

  describe.skipIf(!existsSync(python))("python reducer", () => {
    let cached: Record<string, Record<string, any>> | undefined;
    const reference = (): Record<string, Record<string, any>> => (cached ??= (() => {
      const jobs = open.map((f) => ({ id: f.id, player: f.player, lines: f.lines, team: f.team, cuts: [f.lines.length] }));
      const out = spawnSync(python, [join(here, "reduce_points.py")], {
        input: JSON.stringify({ engine_data: join(FORMAT_DIR, "engine-data.json"), jobs }),
        maxBuffer: 1 << 30,
        encoding: "utf8",
      });
      if (out.status !== 0) throw new Error(out.stderr);
      return JSON.parse(out.stdout) as Record<string, Record<string, any>>;
    })());

    for (const fixture of open) {
      it(`${fixture.id}: the sheets of both sides are equal`, () => {
        const mine = run(fixture.lines, fixture.player).frames.at(-1)!.state;
        const theirs = reference()[fixture.id]![String(fixture.lines.length)];
        expect(theirs.refused).toBeUndefined();
        expect(mine.own.sheet).toEqual(theirs.own.sheet);
        expect(mine.opponent.sheet).toEqual(theirs.opponent.sheet);
      });
    }
  });
});
