import { readFileSync } from "node:fs";
import { basename, join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  TEAM_VERDICTS,
  type TeamVerdictLog,
  TEAMS_DIR,
  teamFixtureFiles,
  teamFixtureManifest,
  teamVerdicts,
} from "../src/golden.ts";
import {
  exportPaste,
  importPaste,
  loadEngine,
  resolveFormat,
  TeamImportError,
  validateTeam,
} from "../src/index.ts";

const engine = loadEngine();
const paste = (file: string): string => readFileSync(join(TEAMS_DIR, file), "utf8");

describe("PokePaste import and export", () => {
  it("round-trip the legal fixture byte for byte", () => {
    const text = paste("legal-synthetic.txt");
    expect(exportPaste(engine, importPaste(engine, text))).toBe(text);
  });

  // Champions Stat Points travel in Showdown's `EVs:` line (cap 32 each).
  it("keep Champions Stat Points, nickname and gender", () => {
    const [venusaur, charizard] = importPaste(engine, paste("legal-synthetic.txt"));
    expect(venusaur?.evs).toEqual({ hp: 32, atk: 0, def: 2, spa: 32, spd: 0, spe: 0 });
    expect(charizard).toMatchObject({ name: "Ember", species: "Charizard", gender: "F" });
  });

  const rejected = (text: string): TeamImportError => {
    try {
      importPaste(engine, text);
    } catch (error) {
      if (error instanceof TeamImportError) return error;
      throw error;
    }
    throw new Error("import was accepted");
  };

  it("refuse a line the pinned engine would silently drop", () => {
    expect(rejected("Venusaur\nSPs: 32 HP\n").lines).toEqual(["SPs: 32 HP"]);
  });

  // Each Pokémon is checked against its own export, case-sensitively, so a
  // lost line cannot hide behind the same text in another set.
  it("refuse a lost line that another Pokémon's export happens to contain", () => {
    const sets = paste("legal-synthetic.txt").split(/\n\s*\n/);
    const lowered = [sets[0]?.replace("EVs: 32 HP / 2 Def / 32 SpA", "evs: 2 hp / 32 spa / 32 spe"), ...sets.slice(1)];
    expect(rejected(lowered.join("\n\n")).lines).toEqual(["evs: 2 hp / 32 spa / 32 spe"]);
    const overwritten = [sets[0]?.replace("Level: 50", "EVs: 2 HP / 32 SpA / 32 Spe\nLevel: 50"), ...sets.slice(1)];
    expect(rejected(overwritten.join("\n\n")).lines).toEqual(["EVs: 2 HP / 32 SpA / 32 Spe"]);
  });

  it("refuse a line the engine parsed but exports differently", () => {
    expect(rejected("Venusaur\nAbility: Overgrow\nLevel: 100\n- Protect\n").lines).toEqual(["Level: 100"]);
  });

  it("refuse text outside a Pokémon block, like a teambuilder header", () => {
    expect(rejected(`=== [gen9] Team ===\n\n${paste("legal-synthetic.txt")}`).message).toMatch(
      /7 blocks but the pinned engine read 6 Pokémon/,
    );
  });

  it("refuse a paste with no Pokémon", () => {
    expect(() => importPaste(engine, "")).toThrow(/no Pokémon found/);
  });
});

describe("team fixtures", () => {
  const manifest = teamFixtureManifest();

  it("live in the directory named after their format", () => {
    expect(basename(TEAMS_DIR)).toBe(manifest.format_id);
  });

  it("list exactly the paste files on disk, with provenance", () => {
    expect(Object.keys(manifest.fixtures).sort()).toEqual(teamFixtureFiles());
    expect(manifest.provenance).toMatch(/^synthetic: /);
    expect(manifest.license).toBe("CC0-1.0");
  });

  it("get the verdict and reason the manifest expects from the pinned validator", () => {
    const format = resolveFormat(engine, manifest.format_id);
    for (const [file, { expect: expected, reason }] of Object.entries(manifest.fixtures)) {
      const verdict = validateTeam(engine, format, importPaste(engine, paste(file)));
      expect(verdict.valid, file).toBe(expected === "legal");
      if (expected === "legal") expect(verdict.problems, file).toEqual([]);
      else expect(verdict.problems[0], file).toContain(reason);
    }
  });

  // A Showdown pin bump that changes a verdict or its wording fails here;
  // re-capture with scripts/capture-golden.ts and review the diff.
  it("match the committed validator log", () => {
    const committed = JSON.parse(paste(TEAM_VERDICTS)) as TeamVerdictLog;
    expect(committed).toEqual(teamVerdicts(engine));
  });
});
