import { readFileSync } from "node:fs";
import { join } from "node:path";
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

  it("refuse a line the pinned engine would silently drop", () => {
    expect(() => importPaste(engine, "Venusaur\nSPs: 32 HP\n")).toThrow(TeamImportError);
    expect(() => importPaste(engine, "Venusaur\nSPs: 32 HP\n")).toThrow(/ignored: SPs: 32 HP/);
  });

  it("refuse a paste with no Pokémon", () => {
    expect(() => importPaste(engine, "")).toThrow(/no Pokémon found/);
  });
});

describe("team fixtures", () => {
  const manifest = teamFixtureManifest();

  it("list exactly the paste files on disk, with provenance", () => {
    expect(Object.keys(manifest.fixtures).sort()).toEqual(teamFixtureFiles());
    expect(manifest.provenance).toMatch(/^synthetic: /);
    expect(manifest.license).toBe("CC0-1.0");
  });

  it("get the verdict the manifest expects from the pinned validator", () => {
    const format = resolveFormat(engine, manifest.format_id);
    for (const [file, { expect: expected }] of Object.entries(manifest.fixtures)) {
      const verdict = validateTeam(engine, format, importPaste(engine, paste(file)));
      expect(verdict.valid, file).toBe(expected === "legal");
      expect(verdict.problems.length === 0, file).toBe(verdict.valid);
    }
  });

  // A Showdown pin bump that changes a verdict or its wording fails here;
  // re-capture with scripts/capture-golden.ts and review the diff.
  it("match the committed validator log", () => {
    const committed = JSON.parse(paste(TEAM_VERDICTS)) as TeamVerdictLog;
    expect(committed).toEqual(teamVerdicts(engine));
  });
});
