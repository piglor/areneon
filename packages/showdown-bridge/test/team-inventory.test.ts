import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, symlinkSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, describe, expect, it } from "vitest";
import { loadEngine } from "../src/engine.ts";
import { TEAMS_DIR } from "../src/golden.ts";
import { loadRulesSnapshot } from "../src/rules-load.ts";
import {
  buildReport,
  countReport,
  gapsNaming,
  InventoryError,
  inventoryTeam,
  parseInventoryArgs,
  privateDataRoot,
  reportPath,
  runInventory,
  teamFiles,
  writePrivate,
} from "../src/team-inventory.ts";

const engine = loadEngine();
const snapshot = loadRulesSnapshot(engine.commit);
const FORMAT = "gen9championsvgc2026regmc";
const fixture = (name: string): string => readFileSync(join(TEAMS_DIR, name), "utf8");

// Everything here is synthetic: the committed fixtures, or a copy of one.
const scratch = mkdtempSync(join(tmpdir(), "inventory-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));

let n = 0;
function privateDir(files: Record<string, string>): string {
  const root = join(scratch, `case-${n++}`, "private-data");
  const teams = join(root, "teams");
  mkdirSync(teams, { recursive: true });
  for (const [name, text] of Object.entries(files)) writeFileSync(join(teams, name), text);
  return teams;
}

const megaPaste = fixture("legal-synthetic.txt").replace("Leftovers", "Venusaurite");

describe("inventoryTeam", () => {
  it("lists a legal team's features and flags only what a gap names", () => {
    const team = inventoryTeam(engine, snapshot, FORMAT, "a.txt", fixture("legal-synthetic.txt"));
    expect(team).toMatchObject({ file: "a.txt", status: "legal", problems: [], import_error: null });
    expect(team.pokemon.map((p) => p.species)).toEqual([
      "Venusaur", "Charizard", "Blastoise", "Pikachu", "Snorlax", "Gengar",
    ]);
    expect(team.pokemon[0]).toEqual({
      species: "Venusaur",
      item: "Leftovers",
      ability: "Overgrow",
      moves: ["Giga Drain", "Sludge Bomb", "Protect", "Sleep Powder"],
      mega: null,
    });
    // mega-request-shape names Charizard (separate X and Y stones); nothing else is named.
    expect(team.quarantined).toEqual([
      { kind: "species", name: "Charizard", gaps: [{ gap_id: "mega-request-shape", status: "unverified" }] },
    ]);
  });

  it("reports the validator's verdict verbatim for an illegal team", () => {
    const team = inventoryTeam(engine, snapshot, FORMAT, "b.txt", fixture("illegal-species-clause.txt"));
    expect(team.status).toBe("illegal");
    expect(team.problems.join(" ")).toContain("Species Clause");
  });

  it("records a refused paste as import_failed, not a throw", () => {
    const team = inventoryTeam(engine, snapshot, FORMAT, "c.txt", "not a team");
    expect(team).toMatchObject({ status: "import_failed", pokemon: [], quarantined: [] });
    expect(team.import_error).not.toBeNull();
  });

  it("rethrows what is not an import refusal", () => {
    expect(() => inventoryTeam(engine, snapshot, "gen9ou", "d.txt", fixture("legal-synthetic.txt"))).toThrow(
      /not a supported/,
    );
  });

  it("flags Mega use and the gap that names Mega Evolution", () => {
    const team = inventoryTeam(engine, snapshot, FORMAT, "m.txt", megaPaste);
    expect(team.pokemon[0]?.mega).toBe("Venusaur-Mega");
    const mega = team.quarantined.find((f) => f.name === "Mega Evolution");
    expect(mega?.kind).toBe("mechanic");
    expect(mega?.gaps.map((g) => g.gap_id)).toEqual(["mega-evolution", "mega-request-shape"]);
  });

  it("takes its flags from the gap records", () => {
    const withGap = structuredClone(snapshot);
    withGap.gaps["x-protect"] = {
      area: "Protect",
      status: "unsupported",
      description: "Charcoal is odd",
      evidence: [{ kind: "engine", path: "data/moves.ts", symbols: ["sleepPowder"] }],
    };
    const team = inventoryTeam(engine, withGap, FORMAT, "p.txt", fixture("legal-synthetic.txt"));
    const flagged = Object.fromEntries(team.quarantined.map((f) => [f.name, f.gaps]));
    expect(Object.keys(flagged).sort()).toEqual(["Charcoal", "Charizard", "Protect", "Sleep Powder"]);
    expect(flagged["Protect"]).toEqual([{ gap_id: "x-protect", status: "unsupported" }]);
  });
});

describe("gapsNaming", () => {
  it("matches whole words only and ignores an empty name", () => {
    expect(gapsNaming(snapshot, { kind: "move", name: "Wish" }).map((g) => g.gap_id)).toContain("override-parity");
    expect(gapsNaming(snapshot, { kind: "move", name: "Wis" })).toEqual([]);
    expect(gapsNaming(snapshot, { kind: "item", name: " " })).toEqual([]);
  });
});

describe("paths", () => {
  it("accepts a path inside private-data and names its root", () => {
    const teams = privateDir({ "a.txt": fixture("legal-synthetic.txt") });
    expect(privateDataRoot(teams)).toBe(join(teams, ".."));
    expect(reportPath(teams)).toBe(join(teams, "..", "reports", "team-inventory.json"));
  });

  it("refuses a directory with no private-data ancestor", () => {
    expect(() => teamFiles(TEAMS_DIR)).toThrow(InventoryError);
    expect(() => privateDataRoot(join(scratch, "elsewhere"))).toThrow(/outside a private-data/);
    expect(() => privateDataRoot("/")).toThrow(InventoryError);
  });

  it("refuses a private-data link that leads out", () => {
    const outside = join(scratch, "outside");
    mkdirSync(outside);
    const base = join(scratch, "linked");
    mkdirSync(base);
    symlinkSync(outside, join(base, "private-data"));
    expect(() => teamFiles(join(base, "private-data"))).toThrow(/outside a private-data/);
  });

  it("refuses a team file that is a link out of private-data", () => {
    const teams = privateDir({});
    const target = join(scratch, "stray.txt");
    writeFileSync(target, "x");
    symlinkSync(target, join(teams, "stray.txt"));
    expect(() => teamFiles(teams)).toThrow(/stray\.txt: not a regular file/);
  });

  it("refuses an empty or unreadable teams directory", () => {
    expect(() => teamFiles(privateDir({ "notes.md": "x" }))).toThrow(/no \.txt/);
    expect(() => teamFiles(join(privateDir({}), "missing"))).toThrow(/cannot be read/);
  });

  it("writes only inside private-data, with private modes", () => {
    const teams = privateDir({});
    const file = join(teams, "..", "reports", "out.json");
    writePrivate(file, "{}");
    expect(statSync(file).mode & 0o777).toBe(0o600);
    expect(() => writePrivate(join(scratch, "leak.json"), "{}")).toThrow(/outside a private-data/);
    expect(existsSync(join(scratch, "leak.json"))).toBe(false);
    expect(() => writePrivate(join(teams, "..", "..", "leak.json"), "{}")).toThrow(InventoryError);
    expect(existsSync(join(teams, "..", "..", "leak.json"))).toBe(false);
  });

  it("refuses to write through a link that leaves private-data", () => {
    const teams = privateDir({});
    const out = join(scratch, "out-dir");
    mkdirSync(out);
    symlinkSync(out, join(teams, "..", "reports"));
    expect(() => writePrivate(join(teams, "..", "reports", "x.json"), "{}")).toThrow(/outside/);
    expect(existsSync(join(out, "x.json"))).toBe(false);
  });
});

describe("runInventory", () => {
  const files = {
    "legal.txt": fixture("legal-synthetic.txt"),
    "mega.txt": megaPaste,
    "illegal.txt": fixture("illegal-item-clause.txt"),
    "broken.txt": "not a team",
  };

  it("writes one report under private-data/reports and returns only counts and the path", () => {
    const teams = privateDir(files);
    const { path, counts } = runInventory({ engine, teamsDir: teams, snapshot });
    expect(path).toBe(join(teams, "..", "reports", "team-inventory.json"));
    expect(counts).toEqual({ teams: 4, legal: 2, illegal: 1, import_failed: 1, quarantined_features: 4 });
    expect(JSON.stringify(counts)).not.toContain("Venusaur");
    const report = JSON.parse(readFileSync(path, "utf8")) as ReturnType<typeof buildReport>;
    expect(report).toMatchObject({ data_class: "private", format_id: FORMAT, showdown_commit: engine.commit });
    expect(report.teams.map((t) => t.file)).toEqual(["broken.txt", "illegal.txt", "legal.txt", "mega.txt"]);
    expect(countReport(report)).toEqual(counts);
  });

  it("needs --format when several formats are supported, and names the file on an unexpected failure", () => {
    const teams = privateDir({ "a.txt": fixture("legal-synthetic.txt") });
    const two = structuredClone(snapshot);
    two.formats["other"] = structuredClone(two.formats[FORMAT] as object) as (typeof two.formats)[string];
    expect(() => buildReport({ engine, teamsDir: teams, snapshot: two })).toThrow(/pass --format/);
    const broken = { ...engine, sim: { ...engine.sim, Teams: { ...engine.sim.Teams, import: () => { throw new Error("SECRET Venusaur"); } } } };
    const failure = (): unknown => buildReport({ engine: broken, teamsDir: teams, snapshot, formatId: FORMAT });
    expect(failure).toThrow(/unexpected Error on a\.txt/);
    expect(failure).not.toThrow(/SECRET/);
  });

  it("loads the rules snapshot itself when none is passed", () => {
    const teams = privateDir({ "a.txt": fixture("legal-synthetic.txt") });
    expect(buildReport({ engine, teamsDir: teams }).teams).toHaveLength(1);
  });

  it("names no non-Error failure", () => {
    const teams = privateDir({ "a.txt": fixture("legal-synthetic.txt") });
    const broken = { ...engine, sim: { ...engine.sim, Teams: { ...engine.sim.Teams, import: () => { throw "SECRET"; } } } };
    expect(() => buildReport({ engine: broken, teamsDir: teams, snapshot })).toThrow(/unexpected error on a\.txt/);
  });
});

describe("parseInventoryArgs", () => {
  it("takes the option, then the environment, then the default", () => {
    expect(parseInventoryArgs([], {}, "/d")).toEqual({ teamsDir: "/d" });
    expect(parseInventoryArgs([], { ARENEON_TEAMS_DIR: "/e" }, "/d")).toEqual({ teamsDir: "/e" });
    expect(parseInventoryArgs(["--teams-dir", "/o", "--format", "f"], { ARENEON_TEAMS_DIR: "/e" }, "/d")).toEqual({
      teamsDir: "/o",
      formatId: "f",
    });
  });

  it("refuses an unknown or incomplete option", () => {
    for (const argv of [["--nope", "x"], ["--teams-dir"], ["--teams-dir", "--format"]]) {
      expect(() => parseInventoryArgs(argv, {}, "/d")).toThrow(/usage/);
    }
  });
});

describe("scripts/team-inventory.ts", () => {
  const script = fileURLToPath(new URL("../scripts/team-inventory.ts", import.meta.url));
  const run = (args: string[], env: Record<string, string> = {}): ReturnType<typeof spawnSync> =>
    spawnSync(process.execPath, [script, ...args], { encoding: "utf8", env: { ...process.env, ...env } });

  it("prints counts and the report path only", () => {
    const teams = privateDir({ "a.txt": fixture("legal-synthetic.txt") });
    const result = run(["--teams-dir", teams]);
    expect(result.status).toBe(0);
    expect(JSON.parse(String(result.stdout))).toMatchObject({ teams: 1, legal: 1, report: reportPath(teams) });
    expect(String(result.stdout)).not.toMatch(/Venusaur|Leftovers/);
  });

  it("refuses a directory outside private-data (option or environment) without echoing team text", () => {
    for (const result of [run(["--teams-dir", TEAMS_DIR]), run([], { ARENEON_TEAMS_DIR: TEAMS_DIR })]) {
      expect(result.status).toBe(1);
      expect(String(result.stderr)).toMatch(/outside a private-data/);
      expect(String(result.stderr)).not.toMatch(/Venusaur/);
    }
  });

  it("exits 2 with the usage on a bad option", () => {
    const result = run(["--nope", "x"]);
    expect(result.status).toBe(2);
    expect(String(result.stderr)).toMatch(/^usage/);
  });

  it("names only the error class for a failure it did not expect", () => {
    const result = run(["--teams-dir", privateDir({ "a.txt": "x" }), "--format", "gen9ou"]);
    expect(result.status).toBe(1);
    expect(String(result.stderr)).toBe("failed: UnsupportedFormatError\n");
  });
});
