import { chmodSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, describe, expect, it } from "vitest";
import { loadEngine } from "../src/engine.ts";
import { TEAMS_DIR, teamFixtureFiles } from "../src/golden.ts";
import { loadRulesSnapshot } from "../src/rules-load.ts";
import {
  PasteDirectorySource,
  privateDataRoot,
  type PrivateTeamEntry,
  type PrivateTeamSource,
  teamFiles,
} from "../src/private-team-source.ts";
import {
  buildReport,
  countReport,
  gapMatchCounts,
  gapsNaming,
  InventoryError,
  type InventoryReport,
  type InventoryTerminal,
  inventoryTeam,
  parseInventoryArgs,
  runInventory,
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

/** A source that holds its teams in memory: the port assumes no file system. */
function memorySource(entries: PrivateTeamEntry[]): PrivateTeamSource {
  return {
    async *entries() {
      yield* entries;
    },
  };
}
const dirSource = (files: Record<string, string>): PasteDirectorySource => new PasteDirectorySource(privateDir(files));

/** Every file and directory under `dir`, relative, sorted. */
function tree(dir: string): string[] {
  return readdirSync(dir, { recursive: true }).map(String).sort();
}

const PINNED_FLAGS: Record<string, string[]> = {
  "illegal-five-pokemon.txt": ["species:Charizard=mega-request-shape"],
  "illegal-item-clause.txt": ["species:Charizard=mega-request-shape"],
  "illegal-not-in-roster.txt": ["species:Charizard=mega-request-shape"],
  "illegal-restricted-legendary.txt": ["species:Charizard=mega-request-shape"],
  "illegal-species-clause.txt": ["species:Charizard=mega-request-shape"],
  "illegal-stat-points-over-32.txt": ["species:Charizard=mega-request-shape"],
  "illegal-stat-points-over-total.txt": ["species:Charizard=mega-request-shape"],
  "illegal-unlearnable-move.txt": ["species:Charizard=mega-request-shape"],
  "illegal-zero-stat-points.txt": ["species:Charizard=mega-request-shape"],
  // "Trick" is a false positive: the trick-room gap prose says "Trick Room".
  "legal-opponent.txt": ["move:Trick=trick-room"],
  "legal-synthetic.txt": ["species:Charizard=mega-request-shape"],
};
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
      base_species: "Venusaur",
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

  it("does not call an unrelated item's holder a Mega user", () => {
    const team = inventoryTeam(engine, snapshot, FORMAT, "n.txt", fixture("legal-synthetic.txt").replace("Charcoal", "Venusaurite"));
    expect(team.pokemon[1]?.mega).toBeNull();
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

describe("real gap records", () => {
  // Pins what the committed gap records flag on the committed fixture teams, so
  // editing a record in rules/gaps/ shows up here. Advisory name matches only.
  it("flag the fixture teams as pinned", () => {
    const flags = Object.fromEntries(
      teamFixtureFiles().map((file) => {
        const team = inventoryTeam(engine, snapshot, FORMAT, file, fixture(file));
        return [file, team.quarantined.map((f) => `${f.kind}:${f.name}=${f.gaps.map((g) => g.gap_id).join("+")}`)];
      }),
    );
    expect(flags).toEqual(PINNED_FLAGS);
  });

  it("list every gap record as format-wide, flagged or not", async () => {
    const report = await buildReport({ engine, source: dirSource({ "a.txt": fixture("legal-synthetic.txt") }), snapshot });
    expect(report.flags_are_advisory).toBe(true);
    expect(report.flags_note).toMatch(/not a complete list/);
    expect(report.format_wide_gaps.map((g) => g.gap_id)).toEqual(Object.keys(snapshot.gaps).sort());
    expect(report.format_wide_gaps[0]).toEqual({
      gap_id: expect.any(String),
      status: expect.any(String),
      area: expect.any(String),
    });
  });

  it("match a forme by its base species and read a pasted Mega forme through it", () => {
    const withGap = structuredClone(snapshot);
    withGap.gaps["x-venusaur"] = { area: "Venusaur", status: "unverified", description: "d", evidence: [] };
    // The engine's import turns a pasted X-Mega back into X, so present the Mega forme to the lookup.
    const realMod = engine.sim.Dex.mod.bind(engine.sim.Dex);
    const dexMod: typeof engine.sim.Dex.mod = (mod) => {
      const dex = realMod(mod);
      return {
        items: dex.items,
        species: {
          get: (name) => (name === "Gengar" ? { exists: false, name, baseSpecies: name } : name === "Venusaur" ? { exists: true, name: "Venusaur-Mega", baseSpecies: "Venusaur" } : dex.species.get(name)),
        },
      };
    };
    const asMega = { ...engine, sim: { ...engine.sim, Dex: { ...engine.sim.Dex, mod: dexMod } } };
    const team = inventoryTeam(asMega, withGap, FORMAT, "v.txt", megaPaste);
    expect(team.pokemon[0]).toMatchObject({ species: "Venusaur", base_species: "Venusaur", mega: "Venusaur-Mega" });
    expect(team.pokemon.at(-1)?.base_species).toBe("Gengar");
    expect(team.quarantined.find((f) => f.name === "Venusaur")?.gaps[0]?.gap_id).toBe("x-venusaur");
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
});

describe("PasteDirectorySource", () => {
  const collect = async (source: PrivateTeamSource): Promise<PrivateTeamEntry[]> => {
    const out: PrivateTeamEntry[] = [];
    for await (const entry of source.entries()) out.push(entry);
    return out;
  };

  it("yields each *.txt as a base name and its text, sorted", async () => {
    const entries = await collect(dirSource({ "b.txt": "two", "a.txt": "one", "skip.md": "x" }));
    expect(entries).toEqual([
      { name: "a.txt", paste: "one" },
      { name: "b.txt", paste: "two" },
    ]);
  });

  it("keeps the private-data check: nothing is yielded from outside", async () => {
    await expect(collect(new PasteDirectorySource(TEAMS_DIR))).rejects.toThrow(/outside a private-data/);
  });

  it("names the file, not its content, when it cannot be read", async () => {
    const teams = privateDir({ "a.txt": "SECRET Venusaur" });
    chmodSync(join(teams, "a.txt"), 0o000);
    try {
      if (process.getuid?.() === 0) return; // root reads anything
      const failure = await collect(new PasteDirectorySource(teams)).catch((e: unknown) => e);
      expect(failure).toBeInstanceOf(InventoryError);
      expect(String((failure as Error).message)).toBe("a.txt cannot be read");
    } finally {
      chmodSync(join(teams, "a.txt"), 0o600);
    }
  });
});

describe("inventory over the port", () => {
  const files = {
    "legal.txt": fixture("legal-synthetic.txt"),
    "mega.txt": megaPaste,
    "illegal.txt": fixture("illegal-item-clause.txt"),
    "broken.txt": "not a team",
  };

  it("works on a fake in-memory source with no file system", async () => {
    const source = memorySource([
      { name: "one", paste: fixture("legal-synthetic.txt") },
      { name: "two", paste: "not a team" },
    ]);
    const report = await buildReport({ engine, source, snapshot });
    expect(report.teams.map((t) => [t.file, t.status])).toEqual([
      ["one", "legal"],
      ["two", "import_failed"],
    ]);
  });

  it("reads a directory source as before: same report as the pre-port reader", async () => {
    const report = await buildReport({ engine, source: dirSource(files), snapshot });
    expect(report).toMatchObject({ data_class: "private", format_id: FORMAT, showdown_commit: engine.commit });
    expect(report.teams.map((t) => t.file)).toEqual(["broken.txt", "illegal.txt", "legal.txt", "mega.txt"]);
    expect(countReport(report)).toEqual({ teams: 4, legal: 2, illegal: 1, import_failed: 1, quarantined_features: 4 });
  });

  it("counts gap-name matches per gap, once per team", async () => {
    const report = await buildReport({ engine, source: dirSource(files), snapshot });
    expect(gapMatchCounts(report)).toEqual({ "mega-evolution": 1, "mega-request-shape": 3 });
  });

  it("refuses a source that holds no teams", async () => {
    await expect(buildReport({ engine, source: memorySource([]), snapshot })).rejects.toThrow(/no teams/);
  });

  it("needs --format when several formats are supported, and names the label on an unexpected failure", async () => {
    const source = memorySource([{ name: "a", paste: fixture("legal-synthetic.txt") }]);
    const two = structuredClone(snapshot);
    two.formats["other"] = structuredClone(two.formats[FORMAT] as object) as (typeof two.formats)[string];
    await expect(buildReport({ engine, source, snapshot: two })).rejects.toThrow(/pass --format/);
    const broken = { ...engine, sim: { ...engine.sim, Teams: { ...engine.sim.Teams, import: () => { throw new Error("SECRET Venusaur"); } } } };
    const failure = buildReport({ engine: broken, source, snapshot, formatId: FORMAT });
    await expect(failure).rejects.toThrow(/unexpected Error on a$/);
    await expect(failure).rejects.not.toThrow(/SECRET/);
  });

  it("loads the rules snapshot itself when none is passed", async () => {
    const report = await buildReport({ engine, source: memorySource([{ name: "a", paste: fixture("legal-synthetic.txt") }]) });
    expect(report.teams).toHaveLength(1);
  });

  it("names no non-Error failure", async () => {
    const broken = { ...engine, sim: { ...engine.sim, Teams: { ...engine.sim.Teams, import: () => { throw "SECRET"; } } } };
    await expect(
      buildReport({ engine: broken, source: memorySource([{ name: "a", paste: "x" }]), snapshot }),
    ).rejects.toThrow(/unexpected error on a$/);
  });
});

describe("runInventory", () => {
  const files = { "legal.txt": fixture("legal-synthetic.txt"), "broken.txt": "not a team" };
  const terminalOf = (isTty: boolean, sink: string[]): InventoryTerminal => ({
    stdoutIsTerminal: isTty,
    writeDetail: (text) => void sink.push(text),
  });

  it("sends the detail to the terminal only and returns counts, with no new file", async () => {
    const teams = privateDir(files);
    const before = tree(scratch);
    const sink: string[] = [];
    const summary = await runInventory({ engine, source: new PasteDirectorySource(teams), snapshot }, terminalOf(true, sink));
    expect(tree(scratch)).toEqual(before);
    expect(summary.counts).toEqual({ teams: 2, legal: 1, illegal: 0, import_failed: 1, quarantined_features: 1 });
    expect(summary.gap_matches).toEqual({ "mega-request-shape": 1 });
    expect(JSON.stringify(summary)).not.toMatch(/Venusaur|Leftovers|legal\.txt/);
    expect(sink).toHaveLength(1);
    const report = JSON.parse(sink[0] ?? "") as InventoryReport;
    expect(report.teams.map((t) => t.file)).toEqual(["broken.txt", "legal.txt"]);
    expect(report.teams[1]?.pokemon[0]?.species).toBe("Venusaur");
  });

  it("refuses when stdout is not a terminal, before reading any team or sending any detail", async () => {
    let read = false;
    const source: PrivateTeamSource = {
      // eslint-disable-next-line require-yield -- must not be reached
      async *entries() {
        read = true;
      },
    };
    const sink: string[] = [];
    await expect(runInventory({ engine, source, snapshot }, terminalOf(false, sink))).rejects.toThrow(/not a terminal/);
    expect(read).toBe(false);
    expect(sink).toEqual([]);
  });

  it("canary: a team's text is in the detail and nowhere in a refusal or an error", async () => {
    const canary = fixture("legal-synthetic.txt").replace("Venusaur", "Canarymon");
    const sink: string[] = [];
    const failures: string[] = [];
    // an unknown species is an import failure, kept in the private detail
    await runInventory({ engine, source: memorySource([{ name: "c", paste: canary }]), snapshot }, terminalOf(true, sink));
    expect(sink.join("")).toContain("Canarymon");
    for (const attempt of [
      () => runInventory({ engine, source: memorySource([{ name: "c", paste: canary }]), snapshot }, terminalOf(false, [])),
      () => runInventory({ engine, source: memorySource([{ name: "c", paste: canary }]), snapshot, formatId: "gen9ou" }, terminalOf(true, [])),
      () => runInventory({ engine, source: new PasteDirectorySource(TEAMS_DIR), snapshot }, terminalOf(true, [])),
    ]) {
      failures.push(String(await attempt().then(() => "", (e: unknown) => (e as Error).message)));
    }
    expect(failures.every((m) => m !== "")).toBe(true);
    expect(failures.join("\n")).not.toMatch(/Canarymon|Venusaur|Leftovers/);
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

  it("refuses when stdout is not a terminal, writes no file and prints no team text", () => {
    const teams = privateDir({ "a.txt": fixture("legal-synthetic.txt") });
    const before = tree(scratch);
    const result = run(["--teams-dir", teams]);
    expect(result.status).toBe(1);
    expect(String(result.stdout)).toBe("");
    expect(String(result.stderr)).toBe("refusing: stdout is not a terminal; run it in an interactive terminal\n");
    expect(tree(scratch)).toEqual(before);
  });

  it("never reads a directory outside private-data either (option or environment)", () => {
    for (const result of [run(["--teams-dir", TEAMS_DIR]), run([], { ARENEON_TEAMS_DIR: TEAMS_DIR })]) {
      expect(result.status).toBe(1);
      expect(String(result.stdout)).toBe("");
      expect(String(result.stderr)).not.toMatch(/Venusaur/);
    }
  });

  it("exits 2 with the usage on a bad option", () => {
    const result = run(["--nope", "x"]);
    expect(result.status).toBe(2);
    expect(String(result.stderr)).toMatch(/^usage/);
  });
});
