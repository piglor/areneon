import fs, { chmodSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { syncBuiltinESMExports } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, describe, expect, it, vi } from "vitest";
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
  terminalRefusal,
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
        abilities: dex.abilities,
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

/** Every way this module's code could emit data, observed while `fn` runs. */
const FS_WRITERS = [
  "writeFileSync", "writeSync", "appendFileSync", "createWriteStream", "mkdirSync", "renameSync",
  "copyFileSync", "cpSync", "rmSync", "unlinkSync", "symlinkSync", "truncateSync", "linkSync",
] as const;
const FS_PROMISE_WRITERS = ["writeFile", "appendFile", "mkdir", "rename", "copyFile", "cp", "rm", "unlink", "open"] as const;

async function observe<T>(fn: () => Promise<T>): Promise<{ value: T; calls: string[] }> {
  const calls: string[] = [];
  const spies: { mockRestore(): void }[] = [];
  const record = (name: string) => (): never | boolean => {
    calls.push(name);
    return true;
  };
  for (const name of FS_WRITERS) {
    const real = fs[name] as (...args: unknown[]) => unknown;
    spies.push(vi.spyOn(fs, name as "writeFileSync").mockImplementation(((...args: unknown[]) => { calls.push(`fs.${name}`); return real(...args); }) as never));
  }
  for (const name of FS_PROMISE_WRITERS) {
    const real = fs.promises[name] as (...args: unknown[]) => unknown;
    spies.push(vi.spyOn(fs.promises, name as "writeFile").mockImplementation(((...args: unknown[]) => { calls.push(`fs.promises.${name}`); return real(...args); }) as never));
  }
  const realOpen = fs.openSync as (...args: unknown[]) => number;
  spies.push(vi.spyOn(fs, "openSync").mockImplementation(((...args: unknown[]) => {
    const flags = args[1];
    if (flags !== undefined && flags !== "r" && flags !== 0) calls.push("fs.openSync(write)");
    return realOpen(...args);
  }) as never));
  for (const name of ["log", "info", "warn", "error", "debug"] as const) {
    spies.push(vi.spyOn(console, name).mockImplementation(record(`console.${name}`)));
  }
  spies.push(vi.spyOn(process.stdout, "write").mockImplementation(record("stdout.write")));
  spies.push(vi.spyOn(process.stderr, "write").mockImplementation(record("stderr.write")));
  syncBuiltinESMExports();
  try {
    return { value: await fn(), calls };
  } finally {
    for (const spy of spies.reverse()) spy.mockRestore();
    syncBuiltinESMExports();
  }
}

describe("runInventory", () => {
  const files = { "legal.txt": fixture("legal-synthetic.txt"), "broken.txt": "not a team" };
  interface FakeTerminal extends InventoryTerminal {
    writes: string[];
    opened: number;
    closed: number;
  }
  const terminalOf = (refusal: string | null, tty = true): FakeTerminal => {
    const t: FakeTerminal = {
      writes: [],
      opened: 0,
      closed: 0,
      refusal: () => refusal,
      openDetail() {
        if (!tty) throw new InventoryError("refusing: no controlling terminal to show the detail on");
        t.opened++;
        return { write: (text) => void t.writes.push(text), close: () => void t.closed++ };
      },
    };
    return t;
  };

  it("sends the detail to the terminal only and returns counts; no write, print or file of any other kind", async () => {
    const teams = privateDir(files);
    const before = tree(scratch);
    const terminal = terminalOf(null);
    const { value: summary, calls } = await observe(() =>
      runInventory({ engine, source: new PasteDirectorySource(teams), snapshot }, terminal),
    );
    expect(calls).toEqual([]);
    expect(tree(scratch)).toEqual(before);
    expect(summary.counts).toEqual({ teams: 2, legal: 1, illegal: 0, import_failed: 1, quarantined_features: 1 });
    expect(summary.gap_matches).toEqual({ "mega-request-shape": 1 });
    expect(JSON.stringify(summary)).not.toMatch(/Venusaur|Leftovers|legal\.txt/);
    expect(terminal.writes).toHaveLength(1);
    expect([terminal.opened, terminal.closed]).toEqual([1, 1]);
    const report = JSON.parse(terminal.writes[0] ?? "") as InventoryReport;
    expect(report.teams.map((t) => t.file)).toEqual(["broken.txt", "legal.txt"]);
    expect(report.teams[1]?.pokemon[0]?.species).toBe("Venusaur");
  });

  describe("negative controls: the observer sees what a regression would do", () => {
    const run = (): Parameters<typeof runInventory>[0] => ({
      engine,
      source: memorySource([{ name: "a", paste: fixture("legal-synthetic.txt") }]),
      snapshot,
    });
    const mutants: Record<string, (r: InventoryReport) => void> = {
      "console.log of the report": (r) => console.log(JSON.stringify(r)),
      "console.error of the report": (r) => console.error(JSON.stringify(r)),
      "stdout.write": (r) => process.stdout.write(JSON.stringify(r)),
      "stderr.write": (r) => process.stderr.write(JSON.stringify(r)),
      "a file in the working directory": (r) => fs.writeFileSync(join(scratch, "leak-cwd.json"), JSON.stringify(r)),
      "a file in /tmp": (r) => fs.appendFileSync(join(scratch, "leak-tmp.json"), JSON.stringify(r)),
      "a write stream": (r) => fs.createWriteStream(join(scratch, "leak-stream.json")).end(JSON.stringify(r)),
      "a write on an opened descriptor": (r) => {
        const fd = fs.openSync(join(scratch, "leak-fd.json"), "w");
        fs.writeSync(fd, JSON.stringify(r));
        fs.closeSync(fd);
      },
      "a directory": () => fs.mkdirSync(join(scratch, "leak-dir")),
      "a rename": () => { fs.writeFileSync(join(scratch, "leak-r1"), "x"); fs.renameSync(join(scratch, "leak-r1"), join(scratch, "leak-r2")); },
    };
    for (const [name, mutate] of Object.entries(mutants)) {
      it(`catches ${name}`, async () => {
        const { calls } = await observe(async () => {
          const summary = await runInventory(run(), terminalOf(null));
          // The mutant: the same run, plus one extra emission of the full report.
          mutate(await buildReport(run()));
          return summary;
        });
        expect(calls.length).toBeGreaterThan(0);
      });
    }

    it("catches a leak through the promises API", async () => {
      const { calls } = await observe(async () => {
        await fs.promises.writeFile(join(scratch, "leak-p.json"), "x");
      });
      expect(calls).toEqual(["fs.promises.writeFile"]);
    });
  });

  it("refuses, before opening the terminal or reading any team, when the session is not interactive", async () => {
    let read = false;
    const source: PrivateTeamSource = {
      // eslint-disable-next-line require-yield -- must not be reached
      async *entries() {
        read = true;
      },
    };
    const terminal = terminalOf("refusing: nope");
    await expect(runInventory({ engine, source, snapshot }, terminal)).rejects.toThrow(/refusing: nope/);
    expect([read, terminal.opened, terminal.writes]).toEqual([false, 0, []]);
  });

  it("refuses before reading any team when there is no terminal device to open", async () => {
    let read = false;
    const source: PrivateTeamSource = {
      // eslint-disable-next-line require-yield -- must not be reached
      async *entries() {
        read = true;
      },
    };
    await expect(runInventory({ engine, source, snapshot }, terminalOf(null, false))).rejects.toThrow(/no controlling terminal/);
    expect(read).toBe(false);
  });

  it("closes the terminal even when the build fails", async () => {
    const terminal = terminalOf(null);
    await expect(runInventory({ engine, source: memorySource([]), snapshot }, terminal)).rejects.toThrow(/no teams/);
    expect([terminal.opened, terminal.closed, terminal.writes]).toEqual([1, 1, []]);
  });

  it("canary: a team's text is in the detail and nowhere in a refusal or an error", async () => {
    const canary = fixture("legal-synthetic.txt").replace("Venusaur", "Canarymon");
    const source = (): PrivateTeamSource => memorySource([{ name: "c", paste: canary }]);
    const ok = terminalOf(null);
    await runInventory({ engine, source: source(), snapshot }, ok);
    expect(ok.writes.join("")).toContain("Canarymon");
    const failures: string[] = [];
    for (const attempt of [
      () => runInventory({ engine, source: source(), snapshot }, terminalOf("refusing: nope")),
      () => runInventory({ engine, source: source(), snapshot, formatId: "gen9ou" }, terminalOf(null)),
      () => runInventory({ engine, source: new PasteDirectorySource(TEAMS_DIR), snapshot }, terminalOf(null)),
    ]) {
      failures.push(String(await attempt().then(() => "", (e: unknown) => (e as Error).message)));
    }
    expect(failures.every((m) => m !== "")).toBe(true);
    expect(failures.join("\n")).not.toMatch(/Canarymon|Venusaur|Leftovers/);
  });
});

describe("terminalRefusal", () => {
  const all = { stdin: true, stdout: true, stderr: true };
  it("passes only an interactive terminal session without an agent variable", () => {
    expect(terminalRefusal({}, all)).toBeNull();
    expect(terminalRefusal({ CLAUDECODE: "1" }, all)).toMatch(/agent session/);
    expect(terminalRefusal({ AI_AGENT: "x" }, all)).toMatch(/agent session/);
    for (const off of ["stdin", "stdout", "stderr"] as const) {
      expect(terminalRefusal({}, { ...all, [off]: false })).toMatch(/must all be a terminal/);
    }
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
  // The agent variables are dropped unless the test sets one, so the run behaves the same in any session.
  const cleanEnv = (extra: Record<string, string> = {}): Record<string, string> =>
    Object.fromEntries(
      Object.entries({ ...process.env, ...extra }).filter(
        ([key]) => key in extra || (key !== "CLAUDECODE" && key !== "AI_AGENT"),
      ),
    ) as Record<string, string>;
  const run = (args: string[], env: Record<string, string> = {}): ReturnType<typeof spawnSync> =>
    spawnSync(process.execPath, [script, ...args], { encoding: "utf8", env: cleanEnv(env) });

  it("refuses when the stdio are pipes, writes no file and prints no team text", () => {
    const teams = privateDir({ "a.txt": fixture("legal-synthetic.txt") });
    const before = tree(scratch);
    const result = run(["--teams-dir", teams]);
    expect(result.status).toBe(1);
    expect(String(result.stdout)).toBe("");
    expect(String(result.stderr)).toBe(
      "refusing: stdin, stdout and stderr must all be a terminal; run it in an interactive terminal\n",
    );
    expect(tree(scratch)).toEqual(before);
  });

  it("exits 2 with the usage on a bad option", () => {
    const result = run(["--nope", "x"]);
    expect(result.status).toBe(2);
    expect(String(result.stderr)).toMatch(/^usage/);
  });

  // The output boundary (the /dev/tty write, the terminal decision, the one stdout
  // line) is in the script, so it is run for real under a pseudo-terminal. python3's
  // pty module needs no dependency; every CI runner has python3. Skipped, loudly in
  // the name, where python3 is missing. A pty has one transcript for the controlling
  // terminal and the three stdio, so the assertions are on content and order: the
  // detail (with the canary) once, then exactly one summary line without it.
  const hasPython = spawnSync("python3", ["-V"]).status === 0;
  const PTY_RUNNER = `
import json, os, pty, sys
mode = sys.argv[1]
argv = sys.argv[2:]
master, slave = os.openpty()
if mode == "ctty":
    pid, master = pty.fork()
    if pid == 0:
        os.execv(argv[0], argv)
else:
    pid = os.fork()
    if pid == 0:
        os.setsid()  # a new session: the pty is not its controlling terminal
        for fd in (0, 1, 2):
            os.dup2(slave, fd)
        os.execv(argv[0], argv)
    os.close(slave)
chunks = []
while True:
    try:
        data = os.read(master, 65536)
    except OSError:
        break
    if not data:
        break
    chunks.append(data)
_, status = os.waitpid(pid, 0)
print(json.dumps({"status": os.waitstatus_to_exitcode(status), "out": b"".join(chunks).decode("utf8", "replace")}))
`;
  const underPty = (
    mode: "ctty" | "no-ctty",
    args: string[],
    env: Record<string, string> = {},
  ): { status: number; out: string } => {
    const result = spawnSync("python3", ["-I", "-c", PTY_RUNNER, mode, process.execPath, script, ...args], {
      encoding: "utf8",
      env: cleanEnv(env),
      timeout: 60_000,
    });
    expect(result.status, String(result.stderr)).toBe(0);
    return JSON.parse(String(result.stdout)) as { status: number; out: string };
  };
  const canaryTeam = fixture("legal-synthetic.txt"); // Venusaur and Leftovers are the canaries
  const lines = (out: string): string[] => out.replace(/\r/g, "").split("\n").filter((l) => l !== "");

  describe.skipIf(!hasPython)("under a pseudo-terminal (needs python3)", () => {
    it("shows the detail on the terminal, then one summary line; exit 0; no file", () => {
      const teams = privateDir({ "a.txt": canaryTeam, "b.txt": "not a team" });
      const before = tree(scratch);
      const { status, out } = underPty("ctty", ["--teams-dir", teams]);
      expect(status).toBe(0);
      expect(tree(scratch)).toEqual(before);
      const detailEnd = out.lastIndexOf("}\r\n{") + 3; // the report's closing brace, then the summary
      expect(out.slice(0, detailEnd)).toContain("Leftovers");
      const summary = out.slice(detailEnd);
      expect(summary).not.toMatch(/Venusaur|Leftovers|a\.txt/);
      expect(lines(summary)).toHaveLength(1);
      expect(JSON.parse(lines(summary)[0] ?? "")).toEqual({
        teams: 2, legal: 1, illegal: 0, import_failed: 1, quarantined_features: 1,
        gap_matches: { "mega-request-shape": 1 },
      });
      expect(JSON.parse(out.slice(0, detailEnd).replace(/\r/g, ""))).toMatchObject({ data_class: "private" });
    });

    it("refuses without a controlling terminal, before reading any team and with no detail", () => {
      const teams = privateDir({ "a.txt": canaryTeam });
      const { status, out } = underPty("no-ctty", ["--teams-dir", teams]);
      expect(status).toBe(1);
      expect(lines(out)).toEqual(["refusing: no controlling terminal to show the detail on"]);
      // The same outcome for a directory that would fail the path check: the terminal decision came first.
      expect(lines(underPty("no-ctty", ["--teams-dir", TEAMS_DIR]).out)).toEqual(lines(out));
    });

    it("reaches the path check on a terminal: a directory outside private-data is refused (option or environment)", () => {
      for (const result of [underPty("ctty", ["--teams-dir", TEAMS_DIR]), underPty("ctty", [], { ARENEON_TEAMS_DIR: TEAMS_DIR })]) {
        expect(result.status).toBe(1);
        expect(lines(result.out)).toEqual(["refusing a path outside a private-data/ directory"]);
      }
    });

    it("refuses an agent session even on a terminal", () => {
      const teams = privateDir({ "a.txt": canaryTeam });
      const { status, out } = underPty("ctty", ["--teams-dir", teams], { CLAUDECODE: "1" });
      expect(status).toBe(1);
      expect(lines(out)).toEqual(["refusing: an agent session is running this; use your own terminal"]);
    });

    it("names only the error class for a failure it did not expect", () => {
      const teams = privateDir({ "a.txt": "x" });
      const { status, out } = underPty("ctty", ["--teams-dir", teams, "--format", "gen9ou"]);
      expect(status).toBe(1);
      expect(lines(out)).toEqual(["failed: UnsupportedFormatError"]);
    });
  });
});
