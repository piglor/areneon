import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, afterEach, describe, expect, it } from "vitest";
import { loadEngine } from "../src/engine.ts";
import { loadRulesSnapshot } from "../src/rules-load.ts";
import { assertOwnerSession, type OwnerSession } from "../src/owner-session.ts";
import { assertPrivateRoot, writePrivateFile } from "../src/private-root.ts";
import { PasteDirectorySource } from "../src/private-team-source.ts";
import { canonicalTeamHash } from "../src/team-records.ts";
import { runInventory, terminalRefusal } from "../src/team-inventory.ts";
import { teamHash } from "../src/team-class.ts";
import {
  type CanaryWorld,
  errorSurface,
  expectNoLeak,
  findLeaks,
  makeCanaryWorld,
  nickPaste,
  observe,
  worldIsPrivate,
} from "./support/private-canary.ts";

const engine = loadEngine();
const snapshot = loadRulesSnapshot(engine.commit);
const worlds: CanaryWorld[] = [];
const need = <T>(value: T | undefined): T => {
  if (value === undefined) throw new Error("missing");
  return value;
};
const team0 = (w: CanaryWorld): CanaryWorld["teams"][number] => need(w.teams[0]);
const set0 = (w: CanaryWorld): CanaryWorld["teams"][number]["sets"][number] => need(team0(w).sets[0]);
const set1 = (w: CanaryWorld): CanaryWorld["teams"][number]["sets"][number] => need(need(w.teams[1]).sets[0]);
const marker0 = (w: CanaryWorld): string => need(w.markers[0]);
const marker1 = (w: CanaryWorld): string => need(w.markers[1]);
const world = (): CanaryWorld => {
  const w = makeCanaryWorld(engine);
  worlds.push(w);
  return w;
};
afterEach(() => {
  while (worlds.length > 0) worlds.pop()?.cleanup();
});

const terminalSession: OwnerSession = {
  env: {},
  isTerminal: { stdin: true, stdout: true, stderr: true },
  openTerminal: () => ({ write: () => undefined, close: () => undefined }),
};

/** A guarded command built from the shared pieces: the guard, the private root, the inventory. */
async function guardedCommand(w: CanaryWorld, detail: string[]): Promise<void> {
  assertOwnerSession(terminalSession);
  assertPrivateRoot(w.root);
  const summary = await runInventory(
    { engine, source: new PasteDirectorySource(w.teamsDir), snapshot },
    {
      refusal: () => terminalRefusal(terminalSession.env, terminalSession.isTerminal),
      openDetail: () => ({ write: (text) => void detail.push(text), close: () => undefined }),
    },
  );
  console.log(JSON.stringify(summary.counts));
  writePrivateFile(w.root, "summary.json", JSON.stringify(summary.counts));
}

describe("the stand-in world", () => {
  it("holds private-classed teams, marker nicknames the engine keeps, and a root the private-root rule accepts", () => {
    const w = world();
    expect(worldIsPrivate(w)).toBe(true);
    expect(w.teams).toHaveLength(2);
    expect(w.markers).toHaveLength(w.teams.reduce((n, t) => n + t.sets.length, 0));
    for (const team of w.teams) expect(team.sets.map((s) => s.name)).toEqual(expect.arrayContaining(w.markers.filter((m) => team.paste.includes(m))));
    expect(assertPrivateRoot(w.root)).toBe(w.root);
    const kinds = new Set(w.needles.map((n) => n.kind));
    for (const kind of ["marker", "species", "item", "ability", "move", "set_line", "team_hash", "paste", "file_name"]) expect(kinds).toContain(kind);
    expect(w.needles.map((n) => n.text)).toEqual(expect.arrayContaining([teamHash(team0(w).sets), canonicalTeamHash(team0(w).sets)]));
    expect(new Set(w.markers).size).toBe(w.markers.length);
    expect(w.markers.some((m) => world().markers.includes(m))).toBe(false);
  });

  it("refuses more teams than the fixture pool holds", () => {
    expect(() => makeCanaryWorld(engine, 10_000)).toThrow(/not enough stand-in teams/);
  });

  it("nicknames every Pokemon block", () => {
    expect(nickPaste("A @ x\nb\n\nC @ y\nd", (i) => `N${i}`)).toBe("N0 (A) @ x\nb\n\nN1 (C) @ y\nd");
  });
});

describe("a clean guarded command", () => {
  it("leaks nothing: counts on stdout, a summary in the private root, the detail only on the terminal", async () => {
    const w = world();
    const detail: string[] = [];
    const observation = await observe(w, (x) => guardedCommand(x, detail));
    expect(observation.thrown).toBeNull();
    expect(JSON.parse(observation.stdout)).toMatchObject({ teams: 2 });
    expect(readFileSync(join(w.root, "summary.json"), "utf8")).toContain("teams");
    expect(findLeaks(w, observation)).toEqual([]);
    expect(() => expectNoLeak(w, observation)).not.toThrow();
    // The terminal detail does hold the team, so the scan would see it on any other surface.
    expect(findLeaks(w, { ...observation, stdout: detail.join("") }).length).toBeGreaterThan(0);
  });

  it("an exception with fixed words only is clean too", async () => {
    const w = world();
    const observation = await observe(w, () => {
      assertOwnerSession({ ...terminalSession, env: { CLAUDECODE: "1" } });
    });
    expect(observation.thrown).toContain("agent session");
    expect(findLeaks(w, observation)).toEqual([]);
  });
});

describe("negative controls: the harness flags a leak on each surface", () => {
  const species = (w: CanaryWorld): string => set0(w).species;
  const flagged = async (run: (w: CanaryWorld) => unknown, surface: string, kind?: string, watch?: (w: CanaryWorld) => string[]): Promise<void> => {
    const w = world();
    const observation = await observe(w, run as (x: CanaryWorld) => void, watch?.(w));
    const leaks = findLeaks(w, observation);
    expect(leaks.map((l) => l.surface)).toContain(surface);
    if (kind !== undefined) expect(leaks.map((l) => l.kind)).toContain(kind);
    expect(() => expectNoLeak(w, observation)).toThrow(/private data on/);
  };

  it("stdout", () => flagged((w) => void process.stdout.write(`${marker0(w)}\n`), "stdout", "marker"));
  it("stderr", () => flagged((w) => void process.stderr.write(`${species(w)}\n`), "stderr", "species"));
  it("console.log and console.error", async () => {
    await flagged((w) => console.log(set0(w).item), "stdout", "item");
    await flagged((w) => console.error(set0(w).ability), "stderr", "ability");
    await flagged((w) => console.warn({ move: set0(w).moves[0] }), "stderr", "move");
  });
  it("a team hash in a log line", () => flagged((w) => console.log(`team ${teamHash(team0(w).sets)}`), "stdout", "team_hash"));
  it("a set line", () => flagged((w) => console.log(team0(w).paste.split("\n")[1]), "stdout", "set_line"));
  it("an exception message", () =>
    flagged((w) => {
      throw new Error(`bad team ${marker0(w)}`);
    }, "exception", "marker"));
  it("an exception stack frame or own field", async () => {
    await flagged((w) => {
      const error = new Error("fixed") as Error & { detail: string };
      error.detail = species(w);
      throw error;
    }, "exception", "species");
    await flagged((w) => {
      throw new Error("outer", { cause: Object.assign(new Error("inner"), { team: need(set0(w).moves[0]) }) });
    }, "exception", "move");
  });
  it("a cause, an aggregate member and a thrown string", async () => {
    await flagged((w) => {
      throw new Error("fixed", { cause: new Error(marker1(w)) });
    }, "exception", "marker");
    await flagged((w) => {
      throw new AggregateError([new Error(species(w))], "many");
    }, "exception", "species");
    await flagged((w) => {
      throw marker0(w);
    }, "exception", "marker");
  });
  it("an extra surface (a log, a run record, a dashboard)", () =>
    flagged((w) => ({ surfaces: { dashboard: `<p>${set1(w).species}</p>` } }), "surface:dashboard", "species"));
  it("a file written outside the private root (content)", () =>
    flagged((w) => writeFileSync(join(w.outside, "report.txt"), team0(w).paste), "file content", "paste"));
  it("a file written outside the private root (name)", () =>
    flagged((w) => writeFileSync(join(w.outside, `${marker0(w)}.json`), "{}"), "file name", "marker"));
  it("a file written in a watched directory", async () => {
    const watched = mkdtempSync(join(tmpdir(), "canary-watch-"));
    try {
      writeFileSync(join(watched, "keep.txt"), "old");
      await flagged((w) => writeFileSync(join(watched, "new.txt"), marker0(w)), "file content", "marker", () => [watched]);
    } finally {
      rmSync(watched, { recursive: true, force: true });
    }
  });
  it("a file in the private root that others can read", async () => {
    const w = world();
    const observation = await observe(w, (x) => {
      writeFileSync(join(x.root, "open.txt"), "no team text, only a loose mode");
      chmodSync(join(x.root, "open.txt"), 0o644);
    });
    expect(findLeaks(w, observation)).toEqual([{ surface: "private root", kind: "mode" }]);
    expect(() => expectNoLeak(w, observation)).toThrow(/private root \(mode\)/);
  });
  it("a team in a file of the private root is not a leak (that is where it belongs)", async () => {
    const w = world();
    const observation = await observe(w, (x) => void writePrivateFile(x.root, "teams.txt", team0(x).paste));
    expect(findLeaks(w, observation)).toEqual([]);
  });
});

describe("the capture itself", () => {
  it("restores stdout, stderr and the console, also when the command throws", async () => {
    const w = world();
    const before = [process.stdout.write, process.stderr.write];
    await observe(w, () => {
      throw new Error("x");
    });
    expect([process.stdout.write, process.stderr.write]).toEqual(before);
    const again = await observe(w, () => void process.stdout.write("second"));
    expect(again.stdout).toBe("second");
  });

  it("captures binary writes and non-string console arguments", async () => {
    const w = world();
    const observation = await observe(w, () => {
      process.stdout.write(Buffer.from("raw-bytes"));
      console.info("a", { b: 1 });
    });
    expect(observation.stdout).toBe('raw-bytesa {"b":1}\n');
  });

  it("an error surface survives unserialisable fields and a long cause chain", () => {
    const loop: Record<string, unknown> = {};
    loop["self"] = loop;
    const error = Object.assign(new Error("e"), { loop });
    expect(errorSurface(error)).toContain("Error: e");
    expect(errorSurface(loop)).toContain("[object Object]");
    let deep: Error = new Error("bottom");
    for (let i = 0; i < 10; i++) deep = new Error(`level${i}`, { cause: deep });
    expect(errorSurface(deep)).toContain("level9");
  });

  it("leaves nothing behind", () => {
    const w = makeCanaryWorld(engine);
    const root = w.root;
    mkdirSync(join(w.outside, "x"));
    w.cleanup();
    expect(() => readFileSync(join(root, "nope"))).toThrow();
  });
});

afterAll(() => {
  for (const w of worlds) w.cleanup();
});
