// The privacy canary harness (ADR-015 erratum E1): a reusable test helper for
// any command that reads private teams. It builds a throwaway world of
// synthetic stand-in teams (committed fixture teams from `fixtures/teams/`,
// not in any `known_synthetic` set, each Pokemon given a distinctive
// nickname marker) inside a 0700 `private-data` root, runs a command, and
// captures everything that could leak: stdout, stderr and console output,
// the exception (message, name, stack, own fields, cause chain), any extra
// surface the command returns (logs, run records, dashboards) and every file
// or file name written outside the private root. It then looks for any
// marker, species, item, ability, move, set line, team hash or paste of the
// stand-ins. Its own negative controls (private-canary.test.ts) show it
// flags a leak on each surface. Only synthetic teams; never a real team.

import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { importPaste } from "../../src/teams.ts";
import type { Engine, PokemonSet } from "../../src/engine.ts";
import { canonicalTeamHash } from "../../src/team-records.ts";
import { classOfSets, teamHash } from "../../src/team-class.ts";

const ROOT = fileURLToPath(new URL("../../../../", import.meta.url));
const DEV_TEAMS = join(ROOT, "fixtures", "teams", "gen9championsvgc2026regmc", "datasets", "fixtures-v1", "teams-dev.jsonl");

export type NeedleKind = "marker" | "species" | "item" | "ability" | "move" | "set_line" | "team_hash" | "paste" | "file_name";
export interface Needle {
  kind: NeedleKind;
  text: string;
}

export interface StandInTeam {
  name: string;
  paste: string;
  sets: PokemonSet[];
}

export interface CanaryWorld {
  /** The 0700 private output root (a `private-data` path segment). */
  root: string;
  /** Where the stand-in teams are staged (inside the root's parent, also under `private-data`). */
  teamsDir: string;
  /** A fresh directory outside the private root: nothing the command writes here may hold a needle. */
  outside: string;
  teams: StandInTeam[];
  needles: Needle[];
  markers: string[];
  cleanup(): void;
}

const MIN_NEEDLE = 4;

/** Give every Pokemon of a paste a nickname: `Nick (Species) @ Item`. */
export function nickPaste(paste: string, nick: (index: number) => string): string {
  return paste
    .split("\n\n")
    .map((block, i) => block.replace(/^([^@\n(]*?)( @ )/, (_all, species: string, at: string) => `${nick(i)} (${species})${at}`))
    .join("\n\n");
}

let counter = 0;

/** Build a world of `count` stand-in teams. Distinct markers per world and team. */
export function makeCanaryWorld(engine: Engine, count = 2): CanaryWorld {
  const base = mkdtempSync(join(tmpdir(), "canary-world-"));
  const id = `${process.pid}x${counter++}`;
  const parent = join(base, "private-data");
  const root = join(parent, "out");
  const teamsDir = join(parent, "teams");
  const outside = join(base, "outside");
  for (const dir of [parent, root, teamsDir, outside]) mkdirSync(dir, { mode: 0o700 });

  const pool = readFileSync(DEV_TEAMS, "utf8").split("\n").filter(Boolean).map((line) => (JSON.parse(line) as { paste: string }).paste);
  if (pool.length < count) throw new Error("not enough stand-in teams");
  const markers: string[] = [];
  const teams: StandInTeam[] = pool.slice(0, count).map((source, t) => {
    const paste = nickPaste(source, (i) => {
      const marker = `CanaryNick${id}t${t}p${i}`;
      markers.push(marker);
      return marker;
    });
    const sets = importPaste(engine, paste); // the engine's own round trip: the nicknames survive
    return { name: `canary-team-${id}-${t}.txt`, paste, sets };
  });
  for (const team of teams) writeFileSync(join(teamsDir, team.name), team.paste, { mode: 0o600 });

  const needles: Needle[] = markers.map((text) => ({ kind: "marker", text }));
  const add = (kind: NeedleKind, text: string): void => {
    if (text.length >= MIN_NEEDLE) needles.push({ kind, text });
  };
  for (const team of teams) {
    add("file_name", team.name);
    add("paste", team.paste.trim());
    add("team_hash", teamHash(team.sets));
    add("team_hash", canonicalTeamHash(team.sets));
    for (const set of team.sets) {
      add("species", set.species);
      add("item", set.item);
      add("ability", set.ability);
      for (const move of set.moves) add("move", move);
    }
    for (const line of team.paste.split("\n")) add("set_line", line.trim());
  }
  const unique = [...new Map(needles.map((n) => [`${n.kind}:${n.text}`, n])).values()];
  return { root, teamsDir, outside, teams, needles: unique, markers, cleanup: () => rmSync(base, { recursive: true, force: true }) };
}

/** True if no stand-in is in a `known_synthetic` set: the world's teams count as private. */
export const worldIsPrivate = (world: CanaryWorld): boolean =>
  world.teams.every((team) => classOfSets(team.sets, new Set<string>()) === "private");

export interface Observation {
  stdout: string;
  stderr: string;
  /** All text an exception exposes, or null if none was thrown. */
  thrown: string | null;
  /** Extra surfaces the command hands back (a log, a run record, a dashboard), by name. */
  surfaces: Record<string, string>;
  /** Files created or changed outside the private root, by path relative to the watched directory. */
  files: { path: string; text: string }[];
  /** Paths inside the private root that group or other can reach. */
  loosePaths: string[];
}

export interface CommandResult {
  /** Extra text to scan, by surface name. */
  surfaces?: Record<string, string>;
}

/** Everything an error shows: name, message, stack, own fields, the cause chain and nested errors. */
export function errorSurface(error: unknown, depth = 0): string {
  if (depth > 5) return "";
  if (!(error instanceof Error)) {
    let shown: string;
    try {
      shown = JSON.stringify(error) ?? "";
    } catch {
      shown = "";
    }
    return `${String(error)}\n${shown}`;
  }
  const parts = [String(error), error.name, error.message, error.stack ?? ""];
  try {
    parts.push(JSON.stringify(error, Object.getOwnPropertyNames(error)));
  } catch {
    // unserialisable fields: the string forms above still count
  }
  if ("cause" in error) parts.push(errorSurface(error.cause, depth + 1));
  if (error instanceof AggregateError) for (const inner of error.errors as unknown[]) parts.push(errorSurface(inner, depth + 1));
  return parts.join("\n");
}

type Listing = Map<string, string>;
function listing(dir: string, base = dir, into: Listing = new Map()): Listing {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    const stat = statSync(path, { throwIfNoEntry: false });
    if (stat === undefined) continue;
    into.set(relative(base, path), `${stat.mtimeMs}:${stat.size}:${stat.mode}`);
    if (entry.isDirectory()) listing(path, base, into);
  }
  return into;
}

function loose(dir: string): string[] {
  const out: string[] = [];
  for (const rel of listing(dir).keys()) {
    const stat = statSync(join(dir, rel));
    if ((stat.mode & 0o077) !== 0) out.push(rel);
  }
  return out;
}

/**
 * Run `command` in the world and collect every surface. stdout, stderr and the
 * console are captured (and restored, also on a throw); a thrown error is
 * recorded, not rethrown. `watch` lists directories besides `world.outside`
 * whose new or changed files are scanned too.
 */
export async function observe(
  world: CanaryWorld,
  command: (world: CanaryWorld) => unknown,
  watch: string[] = [],
): Promise<Observation> {
  const dirs = [world.outside, ...watch];
  const before = dirs.map((dir) => listing(dir));
  let stdout = "";
  let stderr = "";
  const out = process.stdout.write;
  const err = process.stderr.write;
  const consoleMethods = ["log", "info", "debug", "warn", "error", "trace"] as const;
  const originals = consoleMethods.map((m) => console[m]);
  const text = (chunk: unknown): string => (typeof chunk === "string" ? chunk : Buffer.from(chunk as Uint8Array).toString("utf8"));
  process.stdout.write = ((chunk: unknown) => ((stdout += text(chunk)), true)) as typeof process.stdout.write;
  process.stderr.write = ((chunk: unknown) => ((stderr += text(chunk)), true)) as typeof process.stderr.write;
  consoleMethods.forEach((m) => {
    console[m] = (...args: unknown[]): void => {
      (m === "log" || m === "info" || m === "debug" ? (s: string) => (stdout += s) : (s: string) => (stderr += s))(`${args.map((a) => (typeof a === "string" ? a : JSON.stringify(a))).join(" ")}\n`);
    };
  });
  let thrown: string | null = null;
  let surfaces: Record<string, string> = {};
  try {
    const result = await command(world);
    surfaces = (result as CommandResult | undefined)?.surfaces ?? {};
  } catch (error) {
    thrown = errorSurface(error);
  } finally {
    process.stdout.write = out;
    process.stderr.write = err;
    consoleMethods.forEach((m, i) => {
      console[m] = originals[i] as never;
    });
  }
  const files: { path: string; text: string }[] = [];
  dirs.forEach((dir, i) => {
    const prior = before[i] as Listing;
    for (const [rel, signature] of listing(dir)) {
      if (prior.get(rel) === signature) continue;
      const full = join(dir, rel);
      files.push({ path: rel, text: statSync(full).isFile() ? readFileSync(full, "utf8") : "" });
    }
  });
  return { stdout, stderr, thrown, surfaces, files, loosePaths: loose(world.root) };
}

export interface Leak {
  surface: string;
  kind: NeedleKind | "mode";
}

/** Every needle found on every surface; a file name counts as a surface of its own. */
export function findLeaks(world: CanaryWorld, observation: Observation): Leak[] {
  const surfaces: [string, string][] = [
    ["stdout", observation.stdout],
    ["stderr", observation.stderr],
    ["exception", observation.thrown ?? ""],
    ...Object.entries(observation.surfaces).map(([name, value]): [string, string] => [`surface:${name}`, value]),
    ...observation.files.flatMap((file): [string, string][] => [
      ["file name", file.path],
      ["file content", file.text],
    ]),
  ];
  const leaks: Leak[] = [];
  for (const [surface, content] of surfaces) {
    for (const needle of world.needles) if (content.includes(needle.text)) leaks.push({ surface, kind: needle.kind });
  }
  if (observation.loosePaths.length > 0) leaks.push({ surface: "private root", kind: "mode" });
  return leaks;
}

/** Throw unless the observation is clean. The message names surfaces and kinds, never a path. */
export function expectNoLeak(world: CanaryWorld, observation: Observation): void {
  const leaks = findLeaks(world, observation);
  if (leaks.length > 0) {
    const seen = [...new Set(leaks.map((l) => `${l.surface} (${l.kind})`))];
    throw new Error(`private data on: ${seen.join(", ")}`);
  }
}
