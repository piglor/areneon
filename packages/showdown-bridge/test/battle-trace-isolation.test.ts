import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { describe, expect, it } from "vitest";

// ADR-030 common rule 1 (training games only) and ADR-028 section 5 F0, #1462: the forced-draw trace recorder and the Showdown
// battle backend (which install a random source on a battle) and the committed-battle list are reachable from the bridge's own
// files and the composition root's differential files only, never from an evaluation, progress, private-test, sealed or A2
// path. A source scan, with negative controls, over everything that ships or runs outside a test.

const ROOT = join(import.meta.dirname, "..", "..", "..");

/** What only the allowed files may name: the modules and every export they hold that drives an engine. */
export const RECORDER_NAMES = /\b(recordBattleTrace|showdownBattleBackend|committedBattles|recordingSource|forcingSource|installSource|interceptedSource|tracedStreamClass|stepRequests|agreesWithRecord)\b|["'][^"']*\/battle-(trace|replay|trace-corpus)(\.ts)?["']/;

/** The bridge's own files that hold or re-export them (and the battle-state projection's trace reader and its fixture capture, G4), the engine-free core (its own trace reader shares the file name; it imports no engine) and the composition root's differential files. */
export const ALLOWED = /^packages\/showdown-bridge\/src\/(battle-trace|battle-replay|battle-trace-corpus|battle-trace-samples|index|projection-points)\.ts$|^packages\/showdown-bridge\/scripts\/capture-projection-points\.ts$|^packages\/differential\/src\/[a-z-]+\.ts$|^packages\/team-dataset\/(src|scripts)\/differential[a-z.-]*\.ts$/;

/** Paths that must never reach them, whatever the allowlist says: evaluation, progress, private test, sealed runs and A2. */
export const FORBIDDEN_PATH = /(evaluation|progress|private|sealed|bot-client|a2[-_]|session)/;

export function scanFile(file: string, text: string): string[] {
  if (!RECORDER_NAMES.test(text)) return [];
  if (FORBIDDEN_PATH.test(file)) return [`${file}: an evaluation, progress, private-test, sealed or A2 path names the recorder`];
  return ALLOWED.test(file) ? [] : [`${file}: names the recorder outside the allowed files`];
}

const SKIP = new Set(["node_modules", "dist", "vendor", "fixtures", "test", "tests", ".venv", ".git", "coverage", "private-data", "docs", "__pycache__"]);

function walk(dir: string, out: string[]): string[] {
  for (const name of readdirSync(dir)) {
    if (SKIP.has(name) || name.startsWith(".")) continue;
    const path = join(dir, name);
    if (statSync(path).isDirectory()) walk(path, out);
    else if (/\.(ts|cjs|js|mjs|py)$/.test(name) && !name.endsWith(".test.ts") && !name.endsWith(".d.ts")) out.push(path);
  }
  return out;
}

const shipped = (): string[] => ["packages", "ml", "scripts", "services", "crates"].flatMap((d) => (statSync(join(ROOT, d), { throwIfNoEntry: false })?.isDirectory() ? walk(join(ROOT, d), []) : []));

describe("the trace recorder and the battle backend stay off every evaluation path", () => {
  it("only the bridge's own files and the composition root's differential files name them", () => {
    const files = shipped().map((f) => relative(ROOT, f));
    expect(files.length).toBeGreaterThan(100);
    expect(files.flatMap((file) => scanFile(file, readFileSync(join(ROOT, file), "utf8")))).toEqual([]);
    const naming = files.filter((file) => RECORDER_NAMES.test(readFileSync(join(ROOT, file), "utf8")));
    expect(naming).toEqual(expect.arrayContaining(["packages/showdown-bridge/src/battle-replay.ts", "packages/showdown-bridge/src/index.ts", "packages/team-dataset/src/differential-battle.ts"]));
  });

  it("negative controls: an import or a name of the recorder outside the allowed files is flagged, on an evaluation path always", () => {
    expect(scanFile("packages/showdown-bridge/src/x.ts", 'import { recordBattleTrace } from "@areneon/showdown-bridge";')).toHaveLength(1);
    expect(scanFile("packages/showdown-bridge/src/x.ts", 'import { a } from "./battle-replay.ts";')).toHaveLength(1);
    expect(scanFile("packages/team-dataset/src/loop-pool.ts", "showdownBattleBackend(engine);")).toHaveLength(1);
    expect(scanFile("packages/team-dataset/src/differential-progress.ts", "committedBattles();")).toEqual([expect.stringContaining("evaluation, progress")]);
    expect(scanFile("packages/showdown-bridge/src/progress-evaluation.ts", "installSource(b, s);")).toHaveLength(1);
    expect(scanFile("packages/bot-client/src/session.ts", "forcingSource(p);")).toHaveLength(1);
    expect(scanFile("packages/team-dataset/src/differential-battle.ts", "recordBattleTrace(e, s);")).toEqual([]);
    expect(scanFile("packages/showdown-bridge/src/x.ts", "export const recordBattleTraceSize = 1;")).toEqual([]);
    expect(scanFile("packages/showdown-bridge/src/projection-points.ts", "recordBattleTrace(e, s);")).toEqual([]);
    expect(scanFile("packages/showdown-bridge/scripts/capture-other.ts", "committedBattles();")).toHaveLength(1);
  });
});
