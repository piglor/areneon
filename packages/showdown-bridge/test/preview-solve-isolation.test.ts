// The Team Preview solver is evaluation only (ADR-029 section 5, N12): no training, Episode or loader path may reach it, and it
// reads only the player's own view (section 7). A source scan with negative controls.
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { describe, expect, it } from "vitest";

const ROOT = new URL("../../..", import.meta.url).pathname;
const SOLVER_FILES = ["preview-solve.ts", "preview-solve-config.ts", "preview-solve-config-samples.ts", "samples.ts", "index.ts"];
/** The shipped modules that write training data or Episodes. */
const TRAINING_FILES = ["training-loader.ts", "episode-io.ts", "episode.ts", "league-generation.ts", "teacher-generation.ts", "play-generation.ts", "generation-cli.ts", "feature-recompute.ts"];
/** What the solver must never import or name: the writers of training data and the running battle's records. */
const SOLVER_FORBIDDEN_IMPORTS = ["training-loader", "episode-io", "episode.ts", "league-generation", "teacher-generation", "play-generation", "private-records", "private-snapshot", "runner.ts"];
const SOLVER_BANNED_NAMES = ["BattleRecord", "OmniscientLog", "HiddenVerdicts", "hidden_verdicts", "omniscient_log", "CloneJudge", "runBattle", "recordedChannel"];

interface Finding {
  file: string;
  what: string;
}

export function scanSource(file: string, text: string): Finding[] {
  const name = file.split("/").at(-1) as string;
  const found: Finding[] = [];
  if (name === "preview-solve.ts") {
    for (const forbidden of SOLVER_FORBIDDEN_IMPORTS) {
      // A type-only import of the runner's `Actor` and `Player` is how every actor module names the actor type; the runner's records are banned by name.
      if (forbidden !== "runner.ts" && new RegExp(`from "[^"]*${forbidden}[^"]*"`).test(text)) found.push({ file, what: `imports ${forbidden}` });
    }
    for (const banned of SOLVER_BANNED_NAMES) if (text.includes(banned)) found.push({ file, what: `names ${banned}` });
    if (/writeFile|appendFile|createWriteStream/.test(text)) found.push({ file, what: "writes a file" });
  } else if (TRAINING_FILES.includes(name) && /preview-solve/.test(text)) {
    found.push({ file, what: "reaches the solver" });
  }
  return found;
}

const SKIP = new Set(["node_modules", "dist", "vendor", "fixtures", "test", "tests", ".venv", ".git", "coverage", "private-data", "docs", "__pycache__"]);
function walk(dir: string, out: string[]): string[] {
  for (const name of readdirSync(dir)) {
    if (SKIP.has(name) || name.startsWith(".")) continue;
    const path = join(dir, name);
    if (statSync(path).isDirectory()) walk(path, out);
    else if (/\.(ts|cjs|js|mjs)$/.test(name) && !name.endsWith(".test.ts") && !name.endsWith(".d.ts")) out.push(path);
  }
  return out;
}

describe("the Team Preview solver is evaluation only", () => {
  it("no training or Episode module reaches it, and it imports none and names no battle record", () => {
    const files = ["packages", "ml", "scripts", "services", "crates"].flatMap((d) => (statSync(join(ROOT, d), { throwIfNoEntry: false })?.isDirectory() ? walk(join(ROOT, d), []) : []));
    const names = files.map((f) => relative(ROOT, f));
    expect(names).toContain("packages/showdown-bridge/src/preview-solve.ts");
    expect(names).toContain("packages/showdown-bridge/src/training-loader.ts");
    for (const name of SOLVER_FILES) expect(names.some((n) => n.endsWith(`/${name}`)), name).toBe(true);
    const findings = files.flatMap((file) => scanSource(relative(ROOT, file), readFileSync(file, "utf8")));
    expect(findings).toEqual([]);
  });

  it("negative controls: the scan flags a training writer in the solver and the solver reached from the loader", () => {
    expect(scanSource("src/preview-solve.ts", 'import { x } from "./training-loader.ts";').map((f) => f.what)).toEqual(["imports training-loader"]);
    expect(scanSource("src/preview-solve.ts", "const r: BattleRecord = 1;").map((f) => f.what)).toEqual(["names BattleRecord"]);
    expect(scanSource("src/preview-solve.ts", 'writeFileSync("x", "y");').map((f) => f.what)).toEqual(["writes a file"]);
    expect(scanSource("src/training-loader.ts", 'import { x } from "./preview-solve.ts";').map((f) => f.what)).toEqual(["reaches the solver"]);
    expect(scanSource("src/policy-actor.ts", 'import { x } from "./preview-solve.ts";')).toEqual([]);
    expect(scanSource("src/preview-solve.ts", 'import { Actor } from "./runner.ts";')).toEqual([]);
  });
});
