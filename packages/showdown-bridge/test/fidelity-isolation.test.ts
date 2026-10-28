import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { describe, expect, it } from "vitest";

// ADR-026 Evidence 1 (B2), isolation: the fork of a live battle and the copy of its random-number state exist only
// in tests. A player at play time has neither the live battle nor its seed. This is a source scan, with negative
// controls, over everything that ships or runs outside a test.

const ROOT = join(import.meta.dirname, "..", "..", "..");

/** What only the oracle (test/fidelity/oracle.ts) may name. */
export const ORACLE_IDENTIFIERS = ["copyRandomState", "captureDecisions", "translateChoice", "stateDelta", "OracleSpec", "prng.clone("] as const;

/** Files outside tests that may rebuild a battle from JSON: each forks a battle IT built (a scenario world or a probe clone), reviewed here. */
export const FROM_JSON_ALLOWED = ["choice-judge.ts", "mechanics-damage.ts", "reply-model.ts", "search-turn.ts"];
/** Files outside tests that may assign a battle's `prng`: the mechanics probes install their own recording source. */
export const PRNG_ASSIGN_ALLOWED = ["mechanics-damage.ts", "mechanics-speed.ts", "search-turn.ts"];
// search-turn.ts (ADR-026 slice R): clones the search's OWN scenario world (a battle the world builder made, never the live battle) and installs
// the declared per-world engine seed (`search-engine/1`, the harness's search stream) as its generator; it copies no random state from any battle.

export interface Finding {
  file: string;
  what: string;
}

/** Scan one shipped source file. Pure, so the controls below can feed it doctored text. */
export function scanSource(file: string, text: string): Finding[] {
  const name = file.split("/").at(-1) as string;
  const found: Finding[] = [];
  for (const id of ORACLE_IDENTIFIERS) if (text.includes(id)) found.push({ file, what: `names the oracle identifier ${id}` });
  if (/["'][^"']*test\/fidelity\//.test(text)) found.push({ file, what: "imports the fidelity tooling" });
  if (/\bBattle\.fromJSON\(|\.fromJSON\(/.test(text) && !FROM_JSON_ALLOWED.includes(name)) found.push({ file, what: "rebuilds a battle from JSON" });
  if (/\.prng\s*=[^=]/.test(text) && !PRNG_ASSIGN_ALLOWED.includes(name)) found.push({ file, what: "assigns a battle's random-number source" });
  return found;
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

describe("the fork oracle and the random-state copy are reachable only from tests", () => {
  it("no shipped source (src, scripts, web, testing, ml, repository scripts) names the oracle, imports the fidelity tooling, forks a battle outside the reviewed files or assigns a random source", () => {
    const files = shipped();
    expect(files.length).toBeGreaterThan(100);
    const findings = files.flatMap((file) => scanSource(relative(ROOT, file), readFileSync(file, "utf8")));
    expect(findings).toEqual([]);
  });

  it("the scan covers the world builder, the fork judge and the reply model it must not be reachable from", () => {
    const names = shipped().map((f) => relative(ROOT, f));
    for (const f of ["world-builder.ts", "world-state.ts", "choice-judge.ts", "reply-model.ts", "damage-aware-actor.ts"]) expect(names).toContain(`packages/showdown-bridge/src/${f}`);
  });

  it("the oracle lives under test/ and the package exports only its own entry point", () => {
    expect(statSync(join(ROOT, "packages/showdown-bridge/test/fidelity/oracle.ts")).isFile()).toBe(true);
    const pkg = JSON.parse(readFileSync(join(ROOT, "packages/showdown-bridge/package.json"), "utf8")) as { exports: Record<string, string> };
    expect(pkg.exports).toEqual({ ".": "./src/index.ts" });
    const index = readFileSync(join(ROOT, "packages/showdown-bridge/src/index.ts"), "utf8");
    expect(index).not.toMatch(/fidelity|oracle/i);
  });

  it("negative controls: the scan flags each way the oracle could leak into shipped code", () => {
    const clean = "export const x = 1;\n";
    expect(scanSource("src/a.ts", clean)).toEqual([]);
    for (const id of ORACLE_IDENTIFIERS) expect(scanSource("src/a.ts", `${id}(x);`).map((f) => f.what)).toEqual([`names the oracle identifier ${id}`]);
    expect(scanSource("src/a.ts", 'import { x } from "../test/fidelity/oracle.ts";').some((f) => f.what === "imports the fidelity tooling")).toBe(true);
    expect(scanSource("src/a.ts", "const b = engine.sim.Battle.fromJSON(live.toJSON());").map((f) => f.what)).toEqual(["rebuilds a battle from JSON"]);
    expect(scanSource("src/choice-judge.ts", "Battle.fromJSON(x);")).toEqual([]);
    expect(scanSource("src/a.ts", "world.battle.prng = live.prng;").map((f) => f.what)).toEqual(["assigns a battle's random-number source"]);
    expect(scanSource("src/mechanics-speed.ts", "battle.prng = refusing;")).toEqual([]);
    // The world builder is in no allowlist: forking or copying there would be a finding.
    expect(scanSource("src/world-builder.ts", "x.prng = y;").length).toBe(1);
    expect(scanSource("src/world-builder.ts", "x.fromJSON(y);").length).toBe(1);
  });
});
