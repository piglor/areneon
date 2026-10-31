import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { describe, expect, it } from "vitest";
import type { BestResponseOptions, PreviewResult } from "../src/best-response.ts";
import type { episodeHash } from "../src/episode.ts";
import type { loadTrainingSet } from "../src/training-loader.ts";

// ADR-029 section 2 (#1257), isolation: the best-response responder has white-box access to the evaluated policy's weights, so it is
// an evaluation tool and nothing else. Its modules are imported by the best-response command and its own tests alone, and they
// import nothing that writes an Episode, a training example or a stream. Source scans with negative controls, plus a type test.

const ROOT = join(import.meta.dirname, "..", "..", "..");

/** The shipped files that may name the responder's modules or entry points. */
export const RESPONDER_FILES = ["best-response.ts", "best-response-cli.ts", "best-response-record.ts", "best-response-samples.ts", "samples.ts", "index.ts", "best-response.ts"];
/** What the responder must never import: the writers of training data and of Episodes. */
export const FORBIDDEN_IMPORTS = ["training-loader", "episode-io", "episode.ts", "league-generation", "teacher-generation", "play-generation", "private-records", "private-snapshot"];
const RESPONDER_NAMES = ["decidePreview", "playBestResponse", "chooseResponse", "columnWeights", "runBestResponseCli", "best-response.ts", "best-response-cli.ts"];

export interface Finding {
  file: string;
  what: string;
}

/** Scan one shipped source file. Pure, so the controls below can feed it doctored text. */
export function scanSource(file: string, text: string): Finding[] {
  const name = file.split("/").at(-1) as string;
  const found: Finding[] = [];
  const responder = name === "best-response.ts" || name === "best-response-cli.ts";
  if (responder) {
    for (const forbidden of FORBIDDEN_IMPORTS) if (new RegExp(`from "[^"]*${forbidden}[^"]*"`).test(text)) found.push({ file, what: `imports ${forbidden}` });
    if (/writeFile|appendFile|createWriteStream/.test(text) && name === "best-response.ts") found.push({ file, what: "writes a file" });
  } else if (!RESPONDER_FILES.includes(name) && !file.includes("/test/") && RESPONDER_NAMES.some((id) => text.includes(id))) {
    found.push({ file, what: "names the responder" });
  }
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

describe("the best-response responder is reachable only from the best-response command", () => {
  it("no shipped source but the command's own names the responder, and the responder imports no training-data writer", () => {
    const files = shipped();
    expect(files.length).toBeGreaterThan(100);
    const names = files.map((f) => relative(ROOT, f));
    expect(names).toContain("packages/showdown-bridge/src/best-response.ts");
    expect(names).toContain("packages/team-dataset/scripts/best-response.ts");
    const findings = files.flatMap((file) => {
      const text = readFileSync(file, "utf8");
      // The team-dataset script and the Python command are the command's own entry points.
      return /(best-response\.ts|best_response\.py)$/.test(file) && !file.includes("showdown-bridge/src") ? [] : scanSource(relative(ROOT, file), text);
    });
    expect(findings).toEqual([]);
  });

  it("negative controls: the scan flags a training-data writer in the responder and the responder named elsewhere", () => {
    expect(scanSource("src/best-response.ts", 'import { x } from "./training-loader.ts";').map((f) => f.what)).toEqual(["imports training-loader"]);
    expect(scanSource("src/best-response-cli.ts", 'import { x } from "./episode-io.ts";').map((f) => f.what)).toEqual(["imports episode-io"]);
    expect(scanSource("src/best-response.ts", "writeFileSync(a, b);").map((f) => f.what)).toEqual(["writes a file"]);
    expect(scanSource("src/league-generation.ts", "await playBestResponse(o);").map((f) => f.what)).toEqual(["names the responder"]);
    expect(scanSource("src/league-generation.ts", 'import "./best-response.ts";').map((f) => f.what)).toEqual(["names the responder"]);
    expect(scanSource("src/best-response.ts", "const x = 1;")).toEqual([]);
    expect(scanSource("src/index.ts", 'export { playBestResponse } from "./best-response.ts";')).toEqual([]);
  });

  // A type test: a responder result is two plain fields (a record of counts and choice strings), not an Episode, a training stream
  // or the loader's input, so there is no path from a decision to training data.
  it("a responder result is not training data (type test)", () => {
    type Takes<F extends (...a: never[]) => unknown> = Parameters<F>[0];
    const result = null as unknown as PreviewResult;
    const options = null as unknown as BestResponseOptions;
    // @ts-expect-error a PreviewResult is not an Episode
    const asEpisode: Takes<typeof episodeHash> = result;
    // @ts-expect-error the best-response options are not the training loader's input
    const asTraining: Takes<typeof loadTrainingSet> = options;
    expect([asEpisode, asTraining]).toBeDefined();
  });
});
