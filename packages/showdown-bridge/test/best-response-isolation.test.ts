import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { describe, expect, it } from "vitest";
import type { BestResponseOptions, PreviewContext, PreviewResult, ResponderDeps } from "../src/best-response.ts";
import type { episodeHash } from "../src/episode.ts";
import type { loadTrainingSet } from "../src/training-loader.ts";

// ADR-029 section 2 (#1257), isolation: the best-response responder has white-box access to the evaluated policy's weights, so it is
// an evaluation tool and nothing else. Its modules are imported by the best-response command and its own tests alone, and they
// import nothing that writes an Episode, a training example or a stream. Source scans with negative controls, plus a type test.

const ROOT = join(import.meta.dirname, "..", "..", "..");

/** The shipped files that may name the responder's modules or entry points. */
export const RESPONDER_FILES = ["best-response.ts", "best-response-cli.ts", "best-response-record.ts", "best-response-samples.ts", "samples.ts", "index.ts"];
/** What the responder must never import: the writers of training data and of Episodes. */
export const FORBIDDEN_IMPORTS = ["training-loader", "episode-io", "episode.ts", "league-generation", "teacher-generation", "play-generation", "private-records", "private-snapshot"];
const RESPONDER_NAMES = ["decidePreview", "playBestResponse", "chooseResponse", "columnWeights", "runBestResponseCli", "best-response.ts", "best-response-cli.ts"];

/** What the responder module may never name: the running battle, the omniscient log, hidden verdicts, the runner's records. */
export const BANNED_IN_RESPONDER = ["BattleRecord", "OmniscientLog", "HiddenVerdicts", "hidden_verdicts", "omniscient_log", "CloneJudge", "runBattle", "recordedChannel"];
/** What the Python command module may never import: learners, training streams, Episodes. */
export const PYTHON_FORBIDDEN = ["areneon_ml.learner", "areneon_ml.loop.driver", "areneon_ml.loop.warm_start", "training", "episode"];

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
    for (const banned of BANNED_IN_RESPONDER) if (name === "best-response.ts" && text.includes(banned)) found.push({ file, what: `names ${banned}` });
    if (/writeFile|appendFile|createWriteStream/.test(text) && name === "best-response.ts") found.push({ file, what: "writes a file" });
  } else if (name === "best_response.py") {
    for (const forbidden of PYTHON_FORBIDDEN) if (new RegExp(`^\\s*(from|import)\\s+\\S*${forbidden}`, "m").test(text)) found.push({ file, what: `imports ${forbidden}` });
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
      return /best-response\.ts$/.test(file) && !file.includes("showdown-bridge/src") ? [] : scanSource(relative(ROOT, file), text);
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
    for (const banned of BANNED_IN_RESPONDER) expect(scanSource("src/best-response.ts", `function f(r: ${banned}) {}`).map((f) => f.what)).toEqual([`names ${banned}`]);
    expect(scanSource("ml/areneon_ml/progress/best_response.py", "from areneon_ml.learner.data import trajectories\n").map((f) => f.what)).toEqual(["imports areneon_ml.learner"]);
    expect(scanSource("ml/areneon_ml/progress/best_response.py", "import json\nfrom areneon_ml.progress.stats import pair_interval\n")).toEqual([]);
    expect(scanSource("src/index.ts", 'export { playBestResponse } from "./best-response.ts";')).toEqual([]);
  });

  // The key sets of what a decision reads (ADR-029 section 7): no field for the other team, the running battle, the omniscient log,
  // hidden verdicts or the battle seed beyond the harness label the stream function hashes.
  it("a decision reads only the seat's own view and the harness labels (type test)", () => {
    const context: (keyof PreviewContext)[] = ["seat", "channel", "team", "battle", "seed", "game", "request_seq", "channel_class"];
    const deps: (keyof ResponderDeps)[] = ["engine", "vocabulary", "format_id", "teams", "settings", "client", "policy_id", "mode"];
    type Exact<T, K extends PropertyKey> = [keyof T] extends [K] ? ([K] extends [keyof T] ? true : never) : never;
    const contextExact: Exact<PreviewContext, (typeof context)[number]> = true;
    const depsExact: Exact<ResponderDeps, (typeof deps)[number]> = true;
    expect([contextExact, depsExact, context.length, deps.length]).toEqual([true, true, 8, 8]);
    // @ts-expect-error there is no field for the opponent's team
    const noTeam: Pick<PreviewContext, "opponent_team"> = null;
    // @ts-expect-error there is no field for a battle record
    const noRecord: Pick<ResponderDeps, "record"> = null;
    // @ts-expect-error there is no field for the omniscient log
    const noLog: Pick<PreviewContext, "omniscient_log"> = null;
    // @ts-expect-error there is no field for hidden verdicts
    const noVerdicts: Pick<ResponderDeps, "hidden_verdicts"> = null;
    expect([noTeam, noRecord, noLog, noVerdicts]).toBeDefined();
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
