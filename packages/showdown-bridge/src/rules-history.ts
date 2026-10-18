import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { parseChannel } from "./rules-load.ts";
import { type ChannelFile, RulesError } from "./rules.ts";

// CI-only internals of scripts/check-rules.ts, deliberately not exported
// from the package index: nothing at runtime depends on git history.

/** The bits of git the channel check needs; injected so the policy is testable. */
export interface GitReader {
  /** Run `git <args>` and return trimmed stdout, or `undefined` if it fails. */
  (...args: string[]): string | undefined;
}

/** Run a program without a shell and return its stdout. */
export type Exec = (
  file: string,
  args: string[],
  options: { encoding: "utf8"; timeout: number; cwd: string },
) => string;

/** The repository root. Git runs here so paths mean the same from any directory. */
export const REPO_ROOT = fileURLToPath(new URL("../../../", import.meta.url));

/** A `GitReader` that runs git in `cwd`. */
export function gitAt(cwd: string): GitReader {
  return (...args) => {
    try {
      return execFileSync("git", args, { cwd, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim();
    } catch {
      return undefined;
    }
  };
}

/** Pathspecs anchored at the repository root, whatever git's working directory. */
const fromTop = (path: string): string => `:(top)${path}`;

/** What one historical builder printed. */
export interface BuiltSnapshot {
  hash: string;
  showdown_commit: string;
}

/** The dependency-free builder every rules-aware commit carries (scripts/rules-hash.ts). */
export const RULES_BUILDER = "packages/showdown-bridge/scripts/rules-hash.ts";

/**
 * A snapshot's bytes depend on the records and on any bridge code the
 * builder runs, so every commit touching either is rebuilt.
 */
export const RULES_HISTORY_PATHS: readonly string[] = ["rules", "packages/showdown-bridge"];

/** A historical builder that hangs is killed, so it cannot stall CI. */
export const RULES_BUILDER_TIMEOUT_MS = 60_000;

/** Whether `path` exists at `commit`. Throws when git itself fails. */
function existsAt(git: GitReader, commit: string, path: string): boolean {
  const listed = git("ls-tree", "--full-tree", "--name-only", commit, "--", path);
  if (listed === undefined) throw new RulesError(`git could not list ${path} at ${commit}`);
  return listed !== "";
}

/**
 * The commit the channel file is compared against. CI passes `HEAD^1`
 * through `RULES_BASE_REF`: the first parent of a PR merge commit is the
 * main tip it merges into, and of a squash commit on main the previous tip.
 * Locally it is the merge-base with origin/main; without one the history
 * checks are skipped with a warning (CI always runs them).
 */
export function resolveBase(
  env: { RULES_BASE_REF?: string | undefined },
  git: GitReader,
  warn: (message: string) => void,
): string | undefined {
  if (env.RULES_BASE_REF) {
    const base = git("rev-parse", "--verify", `${env.RULES_BASE_REF}^{commit}`);
    if (base === undefined) {
      throw new RulesError(`RULES_BASE_REF=${env.RULES_BASE_REF} does not resolve to a commit`);
    }
    return base;
  }
  const base = git("merge-base", "HEAD", "origin/main");
  if (base === undefined) {
    warn("no merge-base with origin/main: channel seq and history are not checked (CI checks them)");
  }
  return base;
}

/** `rules/channels/<channel>.json` at `base`, or `undefined` if the channel is new there. */
export function baseChannel(base: string, channel: string, git: GitReader): ChannelFile | undefined {
  const path = `rules/channels/${channel}.json`;
  if (!existsAt(git, base, path)) return undefined;
  const where = `${base}:${path}`;
  const text = git("show", where);
  if (text === undefined) throw new RulesError(`git could not read ${where}`);
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch (error) {
    throw new RulesError(`${where}: not valid JSON (${(error as Error).message})`);
  }
  return parseChannel(channel, value, where);
}

/**
 * Rebuild the snapshot of every first-parent commit up to `base` that
 * changed the rules, each with that commit's own builder. Commits from
 * before the builder existed contribute nothing. A builder that no longer
 * runs is reported and skipped: its snapshot just stops being an allowed
 * channel target, which fails safe instead of blocking every later PR.
 */
export function historicalSnapshots(
  base: string,
  git: GitReader,
  build: (commit: string) => BuiltSnapshot,
  warn: (message: string) => void,
): BuiltSnapshot[] {
  const listed = git("rev-list", "--first-parent", base, "--", ...RULES_HISTORY_PATHS.map(fromTop));
  if (listed === undefined) throw new RulesError(`git could not list the history of ${base}`);
  const built: BuiltSnapshot[] = [];
  for (const commit of listed.split("\n").filter(Boolean)) {
    if (!existsAt(git, commit, RULES_BUILDER)) continue;
    try {
      built.push(build(commit));
    } catch (error) {
      const reason = error instanceof RulesError ? error.message : `its rules builder failed (${(error as Error).message})`;
      warn(`skipping ${commit}: ${reason}`);
    }
  }
  return built;
}

const run: Exec = (file, args, options) => execFileSync(file, args, options);

/** Run `commit`'s own builder from a bare archive: no node_modules, no shell. */
export function buildAt(commit: string, exec: Exec = run): BuiltSnapshot {
  const dir = mkdtempSync(join(tmpdir(), "areneon-rules-"));
  const options = { encoding: "utf8", timeout: RULES_BUILDER_TIMEOUT_MS, cwd: REPO_ROOT } as const;
  try {
    const archive = join(dir, "tree.tar");
    try {
      exec("git", ["archive", "-o", archive, commit, ...RULES_HISTORY_PATHS], options);
    } catch (error) {
      throw new RulesError(`git could not archive ${commit} (${(error as Error).message})`);
    }
    exec("tar", ["-xf", archive, "-C", dir], options);
    return JSON.parse(exec("node", [join(dir, RULES_BUILDER)], options)) as BuiltSnapshot;
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}
