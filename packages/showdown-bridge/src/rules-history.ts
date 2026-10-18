/** The bits of git the channel check needs; injected so the policy is testable. */
export interface GitReader {
  /** Run `git <args>` and return trimmed stdout, or `undefined` if it fails. */
  (...args: string[]): string | undefined;
}

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

/**
 * The commit the channel file is compared against. CI passes `HEAD^1`
 * through `RULES_BASE_REF`: the first parent of a PR merge commit is the
 * main tip it merges into, and of a squash commit on main the previous tip.
 * Locally it is the merge-base with origin/main.
 */
export function chooseBase(env: { RULES_BASE_REF?: string | undefined }, git: GitReader): string | undefined {
  if (env.RULES_BASE_REF) return git("rev-parse", "--verify", `${env.RULES_BASE_REF}^{commit}`);
  return git("merge-base", "HEAD", "origin/main");
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
  const listed = git("rev-list", "--first-parent", base, "--", ...RULES_HISTORY_PATHS) ?? "";
  const built: BuiltSnapshot[] = [];
  for (const commit of listed.split("\n").filter(Boolean)) {
    if (git("cat-file", "-e", `${commit}:${RULES_BUILDER}`) === undefined) continue;
    try {
      built.push(build(commit));
    } catch (error) {
      warn(`skipping ${commit}: its rules builder failed (${(error as Error).message})`);
    }
  }
  return built;
}
