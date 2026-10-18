import { existsSync, readFileSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { DEFAULT_RULES_DIR, PINNED_SHOWDOWN_COMMIT, RulesError } from "../src/index.ts";
import {
  baseChannel,
  buildAt,
  type BuiltSnapshot,
  type Exec,
  type GitReader,
  historicalSnapshots,
  resolveBase,
  RULES_BUILDER,
  RULES_BUILDER_TIMEOUT_MS,
  RULES_HISTORY_PATHS,
} from "../src/rules-history.ts";

const REPO = fileURLToPath(new URL("../../../", import.meta.url));

/** A scripted git: each call is matched on its joined arguments. */
function fakeGit(answers: Record<string, string | undefined>): GitReader & { calls: string[] } {
  const calls: string[] = [];
  const git = (...args: string[]): string | undefined => {
    const key = args.join(" ");
    calls.push(key);
    return answers[key];
  };
  return Object.assign(git, { calls });
}

const built = (commit: string): BuiltSnapshot => ({ hash: commit.repeat(64).slice(0, 64), showdown_commit: "p" });
const builderAt = (commit: string): string => `ls-tree --name-only ${commit} -- ${RULES_BUILDER}`;
const noWarnings = (message: string): never => {
  throw new Error(`unexpected warning: ${message}`);
};

describe("resolveBase", () => {
  it("resolves RULES_BASE_REF when set", () => {
    const git = fakeGit({ "rev-parse --verify HEAD^1^{commit}": "abc" });
    expect(resolveBase({ RULES_BASE_REF: "HEAD^1" }, git, noWarnings)).toBe("abc");
  });

  it("fails on an unresolvable RULES_BASE_REF", () => {
    expect(() => resolveBase({ RULES_BASE_REF: "nope" }, fakeGit({}), noWarnings)).toThrow(
      /RULES_BASE_REF=nope does not resolve to a commit/,
    );
  });

  it("falls back to the merge-base with origin/main", () => {
    const git = fakeGit({ "merge-base HEAD origin/main": "def" });
    expect(resolveBase({}, git, noWarnings)).toBe("def");
    expect(resolveBase({ RULES_BASE_REF: "" }, git, noWarnings)).toBe("def");
  });

  it("warns when no base resolves locally", () => {
    const warnings: string[] = [];
    expect(resolveBase({}, fakeGit({}), (m) => warnings.push(m))).toBeUndefined();
    expect(warnings).toEqual([expect.stringMatching(/no merge-base with origin\/main/)]);
  });
});

describe("baseChannel", () => {
  const path = "rules/channels/stable.json";
  const listing = `ls-tree --name-only base -- ${path}`;
  const channel = { channel: "stable", seq: 1, hash: "a".repeat(64) };

  it("reads and validates the channel at the base", () => {
    const git = fakeGit({ [listing]: path, [`show base:${path}`]: JSON.stringify(channel) });
    expect(baseChannel("base", "stable", git)).toEqual(channel);
  });

  it("returns undefined when the channel is new at the base", () => {
    expect(baseChannel("base", "stable", fakeGit({ [listing]: "" }))).toBeUndefined();
  });

  it("fails when git cannot list or read the base", () => {
    expect(() => baseChannel("base", "stable", fakeGit({}))).toThrow(/git could not list/);
    expect(() => baseChannel("base", "stable", fakeGit({ [listing]: path }))).toThrow(/git could not read base:/);
  });

  it("names the base file when it is not valid JSON or not a channel", () => {
    const git = (text: string): GitReader => fakeGit({ [listing]: path, [`show base:${path}`]: text });
    expect(() => baseChannel("base", "stable", git("["))).toThrow(RulesError);
    expect(() => baseChannel("base", "stable", git("["))).toThrow(/base:rules\/channels\/stable\.json: not valid JSON/);
    expect(() => baseChannel("base", "stable", git(JSON.stringify({ ...channel, seq: 0 })))).toThrow(
      /base:rules\/channels\/stable\.json/,
    );
  });
});

describe("historicalSnapshots", () => {
  const listing = `rev-list --first-parent base -- ${RULES_HISTORY_PATHS.join(" ")}`;

  it("rebuilds every rules-touching commit that carries the builder", () => {
    const git = fakeGit({
      [listing]: "c3\nc2\nc1",
      [builderAt("c3")]: RULES_BUILDER,
      [builderAt("c2")]: RULES_BUILDER,
      [builderAt("c1")]: "",
    });
    expect(historicalSnapshots("base", git, built, noWarnings)).toEqual([built("c3"), built("c2")]);
  });

  it("skips and reports a builder that no longer runs", () => {
    const git = fakeGit({ [listing]: "c2\nc1", [builderAt("c2")]: RULES_BUILDER, [builderAt("c1")]: RULES_BUILDER });
    const warnings: string[] = [];
    const build = (commit: string): BuiltSnapshot => {
      if (commit === "c2") throw new Error("boom");
      return built(commit);
    };
    expect(historicalSnapshots("base", git, build, (m) => warnings.push(m))).toEqual([built("c1")]);
    expect(warnings).toEqual(["skipping c2: its rules builder failed (boom)"]);
  });

  it("returns nothing for an empty history", () => {
    expect(historicalSnapshots("base", fakeGit({ [listing]: "" }), built, noWarnings)).toEqual([]);
  });

  // A git failure is not an empty history: treating it as one would drop
  // allowed snapshots without saying why.
  it("fails when git cannot list the history or a commit's tree", () => {
    expect(() => historicalSnapshots("base", fakeGit({}), built, noWarnings)).toThrow(/git could not list the history/);
    expect(() => historicalSnapshots("base", fakeGit({ [listing]: "c1" }), built, noWarnings)).toThrow(
      /git could not list .* at c1/,
    );
  });

  // Everything the builder runs must lie under the watched paths, or a
  // commit changing it would not be rebuilt.
  it("watches every file the builder reads or imports", () => {
    const seen = new Set<string>();
    const visit = (file: string): void => {
      if (seen.has(file)) return;
      seen.add(file);
      for (const [, spec] of readFileSync(file, "utf8").matchAll(/from "(\.{1,2}\/[^"]+)"/g)) {
        visit(resolve(dirname(file), spec as string));
      }
    };
    visit(resolve(REPO, RULES_BUILDER));
    const read = [...seen, DEFAULT_RULES_DIR].map((file) => relative(REPO, file));
    expect(read.length).toBeGreaterThan(2);
    for (const file of read) {
      expect(RULES_HISTORY_PATHS.some((path) => file.startsWith(`${path}/`) || file === path), file).toBe(true);
    }
  });
});

describe("buildAt", () => {
  it("rebuilds a commit's snapshot from a bare archive of the watched paths", () => {
    expect(buildAt("HEAD")).toEqual({ hash: expect.stringMatching(/^[0-9a-f]{64}$/), showdown_commit: PINNED_SHOWDOWN_COMMIT });
  });

  it("bounds every step with the builder timeout and cleans up", () => {
    const calls: { file: string; args: string[]; timeout: number }[] = [];
    const exec: Exec = (file, args, options) => {
      calls.push({ file, args, timeout: options.timeout });
      if (file === "node") throw new Error("spawnSync node ETIMEDOUT");
      return "";
    };
    expect(() => buildAt("c1", exec)).toThrow(/ETIMEDOUT/);
    expect(calls.map(({ file }) => file)).toEqual(["git", "tar", "node"]);
    expect(calls.every(({ timeout }) => timeout === RULES_BUILDER_TIMEOUT_MS)).toBe(true);
    expect(calls[0]?.args.slice(-RULES_HISTORY_PATHS.length)).toEqual(RULES_HISTORY_PATHS);
    expect(existsSync(dirname(calls[0]?.args[2] as string))).toBe(false);
  });
});
