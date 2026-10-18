import { describe, expect, it } from "vitest";
import {
  type BuiltSnapshot,
  chooseBase,
  type GitReader,
  historicalSnapshots,
  RULES_BUILDER,
  RULES_HISTORY_PATHS,
} from "../src/index.ts";

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

describe("chooseBase", () => {
  it("resolves RULES_BASE_REF when set", () => {
    const git = fakeGit({ "rev-parse --verify HEAD^1^{commit}": "abc" });
    expect(chooseBase({ RULES_BASE_REF: "HEAD^1" }, git)).toBe("abc");
  });

  it("returns undefined for an unresolvable RULES_BASE_REF", () => {
    expect(chooseBase({ RULES_BASE_REF: "nope" }, fakeGit({}))).toBeUndefined();
  });

  it("falls back to the merge-base with origin/main", () => {
    const git = fakeGit({ "merge-base HEAD origin/main": "def" });
    expect(chooseBase({}, git)).toBe("def");
    expect(chooseBase({ RULES_BASE_REF: "" }, git)).toBe("def");
  });
});

describe("historicalSnapshots", () => {
  const listing = `rev-list --first-parent base -- ${RULES_HISTORY_PATHS.join(" ")}`;

  it("rebuilds every rules-touching commit that carries the builder", () => {
    const git = fakeGit({
      [listing]: "c3\nc2\nc1",
      [`cat-file -e c3:${RULES_BUILDER}`]: "",
      [`cat-file -e c2:${RULES_BUILDER}`]: "",
    });
    const warnings: string[] = [];
    expect(historicalSnapshots("base", git, built, (m) => warnings.push(m))).toEqual([built("c3"), built("c2")]);
    expect(warnings).toEqual([]);
  });

  // Everything the builder runs lives under these paths, so a commit that
  // only changes, say, canonicalJson is still rebuilt.
  it("watches the records and the whole bridge", () => {
    expect(RULES_HISTORY_PATHS).toEqual(["rules", "packages/showdown-bridge"]);
  });

  it("skips and reports a builder that no longer runs", () => {
    const git = fakeGit({ [listing]: "c2\nc1", [`cat-file -e c2:${RULES_BUILDER}`]: "", [`cat-file -e c1:${RULES_BUILDER}`]: "" });
    const warnings: string[] = [];
    const build = (commit: string): BuiltSnapshot => {
      if (commit === "c2") throw new Error("boom");
      return built(commit);
    };
    expect(historicalSnapshots("base", git, build, (m) => warnings.push(m))).toEqual([built("c1")]);
    expect(warnings).toEqual(["skipping c2: its rules builder failed (boom)"]);
  });

  it("returns nothing when git lists nothing", () => {
    expect(historicalSnapshots("base", fakeGit({}), built, () => {})).toEqual([]);
  });
});
