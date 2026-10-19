// A stand-in for the gh CLI that replays GitHub's recorded answers for the
// first rules-publish run (fixtures/rules-releases/): the Releases list, the
// tag refs, the asset downloads and `gh attestation verify`. The unit tests
// use it as a CommandRunner; the end-to-end image runs it as its `gh`
// (fake-gh-cli.ts), so the production CLI runs unchanged with no network.
// Test-only: the production image contains neither file.

import { createHash } from "node:crypto";
import { copyFileSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import type { CommandResult } from "../src/index.ts";

export const FIXTURES = fileURLToPath(new URL("../../../fixtures/rules-releases/", import.meta.url));
const fixture = (name: string): string => readFileSync(join(FIXTURES, name), "utf8");

const REPO = "piglor/areneon";
const ok = (stdout: string): CommandResult => ({ code: 0, stdout, stderr: "" });
const fail = (stderr: string): CommandResult => ({ code: 1, stdout: "", stderr });

interface RecordedRef {
  ref: string;
  object: { sha: string; type: string };
}

export interface FakeGhOptions {
  /** stdout of `gh api --paginate --slurp repos/…/releases…`; default the recording. */
  readonly releases?: string;
  /** `gh api …/git/ref/tags/<tag>` answers by tag; default the recording. */
  readonly refs?: Readonly<Record<string, unknown>>;
  /** `gh api …/git/tags/<sha>` answers (annotated tags) by sha. */
  readonly tags?: Readonly<Record<string, unknown>>;
  /** The file served for a downloaded asset name; default the fixture of that name. */
  readonly served?: (name: string) => string;
}

/** The recorded answer of every rules Release tag ref. */
export const RECORDED_REFS: Readonly<Record<string, RecordedRef>> = Object.fromEntries(
  (JSON.parse(fixture("gh-api-tag-refs.json")) as RecordedRef[]).map((ref) => [ref.ref.slice("refs/tags/".length), ref]),
);

const VERIFIED = fixture("gh-verify-rules-publish.json");
const UNSIGNED = fixture("gh-verify-unsigned.stderr");
/** The sha256 subjects the recorded attestation names. */
const ATTESTED = new Set(
  (JSON.parse(VERIFIED) as { verificationResult: { statement: { subject: { digest: { sha256: string } }[] } } }[])
    .flatMap((result) => result.verificationResult.statement.subject)
    .map((subject) => subject.digest.sha256),
);

/** Answer one gh invocation (its argv without `gh`) from the recordings. */
export function fakeGh(args: readonly string[], options: FakeGhOptions = {}): CommandResult {
  const [command, ...rest] = args;
  if (command === "api") return api(rest, options);
  if (command === "release" && rest[0] === "download") return download(rest.slice(1), options);
  if (command === "attestation" && rest[0] === "verify") return verify(rest);
  return fail(`fake gh: unsupported command: ${args.join(" ")}`);
}

function api(args: readonly string[], options: FakeGhOptions): CommandResult {
  const releases = `repos/${REPO}/releases?per_page=100`;
  if (args.length === 3 && args[0] === "--paginate" && args[1] === "--slurp" && args[2] === releases) {
    return ok(options.releases ?? fixture("gh-api-releases.json"));
  }
  const [path] = args;
  if (args.length !== 1 || path === undefined) return fail(`fake gh: unsupported api call: ${args.join(" ")}`);
  const refPrefix = `repos/${REPO}/git/ref/tags/`;
  const tagPrefix = `repos/${REPO}/git/tags/`;
  const answer = path.startsWith(refPrefix)
    ? (options.refs ?? RECORDED_REFS)[path.slice(refPrefix.length)]
    : path.startsWith(tagPrefix)
      ? options.tags?.[path.slice(tagPrefix.length)]
      : undefined;
  return answer === undefined ? fail("gh: Not Found (HTTP 404)") : ok(JSON.stringify(answer));
}

function download(args: readonly string[], options: FakeGhOptions): CommandResult {
  const [tag, repoFlag, repo, patternFlag, name, dirFlag, dir] = args;
  if (args.length !== 7 || repoFlag !== "--repo" || repo !== REPO || patternFlag !== "--pattern" || dirFlag !== "--dir") {
    return fail(`fake gh: unsupported download: ${args.join(" ")}`);
  }
  if (name !== `${tag}.json`) return fail(`no assets match the file pattern`);
  copyFileSync(options.served?.(name as string) ?? join(FIXTURES, name), join(dir as string, name));
  return ok("");
}

function verify(args: readonly string[]): CommandResult {
  const file = args.at(-1) as string;
  const digest = createHash("sha256").update(readFileSync(file)).digest("hex");
  return ATTESTED.has(digest) ? ok(VERIFIED) : fail(UNSIGNED);
}
