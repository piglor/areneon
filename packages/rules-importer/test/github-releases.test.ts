import { createHash } from "node:crypto";
import { mkdtemp, readdir, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { planImport } from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import {
  AttestationError,
  type AttestationVerifier,
  type CommandRunner,
  GhAttestationVerifier,
  GitHubReleaseSource,
  ReleaseSourceError,
} from "../src/index.ts";
import { fakeGh, type FakeGhOptions, FIXTURES, RECORDED_REFS } from "./fake-gh.ts";

const SNAPSHOT_TAG = "rules-snapshot-7c618d78d6f569e875d7e5be4863152599dc8f4207f6e10cfd5f12301a853fe8";
const CHANNEL_TAG = "rules-channel-stable-b226c3bfe889f8cf775933b0a964a44ef6d69913";
const SNAPSHOT_COMMIT = "35f9c5f04aecbe1dfc38e55d2599b1013f1b43b7";
const CHANNEL_COMMIT = "b226c3bfe889f8cf775933b0a964a44ef6d69913";
const RELEASES = JSON.parse(await readFile(join(FIXTURES, "gh-api-releases.json"), "utf8")) as Record<string, unknown>[][];
const bytesOf = async (tag: string): Promise<Uint8Array> => new Uint8Array(await readFile(join(FIXTURES, `${tag}.json`)));

interface Setup {
  gh?: FakeGhOptions;
  verifier?: AttestationVerifier;
}

/** A source on the fake gh, its private temp root, and every gh argv it ran. */
async function setup({ gh = {}, verifier }: Setup = {}) {
  const calls: string[][] = [];
  const run: CommandRunner = async (command, args) => {
    expect(command).toBe("gh");
    calls.push([...args]);
    return fakeGh(args, gh);
  };
  const tmpRoot = await mkdtemp(join(tmpdir(), "rules-importer-test-"));
  const source = new GitHubReleaseSource({ verifier: verifier ?? new GhAttestationVerifier({ run }), run, tmpRoot });
  return { source, calls, tmpRoot };
}

/** The recorded Releases list with release `index` replaced by `edit` of it. */
function releases(index: number, edit: (release: Record<string, unknown>) => unknown): string {
  const page = RELEASES[0] as Record<string, unknown>[];
  return JSON.stringify([page.map((release, i) => (i === index ? edit(structuredClone(release)) : release))]);
}
const firstAsset = (release: Record<string, unknown>) => (release.assets as Record<string, unknown>[])[0] as Record<string, unknown>;

async function refusal(options: Setup): Promise<string> {
  const { source, tmpRoot } = await setup(options);
  const error = await source.list().then(
    () => new Error("list resolved"),
    (e: unknown) => e as Error,
  );
  // The private download directory is removed on failure too.
  expect(await readdir(tmpRoot)).toEqual([]);
  expect(error).toBeInstanceOf(ReleaseSourceError);
  return error.message;
}

describe("GitHubReleaseSource", () => {
  it("lists every rules Release with its tag's commit and the attested bytes", async () => {
    const { source, calls, tmpRoot } = await setup();
    const assets = await source.list();
    expect(assets).toEqual([
      { tag: SNAPSHOT_TAG, name: `${SNAPSHOT_TAG}.json`, commit: SNAPSHOT_COMMIT, bytes: await bytesOf(SNAPSHOT_TAG) },
      { tag: CHANNEL_TAG, name: `${CHANNEL_TAG}.json`, commit: CHANNEL_COMMIT, bytes: await bytesOf(CHANNEL_TAG) },
    ]);
    expect(calls.map((args) => args.slice(0, 2).join(" "))).toEqual([
      "api --paginate",
      `api repos/piglor/areneon/git/ref/tags/${SNAPSHOT_TAG}`,
      "release download",
      "attestation verify",
      `api repos/piglor/areneon/git/ref/tags/${CHANNEL_TAG}`,
      "release download",
      "attestation verify",
    ]);
    // Each Release downloads into its own directory under one private root, removed afterwards.
    const [, , snapshotDownload, snapshotVerify] = calls;
    expect(snapshotDownload?.slice(2, 7)).toEqual([SNAPSHOT_TAG, "--repo", "piglor/areneon", "--pattern", `${SNAPSHOT_TAG}.json`]);
    expect(snapshotVerify?.at(-1)).toBe(join(snapshotDownload?.at(-1) as string, `${SNAPSHOT_TAG}.json`));
    expect(snapshotDownload?.at(-1)).toMatch(new RegExp(`^${tmpRoot}/areneon-rules-[^/]+/0$`));
    expect(await readdir(tmpRoot)).toEqual([]);
  });

  it("feeds the planner a valid release set", async () => {
    const { source } = await setup();
    const plan = await planImport(source, { snapshots: new Set() });
    expect(plan.steps.map((step) => [step.kind, step.source_commit])).toEqual([
      ["snapshot", SNAPSHOT_COMMIT],
      ["channel", CHANNEL_COMMIT],
    ]);
  });

  it("imports only the bytes the verifier returns, never the downloaded file", async () => {
    const attested = new TextEncoder().encode("attested");
    const verifier: AttestationVerifier = { verify: async () => attested };
    // Without GitHub's digest only the attestation binds the bytes.
    const gh = { releases: JSON.stringify([RELEASES[0]?.map((r) => ({ ...r, assets: [{ ...firstAsset(r), digest: null }] }))]) };
    const { source } = await setup({ gh, verifier });
    expect((await source.list()).map((asset) => asset.bytes)).toEqual([attested, attested]);
  });

  it("refuses attested bytes whose sha256 is not the digest GitHub reports", async () => {
    const verifier: AttestationVerifier = { verify: async () => new TextEncoder().encode("other") };
    const other = createHash("sha256").update("other").digest("hex");
    expect(await refusal({ verifier })).toContain(`attested bytes have sha256 ${other}`);
  });

  it("propagates the verifier's refusal and removes the download", async () => {
    const { source, tmpRoot } = await setup({ gh: { served: () => join(FIXTURES, "gh-verify-unsigned.stderr") } });
    await expect(source.list()).rejects.toBeInstanceOf(AttestationError);
    expect(await readdir(tmpRoot)).toEqual([]);
  });

  it("ignores drafts and Releases that are not rules Releases", async () => {
    const extra = [{ tag_name: "v1.0.0", draft: false, prerelease: true, assets: [] }];
    const gh = { releases: releases(0, (r) => ({ ...r, draft: true })).replace(/^\[/, `[${JSON.stringify(extra)},`) };
    const { source } = await setup({ gh });
    expect((await source.list()).map((asset) => asset.tag)).toEqual([CHANNEL_TAG]);
  });

  it("dereferences an annotated tag to its commit", async () => {
    const tagObject = "c".repeat(40);
    const nested = "d".repeat(40);
    const gh = {
      refs: { ...RECORDED_REFS, [CHANNEL_TAG]: { ref: `refs/tags/${CHANNEL_TAG}`, object: { sha: tagObject, type: "tag" } } },
      tags: {
        [tagObject]: { object: { sha: nested, type: "tag" } },
        [nested]: { object: { sha: CHANNEL_COMMIT, type: "commit" } },
      },
    };
    const { source } = await setup({ gh });
    expect((await source.list()).map((asset) => asset.commit)).toEqual([SNAPSHOT_COMMIT, CHANNEL_COMMIT]);
  });

  it("refuses a tag chain longer than it follows", async () => {
    const loop = "e".repeat(40);
    const gh = {
      refs: { ...RECORDED_REFS, [SNAPSHOT_TAG]: { ref: `refs/tags/${SNAPSHOT_TAG}`, object: { sha: loop, type: "tag" } } },
      tags: { [loop]: { object: { sha: loop, type: "tag" } } },
    };
    expect(await refusal({ gh })).toContain(`does not resolve to a commit (tag ${loop})`);
  });

  const ref = (object: unknown, name = `refs/tags/${SNAPSHOT_TAG}`) => ({
    refs: { ...RECORDED_REFS, [SNAPSHOT_TAG]: { ref: name, object } },
  });

  it.each([
    ["a tag pointing at a tree", ref({ sha: "f".repeat(40), type: "tree" }), "does not resolve to a commit (tree"],
    ["a short object id", ref({ sha: "abc", type: "commit" }), "points at no 40-hex object"],
    ["a ref for another tag", ref({ sha: SNAPSHOT_COMMIT, type: "commit" }, "refs/tags/other"), "did not resolve to exactly"],
    ["a ref list instead of one ref", { refs: { ...RECORDED_REFS, [SNAPSHOT_TAG]: [] } }, "did not resolve to exactly"],
    ["an annotated tag that is gone", ref({ sha: "a".repeat(40), type: "tag" }), "Not Found (HTTP 404)"],
    ["an annotated tag without an object", { ...ref({ sha: "a".repeat(40), type: "tag" }), tags: { ["a".repeat(40)]: [] } }, "points at no 40-hex object"],
    ["a missing tag", { refs: {} }, "exited 1: gh: Not Found (HTTP 404)"],
    ["malformed JSON", { releases: "[[" }, "printed malformed JSON"],
    ["a list that is not of pages", { releases: "[{}]" }, "did not return a list of pages"],
    ["a non-list", { releases: "{}" }, "did not return a list of pages"],
    ["a Release without a tag", { releases: releases(0, ({ tag_name, ...r }) => r) }, "without a tag_name and draft flag"],
    ["a Release without a draft flag", { releases: releases(0, ({ draft, ...r }) => r) }, "without a tag_name and draft flag"],
    ["a Release that is not an object", { releases: releases(0, () => "x") }, "without a tag_name and draft flag"],
    ["a tag with unsafe characters", { releases: releases(0, (r) => ({ ...r, tag_name: "rules-snapshot-../x" })) }, "unexpected characters"],
    ["a pre-release", { releases: releases(1, (r) => ({ ...r, prerelease: true })) }, `Release ${CHANNEL_TAG} is a pre-release`],
    ["a Release without a pre-release flag", { releases: releases(1, ({ prerelease, ...r }) => r) }, "is a pre-release"],
    ["a Release listed twice", { releases: releases(1, () => RELEASES[0]?.[0]) }, "is listed twice"],
    ["a Release without assets", { releases: releases(0, (r) => ({ ...r, assets: [] })) }, "has no assets"],
    ["a Release whose assets are not a list", { releases: releases(0, (r) => ({ ...r, assets: {} })) }, "has no assets"],
    ["an asset name with a path", { releases: releases(0, (r) => ({ ...r, assets: [{ ...firstAsset(r), name: "../x.json" }] })) }, "without a plain name"],
    ["an asset that is not an object", { releases: releases(0, (r) => ({ ...r, assets: [null] })) }, "without a plain name"],
    ["two assets of one name", { releases: releases(0, (r) => ({ ...r, assets: [firstAsset(r), firstAsset(r)] })) }, "has two assets named"],
    ["an asset still uploading", { releases: releases(0, (r) => ({ ...r, assets: [{ ...firstAsset(r), state: "starter" }] })) }, "is not fully uploaded"],
    ["a digest of another algorithm", { releases: releases(0, (r) => ({ ...r, assets: [{ ...firstAsset(r), digest: "sha512:00" }] })) }, "unexpected digest"],
    ["a digest that is not text", { releases: releases(0, (r) => ({ ...r, assets: [{ ...firstAsset(r), digest: 7 }] })) }, "unexpected digest"],
    ["an asset gh cannot download", { releases: releases(0, (r) => ({ ...r, assets: [{ ...firstAsset(r), name: "other.json" }] })) }, "gh release download exited 1: no assets match"],
  ])("refuses %s", async (_, gh, message) => {
    expect(await refusal({ gh: gh as FakeGhOptions })).toContain(message);
  });

  it("refuses when gh cannot be run", async () => {
    const run: CommandRunner = async () => {
      throw new Error("spawn gh ENOENT");
    };
    const source = new GitHubReleaseSource({ verifier: new GhAttestationVerifier({ run }), run, gh: "gh" });
    await expect(source.list()).rejects.toThrow("could not run gh api: Error: spawn gh ENOENT");
  });

  it("defaults to gh on PATH and the rules-publish repository", async () => {
    const seen: string[] = [];
    const run: CommandRunner = async (command, args) => {
      seen.push(command, ...args);
      return { code: 1, stdout: "", stderr: "offline" };
    };
    const source = new GitHubReleaseSource({ verifier: new GhAttestationVerifier({ run }), run });
    await expect(source.list()).rejects.toThrow("gh api --paginate exited 1: offline");
    expect(seen).toEqual(["gh", "api", "--paginate", "--slurp", "repos/piglor/areneon/releases?per_page=100"]);
  });
});
