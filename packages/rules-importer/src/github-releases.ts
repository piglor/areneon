// The importer's real ReleaseSource (ADR-007 §6): the published rules
// Releases of piglor/areneon, read through the gh CLI. Every gh call goes
// through the injected CommandRunner, so tests drive it without network.
// Each asset is downloaded into a fresh private directory and attested; only
// the bytes the AttestationVerifier returns become the asset's bytes, and the
// downloaded path is never read again.

import { createHash } from "node:crypto";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ReleaseAsset, ReleaseSource } from "@areneon/showdown-bridge";
import {
  type AttestationVerifier,
  type CommandResult,
  type CommandRunner,
  RULES_PUBLISH_POLICY,
  spawnRunner,
} from "./attestation.ts";

export class ReleaseSourceError extends Error {
  override name = "ReleaseSourceError";
}

export interface GitHubReleaseSourceOptions {
  /** Attests every downloaded asset; its returned bytes are what is imported. */
  readonly verifier: AttestationVerifier;
  readonly run?: CommandRunner;
  /** The gh executable; default `gh` on PATH. */
  readonly gh?: string;
  /** `owner/name`; default the repository rules-publish releases from. */
  readonly repo?: string;
  /** Where the private download directory is made; default the OS temp dir. */
  readonly tmpRoot?: string;
}

/** The tag prefixes rules-publish gives its Releases. */
const RULES_TAG = /^rules-(?:snapshot|channel)-/;
// Tags and asset names go into API paths and gh arguments: plain characters only.
const SAFE_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;
const COMMIT = /^[0-9a-f]{40}$/;
const DIGEST = /^sha256:([0-9a-f]{64})$/;
/** gh's jq projection of each Release: only the fields read here. */
export const RELEASES_JQ = ".[] | {tag_name, draft, prerelease, assets: [.assets[] | {name, state, digest}]}";
/** An annotated tag may point at another tag; follow at most this many. */
const MAX_TAG_DEPTH = 4;

interface Asset {
  name: string;
  digest: string | undefined;
}

interface Release {
  tag: string;
  assets: Asset[];
}

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/**
 * Lists every non-draft Release of the repository whose tag starts with
 * `rules-snapshot-` or `rules-channel-`, with the commit its tag points at
 * and each asset's attested bytes. Other Releases are ignored. Fails closed
 * with a ReleaseSourceError on anything unexpected: a gh failure, an
 * unexpected API shape, a pre-release or asset-less rules Release, an asset
 * not fully uploaded, a tag that does not resolve to a commit, or a download
 * whose digest differs from the one GitHub reports.
 */
export class GitHubReleaseSource implements ReleaseSource {
  readonly #verifier: AttestationVerifier;
  readonly #run: CommandRunner;
  readonly #gh: string;
  readonly #repo: string;
  readonly #tmpRoot: string;

  constructor(options: GitHubReleaseSourceOptions) {
    this.#verifier = options.verifier;
    this.#run = options.run ?? spawnRunner;
    this.#gh = options.gh ?? "gh";
    this.#repo = options.repo ?? RULES_PUBLISH_POLICY.repo;
    this.#tmpRoot = options.tmpRoot ?? tmpdir();
  }

  async list(): Promise<ReleaseAsset[]> {
    const releases = await this.#releases();
    const dir = await mkdtemp(join(this.#tmpRoot, "areneon-rules-"));
    try {
      const assets: ReleaseAsset[] = [];
      for (const [index, release] of releases.entries()) {
        const commit = await this.#commitOf(release.tag);
        // One subdirectory per Release, so two Releases' files cannot collide.
        const releaseDir = join(dir, String(index));
        await mkdir(releaseDir, { mode: 0o700 });
        for (const asset of release.assets) {
          const bytes = await this.#download(release.tag, asset, releaseDir);
          assets.push({ tag: release.tag, name: asset.name, commit, bytes });
        }
      }
      return assets;
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  }

  async #call(args: string[]): Promise<string> {
    let result: CommandResult;
    try {
      result = await this.#run(this.#gh, args);
    } catch (cause) {
      throw new ReleaseSourceError(`could not run ${this.#gh} ${args[0]}: ${String(cause)}`, { cause });
    }
    if (result.code !== 0) {
      throw new ReleaseSourceError(`${this.#gh} ${args.slice(0, 2).join(" ")} exited ${String(result.code)}: ${result.stderr.trim()}`);
    }
    return result.stdout;
  }

  async #json(args: string[]): Promise<unknown> {
    const stdout = await this.#call(args);
    try {
      return JSON.parse(stdout) as unknown;
    } catch (cause) {
      throw new ReleaseSourceError(`${this.#gh} ${args.slice(0, 2).join(" ")} printed malformed JSON`, { cause });
    }
  }

  /**
   * Every non-draft rules Release, across all pages of the Releases API, as
   * one compact JSON line each (RELEASES_JQ), so gh's output stays small.
   */
  async #releases(): Promise<Release[]> {
    const args = ["api", "--paginate", "--jq", RELEASES_JQ, `repos/${this.#repo}/releases?per_page=100`];
    const lines = (await this.#call(args)).split("\n").filter((line) => line !== "");
    const releases: Release[] = [];
    const tags = new Set<string>();
    for (const line of lines) {
      const value = parseLine(line);
      if (!isObject(value) || typeof value.tag_name !== "string" || typeof value.draft !== "boolean") {
        throw new ReleaseSourceError("the Releases API returned a Release without a tag_name and draft flag");
      }
      const tag = value.tag_name;
      if (!RULES_TAG.test(tag) || value.draft) continue;
      if (!SAFE_NAME.test(tag)) throw new ReleaseSourceError(`Release tag ${JSON.stringify(tag)} has unexpected characters`);
      if (tags.has(tag)) throw new ReleaseSourceError(`Release ${tag} is listed twice`);
      tags.add(tag);
      if (value.prerelease !== false) throw new ReleaseSourceError(`Release ${tag} is a pre-release`);
      releases.push({ tag, assets: assetsOf(tag, value.assets) });
    }
    return releases;
  }

  /** The commit `tag` points at, dereferencing annotated tags. */
  async #commitOf(tag: string): Promise<string> {
    const ref = await this.#json(["api", `repos/${this.#repo}/git/ref/tags/${tag}`]);
    if (!isObject(ref) || ref.ref !== `refs/tags/${tag}`) {
      throw new ReleaseSourceError(`tag ${tag} did not resolve to exactly refs/tags/${tag}`);
    }
    let target = ref.object;
    for (let depth = 0; ; depth++) {
      if (!isObject(target) || typeof target.sha !== "string" || !COMMIT.test(target.sha)) {
        throw new ReleaseSourceError(`tag ${tag} points at no 40-hex object`);
      }
      if (target.type === "commit") return target.sha;
      if (target.type !== "tag" || depth === MAX_TAG_DEPTH) {
        throw new ReleaseSourceError(`tag ${tag} does not resolve to a commit (${String(target.type)} ${target.sha})`);
      }
      const annotated = await this.#json(["api", `repos/${this.#repo}/git/tags/${target.sha}`]);
      target = isObject(annotated) ? annotated.object : undefined;
    }
  }

  /** Download one asset into `dir`, attest it, and return the attested bytes. */
  async #download(tag: string, asset: Asset, dir: string): Promise<Uint8Array> {
    await this.#call(["release", "download", tag, "--repo", this.#repo, "--pattern", asset.name, "--dir", dir]);
    const bytes = await this.#verifier.verify(join(dir, asset.name));
    if (asset.digest !== undefined) {
      const actual = createHash("sha256").update(bytes).digest("hex");
      if (actual !== asset.digest) {
        throw new ReleaseSourceError(`${tag}/${asset.name}: attested bytes have sha256 ${actual}, GitHub reports ${asset.digest}`);
      }
    }
    return bytes;
  }
}

function parseLine(line: string): unknown {
  try {
    return JSON.parse(line) as unknown;
  } catch (cause) {
    throw new ReleaseSourceError("the Releases API listing has a line that is not JSON", { cause });
  }
}

function assetsOf(tag: string, value: unknown): Asset[] {
  if (!Array.isArray(value) || value.length === 0) throw new ReleaseSourceError(`Release ${tag} has no assets`);
  const names = new Set<string>();
  return value.map((asset: unknown) => {
    if (!isObject(asset) || typeof asset.name !== "string" || !SAFE_NAME.test(asset.name)) {
      throw new ReleaseSourceError(`Release ${tag} has an asset without a plain name`);
    }
    const { name } = asset;
    if (names.has(name)) throw new ReleaseSourceError(`Release ${tag} has two assets named ${name}`);
    names.add(name);
    if (asset.state !== "uploaded") throw new ReleaseSourceError(`${tag}/${name} is not fully uploaded`);
    // GitHub reports a digest for assets uploaded since mid-2025; older ones have null.
    if (asset.digest === null || asset.digest === undefined) return { name, digest: undefined };
    const digest = typeof asset.digest === "string" ? DIGEST.exec(asset.digest) : null;
    if (!digest) throw new ReleaseSourceError(`${tag}/${name} has an unexpected digest ${JSON.stringify(asset.digest)}`);
    return { name, digest: digest[1] };
  });
}
