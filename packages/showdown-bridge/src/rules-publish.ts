// What rules-publish (ADR-007 §6) releases for one main commit, and how.
// CI-only like rules-history.ts, deliberately not exported from the package
// index: the workflow runs scripts/rules-publish.ts, which wires git and the
// GitHub CLI to these functions.

import { createHash } from "node:crypto";
import { appendFileSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { type ChannelFile, jcsBytes, RULES_SCHEMA_VERSION, RulesError, type RulesSnapshot, rulesSnapshotHash, snapshotBytes } from "./rules.ts";

/** One file to release, under its own tag, at its own commit. */
export interface Artifact {
  kind: "snapshot" | "channel";
  /** The Release tag, and the file name without `.json`. */
  tag: string;
  /** The commit the Release's tag points at. */
  commit: string;
  bytes: Buffer;
}

export const snapshotTag = (hash: string): string => `rules-snapshot-${hash}`;
export const channelTag = (channel: string, commit: string): string => `rules-channel-${channel}-${commit}`;

/** The exact snapshot bytes, named after their hash: published once per hash. */
export function snapshotArtifact(snapshot: RulesSnapshot, commit: string): Artifact {
  return { kind: "snapshot", tag: snapshotTag(rulesSnapshotHash(snapshot)), commit, bytes: snapshotBytes(snapshot) };
}

/** A channel change: `jcs({schema_version, channel, seq, hash, source_commit})`. */
export function channelArtifact(channel: ChannelFile, sourceCommit: string): Artifact {
  const body = {
    schema_version: RULES_SCHEMA_VERSION,
    channel: channel.channel,
    seq: channel.seq,
    hash: channel.hash,
    source_commit: sourceCommit,
  };
  return { kind: "channel", tag: channelTag(channel.channel, sourceCommit), commit: sourceCommit, bytes: jcsBytes(body) };
}

export interface PlanInput {
  /** The snapshot built from this commit's records. */
  snapshot: RulesSnapshot;
  /** This commit. */
  commit: string;
  /** Every channel file at this commit, with the commit that last changed it. */
  channels: readonly { file: ChannelFile; changedIn: string }[];
  /** Whether a usable Release exists for this tag. */
  released: (tag: string) => boolean;
}

/**
 * The artifacts this commit must release, snapshots first so a channel never
 * names a snapshot that is not out yet:
 * - its own snapshot, unless released;
 * - for each channel, the artifact of its current file, keyed by the commit
 *   that last changed it, unless released. A channel change from an earlier
 *   run that never published (a first run, a cancelled one) is published
 *   now, so the importer always reaches the current channel. Channel steps
 *   superseded before any run released them are not in the history; the
 *   importer orders by seq, so the active channel is still right.
 * A channel must name this commit's snapshot or a released one; otherwise
 * the importer would wait forever, so planning fails instead. Rebuilding an
 * older, never-released snapshot is out of scope: re-run that commit's
 * rules-publish, or revert the records in a PR.
 */
export function plan({ snapshot, commit, channels, released }: PlanInput): Artifact[] {
  const own = snapshotArtifact(snapshot, commit);
  const ownHash = rulesSnapshotHash(snapshot);
  const artifacts = released(own.tag) ? [] : [own];
  for (const { file, changedIn } of channels) {
    if (file.hash !== ownHash && !released(snapshotTag(file.hash))) {
      throw new RulesError(
        `rules/channels/${file.channel}.json names snapshot ${file.hash}, which is neither this commit's (${ownHash}) nor released`,
      );
    }
    const artifact = channelArtifact(file, changedIn);
    if (!released(artifact.tag)) artifacts.push(artifact);
  }
  return artifacts;
}

/** `gh <args>`: its stdout, or `undefined` if it failed. Injected so publishing is testable. */
export type GitHubCli = (args: string[]) => string | undefined;

/**
 * Whether `tag` has a usable Release: published (not a draft) and carrying
 * its artifact file. A bare git tag, a draft or an upload that failed half
 * way does not count.
 */
export function isReleased(tag: string, gh: GitHubCli): boolean {
  const out = gh(["release", "view", tag, "--json", "isDraft,assets"]);
  if (out === undefined) return false;
  const view = JSON.parse(out) as { isDraft: boolean; assets: { name: string }[] };
  return !view.isDraft && view.assets.some((asset) => asset.name === `${tag}.json`);
}

/**
 * Create each planned artifact's Release, in plan order, at its commit.
 * `fileOf` names the artifact's file on disk. Returns the tags it created.
 */
export function publish(artifacts: readonly Artifact[], fileOf: (a: Artifact) => string, gh: GitHubCli): string[] {
  const created: string[] = [];
  for (const artifact of artifacts) {
    const notes = `ADR-007 ${artifact.kind} artifact, attested by rules-publish (build provenance).`;
    const args = ["release", "create", artifact.tag, fileOf(artifact), "--target", artifact.commit];
    if (gh([...args, "--title", artifact.tag, "--notes", notes]) === undefined) {
      // Another run (for another commit) may have released it meanwhile.
      if (isReleased(artifact.tag, gh)) continue;
      throw new RulesError(`could not create Release ${artifact.tag}`);
    }
    created.push(artifact.tag);
  }
  return created;
}

/** Where the plan job writes, and the publish job reads, the plan's manifest and files. */
export const PLAN_FILE = "plan.json";
export const ARTIFACTS_DIR = "artifacts";
export const artifactPath = (dir: string, artifact: Pick<Artifact, "tag">): string =>
  join(dir, ARTIFACTS_DIR, `${artifact.tag}.json`);
/** Where `verify` puts the release set: only these files are attested and released. */
export const VERIFIED_DIR = "verified";
export const verifiedPath = (dir: string, artifact: Pick<Artifact, "tag">): string =>
  join(dir, VERIFIED_DIR, `${artifact.tag}.json`);

/** One planned artifact as the manifest records it: its file is `<tag>.json`. */
export interface PlannedEntry {
  kind: Artifact["kind"];
  tag: string;
  commit: string;
  sha256: string;
}

/**
 * The plan job's output: the commit it planned for and every artifact still
 * to release, in release order, with the sha256 of the bytes it built.
 */
export interface PlanManifest {
  commit: string;
  artifacts: PlannedEntry[];
}

const sha256 = (bytes: Buffer): string => createHash("sha256").update(bytes).digest("hex");
const HEX40 = /^[0-9a-f]{40}$/;
const HEX64 = /^[0-9a-f]{64}$/;
// A tag is a file name: lowercase letters, digits and hyphens only, so it can
// never leave the artifacts directory.
const TAG = /^rules-(snapshot|channel)-[a-z0-9-]+$/;

export function planManifest(commit: string, artifacts: readonly Artifact[]): PlanManifest {
  return {
    commit,
    artifacts: artifacts.map((a) => ({ kind: a.kind, tag: a.tag, commit: a.commit, sha256: sha256(a.bytes) })),
  };
}

/** The plan job's step output: the publish job (and its approval) runs only on `publish=true`. */
export const publishOutput = (manifest: PlanManifest): string => `publish=${manifest.artifacts.length > 0}`;

/** Write the manifest and each artifact's bytes under `dir`, for the workflow artifact. */
export function writePlan(dir: string, commit: string, artifacts: readonly Artifact[]): PlanManifest {
  const manifest = planManifest(commit, artifacts);
  mkdirSync(join(dir, ARTIFACTS_DIR), { recursive: true });
  for (const a of artifacts) writeFileSync(artifactPath(dir, a), a.bytes);
  writeFileSync(join(dir, PLAN_FILE), `${JSON.stringify(manifest, null, 2)}\n`);
  return manifest;
}

const isObject = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value);
const hasKeys = (value: Record<string, unknown>, keys: string[]): boolean =>
  Object.keys(value).sort().join() === [...keys].sort().join();

/** Parse a manifest, failing closed on any shape `planManifest` does not write. */
export function parseManifest(text: string): PlanManifest {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    throw new RulesError(`${PLAN_FILE} is not JSON`);
  }
  if (!isObject(value) || !hasKeys(value, ["commit", "artifacts"]) || !Array.isArray(value.artifacts)) {
    throw new RulesError(`${PLAN_FILE} must hold exactly commit and artifacts`);
  }
  if (typeof value.commit !== "string" || !HEX40.test(value.commit)) {
    throw new RulesError(`${PLAN_FILE}: commit must be a 40-hex commit`);
  }
  const tags = new Set<string>();
  const artifacts = value.artifacts.map((entry: unknown, i): PlannedEntry => {
    if (
      !isObject(entry) ||
      !hasKeys(entry, ["kind", "tag", "commit", "sha256"]) ||
      (entry.kind !== "snapshot" && entry.kind !== "channel") ||
      typeof entry.tag !== "string" ||
      !TAG.test(entry.tag) ||
      typeof entry.commit !== "string" ||
      !HEX40.test(entry.commit) ||
      typeof entry.sha256 !== "string" ||
      !HEX64.test(entry.sha256) ||
      tags.has(entry.tag)
    ) {
      throw new RulesError(`${PLAN_FILE}: artifacts[${i}] is not a planned artifact`);
    }
    tags.add(entry.tag);
    return { kind: entry.kind, tag: entry.tag, commit: entry.commit, sha256: entry.sha256 };
  });
  return { commit: value.commit, artifacts };
}

/** Read what the plan job wrote: its manifest and every file in its artifacts directory. */
export function readPlan(dir: string): { manifest: PlanManifest; files: Map<string, Buffer> } {
  let text: string;
  let names: string[];
  try {
    text = readFileSync(join(dir, PLAN_FILE), "utf8");
    names = readdirSync(join(dir, ARTIFACTS_DIR));
  } catch {
    throw new RulesError(`${dir} does not hold a plan (${PLAN_FILE} and ${ARTIFACTS_DIR}/)`);
  }
  const files = new Map(names.map((name) => [name, readFileSync(join(dir, ARTIFACTS_DIR, name))]));
  return { manifest: parseManifest(text), files };
}

/**
 * The publish job's check before it attests and releases anything, so the
 * attested bytes are exactly the plan job's and still what this commit plans:
 * - the manifest is for `head`, the commit this job checked out;
 * - the files are exactly the manifest's, each matching its sha256;
 * - every artifact `replanned` (the plan again, now) is in the manifest with
 *   the same kind, commit and sha256. One the manifest has but the plan no
 *   longer does was released meanwhile (by another run) and is skipped.
 * Returns the artifacts to release, with the downloaded bytes, in plan order.
 */
export function verifyPlan(
  manifest: PlanManifest,
  head: string,
  files: ReadonlyMap<string, Buffer>,
  replanned: readonly Artifact[],
): { release: Artifact[]; skipped: string[] } {
  const problems: string[] = [];
  if (manifest.commit !== head) problems.push(`the plan is for ${manifest.commit}, not this commit ${head}`);
  const entries = new Map(manifest.artifacts.map((e) => [`${e.tag}.json`, e]));
  for (const name of files.keys()) if (!entries.has(name)) problems.push(`${name} is not in the manifest`);
  for (const [name, entry] of entries) {
    const found = files.get(name);
    if (found === undefined) problems.push(`${name} is in the manifest but missing`);
    else if (sha256(found) !== entry.sha256) problems.push(`${name} does not match its sha256 in the manifest`);
  }
  for (const a of replanned) {
    const entry = entries.get(`${a.tag}.json`);
    if (entry === undefined) problems.push(`${a.tag} is planned now but not in the manifest`);
    else if (entry.kind !== a.kind || entry.commit !== a.commit || entry.sha256 !== sha256(a.bytes)) {
      problems.push(`${a.tag} in the manifest differs from what this commit plans`);
    }
  }
  if (problems.length > 0) throw new RulesError(`the plan does not match:\n- ${problems.join("\n- ")}`);
  const planned = new Set(replanned.map((a) => a.tag));
  return {
    release: replanned.map((a) => ({ ...a, bytes: files.get(`${a.tag}.json`) as Buffer })),
    skipped: manifest.artifacts.filter((e) => !planned.has(e.tag)).map((e) => e.tag),
  };
}

/**
 * Append a step output line (`name=value`) to `$GITHUB_OUTPUT` when the
 * runner set it; outside Actions it is only printed by the caller.
 */
export function emitOutput(line: string, githubOutput: string | undefined): void {
  if (githubOutput) appendFileSync(githubOutput, `${line}\n`);
}

/**
 * Write exactly the release set `verifyPlan` returned into `<dir>/verified/`
 * (which must not exist yet), so the attest step signs only verified bytes and
 * never an entry that was skipped. Returns the step output `attest=true|false`.
 */
export function writeVerified(dir: string, release: readonly Artifact[]): string {
  try {
    mkdirSync(join(dir, VERIFIED_DIR));
  } catch {
    throw new RulesError(`${join(dir, VERIFIED_DIR)} already exists`);
  }
  for (const a of release) writeFileSync(verifiedPath(dir, a), a.bytes);
  return `attest=${release.length > 0}`;
}

/**
 * `release` publishes from `<dir>/verified/`: it must hold exactly the
 * release set verified again now, byte for byte.
 */
export function checkVerified(dir: string, release: readonly Artifact[]): void {
  let names: string[];
  try {
    names = readdirSync(join(dir, VERIFIED_DIR));
  } catch {
    throw new RulesError(`${join(dir, VERIFIED_DIR)} is missing: run verify first`);
  }
  const problems: string[] = [];
  const expected = new Map(release.map((a) => [`${a.tag}.json`, a.bytes]));
  for (const name of names) if (!expected.has(name)) problems.push(`${name} is not in the release set`);
  for (const [name, bytes] of expected) {
    if (!names.includes(name)) problems.push(`${name} is in the release set but was not verified`);
    else if (!readFileSync(join(dir, VERIFIED_DIR, name)).equals(bytes)) problems.push(`${name} changed since verify`);
  }
  if (problems.length > 0) throw new RulesError(`${join(dir, VERIFIED_DIR)} is not the release set:\n- ${problems.join("\n- ")}`);
}
