// What rules-publish (ADR-007 §6) releases for one main commit, and how.
// CI-only like rules-history.ts, deliberately not exported from the package
// index: the workflow runs scripts/rules-publish.ts, which wires git and the
// GitHub CLI to these functions.

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
 *   now, so the importer always reaches the current channel.
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

/**
 * The directory `build` wrote must hold exactly the planned files, byte for
 * byte, so `release` publishes nothing else.
 */
export function checkBuilt(artifacts: readonly Artifact[], files: ReadonlyMap<string, Buffer>): string[] {
  const problems: string[] = [];
  const planned = new Map(artifacts.map((a) => [`${a.tag}.json`, a.bytes]));
  for (const [name, bytes] of planned) {
    const found = files.get(name);
    if (found === undefined) problems.push(`${name} was planned but not built`);
    else if (!found.equals(bytes)) problems.push(`${name} differs from the planned bytes`);
  }
  for (const name of files.keys()) if (!planned.has(name)) problems.push(`${name} was not planned`);
  return problems;
}
