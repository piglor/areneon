// What rules-publish (ADR-007 §6) releases for one main commit, and how.
// CI-only like rules-history.ts, deliberately not exported from the package
// index: the workflow runs scripts/rules-publish.ts, which wires git and the
// GitHub CLI to these functions.

import { jcsBytes, RULES_SCHEMA_VERSION, type ChannelFile, RulesError, type RulesSnapshot, rulesSnapshotHash, snapshotBytes } from "./rules.ts";

/** One file to release, under its own tag. */
export interface Artifact {
  kind: "snapshot" | "channel";
  /** The Release tag, and the file name without `.json`. */
  tag: string;
  bytes: Buffer;
}

export const snapshotTag = (hash: string): string => `rules-snapshot-${hash}`;

/** The exact snapshot bytes, named after their hash: published once per hash. */
export function snapshotArtifact(snapshot: RulesSnapshot): Artifact {
  return { kind: "snapshot", tag: snapshotTag(rulesSnapshotHash(snapshot)), bytes: snapshotBytes(snapshot) };
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
  return { kind: "channel", tag: `rules-channel-${channel.channel}-${sourceCommit}`, bytes: jcsBytes(body) };
}

const sameChannel = (a: ChannelFile, b: ChannelFile | undefined): boolean =>
  b !== undefined && a.channel === b.channel && a.seq === b.seq && a.hash === b.hash;

export interface PlanInput {
  /** The snapshot built from this commit's records. */
  snapshot: RulesSnapshot;
  /** Every channel file at this commit. */
  channels: readonly ChannelFile[];
  /** The same channel's file at the first parent, if it existed there. */
  base: (channel: string) => ChannelFile | undefined;
  /** This commit. */
  sourceCommit: string;
  /** Whether a snapshot Release for this hash was already published. */
  published: (hash: string) => boolean;
}

/**
 * The artifacts this commit publishes: its snapshot, and one channel
 * artifact per channel file it changes. A changed channel must name a
 * snapshot the importer can find, this commit's own or an already
 * published one; otherwise the importer would wait forever, so publishing
 * fails instead.
 */
export function plan({ snapshot, channels, base, sourceCommit, published }: PlanInput): Artifact[] {
  const own = snapshotArtifact(snapshot);
  const ownHash = rulesSnapshotHash(snapshot);
  const artifacts = [own];
  for (const channel of channels) {
    if (sameChannel(channel, base(channel.channel))) continue;
    if (channel.hash !== ownHash && !published(channel.hash)) {
      throw new RulesError(
        `rules/channels/${channel.channel}.json names snapshot ${channel.hash}, which is neither this commit's (${ownHash}) nor published`,
      );
    }
    artifacts.push(channelArtifact(channel, sourceCommit));
  }
  return artifacts;
}

/** `gh <args>`: true if it succeeded. Injected so publishing is testable. */
export type GitHubCli = (args: string[]) => boolean;

/**
 * Create each artifact's Release, at `sourceCommit`, unless its tag already
 * exists: publishing the same commit again changes nothing. `fileOf` names
 * the artifact's file on disk. Returns the tags it created.
 */
export function publish(artifacts: readonly Artifact[], sourceCommit: string, fileOf: (a: Artifact) => string, gh: GitHubCli): string[] {
  const created: string[] = [];
  for (const artifact of artifacts) {
    if (gh(["release", "view", artifact.tag])) continue;
    const notes = `ADR-007 ${artifact.kind} artifact, attested by rules-publish (build provenance).`;
    const ok = gh(["release", "create", artifact.tag, fileOf(artifact), "--target", sourceCommit, "--title", artifact.tag, "--notes", notes]);
    if (!ok) throw new RulesError(`could not create Release ${artifact.tag}`);
    created.push(artifact.tag);
  }
  return created;
}
