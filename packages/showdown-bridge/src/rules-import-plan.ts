// What the rules importer (ADR-007 §6) loads from the released artifacts,
// and in what order. Pure: the Releases come in through the `ReleaseSource`
// port, and nothing here touches the network or the database. Unlike
// rules-publish.ts (CI-only), this is exported from the package index,
// because the importer that runs it is a consumer of the package.

import { parseChannelArtifact, parseSnapshot } from "./rules-load.ts";
import { jcsBytes, sha256Hex } from "./rules.ts";

/** One file of a published (non-draft) Release. */
export interface ReleaseAsset {
  tag: string;
  name: string;
  /** The commit the Release's tag points at (rules-publish's `Artifact.commit`). */
  commit: string;
  bytes: Uint8Array;
}

/** Lists the assets of every published rules Release; the real one wraps GitHub. */
export interface ReleaseSource {
  list(): Promise<ReleaseAsset[]>;
}

export interface ImportSnapshot {
  kind: "snapshot";
  hash: string;
  /** The commit the snapshot's Release tag points at. */
  source_commit: string;
  bytes: Uint8Array;
}

export interface ImportChannel {
  kind: "channel";
  channel: string;
  seq: number;
  hash: string;
  source_commit: string;
  bytes: Uint8Array;
}

export type ImportStep = ImportSnapshot | ImportChannel;

/** All snapshots first (by hash), then the channel steps by (channel, seq). */
export interface ImportPlan {
  steps: ImportStep[];
}

export class ImportPlanError extends Error {
  override name = "ImportPlanError";
}

// The names rules-publish gives its files: `<tag>.json`, the tag being
// `rules-snapshot-<sha256>` or `rules-channel-<channel>-<commit>`.
const SNAPSHOT_NAME = /^rules-snapshot-([0-9a-f]{64})\.json$/;
const CHANNEL_NAME = /^rules-channel-([a-z0-9]+(?:-[a-z0-9]+)*)-([0-9a-f]{40})\.json$/;

const COMMIT = /^[0-9a-f]{40}$/;

const compare = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);

const decoder = new TextDecoder("utf-8", { fatal: true });

/** The asset's JSON, which must be exactly its JCS bytes as published. */
function parseCanonical(asset: ReleaseAsset): unknown {
  let value: unknown;
  try {
    value = JSON.parse(decoder.decode(asset.bytes)) as unknown;
  } catch (error) {
    throw new ImportPlanError(`${asset.name} is not UTF-8 JSON`, { cause: error });
  }
  if (value === null || typeof value !== "object" || !jcsBytes(value).equals(asset.bytes)) {
    throw new ImportPlanError(`${asset.name} is not in canonical (JCS) form`);
  }
  return value;
}

/** Run a rules validator, reporting its refusal as an ImportPlanError. */
function validated<T>(parse: () => T): T {
  try {
    return parse();
  } catch (error) {
    throw new ImportPlanError((error as Error).message, { cause: error });
  }
}

function snapshotStep(asset: ReleaseAsset, hash: string): ImportSnapshot {
  if (sha256Hex(asset.bytes) !== hash) {
    throw new ImportPlanError(`${asset.name}: sha256 of its bytes is not ${hash}`);
  }
  const value = parseCanonical(asset);
  validated(() => parseSnapshot(value, asset.name));
  return { kind: "snapshot", hash, source_commit: asset.commit, bytes: asset.bytes };
}

function channelStep(asset: ReleaseAsset, channel: string, commit: string): ImportChannel {
  const value = parseCanonical(asset);
  const body = validated(() => parseChannelArtifact(value, asset.name));
  if (body.channel !== channel || body.source_commit !== commit || asset.commit !== commit) {
    throw new ImportPlanError(
      `${asset.name} (Release at ${asset.commit}) holds channel ${body.channel} at ${body.source_commit}, not ${channel} at ${commit}`,
    );
  }
  const { seq, hash, source_commit } = body;
  return { kind: "channel", channel, seq, hash, source_commit, bytes: asset.bytes };
}

/**
 * Plan the import of every released rules artifact. Each asset must be named
 * `<tag>.json` after its own Release tag, and carry the 40-hex commit that
 * tag points at; a channel's must be the commit in its name and body.
 * A snapshot's bytes must hash to
 * the hash in its name; each artifact must be canonical JCS and valid
 * against the schemas of its own `schema_version`. A channel step must name
 * a snapshot that is in this release set or in `known.snapshots` (already
 * imported), and no channel may have two steps with one seq. Every released
 * snapshot is planned, imported or not, so the importer can check stored
 * bytes against released ones. The order is fixed by content (snapshots by
 * hash, then channel steps by channel and seq), never by listing order, so
 * the same release set always gives the same plan. Fails closed: any
 * refusal, including the source's own failure, is an `ImportPlanError`.
 */
export async function planImport(
  source: ReleaseSource,
  known: { snapshots: ReadonlySet<string> },
): Promise<ImportPlan> {
  let assets: ReleaseAsset[];
  try {
    assets = await source.list();
  } catch (error) {
    throw new ImportPlanError(`could not list the released rules artifacts: ${(error as Error).message}`, {
      cause: error,
    });
  }
  const names = new Set<string>();
  const snapshots: ImportSnapshot[] = [];
  const channels: ImportChannel[] = [];
  for (const asset of assets) {
    if (asset.name !== `${asset.tag}.json`) {
      throw new ImportPlanError(`asset ${asset.name} is not named after its Release ${asset.tag}`);
    }
    if (names.has(asset.name)) throw new ImportPlanError(`asset ${asset.name} is listed twice`);
    names.add(asset.name);
    if (typeof asset.commit !== "string" || !COMMIT.test(asset.commit)) {
      throw new ImportPlanError(`asset ${asset.name}: its Release commit ${String(asset.commit)} is not a 40-hex commit`);
    }
    const snapshot = SNAPSHOT_NAME.exec(asset.name);
    const channel = CHANNEL_NAME.exec(asset.name);
    if (snapshot) snapshots.push(snapshotStep(asset, snapshot[1] as string));
    else if (channel) channels.push(channelStep(asset, channel[1] as string, channel[2] as string));
    else throw new ImportPlanError(`asset ${asset.name} is neither a rules snapshot nor a channel artifact`);
  }
  const available = new Set([...known.snapshots, ...snapshots.map((step) => step.hash)]);
  const steps = new Set<string>();
  for (const step of channels) {
    if (!available.has(step.hash)) {
      throw new ImportPlanError(
        `channel ${step.channel} seq ${step.seq} names snapshot ${step.hash}, which is neither released nor imported`,
      );
    }
    const key = `${step.channel} ${step.seq}`;
    if (steps.has(key)) throw new ImportPlanError(`channel ${step.channel} has two steps with seq ${step.seq}`);
    steps.add(key);
  }
  snapshots.sort((a, b) => compare(a.hash, b.hash));
  channels.sort((a, b) => compare(a.channel, b.channel) || a.seq - b.seq);
  return { steps: [...snapshots, ...channels] };
}
