import { describe, expect, it } from "vitest";
import { PINNED_SHOWDOWN_COMMIT } from "../src/engine.ts";
import { ImportPlanError, planImport, type ReleaseAsset, type ReleaseSource } from "../src/rules-import-plan.ts";
import { loadRulesSnapshot } from "../src/rules-load.ts";
import { channelArtifact, snapshotArtifact } from "../src/rules-publish.ts";
import { jcsBytes, rulesSnapshotHash, type RulesSnapshot, sha256Hex } from "../src/rules.ts";

// Assets are built by rules-publish's own artifact builders, so they are
// byte for byte what a rules-publish run releases for these records.
const COMMIT = "b".repeat(40);
const LATER = "c".repeat(40);
const snapshot = loadRulesSnapshot(PINNED_SHOWDOWN_COMMIT);
const own = rulesSnapshotHash(snapshot);
const [[gapId, gap]] = Object.entries(snapshot.gaps) as [[string, RulesSnapshot["gaps"][string]]];
const older: RulesSnapshot = { ...snapshot, gaps: { ...snapshot.gaps, [gapId]: { ...gap, description: "an earlier wording" } } };
const old = rulesSnapshotHash(older);

const asAsset = ({ tag, bytes }: { tag: string; bytes: Uint8Array }): ReleaseAsset => ({ tag, name: `${tag}.json`, bytes });
const snapshotAsset = (value: RulesSnapshot) => asAsset(snapshotArtifact(value, COMMIT));
const channelAsset = (seq: number, hash: string, commit: string) =>
  asAsset(channelArtifact({ channel: "stable", seq, hash }, commit));
/** A snapshot asset for any document, named after the sha256 of its JCS bytes. */
const rawSnapshot = (value: object): ReleaseAsset => {
  const bytes = jcsBytes(value);
  return asAsset({ tag: `rules-snapshot-${sha256Hex(bytes)}`, bytes });
};
const rawChannel = (value: object, commit = COMMIT): ReleaseAsset =>
  asAsset({ tag: `rules-channel-stable-${commit}`, bytes: jcsBytes(value) });
const channelBody = { schema_version: 1, channel: "stable", seq: 5, hash: own, source_commit: COMMIT };

const source = (assets: ReleaseAsset[]): ReleaseSource => ({ list: () => Promise.resolve(assets) });
const none = { snapshots: new Set<string>() };
const plan = (assets: ReleaseAsset[], known = none) => planImport(source(assets), known);
const refuses = (assets: ReleaseAsset[], message: RegExp, known = none) =>
  expect(plan(assets, known)).rejects.toThrow(expect.objectContaining({ name: "ImportPlanError", message: expect.stringMatching(message) }));

describe("planImport", () => {
  const step6 = channelAsset(6, own, LATER);
  const step5 = channelAsset(5, old, COMMIT);
  const current = snapshotAsset(snapshot);
  const earlier = snapshotAsset(older);
  const release = [step6, current, step5, earlier];

  it("plans every snapshot first, by hash, then the channel steps by seq", async () => {
    const { steps } = await plan(release);
    expect(steps.map((step) => [step.kind, step.hash])).toEqual([
      ...[own, old].sort().map((hash) => ["snapshot", hash]),
      ["channel", old],
      ["channel", own],
    ]);
    expect(steps.slice(2)).toEqual([
      { kind: "channel", channel: "stable", seq: 5, hash: old, source_commit: COMMIT, bytes: step5.bytes },
      { kind: "channel", channel: "stable", seq: 6, hash: own, source_commit: LATER, bytes: step6.bytes },
    ]);
    const first = steps.find((step) => step.hash === own && step.kind === "snapshot");
    expect(first?.bytes).toBe(current.bytes);
  });

  it("gives the same plan for any listing order", async () => {
    const expected = await plan(release);
    expect(await plan([...release].reverse())).toEqual(expected);
    expect(await plan([step5, current, earlier, step6])).toEqual(expected);
  });

  it("plans nothing for an empty release set", async () => {
    expect(await plan([])).toEqual({ steps: [] });
  });

  it("allows a channel naming a snapshot that is already imported, not released", async () => {
    const { steps } = await plan([channelAsset(5, old, COMMIT)], { snapshots: new Set([old]) });
    expect(steps.map((step) => step.kind)).toEqual(["channel"]);
  });

  it("refuses a channel naming a snapshot neither released nor imported", async () => {
    await refuses([channelAsset(5, old, COMMIT), snapshotAsset(snapshot)], /names snapshot .* neither released nor imported/);
    await refuses([channelAsset(5, old, COMMIT)], /neither released nor imported/, { snapshots: new Set([own]) });
  });

  it("refuses two steps of one channel with the same seq", async () => {
    await refuses([snapshotAsset(snapshot), channelAsset(5, own, COMMIT), channelAsset(5, own, LATER)], /two steps with seq 5/);
  });

  it("refuses snapshot bytes whose sha256 is not the hash in their name", async () => {
    const real = snapshotAsset(snapshot);
    await refuses([{ ...real, bytes: snapshotAsset(older).bytes }], /sha256 of its bytes is not/);
    const tag = `rules-snapshot-${old}`;
    await refuses([{ tag, name: `${tag}.json`, bytes: real.bytes }], /sha256 of its bytes is not/);
  });

  it("refuses an unknown schema_version", async () => {
    await refuses([rawSnapshot({ ...snapshot, schema_version: 2 })], /no regulation schema for schema_version 2/);
    await refuses([rawSnapshot({ ...snapshot, schema_version: 2, regulations: {}, gaps: {} })], /no format schema for schema_version 2/);
    await refuses([rawSnapshot({ ...snapshot, schema_version: "1" })], /schema_version must be an integer/);
    await refuses([rawChannel({ ...channelBody, schema_version: 2 })], /no channel-artifact schema for schema_version 2/);
  });

  it("refuses a malformed asset name", async () => {
    const real = snapshotAsset(snapshot);
    const named = (name: string, tag = name.replace(/\.json$/, "")): ReleaseAsset => ({ ...real, tag, name });
    await refuses([named("rules-snapshot-abc.json")], /neither a rules snapshot nor a channel artifact/);
    await refuses([named(`rules-snapshot-${own.toUpperCase()}.json`)], /neither a rules snapshot nor a channel artifact/);
    await refuses([named(`rules-channel-stable-${"b".repeat(39)}.json`)], /neither a rules snapshot nor a channel artifact/);
    await refuses([named(`rules-channel--${COMMIT}.json`)], /neither a rules snapshot nor a channel artifact/);
    await refuses([named("notes.txt", "notes")], /not named after its Release/);
    await refuses([named(`${real.tag}.json`, `rules-snapshot-${old}`)], /not named after its Release/);
    await refuses([real, real], /listed twice/);
  });

  it("refuses a channel artifact released under another channel's or commit's name", async () => {
    await refuses([snapshotAsset(snapshot), rawChannel(channelBody, LATER)], /holds channel stable at b+, not stable at c+/);
    const wrongChannel = { ...rawChannel(channelBody), tag: `rules-channel-beta-${COMMIT}`, name: `rules-channel-beta-${COMMIT}.json` };
    await refuses([snapshotAsset(snapshot), wrongChannel], /not beta at/);
  });

  it("refuses a channel artifact its schema rejects", async () => {
    await refuses([rawChannel({ ...channelBody, seq: 0 })], /must be >= 1/);
    await refuses([rawChannel({ ...channelBody, extra: true })], /additional properties/);
  });

  it("refuses bytes that are not the canonical JCS form of an object", async () => {
    const pretty = (asset: ReleaseAsset): ReleaseAsset => ({
      ...asset,
      bytes: Buffer.from(JSON.stringify(JSON.parse(Buffer.from(asset.bytes).toString("utf8")), null, 2)),
    });
    await refuses([pretty(channelAsset(5, own, COMMIT))], /not in canonical \(JCS\) form/);
    const reordered = Buffer.from(`{"seq":5,${jcsBytes(channelBody).toString("utf8").slice(1).replace(',"seq":5', "")}`);
    await refuses([{ ...rawChannel(channelBody), bytes: reordered }], /not in canonical/);
    await refuses([{ ...rawChannel(channelBody), bytes: Buffer.from("1") }], /not in canonical/);
    await refuses([{ ...rawChannel(channelBody), bytes: Buffer.from("null") }], /not in canonical/);
    await refuses([{ ...rawChannel(channelBody), bytes: Buffer.from("{") }], /not UTF-8 JSON/);
    await refuses([{ ...rawChannel(channelBody), bytes: Uint8Array.from([0x22, 0xff, 0x22]) }], /not UTF-8 JSON/);
    const bytes = Buffer.from(JSON.stringify(JSON.parse(snapshotArtifact(snapshot, COMMIT).bytes.toString("utf8")), null, 1));
    await refuses([asAsset({ tag: `rules-snapshot-${sha256Hex(bytes)}`, bytes })], /not in canonical/);
  });

  it("refuses a snapshot that is not what its own records build", async () => {
    const [[formatId, format]] = Object.entries(snapshot.formats) as [[string, RulesSnapshot["formats"][string]]];
    const withFormat = (value: unknown) => rawSnapshot({ ...snapshot, formats: { ...snapshot.formats, [formatId]: value } });
    await refuses([rawSnapshot([snapshot])], /exactly the keys/);
    await refuses([rawSnapshot({ ...snapshot, extra: 1 })], /exactly the keys/);
    await refuses([rawSnapshot({ ...snapshot, showdown_commit: "main" })], /showdown_commit must be a 40-hex commit/);
    await refuses([rawSnapshot({ ...snapshot, gaps: [] })], /gaps must be an object/);
    await refuses([withFormat("doubles")], new RegExp(`formats.${formatId} must be an object`));
    await refuses([withFormat({ ...format, game_type: "singles" })], new RegExp(`formats/${formatId}.json: .*allowed values`));
    await refuses([withFormat({ ...format, regulation: "Z-9" })], /unknown regulation Z-9/);
    await refuses([withFormat({ ...format, format_id: formatId })], /not the snapshot its records build/);
    await refuses([rawSnapshot({ ...snapshot, formats: {} })], /no format records/);
  });

  it("reports a failing source as an ImportPlanError", async () => {
    const failing: ReleaseSource = { list: () => Promise.reject(new Error("rate limited")) };
    const error = await planImport(failing, none).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ImportPlanError);
    expect((error as Error).message).toMatch(/could not list the released rules artifacts: rate limited/);
  });
});
