import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { Ajv2020 } from "ajv/dist/2020.js";
import { describe, expect, it } from "vitest";
import { PINNED_SHOWDOWN_COMMIT } from "../src/engine.ts";
import { loadRulesSnapshot } from "../src/rules-load.ts";
import {
  type Artifact,
  channelArtifact,
  channelTag,
  checkBuilt,
  type GitHubCli,
  isReleased,
  plan,
  publish,
  snapshotArtifact,
  snapshotTag,
} from "../src/rules-publish.ts";
import { RulesError, rulesSnapshotHash } from "../src/rules.ts";

const SCHEMAS = new URL("../../schema/rules/v1/", import.meta.url);
const schema = (name: string) => JSON.parse(readFileSync(new URL(`${name}.schema.json`, SCHEMAS), "utf8")) as Record<string, unknown>;
const snapshot = loadRulesSnapshot(PINNED_SHOWDOWN_COMMIT);
const own = rulesSnapshotHash(snapshot);
const HEAD = "c".repeat(40);
const EARLIER = "b".repeat(40);
const OLD = "0".repeat(64);
const stable = (seq: number, hash: string) => ({ channel: "stable", seq, hash });

describe("the artifacts", () => {
  it("name a snapshot after the sha256 of its exact bytes", () => {
    const artifact = snapshotArtifact(snapshot, HEAD);
    expect([artifact.tag, artifact.commit]).toEqual([snapshotTag(own), HEAD]);
    expect(createHash("sha256").update(artifact.bytes).digest("hex")).toBe(own);
  });

  it("make a channel artifact the JCS bytes of the channel file plus its commit", () => {
    const artifact = channelArtifact(stable(6, own), EARLIER);
    expect([artifact.tag, artifact.commit]).toEqual([channelTag("stable", EARLIER), EARLIER]);
    expect(artifact.bytes.toString("utf8")).toBe(
      `{"channel":"stable","hash":"${own}","schema_version":1,"seq":6,"source_commit":"${EARLIER}"}`,
    );
    const valid = new Ajv2020({ strict: true }).compile(schema("channel-artifact"));
    expect(valid(JSON.parse(artifact.bytes.toString("utf8")))).toBe(true);
    expect(valid({ ...JSON.parse(artifact.bytes.toString("utf8")), extra: 1 })).toBe(false);
  });

  it("allow exactly the channels the channel file schema allows", () => {
    const enumOf = (s: Record<string, unknown>) => (s.properties as { channel: { enum: string[] } }).channel.enum;
    expect(enumOf(schema("channel-artifact"))).toEqual(enumOf(schema("channel")));
  });
});

describe("plan", () => {
  const input = (channelHash: string, released: (tag: string) => boolean) => ({
    snapshot,
    commit: HEAD,
    channels: [{ file: stable(6, channelHash), changedIn: EARLIER }],
    released,
  });

  it("releases the snapshot first, then the current channel, keyed by the commit that last changed it", () => {
    const artifacts = plan(input(own, () => false));
    expect(artifacts.map((a) => a.tag)).toEqual([snapshotTag(own), channelTag("stable", EARLIER)]);
  });

  it("skips whatever is already released, so a re-run releases nothing", () => {
    expect(plan(input(own, () => true))).toEqual([]);
    expect(plan(input(own, (tag) => tag === snapshotTag(own))).map((a) => a.kind)).toEqual(["channel"]);
  });

  it("allows a channel naming an older, released snapshot, and asks about that snapshot", () => {
    const asked: string[] = [];
    const artifacts = plan(
      input(OLD, (tag) => {
        asked.push(tag);
        return tag === snapshotTag(OLD);
      }),
    );
    expect(asked).toContain(snapshotTag(OLD));
    expect(artifacts.map((a) => a.kind)).toEqual(["snapshot", "channel"]);
  });

  it("refuses a channel naming a snapshot the importer could never find", () => {
    expect(() => plan(input(OLD, () => false))).toThrow(RulesError);
    expect(() => plan(input(OLD, () => false))).toThrow(/neither this commit's .* nor released/);
  });
});

describe("isReleased", () => {
  const tag = snapshotTag(own);
  const view = (isDraft: boolean, assets: string[]): GitHubCli => () =>
    JSON.stringify({ isDraft, assets: assets.map((name) => ({ name })) });

  it("counts only a published Release that carries its file", () => {
    expect(isReleased(tag, view(false, [`${tag}.json`]))).toBe(true);
    expect(isReleased(tag, view(true, [`${tag}.json`]))).toBe(false);
    expect(isReleased(tag, view(false, []))).toBe(false);
    expect(isReleased(tag, () => undefined)).toBe(false);
  });
});

describe("publish and checkBuilt", () => {
  const artifacts: Artifact[] = [snapshotArtifact(snapshot, HEAD), channelArtifact(stable(6, own), EARLIER)];
  const file = (a: Artifact) => `dist/${a.tag}.json`;

  it("creates each Release in plan order, at its own commit", () => {
    const calls: string[][] = [];
    expect(publish(artifacts, file, (args) => (calls.push(args), ""))).toEqual(artifacts.map((a) => a.tag));
    expect(calls.map((c) => [c[2], c[5]])).toEqual([
      [snapshotTag(own), HEAD],
      [channelTag("stable", EARLIER), EARLIER],
    ]);
  });

  it("fails when a Release cannot be created", () => {
    expect(() => publish(artifacts, file, () => undefined)).toThrow(/could not create Release rules-snapshot-/);
  });

  it("accepts exactly the planned files and nothing else", () => {
    const built = new Map(artifacts.map((a) => [`${a.tag}.json`, a.bytes]));
    expect(checkBuilt(artifacts, built)).toEqual([]);
    const stray = new Map([...built, ["evil.json", Buffer.from("{}")]]);
    expect(checkBuilt(artifacts, stray)).toEqual(["evil.json was not planned"]);
    const changed = new Map(built).set(`${snapshotTag(own)}.json`, Buffer.from("{}"));
    expect(checkBuilt(artifacts, changed)).toEqual([`${snapshotTag(own)}.json differs from the planned bytes`]);
    expect(checkBuilt(artifacts, new Map())).toHaveLength(2);
  });
});
