import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { Ajv2020 } from "ajv/dist/2020.js";
import { describe, expect, it } from "vitest";
import { PINNED_SHOWDOWN_COMMIT } from "../src/engine.ts";
import { loadRulesSnapshot } from "../src/rules-load.ts";
import { type Artifact, channelArtifact, type GitHubCli, plan, publish, snapshotArtifact } from "../src/rules-publish.ts";
import { RulesError, rulesSnapshotHash } from "../src/rules.ts";

const SCHEMAS = new URL("../../schema/rules/v1/", import.meta.url);
const schema = (name: string) => JSON.parse(readFileSync(new URL(`${name}.schema.json`, SCHEMAS), "utf8")) as Record<string, unknown>;
const snapshot = loadRulesSnapshot(PINNED_SHOWDOWN_COMMIT);
const own = rulesSnapshotHash(snapshot);
const COMMIT = "c".repeat(40);
const OLD = "0".repeat(64);
const stable = (seq: number, hash: string) => ({ channel: "stable", seq, hash });

describe("the artifacts", () => {
  it("name a snapshot after the sha256 of its exact bytes", () => {
    const artifact = snapshotArtifact(snapshot);
    expect(artifact.tag).toBe(`rules-snapshot-${own}`);
    expect(createHash("sha256").update(artifact.bytes).digest("hex")).toBe(own);
  });

  it("make a channel artifact the JCS bytes of the channel file plus its commit", () => {
    const artifact = channelArtifact(stable(6, own), COMMIT);
    expect(artifact.tag).toBe(`rules-channel-stable-${COMMIT}`);
    expect(artifact.bytes.toString("utf8")).toBe(
      `{"channel":"stable","hash":"${own}","schema_version":1,"seq":6,"source_commit":"${COMMIT}"}`,
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
  const input = (channelHash: string, base: ReturnType<typeof stable> | undefined, published = false) => ({
    snapshot,
    channels: [stable(6, channelHash)],
    base: () => base,
    sourceCommit: COMMIT,
    published: () => published,
  });

  it("always publishes the commit's own snapshot, and no channel artifact if no channel changed", () => {
    expect(plan(input(own, stable(6, own))).map((a) => a.tag)).toEqual([`rules-snapshot-${own}`]);
  });

  it("adds a channel artifact for a changed or new channel", () => {
    expect(plan(input(own, stable(5, own))).map((a) => a.kind)).toEqual(["snapshot", "channel"]);
    expect(plan(input(own, undefined)).map((a) => a.kind)).toEqual(["snapshot", "channel"]);
  });

  it("allows a rollback to an already published snapshot", () => {
    expect(plan(input(OLD, stable(5, own), true)).at(-1)?.tag).toBe(`rules-channel-stable-${COMMIT}`);
  });

  it("refuses a channel naming a snapshot the importer could never find", () => {
    expect(() => plan(input(OLD, stable(5, own)))).toThrow(RulesError);
    expect(() => plan(input(OLD, stable(5, own)))).toThrow(/neither this commit's .* nor published/);
  });
});

describe("publish", () => {
  const artifacts: Artifact[] = [snapshotArtifact(snapshot), channelArtifact(stable(6, own), COMMIT)];
  const file = (a: Artifact) => `dist/${a.tag}.json`;

  it("creates each missing Release at the commit and skips those that exist", () => {
    const calls: string[][] = [];
    const gh: GitHubCli = (args) => {
      calls.push(args);
      return !(args[1] === "view" && args[2] === artifacts[1]?.tag);
    };
    expect(publish(artifacts, COMMIT, file, gh)).toEqual([artifacts[1]?.tag]);
    const create = calls.find((c) => c[1] === "create");
    expect(create?.slice(0, 6)).toEqual(["release", "create", artifacts[1]?.tag, file(artifacts[1] as Artifact), "--target", COMMIT]);
  });

  it("fails when a Release cannot be created", () => {
    const gh: GitHubCli = () => false;
    expect(() => publish(artifacts, COMMIT, file, gh)).toThrow(/could not create Release rules-snapshot-/);
  });
});
