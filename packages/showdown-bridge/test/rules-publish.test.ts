import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Ajv2020 } from "ajv/dist/2020.js";
import { describe, expect, it } from "vitest";
import { PINNED_SHOWDOWN_COMMIT } from "../src/engine.ts";
import { loadRulesSnapshot } from "../src/rules-load.ts";
import {
  type Artifact,
  channelArtifact,
  artifactPath,
  channelTag,
  checkVerified,
  emitOutput,
  type GitHubCli,
  isReleased,
  PLAN_FILE,
  type PlanManifest,
  parseManifest,
  plan,
  planManifest,
  publish,
  publishOutput,
  readPlan,
  VERIFIED_DIR,
  verifiedPath,
  verifyPlan,
  writePlan,
  writeVerified,
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

describe("publish", () => {
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

  it("treats a Release another run created meanwhile as done", () => {
    const gh: GitHubCli = (args) =>
      args[1] === "create" ? undefined : JSON.stringify({ isDraft: false, assets: [{ name: `${args[2]}.json` }] });
    expect(publish(artifacts, file, gh)).toEqual([]);
  });
});

describe("the plan job's manifest", () => {
  const snap = snapshotArtifact(snapshot, HEAD);
  const chan = channelArtifact(stable(6, own), EARLIER);
  const sha = (bytes: Buffer) => createHash("sha256").update(bytes).digest("hex");
  const tmp = () => mkdtempSync(join(tmpdir(), "rules-publish-"));

  it("says publish=false for an empty plan, so no approval is asked", () => {
    expect(publishOutput(planManifest(HEAD, plan({ snapshot, commit: HEAD, channels: [], released: () => true })))).toBe(
      "publish=false",
    );
  });

  it("says publish=true for a new snapshot, recording each file's sha256 in release order", () => {
    const manifest = planManifest(HEAD, plan({ snapshot, commit: HEAD, channels: [], released: () => false }));
    expect(publishOutput(manifest)).toBe("publish=true");
    expect(manifest).toEqual({ commit: HEAD, artifacts: [{ kind: "snapshot", tag: snap.tag, commit: HEAD, sha256: own }] });
  });

  it("round-trips through the workflow artifact directory", () => {
    const dir = tmp();
    const manifest = writePlan(dir, HEAD, [snap, chan]);
    expect(readFileSync(artifactPath(dir, chan))).toEqual(chan.bytes);
    const read = readPlan(dir);
    expect(read.manifest).toEqual(manifest);
    expect([...read.files.keys()].sort()).toEqual([`${chan.tag}.json`, `${snap.tag}.json`]);
    expect(manifest.artifacts.map((e) => e.sha256)).toEqual([sha(snap.bytes), sha(chan.bytes)]);
    const empty = tmp();
    expect(publishOutput(writePlan(empty, HEAD, []))).toBe("publish=false");
    expect(readPlan(empty).files.size).toBe(0);
  });

  it("refuses a directory without a plan", () => {
    expect(() => readPlan(tmp())).toThrow(/does not hold a plan/);
  });

  it("refuses any manifest shape it did not write", () => {
    const good = planManifest(HEAD, [snap]);
    const entry = good.artifacts[0];
    expect(parseManifest(JSON.stringify(good))).toEqual(good);
    const bad: unknown[] = [
      null,
      { commit: HEAD },
      { commit: HEAD, artifacts: {} },
      { commit: "HEAD", artifacts: [] },
      { ...good, extra: 1 },
      { commit: HEAD, artifacts: [{ ...entry, extra: 1 }] },
      { commit: HEAD, artifacts: [{ ...entry, kind: "other" }] },
      { commit: HEAD, artifacts: [{ ...entry, tag: "../../evil" }] },
      { commit: HEAD, artifacts: [{ ...entry, tag: 1 }] },
      { commit: HEAD, artifacts: [{ ...entry, commit: "x" }] },
      { commit: HEAD, artifacts: [{ ...entry, sha256: "x" }] },
      { commit: HEAD, artifacts: [entry, entry] },
    ];
    for (const value of bad) expect(() => parseManifest(JSON.stringify(value))).toThrow(RulesError);
    expect(() => parseManifest("{")).toThrow(/not JSON/);
  });
});

describe("verifyPlan (the publish job's check before attesting)", () => {
  const snap = snapshotArtifact(snapshot, HEAD);
  const chan = channelArtifact(stable(6, own), EARLIER);
  const manifest: PlanManifest = planManifest(HEAD, [snap, chan]);
  const built = new Map([snap, chan].map((a) => [`${a.tag}.json`, a.bytes]));

  it("passes the plan job's files and returns them, in plan order", () => {
    const { release, skipped } = verifyPlan(manifest, HEAD, built, [snap, chan]);
    expect(release.map((a) => a.tag)).toEqual([snap.tag, chan.tag]);
    expect(release[0]?.bytes).toBe(built.get(`${snap.tag}.json`));
    expect(skipped).toEqual([]);
  });

  it("skips what another run released meanwhile", () => {
    const { release, skipped } = verifyPlan(manifest, HEAD, built, [chan]);
    expect([release.map((a) => a.tag), skipped]).toEqual([[chan.tag], [snap.tag]]);
  });

  it("refuses a file whose bytes do not match the manifest's sha256", () => {
    const changed = new Map(built).set(`${snap.tag}.json`, Buffer.from("{}"));
    expect(() => verifyPlan(manifest, HEAD, changed, [snap, chan])).toThrow(
      `${snap.tag}.json does not match its sha256 in the manifest`,
    );
  });

  it("refuses a manifest that disagrees with the plan made again now", () => {
    const forged = { ...manifest, artifacts: manifest.artifacts.map((e) => ({ ...e, sha256: "0".repeat(64) })) };
    expect(() => verifyPlan(forged, HEAD, built, [snap, chan])).toThrow(
      `${snap.tag} in the manifest differs from what this commit plans`,
    );
    const moved = { ...manifest, artifacts: manifest.artifacts.map((e) => ({ ...e, commit: OLD.slice(0, 40) })) };
    expect(() => verifyPlan(moved, HEAD, built, [snap, chan])).toThrow(/differs from what this commit plans/);
    const relabelled = { ...manifest, artifacts: manifest.artifacts.map((e) => ({ ...e, kind: "channel" as const })) };
    expect(() => verifyPlan(relabelled, HEAD, built, [snap, chan])).toThrow(/differs from what this commit plans/);
  });

  it("refuses missing, stray and unplanned files, and another commit's plan", () => {
    const only = planManifest(HEAD, [chan]);
    const onlyFiles = new Map([[`${chan.tag}.json`, chan.bytes]]);
    expect(() => verifyPlan(only, HEAD, onlyFiles, [snap, chan])).toThrow(`${snap.tag} is planned now but not in the manifest`);
    expect(() => verifyPlan(manifest, HEAD, onlyFiles, [snap, chan])).toThrow(`${snap.tag}.json is in the manifest but missing`);
    const stray = new Map(built).set("evil.json", Buffer.from("{}"));
    expect(() => verifyPlan(manifest, HEAD, stray, [snap, chan])).toThrow("evil.json is not in the manifest");
    expect(() => verifyPlan(manifest, EARLIER, built, [snap, chan])).toThrow(`the plan is for ${HEAD}, not this commit ${EARLIER}`);
  });

  it("refuses a file swapped into the downloaded directory on disk", () => {
    const dir = mkdtempSync(join(tmpdir(), "rules-publish-"));
    writePlan(dir, HEAD, [snap, chan]);
    writeFileSync(artifactPath(dir, snap), "{}");
    const { manifest: read, files } = readPlan(dir);
    expect(read).toEqual(JSON.parse(readFileSync(join(dir, PLAN_FILE), "utf8")));
    expect(() => verifyPlan(read, HEAD, files, [snap, chan])).toThrow(/does not match its sha256/);
  });
});

describe("the verified release set (what the publish job attests and releases)", () => {
  const snap = snapshotArtifact(snapshot, HEAD);
  const chan = channelArtifact(stable(6, own), EARLIER);
  const planned = () => {
    const dir = mkdtempSync(join(tmpdir(), "rules-publish-"));
    writePlan(dir, HEAD, [snap, chan]);
    return dir;
  };
  const verify = (dir: string, replanned: Artifact[]) => {
    const { manifest, files } = readPlan(dir);
    return verifyPlan(manifest, HEAD, files, replanned).release;
  };

  it("holds only the entries the plan made again still has, never a skipped one", () => {
    const dir = planned();
    const release = verify(dir, [chan]);
    expect(writeVerified(dir, release)).toBe("attest=true");
    expect(readdirSync(join(dir, VERIFIED_DIR))).toEqual([`${chan.tag}.json`]);
    expect(readFileSync(verifiedPath(dir, chan))).toEqual(chan.bytes);
    expect(existsSync(verifiedPath(dir, snap))).toBe(false);
    expect(() => checkVerified(dir, release)).not.toThrow();
  });

  it("gives attest=false and an empty set when every entry was released meanwhile", () => {
    const dir = planned();
    const release = verify(dir, []);
    expect(writeVerified(dir, release)).toBe("attest=false");
    expect(readdirSync(join(dir, VERIFIED_DIR))).toEqual([]);
    expect(() => checkVerified(dir, release)).not.toThrow();
    const calls: string[][] = [];
    expect(publish(release, (a) => verifiedPath(dir, a), (args) => (calls.push(args), ""))).toEqual([]);
    expect(calls).toEqual([]);
  });

  it("refuses to overwrite an existing verified set", () => {
    const dir = planned();
    writeVerified(dir, [chan]);
    expect(() => writeVerified(dir, [chan])).toThrow(/already exists/);
  });

  it("keeps the cause when the verified set cannot be created for another reason", () => {
    const missing = join(planned(), "no-such-dir");
    let caught: unknown;
    try {
      writeVerified(missing, [chan]);
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(RulesError);
    expect((caught as Error).message).toMatch(/^could not create .*no-such-dir.*verified: ENOENT/);
    expect(((caught as Error).cause as NodeJS.ErrnoException).code).toBe("ENOENT");
  });

  it("makes release refuse a verified set that is missing, short, padded or changed", () => {
    const dir = planned();
    expect(() => checkVerified(dir, [snap, chan])).toThrow(/run verify first/);
    writeVerified(dir, [chan]);
    expect(() => checkVerified(dir, [snap, chan])).toThrow(`${snap.tag}.json is in the release set but was not verified`);
    expect(() => checkVerified(dir, [])).toThrow(`${chan.tag}.json is not in the release set`);
    writeFileSync(verifiedPath(dir, chan), "{}");
    expect(() => checkVerified(dir, [chan])).toThrow(`${chan.tag}.json changed since verify`);
  });
});

describe("emitOutput", () => {
  it("appends the step output line to $GITHUB_OUTPUT, and does nothing without it", () => {
    const file = join(mkdtempSync(join(tmpdir(), "rules-publish-")), "out");
    emitOutput("publish=true", file);
    emitOutput("attest=false", file);
    emitOutput("publish=false", undefined);
    emitOutput("publish=false", "");
    expect(readFileSync(file, "utf8")).toBe("publish=true\nattest=false\n");
  });
});
