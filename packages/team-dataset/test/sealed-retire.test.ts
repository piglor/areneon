/* eslint-disable @typescript-eslint/no-non-null-assertion -- test fixtures: the arrays are known to be filled */
import { createHash } from "node:crypto";
import { mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { appendEntry, exportPaste, importPaste, retiredManifestHash, sealedManifestHash } from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it } from "vitest";
import { checkTeams, readRevealDir, REVEAL_FILES, retireSealedSet, revealHash, SealedRetireError, verifyRetiredSealed, type RevealBundle, type VerifyInput } from "../src/sealed-retire.ts";
import { generateSealedSet, readStoredSet, referenceTeams } from "../src/sealed.ts";
import { cfg, counterSecret, evaluation, freshStore, publicManifest, publicTeams } from "./sealed-support.ts";
import { engine, format, light, table, vocabulary } from "./support.ts";

const stores: { remove: () => void }[] = [];
afterAll(() => stores.forEach((s) => s.remove()));
const fresh = () => {
  const s = freshStore();
  stores.push(s);
  return s.dir;
};

const storeDir = fresh();
const made = generateSealedSet({
  engine, format, table, vocabulary, labelSet: light, config: cfg, evaluation, publicManifest, known: publicTeams,
  storeDir, setId: "set-r", generation: 1, planId: "plan-r", generatorCommit: "b".repeat(40), maxAttempts: 3, secret: counterSecret("retire"),
});
const stored = readStoredSet(storeDir, "set-r");

const draft = (extra: Record<string, unknown>) => ({ recorded_at: "2026-11-01T10:00:00Z", operator: "owner", ...extra }) as never;
function register(reveal: string, opts: { retire?: boolean } = {}): string {
  let text = appendEntry("", draft({ kind: "set_commitment", commitment: made.commitment }), [evaluation]).text;
  if (opts.retire === false) return text;
  text = appendEntry(text, draft({ kind: "incident", set_id: "set-r", cause: "sealed_artifact_read", description: "A test incident.", affected_candidates: [] }), [evaluation]).text;
  return appendEntry(text, draft({ kind: "set_retirement", set_id: "set-r", reason: "contaminated", reveal_hash: reveal }), [evaluation]).text;
}

describe("retiring a sealed set (L5)", () => {
  const hash = revealHash(storeDir, "set-r");

  it("gives the reveal hash from the store, without writing anything and without the secrets", () => {
    expect(hash).toMatch(/^[0-9a-f]{64}$/);
    expect(hash).toBe(revealHash(storeDir, "set-r"));
    expect(hash.includes(stored.key)).toBe(false);
  });

  it("refuses to reveal a set the register has not retired", () => {
    const out = join(fresh(), "bundle");
    expect(() => retireSealedSet({ storeDir, setId: "set-r", outDir: out, registerText: register(hash, { retire: false }), configs: [evaluation] })).toThrow(/has not retired set set-r/);
    expect(() => retireSealedSet({ storeDir, setId: "set-r", outDir: out, registerText: "", configs: [evaluation] })).toThrow(/no set set-r/);
    expect(() => retireSealedSet({ storeDir, setId: "set-r", outDir: out, registerText: register("0".repeat(64)), configs: [evaluation] })).toThrow(/not the hash of this set's reveal/);
    expect(readdirSyncSafe(out)).toEqual([]);
  });

  it("refuses a register that committed another set under this id", () => {
    const other = generateSealedSet({
      engine, format, table, vocabulary, labelSet: light, config: cfg, evaluation, publicManifest, known: publicTeams,
      storeDir: fresh(), setId: "set-r", generation: 1, planId: "plan-r", generatorCommit: "b".repeat(40), maxAttempts: 3, secret: counterSecret("other"),
    });
    let text = appendEntry("", draft({ kind: "set_commitment", commitment: other.commitment }), [evaluation]).text;
    text = appendEntry(text, draft({ kind: "incident", set_id: "set-r", cause: "sealed_artifact_read", description: "x", affected_candidates: [] }), [evaluation]).text;
    text = appendEntry(text, draft({ kind: "set_retirement", set_id: "set-r", reason: "contaminated", reveal_hash: hash }), [evaluation]).text;
    expect(() => retireSealedSet({ storeDir, setId: "set-r", outDir: join(fresh(), "b"), registerText: text, configs: [evaluation] })).toThrow(/committed another set/);
  });

  it("refuses a store whose logs are not the ones its manifest names", () => {
    const path = join(storeDir, "sealed", "set-r", "rejections.jsonl");
    const old = readFileSync(path, "utf8");
    writeFileSync(path, `${old}\n`);
    try {
      expect(() => revealHash(storeDir, "set-r")).toThrow(SealedRetireError);
    } finally {
      writeFileSync(path, old);
    }
  });

  it("writes the bundle to a new private directory once the register has retired the set, and a third party can verify it", () => {
    const out = join(fresh(), "bundle");
    const done = retireSealedSet({ storeDir, setId: "set-r", outDir: out, registerText: register(hash), configs: [evaluation] });
    expect(done).toBe(hash);
    expect(readdirSync(out).sort()).toEqual(Object.values(REVEAL_FILES).sort());
    expect(statSync(out).mode & 0o7777).toBe(0o700);
    for (const f of readdirSync(out)) expect(statSync(join(out, f)).mode & 0o7777).toBe(0o600);
    expect(() => retireSealedSet({ storeDir, setId: "set-r", outDir: out, registerText: register(hash), configs: [evaluation] })).toThrow(/EEXIST/);
    const bundle = readRevealDir(out);
    const retired = bundle.retired as { reveal: { key: string; salt: string } };
    expect(retired.reveal).toEqual({ key: stored.key, salt: stored.salt });
    expect(retiredManifestHash(bundle.retired as never)).toBe(hash);
    expect(sealedManifestHash(bundle.manifest as never)).toBe((bundle.retired as { set_manifest_hash: string }).set_manifest_hash);
    const report = verifyRetiredSealed(verify(bundle));
    expect(report.teams).toBe(24);
    expect(report.replayed.draws).toBe(made.manifest.rejection_log.draws);
    expect(report.replayed.overflows).toBe(made.manifest.rejection_log.overflows);
  });
});

const readdirSyncSafe = (dir: string): string[] => {
  try {
    return readdirSync(dir);
  } catch {
    return [];
  }
};

const verify = (bundle: RevealBundle, over: Partial<VerifyInput> = {}): VerifyInput => ({
  engine, format, table, vocabulary, labelSet: light, config: cfg, evaluation, bundle, known: publicTeams, ...over,
});

/** The reveal bundle as retirement writes it, in memory. */
function bundle(): RevealBundle {
  const out = join(fresh(), "b");
  const hash = revealHash(storeDir, "set-r");
  retireSealedSet({ storeDir, setId: "set-r", outDir: out, registerText: register(hash), configs: [evaluation] });
  return readRevealDir(out);
}

/** Change the bundle and keep its internal hashes consistent, so that only the deeper checks can notice. */
function rebound(b: RevealBundle, change: (manifest: Record<string, unknown>) => void): RevealBundle {
  const manifest = structuredClone(b.manifest) as Record<string, unknown>;
  change(manifest);
  const retired = { ...(b.retired as Record<string, unknown>), set_manifest_hash: sealedManifestHash(manifest as never) };
  return { ...b, manifest, retired };
}

describe("verifying a reveal", () => {
  it("fails when a file is changed (tamper)", () => {
    const b = bundle();
    expect(() => verifyRetiredSealed(verify({ ...b, teams: `${b.teams}\n` }))).toThrow(/teams.jsonl is not the file the reveal names/);
    expect(() => verifyRetiredSealed(verify({ ...b, rejections: `${b.rejections}\n` }))).toThrow(/rejections.jsonl is not the file the reveal names/);
    expect(() => verifyRetiredSealed(verify({ ...b, discards: `${b.discards}\n` }))).toThrow(/discards.jsonl is not the file the reveal names/);
  });

  it("fails when the teams change even if the reveal names the new bytes: the salt no longer opens the commitment", () => {
    const b = bundle();
    const lines = b.teams.split("\n").filter(Boolean);
    const swapped = `${[lines[1], lines[0], ...lines.slice(2)].join("\n")}\n`;
    const retired = b.retired as { files: Record<string, string> };
    const files = { ...retired.files, teams_sha256: recordHashText(swapped) };
    expect(() => verifyRetiredSealed(verify({ ...b, teams: swapped, retired: { ...retired, files } }))).toThrow(/revealed salt and teams do not open the set commitment/);
  });

  it("fails on a wrong key, a wrong salt, and a key equal to the salt", () => {
    const b = bundle();
    const retired = b.retired as { reveal: { key: string; salt: string } };
    expect(() => verifyRetiredSealed(verify({ ...b, retired: { ...retired, reveal: { ...retired.reveal, key: "1".repeat(64) } } }))).toThrow(/does not open the key commitment/);
    expect(() => verifyRetiredSealed(verify({ ...b, retired: { ...retired, reveal: { ...retired.reveal, salt: "1".repeat(64) } } }))).toThrow(/do not open the set commitment/);
    expect(() => verifyRetiredSealed(verify({ ...b, retired: { ...retired, reveal: { key: "1".repeat(64), salt: "1".repeat(64) } } }))).toThrow(/key and the salt must differ/);
  });

  it("fails on records that do not agree: set, manifest hash, commitments, parameters", () => {
    const b = bundle();
    const retired = b.retired as Record<string, unknown>;
    expect(() => verifyRetiredSealed(verify({ ...b, retired: { ...retired, set_id: "set-z" } }))).toThrow(/three records name different sets/);
    expect(() => verifyRetiredSealed(verify({ ...b, retired: { ...retired, set_manifest_hash: "0".repeat(64) } }))).toThrow(/another set manifest/);
    expect(() => verifyRetiredSealed(verify(rebound(b, (m) => (m.set_commitment = "0".repeat(64)))))).toThrow(/set manifest and the commitment disagree/);
    expect(() => verifyRetiredSealed(verify({ ...b, commitment: { ...(b.commitment as object), b_sealed: 2 } }))).toThrow(/does not fit the parameters/);
    expect(() => verifyRetiredSealed(verify({ ...b, manifest: { not: "a manifest" } }))).toThrow(/set-manifest.json/);
    expect(() => verifyRetiredSealed(verify({ ...b, retired: { not: "a reveal" } }))).toThrow(/retired-manifest.json/);
    expect(() => verifyRetiredSealed(verify({ ...b, commitment: { not: "a commitment" } }))).toThrow(/commitment.json/);
    expect(() => verifyRetiredSealed(verify(rebound(b, (m) => (m.rejection_log = { ...(m.rejection_log as object), log_sha256: "0".repeat(64) }))))).toThrow(/rejections.jsonl is not the log the set manifest names/);
    expect(() => verifyRetiredSealed(verify(rebound(b, (m) => (m.discards = { ...(m.discards as object), log_sha256: "0".repeat(64) }))))).toThrow(/discards.jsonl is not the log the set manifest names/);
  });

  it("fails on another generator, label set, vocabulary, version manifest or d_team", () => {
    const b = bundle();
    const gen = (change: Record<string, unknown>) => rebound(b, (m) => (m.generator = { ...(m.generator as object), ...change }));
    expect(() => verifyRetiredSealed(verify(gen({ config_hash: "0".repeat(64) })))).toThrow(/another config, label set or vocabulary/);
    expect(() => verifyRetiredSealed(verify(gen({ manifest: { ...made.manifest.generator.manifest, ruleset_hash: "0".repeat(64) } })))).toThrow(/another version manifest/);
    expect(() => verifyRetiredSealed(verify(rebound(b, (m) => (m.d_team_differing_fields = 3))))).toThrow(/another d_team/);
  });

  it("fails when the reference pool is not the one the set names (L1/L2 are replayed against that pool)", () => {
    const b = bundle();
    expect(() => verifyRetiredSealed(verify(b, { known: publicTeams.slice(1) }))).toThrow(/not the reference pool the set names/);
    expect(() => verifyRetiredSealed(verify(b, { known: [...publicTeams, { paste: stored.teams[0]!.paste }] }))).toThrow(/not the reference pool/);
  });

  it("fails when regenerating from the revealed key does not give the set, or its rejection log", () => {
    const b = bundle();
    expect(() => verifyRetiredSealed(verify(rebound(b, (m) => (m.plan_id = "another-plan"))))).toThrow(/gives other teams/);
    expect(() => verifyRetiredSealed(verify(rebound(b, (m) => (m.rejection_log = { ...(m.rejection_log as object), rejections: 999 }))))).toThrow(/replayed against the reference pool are not the logged ones/);
    expect(() => verifyRetiredSealed(verify(rebound(b, (m) => (m.d_max = 3))))).toThrow(/leaves strata .* short/);
  });

  it("checks each team again for L1, L2 and legality", () => {
    const teams = stored.teams;
    const reference = referenceTeams(engine, publicTeams, 2);
    expect(() => checkTeams(verify(bundle()), teams, reference)).not.toThrow();
    const known = referenceTeams(engine, [{ paste: teams[0]!.paste }], 2);
    expect(() => checkTeams(verify(bundle()), teams, known)).toThrow(/is a known team \(L1\)/);
    const sets = importPaste(engine, teams[0]!.paste);
    sets[0]!.nature = sets[0]!.nature === "Hardy" ? "Serious" : "Hardy";
    const near = referenceTeams(engine, [{ paste: exportPaste(engine, sets) }], 2);
    expect(() => checkTeams(verify(bundle()), teams, near)).toThrow(/within d_team of a known team \(L2\)/);
    const illegal = importPaste(engine, teams[0]!.paste);
    illegal[0]!.moves = ["splash", "splash", "splash", "splash"];
    expect(() => checkTeams(verify(bundle()), [{ ...teams[0]!, paste: exportPaste(engine, illegal) }], reference)).toThrow(/refused by the pinned validator/);
  });

  it("reads the bundle from a directory", () => {
    const out = join(fresh(), "x");
    mkdirSync(out);
    expect(() => readRevealDir(out)).toThrow(/ENOENT/);
    writeFileSync(join(out, REVEAL_FILES.retired), "{ nope");
    expect(() => readRevealDir(out)).toThrow(/retired-manifest.json is not valid JSON/);
  });
});

const recordHashText = (text: string): string => createHash("sha256").update(text, "utf8").digest("hex");
