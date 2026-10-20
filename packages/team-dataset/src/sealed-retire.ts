// Retirement and reveal of a sealed set (evaluation protocol L5), and the
// check a third party runs on the reveal.
//
// Retire. A set may be revealed only after the register logs its retirement
// (a `set_retirement` entry whose `reveal_hash` is the hash of the reveal this
// module builds). `revealHash` gives that hash first, from the store, without
// writing anything; `retireSealedSet` then checks the register and writes the
// bundle to a new directory: the RetiredSealedManifest (the key and the salt),
// the SealedTeamSetManifest, the commitment, the teams, the rejection log and
// the discard log. From then on the set belongs to the training pool.
//
// Verify. `verifyRetiredSealed` needs no store. It opens both commitments,
// checks the strata with the reviewed labeller (`checkReveal`), regenerates the
// set from the revealed key against the reference pool (which replays every
// L1 and L2 rejection: the log must be byte-equal) and checks each team again
// for legality, L1 and L2. Any difference is a `SealedVerifyError`.

import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  canonicalTeamHash,
  checkCommitment,
  checkReveal,
  commitmentHash,
  type Engine,
  type EngineData,
  type EvaluationConfig,
  importPaste,
  jcs,
  keyCommitment,
  readRetiredSealedManifest,
  readSealedSetCommitment,
  readSealedTeamSetManifest,
  recordHash,
  type ResolvedFormat,
  type RetiredSealedManifest,
  retiredManifestHash,
  type SealedSetCommitment,
  sealedManifestHash,
  setCommitment,
  stratumLabel,
  type TeamGeneratorConfig,
  type TeamLabelSet,
  type TeamVocabulary,
  teamVocabularyHash,
  validateTeam,
  verifyRegister,
} from "@areneon/showdown-bridge";
import { pretty } from "./dataset-io.ts";
import { Drawer } from "./draw.ts";
import { Proposal } from "./proposal.ts";
import {
  fillSealed,
  type KnownTeam,
  parseSealedTeams,
  planOfManifest,
  readStoredSet,
  referenceTeams,
  SEALED_FILES,
  type SealedTeam,
  type StoredSet,
  textSha256,
} from "./sealed.ts";
import { createPrivateDir, parseSecretJson, writeSecretFile } from "./sealed-store.ts";

export class SealedRetireError extends Error {
  override name = "SealedRetireError";
}

export class SealedVerifyError extends Error {
  override name = "SealedVerifyError";
}

/** The files of a reveal bundle. */
export const REVEAL_FILES = {
  retired: "retired-manifest.json",
  manifest: "set-manifest.json",
  commitment: "commitment.json",
  teams: "teams.jsonl",
  rejections: "rejections.jsonl",
  discards: "discards.jsonl",
} as const;

const sha256File = (text: string): string => createHash("sha256").update(text, "utf8").digest("hex");

/** The reveal of a stored set: its RetiredSealedManifest and the three files it names. */
export function revealOf(stored: StoredSet): RetiredSealedManifest {
  const { manifest, texts } = stored;
  if (textSha256(texts.rejections) !== manifest.rejection_log.log_sha256) throw new SealedRetireError("the stored rejection log is not the one the manifest names");
  if (textSha256(texts.discards) !== manifest.discards.log_sha256) throw new SealedRetireError("the stored discard log is not the one the manifest names");
  return readRetiredSealedManifest({
    schema_version: 1,
    data_class: "synthetic",
    set_id: manifest.set_id,
    set_manifest_hash: sealedManifestHash(manifest),
    reveal: { key: stored.key, salt: stored.salt },
    files: { teams_sha256: textSha256(texts.teams), rejections_sha256: textSha256(texts.rejections), discards_sha256: textSha256(texts.discards) },
  });
}

/** The `reveal_hash` for the register's `set_retirement` entry: a hash only, so it may be printed. */
export function revealHash(storeDir: string, setId: string): string {
  return retiredManifestHash(revealOf(readStoredSet(storeDir, setId)));
}

export interface RetireInput {
  storeDir: string;
  setId: string;
  /** A new directory (its parent must exist) for the bundle; the bundle is public once published, but it is written 0600 in a 0700 directory. */
  outDir: string;
  /** The register's text and the records its commitments name. */
  registerText: string;
  configs: readonly EvaluationConfig[];
}

/** Write the reveal bundle of a set the register has retired. Returns the reveal hash. */
export function retireSealedSet(input: RetireInput): string {
  const stored = readStoredSet(input.storeDir, input.setId);
  const retired = revealOf(stored);
  const state = verifyRegister(input.registerText, input.configs);
  const set = state.sets.get(input.setId);
  if (set === undefined) throw new SealedRetireError(`the register has no set ${input.setId}`);
  if (commitmentHash(set.commitment) !== commitmentHash(stored.commitment)) throw new SealedRetireError("the register committed another set under this id");
  const entry = state.entries.find((e) => e.kind === "set_retirement" && e.set_id === input.setId);
  if (entry === undefined || entry.kind !== "set_retirement" || !set.retirementLogged) throw new SealedRetireError(`the register has not retired set ${input.setId}: log set_retirement with the reveal hash first`);
  const hash = retiredManifestHash(retired);
  if (entry.reveal_hash !== hash) throw new SealedRetireError("the register's reveal_hash is not the hash of this set's reveal");
  createPrivateDir(input.outDir);
  const out = (file: string, text: string): void => writeSecretFile(join(input.outDir, file), text);
  out(REVEAL_FILES.retired, pretty(retired));
  out(REVEAL_FILES.manifest, pretty(stored.manifest));
  out(REVEAL_FILES.commitment, pretty(stored.commitment));
  out(REVEAL_FILES.teams, stored.texts.teams);
  out(REVEAL_FILES.rejections, stored.texts.rejections);
  out(REVEAL_FILES.discards, stored.texts.discards);
  return hash;
}

/** A bundle as files: the three records parsed, the three logs as text. */
export interface RevealBundle {
  retired: unknown;
  manifest: unknown;
  commitment: unknown;
  teams: string;
  rejections: string;
  discards: string;
}

export function readRevealDir(dir: string): RevealBundle {
  const text = (file: string): string => readFileSync(join(dir, file), "utf8");
  return {
    retired: parseSecretJson(text(REVEAL_FILES.retired), REVEAL_FILES.retired),
    manifest: parseSecretJson(text(REVEAL_FILES.manifest), REVEAL_FILES.manifest),
    commitment: parseSecretJson(text(REVEAL_FILES.commitment), REVEAL_FILES.commitment),
    teams: text(REVEAL_FILES.teams),
    rejections: text(REVEAL_FILES.rejections),
    discards: text(REVEAL_FILES.discards),
  };
}

export interface VerifyInput {
  engine: Engine;
  format: ResolvedFormat;
  table: EngineData;
  vocabulary: TeamVocabulary;
  labelSet: TeamLabelSet;
  config: TeamGeneratorConfig;
  evaluation: EvaluationConfig;
  bundle: RevealBundle;
  /** The reference pool: every team the set was checked against (the public splits, released datasets, the training pool, earlier retired sets). */
  known: readonly KnownTeam[];
}

export interface VerifyReport {
  set_id: string;
  teams: number;
  /** The replayed log: draws and the four kinds of skipped candidate. */
  replayed: { draws: number; rejections: number; duplicates: number; near_duplicates: number; overflows: number };
}

const bad = (message: string): never => {
  throw new SealedVerifyError(message);
};

function read<T>(reader: (value: unknown) => T, value: unknown, what: string): T {
  try {
    return reader(value);
  } catch (error) {
    return bad(`${what}: ${(error as Error).message}`);
  }
}

/** Verify a reveal. Returns a report, or throws `SealedVerifyError` naming the first thing that differs. */
export function verifyRetiredSealed(input: VerifyInput): VerifyReport {
  const { bundle, engine, evaluation } = input;
  const retired = read(readRetiredSealedManifest, bundle.retired, "retired-manifest.json");
  const manifest = read(readSealedTeamSetManifest, bundle.manifest, "set-manifest.json");
  const commitment: SealedSetCommitment = read(readSealedSetCommitment, bundle.commitment, "commitment.json");
  try {
    checkCommitment(commitment, evaluation);
  } catch (error) {
    bad(`the commitment does not fit the parameters: ${(error as Error).message}`);
  }
  if (retired.set_id !== manifest.set_id || manifest.set_id !== commitment.set_id) bad("the three records name different sets");
  if (retired.set_manifest_hash !== sealedManifestHash(manifest)) bad("the reveal names another set manifest");
  if (manifest.set_commitment !== commitment.set_commitment || manifest.key_commitment !== commitment.key_commitment) bad("the set manifest and the commitment disagree");
  const files: [string, string, string][] = [
    ["teams.jsonl", bundle.teams, retired.files.teams_sha256],
    ["rejections.jsonl", bundle.rejections, retired.files.rejections_sha256],
    ["discards.jsonl", bundle.discards, retired.files.discards_sha256],
  ];
  for (const [name, text, expected] of files) if (sha256File(text) !== expected) bad(`${name} is not the file the reveal names`);
  if (sha256File(bundle.rejections) !== manifest.rejection_log.log_sha256) bad("rejections.jsonl is not the log the set manifest names");
  if (sha256File(bundle.discards) !== manifest.discards.log_sha256) bad("discards.jsonl is not the log the set manifest names");

  let teams;
  try {
    teams = parseSealedTeams(bundle.teams);
  } catch (error) {
    return bad((error as Error).message);
  }
  // L3: both commitments open, and the strata are the declared ones (checkReveal, with the stratum from each team's own label).
  if (keyCommitment(retired.reveal.key) !== commitment.key_commitment) bad("the revealed key does not open the key commitment");
  if (setCommitment(retired.reveal.salt, teams as never) !== commitment.set_commitment) bad("the revealed salt and teams do not open the set commitment");
  if (!checkReveal(commitment, { salt: retired.reveal.salt, key: retired.reveal.key, teams: teams as never }, (team) => stratumLabel((team as unknown as { stratum: number }).stratum), evaluation)) {
    bad("the revealed teams are not the declared count and strata");
  }

  const g = manifest.generator;
  if (g.config_hash !== recordHash(input.config) || g.label_set_hash !== recordHash(input.labelSet) || g.vocabulary_hash !== teamVocabularyHash(input.vocabulary)) bad("the set was made with another config, label set or vocabulary than the ones given");
  if (jcs(g.manifest as never) !== jcs(input.format.manifest as never)) bad("the set was made under another version manifest than the pinned engine's");
  if (manifest.d_team_differing_fields !== evaluation.parameters.d_team.differing_fields) bad("the set used another d_team than the parameters give");
  const reference = referenceTeams(engine, input.known, manifest.d_team_differing_fields);
  if (reference.hash !== manifest.reference_pool_hash) bad("the given reference teams are not the reference pool the set names");

  // Regenerate from the revealed key: the teams and the whole rejection log must come out the same.
  const scratch = mkdtempSync(join(tmpdir(), "sealed-verify-"));
  try {
    const drawer = new Drawer({ engine, format: input.format, proposal: new Proposal(input.vocabulary, input.table, input.config), labelSet: input.labelSet, generationKey: retired.reveal.key });
    const replay = fillSealed({ engine, drawer, stream: { config_hash: g.config_hash, plan_id: manifest.plan_id }, plan: planOfManifest(manifest), known: reference.known, near: reference.near, logPath: join(scratch, SEALED_FILES.rejections) });
    if (!replay.filled) return bad(`regenerating from the revealed key leaves strata ${replay.short.join(", ")} short`);
    if (jcs(replay.teams as never) !== jcs(teams as never)) bad("regenerating from the revealed key gives other teams");
    const log = manifest.rejection_log;
    const s = replay.summary;
    if (s.log_sha256 !== log.log_sha256 || replay.draws !== log.draws || s.rejections !== log.rejections || s.duplicates !== log.duplicates || s.near_duplicates !== log.near_duplicates || s.overflows !== log.overflows) {
      bad("the L1/L2 rejections replayed against the reference pool are not the logged ones");
    }
    checkTeams(input, teams, reference);
    return { set_id: manifest.set_id, teams: teams.length, replayed: { draws: replay.draws, rejections: s.rejections, duplicates: s.duplicates, near_duplicates: s.near_duplicates, overflows: s.overflows } };
  } finally {
    rmSync(scratch, { recursive: true, force: true });
  }
}

/** Each team again, on its own: legal for the pinned validator, and neither equal to nor within d_team of a known team (L1, L2). */
export function checkTeams(input: VerifyInput, teams: readonly SealedTeam[], reference: ReturnType<typeof referenceTeams>): void {
  for (const team of teams) {
    const sets = importPaste(input.engine, team.paste);
    if (reference.known.has(canonicalTeamHash(sets))) bad(`team at draw ${team.k} is a known team (L1)`);
    if (reference.near.near(sets)) bad(`team at draw ${team.k} is within d_team of a known team (L2)`);
    if (!validateTeam(input.engine, input.format, structuredClone(sets)).valid) bad(`team at draw ${team.k} is refused by the pinned validator`);
  }
}
