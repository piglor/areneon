// Sealed team generation inside the isolated evaluation store (ADR-013
// section 2 "Keys and splits" and "Shortfall"; evaluation protocol L1 to L5).
//
// The sealed set is made by the same generator, config and label set as the
// public pool, but
//   - from a 256-bit key of the OS CSPRNG that exists only in the store (L4),
//     separate from every public key and from the sealed battle key (D9);
//   - on the strata the public plan fixed: the D20 merge is NOT run again, so
//     a short bin can never trigger a merge after the public assignment. The
//     draw budget is the public `d_max` over-provisioned by the config's
//     `sealed_margin_permille`; if a bin is still short, the whole generation
//     is discarded, logged (hashes and counts) and redrawn under a new key.
//     Nothing is regenerated selectively;
//   - against everything public: L1 (equal canonical hash) and L2 (within
//     `d_team` differing fields) refuse a candidate known from any public
//     split, released dataset, the training pool or a retired sealed set;
//   - committed with L3: SHA-256(JCS({salt, teams})) and the SHA-256 of the key.
//
// What may leave this module: counts, hashes, the SealedTeamSetManifest and
// the SealedSetCommitment. The key, the salt and the teams are written only
// below the caller's store directory (0600 files in 0700 directories) and are
// never returned, logged or printed. `SealedTeamHashes` (stamping) is built in
// memory from the store and cannot be serialised.
//
// What the protocol leaves open, and this module fixes (journaled on #861):
//   - L2 "within distance d_team" is distance <= d_team (inclusive), and the
//     distance of two teams is the least number of differing fields over all
//     pairings of their sets, a field being species, item, ability, nature,
//     the move set or the Stat Points of one set. Level, IVs, gender and
//     nickname are fixed by the config and are not fields;
//   - two sealed teams are not compared by L2 with each other, only by L1;
//   - the quotas are n_teams shared over the public strata by their pilot
//     hits, largest remainder, each stratum at least n_stratum_min.

import { createHash } from "node:crypto";
import { appendFileSync, renameSync, rmSync } from "node:fs";
import { join } from "node:path";
import { inspect } from "node:util";
import {
  canonicalTeamHash,
  checkCommitment,
  type Engine,
  type EngineData,
  EvaluationConfigError,
  type EvaluationConfig,
  generateSecret,
  importPaste,
  jcs,
  keyCommitment,
  type PokemonSet,
  readSealedSetCommitment,
  readSealedTeamSetManifest,
  recordHash,
  requireComplete,
  type ResolvedFormat,
  type SealedSetCommitment,
  type SealedTeamSetManifest,
  setCommitment,
  stratumLabel,
  type SyntheticTeamHashes,
  syntheticTeamHashes,
  teamHash,
  type TeamDatasetManifest,
  type TeamGeneratorConfig,
  type TeamLabelSet,
  type TeamVocabulary,
  teamVocabularyHash,
} from "@areneon/showdown-bridge";
import { Binner } from "./binner.ts";
import { pretty } from "./dataset-io.ts";
import { KnownHashes, referencePoolHash, sortedUnique } from "./dedupe.ts";
import { Drawer } from "./draw.ts";
import { cellCount } from "./labeller.ts";
import { Proposal } from "./proposal.ts";
import { fillQuantile } from "./provision.ts";
import { RejectionLog, type LogSummary } from "./rejection-log.ts";
import { assertPrivateStore, createPrivateDir, createSecretFile, ensurePrivateDir, parseSecretJson, readSecretFile, writeSecretFile } from "./sealed-store.ts";
import { checkKey, type StreamIdentity } from "./stream.ts";

export class SealedError extends Error {
  override name = "SealedError";
}

/** Every attempt left a bin short: nothing was committed. */
export class SealedShortfallError extends SealedError {
  override name = "SealedShortfallError";
}

const fail = (message: string): never => {
  throw new SealedError(message);
};

/** The files of one sealed set, below `<store>/sealed/<set_id>/`. */
export const SEALED_FILES = {
  key: "key",
  salt: "salt",
  teams: "teams.jsonl",
  rejections: "rejections.jsonl",
  discards: "discards.jsonl",
  manifest: "manifest.json",
  commitment: "commitment.json",
} as const;

export const setDirOf = (store: string, setId: string): string => join(store, "sealed", setId);

const SET_ID = /^[a-z0-9][a-z0-9._-]{0,63}$/;
const sha256 = (text: string): string => createHash("sha256").update(text, "utf8").digest("hex");

/** One sealed team as it is committed and revealed: the draw counter, the engine's paste and the stratum. */
export interface SealedTeam {
  k: number;
  paste: string;
  stratum: number;
}

/** A team the sealed set must not repeat or approach: any object with the engine's paste (a TeamRecord, a sealed team of a retired set). */
export interface KnownTeam {
  paste: string;
}

// ---------------------------------------------------------------------------
// L2: distance between teams.

const FIELDS = 6;
const STATS = ["hp", "atk", "def", "spa", "spd", "spe"] as const;
const toId = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");

/** The six compared fields of one set. */
function setFields(set: PokemonSet): string[] {
  const evs = set.evs as unknown as Record<string, number | undefined>;
  return [toId(set.species), toId(set.item), toId(set.ability), toId(set.nature), set.moves.map(toId).sort().join(","), STATS.map((s) => evs[s] ?? 0).join(",")];
}

const setKey = (fields: readonly string[]): string => fields.join("|");
const MAX_TEAM = 12;

/**
 * The distance of two teams (L2): the least number of differing fields over
 * all pairings of their sets. Of unequal teams, the sets left unpaired count
 * as differing in every field.
 */
export function teamDistance(a: readonly PokemonSet[], b: readonly PokemonSet[]): number {
  return distanceOfFields(a.map(setFields), b.map(setFields));
}

function distanceOfFields(a: readonly string[][], b: readonly string[][]): number {
  const n = Math.max(a.length, b.length);
  if (n > MAX_TEAM) fail(`a team of ${n} sets is too large to match`);
  const cost = (i: number, j: number): number => {
    const x = a[i];
    const y = b[j];
    if (x === undefined || y === undefined) return FIELDS;
    return x.reduce((d, field, f) => d + (field === y[f] ? 0 : 1), 0);
  };
  // best[mask]: the least cost of pairing the first popcount(mask) sets of `a` with the sets of `b` in `mask`
  const best = new Array<number>(1 << n).fill(Infinity);
  best[0] = 0;
  for (let mask = 0; mask < 1 << n; mask++) {
    const i = popcount(mask);
    if (i >= n || best[mask] === Infinity) continue;
    for (let j = 0; j < n; j++) {
      if ((mask & (1 << j)) !== 0) continue;
      const next = mask | (1 << j);
      best[next] = Math.min(best[next] as number, (best[mask] as number) + cost(i, j));
    }
  }
  return best[(1 << n) - 1] as number;
}

function popcount(x: number): number {
  let n = 0;
  for (let v = x; v !== 0; v &= v - 1) n++;
  return n;
}

/**
 * The known teams, indexed to answer "is a candidate within `bound` of any of
 * them" without comparing every pair. A pairing with at most `bound` differing
 * fields leaves at most `bound` sets that are not identical, so a team that
 * shares fewer than (sets - bound) identical sets with the candidate is out; an
 * inverted index on whole sets finds the others, and only those are measured.
 * In memory: the scale-up to the shard store is a later slice.
 */
export class NearIndex {
  readonly #teams: string[][][] = [];
  readonly #postings = new Map<string, Map<number, number>>();
  readonly #bound: number;

  constructor(teams: Iterable<readonly PokemonSet[]>, bound: number) {
    if (!Number.isSafeInteger(bound) || bound < 0) fail("d_team must be a non-negative integer");
    this.#bound = bound;
    for (const team of teams) {
      const id = this.#teams.length;
      const fields = team.map(setFields);
      this.#teams.push(fields);
      for (const f of fields) {
        const key = setKey(f);
        const posting = this.#postings.get(key) ?? new Map<number, number>();
        posting.set(id, (posting.get(id) ?? 0) + 1);
        this.#postings.set(key, posting);
      }
    }
  }

  get size(): number {
    return this.#teams.length;
  }

  /** Whether some known team is within the bound of `team`. */
  near(team: readonly PokemonSet[]): boolean {
    const fields = team.map(setFields);
    const need = fields.length - this.#bound;
    if (need <= 0) return this.#teams.some((known) => distanceOfFields(fields, known) <= this.#bound);
    const wanted = new Map<string, number>();
    for (const f of fields) wanted.set(setKey(f), (wanted.get(setKey(f)) ?? 0) + 1);
    const shared = new Map<number, number>();
    for (const [key, count] of wanted) {
      for (const [id, have] of this.#postings.get(key) ?? []) shared.set(id, (shared.get(id) ?? 0) + Math.min(count, have));
    }
    for (const [id, count] of shared) {
      if (count >= need && distanceOfFields(fields, this.#teams[id] as string[][]) <= this.#bound) return true;
    }
    return false;
  }
}

// ---------------------------------------------------------------------------
// The sealed plan.

export interface SealedStratum {
  stratum: number;
  pilot_hits: number;
  quota: number;
}

export interface SealedPlan {
  teamCount: number;
  strata: SealedStratum[];
  mergedCells: { stratum: number; cells: number[] }[];
  /** The stratum of a base cell under the public D20 merge. */
  stratumOfCell(cell: number): number;
  /** The sealed draw budget: the public `d_max` over-provisioned. */
  dMax: number;
  pilotDraws: number;
  /** Strata whose fill-time quantile is beyond the budget: the generation may fall short and be discarded. */
  warnings: string[];
}

/** ceil(dMax * (1000 + margin) / 1000), exact. */
export function sealedBudget(dMax: number, marginPermille: number): number {
  return Number((BigInt(dMax) * BigInt(1000 + marginPermille) + 999n) / 1000n);
}

/**
 * `teamCount` shared over the strata by their pilot hits: largest remainder,
 * ties to the lower stratum, each stratum at least `min`. The sum is exactly
 * `teamCount`, which the commitment must be (checkCommitment).
 */
export function sealedQuotas(hits: readonly number[], teamCount: number, min: number): number[] {
  const strata = hits.length;
  if (strata * min > teamCount) fail(`${strata} strata of at least ${min} teams need ${strata * min} teams, n_teams is ${teamCount}`);
  if (hits.some((h) => h < 1)) fail("a stratum with no pilot hit never fills");
  const quota = new Array<number>(strata).fill(0);
  const active = new Set(hits.keys());
  for (let again = true; again; ) {
    again = false;
    const budget = BigInt(teamCount - (strata - active.size) * min);
    const total = BigInt([...active].reduce((sum, s) => sum + (hits[s] as number), 0));
    for (const s of [...active]) {
      if (budget * BigInt(hits[s] as number) < BigInt(min) * total) {
        quota[s] = min;
        active.delete(s);
        again = true;
      }
    }
  }
  const members = [...active].sort((a, b) => a - b);
  const budget = BigInt(teamCount - (strata - members.length) * min);
  const total = BigInt(members.reduce((sum, s) => sum + (hits[s] as number), 0));
  const rows = members.map((s) => ({ s, base: (budget * BigInt(hits[s] as number)) / total, rest: (budget * BigInt(hits[s] as number)) % total }));
  let left = Number(budget - rows.reduce((sum, r) => sum + r.base, 0n));
  for (const r of rows) quota[r.s] = Number(r.base);
  for (const r of [...rows].sort((x, y) => (x.rest === y.rest ? x.s - y.s : x.rest > y.rest ? -1 : 1))) {
    if (left-- > 0) quota[r.s] = (quota[r.s] as number) + 1;
  }
  return quota;
}

export interface SealedPlanInput {
  /** The public dataset the sealed set shares its D20 strata with. */
  publicManifest: TeamDatasetManifest;
  teamCount: number;
  nStratumMin: number;
  marginPermille: number;
  /** The config's fill quantile: a stratum whose quantile of fill time is beyond the budget is reported, as the public plan does. */
  fillQuantilePermille: number;
}

/** The sealed plan on the public plan's strata. Refused: a stratum that the budget cannot be expected to fill. */
export function sealedPlan(input: SealedPlanInput): SealedPlan {
  const { publicManifest: m } = input;
  const first = m.splits[0] ?? fail("the public manifest has no split");
  const dMax = sealedBudget(m.d_max, input.marginPermille);
  const quotas = sealedQuotas(first.cells.map((c) => c.pilot_hits), input.teamCount, input.nStratumMin);
  const strata = first.cells.map((c, i) => ({ stratum: c.stratum, pilot_hits: c.pilot_hits, quota: quotas[i] as number }));
  for (const s of strata) {
    // the public cost rule on the over-provisioned budget: expected draws to fill q * pilot_draws / hits
    if (BigInt(s.quota) * BigInt(m.pilot_draws) > BigInt(dMax) * BigInt(s.pilot_hits)) {
      fail(`stratum ${s.stratum} needs about ${Math.ceil((s.quota * m.pilot_draws) / s.pilot_hits)} draws for ${s.quota} teams, over the sealed budget ${dMax}; lower n_teams or raise sealed_margin_permille`);
    }
  }
  const plan = planOf(strata, m.merged_cells ?? [], dMax, m.pilot_draws, input.teamCount);
  for (const x of strata) {
    const quantile = fillQuantile(x.quota, x.pilot_hits, m.pilot_draws, input.fillQuantilePermille);
    if (quantile > dMax) plan.warnings.push(`stratum ${x.stratum}: the ${input.fillQuantilePermille}/1000 fill quantile is ${quantile} draws, beyond the sealed budget ${dMax}; a generation may fall short and be discarded`);
  }
  return plan;
}

function planOf(strata: SealedStratum[], mergedCells: { stratum: number; cells: number[] }[], dMax: number, pilotDraws: number, teamCount: number): SealedPlan {
  const owner = new Map<number, number>();
  for (const g of mergedCells) for (const cell of g.cells) owner.set(cell, g.stratum);
  return { teamCount, strata, mergedCells, stratumOfCell: (cell) => owner.get(cell) ?? cell, dMax, pilotDraws, warnings: [] };
}

/** The plan a manifest records: how a reveal is regenerated. */
export function planOfManifest(manifest: SealedTeamSetManifest): SealedPlan {
  return planOf(manifest.strata, manifest.merged_cells ?? [], manifest.d_max, manifest.pilot_draws, manifest.team_count);
}

// ---------------------------------------------------------------------------
// One generation under one key.

export interface FillArgs {
  engine: Engine;
  drawer: Drawer;
  stream: StreamIdentity;
  plan: SealedPlan;
  known: KnownHashes;
  near: NearIndex;
  /** A private file that exists already: the rejection log is written there as the run goes. */
  logPath: string;
  onProgress?: ((event: { draws: number; open: number }) => void) | undefined;
  progressEvery?: number | undefined;
}

export type FillResult = { filled: true; teams: SealedTeam[]; summary: LogSummary; draws: number } | { filled: false; short: number[]; summary: LogSummary; draws: number };

/** Draw until every stratum's bin is full or the budget is spent. Deterministic given the key (inside `drawer`), the plan and the known teams. */
export function fillSealed(args: FillArgs): FillResult {
  const { engine, drawer, plan } = args;
  const quotas = new Map(plan.strata.map((x) => [x.stratum, x.quota]));
  // The checks run here, in the binner's order (full bin, then L1, then L2), so each refusal has its own reason.
  const binner = new Binner(quotas, { has: () => false });
  const binned = new Set<string>();
  const log = new RejectionLog(args.logPath);
  const every = args.progressEvery ?? 10_000;
  let draws = 0;
  while (!binner.full && draws < plan.dMax) {
    const draw = drawer.draw(args.stream, draws);
    if (draw.status === "rejected") {
      log.write({ k: draw.k, reason: draw.reason, canonical_team_hash: draw.canonical_team_hash, detail: draw.detail });
    } else {
      const stratum = plan.stratumOfCell(draw.cell);
      const hash = draw.canonical_team_hash;
      const open = binner.bin(stratum).length < (quotas.get(stratum) as number);
      const refusal: "duplicate" | "near_duplicate" | undefined = !open
        ? undefined
        : binned.has(hash) || args.known.has(hash)
          ? "duplicate"
          : args.near.near(importPaste(engine, draw.paste))
            ? "near_duplicate"
            : undefined;
      if (refusal !== undefined) log.write({ k: draw.k, reason: refusal, canonical_team_hash: hash, stratum });
      else if (binner.offer(draw, stratum) === "overflow") log.write({ k: draw.k, reason: "overflow", canonical_team_hash: hash, stratum });
      else binned.add(hash);
    }
    draws++;
    if (args.onProgress !== undefined && draws % every === 0) args.onProgress({ draws, open: binner.short.length });
  }
  const summary = log.close();
  const short = binner.short;
  if (short.length > 0) return { filled: false, short, summary, draws };
  const teams = plan.strata.flatMap((s) => binner.bin(s.stratum).map((d) => ({ k: d.k, paste: d.paste, stratum: s.stratum })));
  return { filled: true, teams, summary, draws };
}

// ---------------------------------------------------------------------------
// What the sealed set is checked against.

export interface ReferenceTeams {
  known: KnownHashes;
  near: NearIndex;
  /** SHA-256 of the sorted canonical hashes: the manifest's `reference_pool_hash`. */
  hash: string;
}

/** The known teams of the engine's pastes, with the identities recomputed (a record's own hash fields are not trusted). */
export function referenceTeams(engine: Engine, teams: readonly KnownTeam[], dTeam: number): ReferenceTeams {
  const sets = teams.map((t, n) => {
    if (typeof t?.paste !== "string") return fail(`known team ${n + 1} has no paste`);
    try {
      return importPaste(engine, t.paste);
    } catch (error) {
      return fail(`known team ${n + 1}: ${(error as Error).message}`);
    }
  });
  const sorted = sortedUnique(sets.map((s) => canonicalTeamHash(s)));
  return { known: new KnownHashes(sorted), near: new NearIndex(sets, dTeam), hash: referencePoolHash(sorted) };
}

// ---------------------------------------------------------------------------
// Generation into the store.

export interface SealedGenerateInput {
  engine: Engine;
  format: ResolvedFormat;
  table: EngineData;
  vocabulary: TeamVocabulary;
  labelSet: TeamLabelSet;
  config: TeamGeneratorConfig;
  /** The parameters and rule in force; the parameters must be complete (n_teams, d_team measured). */
  evaluation: EvaluationConfig;
  /** The public dataset whose strata the sealed set shares. */
  publicManifest: TeamDatasetManifest;
  /** Every team the set must not repeat or approach. */
  known: readonly KnownTeam[];
  /** The store directory, private and outside every git working tree. */
  storeDir: string;
  setId: string;
  generation: number;
  planId: string;
  generatorCommit: string;
  /** How many generations may be discarded before the run gives up. */
  maxAttempts: number;
  /** 256 random bits as 64 hex digits: the key and the salt. The OS CSPRNG unless a test passes its own. */
  secret?: (() => string) | undefined;
  /** Called with counts only. */
  onProgress?: ((event: { attempt: number; draws: number; open: number }) => void) | undefined;
  progressEvery?: number | undefined;
}

/** What a run returns: hashes and counts. Never the key, the salt or a team. */
export interface SealedGenerateResult {
  manifest: SealedTeamSetManifest;
  commitment: SealedSetCommitment;
  /** The set's directory in the store. */
  dir: string;
  /** Generations discarded before the one that filled. */
  discarded: number;
  /** Strata the plan expects may fall short (counts only). */
  warnings: string[];
}

const sealedKey = (secret: () => string, what: string): string => checkKey(secret(), what);

/** The generator identity must be the public pool's: the same config, label set, vocabulary and version manifest. */
function checkSamePool(input: SealedGenerateInput): void {
  const g = input.publicManifest.generator;
  if (g.config_hash !== recordHash(input.config)) fail("the public dataset was made with another generator config");
  if (g.label_set_hash !== recordHash(input.labelSet)) fail("the public dataset was made with another label set");
  if (g.vocabulary_hash !== teamVocabularyHash(input.vocabulary)) fail("the public dataset was drawn from another vocabulary");
  if (jcs(g.manifest as never) !== jcs(input.format.manifest as never)) fail("the public dataset was generated under another version manifest than the pinned engine's");
}

/**
 * Generate a sealed set into `<store>/sealed/<set_id>/`. Refused up front:
 * a store that is not private, a set that exists, provisional parameters, a
 * public dataset of another generator, a plan the budget cannot fill. Throws
 * `SealedShortfallError` when every attempt was discarded (the discard log
 * stays in the set's directory).
 */
export function generateSealedSet(input: SealedGenerateInput): SealedGenerateResult {
  if (!SET_ID.test(input.setId)) fail("the set id must be lower case letters, digits, '.', '_' and '-' (at most 64)");
  if (!Number.isSafeInteger(input.generation) || input.generation < 1) fail("the generation must be a positive integer");
  if (!Number.isSafeInteger(input.maxAttempts) || input.maxAttempts < 1) fail("max attempts must be a positive integer");
  const store = assertPrivateStore(input.storeDir);
  const p = input.evaluation.parameters;
  try {
    requireComplete(p);
  } catch (error) {
    return fail((error as EvaluationConfigError).message);
  }
  checkSamePool(input);
  const plan = sealedPlan({ publicManifest: input.publicManifest, teamCount: p.n_teams as number, nStratumMin: p.n_stratum_min, marginPermille: input.config.sealed_margin_permille, fillQuantilePermille: input.config.fill_quantile_permille });
  if (plan.strata.some((s) => s.stratum >= cellCount(input.labelSet))) fail("the public plan names a cell outside the label set");
  const reference = referenceTeams(input.engine, input.known, p.d_team.differing_fields);
  const secret = input.secret ?? generateSecret;

  ensurePrivateDir(join(store, "sealed"));
  const dir = setDirOf(store, input.setId);
  createPrivateDir(dir);
  const discardsPath = join(dir, SEALED_FILES.discards);
  createSecretFile(discardsPath);
  const identity = { config_hash: recordHash(input.config), plan_id: input.planId };
  const proposal = new Proposal(input.vocabulary, input.table, input.config);

  for (let attempt = 1; attempt <= input.maxAttempts; attempt++) {
    const key = sealedKey(secret, "the sealed key");
    const logPath = join(dir, "rejections.attempt");
    createSecretFile(logPath);
    const drawer = new Drawer({ engine: input.engine, format: input.format, proposal, labelSet: input.labelSet, generationKey: key });
    const result = fillSealed({
      engine: input.engine,
      drawer,
      stream: identity,
      plan,
      known: reference.known,
      near: reference.near,
      logPath,
      progressEvery: input.progressEvery,
      onProgress: input.onProgress === undefined ? undefined : (e) => input.onProgress?.({ attempt, ...e }),
    });
    if (!result.filled) {
      // The generation is discarded whole: its key, teams and rejections are dropped, and only counts and the key's hash are kept.
      rmSync(logPath);
      appendFileSync(discardsPath, `${jcs({ attempt, draws: result.draws, key_commitment: keyCommitment(key), short: result.short } as never)}\n`);
      continue;
    }
    return commitSet({ input, store, dir, plan, reference, key, salt: sealedKey(secret, "the salt"), result, logPath, discarded: attempt - 1 });
  }
  throw new SealedShortfallError(`every one of ${input.maxAttempts} generations left a bin short; nothing was committed (the discard log is ${discardsPath})`);
}

interface CommitArgs {
  input: SealedGenerateInput;
  store: string;
  dir: string;
  plan: SealedPlan;
  reference: ReferenceTeams;
  key: string;
  salt: string;
  result: Extract<FillResult, { filled: true }>;
  logPath: string;
  discarded: number;
}

function commitSet(a: CommitArgs): SealedGenerateResult {
  const { input, dir, plan, result } = a;
  const evaluation = input.evaluation;
  const teams = result.teams;
  if (a.key === a.salt) fail("the key and the salt must differ");
  const commitment = readSealedSetCommitment({
    schema_version: 1,
    set_id: input.setId,
    generation: input.generation,
    set_commitment: setCommitment(a.salt, teams as never),
    key_commitment: keyCommitment(a.key),
    team_count: teams.length,
    stratum_counts: plan.strata.map((s) => ({ stratum: stratumLabel(s.stratum), count: s.quota })),
    evaluation_parameters_hash: evaluation.parameters_hash,
    training_gate_rule_hash: evaluation.rule_hash,
    b_sealed: evaluation.parameters.b_sealed,
  });
  checkCommitment(commitment, evaluation);
  const teamsText = teams.map((t) => `${jcs(t as never)}\n`).join("");
  const discardsText = readSecretFile(join(a.dir, SEALED_FILES.discards));
  const manifest = readSealedTeamSetManifest({
    schema_version: 1,
    data_class: "synthetic",
    set_id: input.setId,
    generation: input.generation,
    plan_id: input.planId,
    generator: {
      generator_commit: input.generatorCommit,
      config_hash: recordHash(input.config),
      label_set_hash: recordHash(input.labelSet),
      vocabulary_hash: teamVocabularyHash(input.vocabulary),
      manifest: input.format.manifest,
    },
    d_max: plan.dMax,
    sealed_margin_permille: input.config.sealed_margin_permille,
    d_team_differing_fields: evaluation.parameters.d_team.differing_fields,
    pilot_draws: plan.pilotDraws,
    team_count: teams.length,
    strata: plan.strata,
    ...(plan.mergedCells.length === 0 ? {} : { merged_cells: plan.mergedCells }),
    reference_pool_hash: a.reference.hash,
    rejection_log: { log_sha256: result.summary.log_sha256, draws: result.draws, rejections: result.summary.rejections, duplicates: result.summary.duplicates, near_duplicates: result.summary.near_duplicates, overflows: result.summary.overflows },
    discards: { count: a.discarded, log_sha256: sha256(discardsText) },
    set_commitment: commitment.set_commitment,
    key_commitment: commitment.key_commitment,
  });
  writeSecretFile(join(dir, SEALED_FILES.teams), teamsText);
  writeSecretFile(join(dir, SEALED_FILES.key), `${a.key}\n`);
  writeSecretFile(join(dir, SEALED_FILES.salt), `${a.salt}\n`);
  renameSync(a.logPath, join(dir, SEALED_FILES.rejections));
  writeSecretFile(join(dir, SEALED_FILES.manifest), pretty(manifest));
  writeSecretFile(join(dir, SEALED_FILES.commitment), pretty(commitment));
  return { manifest, commitment, dir, discarded: a.discarded, warnings: plan.warnings };
}

// ---------------------------------------------------------------------------
// Reading a set back inside the store; stamping.

/** The set's records and secrets as the store holds them, after the commitment was recomputed from the salt and the teams. */
export interface StoredSet {
  manifest: SealedTeamSetManifest;
  commitment: SealedSetCommitment;
  teams: SealedTeam[];
  key: string;
  salt: string;
  /** The exact bytes of the three logs and the teams file, for the reveal. */
  texts: { teams: string; rejections: string; discards: string };
}

/** The sets `readStoredSet` returned: the commitment was recomputed for each. Only these may build `SealedTeamHashes`. */
const verifiedSets = new WeakSet<object>();

/** Parse `teams.jsonl`: one JCS line per team. */
export function parseSealedTeams(text: string): SealedTeam[] {
  if (text !== "" && !text.endsWith("\n")) fail("teams.jsonl does not end with a newline");
  return (text === "" ? [] : text.slice(0, -1).split("\n")).map((line, n) => {
    const value = parseSecretJson(line, `teams.jsonl line ${n + 1}`) as SealedTeam;
    if (jcs(value as never) !== line) fail(`teams.jsonl line ${n + 1} is not in canonical (JCS) form`);
    const keys = Object.keys(value).join();
    if (keys !== "k,paste,stratum" || !Number.isSafeInteger(value.k) || typeof value.paste !== "string" || !Number.isSafeInteger(value.stratum)) fail(`teams.jsonl line ${n + 1} is not a sealed team`);
    return value;
  });
}

/** Read a set from the store. The commitment is recomputed from the salt and the teams (L3) and must equal both records. */
export function readStoredSet(storeDir: string, setId: string): StoredSet {
  const store = assertPrivateStore(storeDir);
  const dir = setDirOf(store, setId);
  const text = (file: string): string => readSecretFile(join(dir, file));
  const manifest = readSealedTeamSetManifest(parseSecretJson(text(SEALED_FILES.manifest), SEALED_FILES.manifest));
  const commitment = readSealedSetCommitment(parseSecretJson(text(SEALED_FILES.commitment), SEALED_FILES.commitment));
  const key = text(SEALED_FILES.key).trim();
  const salt = text(SEALED_FILES.salt).trim();
  const teamsText = text(SEALED_FILES.teams);
  const teams = parseSealedTeams(teamsText);
  if (manifest.set_id !== setId || commitment.set_id !== setId) fail(`the records in ${dir} are not set ${setId}`);
  if (setCommitment(salt, teams as never) !== commitment.set_commitment || commitment.set_commitment !== manifest.set_commitment) fail("the stored salt and teams do not open the set commitment");
  if (keyCommitment(key) !== commitment.key_commitment || commitment.key_commitment !== manifest.key_commitment) fail("the stored key does not open the key commitment");
  const stored: StoredSet = { manifest, commitment, teams, key, salt, texts: { teams: teamsText, rejections: text(SEALED_FILES.rejections), discards: text(SEALED_FILES.discards) } };
  verifiedSets.add(stored);
  return stored;
}

/**
 * The hashes of the sealed teams, for stamping inside the store (ADR-013 section
 * 4, "SealedTeamHashes inside the store"). Built only by `loadSealedTeamHashes`,
 * after the L3 commitment was recomputed. It cannot be serialised or printed:
 * `withPublic` is the one way out, and returns a plain set that stays in the
 * memory of the evaluation job (the type cannot stop a caller from writing it;
 * the evaluation job must not, and a canary test pins that this module does not).
 */
export class SealedTeamHashes {
  readonly #positional: ReadonlySet<string>;
  readonly #canonical: ReadonlySet<string>;

  private constructor(positional: ReadonlySet<string>, canonical: ReadonlySet<string>) {
    this.#positional = positional;
    this.#canonical = canonical;
  }

  /** Only a set that `readStoredSet` returned (its L3 commitment recomputed) is accepted. */
  static fromVerified(engine: Engine, stored: StoredSet): SealedTeamHashes {
    if (!verifiedSets.has(stored)) fail("a set that readStoredSet did not return is not trusted");
    const sets = stored.teams.map((t) => importPaste(engine, t.paste));
    return new SealedTeamHashes(new Set(sets.map((s) => teamHash(s))), new Set(sets.map((s) => canonicalTeamHash(s))));
  }

  get size(): number {
    return this.#positional.size;
  }

  /** The positional hashes of the public synthetic teams and the sealed ones, for `stampTeam` and `readEpisode`. */
  withPublic(publicHashes: SyntheticTeamHashes): SyntheticTeamHashes {
    return syntheticTeamHashes(publicHashes, this.#positional);
  }

  /** Whether a team (by canonical hash) is sealed: the repository scan of protocol section 7. */
  hasCanonical(hash: string): boolean {
    return this.#canonical.has(hash);
  }

  toJSON(): never {
    return fail("SealedTeamHashes must never be serialised");
  }

  toString(): string {
    return `SealedTeamHashes(${this.size})`;
  }

  [inspect.custom](): string {
    return this.toString();
  }
}

/** Build `SealedTeamHashes` from the store, after recomputing the L3 commitment. */
export function loadSealedTeamHashes(engine: Engine, storeDir: string, setId: string): SealedTeamHashes {
  const stored = readStoredSet(storeDir, setId);
  return SealedTeamHashes.fromVerified(engine, stored);
}

export { sha256 as textSha256 };
