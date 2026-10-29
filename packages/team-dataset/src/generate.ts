// The generator (ADR-013 section 2): the pilot, the quota plan with the D20
// merge, one draw stream binned in counter order with dedupe, the split fill
// and the records and manifest. Nothing here decides legality (the engine
// does, in `Drawer`) and nothing is written to disk but the rejection log, as
// the run goes; `dataset-io.ts` writes the rest.
//
// Splits. The bins are per stratum, sized by the sum of the strata's quotas
// over the splits. When the bins are full, `assignSplits` (assign-splits.ts)
// deals them to dev, validation and training by the assignment key, so that
// every record carries its split. A reader of `origin.split` can rely on the
// key, not on the stream position.
//
// Shortfall. If `d_max` draws leave a bin short, the dataset is planned again
// with the short strata coarsened by one more level, under a fresh plan id
// (`<plan_id>-r1`, ...) and dataset id, and the first attempt's log is kept
// as `rejections.attempt-<n>.jsonl`.

import { renameSync } from "node:fs";
import { join } from "node:path";
import {
  type Engine,
  importPaste,
  readTeamDatasetManifest,
  readTeamRecord,
  type ResolvedFormat,
  recordHash,
  type TeamDatasetManifest,
  type TeamGeneratorConfig,
  type TeamLabelSet,
  type TeamRecord,
  type TeamVocabulary,
  teamVocabularyHash,
  type EngineData,
} from "@areneon/showdown-bridge";
import { assignSplits } from "./assign-splits.ts";
import { Binner } from "./binner.ts";
import { crossSplitNear, type NearDuplicates, type NearIndex } from "./near-index.ts";
import { type DeliveredComponent, populationOf, proposalCoverage, type ProposalCoverage } from "./pilot-gates.ts";
import type { KnownHashes } from "./dedupe.ts";
import { Drawer } from "./draw.ts";
import { cellCount, cellLabels, cellOf, labelTeam, radices } from "./labeller.ts";
import { type Pilot, runPilot } from "./pilot.ts";
import { groupKey, manifestPlan, planQuotas, type QuotaPlan, SPLITS, type Split, type SplitSizes } from "./plan.ts";
import { Proposal } from "./proposal.ts";
import type { RoleSetsSource } from "./role-set-proposal.ts";
import { provision, type Provision } from "./provision.ts";
import { RejectionLog } from "./rejection-log.ts";
import { checkKey } from "./stream.ts";

export class GenerateError extends Error {
  override name = "GenerateError";
}

/** The known hashes of earlier released datasets, and the identity of that set. */
export interface ReferencePool {
  /** SHA-256 of the sorted canonical hashes (`referencePoolHash`). */
  hash: string;
  known: KnownHashes;
}

export interface GenerateInput {
  engine: Engine;
  format: ResolvedFormat;
  table: EngineData;
  vocabulary: TeamVocabulary;
  labelSet: TeamLabelSet;
  config: TeamGeneratorConfig;
  /** The role-set table a `role_sets` component of `config.proposal` reads (its hash must be the one the config names). */
  roleSets?: RoleSetsSource | undefined;
  /** HMAC keys, 64 hex digits each. Public for a public dataset. */
  generationKey: string;
  assignmentKey: string;
  datasetId: string;
  planId: string;
  sizes: SplitSizes;
  generatorCommit: string;
  reference: ReferencePool;
  /** A recorded pilot; without one, `config.pilot_draws` pilot draws are run. */
  pilot?: Pilot | undefined;
  /**
   * A pilot of the uniform config (this config without `proposal`) at the same `d_max`: the reference plan of gate P4
   * (ADR-020 section 6). Without it a mixture's P4 is reported as not evaluated.
   */
  uniformPilot?: Pilot | undefined;
  /** `d_team` for the cross-split near-duplicate counts of `coverage.json` (counts only); without it none are reported. */
  dTeam?: number | undefined;
  /**
   * An evaluation-only dataset (ADR-020 section 7): the config must be a pure component, and every accepted team
   * within `d_team` differing fields of a team in `index` (built from the reference datasets' teams) is dropped and
   * logged as `near_reference`. The manifest marks the dataset evaluation-only.
   */
  exclude?: { d_team: number; index: NearIndex } | undefined;
  /** Where the rejection log is written. */
  logDir: string;
  /** Called every `progressEvery` draws. */
  onProgress?: (event: { draws: number; open: number }) => void;
  progressEvery?: number;
}

export interface Attempt {
  plan_id: string;
  dataset_id: string;
  /** The strata that were short when `d_max` draws were spent. */
  short: number[];
}

export interface GenerateResult {
  manifest: TeamDatasetManifest;
  /** The records of each generated split, in the dataset's fixed order. */
  records: Partial<Record<Split, TeamRecord[]>>;
  plan: QuotaPlan;
  pilot: Pilot;
  /** How many pilot draws this call ran: none when a recorded pilot was given. */
  pilotDrawsRun: number;
  coverage: CoverageReport;
  /** The plan and dataset ids of the attempt that filled; they differ from the input after a re-plan. */
  planId: string;
  datasetId: string;
  logPath: string;
  attempts: Attempt[];
}

export interface CoverageReport {
  dataset_id: string;
  plan_id: string;
  config_hash: string;
  label_set_hash: string;
  vocabulary_hash: string;
  labels: { label_id: string; values: string[] }[];
  pilot: { draws: number; rejected: number };
  /** Every base cell: its label values (in the label set's order), pilot hits, stratum and the teams it holds in each split (inside a merged stratum the cells share the quota, so these show how it was spread). */
  cells: { cell: number; values: string[]; pilot_hits: number; stratum: number; held: Partial<Record<Split, number>> }[];
  /** Every stratum after the merge, with the teams it holds in each split (its quotas, as every bin is full). */
  strata: { stratum: number; level: number; cells: number; pilot_hits: number; quotas: Partial<Record<Split, number>> }[];
  /** Requested and delivered teams per split: the quotas are proportional with a floor of n_stratum_min per stratum, so a split can hold more than was asked. */
  splits: { split: Split; requested: number; teams: number }[];
  /** 0 for the first plan; n for the n-th re-plan after a shortfall. */
  replan: number;
  stream: { draws: number; accepted: number; rejections: number; duplicates: number; overflows: number };
  provisioning: Provision;
  attempts: Attempt[];
  dedupe: { reference_hashes: number; prefix_bytes: number; prefix_misses: number; exact_checks: number };
  /** E-13.6, only with a mixture proposal or an evaluation-only run (an ordinary uniform dataset keeps its bytes). */
  population?: string;
  /** Pilot gates P1 to P4 and the per-component counts; only with a mixture proposal. */
  proposal?: ProposalCoverage;
  /** Only when `dTeam` was given. */
  near_duplicates?: NearDuplicates;
}

export function generate(input: GenerateInput): GenerateResult {
  const { engine, format, table, vocabulary, labelSet, config } = input;
  const generationKey = checkKey(input.generationKey, "the generation key");
  const assignmentKey = checkKey(input.assignmentKey, "the assignment key");
  if (input.exclude !== undefined && (config.proposal?.components.length ?? 1) !== 1) throw new GenerateError("an evaluation-only dataset needs a pure component config (one component)");
  const configHash = recordHash(config);
  const labelSetHash = recordHash(labelSet);
  const vocabularyHash = teamVocabularyHash(vocabulary);
  const drawer = new Drawer({ engine, format, proposal: new Proposal(vocabulary, table, config, input.roleSets), labelSet, generationKey });
  const identity = { config_hash: configHash, label_set_hash: labelSetHash, vocabulary_hash: vocabularyHash };
  const pilotDrawsRun = input.pilot === undefined ? config.pilot_draws : 0;
  const pilot = input.pilot ?? runPilot(drawer, identity, labelSet, config.pilot_draws);
  if (pilot.draws !== config.pilot_draws) throw new GenerateError(`the pilot has ${pilot.draws} draws, the config says ${config.pilot_draws}`);
  if (pilot.config_hash !== configHash || pilot.label_set_hash !== labelSetHash || pilot.vocabulary_hash !== vocabularyHash) {
    throw new GenerateError("the pilot was recorded for another config, label set or vocabulary");
  }
  if (config.proposal !== undefined && pilot.components === undefined) throw new GenerateError("the pilot has no per-component counts; record it again with this config");
  const uniformPilot = input.uniformPilot;
  if (uniformPilot !== undefined) {
    const plain = structuredClone(config);
    delete plain.proposal;
    if (config.proposal === undefined) throw new GenerateError("a uniform reference pilot is for a config with a proposal");
    if (uniformPilot.config_hash !== recordHash(plain) || uniformPilot.label_set_hash !== labelSetHash || uniformPilot.vocabulary_hash !== vocabularyHash || uniformPilot.draws !== config.pilot_draws) {
      throw new GenerateError("the uniform reference pilot was not recorded for this config without its proposal, label set and vocabulary");
    }
  }
  const dims = radices(labelSet);
  const known = input.reference.known;
  const attempts: Attempt[] = [];
  const forced = new Set<string>();

  for (let attempt = 0; ; attempt++) {
    const suffix = attempt === 0 ? "" : `-r${attempt}`;
    const planId = `${input.planId}${suffix}`;
    const datasetId = `${input.datasetId}${suffix}`;
    const plan = planQuotas({
      cellHits: pilot.cell_hits,
      radices: dims,
      pilotDraws: pilot.draws,
      pilotMinHits: config.pilot_min_hits,
      dMax: config.d_max,
      nStratumMin: config.n_stratum_min,
      sizes: input.sizes,
      forced,
    });
    const quotas = new Map(plan.strata.map((s) => [s.stratum, s.total]));
    const lookups = { misses: known.prefixMisses, checks: known.exactChecks };
    const exclude = input.exclude;
    const binner = new Binner(quotas, known, exclude === undefined ? undefined : (draw) => exclude.index.near(importPaste(engine, draw.paste)));
    const logPath = join(input.logDir, "rejections.jsonl");
    const log = new RejectionLog(logPath);
    const stream = { config_hash: configHash, plan_id: planId };
    let draws = 0;
    let accepted = 0;
    const every = input.progressEvery ?? 10_000;
    while (!binner.full && draws < config.d_max) {
      const draw = drawer.draw(stream, draws);
      if (draw.status === "rejected") {
        log.write({ k: draw.k, reason: draw.reason, canonical_team_hash: draw.canonical_team_hash, detail: draw.detail });
      } else {
        accepted++;
        const stratum = plan.stratumOf[draw.cell] as number;
        const offer = binner.offer(draw, stratum);
        if (offer !== "binned") log.write({ k: draw.k, reason: offer, canonical_team_hash: draw.canonical_team_hash, stratum });
      }
      draws++;
      if (input.onProgress !== undefined && draws % every === 0) input.onProgress({ draws, open: binner.short.length });
    }
    const summary = log.close();
    const short = binner.short;
    if (short.length > 0) {
      attempts.push({ plan_id: planId, dataset_id: datasetId, short });
      if (attempt >= dims.length) throw new GenerateError(`bins are still short after ${attempt + 1} plans`);
      renameSync(logPath, join(input.logDir, `rejections.attempt-${attempt}.jsonl`));
      for (const stratum of short) forced.add(groupKey((plan.strata.find((s) => s.stratum === stratum) as { level: number }).level, stratum));
      continue;
    }
    attempts.push({ plan_id: planId, dataset_id: datasetId, short: [] });

    const records = fillSplits({ plan, binner, assignmentKey, datasetId, labelSetHash, nStratumMin: config.n_stratum_min });
    for (const list of Object.values(records)) for (const record of list) checkRecord(engine, record, labelSet);
    const manifest = readTeamDatasetManifest(
      {
        schema_version: 1,
        data_class: "synthetic",
        dataset_id: datasetId,
        plan_id: planId,
        generator: {
          generator_commit: input.generatorCommit,
          config_hash: configHash,
          label_set_hash: labelSetHash,
          vocabulary_hash: vocabularyHash,
          manifest: format.manifest,
        },
        key: { kind: "public", generation_key: generationKey, assignment_key: assignmentKey },
        d_max: config.d_max,
        pilot_draws: pilot.draws,
        ...manifestPlan(plan),
        reference_pool_hash: input.reference.hash,
        ...(input.exclude === undefined ? {} : { evaluation_only: { d_team: input.exclude.d_team, near_references: summary.near_references } }),
        rejection_log: { log_sha256: summary.log_sha256, draws, rejections: summary.rejections, duplicates: summary.duplicates, overflows: summary.overflows },
      },
      { config, labelSet },
    );
    const held = new Map<string, number>();
    for (const split of plan.splits) {
      for (const r of records[split] as TeamRecord[]) {
        const key = `${split}:${cellOf(labelSet, r.labels)}`;
        held.set(key, (held.get(key) ?? 0) + 1);
      }
    }
    const coverage: CoverageReport = {
      dataset_id: datasetId,
      plan_id: planId,
      config_hash: configHash,
      label_set_hash: labelSetHash,
      vocabulary_hash: vocabularyHash,
      labels: labelSet.labels.map((l) => ({ label_id: l.label_id, values: [...l.values.map((v) => v.value), l.default] })),
      pilot: { draws: pilot.draws, rejected: pilot.rejected },
      cells: Array.from({ length: cellCount(labelSet) }, (_, cell) => ({
        cell,
        values: cellLabels(labelSet, cell).map((l) => l.value),
        pilot_hits: pilot.cell_hits[cell] as number,
        stratum: plan.stratumOf[cell] as number,
        held: Object.fromEntries(plan.splits.map((split) => [split, (held.get(`${split}:${cell}`) ?? 0)])),
      })),
      strata: plan.strata.map((s) => ({
        stratum: s.stratum,
        level: s.level,
        cells: s.cells,
        pilot_hits: s.pilot_hits,
        quotas: s.quotas,
      })),
      splits: plan.splits.map((split) => ({ split, requested: input.sizes[split] as number, teams: (records[split] as TeamRecord[]).length })),
      replan: attempt,
      stream: { draws, accepted, rejections: summary.rejections, duplicates: summary.duplicates, overflows: summary.overflows },
      provisioning: provision(plan, {
        pilotDraws: pilot.draws,
        fillQuantilePermille: config.fill_quantile_permille,
        rangeMarginPermille: config.range_margin_permille,
        rangeDrawsLog2: config.range_draws_log2,
        dMax: config.d_max,
      }),
      attempts,
      dedupe: {
        reference_hashes: known.size,
        prefix_bytes: known.prefixBytes,
        prefix_misses: known.prefixMisses - lookups.misses,
        exact_checks: known.exactChecks - lookups.checks,
      },
    };
    if (config.proposal !== undefined || input.exclude !== undefined) {
      coverage.population = populationOf(config, input.exclude === undefined ? undefined : `evaluation-only: teams within ${input.exclude.d_team} differing fields of a reference team were dropped`);
    }
    if (input.dTeam !== undefined) {
      const teams = Object.fromEntries(plan.splits.map((split) => [split, (records[split] as TeamRecord[]).map((r) => importPaste(engine, r.paste))]));
      coverage.near_duplicates = crossSplitNear(teams, input.dTeam);
    }
    if (config.proposal !== undefined) {
      const delivered = new Map<string, DeliveredComponent>(config.proposal.components.map((c) => [c.kind, { cells: new Array<number>(cellCount(labelSet)).fill(0), splits: {} }]));
      for (const split of SPLITS) {
        for (const r of records[split] ?? []) {
          const part = delivered.get(drawer.component(stream, r.origin.draw_counter)) as DeliveredComponent;
          const cell = cellOf(labelSet, r.labels);
          part.cells[cell] = (part.cells[cell] as number) + 1;
          part.splits[split] = (part.splits[split] ?? 0) + 1;
        }
      }
      coverage.proposal = proposalCoverage({
        config,
        vocabulary,
        roleSets: input.roleSets,
        pilot,
        plan,
        referenceStrata: uniformPilot === undefined ? null : referenceStrata(uniformPilot, config, dims, input.sizes),
        delivered,
      });
    }
    return { manifest, records, plan, pilot, pilotDrawsRun, coverage, planId, datasetId, logPath, attempts };
  }
}

/** The strata of the uniform config's plan at the same `d_max` and sizes: the reference of P4. */
function referenceStrata(pilot: Pilot, config: TeamGeneratorConfig, dims: number[], sizes: SplitSizes): number {
  return planQuotas({
    cellHits: pilot.cell_hits,
    radices: dims,
    pilotDraws: pilot.draws,
    pilotMinHits: config.pilot_min_hits,
    dMax: config.d_max,
    nStratumMin: config.n_stratum_min,
    sizes,
    forced: new Set(),
  }).strata.length;
}

interface FillInput {
  plan: QuotaPlan;
  binner: Binner;
  assignmentKey: string;
  datasetId: string;
  labelSetHash: string;
  nStratumMin: number;
}

/** Deal the full bins to the splits (`assignSplits`) and build the records. */
function fillSplits(input: FillInput): Partial<Record<Split, TeamRecord[]>> {
  const { plan, binner } = input;
  const pool = plan.strata.flatMap((s) => binner.bin(s.stratum).map((entry) => ({ ...entry, stratum: s.stratum })));
  const dealt = assignSplits(pool, plan, input.assignmentKey, input.nStratumMin);
  const out: Partial<Record<Split, TeamRecord[]>> = {};
  for (const split of plan.splits) {
    out[split] = dealt[split].map((entry, index) => ({
      schema_version: 1,
      data_class: "synthetic",
      team_hash: entry.team_hash,
      canonical_team_hash: entry.canonical_team_hash,
      paste: entry.paste,
      origin: { dataset_id: input.datasetId, split, index, draw_counter: entry.k },
      label_set_hash: input.labelSetHash,
      labels: entry.labels,
    }));
  }
  return out;
}

/** The last gate on a record: the bridge's reader (schema, engine round trip, both hashes, labels follow the set) and the labeller's own output. */
export function checkRecord(engine: Engine, record: TeamRecord, labelSet: TeamLabelSet): void {
  readTeamRecord(engine, record, labelSet);
  const again = labelTeam(labelSet, importPaste(engine, record.paste));
  if (JSON.stringify(again) !== JSON.stringify(record.labels)) throw new GenerateError(`record ${record.canonical_team_hash}: its labels are not the labeller's output`);
}
