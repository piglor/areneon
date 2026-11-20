// The damage kernel as a backend of the differential harness (ADR-030 step 3, #1457), behind the probe's interface: a batch
// of parsed mechanics queries in, their `MechanicsResult`s in query order. The result's envelope (its identity, the set check,
// the scope check and the scenario check) is the probe's own (`probeEnvelope`); only the `damage` computation is the kernel's.
// A query the kernel does not cover is a `KernelAbstain` outcome (never a guess): the harness compares only covered queries,
// and a router hands an abstained query to the engine probe. Training games only; no evaluation path imports this module (a
// source scan in the composition root's tests).
import type { Engine } from "./engine.ts";
import type { MechanicsConfig } from "./mechanics-config.ts";
import { probeEnvelope, type ProbeOutcome } from "./mechanics-probe.ts";
import type { ParsedMechanicsQuery } from "./mechanics-query.ts";
import type { Outcome } from "./mechanics-result.ts";
import type { Settled } from "./probe-batch.ts";
import type { RulesSnapshot } from "./rules.ts";
import { kernelData } from "./damage-kernel-data.ts";
import { checkKernelEffects, type KernelEffects, KernelEffectsError, loadKernelEffects } from "./damage-kernel-effects.ts";
import { KERNEL_FORMAT, KernelAbstain, type KernelContext, type Prepared, prepareQuery, rollOutcomes } from "./damage-kernel.ts";

export interface KernelBackendOptions {
  config: MechanicsConfig;
  snapshot?: RulesSnapshot;
  /** The frozen commit of the kernel's code (40 hex): the backend identity the admission ledger and its fresh seed name. */
  backend_commit: string;
  /** The name reports give it (default `damage-kernel`). */
  id?: string;
}

/** What a kernel backend counts: queries answered, and abstentions by entity (the per-entity fallbacks of run metrics). */
export interface KernelCounts {
  answered: number;
  abstained: number;
  /** `<kind>:<id>` -> abstentions it caused. */
  by_entity: Map<string, number>;
}

export interface KernelBackend {
  readonly id: string;
  readonly backend_format: number;
  readonly backend_commit: string;
  run(batch: readonly ParsedMechanicsQuery[]): Settled[];
  readonly counts: KernelCounts;
}

const COMMIT = /^[0-9a-f]{40}$/;

export class KernelBackendError extends Error {
  override name = "KernelBackendError";
}

/** The kernel behind the probe interface, on the pinned engine's data and the probe's envelope. */
export function kernelBackend(engine: Engine, options: KernelBackendOptions): KernelBackend {
  if (!COMMIT.test(options.backend_commit)) throw new KernelBackendError("a backend commit is 40 hex digits");
  const context = { engine, config: options.config, ...(options.snapshot === undefined ? {} : { snapshot: options.snapshot }) };
  const levels = new Map<string, number | null>();
  const levelOf = (scenarioId: string): number | null => {
    if (!levels.has(scenarioId)) levels.set(scenarioId, engine.sim.Dex.formats.getRuleTable(engine.sim.Dex.formats.get(scenarioId)).adjustLevel);
    return levels.get(scenarioId) as number | null;
  };
  // The reviewed effects record of a format, bound to this engine's pin: another pin's record makes the kernel abstain on everything.
  const effectsOf = new Map<string, KernelEffects | null>();
  const effects = (formatId: string): KernelEffects => {
    if (!effectsOf.has(formatId)) {
      try {
        const loaded = loadKernelEffects(formatId, engine.commit);
        // A mis-kinded entry is a defect of the record: it throws (`KernelRecordError`), never routes silently.
        checkKernelEffects(loaded, kernelData(engine, formatId));
        effectsOf.set(formatId, loaded);
      } catch (error) {
        if (!(error instanceof KernelEffectsError)) throw error;
        effectsOf.set(formatId, null);
      }
    }
    const found = effectsOf.get(formatId);
    if (found === null || found === undefined) throw new KernelAbstain("state:effects_record", "pin");
    return found;
  };
  const counts: KernelCounts = { answered: 0, abstained: 0, by_entity: new Map() };
  const abstain = (error: KernelAbstain): Settled => {
    counts.abstained += 1;
    counts.by_entity.set(error.entity, (counts.by_entity.get(error.entity) ?? 0) + 1);
    return { ok: false, error };
  };
  return {
    id: options.id ?? "damage-kernel",
    backend_format: KERNEL_FORMAT,
    backend_commit: options.backend_commit,
    counts,
    run(batch) {
      const out: (Settled | null)[] = new Array(batch.length).fill(null);
      // The queries that need the roll loop, decoded, then computed together over struct-of-arrays columns.
      const rows: { index: number; prepared: Extract<Prepared, { kind: "rolls" }>; done: (outcome: Outcome) => ProbeOutcome; ctx: KernelContext }[] = [];
      batch.forEach((query, index) => {
        if (query.query.kind !== "damage") {
          out[index] = abstain(new KernelAbstain(`kind:${query.query.kind}`, "not_implemented"));
          return;
        }
        let settled: Settled | null = null;
        try {
          const envelope = probeEnvelope(context, query);
          if (envelope.kind === "done") settled = { ok: true, result: envelope.outcome.result };
          else {
            const ctx: KernelContext = { data: kernelData(engine, query.format_id), effects: effects(query.format_id), ko_blockers: options.config.ko_blockers, level: levelOf(envelope.scenario_id) };
            const prepared = prepareQuery(ctx, query.query, envelope.sets[0] as never, envelope.sets[1] as never);
            if (prepared.kind === "outcome") settled = { ok: true, result: envelope.done(prepared.outcome).result };
            else rows.push({ index, prepared, done: envelope.done, ctx });
          }
        } catch (error) {
          if (!(error instanceof KernelAbstain)) throw error;
          settled = abstain(error);
        }
        if (settled !== null) {
          if (settled.ok) counts.answered += 1;
          out[index] = settled;
        }
      });
      if (rows.length > 0) {
        const outcomes = rollOutcomes(rows[0]?.ctx as KernelContext, rows.map((r) => r.prepared.decoded));
        rows.forEach((row, i) => {
          counts.answered += 1;
          out[row.index] = { ok: true, result: row.done(outcomes[i] as Outcome).result };
        });
      }
      return out as Settled[];
    },
  };
}

/** Whether a kernel outcome is an abstention (the query routes to the engine probe). */
export const abstained = (settled: Settled): boolean => !settled.ok && settled.error instanceof KernelAbstain;
