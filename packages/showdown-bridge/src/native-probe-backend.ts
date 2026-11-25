// The native probe as a backend of the differential harness (#1534, the Ninjask track of #1530): the declared-state layer of the
// vendored fork behind the probe's interface. A batch of parsed mechanics queries in, their `MechanicsResult`s in query order. The
// result's envelope (identity, set check, scope check, scenario check) is the probe's own (`probeEnvelope`); only the `answer` of a
// `damage`, `speed_order` or `stats` query is the native engine's, from its calculators (`ninjask::calc`) on a scenario battle it
// builds the way the probe builds its own. Training games only: no evaluation path imports this module (a source scan).
//
// The layer never guesses. What it does not implement (a history block, a move with a step before the hit it does not model, a
// declared state the engine refuses) comes back as an abstention; the backend returns a `NativeAbstain` outcome, the caller routes
// the query to the pinned probe, and admission leaves the entity unadmitted. Showdown stays the oracle.
import type { Engine, PokemonSet } from "./engine.ts";
import type { MechanicsConfig } from "./mechanics-config.ts";
import { probeEnvelope, type ProbeEnvelope } from "./mechanics-probe.ts";
import type { ParsedMechanicsQuery } from "./mechanics-query.ts";
import { type Outcome } from "./mechanics-result.ts";
import { hasHistory, type BatchResponse, type NativeOutcome, NativeProtocolError, readResponse, type SetsResponse, wireQuery, type WireQuery, type WireSet, wireSet } from "./native-probe-wire.ts";
import { type NativeProbeCommand, NativeProbeError, NativeProbeProcess } from "./native-probe-process.ts";
import type { Settled } from "./probe-batch.ts";
import type { RulesSnapshot } from "./rules.ts";

/** Bumped when the wire form or the way an answer is read changes (the admission ledger names it with the fork's commit). */
export const NATIVE_PROBE_FORMAT = 1;

/** The native layer does not answer a query: the entity it does not implement (the pinned probe answers it). */
export class NativeAbstain extends Error {
  override name = "NativeAbstain";
  /** `<kind>:<id>` of what decided it (`move:suckerpunch`, `ability:protean`, `state:status:slp`, `query:history`, `native:process`...). */
  readonly entity: string;
  /** A fixed word for why. */
  readonly why: string;
  constructor(entity: string, why: string) {
    super(`the native probe does not cover ${entity} (${why})`);
    this.entity = entity;
    this.why = why;
  }
}

/** Whether a settled outcome is an abstention (the query routes to the pinned probe). */
export const nativeAbstained = (settled: Settled): boolean => !settled.ok && settled.error instanceof NativeAbstain;

export interface NativeProbeBackendOptions {
  config: MechanicsConfig;
  snapshot?: RulesSnapshot;
  command: NativeProbeCommand;
  /** The fork's commit the binary was built from (40 hex): the backend identity the admission ledger and its fresh seed name. */
  backend_commit: string;
  /** The name reports give it (default `native-probe`). */
  id?: string;
  /** Milliseconds a response may take (default two minutes). */
  timeout_ms?: number;
  /** Milliseconds the process may take to start (default 30 seconds). */
  start_ms?: number;
}

/** What a backend counts: queries answered, left to the pinned probe and abstained, by entity; and where the time went. */
export interface NativeCounts {
  answered: number;
  unsupported: number;
  abstained: number;
  /** `<kind>:<id>` -> abstentions it caused. */
  by_entity: Map<string, number>;
  /** Frames the native process answered and restarts after a dead process. */
  frames: number;
  restarts: number;
  /** Milliseconds in the pinned envelope, in the transport and the native layer (summed over `run` calls). */
  envelope_ms: number;
  native_ms: number;
}

export interface NativeProbeBackend {
  readonly id: string;
  readonly backend_format: number;
  readonly backend_commit: string;
  run(batch: readonly ParsedMechanicsQuery[]): Settled[];
  readonly counts: NativeCounts;
  close(): void;
}

const COMMIT = /^[0-9a-f]{40}$/;

/** A failure of the process or its frames is the batch's to route; any other error is a defect and propagates. */
export function recover(error: unknown): void {
  if (!(error instanceof NativeProbeError) && !(error instanceof NativeProtocolError)) throw error;
}
/** After this many consecutive dead processes the backend stops restarting and abstains everything. */
const MAX_RESTARTS = 3;

export class NativeProbeBackendError extends Error {
  override name = "NativeProbeBackendError";
}

interface Pending {
  index: number;
  envelope: Extract<ProbeEnvelope, { kind: "ready" }>;
  wire: WireQuery;
  ids: number[];
}

export function nativeProbeBackend(engine: Engine, options: NativeProbeBackendOptions): NativeProbeBackend {
  if (!COMMIT.test(options.backend_commit)) throw new NativeProbeBackendError("a backend commit is 40 hex digits");
  const context = { engine, config: options.config, ...(options.snapshot === undefined ? {} : { snapshot: options.snapshot }) };
  const counts: NativeCounts = { answered: 0, unsupported: 0, abstained: 0, by_entity: new Map(), frames: 0, restarts: 0, envelope_ms: 0, native_ms: 0 };
  let process: NativeProbeProcess | null = null;
  let deaths = 0;
  let started = false;
  // The sets registered with the running process, by their wire text, and the ones the dex refused.
  let registered = new Map<string, number>();
  let refused = new Set<number>();
  const encoder = new TextEncoder();

  const ensure = (): NativeProbeProcess => {
    if (process !== null) return process;
    if (deaths > MAX_RESTARTS) throw new NativeProbeError("the native probe died too often");
    if (started) counts.restarts += 1;
    started = true;
    process = new NativeProbeProcess(options.command, { timeout_ms: options.timeout_ms, start_ms: options.start_ms });
    registered = new Map();
    refused = new Set();
    const blockers = options.config.ko_blockers;
    readResponse<{ ok: boolean }>(process.request(encoder.encode(JSON.stringify({ op: "blockers", items: blockers.items, abilities: blockers.abilities }))), "ok");
    return process;
  };

  const abstain = (error: NativeAbstain): Settled => {
    counts.abstained += 1;
    counts.by_entity.set(error.entity, (counts.by_entity.get(error.entity) ?? 0) + 1);
    return { ok: false, error };
  };

  return {
    id: options.id ?? "native-probe",
    backend_format: NATIVE_PROBE_FORMAT,
    backend_commit: options.backend_commit,
    counts,
    close() {
      process?.close();
      process = null;
    },
    run(batch) {
      const out: (Settled | null)[] = new Array(batch.length).fill(null);
      const pending: Pending[] = [];
      const newSets: [number, WireSet][] = [];
      const started = performance.now();
      // The process (and the set table it holds) is settled before any id is handed out: a restart empties the table.
      let proc: NativeProbeProcess | null = null;
      try {
        proc = ensure();
      } catch (error) {
        recover(error);
        deaths += 1;
        process?.close();
        process = null;
      }
      batch.forEach((query, index) => {
        try {
          const envelope = probeEnvelope(context, query);
          if (envelope.kind === "done") {
            out[index] = { ok: true, result: envelope.outcome.result };
            counts.answered += 1;
            return;
          }
          if (hasHistory(query.query)) {
            out[index] = abstain(new NativeAbstain("query:history", "a declared history is not implemented"));
            return;
          }
          const ids = envelope.sets.map((set: PokemonSet) => {
            const wire = wireSet(set);
            const key = JSON.stringify(wire);
            let id = registered.get(key);
            if (id === undefined) {
              id = registered.size;
              registered.set(key, id);
              newSets.push([id, wire]);
            }
            return id;
          });
          const wire = wireQuery(query.query, ids);
          if (wire === null) {
            out[index] = abstain(new NativeAbstain(`kind:${query.query.kind}`, "not implemented"));
            return;
          }
          pending.push({ index, envelope, wire, ids });
        } catch (error) {
          out[index] = { ok: false, error };
        }
      });
      const mid = performance.now();
      counts.envelope_ms += mid - started;
      if (pending.length > 0) {
        let results: NativeOutcome[] | null = null;
        try {
          if (proc === null) throw new NativeProbeError("the native probe is not running");
          if (newSets.length > 0) {
            const response = readResponse<SetsResponse>(proc.request(encoder.encode(JSON.stringify({ op: "sets", sets: newSets }))), "failed");
            for (const [id] of response.failed) refused.add(id);
            counts.frames += 1;
          }
          const frame = readResponse<BatchResponse>(proc.request(encoder.encode(JSON.stringify({ op: "batch", queries: pending.map((p) => p.wire) }))), "results");
          counts.frames += 1;
          if (frame.results.length !== pending.length) throw new NativeProtocolError("the native probe answered another number of queries than it was sent");
          results = frame.results;
          deaths = 0;
        } catch (error) {
          recover(error);
          deaths += 1;
          process?.close();
          process = null;
        }
        pending.forEach((p, i) => {
          const native = results?.[i];
          if (native === undefined) {
            out[p.index] = abstain(new NativeAbstain("native:process", "the native probe is not running"));
            return;
          }
          if (p.ids.some((id) => refused.has(id))) {
            out[p.index] = abstain(new NativeAbstain("state:set", "the native dex does not know the set"));
            return;
          }
          if (native.kind === "abstain") {
            out[p.index] = abstain(new NativeAbstain(native.entity, native.why));
            return;
          }
          const outcome: Outcome = native.kind === "computed" ? { kind: "computed", answer: native.answer } : { kind: "unsupported", reason: native.reason, position: native.position };
          if (native.kind === "computed") counts.answered += 1;
          else counts.unsupported += 1;
          out[p.index] = { ok: true, result: p.envelope.done(outcome).result };
        });
      }
      counts.native_ms += performance.now() - mid;
      return out as Settled[];
    },
  };
}

/**
 * A query in the native layer's own form, with its sets numbered from 0 in the order the query declares them: what a test of the
 * fork (`tests/declared.rs`) reads. Null for a query the backend never sends (refused or out of scope in the envelope, a history
 * block, a kind the layer does not answer).
 */
export function nativeCase(context: { engine: Engine; config: MechanicsConfig; snapshot?: RulesSnapshot }, query: ParsedMechanicsQuery): { sets: WireSet[]; query: WireQuery } | null {
  const envelope = probeEnvelope(context, query);
  if (envelope.kind === "done" || hasHistory(query.query)) return null;
  const sets = envelope.sets.map((set: PokemonSet) => wireSet(set));
  const wire = wireQuery(query.query, sets.map((_, i) => i));
  return wire === null ? null : { sets, query: wire };
}
