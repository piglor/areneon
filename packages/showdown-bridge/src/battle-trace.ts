/* eslint-disable @typescript-eslint/no-explicit-any */
// The forced-draw trace recorder on the unmodified pinned engine (ADR-028 section 5 F0, ADR-030 groundwork G2, #1462). The
// engine draws every random number through `battle.prng`; the recorder replaces it, at run time and on a battle `runBattle`
// plays, with a source that delegates to the engine's own generator and records EVERY member call (method, arguments, result,
// in order). The forcing source (the oracle backend's, `battle-replay.ts`) answers the same calls from a trace instead: the
// engine still runs its own generator in lock step, so its state and serialisation stay the engine's, and a call the trace has
// no matching answer for gets the generator's own answer and is reported as a draw request the comparison flags.
//
// Isolation (ADR-030 rule 1): this file and `battle-replay.ts` are the only bridge files that install a random source on a
// live battle outside the mechanics probes' scenario battles (`fidelity-isolation.test.ts` lists this file); a source scan
// (`battle-trace-isolation.test.ts`) keeps both out of evaluation, progress, private-test, sealed and A2 paths. A trace holds
// both sides' hidden information: an evaluation artifact, never an Episode, never actor input.
import { createHash } from "node:crypto";
import type { Action, BattleTrace, Draw, MaskEntry, Request, Step } from "@areneon/schema/types/v1/battle-trace.ts";
import { CloneJudge, legalChoiceSet } from "./choice-judge.ts";
import { enumerateCandidates } from "./choices.ts";
import type { Engine, PokemonSet, SimBattle } from "./engine.ts";
import { decodeRequest, type DecodedRequest } from "./request.ts";
import { type Actor, type BattleRecord, type HiddenVerdicts, runBattle } from "./runner.ts";
import type { RulesSnapshot } from "./rules.ts";

export type { Action, BattleTrace, Draw, MaskEntry, PokemonSet, Request, Step };

/** Bumped when the recorder records anything differently. */
export const TRACE_FORMAT = 1;

export type Seat = "p1" | "p2";
const SEATS: readonly Seat[] = ["p1", "p2"];

export class TraceRecordError extends Error {
  override name = "TraceRecordError";
}

/** The members of the engine's PRNG that draw nothing and are passed through. */
const PASS = new Set(["getSeed", "clone"]);

const text = (value: unknown): string => (typeof value === "string" ? value : String(value));
const indices = (n: number): number[] => Array.from({ length: n }, (_, i) => i);

/**
 * Decides what a call answers. `natural` is the engine generator's own result (encoded); the return value is the result to use
 * (encoded the same way). The recorder returns `natural`; the forcing source returns a forced result when the trace has one.
 */
export type Answer = (method: string, args: string[], natural: string[]) => string[];

/**
 * A random source over the engine's generator `target`: every string-named member function but `getSeed` and `clone` is
 * intercepted. `random`, `randomChance`, `sample` and `shuffle` are run on the generator exactly as the engine would run them
 * (`sample` and `shuffle` on an index array, so their results are integers), and their answers decoded back; any other member
 * is called on the generator and answered with what it returns.
 */
export function interceptedSource(target: any, answer: Answer): any {
  return new Proxy(target, {
    get(t, name) {
      const value = Reflect.get(t, name);
      if (typeof name !== "string" || typeof value !== "function" || PASS.has(name)) return typeof value === "function" ? value.bind(t) : value;
      switch (name) {
        case "random":
          return (...args: unknown[]): number => {
            const given = args.filter((a) => a !== undefined);
            return Number(answer(name, given.map(text), [text(t.random(...args))])[0]);
          };
        case "randomChance":
          return (numerator: number, denominator: number): boolean => answer(name, [text(numerator), text(denominator)], [text(t.randomChance(numerator, denominator))])[0] === "true";
        case "sample":
          return (items: readonly unknown[]): unknown => items[Number(answer(name, [text(items.length)], [text(t.sample(indices(items.length)))])[0])];
        case "shuffle":
          return (items: unknown[], ...range: unknown[]): void => {
            const order = indices(items.length);
            t.shuffle(order, ...range);
            const given = range.filter((a) => a !== undefined);
            const used = answer(name, [text(items.length), ...given.map(text)], order.map(text));
            const before = [...items];
            used.forEach((from, i) => {
              items[i] = before[Number(from)];
            });
          };
        default:
          return (...args: unknown[]): unknown => {
            const result = value.apply(t, args);
            answer(name, args.map(text), [text(result)]);
            return result;
          };
      }
    },
  });
}

/** A recording source: the engine's own answers, every call logged to `log()` (the current write's draw list). */
export function recordingSource(target: any, log: () => Draw[]): any {
  return interceptedSource(target, (method, args, natural) => {
    log().push({ method, args, result: natural });
    return natural;
  });
}

/** A forcing source: answers calls from the draws `load` gave, in order; `requests()` lists the calls made since, with the answers used. */
export interface ForcingSource {
  source: any;
  load(draws: readonly Draw[]): void;
  requests(): Draw[];
}

const sameCall = (d: Draw | undefined, method: string, args: string[]): d is Draw => d !== undefined && d.method === method && d.args.length === args.length && d.args.every((a, i) => a === args[i]);

export function forcingSource(target: any): ForcingSource {
  let queue: readonly Draw[] = [];
  let made: Draw[] = [];
  const source = interceptedSource(target, (method, args, natural) => {
    const forced = queue[made.length];
    const result = sameCall(forced, method, args) ? [...forced.result] : natural;
    made.push({ method, args, result });
    return result;
  });
  return {
    source,
    load(draws) {
      queue = draws;
      made = [];
    },
    requests: () => made,
  };
}

/** Install `source` as a live battle's random source (the one assignment the isolation scan allows here). */
export const installSource = (battle: any, source: any): void => {
  battle.prng = source;
};

// The engine's channel split (its `extractChannelMessages`, which `getPlayerStreams` applies to every `update`), for the
// omniscient channel and the two players.
const SPLIT = /^\|split\|p([1234])\n(.*)\n(.*)|.+/gm;

/** What one engine message gives each channel, as the player streams deliver it (one chunk, split into lines, `|t:|` dropped). */
export interface Emitted {
  omniscient: string[];
  p1: string[];
  p2: string[];
}

const chunkLines = (lines: string[]): string[] => (lines.length === 0 ? [""] : lines).filter((l) => !l.startsWith("|t:|"));

/** Split one `pushMessage(type, data)` of the engine into the lines each channel gets (`end` gives none). */
export function emitted(type: string, data: string): Emitted {
  if (type === "update") {
    const out: Record<"-1" | "1" | "2", string[]> = { "-1": [], 1: [], 2: [] };
    for (const [line, player, secret, shared] of data.matchAll(SPLIT)) {
      for (const channel of ["-1", "1", "2"] as const) {
        const value = player === undefined ? line : channel === "-1" || channel === player ? secret : shared;
        if (value) out[channel].push(value);
      }
    }
    return { omniscient: chunkLines(out["-1"]), p1: chunkLines(out[1]), p2: chunkLines(out[2]) };
  }
  if (type === "sideupdate") {
    const at = data.indexOf("\n");
    const side = data.slice(0, at);
    const lines = chunkLines(data.slice(at + 1).split("\n"));
    return { omniscient: [], p1: side === "p1" ? lines : [], p2: side === "p2" ? lines : [] };
  }
  return { omniscient: [], p1: [], p2: [] };
}

/** Hooks of a traced BattleStream. */
export interface StreamHooks {
  /** Called with the battle once `>start` built it: install the random source here. */
  started(battle: any): void;
  write?(begin: boolean, battle: any): void;
  line?(type: string, message: string): void;
  message(type: string, data: string): void;
  /** Whether the stream still delivers messages to its readers (the recorder's runner reads them; the replay has none). */
  forward: boolean;
  error?(error: unknown): void;
}

/** A BattleStream subclass that reports every write, input line and engine message, synchronously and in order. */
export function tracedStreamClass(engine: Engine, hooks: StreamHooks): new () => any {
  const Base = engine.sim.BattleStream as unknown as new () => any;
  return class Traced extends Base {
    _write(chunk: string): void {
      hooks.write?.(true, this.battle);
      super._write(chunk);
      hooks.write?.(false, this.battle);
    }
    _writeLine(type: string, message: string): void {
      hooks.line?.(type, message);
      super._writeLine(type, message);
      if (type === "start") hooks.started(this.battle);
    }
    pushMessage(type: string, data: string): void {
      hooks.message(type, data);
      if (hooks.forward) super.pushMessage(type, data);
    }
    pushError(error: unknown, recoverable?: boolean): void {
      hooks.error?.(error);
      if (hooks.forward) super.pushError(error, recoverable);
    }
  };
}

/** SHA-256 of the JSON text of the engine's serialised state, its log's wall-clock `|t:|` lines dropped (the only part that is not reproducible). */
export function stateHash(battle: any): string {
  const state = battle.toJSON();
  return createHash("sha256").update(JSON.stringify({ ...state, log: (state.log as string[]).filter((l) => !l.startsWith("|t:|")) })).digest("hex");
}

/** The mask of one request: every candidate in canonical order, non-pending ones legal, pending ones with the engine's hidden verdict. */
export function maskOf(request: DecodedRequest, verdicts: readonly { choice: string; legal: boolean }[]): MaskEntry[] {
  const byChoice = new Map(verdicts.map((v) => [v.choice, v.legal]));
  return enumerateCandidates(request).candidates.map(({ choice, pending }) => {
    if (!pending) return { choice, verdict: "legal" as const };
    const legal = byChoice.get(choice);
    if (legal === undefined) throw new TraceRecordError("a pending candidate has no hidden verdict");
    return { choice, verdict: legal ? ("pending_legal" as const) : ("pending_illegal" as const) };
  });
}

/**
 * The requests of a step as the battle stands after it: each request's identity and, unless it is a wait request or the battle
 * has ended, the mask of its candidates judged by the engine's `CloneJudge` on this battle (pending ones with their hidden
 * verdict). Judging at the step's end makes the mask a function of the battle, never of when a runner's player loop got to it
 * (a request the runner's turn cap overtook is judged here and unjudged there). `seqs` is advanced.
 */
export function stepRequests(engine: Engine, battle: SimBattle, bodies: Record<Seat, readonly string[]>, seqs: Record<Seat, number>): Record<Seat, Request[]> {
  const out: Record<Seat, Request[]> = { p1: [], p2: [] };
  for (const seat of SEATS) {
    for (const body of bodies[seat]) {
      const decoded = decodeRequest(body);
      const request = { ...decoded, request_seq: (seqs[seat] += 1) };
      if (decoded.phase === "wait" || battle.ended) {
        out[seat].push({ request_seq: request.request_seq, request_hash: decoded.request_hash });
        continue;
      }
      const judge = new CloneJudge(engine, battle, seat);
      const legal = legalChoiceSet({ request, showdown_commit: engine.commit, judge });
      const verdicts = legal.pending.length > 0 ? judge.judge(decoded.request_hash, legal.pending) : [];
      out[seat].push({ request_seq: request.request_seq, request_hash: decoded.request_hash, mask: maskOf(decoded, verdicts) });
    }
  }
  return out;
}

/** What one write to the stream did. */
interface Write {
  actions: Action[];
  draws: Draw[];
  channels: Record<Seat, string[]>;
  omniscient: string[];
  bodies: Record<Seat, string[]>;
  requests: Record<Seat, Request[]>;
  ended: boolean;
  state: string;
  unsupported: boolean;
}

const newWrite = (): Write => ({ actions: [], draws: [], channels: { p1: [], p2: [] }, omniscient: [], bodies: { p1: [], p2: [] }, requests: { p1: [], p2: [] }, ended: false, state: "", unsupported: false });

/** One battle to record: as `runBattle` takes it, plus a name. */
export interface TraceSpec {
  battle_id: string;
  format_id: string;
  seed: string;
  teams: Record<Seat, PokemonSet[]>;
  actors: Record<Seat, Actor>;
  max_turns: number;
  snapshot?: RulesSnapshot;
}

/** Collect a message's lines into a step's or write's output (the bodies of its `|request|` lines apart). */
export function addEmitted(into: { channels: Record<Seat, string[]>; omniscient: string[]; bodies: Record<Seat, string[]> }, type: string, data: string): void {
  const out = emitted(type, data);
  into.omniscient.push(...out.omniscient);
  for (const seat of SEATS) {
    into.channels[seat].push(...out[seat]);
    for (const line of out[seat]) if (line.startsWith("|request|")) into.bodies[seat].push(line.slice("|request|".length));
  }
}

const closes = (w: Pick<Write, "ended" | "bodies">, first: boolean): boolean => first || w.ended || SEATS.some((seat) => w.bodies[seat].length > 0);

/**
 * Play `spec` with `runBattle` on the unmodified engine, recording every draw, write and message, and build its trace. The runner
 * plays as it always does (its own judge, its own channels). At the end of each step the recorder judges the step's requests on
 * the live battle (`stepRequests`), as the replay does; where the runner judged a request too, its legal set and hidden
 * verdicts must agree. Refused when an actor forfeits (a harness command the trace has no action for).
 */
/** Test seam: how the recorder wraps the engine's generator (default `recordingSource`); a test proves its check has teeth with it. */
export interface RecordOptions {
  source?: (prng: any, log: () => Draw[]) => any;
  /**
   * Called with the live battle at the end of every step, in step order (step 0: the start), after its requests are judged: the
   * battle-state projection's trace reader (`projection-points.ts`, G4) reads `toJSON()` here. It must not change the battle.
   */
  onStep?: (step: number, battle: any) => void;
}

export async function recordBattleTrace(engine: Engine, spec: TraceSpec, options: RecordOptions = {}): Promise<{ trace: BattleTrace; record: BattleRecord }> {
  const wrap = options.source ?? recordingSource;
  const writes: Write[] = [];
  const seqs: Record<Seat, number> = { p1: 0, p2: 0 };
  let current = newWrite();
  let steps = 0;
  const Traced = tracedStreamClass(engine, {
    forward: true,
    started: (battle) => installSource(battle, wrap(battle.prng, () => current.draws)),
    write(begin, battle) {
      if (begin) {
        current = newWrite();
        return;
      }
      current.ended = battle.ended === true;
      if (closes(current, writes.length === 0)) {
        current.requests = stepRequests(engine, battle, current.bodies, seqs);
        current.state = stateHash(battle);
        options.onStep?.(steps, battle);
        steps += 1;
      }
      writes.push(current);
    },
    line(type, message) {
      if (type === "p1" || type === "p2") current.actions.push({ kind: "choice", seat: type, choice: message });
      else if (type === "forcetie") current.actions.push({ kind: "command", command: "forcetie" });
      else if (type !== "start" && type !== "player") current.unsupported = true;
    },
    message: (type, data) => addEmitted(current, type, data),
  });
  const record = await runBattle({
    engine: { ...engine, sim: { ...engine.sim, BattleStream: Traced as never } },
    formatId: spec.format_id,
    seed: spec.seed,
    teams: spec.teams,
    actors: spec.actors,
    maxTurns: spec.max_turns,
    ...(spec.snapshot === undefined ? {} : { snapshot: spec.snapshot }),
  });
  if (record.forfeit !== undefined || writes.some((w) => w.unsupported)) throw new TraceRecordError("a forfeit or another harness command cannot be traced");
  const trace = buildTrace(engine, spec, record, writes);
  agreesWithRecord(trace, record);
  return { trace, record };
}

const emptyStep = (): Step => ({ actions: [], draws: [], channels: { p1: [], p2: [] }, omniscient: [], requests: { p1: [], p2: [] }, state_sha256: "" });

function buildTrace(engine: Engine, spec: TraceSpec, record: BattleRecord, writes: Write[]): BattleTrace {
  const steps: Step[] = [];
  let open = emptyStep();
  for (const [i, w] of writes.entries()) {
    open.actions.push(...w.actions);
    open.draws.push(...w.draws);
    open.omniscient.push(...w.omniscient);
    for (const seat of SEATS) {
      open.channels[seat].push(...w.channels[seat]);
      open.requests[seat].push(...w.requests[seat]);
    }
    if (closes(w, i === 0)) {
      open.state_sha256 = w.state;
      steps.push(open);
      open = emptyStep();
      if (w.ended) break;
    }
  }
  return {
    schema_version: 1,
    data_class: "synthetic",
    trace_format: TRACE_FORMAT,
    showdown_commit: engine.commit,
    format_id: spec.format_id,
    battle_id: spec.battle_id,
    seed: spec.seed,
    teams: structuredClone(spec.teams) as unknown as BattleTrace["teams"],
    team_hashes: { ...record.team_hashes },
    steps: steps as BattleTrace["steps"],
    outcome: { winner: record.winner, turns: record.turns },
  };
}

/** The trace must say what the runner recorded: its requests' hashes, and its legal sets and hidden verdicts where it judged. */
export function agreesWithRecord(trace: BattleTrace, record: BattleRecord): void {
  const verdicts = new Map(record.hidden_verdicts.map((h: HiddenVerdicts) => [`${h.player}:${h.request_seq}`, h.verdicts]));
  for (const seat of SEATS) {
    const requests = trace.steps.flatMap((s) => s.requests[seat]);
    if (requests.length !== record.requests[seat].length) throw new TraceRecordError("the trace and the runner saw another number of requests");
    for (const [i, r] of requests.entries()) {
      const judged = record.requests[seat][i] as BattleRecord["requests"][Seat][number];
      if (judged.request_hash !== r.request_hash) throw new TraceRecordError("the runner's request does not match the line it was sent");
      if (judged.legal === undefined) continue;
      const mask = r.mask ?? [];
      const hidden = new Map((verdicts.get(`${seat}:${r.request_seq}`) ?? []).map((v) => [v.choice, v.legal]));
      const same =
        JSON.stringify(mask.filter((m) => m.verdict === "legal").map((m) => m.choice)) === JSON.stringify(judged.legal.choices) &&
        JSON.stringify(mask.filter((m) => m.verdict !== "legal").map((m) => m.choice)) === JSON.stringify(judged.legal.pending) &&
        mask.every((m) => m.verdict === "legal" || hidden.get(m.choice) === (m.verdict === "pending_legal"));
      if (!same) throw new TraceRecordError("the runner's legal set or hidden verdicts differ from the trace's mask");
    }
  }
}
