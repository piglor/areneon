// The TS side of the policy worker (ADR-010 §2): spawns the Python worker with
// stdio only, opens the session with the handshake (ADR-008 §3), and answers
// `ask` with a re-checked Recommendation. Every failure is classified for
// `guardActor`: a dead worker is a `PolicyWorkerError`, a bad reply a
// `PolicyReplyError` (a policy forfeit), an unsupported candidate an
// `UnsupportedCandidateError` and a query the client itself cannot send a
// `PolicyProtocolError` (both harness incidents). Nothing is retried; a
// crashed worker is restarted between games (`restart`), never mid-game.
//
// Overrun rule: `guardActor` cannot cancel a query that ran past its budget,
// so that query stays outstanding (its battle id and its `n_parallel` slot
// stay taken) until the worker answers, and a serial worker makes the next
// game's queries wait behind it. Call `beginGame()` before every game, while
// no game is running on the client: it restarts the worker if a query was
// abandoned or the session has failed, so the next game gets a clean worker
// and is never charged for the last one's overrun. Without a restart it does
// nothing.
//
// Frames are as strict as the worker's: a reply line must be the JCS bytes of
// its value (no duplicate keys, no spacing, no floats), or it is invalid.

import { type ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import type { Ready } from "@areneon/schema/types/v1/policy-handshake.ts";
import type { Recommendation } from "@areneon/schema/types/v1/recommendation.ts";
import { type Json, jcs } from "./canonical-json.ts";
import { PolicyWorkerError, UnsupportedCandidateError } from "./evaluation-harness.ts";
import { answersRequest } from "./episode.ts";
import {
  contractVersions,
  encodeFrame,
  PolicyProtocolError,
  type PolicyQuery,
  type PolicyQueryInput,
  readHandshake,
  readPolicyQuery,
  readPolicyReply,
} from "./policy-protocol.ts";

/** What the actor needs from a worker: one query, one re-checked answer. */
export interface PolicyClient {
  /**
   * Ask for one choice. `battle` names the game: a second query for a battle
   * that still has one outstanding is refused, so the worker never holds two.
   */
  ask(battle: string, query: PolicyQueryInput): Promise<Recommendation>;
}

/** The worker's reply is invalid: schema-invalid, for another query, wrong identity, or a choice outside the candidates. A policy forfeit. */
export class PolicyReplyError extends Error {
  override name = "PolicyReplyError";
}

/** The worker is not the one the run expects. Raised before any battle starts. */
export class PolicyHandshakeError extends Error {
  override name = "PolicyHandshakeError";
}

export interface PolicyWorkerOptions {
  /** The program to run, resolved against `env.PATH`. */
  command: string;
  args: readonly string[];
  cwd?: string;
  /** The child's whole environment: it inherits nothing else. */
  env?: Readonly<Record<string, string>>;
  /** The longest line either side may send (`TrainingRunConfig`). */
  max_line_bytes: number;
  /** Most queries outstanding at once, one per battle (`n_parallel`). */
  n_parallel: number;
  /** How long the worker may take to answer the hello. */
  handshake_timeout_ms: number;
  /** What the run expects the worker to have loaded. */
  expect: {
    encoder_version: string;
    engine_data_hash: string;
    checkpoints: Ready["checkpoints"];
  };
}

interface Outstanding {
  query: PolicyQuery;
  resolve: (recommendation: Recommendation) => void;
  reject: (error: Error) => void;
}

const STDERR_TAIL = 2048;
const OUT_OF_MEMORY = /MemoryError|out of memory/i;

/**
 * What a forfeit detail may say of a worker's stderr: the class of its last
 * line (`MemoryError` of `MemoryError: ...`) and nothing else, since the rest
 * may hold values from a query.
 */
function errorClass(stderr: string): string {
  const last = stderr.trim().split("\n").at(-1) ?? "";
  return /^([A-Za-z_][A-Za-z0-9_.]*):/.exec(last)?.[1] ?? "unknown error";
}

/** A `PolicyClient` on a child process speaking the worker protocol. */
export class PolicyWorkerClient implements PolicyClient {
  readonly #options: PolicyWorkerOptions;
  #child: ChildProcessWithoutNullStreams | undefined;
  #failure: Error | undefined;
  #closing = false;
  #stderr = "";
  #rest: Buffer = Buffer.alloc(0);
  #onReady: ((line: unknown) => void) | undefined;
  #rejectReady: ((error: Error) => void) | undefined;
  #nextId = 1;
  readonly #outstanding = new Map<number, Outstanding>();
  readonly #battles = new Set<string>();
  readonly #waiting: (() => void)[] = [];

  constructor(options: PolicyWorkerOptions) {
    if (!Number.isInteger(options.n_parallel) || options.n_parallel < 1) {
      throw new RangeError("n_parallel must be a positive integer");
    }
    if (!Number.isInteger(options.max_line_bytes) || options.max_line_bytes < 1) {
      throw new RangeError("max_line_bytes must be a positive integer");
    }
    this.#options = options;
  }

  /** Spawn the worker and open the session; resolves once the handshake agrees. */
  async start(): Promise<void> {
    if (this.#child !== undefined) throw new Error("the policy worker is already started");
    const { command, args, cwd, env = {}, handshake_timeout_ms, max_line_bytes } = this.#options;
    this.#failure = undefined;
    this.#closing = false;
    this.#stderr = "";
    this.#rest = Buffer.alloc(0);
    this.#nextId = 1;
    const child = spawn(command, [...args], { cwd, env: { ...env }, stdio: ["pipe", "pipe", "pipe"] });
    this.#child = child;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const ready = new Promise<unknown>((resolve, reject) => {
      this.#onReady = resolve;
      timer = setTimeout(
        () => this.#fail(new PolicyHandshakeError(`no handshake reply within ${handshake_timeout_ms} ms`)),
        handshake_timeout_ms,
      );
      child.once("close", () => clearTimeout(timer));
      this.#rejectReady = reject;
    });
    // If the hello cannot be written, `ready` is never awaited: its later rejection is not unhandled.
    ready.catch(() => undefined);
    child.stdout.on("data", (chunk: Buffer) => this.#read(chunk));
    child.stderr.on("data", (chunk: Buffer) => {
      this.#stderr = (this.#stderr + chunk.toString("utf8")).slice(-STDERR_TAIL);
    });
    child.stdin.on("error", () => {
      // A broken pipe: the close handler reports why the worker is gone.
    });
    child.once("error", (error) => this.#fail(new PolicyWorkerError("crash", `cannot start the worker: ${error.message}`)));
    child.once("close", (code, signal) => this.#closed(code, signal));
    try {
      child.stdin.write(encodeFrame({ schema_version: 1, message: { kind: "hello", contracts: contractVersions() } }, max_line_bytes));
      this.#check(await ready);
    } catch (error) {
      await this.close();
      throw error;
    } finally {
      // The timer bounds the handshake only: left running it would end every
      // session that lasts that long, killing the worker mid-game (#912).
      clearTimeout(timer);
    }
  }

  #check(line: unknown): void {
    const { message } = readHandshake(line);
    if (message.kind !== "ready") throw new PolicyHandshakeError("the worker did not answer the hello with ready");
    const { expect } = this.#options;
    const same = (a: unknown, b: unknown): boolean => jcs(a as Json) === jcs(b as Json);
    if (!same(message.contracts, contractVersions())) {
      throw new PolicyHandshakeError(`the worker speaks other contract versions: ${jcs(message.contracts as unknown as Json)}`);
    }
    if (message.encoder_version !== expect.encoder_version) {
      throw new PolicyHandshakeError(`encoder_version ${message.encoder_version} is not the expected ${expect.encoder_version}`);
    }
    if (message.engine_data_hash !== expect.engine_data_hash) {
      throw new PolicyHandshakeError(`engine_data_hash ${message.engine_data_hash} is not the expected ${expect.engine_data_hash}`);
    }
    if (!same(message.checkpoints, expect.checkpoints)) {
      throw new PolicyHandshakeError(`the worker loaded other checkpoints: ${jcs(message.checkpoints as unknown as Json)}`);
    }
  }

  async ask(battle: string, input: PolicyQueryInput): Promise<Recommendation> {
    if (this.#failure !== undefined) throw this.#failure;
    if (this.#child === undefined) throw new PolicyWorkerError("crash", "the policy worker is not started");
    if (this.#battles.has(battle)) {
      throw new PolicyProtocolError(`battle ${battle} already has a query outstanding`);
    }
    this.#battles.add(battle);
    try {
      while (this.#outstanding.size >= this.#options.n_parallel) {
        await new Promise<void>((resolve) => this.#waiting.push(resolve));
        if (this.#failure !== undefined) throw this.#failure;
      }
      const query = readPolicyQuery({ ...input, query_id: this.#nextId++ });
      const frame = encodeFrame(query, this.#options.max_line_bytes);
      return await new Promise<Recommendation>((resolve, reject) => {
        this.#outstanding.set(query.query_id, { query, resolve, reject });
        (this.#child as ChildProcessWithoutNullStreams).stdin.write(frame);
      });
    } finally {
      this.#battles.delete(battle);
      this.#waiting.shift()?.();
    }
  }

  /** End the session: close the worker's stdin and wait for it to exit (killing it after a second). */
  async close(): Promise<void> {
    const child = this.#child;
    if (child === undefined) return;
    this.#closing = true;
    const exited = new Promise<void>((resolve) => child.once("close", () => resolve()));
    child.stdin.end();
    const kill = setTimeout(() => child.kill("SIGKILL"), 1000);
    await exited;
    clearTimeout(kill);
  }

  /**
   * Call before each game, with no game running on this client: restarts the
   * worker if a query was abandoned (it overran its budget) or the session
   * failed; otherwise does nothing. See the overrun rule above.
   */
  async beginGame(): Promise<void> {
    if (this.#child === undefined || this.#failure !== undefined || this.#outstanding.size > 0) {
      await this.restart();
    }
  }

  /** Close, then start again: between games only, never within one. */
  async restart(): Promise<void> {
    await this.close();
    await this.start();
  }

  #read(chunk: Buffer): void {
    const max = this.#options.max_line_bytes;
    this.#rest = Buffer.concat([this.#rest, chunk]);
    while (this.#child !== undefined && !this.#closing) {
      const end = this.#rest.indexOf(10);
      if (end < 0) {
        if (this.#rest.length > max) this.#fail(new PolicyReplyError(`a reply line is longer than ${max} bytes`));
        return;
      }
      const line = this.#rest.subarray(0, end);
      this.#rest = this.#rest.subarray(end + 1);
      if (line.length > max) {
        this.#fail(new PolicyReplyError(`a reply line is longer than ${max} bytes`));
        return;
      }
      this.#line(line);
    }
  }

  #line(bytes: Buffer): void {
    let value: unknown;
    try {
      const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
      value = JSON.parse(text);
      // JSON.parse keeps the last of two equal keys and reads 1.0 as 1; the
      // canonical form of what it kept differs from the line in both cases.
      if (jcs(value as Json) !== text) {
        this.#fail(new PolicyReplyError("a reply line is not in RFC 8785 (JCS) canonical form"));
        return;
      }
    } catch {
      this.#fail(new PolicyReplyError("a reply line is not UTF-8 JSON"));
      return;
    }
    if (this.#onReady !== undefined) {
      const ready = this.#onReady;
      this.#onReady = undefined;
      ready(value);
      return;
    }
    const id = (value as { query_id?: unknown } | null)?.query_id;
    const pending = typeof id === "number" ? this.#outstanding.get(id) : undefined;
    if (pending === undefined) {
      this.#fail(new PolicyReplyError("a reply names no outstanding query"));
      return;
    }
    this.#outstanding.delete(pending.query.query_id);
    try {
      pending.resolve(this.#answer(pending.query, value));
    } catch (error) {
      pending.reject(error as Error);
    }
  }

  /** The Recommendation of a reply, once it is of the query's request and among its candidates. */
  #answer(query: PolicyQuery, value: unknown): Recommendation {
    let reply;
    try {
      reply = readPolicyReply(value).reply;
    } catch (error) {
      throw new PolicyReplyError((error as Error).message);
    }
    if (reply.kind === "error") {
      if (reply.error === "unsupported_candidate") throw new UnsupportedCandidateError(reply.detail);
      throw new PolicyProtocolError(`the worker refused the query: ${reply.detail}`);
    }
    const { recommendation } = reply;
    if (!answersRequest(recommendation, query.identity)) {
      throw new PolicyReplyError("the reply answers another request than the query's");
    }
    if (recommendation.source_policy_id !== query.policy_id) {
      throw new PolicyReplyError(`the reply names policy ${recommendation.source_policy_id}, not ${query.policy_id}`);
    }
    if (!query.candidates.some((candidate) => candidate.choice === recommendation.choice)) {
      throw new PolicyReplyError(`the reply's choice "${recommendation.choice}" is not among the candidates`);
    }
    return recommendation;
  }

  #closed(code: number | null, signal: NodeJS.Signals | null): void {
    const tail = this.#stderr.trim();
    const cause = OUT_OF_MEMORY.test(tail) || signal === "SIGKILL" ? "out_of_memory" : "crash";
    const summary = tail === "" ? "" : `; ${errorClass(tail)}`;
    const how = signal === null ? `exit code ${code}` : `signal ${signal}`;
    this.#child = undefined;
    if (this.#closing) {
      this.#fail(new PolicyWorkerError("crash", "the policy worker was closed"), false);
    } else {
      this.#fail(new PolicyWorkerError(cause, `the policy worker ended (${how}${summary})`), false);
    }
  }

  /** The session is over: nothing more is sent or accepted, and everything outstanding fails with `error`. */
  #fail(error: Error, kill = true): void {
    this.#failure ??= error;
    if (kill) this.#child?.kill("SIGKILL");
    this.#rejectReady?.(error);
    this.#onReady = undefined;
    const failed = [...this.#outstanding.values()];
    this.#outstanding.clear();
    for (const pending of failed) pending.reject(error);
    for (const resume of this.#waiting.splice(0)) resume();
  }
}
