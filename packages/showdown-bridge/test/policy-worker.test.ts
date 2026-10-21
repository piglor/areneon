import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";
import { PolicyWorkerError, UnsupportedCandidateError } from "../src/evaluation-harness.ts";
import { PolicyProtocolError, type PolicyQueryInput } from "../src/policy-protocol.ts";
import { PolicyHandshakeError, PolicyReplyError, PolicyWorkerClient, type PolicyWorkerOptions } from "../src/policy-worker.ts";
import { SAMPLES_DIR } from "../src/samples.ts";
import { CHECKPOINTS, ENCODER_VERSION, ENGINE_DATA_HASH, MAX_LINE_BYTES } from "./policy-fake.ts";

const FAKE = fileURLToPath(new URL("./fake-policy-worker.ts", import.meta.url));
const { query_id: _id, ...INPUT } = JSON.parse(readFileSync(`${SAMPLES_DIR}/v1/policy-query/valid/battle.json`, "utf8")) as PolicyQueryInput & { query_id: number };

const opened: PolicyWorkerClient[] = [];
afterEach(async () => {
  await Promise.all(opened.splice(0).map((client) => client.close()));
});

function client(mode: string, change: Partial<PolicyWorkerOptions> = {}): PolicyWorkerClient {
  const made = new PolicyWorkerClient({
    command: process.execPath,
    args: [FAKE, mode],
    env: {},
    max_line_bytes: MAX_LINE_BYTES,
    n_parallel: 1,
    handshake_timeout_ms: 10_000,
    expect: { encoder_version: ENCODER_VERSION, engine_data_hash: ENGINE_DATA_HASH, checkpoints: CHECKPOINTS },
    ...change,
  });
  opened.push(made);
  return made;
}

async function started(mode: string, change: Partial<PolicyWorkerOptions> = {}): Promise<PolicyWorkerClient> {
  const made = client(mode, change);
  await made.start();
  return made;
}

describe("the handshake (ADR-008 section 3)", () => {
  it("opens a session with a worker that speaks the same contracts and loaded what is expected", async () => {
    const worker = await started("ok");
    const recommendation = await worker.ask("g1", INPUT);
    expect(recommendation.identity).toEqual(INPUT.identity);
    expect(INPUT.candidates.map((c) => c.choice)).toContain(recommendation.choice);
  });

  it.each([
    ["bad-contracts", /other contract versions/],
    ["bad-encoder", /encoder_version other-1 is not the expected fake-1/],
    ["bad-engine", /engine_data_hash 0{64} is not the expected e{64}/],
    ["bad-checkpoints", /other checkpoints/],
    ["no-ready", /did not answer the hello with ready/],
  ])("refuses a worker that differs (%s) before any battle", async (mode, message) => {
    await expect(client(mode).start()).rejects.toThrow(PolicyHandshakeError);
    await expect(client(mode).start()).rejects.toThrow(message);
  });

  it("refuses a worker that does not answer the hello in time", async () => {
    await expect(client("silent", { handshake_timeout_ms: 300 }).start()).rejects.toThrow(/no handshake reply within 300 ms/);
  });

  it("does not end a session that outlives the handshake timeout", async () => {
    const worker = await started("ok", { handshake_timeout_ms: 2000 });
    await new Promise((resolve) => setTimeout(resolve, 2300));
    const recommendation = await worker.ask("g1", INPUT);
    expect(INPUT.candidates.map((c) => c.choice)).toContain(recommendation.choice);
  }, 15_000);

  it("closes the worker and clears the timer when the hello cannot be written", async () => {
    const cleared = vi.spyOn(globalThis, "clearTimeout");
    try {
      const worker = client("ok", { max_line_bytes: 8 });
      await expect(worker.start()).rejects.toThrow();
      // the failed start left no child behind: a second start is not "already started"
      await expect(worker.start()).rejects.not.toThrow(/already started/);
      expect(cleared).toHaveBeenCalled();
    } finally {
      cleared.mockRestore();
    }
  });

  it("refuses a hello answered with something that is not a handshake", async () => {
    await expect(client("garbage-ready").start()).rejects.toThrow(/not UTF-8 JSON/);
  });

  it("reports a worker that cannot be started as a crash", async () => {
    const error = await client("ok", { command: "/nonexistent/worker" }).start().catch((e: unknown) => e);
    expect(error).toBeInstanceOf(PolicyWorkerError);
    expect((error as PolicyWorkerError).worker_cause).toBe("crash");
  });

  it("cannot be started twice, nor asked before it is started", async () => {
    const worker = await started("ok");
    await expect(worker.start()).rejects.toThrow(/already started/);
    await expect(client("ok").ask("g1", INPUT)).rejects.toThrow(PolicyWorkerError);
  });

  it("refuses options that make no sense", () => {
    expect(() => client("ok", { n_parallel: 0 })).toThrow(RangeError);
    expect(() => client("ok", { max_line_bytes: 0 })).toThrow(RangeError);
  });
});

describe("failures are classified for guardActor (ADR-010 section 2)", () => {
  it.each([
    ["crash", "crash", /exit code 3/],
    ["oom", "out_of_memory", /MemoryError/],
    ["kill", "out_of_memory", /signal SIGKILL/],
  ])("a worker that dies (%s) is a PolicyWorkerError, and nothing is retried", async (mode, cause, detail) => {
    const worker = await started(mode);
    const error = await worker.ask("g1", INPUT).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(PolicyWorkerError);
    expect((error as PolicyWorkerError).worker_cause).toBe(cause);
    expect((error as Error).message).toMatch(detail);
    // The session is over: a later query fails at once with the same cause.
    await expect(worker.ask("g2", INPUT)).rejects.toBe(error);
  });

  it.each([
    ["malformed", /not UTF-8 JSON/],
    ["bad-utf8", /not UTF-8 JSON/],
    ["spaced", /not in RFC 8785/],
    ["duplicate-key", /not in RFC 8785/],
    ["unknown-id", /names no outstanding query/],
    ["long", /longer than 4000000 bytes/],
    ["long-unterminated", /longer than 4000000 bytes/],
  ])("a malformed frame (%s) is an invalid reply and ends the session", async (mode, message) => {
    const worker = await started(mode);
    const error = await worker.ask("g1", INPUT).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(PolicyReplyError);
    expect((error as Error).message).toMatch(message);
    await expect(worker.ask("g2", INPUT)).rejects.toBe(error);
  });

  it.each([
    ["bad-schema", /not a PolicyReply/],
    ["wrong-identity", /another request/],
    ["wrong-choice", /not among the candidates/],
    ["wrong-policy", /names policy policy:other/],
  ])("an invalid reply (%s) is a PolicyReplyError", async (mode, message) => {
    const worker = await started(mode);
    await expect(worker.ask("g1", INPUT)).rejects.toThrow(PolicyReplyError);
    await expect(worker.ask("g2", INPUT)).rejects.toThrow(message);
  });

  it("a schema-invalid reply for one query leaves the session usable for the next game", async () => {
    const worker = await started("bad-then-ok");
    await expect(worker.ask("g1", INPUT)).rejects.toThrow(PolicyReplyError);
    expect((await worker.ask("g2", INPUT)).identity).toEqual(INPUT.identity);
  });

  it("an unsupported candidate is a harness incident, not a policy failure", async () => {
    const worker = await started("unsupported");
    await expect(worker.ask("g1", INPUT)).rejects.toThrow(UnsupportedCandidateError);
  });

  it("a refused query is a protocol fault of the harness", async () => {
    const worker = await started("invalid-query");
    await expect(worker.ask("g1", INPUT)).rejects.toThrow(PolicyProtocolError);
  });

  it("refuses to send a query above the candidate cap, or above the line limit", async () => {
    const worker = await started("ok");
    const many: PolicyQueryInput = { ...INPUT, candidates: Array.from({ length: 1025 }, (_, i) => ({ choice: `team ${i}`, pending: false })) as PolicyQueryInput["candidates"] };
    await expect(worker.ask("g1", many)).rejects.toThrow(/not a PolicyQuery/);
    const small = await started("ok", { max_line_bytes: 3000 });
    await expect(small.ask("g1", INPUT)).rejects.toThrow(/exceeds the maximum line length 3000/);
  });
});

describe("a worker's stderr", () => {
  it("never reaches a forfeit detail beyond the class of its last line", async () => {
    const worker = await started("leaky");
    const error = (await worker.ask("g1", INPUT).catch((e: unknown) => e)) as PolicyWorkerError;
    expect(error).toBeInstanceOf(PolicyWorkerError);
    expect(error.message).toContain("ValueError");
    for (const leaked of ["bad value", "species", "Traceback"]) expect(error.message).not.toContain(leaked);
  });
});

describe("one query per battle, at most n_parallel at once (ADR-010 section 2)", () => {
  it("refuses a second query for a battle that has one outstanding", async () => {
    const worker = await started("slow");
    const first = worker.ask("g1", INPUT);
    await expect(worker.ask("g1", INPUT)).rejects.toThrow(/already has a query outstanding/);
    await first;
    // Answered: the battle may ask again.
    await worker.ask("g1", INPUT);
  });

  it("holds a query back while n_parallel are outstanding", async () => {
    const serial = await started("concurrency", { n_parallel: 1 });
    const replies = await Promise.all([serial.ask("g1", INPUT), serial.ask("g2", INPUT), serial.ask("g3", INPUT)]);
    expect(replies.map((r) => r.elapsed_ms)).toEqual([1, 1, 1]);
    const parallel = await started("concurrency", { n_parallel: 2 });
    const both = await Promise.all([parallel.ask("g1", INPUT), parallel.ask("g2", INPUT), parallel.ask("g3", INPUT)]);
    expect(Math.max(...both.map((r) => r.elapsed_ms))).toBe(2);
  });

  it("fails the queries that wait when the worker dies", async () => {
    const worker = await started("crash");
    const results = await Promise.allSettled([worker.ask("g1", INPUT), worker.ask("g2", INPUT)]);
    expect(results.map((r) => r.status)).toEqual(["rejected", "rejected"]);
  });
});

describe("restart and close", () => {
  it("beginGame leaves a healthy worker running, and restarts one that has an abandoned query", async () => {
    const worker = await started("slow");
    await worker.beginGame();
    const abandoned = worker.ask("g1", INPUT).catch((e: unknown) => e);
    // The query is still outstanding (a guard gave up on it): its battle id is taken.
    await expect(worker.ask("g1", INPUT)).rejects.toThrow(/already has a query outstanding/);
    await worker.beginGame();
    expect(await abandoned).toBeInstanceOf(PolicyWorkerError);
    expect((await worker.ask("g1", INPUT)).identity).toEqual(INPUT.identity);
  });

  it("restarts a crashed worker between games, with a fresh session", async () => {
    const worker = await started("crash");
    await expect(worker.ask("g1", INPUT)).rejects.toThrow(PolicyWorkerError);
    await worker.restart();
    // Still the crashing fake: the new session begins, and fails on its own query.
    await expect(worker.ask("g2", INPUT)).rejects.toThrow(/exit code 3/);
  });

  it("closes cleanly and is idempotent", async () => {
    const worker = await started("ok");
    await worker.close();
    await worker.close();
    await expect(worker.ask("g1", INPUT)).rejects.toThrow(/closed/);
  });

  it("closes a worker that is slow to end by killing it", async () => {
    const worker = await started("slow");
    const pending = worker.ask("g1", INPUT).catch((e: unknown) => e);
    await worker.close();
    expect(await pending).toBeInstanceOf(PolicyWorkerError);
  });
});
