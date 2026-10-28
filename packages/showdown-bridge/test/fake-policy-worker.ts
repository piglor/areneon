// A stand-in policy worker for the TS client tests: speaks the stdio protocol
// in Node (so no Python is needed) and misbehaves in the way argv[2] names.
import { createInterface } from "node:readline";
import { contractVersions, encodeFrame, type PolicyQuery } from "../src/policy-protocol.ts";
import { CHECKPOINTS, ENCODER_VERSION, ENGINE_DATA_HASH, MAX_LINE_BYTES } from "./policy-fake.ts";

const mode = process.argv[2] ?? "ok";
const write = (value: unknown): void => {
  process.stdout.write(encodeFrame(value, MAX_LINE_BYTES * 4));
};

const contracts = contractVersions();
const ready = {
  schema_version: 1,
  message: {
    kind: "ready",
    contracts: mode === "bad-contracts" ? [{ ...contracts[0], schema_sha256: "0".repeat(64) }, ...contracts.slice(1)] : contracts,
    encoder_version: mode === "bad-encoder" ? "other-1" : ENCODER_VERSION,
    engine_data_hash: mode === "bad-engine" ? "0".repeat(64) : ENGINE_DATA_HASH,
    checkpoints: mode === "bad-checkpoints" ? [CHECKPOINTS[0]] : CHECKPOINTS,
    // The feature set this worker's checkpoints need (ADR-022 section 3), when the test names one.
    ...(process.env.FAKE_FEATURE_SET === undefined ? {} : { feature_set: JSON.parse(process.env.FAKE_FEATURE_SET) as unknown }),
  },
};

const recommendation = (query: PolicyQuery, change: Record<string, unknown> = {}) => {
  const pool = query.candidates.filter((c) => !c.pending);
  const candidates = pool.length > 0 ? pool : query.candidates;
  const pick = query.sampling.kind === "sample" ? Number(query.sampling.u % candidates.length) : 0;
  return {
    schema_version: 1,
    identity: query.identity,
    choice: (candidates[pick] as { choice: string }).choice,
    source_policy_id: query.policy_id,
    elapsed_ms: 1,
    deadline_outcome: "in_budget",
    ...change,
  };
};

// An evaluation of an `evaluate` query (ADR-026 section 5): invented prior, one entry per candidate, in the query's order.
const evaluation = (query: PolicyQuery, change: Record<string, unknown> = {}) => ({
  identity: query.identity,
  prior_ppm: query.candidates.map((_, i) => (i === 0 ? 1000000 - (query.candidates.length - 1) * 1000 : 1000)),
  value_ppm: 500000,
  ...change,
});

let handshaken = false;
let inflight = 0;
let peak = 0;
const lines = createInterface({ input: process.stdin });
lines.on("line", (text) => {
  if (!handshaken) {
    handshaken = true;
    if (mode === "silent") return;
    if (mode === "garbage-ready") return void process.stdout.write("this is not json\n");
    if (mode === "no-ready") return void write({ schema_version: 1, message: { kind: "hello", contracts } });
    write(ready);
    return;
  }
  const query = JSON.parse(text) as PolicyQuery;
  const reply = (body: unknown, extra: Record<string, unknown> = {}): void =>
    write({ schema_version: 1, query_id: query.query_id, reply: body, ...extra });
  const ok = (change: Record<string, unknown> = {}): void => reply({ kind: "recommendation", recommendation: recommendation(query, change) });
  if (query.sampling.kind === "evaluate" && (mode.startsWith("eval-") || mode === "ok")) {
    const bad = { ...evaluation(query), prior_ppm: [] };
    const body =
      mode === "eval-short" ? evaluation(query, { prior_ppm: evaluation(query).prior_ppm.slice(1) })
      : mode === "eval-long" ? evaluation(query, { prior_ppm: [...evaluation(query).prior_ppm, 0] })
      : mode === "eval-wrong-identity" ? evaluation(query, { identity: { ...query.identity, request_seq: query.identity.request_seq + 1 } })
      : evaluation(query);
    if (mode === "eval-bad-schema") reply({ kind: "evaluation", evaluation: bad });
    else if (mode === "eval-recommendation") ok();
    else reply({ kind: "evaluation", evaluation: body });
    return;
  }
  switch (mode) {
    case "crash":
      process.exit(3);
      break;
    case "oom":
      process.stderr.write("MemoryError: cannot allocate\n");
      process.exit(1);
      break;
    case "leaky":
      process.stderr.write(`Traceback (most recent call last):\nValueError: bad value ${JSON.stringify(query.observation.team)}\n`);
      process.exit(1);
      break;
    case "kill":
      process.kill(process.pid, "SIGKILL");
      break;
    case "malformed":
      process.stdout.write("{not json\n");
      break;
    case "bad-utf8":
      process.stdout.write(Buffer.from([0xff, 0xfe, 0x0a]));
      break;
    case "spaced":
      process.stdout.write(`${JSON.stringify({ schema_version: 1, query_id: query.query_id, reply: { kind: "error", error: "invalid_query", detail: "x" } }, null, 1).replaceAll("\n", "")}\n`);
      break;
    case "duplicate-key":
      process.stdout.write(`{"query_id":${query.query_id},"query_id":${query.query_id},"reply":{"detail":"x","error":"invalid_query","kind":"error"},"schema_version":1}\n`);
      break;
    case "unknown-id":
      write({ schema_version: 1, query_id: query.query_id + 100, reply: { kind: "error", error: "invalid_query", detail: "x" } });
      break;
    case "long":
      process.stdout.write(`${"x".repeat(MAX_LINE_BYTES + 10)}\n`);
      break;
    case "long-unterminated":
      process.stdout.write("x".repeat(MAX_LINE_BYTES + 10));
      break;
    case "bad-schema":
      reply({ kind: "recommendation" });
      break;
    case "wrong-identity":
      ok({ identity: { ...query.identity, request_seq: query.identity.request_seq + 1 } });
      break;
    case "wrong-choice":
      ok({ choice: "move 9 9" });
      break;
    case "wrong-policy":
      ok({ source_policy_id: "policy:other" });
      break;
    case "unsupported":
      reply({ kind: "error", error: "unsupported_candidate", detail: "a candidate has a shape the worker does not know" });
      break;
    case "invalid-query":
      reply({ kind: "error", error: "invalid_query", detail: "refused" });
      break;
    case "slow":
      setTimeout(() => ok(), 400);
      break;
    case "concurrency":
      peak = Math.max(peak, ++inflight);
      setTimeout(() => {
        ok({ elapsed_ms: peak });
        inflight -= 1;
      }, 50);
      break;
    case "bad-then-ok":
      if (query.query_id === 1) reply({ kind: "recommendation" });
      else ok();
      break;
    default:
      ok();
  }
});
lines.on("close", () => process.exit(0));
