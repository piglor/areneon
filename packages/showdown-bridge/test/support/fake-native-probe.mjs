#!/usr/bin/env node
// A stand-in for the `ninjask-probe` binary: the same arguments (`--dex PATH --in PATH --out PATH`), the same frames
// (`[u32 little-endian length][JSON]`), canned behaviour chosen by the environment (FAKE_MODE), so the transport and the backend are
// tested without the vendored engine. It holds no game content: `stats` answers a fixed table, every other query abstains.
import { closeSync, openSync, readSync, writeSync } from "node:fs";
import { Buffer } from "node:buffer";
import process from "node:process";
import { setTimeout } from "node:timers";

const arg = (name) => process.argv[process.argv.indexOf(name) + 1];
const mode = process.env.FAKE_MODE ?? "plain";
if (mode === "no-start") process.exit(3);
const input = openSync(arg("--in"), "r");
const output = openSync(arg("--out"), "w");
const readExactly = (n) => {
  const buffer = Buffer.alloc(n);
  let at = 0;
  while (at < n) {
    const got = readSync(input, buffer, at, n - at, null);
    if (got === 0) return null;
    at += got;
  }
  return buffer;
};
const send = (value, raw) => {
  const body = raw ?? Buffer.from(JSON.stringify(value));
  const head = Buffer.alloc(4);
  head.writeUInt32LE(body.length, 0);
  writeSync(output, head);
  writeSync(output, body);
};
if (mode === "deaf") await new Promise((resolve) => setTimeout(resolve, 60_000));
if (mode === "closed-input") {
  closeSync(input);
  await new Promise((resolve) => setTimeout(resolve, 60_000));
}
let batches = 0;
for (;;) {
  const head = readExactly(4);
  if (head === null) break;
  const frame = JSON.parse(readExactly(head.readUInt32LE(0)).toString("utf8"));
  if (frame.op === "blockers") send({ ok: true });
  else if (frame.op === "sets") send({ failed: mode === "refuse-sets" ? frame.sets.map(([id]) => [id, "unknown species"]) : [] });
  else if (frame.op === "batch") {
    batches += 1;
    if (mode === "die" && batches === 1) process.exit(4);
    if (mode === "garbage") send(null, Buffer.from("not json"));
    else if (mode === "error") send({ error: "the frame is not a request" });
    else if (mode === "short") send({ results: [] });
    else if (mode === "silent") await new Promise((resolve) => setTimeout(resolve, 60_000));
    else if (mode === "huge") send({ results: frame.queries.map(() => ({ kind: "abstain", entity: "move:" + "x".repeat(70_000), why: "big" })) });
    else
      send({
        results: frame.queries.map((q) => {
          if (q.kind === "stats") return { kind: "computed", answer: { kind: "stats", stats: { hp: 1, atk: 2, def: 3, spa: 4, spd: 5, spe: 6 } } };
          if (mode === "unsupported") return { kind: "unsupported", reason: "query_kind", position: null };
          return { kind: "abstain", entity: "move:fake", why: "the stand-in answers stats only" };
        }),
      });
  } else send({ error: "unknown op" });
}
closeSync(input);
closeSync(output);
