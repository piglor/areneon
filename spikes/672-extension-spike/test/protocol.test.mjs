// SPIKE (#672): unit checks for the isolated-world protocol logic.
//   node --test test/protocol.test.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import "../extension/protocol.js";
import { canonicalJson as bridgeCanonical } from "../../../packages/showdown-bridge/src/request-identity.ts";

const P = globalThis.AreneonProtocol;
const GOLDEN = new URL("../../../fixtures/showdown/gen9championsvgc2026regmc/", import.meta.url);
const env = (extra) => ({ ns: P.NS, v: 1, ...extra });
const frame = (socket, data) => env({ type: "frame", socket, url: "ws://127.0.0.1:1/s", data });

test("request_hash matches the committed headless golden fixtures", async () => {
  for (const file of ["preview", "turn", "force-switch"]) {
    const fixture = JSON.parse(readFileSync(new URL(`${file}.request.json`, GOLDEN), "utf8"));
    assert.equal(P.canonicalJson(fixture.request), bridgeCanonical(fixture.request));
    assert.equal(await P.requestHash({ ...fixture.request, rqid: 99 }), fixture.request_hash);
  }
});

test("envelope validation rejects forged or foreign shapes", () => {
  assert.equal(P.validateEnvelope(null), "not-ours");
  assert.equal(P.validateEnvelope(env({ v: 2, type: "hello", wrapper: P.WRAPPER })), "bad-version");
  assert.equal(P.validateEnvelope(env({ type: "hello", wrapper: "other" })), "unknown-wrapper");
  assert.equal(P.validateEnvelope(env({ type: "nope" })), "bad-type");
  assert.equal(P.validateEnvelope({ ...frame(1, "|x"), extra: 1 }), "bad-shape");
  assert.equal(P.validateEnvelope(frame(0, "|x")), "bad-socket");
  assert.equal(P.validateEnvelope(frame(1, 7)), "bad-frame");
  assert.equal(P.validateEnvelope(env({ type: "open", socket: 1, url: "https://127.0.0.1/" })), "bad-url");
  assert.equal(P.validateEnvelope(env({ type: "open", socket: 1, url: "wss://evil.example/" })), "foreign-host");
  assert.equal(P.validateEnvelope(env({ type: "open", socket: 1, url: "wss://sim3.psim.us/showdown/websocket" })).type, "open");
});

test("SockJS and raw framing", () => {
  assert.deepEqual(P.unframe("o"), { transport: "sockjs", messages: [] });
  assert.deepEqual(P.unframe('a[">battle-x\\n|turn|1"]'), { transport: "sockjs", messages: [">battle-x\n|turn|1"] });
  assert.deepEqual(P.unframe("|challstr|x"), { transport: "raw", messages: ["|challstr|x"] });
  assert.equal(P.unframe("a[1]"), null);
  assert.equal(P.unframe("{}"), null);
});

test("tracker: new, resent, stale and conflicting requests", async () => {
  const t = new P.Tracker();
  await t.accept(env({ type: "hello", wrapper: P.WRAPPER }));
  await t.accept(frame(1, "|updateuser| Alice|1|1|{}"));
  const req = (rqid, extra) =>
    `>battle-a-1\n|request|${JSON.stringify({ rqid, side: { id: "p1", name: "Alice", pokemon: [] }, wait: true, ...extra })}`;
  await t.accept(frame(1, req(3)));
  await t.accept(frame(1, req(3)));
  await t.accept(frame(1, req(1)));
  let room = t.snapshot().rooms[0];
  assert.deepEqual(room.history.map((o) => [o.request_seq, o.rqid, o.event]), [[1, 3, "new"], [1, 3, "resent"]]);
  assert.equal(t.snapshot().rejected["stale-rqid"], 1);
  await t.accept(frame(1, req(3, { x: 1 })));
  room = t.snapshot().rooms[0];
  assert.equal(room.status, "fail-closed");
  assert.equal(room.current, null);
});

test("tracker: a request without rqid or before the handshake fails closed", async () => {
  const t = new P.Tracker();
  await t.accept(env({ type: "hello", wrapper: P.WRAPPER }));
  const body = JSON.stringify({ side: { id: "p1", name: "a", pokemon: [] }, wait: true });
  await t.accept(frame(1, `>battle-b-1\n|request|${body}`));
  await t.accept(frame(2, "|updateuser| a|1|1|{}"));
  await t.accept(frame(2, `>battle-c-1\n|request|${body}`));
  const rooms = Object.fromEntries(t.snapshot().rooms.map((r) => [r.roomid, r]));
  assert.equal(rooms["battle-b-1"].why, "request before handshake");
  assert.equal(rooms["battle-c-1"].why, "server request has no rqid");
});
