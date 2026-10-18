// SPIKE (#672): unit checks for the MAIN-world frame filter and the
// isolated-world protocol logic.
//   node --test test/protocol.test.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import "../extension/protocol.js";
import "../extension/main-world.js";
import { canonicalJson as bridgeCanonical } from "../../../packages/showdown-bridge/src/request-identity.ts";

const P = globalThis.AreneonProtocol;
const F = globalThis.AreneonFrameFilter;
const GOLDEN = new URL("../../../fixtures/showdown/gen9championsvgc2026regmc/", import.meta.url);
const env = (extra) => ({ ns: P.NS, v: 1, ...extra });
const lines = (socket, roomid, ls) =>
  env({ type: "lines", socket, url: "ws://127.0.0.1:1/s", transport: "raw", roomid, lines: ls });

test("request_hash matches the committed headless golden fixtures", async () => {
  for (const file of ["preview", "turn", "force-switch"]) {
    const fixture = JSON.parse(readFileSync(new URL(`${file}.request.json`, GOLDEN), "utf8"));
    assert.equal(P.canonicalJson(fixture.request), bridgeCanonical(fixture.request));
    assert.equal(await P.requestHash({ ...fixture.request, rqid: 99 }), fixture.request_hash);
  }
});

test("the wrapper never forwards |challstr| or other login lines", () => {
  const challstr = "|challstr|4|0123456789abcdef";
  const frames = [
    challstr,
    `a${JSON.stringify([challstr])}`,
    `a${JSON.stringify([`|updateuser| Guest 1|0|170|{"blockChallenges":false}\n${challstr}`])}`,
    "|nametaken|p1|Your authentication token was invalid.",
    "|queryresponse|userdetails|{}",
    `>battle-x-1\n${challstr}`,
  ];
  for (const frame of frames) {
    const out = F.filterFrame(frame);
    const forwarded = out.messages.flatMap((m) => m.lines).join("\n");
    assert.ok(!forwarded.includes("challstr"), frame);
    assert.ok(!forwarded.includes("0123456789abcdef"), frame);
    assert.ok(!forwarded.includes("nametaken") && !forwarded.includes("blockChallenges"), frame);
  }
  // The display name survives, its settings JSON does not.
  assert.deepEqual(F.filterFrame(frames[2]).messages, [{ roomid: "", lines: ["|updateuser| Guest 1"] }]);
});

test("the wrapper keeps only whitelisted battle lines, trimmed", () => {
  const frame = [
    ">battle-x-1",
    "|init|battle",
    "|player|p2|Bob|avatar|1500",
    "|poke|p2|Gengar, L50, M|",
    "|showteam|p2|Gengar||Leftovers|Levitate|Shadow Ball|||||50|",
    '|request|{"rqid":1}',
    "|c|Bob|hi",
    "|switch|p2a: Gengar|Gengar, L50, M|100/100",
    "|turn|1",
  ].join("\n");
  const out = F.filterFrame(frame);
  assert.equal(out.transport, "raw");
  assert.deepEqual(out.messages[0].lines, [
    "|init|battle",
    "|player|p2|Bob",
    "|poke|p2|Gengar, L50, M",
    "|showteam|p2",
    '|request|{"rqid":1}',
    "|turn|1",
  ]);
  for (const line of out.messages[0].lines) assert.ok(P.allowedLine("battle-x-1", line), line);
  assert.equal(F.filterFrame("o").transport, "sockjs");
  assert.equal(F.filterFrame("a[1]"), null);
  assert.equal(F.filterFrame("{}"), null);
});

test("envelope validation rejects forged, foreign or non-whitelisted input", () => {
  assert.equal(P.validateEnvelope(null), "not-ours");
  assert.equal(P.validateEnvelope(env({ v: 2, type: "hello", wrapper: P.WRAPPER })), "bad-version");
  assert.equal(P.validateEnvelope(env({ type: "hello", wrapper: "other" })), "unknown-wrapper");
  assert.equal(P.validateEnvelope(env({ type: "frame" })), "bad-type");
  assert.equal(P.validateEnvelope({ ...lines(1, "", ["|updateuser| a"]), extra: 1 }), "bad-shape");
  assert.equal(P.validateEnvelope(lines(0, "", ["|updateuser| a"])), "bad-socket");
  assert.equal(P.validateEnvelope(lines(1, "", ["|challstr|4|abc"])), "line-not-allowed");
  assert.equal(P.validateEnvelope(lines(1, "battle-x-1", ["|showteam|p2|Gengar||Leftovers"])), "line-not-allowed");
  assert.equal(P.validateEnvelope(lines(1, "battle-x-1", [])), "bad-lines");
  assert.equal(P.validateEnvelope(env({ type: "open", socket: 1, url: "https://127.0.0.1/" })), "bad-url");
  assert.equal(P.validateEnvelope(env({ type: "open", socket: 1, url: "wss://evil.example/" })), "foreign-host");
  assert.equal(P.validateEnvelope(env({ type: "open", socket: 1, url: "wss://sim3.psim.us/showdown/websocket" })).type, "open");
});

async function tracker(user = "Alice") {
  const t = new P.Tracker();
  await t.accept(env({ type: "hello", wrapper: P.WRAPPER }));
  await t.accept(lines(1, "", [`|updateuser| ${user}`]));
  return t;
}
const req = (rqid, side = {}, extra = {}) =>
  `|request|${JSON.stringify({ rqid, side: { id: "p1", name: "Alice", pokemon: [], ...side }, wait: true, ...extra })}`;
const roomOf = (t, roomid) => t.snapshot().rooms.find((r) => r.roomid === roomid);

test("tracker: new, resent, stale and conflicting requests", async () => {
  const t = await tracker();
  for (const rqid of [3, 3, 1]) await t.accept(lines(1, "battle-a-1", [req(rqid)]));
  let room = roomOf(t, "battle-a-1");
  assert.deepEqual(room.history.map((o) => [o.request_seq, o.rqid, o.event]), [[1, 3, "new"], [1, 3, "resent"]]);
  assert.equal(t.snapshot().rejected["stale-rqid"], 1);
  await t.accept(lines(1, "battle-a-1", [req(3, {}, { x: 1 })]));
  room = roomOf(t, "battle-a-1");
  assert.equal(room.status, "fail-closed");
  assert.equal(room.current, null);
});

test("tracker: seat mismatch fails the room closed", async () => {
  const t = await tracker();
  await t.accept(lines(1, "battle-s-1", ["|init|battle", "|player|p1|Bob", "|player|p2|Alice"]));
  await t.accept(lines(1, "battle-s-1", [req(2)]));
  const room = roomOf(t, "battle-s-1");
  assert.equal(room.status, "fail-closed");
  assert.equal(room.why, "request side seat mismatch");
  assert.equal(room.current, null);
});

test("tracker: spectator, other user, no rqid, bad shape, no handshake, unrecognised socket", async () => {
  const t = await tracker();
  await t.accept(lines(1, "battle-v-1", ["|init|battle", "|player|p1|Bob", "|player|p2|Carol"]));
  await t.accept(lines(1, "battle-o-1", [req(1, { name: "Bob" })]));
  await t.accept(lines(1, "battle-n-1", [`|request|${JSON.stringify({ side: { id: "p1", name: "Alice", pokemon: [] }, wait: true })}`]));
  await t.accept(lines(1, "battle-b-1", [req(1, {}, { wait: false, newShape: true })]));
  await t.accept(env({ type: "open", socket: 2, url: "ws://localhost:1/other" }));
  await t.accept(env({ type: "lines", socket: 2, url: "ws://localhost:1/other", transport: "raw", roomid: "battle-h-1", lines: [req(1)] }));
  await t.accept(env({ type: "unrecognised", socket: 1, url: "ws://127.0.0.1:1/s" }));
  assert.equal(roomOf(t, "battle-v-1").status, "rejected: spectator");
  assert.equal(t.snapshot().rejected["request-not-for-current-user"], 1);
  assert.equal(roomOf(t, "battle-n-1").why, "socket: unrecognised frame");
  assert.equal(roomOf(t, "battle-b-1").status, "fail-closed");
  assert.equal(roomOf(t, "battle-h-1").why, "request before handshake");
  assert.equal(t.snapshot().sockets[0].status, "unrecognised");
});

test("tracker: missing rqid and unknown phase fail closed", async () => {
  const t = await tracker();
  await t.accept(lines(1, "battle-n-1", [`|request|${JSON.stringify({ side: { id: "p1", name: "Alice", pokemon: [] }, wait: true })}`]));
  await t.accept(lines(1, "battle-b-1", [req(1, {}, { wait: false, newShape: true })]));
  assert.equal(roomOf(t, "battle-n-1").why, "server request has no rqid");
  assert.equal(roomOf(t, "battle-b-1").why, "unrecognised request phase");
});

test("tracker: an unknown wrapper version withholds every observation", async () => {
  const t = await tracker();
  await t.accept(lines(1, "battle-w-1", [req(1)]));
  assert.ok(roomOf(t, "battle-w-1").current);
  await t.accept(env({ v: 2, type: "hello", wrapper: P.WRAPPER }));
  assert.equal(t.snapshot().client, "unrecognised");
  assert.equal(roomOf(t, "battle-w-1").current, null);
});
