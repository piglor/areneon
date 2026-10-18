// SPIKE (#672, EXP-01 / EXP-08): throwaway end-to-end probe. No human needed.
//
// Boots the pinned Showdown server on 127.0.0.1, loads the unpacked extension
// into Playwright's Chromium, and plays the golden battle (same format, seed,
// synthetic team and engine-default choices as the headless fixtures) with
// p1 in the browser page and p2 as a Node WebSocket client. It records undo,
// reconnect, stale-rqid, spectator, OTS-declined preview and fail-closed
// behaviour, and compares the panel's request_hash sequence with the
// headless path. Never touches play.pokemonshowdown.com.
//
//   node test/e2e.mjs            (after `make showdown` and `pnpm install --ignore-workspace`)
import { spawn } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import http from "node:http";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";
import { chromium } from "playwright-core";
import { loadEngine } from "../../../packages/showdown-bridge/src/engine.ts";
import {
  GOLDEN_FORMAT,
  GOLDEN_SEED,
  goldenBattle,
  syntheticTeam,
} from "../../../packages/showdown-bridge/src/golden.ts";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");
const EXTENSION = join(root, "extension");
const PS_PORT = 18672;
const CONTROL_PORT = 18673;
const PAGE_PORT = 18674;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function until(what, fn, timeout = 15000) {
  const end = Date.now() + timeout;
  for (;;) {
    const value = await fn();
    if (value) return value;
    if (Date.now() > end) throw new Error(`timed out waiting for ${what}`);
    await sleep(50);
  }
}

function startServer() {
  const child = spawn(process.execPath, [join(root, "harness/server.cjs"), String(PS_PORT), String(CONTROL_PORT)], {
    stdio: ["ignore", "pipe", "pipe"],
  });
  let log = "";
  child.stderr.on("data", (d) => (log += d));
  return new Promise((resolve, reject) => {
    child.stdout.on("data", (d) => {
      log += d;
      if (log.includes("READY")) resolve(child);
    });
    child.on("exit", (code) => reject(new Error(`server exited ${code}: ${log}`)));
  });
}

function startPageServer() {
  const html = readFileSync(join(root, "harness/client.html"));
  const server = http.createServer((req, res) => {
    if (!req.url.startsWith("/client.html")) {
      res.statusCode = 404;
      return res.end();
    }
    res.setHeader("content-type", "text/html");
    res.end(html);
  });
  return new Promise((resolve) => server.listen(PAGE_PORT, "127.0.0.1", () => resolve(server)));
}

async function createBattle(players, seed) {
  const res = await fetch(`http://127.0.0.1:${CONTROL_PORT}/battle`, {
    method: "POST",
    body: JSON.stringify({ format: GOLDEN_FORMAT, seed, players, team: syntheticTeam() }),
  });
  const body = await res.json();
  if (!res.ok) throw new Error(body.error);
  return body.roomid;
}

/** A Node Showdown client used for p2 and for the spectated battle's players. */
class NodePlayer {
  constructor(name, { auto = false } = {}) {
    this.name = name;
    this.auto = auto;
    this.requests = new Map();
    this.answered = new Map();
    this.ended = new Set();
    this.named = false;
  }
  connect() {
    this.ws = new WebSocket(`ws://127.0.0.1:${PS_PORT}/showdown/websocket`);
    this.ws.addEventListener("message", (e) => this.onMessage(String(e.data)));
    return new Promise((r) => this.ws.addEventListener("open", r, { once: true })).then(async () => {
      this.ws.send(`|/trn ${this.name},0,`);
      await until(`${this.name} named`, () => this.named);
    });
  }
  onMessage(message) {
    const lines = message.split("\n");
    const roomid = lines[0].startsWith(">") ? lines.shift().slice(1) : "";
    for (const line of lines) {
      if (line.startsWith("|updateuser|") && line.split("|")[2].trim() === this.name) this.named = true;
      if (line.startsWith("|request|") && line.length > 9) {
        const request = JSON.parse(line.slice(9));
        this.requests.set(roomid, request);
        if (this.auto) this.answer(roomid);
      }
      if (line.startsWith("|win|") || line === "|tie") this.ended.add(roomid);
    }
  }
  /** The open request for a room that this player has not answered yet. */
  pending(roomid) {
    const request = this.requests.get(roomid);
    if (!request || request.wait || this.answered.get(roomid) === request.rqid) return null;
    return request;
  }
  answer(roomid) {
    const request = this.pending(roomid);
    if (!request) return false;
    if (request.teamPreview) this.ws.send(`${roomid}|/rejectopenteamsheets`);
    this.answered.set(roomid, request.rqid);
    this.ws.send(`${roomid}|/choose default|${request.rqid}`);
    return true;
  }
  close() {
    this.ws.close();
  }
}

/** p1: the browser page with the extension. */
class PagePlayer {
  constructor(page) {
    this.page = page;
    this.answered = new Map();
  }
  async connect(transport) {
    await this.page.evaluate((t) => window.ps.connect(t), transport);
    await this.page.evaluate(() => window.ps.send("|/trn p1,0,"));
    await until("p1 named", () => this.hasLine(/^\|updateuser\|\s?p1\|1\|/));
  }
  send(text) {
    return this.page.evaluate((t) => window.ps.send(t), text);
  }
  messages() {
    return this.page.evaluate(() => window.ps.messages);
  }
  async hasLine(re) {
    return (await this.messages()).some((m) => m.split("\n").some((l) => re.test(l)));
  }
  /** Latest request this page received for a room (by rqid). */
  async request(roomid) {
    let best = null;
    for (const m of await this.messages()) {
      if (!m.startsWith(`>${roomid}\n`)) continue;
      for (const line of m.split("\n")) {
        if (line.startsWith("|request|") && line.length > 9) {
          const r = JSON.parse(line.slice(9));
          if (!best || r.rqid >= best.rqid) best = r;
        }
      }
    }
    return best;
  }
  async pending(roomid) {
    const r = await this.request(roomid);
    if (!r || r.wait || this.answered.get(roomid) === r.rqid) return null;
    return r;
  }
  async choose(roomid, request) {
    if (request.teamPreview) await this.send(`${roomid}|/rejectopenteamsheets`);
    this.answered.set(roomid, request.rqid);
    await this.send(`${roomid}|/choose default|${request.rqid}`);
  }
  panel() {
    return this.page.evaluate(() => {
      const host = document.querySelector("areneon-672-panel");
      const pre = host?.shadowRoot?.querySelector("#state");
      return pre ? JSON.parse(pre.textContent) : null;
    });
  }
  async room(roomid) {
    return (await this.panel())?.rooms.find((r) => r.roomid === roomid) ?? null;
  }
}

async function headlessP1() {
  const record = await goldenBattle(loadEngine());
  return record.requests.p1.map((r) => ({ request_seq: r.request_seq, phase: r.phase, request_hash: r.request_hash }));
}

async function goldenRun(context, transport, headless) {
  const page = await context.newPage();
  await page.goto(`http://localhost:${PAGE_PORT}/client.html?server=${PS_PORT}`);
  const p1 = new PagePlayer(page);
  await p1.connect(transport);
  const p2 = new NodePlayer("p2");
  await p2.connect();
  const roomid = await createBattle(["p1", "p2"], GOLDEN_SEED);
  const evidence = { transport, roomid };
  let moveRequests = 0;
  let turn1Rqid = null;

  while (!p2.ended.has(roomid)) {
    const mine = await p1.pending(roomid);
    if (mine) {
      if (mine.active && !mine.forceSwitch) moveRequests++;
      if (moveRequests === 1 && !evidence.undo) {
        // Undo: choose, cancel, then check the request identity is unchanged.
        turn1Rqid = mine.rqid;
        await p1.send(`${roomid}|/choose default|${mine.rqid}`);
        await sleep(300);
        const before = (await p1.room(roomid)).current;
        await p1.send(`${roomid}|/undo`);
        await sleep(500);
        const after = (await p1.room(roomid)).current;
        evidence.undo = {
          rqid_before: before.rqid,
          rqid_after: after.rqid,
          request_seq_before: before.request_seq,
          request_seq_after: after.request_seq,
          new_request_after_undo: (await p1.request(roomid)).rqid !== mine.rqid,
          error_lines: (await p1.messages()).flatMap((m) => m.split("\n")).filter((l) => l.startsWith("|error|")),
        };
        assert.deepEqual(after, before, "undo must not change the reconstructed request");
      } else if (moveRequests === 2 && !evidence.reconnect) {
        // Reconnect before choosing: a new socket, same user, same battle.
        const before = (await p1.room(roomid)).current;
        await page.evaluate(() => window.ps.close());
        await sleep(300);
        await page.evaluate(() => (window.ps.archive = [...(window.ps.archive ?? []), ...window.ps.messages.splice(0)]));
        await p1.connect(transport);
        await p1.send(`|/join ${roomid}`);
        await until("request resent", async () => (await p1.request(roomid))?.rqid === mine.rqid);
        await sleep(300);
        const room = await p1.room(roomid);
        evidence.reconnect = {
          rqid_before: before.rqid,
          rqid_after: room.current.rqid,
          request_seq_before: before.request_seq,
          request_seq_after: room.current.request_seq,
          event_after: room.current.event,
          hash_equal: room.current.request_hash === before.request_hash,
        };
        assert.equal(room.current.event, "resent");
        assert.equal(room.current.request_seq, before.request_seq);
      } else if (moveRequests === 3 && !evidence.stale && turn1Rqid !== null) {
        // A choice tagged with an old rqid is refused by the server.
        await p1.send(`${roomid}|/choose default|${turn1Rqid}`);
        const line = await until("stale error", async () =>
          (await p1.messages()).flatMap((m) => m.split("\n")).find((l) => l.includes("too late")),
        );
        evidence.stale = { sent_rqid: turn1Rqid, current_rqid: mine.rqid, server_reply: line };
      }
      await p1.choose(roomid, mine);
      continue;
    }
    if (p2.answer(roomid)) continue;
    await sleep(50);
  }
  await sleep(500);

  const room = await p1.room(roomid);
  const panelNew = room.history
    .filter((o) => o.event === "new")
    .map((o) => ({ request_seq: o.request_seq, phase: o.phase, request_hash: o.request_hash }));
  evidence.panel_p1 = room.history;
  evidence.headless_p1 = headless;
  evidence.exp08_equal = JSON.stringify(panelNew) === JSON.stringify(headless);
  evidence.side_id = room.side_id;
  evidence.status = room.status;

  // OTS-declined preview: what reached p1 before the first turn, other than
  // p1's own request. Both sides bring the same synthetic team, so any move,
  // item or ability name here could only be opponent hidden data.
  const pre = [];
  let started = false;
  // Scan what arrived live (archived at reconnect), not the rejoin replay.
  const live = await page.evaluate(() => [...(window.ps.archive ?? []), ...window.ps.messages]);
  for (const m of live) {
    if (started || !m.startsWith(`>${roomid}\n`)) continue;
    for (const line of m.split("\n").slice(1)) {
      if (line.startsWith("|turn|")) {
        started = true;
        break;
      }
      pre.push(line);
    }
  }
  const hiddenWords = syntheticTeam().flatMap((s) => [s.item, s.ability, ...s.moves]).filter(Boolean);
  const publicLines = pre.filter((l) => !l.startsWith("|request|"));
  evidence.ots = {
    showteam_lines: publicLines.filter((l) => l.startsWith("|showteam|")).length,
    opponent_preview_lines: publicLines.filter((l) => l.startsWith("|poke|p2|")),
    hidden_terms_in_public_lines: hiddenWords.filter((w) => publicLines.some((l) => l.includes(w))),
    public_line_kinds: [...new Set(publicLines.map((l) => l.split("|")[1]))],
    panel_cts: room.cts,
    panel_opponent_preview: room.opponent_preview,
  };
  p2.close();
  await page.close();
  return evidence;
}

async function spectatorRun(context, transport) {
  const page = await context.newPage();
  await page.goto(`http://localhost:${PAGE_PORT}/client.html?server=${PS_PORT}`);
  const p1 = new PagePlayer(page);
  await p1.connect(transport);
  const a = new NodePlayer("p3", { auto: true });
  const b = new NodePlayer("p4", { auto: true });
  await a.connect();
  await b.connect();
  const roomid = await createBattle(["p3", "p4"], GOLDEN_SEED);
  await p1.send(`|/join ${roomid}`);
  await until("spectated battle ends", () => a.ended.has(roomid), 30000);
  await sleep(500);
  const room = await p1.room(roomid);
  const frames = (await p1.messages()).filter((m) => m.startsWith(`>${roomid}\n`)).length;
  a.close();
  b.close();
  await page.close();
  assert.equal(room.role, "spectator");
  assert.equal(room.current, null);
  return { transport, roomid, frames_seen: frames, panel: room };
}

async function failClosedRun(context) {
  const page = await context.newPage();
  await page.goto(`http://localhost:${PAGE_PORT}/client.html?server=${PS_PORT}`);
  const p1 = new PagePlayer(page);
  await until("panel", () => p1.panel());
  const post = (msg) => page.evaluate((m) => window.postMessage(m, location.origin), msg);
  const env = (extra) => ({ ns: "areneon-672", v: 1, ...extra });
  const out = {};
  // 1. Malformed or foreign envelopes are dropped and counted.
  await post(env({ type: "frame", socket: 9, url: "ws://127.0.0.1/x", data: 5 }));
  await post(env({ type: "frame", socket: 9, url: "wss://evil.example/x", data: "|x" }));
  await post(env({ type: "frame", socket: 9, url: "ws://127.0.0.1/x", data: "|x", extra: 1 }));
  await sleep(200);
  out.rejected = (await p1.panel()).rejected;
  // 2. A frame that is not Showdown protocol fails its socket closed.
  await post(env({ type: "frame", socket: 10, url: "ws://127.0.0.1/y", data: "{\"not\":\"showdown\"}" }));
  // 3. A recognised handshake, then an unrecognised request shape: fail closed.
  await post(env({ type: "frame", socket: 11, url: "ws://127.0.0.1/z", data: "|updateuser| p1|1|1|{}" }));
  await post(env({ type: "frame", socket: 11, url: "ws://127.0.0.1/z", data: ">battle-fake-1\n|request|{\"rqid\":1,\"side\":{\"id\":\"p1\",\"name\":\"p1\",\"pokemon\":[]},\"newShape\":true}" }));
  // 4. A request for another user is not taken as the player's own.
  await post(env({ type: "frame", socket: 11, url: "ws://127.0.0.1/z", data: ">battle-fake-2\n|request|{\"rqid\":1,\"side\":{\"id\":\"p2\",\"name\":\"someoneelse\",\"pokemon\":[]},\"wait\":true}" }));
  await sleep(300);
  let panel = await p1.panel();
  out.unframed_socket = panel.sockets.find((s) => s.url === "ws://127.0.0.1/y");
  out.unrecognised_request_room = panel.rooms.find((r) => r.roomid === "battle-fake-1");
  out.other_user_request_room = panel.rooms.find((r) => r.roomid === "battle-fake-2") ?? null;
  out.rejected_after = panel.rejected;
  // 5. A valid request is shown; then an unknown wrapper version withholds everything.
  await post(env({ type: "frame", socket: 11, url: "ws://127.0.0.1/z", data: ">battle-fake-3\n|request|{\"rqid\":1,\"side\":{\"id\":\"p1\",\"name\":\"p1\",\"pokemon\":[]},\"wait\":true}" }));
  await sleep(200);
  out.valid_before_bad_wrapper = (await p1.room("battle-fake-3")).current;
  await post({ ns: "areneon-672", v: 2, type: "hello", wrapper: "b2-websocket" });
  await sleep(200);
  panel = await p1.panel();
  out.client_after_bad_wrapper = panel.client;
  out.valid_after_bad_wrapper = panel.rooms.find((r) => r.roomid === "battle-fake-3").current;
  await page.close();
  assert.equal(out.unframed_socket.status, "unrecognised");
  assert.equal(out.unrecognised_request_room.status, "fail-closed");
  assert.equal(out.unrecognised_request_room.current, null);
  assert.ok(out.valid_before_bad_wrapper);
  assert.equal(out.client_after_bad_wrapper, "unrecognised");
  assert.equal(out.valid_after_bad_wrapper, null);
  return out;
}

const server = await startServer();
const pageServer = await startPageServer();
const userDataDir = mkdtempSync(join(tmpdir(), "areneon-672-"));
let context;
try {
  context = await chromium.launchPersistentContext(userDataDir, {
    channel: "chromium",
    headless: true,
    args: [`--disable-extensions-except=${EXTENSION}`, `--load-extension=${EXTENSION}`],
  });
  const headless = await headlessP1();
  const results = { generated_by: "spikes/672-extension-spike/test/e2e.mjs", golden: [], spectator: [] };
  for (const transport of ["raw", "sockjs"]) {
    const run = await goldenRun(context, transport, headless);
    results.golden.push(run);
    console.log(`${transport}: exp08_equal=${run.exp08_equal} undo=${JSON.stringify(run.undo)} reconnect=${JSON.stringify(run.reconnect)}`);
    results.spectator.push(await spectatorRun(context, transport));
  }
  results.fail_closed = await failClosedRun(context);
  writeFileSync(join(root, "evidence/e2e-result.json"), `${JSON.stringify(results, null, 2)}\n`);
  for (const run of results.golden) {
    assert.ok(run.exp08_equal, `${run.transport}: browser request_hash sequence differs from headless`);
    assert.equal(run.ots.showteam_lines, 0);
    assert.deepEqual(run.ots.hidden_terms_in_public_lines, []);
  }
  console.log("e2e: all assertions passed; evidence/e2e-result.json written");
} finally {
  await context?.close();
  await new Promise((r) => pageServer.close(r));
  server.kill("SIGTERM");
  rmSync(userDataDir, { recursive: true, force: true });
}
