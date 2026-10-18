// SPIKE (#672, ADR-004 EXP-01): throwaway. Written independently from the
// Showdown protocol documentation (vendor/pokemon-showdown/PROTOCOL.md); no
// Showdown client code is copied.
//
// Pure, DOM-free logic for the ISOLATED world: envelope validation, SockJS
// framing, Showdown message parsing and own-request tracking. Loaded as a
// classic content script (it shares the isolated world with isolated.js) and
// also imported by the Node tests, so it only assigns one global.
"use strict";

(() => {
  const NS = "areneon-672";
  const ENVELOPE_VERSION = 1;
  const WRAPPER = "b2-websocket";
  const MAX_FRAME = 1 << 20;
  /** WebSocket hosts the extension trusts: the main Showdown server and localhost. */
  const SOCKET_HOSTS = new Set(["sim3.psim.us", "localhost", "127.0.0.1"]);

  /** Same algorithm as packages/showdown-bridge/src/request-identity.ts. */
  function canonicalJson(value) {
    if (typeof value === "number" && !Number.isFinite(value)) {
      throw new RangeError("canonicalJson: non-finite number");
    }
    if (value === null || typeof value !== "object") return JSON.stringify(value);
    if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
    const keys = Object.keys(value).sort();
    return `{${keys.map((k) => `${JSON.stringify(k)}:${canonicalJson(value[k])}`).join(",")}}`;
  }

  /** SHA-256 of the canonical request JSON without the server-only `rqid`. */
  async function requestHash(request) {
    const { rqid: _rqid, ...rest } = request;
    const bytes = new TextEncoder().encode(canonicalJson(rest));
    const digest = await crypto.subtle.digest("SHA-256", bytes);
    return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
  }

  function toId(text) {
    return String(text).toLowerCase().replace(/[^a-z0-9]+/g, "");
  }

  function isPlainObject(value) {
    return typeof value === "object" && value !== null && !Array.isArray(value);
  }

  /**
   * Validate a MAIN-world message. Anything in the page can post one, so the
   * shape is checked exactly and the socket URL must be an allowed host.
   * Returns the envelope or a string reason for rejection.
   */
  function validateEnvelope(data) {
    if (!isPlainObject(data) || data.ns !== NS) return "not-ours";
    if (data.v !== ENVELOPE_VERSION) return "bad-version";
    const keys = Object.keys(data).sort().join(",");
    switch (data.type) {
      case "hello":
        if (keys !== "ns,type,v,wrapper") return "bad-shape";
        return data.wrapper === WRAPPER ? data : "unknown-wrapper";
      case "open":
      case "close":
        if (keys !== "ns,socket,type,url,v") return "bad-shape";
        break;
      case "frame":
        if (keys !== "data,ns,socket,type,url,v") return "bad-shape";
        if (typeof data.data !== "string" || data.data.length > MAX_FRAME) return "bad-frame";
        break;
      default:
        return "bad-type";
    }
    if (!Number.isSafeInteger(data.socket) || data.socket < 1) return "bad-socket";
    if (typeof data.url !== "string") return "bad-url";
    let url;
    try {
      url = new URL(data.url);
    } catch {
      return "bad-url";
    }
    if (url.protocol !== "ws:" && url.protocol !== "wss:") return "bad-url";
    if (!SOCKET_HOSTS.has(url.hostname)) return "foreign-host";
    return data;
  }

  /**
   * Split one WebSocket frame into Showdown messages. Showdown speaks either
   * raw WebSocket (one message per frame) or SockJS framing over WebSocket
   * (`o` open, `h` heartbeat, `a[...]` messages, `c[...]` close). Returns
   * null for a frame that is neither: the caller fails closed.
   */
  function unframe(frame) {
    if (frame === "o" || frame === "h") return { transport: "sockjs", messages: [] };
    if (frame.startsWith("c[")) return { transport: "sockjs", messages: [] };
    if (frame.startsWith("a[")) {
      let parsed;
      try {
        parsed = JSON.parse(frame.slice(1));
      } catch {
        return null;
      }
      if (!Array.isArray(parsed) || !parsed.every((m) => typeof m === "string")) return null;
      return { transport: "sockjs", messages: parsed };
    }
    if (frame.startsWith("|") || frame.startsWith(">")) {
      return { transport: "raw", messages: [frame] };
    }
    return null;
  }

  /** Split a Showdown message into its room id and protocol lines. */
  function splitMessage(message) {
    const lines = message.split("\n");
    let roomid = "";
    if (lines[0].startsWith(">")) roomid = lines.shift().slice(1).trim();
    return { roomid, lines };
  }

  /** Same phase rules as packages/showdown-bridge/src/request.ts. */
  function phaseOf(request) {
    if (request.teamPreview === true) return "preview";
    if (request.wait === true) return "wait";
    if (Array.isArray(request.forceSwitch)) return "forceSwitch";
    if (Array.isArray(request.active)) return "move";
    return null;
  }

  /**
   * Tracks sockets and battle rooms and reconstructs the player's own
   * request per room. It emits only identity and phase, never the raw
   * request, and never anything about the opponent beyond public preview
   * species details.
   */
  class Tracker {
    constructor() {
      this.client = "waiting-for-wrapper";
      this.rejected = {};
      this.sockets = new Map();
      this.rooms = new Map();
    }

    reject(reason) {
      this.rejected[reason] = (this.rejected[reason] ?? 0) + 1;
    }

    /** Feed one untrusted MAIN-world message. */
    async accept(raw) {
      const envelope = validateEnvelope(raw);
      if (typeof envelope === "string") {
        if (envelope === "unknown-wrapper" || envelope === "bad-version") {
          this.client = "unrecognised";
        }
        if (envelope !== "not-ours") this.reject(envelope);
        return;
      }
      if (envelope.type === "hello") {
        if (this.client === "waiting-for-wrapper") this.client = "ok";
        return;
      }
      let socket = this.sockets.get(envelope.socket);
      if (!socket) {
        socket = { url: envelope.url, status: "unverified", transport: null, userid: null };
        this.sockets.set(envelope.socket, socket);
      }
      if (socket.url !== envelope.url) {
        this.reject("socket-url-changed");
        return;
      }
      if (envelope.type === "close") {
        socket.status = "closed";
        return;
      }
      if (envelope.type !== "frame") return;
      if (socket.status === "unrecognised") return;
      const unframed = unframe(envelope.data);
      if (!unframed || (socket.transport && socket.transport !== unframed.transport)) {
        this.failSocket(socket, "unrecognised frame");
        return;
      }
      socket.transport = unframed.transport;
      for (const message of unframed.messages) await this.message(socket, message);
    }

    failSocket(socket, why) {
      socket.status = "unrecognised";
      socket.why = why;
      for (const room of this.rooms.values()) {
        if (room.socketUrl === socket.url) this.failRoom(room, `socket: ${why}`);
      }
    }

    failRoom(room, why) {
      room.status = "fail-closed";
      room.why = why;
      room.current = null;
    }

    room(roomid, socket) {
      let room = this.rooms.get(roomid);
      if (!room) {
        room = {
          roomid,
          socketUrl: socket.url,
          role: "unknown",
          status: "observing",
          players: {},
          sideId: null,
          current: null,
          history: [],
          opponentPreview: [],
          showteam: [],
        };
        this.rooms.set(roomid, room);
      }
      room.socketUrl = socket.url;
      return room;
    }

    async message(socket, message) {
      const { roomid, lines } = splitMessage(message);
      for (const line of lines) {
        if (line.startsWith("|challstr|")) {
          // A login nonce: proves the Showdown handshake, then dropped.
          if (socket.status === "unverified") socket.status = "showdown";
        } else if (line.startsWith("|updateuser|")) {
          if (socket.status === "unverified") socket.status = "showdown";
          socket.userid = toId(line.split("|")[2] ?? "");
        } else if (roomid.startsWith("battle-")) {
          await this.battleLine(socket, this.room(roomid, socket), line);
        }
      }
    }

    async battleLine(socket, room, line) {
      const parts = line.split("|");
      const cmd = parts[1];
      if (cmd === "init") {
        // A (re)join replays the whole public log; request identity survives.
        if (room.role === "unknown") room.role = "spectator";
        room.opponentPreview = [];
        room.showteam = [];
      } else if (cmd === "player" && parts[3]) {
        room.players[parts[2]] = toId(parts[3]);
        if (room.players[parts[2]] === socket.userid) {
          room.role = "player";
          room.sideId = parts[2];
        }
      } else if (cmd === "poke") {
        // Public team preview details only: species, level, gender.
        if (room.sideId && parts[2] !== room.sideId) room.opponentPreview.push(parts[3]);
      } else if (cmd === "showteam") {
        room.showteam.push(parts[2]);
      } else if (cmd === "request") {
        await this.request(socket, room, line.slice("|request|".length));
      }
    }

    async request(socket, room, body) {
      if (room.status === "fail-closed") return;
      if (body === "") return; // Showdown clears a request with an empty body.
      if (socket.status !== "showdown") return this.failRoom(room, "request before handshake");
      let request;
      try {
        request = JSON.parse(body);
      } catch {
        return this.failRoom(room, "request is not JSON");
      }
      const side = isPlainObject(request) ? request.side : null;
      if (!isPlainObject(side) || !/^p[1-4]$/.test(side.id) || !Array.isArray(side.pokemon)) {
        return this.failRoom(room, "unrecognised request shape");
      }
      const phase = phaseOf(request);
      if (!phase) return this.failRoom(room, "unrecognised request phase");
      if (!Number.isSafeInteger(request.rqid)) return this.failRoom(room, "server request has no rqid");
      if (toId(side.name) !== socket.userid) {
        this.reject("request-not-for-current-user");
        return;
      }
      const seated = room.players[side.id];
      if (seated && seated !== socket.userid) return this.failRoom(room, "request side seat mismatch");
      room.role = "player";
      room.sideId = side.id;
      const hash = await requestHash(request);
      const last = room.current ?? room.history.at(-1);
      let event;
      let seq;
      if (last && request.rqid === last.rqid) {
        if (hash !== last.request_hash) return this.failRoom(room, "same rqid, different request");
        event = "resent";
        seq = last.request_seq;
      } else if (last && request.rqid < last.rqid) {
        this.reject("stale-rqid");
        return;
      } else {
        event = "new";
        seq = (last?.request_seq ?? 0) + 1;
      }
      const observation = {
        side_id: side.id,
        request_seq: seq,
        rqid: request.rqid,
        phase,
        request_hash: hash,
        event,
      };
      room.current = observation;
      room.history.push(observation);
    }

    /** Everything the panel shows. Withholds observations unless the client is recognised. */
    snapshot() {
      const ok = this.client === "ok";
      return {
        client: this.client,
        rejected: { ...this.rejected },
        sockets: [...this.sockets.values()].map((s) => ({ ...s })),
        rooms: [...this.rooms.values()].map((r) => ({
          roomid: r.roomid,
          role: r.role,
          status: r.role === "spectator" ? "rejected: spectator" : r.status,
          why: r.why,
          side_id: r.sideId,
          current: ok && r.role === "player" ? r.current : null,
          history: ok && r.role === "player" ? r.history : [],
          cts: r.showteam.length === 0 ? "no showteam seen" : `showteam seen: ${r.showteam.join(",")}`,
          opponent_preview: r.role === "player" ? r.opponentPreview : [],
        })),
      };
    }
  }

  globalThis.AreneonProtocol = {
    NS,
    WRAPPER,
    canonicalJson,
    requestHash,
    validateEnvelope,
    unframe,
    splitMessage,
    phaseOf,
    Tracker,
  };
})();
