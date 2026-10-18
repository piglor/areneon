// SPIKE (#672, ADR-004 option B2): throwaway. Runs in the page's MAIN world.
//
// It only observes. It subclasses WebSocket so every socket the page opens
// gets one extra `message` listener, unframes raw or SockJS frames, keeps
// only the whitelisted protocol lines the panel needs (never `|challstr|` or
// any other login line), and sends them to the isolated content script over
// a private MessageChannel port, not the shared window.postMessage bus. It
// never wraps `send`, and nothing flows back into this world.
//
// The Node unit tests load this same file: outside a browser it only
// exposes the pure filter functions.
(() => {
  "use strict";
  const NS = "areneon-672";

  /** Battle-room line kinds forwarded, and how many `|` fields survive. */
  const BATTLE_KEEP = {
    init: 3, // |init|battle
    player: 4, // |player|p1|NAME (avatar and rating dropped)
    poke: 4, // |poke|p2|DETAILS (public preview details only)
    showteam: 3, // |showteam|p2 (the packed team itself is dropped)
    request: Infinity, // own request JSON: needed for request_hash and the advisor
    teampreview: 2,
    turn: 3,
    win: 3,
    tie: 2,
  };

  /** Keep the allowed prefix of one line, or null to drop it. */
  function keepLine(roomid, line) {
    const parts = line.split("|");
    if (parts[0] !== "") return null;
    if (roomid.startsWith("battle-")) {
      if (!Object.hasOwn(BATTLE_KEEP, parts[1])) return null;
      const width = BATTLE_KEEP[parts[1]];
      return width === Infinity ? line : parts.slice(0, width).join("|");
    }
    // Outside battle rooms: only the display name, for the own-player check.
    if (roomid === "" && parts[1] === "updateuser") return parts.slice(0, 3).join("|");
    return null;
  }

  /**
   * Unframe one WebSocket frame and filter it. Returns null for a frame that
   * is neither raw Showdown nor SockJS framing, so the isolated world fails
   * that socket closed.
   */
  function filterFrame(frame) {
    let transport;
    let messages;
    if (frame === "o" || frame === "h" || frame.startsWith("c[")) {
      transport = "sockjs";
      messages = [];
    } else if (frame.startsWith("a[")) {
      transport = "sockjs";
      try {
        messages = JSON.parse(frame.slice(1));
      } catch {
        return null;
      }
      if (!Array.isArray(messages) || !messages.every((m) => typeof m === "string")) return null;
    } else if (frame.startsWith("|") || frame.startsWith(">")) {
      transport = "raw";
      messages = [frame];
    } else {
      return null;
    }
    const out = [];
    for (const message of messages) {
      const lines = message.split("\n");
      const roomid = lines[0].startsWith(">") ? lines.shift().slice(1).trim() : "";
      const kept = lines.map((line) => keepLine(roomid, line)).filter((line) => line !== null);
      if (kept.length) out.push({ roomid, lines: kept });
    }
    return { transport, messages: out };
  }

  if (typeof window === "undefined") {
    globalThis.AreneonFrameFilter = { filterFrame, keepLine };
    return;
  }

  // Capture the primitives now, at document_start, so page script that runs
  // later cannot read the port traffic by patching prototypes.
  const apply = Reflect.apply;
  const portPost = MessagePort.prototype.postMessage;
  const portStart = MessagePort.prototype.start;
  const winPost = window.postMessage;
  const listen = EventTarget.prototype.addEventListener;
  const Channel = window.MessageChannel;
  const Native = window.WebSocket;
  if (typeof Native !== "function") return;

  let port = null;
  const backlog = [];
  const send = (message) => {
    const envelope = { ns: NS, v: 1, ...message };
    if (port) apply(portPost, port, [envelope]);
    else if (backlog.length < 1000) backlog.push(envelope);
  };

  // Hand one port to the isolated world. Only the first acknowledged port is
  // used and no port is offered after that, so a page script that later
  // posts "ready" gets nothing.
  function offer() {
    if (port) return;
    const channel = new Channel();
    const mine = channel.port1;
    apply(listen, mine, [
      "message",
      (event) => {
        if (port || event.data !== "ack") return;
        port = mine;
        for (const envelope of backlog.splice(0)) apply(portPost, port, [envelope]);
      },
    ]);
    apply(portStart, mine, []);
    apply(winPost, window, [{ ns: NS, v: 1, type: "port" }, window.location.origin, [channel.port2]]);
  }
  apply(listen, window, [
    "message",
    (event) => {
      const data = event.data;
      if (!port && event.source === window && data && data.ns === NS && data.type === "ready") offer();
    },
  ]);

  let nextSocket = 0;
  class ObservedWebSocket extends Native {
    constructor(url, protocols) {
      super(url, protocols);
      const socket = ++nextSocket;
      const href = String(this.url);
      send({ type: "open", socket, url: href });
      apply(listen, this, [
        "message",
        (event) => {
          if (typeof event.data !== "string") return;
          const filtered = filterFrame(event.data);
          if (!filtered) return send({ type: "unrecognised", socket, url: href });
          for (const { roomid, lines } of filtered.messages) {
            send({ type: "lines", socket, url: href, transport: filtered.transport, roomid, lines });
          }
        },
      ]);
      apply(listen, this, ["close", () => send({ type: "close", socket, url: href })]);
    }
  }

  window.WebSocket = ObservedWebSocket;
  send({ type: "hello", wrapper: "b2-websocket" });
  offer();
})();
