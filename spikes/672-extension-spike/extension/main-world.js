// SPIKE (#672, ADR-004 option B2): throwaway. Runs in the page's MAIN world.
//
// It only observes: it wraps the WebSocket constructor so every socket the
// page opens gets one extra `message` listener, and forwards each text frame
// to the isolated content script with window.postMessage. It never reads or
// wraps `send`, so outgoing login assertions and team uploads never pass
// through it, and it receives nothing back from the extension. SockJS uses a
// native WebSocket underneath, so its frames arrive here too (still framed).
(() => {
  "use strict";
  const NS = "areneon-672";
  const Native = window.WebSocket;
  if (typeof Native !== "function") return;
  let nextSocket = 0;
  const post = (message) => window.postMessage({ ns: NS, v: 1, ...message }, window.location.origin);

  class ObservedWebSocket extends Native {
    constructor(url, protocols) {
      super(url, protocols);
      const socket = ++nextSocket;
      const href = String(this.url);
      post({ type: "open", socket, url: href });
      this.addEventListener("message", (event) => {
        if (typeof event.data === "string") post({ type: "frame", socket, url: href, data: event.data });
      });
      this.addEventListener("close", () => post({ type: "close", socket, url: href }));
    }
  }

  window.WebSocket = ObservedWebSocket;
  post({ type: "hello", wrapper: "b2-websocket" });
})();
