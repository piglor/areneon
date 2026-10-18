// SPIKE (#672): throwaway. Runs in the extension's ISOLATED world.
//
// Receives the MAIN-world wrapper's filtered lines over a private
// MessageChannel port. The port arrives once over window.postMessage; this
// capture listener is registered at document_start, before any page script,
// and stops that message from reaching page listeners. Every envelope is
// still untrusted: AreneonProtocol validates its schema, line whitelist and
// request shape before anything is shown. There is no submit path: nothing
// here writes to the page's socket or posts data back into the MAIN world.
(() => {
  "use strict";
  const { NS, Tracker } = globalThis.AreneonProtocol;
  const tracker = new Tracker();
  let queue = Promise.resolve();
  let pre = null;
  let port = null;

  function render() {
    if (!pre) {
      if (!document.documentElement) return;
      const host = document.createElement("areneon-672-panel");
      host.style.cssText = "position:fixed;right:8px;bottom:8px;z-index:2147483647;max-width:46ch;max-height:40vh;overflow:auto;";
      const shadow = host.attachShadow({ mode: "open" });
      const title = document.createElement("strong");
      title.textContent = "Areneon EXP-01 probe (read-only spike)";
      pre = document.createElement("pre");
      pre.id = "state";
      pre.style.cssText = "margin:0;font:11px/1.3 monospace;white-space:pre-wrap;";
      const box = document.createElement("div");
      box.style.cssText = "background:#fff;color:#111;border:1px solid #888;padding:6px;";
      box.append(title, pre);
      shadow.append(box);
      document.documentElement.append(host);
    }
    pre.textContent = JSON.stringify(tracker.snapshot(), null, 1);
  }

  window.addEventListener(
    "message",
    (event) => {
      if (event.source !== window || event.origin !== window.location.origin) return;
      const data = event.data;
      if (typeof data !== "object" || data === null || data.ns !== NS || data.type !== "port") return;
      event.stopImmediatePropagation();
      if (port || event.ports.length !== 1) return;
      port = event.ports[0];
      port.onmessage = (message) => {
        queue = queue.then(() => tracker.accept(message.data)).then(render, render);
      };
      port.postMessage("ack");
    },
    true,
  );
  // In case the wrapper's first offer was dispatched before this listener.
  window.postMessage({ ns: NS, v: 1, type: "ready" }, window.location.origin);
  document.addEventListener("DOMContentLoaded", render);
})();
