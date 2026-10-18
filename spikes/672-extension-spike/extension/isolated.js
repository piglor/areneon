// SPIKE (#672): throwaway. Runs in the extension's ISOLATED world.
//
// Treats every MAIN-world message as untrusted: AreneonProtocol validates
// the envelope schema, framing and request shape before anything is shown.
// It renders a read-only diagnostic panel. There is no submit path: nothing
// here writes to the page's socket or posts back into the MAIN world.
(() => {
  "use strict";
  const { Tracker } = globalThis.AreneonProtocol;
  const tracker = new Tracker();
  let queue = Promise.resolve();
  let pre = null;

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

  window.addEventListener("message", (event) => {
    // Only same-window, same-origin posts; content is validated by the tracker.
    if (event.source !== window || event.origin !== window.location.origin) return;
    const data = event.data;
    queue = queue.then(() => tracker.accept(data)).then(render, render);
  });
  document.addEventListener("DOMContentLoaded", render);
})();
