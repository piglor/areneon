import { assertLocal } from "./local-url.ts";
import { Listeners, NORMAL_CLOSURE, type Transport, TransportClosedError } from "./transport.ts";

/**
 * The close code for a binary frame: Showdown sends only text. RFC 6455's own
 * 1003 (unsupported data) is not one a WebSocket client may send, so this is
 * 4003 from the private-use range 4000-4999.
 */
export const BINARY_FRAME_CLOSE = 4003;

/** The connection could not be opened. */
export class ConnectError extends Error {
  override name = "ConnectError";
}

/**
 * Opens a WebSocket Transport to a local Showdown server, on Node's built-in
 * `WebSocket`. The URL must pass `assertLocal` before any connection is
 * attempted. Resolves once the connection is open; rejects with a
 * `LocalUrlError` for a refused URL or a `ConnectError` if it never opens.
 * A binary frame closes the connection with `BINARY_FRAME_CLOSE`.
 */
export async function connectWebSocket(url: string): Promise<Transport> {
  const local = assertLocal(url);
  const socket = new WebSocket(local.href);
  const listeners = new Listeners();
  socket.addEventListener("message", (event) => {
    if (typeof event.data === "string") listeners.frame(event.data);
    else socket.close(BINARY_FRAME_CLOSE, "binary frame");
  });
  socket.addEventListener("close", (event) => listeners.close(event.code));
  // A close before open rejects; one after open settles nothing (the promise has resolved).
  await new Promise<void>((resolve, reject) => {
    socket.addEventListener("open", () => resolve());
    socket.addEventListener("close", (event) =>
      reject(new ConnectError(`could not connect to ${local.href} (close code ${event.code})`)),
    );
  });
  return {
    send(line) {
      if (socket.readyState !== WebSocket.OPEN) throw new TransportClosedError("send on a closed transport");
      socket.send(line);
    },
    onFrame: (cb) => listeners.onFrame(cb),
    onClose: (cb) => listeners.onClose(cb),
    close: () => socket.close(NORMAL_CLOSURE),
  };
}
