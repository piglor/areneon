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

export interface WebSocketOptions {
  /** How long to wait for the connection to open. Default 10 000 ms. */
  readonly connectTimeoutMs?: number;
  /**
   * How long to wait, once the client starts closing, for the peer to finish
   * the close handshake. Default 5 000 ms. On expiry the close listeners get
   * the code the client sent, and the transport stops waiting.
   */
  readonly closeTimeoutMs?: number;
}

/**
 * Opens a WebSocket Transport to a local Showdown server, on Node's built-in
 * `WebSocket`. The URL must pass `assertLocal` before any connection is
 * attempted. Resolves once the connection is open; rejects with a
 * `LocalUrlError` for a refused URL, or a `ConnectError` if the socket cannot
 * be constructed, closes before it opens, or does not open in time.
 * A binary frame closes the connection with `BINARY_FRAME_CLOSE`.
 */
export async function connectWebSocket(url: string, options: WebSocketOptions = {}): Promise<Transport> {
  const { connectTimeoutMs = 10_000, closeTimeoutMs = 5_000 } = options;
  const local = assertLocal(url);
  let socket: WebSocket;
  try {
    socket = new WebSocket(local.href);
  } catch (cause) {
    throw new ConnectError(`could not construct a WebSocket for ${local.href}`, { cause });
  }
  const listeners = new Listeners();
  let closeTimer: ReturnType<typeof setTimeout> | undefined;
  const closeWith = (code: number, reason?: string): void => {
    if (closeTimer !== undefined || listeners.closed) return;
    socket.close(code, reason);
    closeTimer = setTimeout(() => listeners.close(code), closeTimeoutMs);
  };
  socket.addEventListener("message", (event) => {
    if (typeof event.data === "string") listeners.frame(event.data);
    else closeWith(BINARY_FRAME_CLOSE, "binary frame");
  });
  socket.addEventListener("close", (event) => {
    clearTimeout(closeTimer);
    listeners.close(event.code);
  });
  // A close before open rejects; one after open settles nothing (the promise has resolved).
  await new Promise<void>((resolve, reject) => {
    const connectTimer = setTimeout(() => {
      reject(new ConnectError(`could not connect to ${local.href} within ${connectTimeoutMs} ms`));
      socket.close();
    }, connectTimeoutMs);
    socket.addEventListener("open", () => {
      clearTimeout(connectTimer);
      resolve();
    });
    socket.addEventListener("close", (event) => {
      clearTimeout(connectTimer);
      reject(new ConnectError(`could not connect to ${local.href} (close code ${event.code})`));
    });
  });
  return {
    send(line) {
      if (socket.readyState !== WebSocket.OPEN) throw new TransportClosedError("send on a closed transport");
      socket.send(line);
    },
    onFrame: (cb) => listeners.onFrame(cb),
    onClose: (cb) => listeners.onClose(cb),
    close: () => closeWith(NORMAL_CLOSURE),
  };
}
