import { Listeners, NORMAL_CLOSURE, type Transport, TransportClosedError } from "./transport.ts";

/**
 * An in-memory Transport for tests: the test plays the server, pushing frames
 * with `push`, closing with `end`, and reading every line the client sent
 * from `sent`. It behaves like the WebSocket transport: a send after the
 * close throws, and a client-side close reports code 1000.
 */
export interface MemoryTransport extends Transport {
  /** Every line the client sent, in order. */
  readonly sent: readonly string[];
  readonly closed: boolean;
  /** Delivers one server frame to the client. Throws once closed. */
  push(frame: string): void;
  /** Closes the connection from the server side with `code`. */
  end(code: number): void;
}

export interface MemoryTransportOptions {
  /** Called after each line the client sends: a scripted peer, which may `push` a reply at once. */
  readonly onSend?: (line: string, transport: MemoryTransport) => void;
}

export function memoryTransport(options: MemoryTransportOptions = {}): MemoryTransport {
  const listeners = new Listeners();
  const sent: string[] = [];
  const transport: MemoryTransport = {
    sent,
    get closed() {
      return listeners.closed;
    },
    send(line) {
      if (listeners.closed) throw new TransportClosedError("send on a closed transport");
      sent.push(line);
      options.onSend?.(line, transport);
    },
    onFrame: (cb) => listeners.onFrame(cb),
    onClose: (cb) => listeners.onClose(cb),
    close: () => listeners.close(NORMAL_CLOSURE),
    push(frame) {
      if (listeners.closed) throw new TransportClosedError("push on a closed transport");
      listeners.frame(frame);
    },
    end: (code) => listeners.close(code),
  };
  return transport;
}
