import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import type { Duplex } from "node:stream";

/**
 * A minimal RFC 6455 server on 127.0.0.1 for the WebSocket transport's tests,
 * on node:http alone (no dependency): the opening handshake, unfragmented
 * text, binary and close frames, and the close handshake. Test-only.
 */
export interface LocalWsServer {
  /** ws://127.0.0.1:<port>/showdown/websocket */
  readonly url: string;
  /** Text frames received from the client, in order. */
  readonly received: string[];
  /** Resolves once a client has completed the handshake. */
  readonly connected: Promise<void>;
  sendText(text: string): void;
  sendBinary(bytes: Uint8Array): void;
  /** Starts the close handshake with `code`. */
  closeWith(code: number): void;
  /** Resolves with the close code the client sent, once it has. */
  readonly clientClose: Promise<number>;
  stop(): Promise<void>;
}

const GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
const TEXT = 0x1;
const BINARY = 0x2;
const CLOSE = 0x8;

function frame(opcode: number, payload: Uint8Array): Buffer {
  const length = payload.length;
  const header =
    length < 126
      ? Buffer.from([0x80 | opcode, length])
      : Buffer.from([0x80 | opcode, 126, length >> 8, length & 0xff]);
  return Buffer.concat([header, payload]);
}

function closePayload(code: number): Buffer {
  return Buffer.from([code >> 8, code & 0xff]);
}

export interface LocalWsServerOptions {
  /** Never answer the client's close frame nor end the connection: a peer that hangs. */
  readonly ignoreClose?: boolean;
}

export async function startLocalWsServer(options: LocalWsServerOptions = {}): Promise<LocalWsServer> {
  const received: string[] = [];
  let socket: Duplex | undefined;
  let closing = false;
  let onConnected: () => void = () => {};
  let onClientClose: (code: number) => void = () => {};
  const connected = new Promise<void>((resolve) => (onConnected = resolve));
  const clientClose = new Promise<number>((resolve) => (onClientClose = resolve));

  const server: Server = createServer((_req, res) => res.writeHead(426).end());
  server.on("upgrade", (req, sock: Duplex) => {
    const key = req.headers["sec-websocket-key"];
    const accept = createHash("sha1").update(`${String(key)}${GUID}`).digest("base64");
    sock.write(
      "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n" +
        `Sec-WebSocket-Accept: ${accept}\r\n\r\n`,
    );
    socket = sock;
    let buffer = Buffer.alloc(0);
    sock.on("data", (chunk: Buffer) => {
      buffer = Buffer.concat([buffer, chunk]);
      for (;;) {
        if (buffer.length < 2) return;
        const opcode = buffer.readUInt8(0) & 0x0f;
        let length = buffer.readUInt8(1) & 0x7f;
        let offset = 2;
        if (length === 126) {
          if (buffer.length < 4) return;
          length = buffer.readUInt16BE(2);
          offset = 4;
        }
        // Client frames are always masked (RFC 6455 §5.3).
        if (buffer.length < offset + 4 + length) return;
        const mask = buffer.subarray(offset, offset + 4);
        const payload = Buffer.from(buffer.subarray(offset + 4, offset + 4 + length));
        for (let i = 0; i < payload.length; i++) payload.writeUInt8(payload.readUInt8(i) ^ mask.readUInt8(i % 4), i);
        buffer = buffer.subarray(offset + 4 + length);
        if (opcode === TEXT) received.push(payload.toString("utf8"));
        if (opcode === CLOSE) {
          onClientClose(payload.length >= 2 ? payload.readUInt16BE(0) : 1005);
          if (options.ignoreClose === true) return;
          if (!closing) sock.write(frame(CLOSE, payload));
          sock.end();
          return;
        }
      }
    });
    onConnected();
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;

  const write = (bytes: Buffer): void => {
    if (socket === undefined) throw new Error("no client connected");
    socket.write(bytes);
  };
  return {
    url: `ws://127.0.0.1:${port}/showdown/websocket`,
    received,
    connected,
    clientClose,
    sendText: (text) => write(frame(TEXT, Buffer.from(text, "utf8"))),
    sendBinary: (bytes) => write(frame(BINARY, bytes)),
    closeWith(code) {
      closing = true;
      write(frame(CLOSE, closePayload(code)));
    },
    stop: () =>
      new Promise<void>((resolve) => {
        socket?.destroy();
        server.close(() => resolve());
      }),
  };
}
