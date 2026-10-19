import { createServer } from "node:http";
import { createServer as createTcpServer, type Socket } from "node:net";
import type { AddressInfo } from "node:net";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  ConnectError,
  connectWebSocket,
  LocalUrlError,
  NORMAL_CLOSURE,
  type Transport,
  TransportClosedError,
  BINARY_FRAME_CLOSE,
} from "../src/index.ts";
import { type LocalWsServer, startLocalWsServer } from "./local-ws-server.ts";

const closeCode = (t: Transport): Promise<number> => new Promise((resolve) => t.onClose(resolve));

describe("connectWebSocket against a real local WebSocket server", () => {
  let server: LocalWsServer | undefined;
  afterEach(async () => {
    await server?.stop();
    server = undefined;
    vi.unstubAllGlobals();
  });

  it("sends lines, receives frames, and closes normally", async () => {
    server = await startLocalWsServer();
    const t = await connectWebSocket(server.url);
    await server.connected;
    // A frame sent before any listener is registered is still delivered.
    server.sendText("|challstr|4|abc");
    const frames: string[] = [];
    const twoFrames = new Promise<void>((resolve) =>
      t.onFrame((f) => {
        frames.push(f);
        if (frames.length === 2) resolve();
      }),
    );
    // Long enough for the 16-bit length form.
    const long = `>battle-1\n${"|".repeat(300)}`;
    server.sendText(long);
    await twoFrames;
    expect(frames).toEqual(["|challstr|4|abc", long]);

    t.send("|/cmd rooms");
    t.send(`|/trn bot,0,${"x".repeat(200)}`);
    await vi.waitFor(() => expect(server?.received).toHaveLength(2));
    expect(server.received[0]).toBe("|/cmd rooms");

    const closed = closeCode(t);
    t.close();
    expect(await closed).toBe(NORMAL_CLOSURE);
    expect(await server.clientClose).toBe(NORMAL_CLOSURE);
    expect(() => t.send("late")).toThrow(TransportClosedError);
  });

  it("reports the server's close code", async () => {
    server = await startLocalWsServer();
    const t = await connectWebSocket(server.url);
    await server.connected;
    const closed = closeCode(t);
    server.closeWith(4000);
    expect(await closed).toBe(4000);
  });

  it("closes with 4003 on a binary frame", async () => {
    server = await startLocalWsServer();
    const t = await connectWebSocket(server.url);
    await server.connected;
    const closed = closeCode(t);
    server.sendBinary(new Uint8Array([1, 2, 3]));
    expect(await server.clientClose).toBe(BINARY_FRAME_CLOSE);
    expect(await closed).toBe(BINARY_FRAME_CLOSE);
  });

  it("rejects with ConnectError when nothing listens on the port", async () => {
    const probe = createServer();
    await new Promise<void>((resolve) => probe.listen(0, "127.0.0.1", resolve));
    const { port } = probe.address() as AddressInfo;
    await new Promise<void>((resolve) => probe.close(() => resolve()));
    await expect(connectWebSocket(`ws://127.0.0.1:${port}/showdown/websocket`)).rejects.toThrow(ConnectError);
  });

  it("ignores a second close and a close after the server closed", async () => {
    server = await startLocalWsServer();
    const t = await connectWebSocket(server.url);
    await server.connected;
    const codes: number[] = [];
    t.onClose((c) => codes.push(c));
    server.closeWith(4000);
    await vi.waitFor(() => expect(codes).toEqual([4000]));
    t.close();
    t.close();
    expect(codes).toEqual([4000]);
  });

  it("reports the client's close code once when the peer never answers the close", async () => {
    server = await startLocalWsServer({ ignoreClose: true });
    const t = await connectWebSocket(server.url, { closeTimeoutMs: 50 });
    await server.connected;
    const codes: number[] = [];
    t.onClose((c) => codes.push(c));
    t.close();
    t.close();
    expect(await server.clientClose).toBe(NORMAL_CLOSURE);
    await vi.waitFor(() => expect(codes).toEqual([NORMAL_CLOSURE]));
    await server.stop();
    server = undefined;
    // The socket's own close event, when it finally comes, is not reported again.
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(codes).toEqual([NORMAL_CLOSURE]);
  });

  it("reports 4003 when the peer never answers the binary-frame close", async () => {
    server = await startLocalWsServer({ ignoreClose: true });
    const t = await connectWebSocket(server.url, { closeTimeoutMs: 50 });
    await server.connected;
    const closed = closeCode(t);
    server.sendBinary(new Uint8Array([1]));
    expect(await closed).toBe(BINARY_FRAME_CLOSE);
  });

  it("rejects with ConnectError when the handshake does not finish in time", async () => {
    const sockets: Socket[] = [];
    const silent = createTcpServer((socket) => sockets.push(socket));
    await new Promise<void>((resolve) => silent.listen(0, "127.0.0.1", resolve));
    const { port } = silent.address() as AddressInfo;
    try {
      await expect(
        connectWebSocket(`ws://127.0.0.1:${port}/showdown/websocket`, { connectTimeoutMs: 50 }),
      ).rejects.toThrow(/within 50 ms/);
    } finally {
      for (const socket of sockets) socket.destroy();
      await new Promise<void>((resolve) => silent.close(() => resolve()));
    }
  });

  it("turns a throw from the WebSocket constructor into ConnectError", async () => {
    vi.stubGlobal(
      "WebSocket",
      // A function expression, not an arrow, so `new` can call it.
      vi.fn(function () {
        throw new SyntaxError("refused by the constructor");
      }),
    );
    const failure = connectWebSocket("ws://127.0.0.1:8000/showdown/websocket");
    await expect(failure).rejects.toThrow(ConnectError);
    await expect(failure).rejects.toMatchObject({ cause: new SyntaxError("refused by the constructor") });
  });

  it("checks the URL with assertLocal before constructing a WebSocket", async () => {
    const ctor = vi.fn();
    vi.stubGlobal("WebSocket", ctor);
    await expect(connectWebSocket("ws://example.com/showdown/websocket")).rejects.toThrow(LocalUrlError);
    await expect(connectWebSocket("ws://2130706433:8000/showdown/websocket")).rejects.toThrow(LocalUrlError);
    await expect(connectWebSocket("ws://127.0.0.1:8000/showdown/websocket#")).rejects.toThrow(LocalUrlError);
    expect(ctor).not.toHaveBeenCalled();
  });
});
