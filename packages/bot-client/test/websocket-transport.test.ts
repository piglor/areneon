import { createServer } from "node:http";
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

  it("checks the URL with assertLocal before constructing a WebSocket", async () => {
    const ctor = vi.fn();
    vi.stubGlobal("WebSocket", ctor);
    await expect(connectWebSocket("ws://example.com/showdown/websocket")).rejects.toThrow(LocalUrlError);
    await expect(connectWebSocket("ws://2130706433:8000/showdown/websocket")).rejects.toThrow(LocalUrlError);
    expect(ctor).not.toHaveBeenCalled();
  });
});
