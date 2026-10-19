import { describe, expect, it } from "vitest";
import { memoryTransport, NORMAL_CLOSURE, TransportClosedError } from "../src/index.ts";

describe("memoryTransport", () => {
  it("records sent lines and delivers pushed frames in order", () => {
    const t = memoryTransport();
    const frames: string[] = [];
    t.onFrame((f) => frames.push(f));
    t.send("|/cmd rooms");
    t.push("|challstr|abc");
    t.push(">battle-1\n|init|battle");
    t.send("|/utm null");
    expect(t.sent).toEqual(["|/cmd rooms", "|/utm null"]);
    expect(frames).toEqual(["|challstr|abc", ">battle-1\n|init|battle"]);
  });

  it("queues frames until the first listener, then delivers to every listener", () => {
    const t = memoryTransport();
    t.push("a");
    t.push("b");
    const first: string[] = [];
    const second: string[] = [];
    t.onFrame((f) => first.push(f));
    t.onFrame((f) => second.push(f));
    t.push("c");
    expect(first).toEqual(["a", "b", "c"]);
    expect(second).toEqual(["c"]);
  });

  it("keeps undelivered frames queued when a listener throws during the backlog replay", () => {
    const t = memoryTransport();
    t.push("a");
    t.push("b");
    t.push("c");
    const seen: string[] = [];
    expect(() =>
      t.onFrame((f) => {
        seen.push(f);
        if (f === "b") throw new Error("listener failed on b");
      }),
    ).toThrow("listener failed on b");
    // "b" was consumed by the throw; "c" is still queued and goes out, in order, before "d".
    expect(seen).toEqual(["a", "b"]);
    t.push("d");
    expect(seen).toEqual(["a", "b", "c", "d"]);
  });

  it("reports a server close once, to listeners registered before and after it", () => {
    const t = memoryTransport();
    const codes: number[] = [];
    t.onClose((c) => codes.push(c));
    expect(t.closed).toBe(false);
    t.end(4000);
    t.end(4001);
    t.close();
    t.onClose((c) => codes.push(c));
    expect(t.closed).toBe(true);
    expect(codes).toEqual([4000, 4000]);
  });

  it("reports a client close as normal closure and refuses later sends and pushes", () => {
    const t = memoryTransport();
    const codes: number[] = [];
    t.onClose((c) => codes.push(c));
    t.close();
    expect(codes).toEqual([NORMAL_CLOSURE]);
    expect(() => t.send("x")).toThrow(TransportClosedError);
    expect(() => t.push("y")).toThrow(TransportClosedError);
    expect(t.sent).toEqual([]);
  });
});
