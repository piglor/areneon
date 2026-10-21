import { describe, expect, it, vi } from "vitest";

// A worker that ends without an answer and without an error (it called
// process.exit, or was stopped from outside): the main thread settles at once
// with the fixed refusal instead of waiting for the wall-clock limit.

const stubs = vi.hoisted(() => ({ terminated: 0 }));
vi.mock("node:worker_threads", async () => {
  const { EventEmitter } = await import("node:events");
  class Worker extends EventEmitter {
    constructor() {
      super();
      setImmediate(() => this.emit("exit", 3));
    }
    terminate(): Promise<number> {
      stubs.terminated += 1;
      return Promise.resolve(0);
    }
  }
  return { Worker };
});

const { decodeIsolated, UNREADABLE } = await import("../../src/shards/private-decode.ts");

describe("a worker that exits without answering", () => {
  it("is an unreadable file with the fixed message well before the time limit", async () => {
    const started = Date.now();
    const error = await decodeIsolated(new Uint8Array([1, 2, 3]), { columns: [], rows: 0 }, 60_000).catch((e: unknown) => e);
    expect((error as Error).message).toBe(UNREADABLE);
    expect(Date.now() - started).toBeLessThan(5000);
    expect(stubs.terminated).toBe(1);
  });
});
