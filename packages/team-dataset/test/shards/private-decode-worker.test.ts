import { describe, expect, it, vi } from "vitest";

// A worker that dies with an error (a rejection the library left unhandled
// inside it, an out-of-memory stop): the main thread gets the fixed refusal
// and the worker is terminated, and nothing the worker said is passed on.

const stubs = vi.hoisted(() => ({ terminated: 0 }));
vi.mock("node:worker_threads", async () => {
  const { EventEmitter } = await import("node:events");
  class Worker extends EventEmitter {
    constructor() {
      super();
      setImmediate(() => this.emit("error", new Error("ERR_WORKER_OUT_OF_MEMORY CanaryLibrary7")));
    }
    terminate(): Promise<number> {
      stubs.terminated += 1;
      return Promise.resolve(0);
    }
  }
  return { Worker };
});

const { decodeIsolated, UNREADABLE } = await import("../../src/shards/private-decode.ts");

describe("a worker that fails", () => {
  it("is an unreadable file with the fixed message, and the worker is terminated", async () => {
    const error = await decodeIsolated(new Uint8Array([1, 2, 3]), { columns: [], rows: 0 }, 5000).catch((e: unknown) => e);
    expect((error as Error).message).toBe(UNREADABLE);
    expect(String((error as Error).stack)).not.toContain("CanaryLibrary7");
    expect(stubs.terminated).toBe(1);
  });
});
