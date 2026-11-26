import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";

// Reading fails with something other than "nothing yet": the process is closed and the error is the transport's.
vi.mock("node:fs", async (original) => {
  const fs = await original<typeof import("node:fs")>();
  return { ...fs, readSync: vi.fn(fs.readSync) };
});
const { readSync } = await import("node:fs");
const { NativeProbeError, NativeProbeProcess } = await import("../src/native-probe-process.ts");

const FAKE = fileURLToPath(new URL("./support/fake-native-probe.mjs", import.meta.url));
afterEach(() => vi.mocked(readSync).mockClear());

describe("a read that fails", () => {
  it("ends the request with the pipe's error code", () => {
    process.env.FAKE_MODE = "plain";
    const p = new NativeProbeProcess({ bin: FAKE, dex: "unused" });
    vi.mocked(readSync).mockImplementationOnce(() => {
      throw Object.assign(new Error("io"), { code: "EIO" });
    });
    expect(() => p.request(Buffer.from(JSON.stringify({ op: "counts" })))).toThrow(NativeProbeError);
    expect(p.alive).toBe(false);
    delete process.env.FAKE_MODE;
  });
});
