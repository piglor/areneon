import { afterEach, describe, expect, it, vi } from "vitest";
import { EXIT_CONFIG } from "../src/index.ts";

afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
  process.exitCode = undefined;
});

describe("cli", () => {
  it("runs the import command on the environment and sets the exit code", async () => {
    vi.stubEnv("DATABASE_URL", "");
    const write = vi.spyOn(process.stdout, "write").mockImplementation(() => true);
    await import("../src/cli.ts");
    expect(process.exitCode).toBe(EXIT_CONFIG);
    expect(write).toHaveBeenCalledWith('{"ok":false,"error":"ConfigError: DATABASE_URL is not set"}\n');
  });
});
