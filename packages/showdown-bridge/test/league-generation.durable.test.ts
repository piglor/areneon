// writeGeneration flushes every file and directory it writes (the Python driver hashes and renames them next).
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it, vi } from "vitest";
import type { GenerationResult } from "../src/league-generation.ts";

const flushed: string[] = [];
vi.mock("node:fs", async (original) => {
  const real = await original<typeof import("node:fs")>();
  const paths = new Map<number, string>();
  return {
    ...real,
    openSync: (path: string, flags: string) => {
      const descriptor = real.openSync(path, flags);
      paths.set(descriptor, path);
      return descriptor;
    },
    fsyncSync: (descriptor: number) => {
      flushed.push(paths.get(descriptor) as string);
      real.fsyncSync(descriptor);
    },
  };
});

const { writeGeneration } = await import("../src/league-generation.ts");

describe("writeGeneration is durable", () => {
  const dir = mkdtempSync(join(tmpdir(), "areneon-durable-"));
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  it("flushes each file after writing it, generation.json last, and then both directories", () => {
    const result = {
      generation: 1,
      plan: [],
      run_record: { schema_version: 1 },
      games: [],
      episodes: [{ a: 1 }],
      episodes_dir: null,
      episode_hashes: ["h1"],
      stream: "line\n",
      stream_path: null,
      set: { examples: 1 },
      training_set_hash: "t",
      manifest: {},
      team_hashes: [],
    } as unknown as GenerationResult;
    writeGeneration(join(dir, "out"), result);
    const out = join(dir, "out");
    expect(flushed).toEqual([
      join(out, "episodes/h1.json"),
      join(out, "stream.jsonl"),
      join(out, "training-set.json"),
      join(out, "plan.json"),
      join(out, "league-run.json"),
      join(out, "generation.json"),
      join(out, "episodes"),
      out,
    ]);
    expect(readFileSync(join(out, "stream.jsonl"), "utf8")).toBe("line\n");
  });
});
