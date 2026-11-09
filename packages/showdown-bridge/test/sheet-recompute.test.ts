// The play-time sheet sidecar's reader and writer (#1315) on their own: engine-free, a temp directory and a stub Episode.
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { OpponentSheet } from "@areneon/schema/types/v1/opponent-sheet.ts";
import { afterAll, describe, expect, it, vi } from "vitest";
import { SAMPLES_DIR } from "../src/samples.ts";

vi.mock("../src/episode-io.ts", () => ({
  decisionObservations: () => Promise.resolve([{ identity: { request_seq: 1 } }, { identity: { request_seq: 3 } }]),
}));
const { readSheetBlocks, writePlaySheets } = await import("../src/sheet-recompute.ts");

const sample = (name: string): OpponentSheet => JSON.parse(readFileSync(join(SAMPLES_DIR, `v1/opponent-sheet/valid/${name}.json`), "utf8")) as OpponentSheet;
const block = sample("belief");
const SOURCE = block.source.hash;
const HASH = "1".repeat(64);
const dir = mkdtempSync(join(tmpdir(), "areneon-sheet-recompute-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const file = join(dir, `${HASH}.sheet.json`);
const put = (value: unknown): void => writeFileSync(file, typeof value === "string" ? value : JSON.stringify(value));
const good = { episode_hash: HASH, source_hash: SOURCE, request_seqs: [1, 3], blocks: [block, block] };
const read = readSheetBlocks(SOURCE, dir);

describe("writePlaySheets", () => {
  const episode = {} as never;
  it("writes one block per decision and nothing for a gap or another source", async () => {
    expect(await writePlaySheets(SOURCE, episode, HASH, new Map([[1, block]]), dir)).toBe(false); // request 3 has no block
    expect(await writePlaySheets("f".repeat(64), episode, HASH, new Map([[1, block], [3, block]]), dir)).toBe(false); // another source
    expect(() => readFileSync(file)).toThrow();
    expect(await writePlaySheets(SOURCE, episode, HASH, new Map([[3, block], [1, block]]), dir)).toBe(true);
    expect(read(HASH, [1, 3])?.length).toBe(2);
  });
});

describe("readSheetBlocks", () => {
  it("reads a good file and refuses each way a file can be stale or damaged", () => {
    put(good);
    expect(read(HASH, [1, 3])).toEqual([block, block]);
    expect(read(HASH, [1, 4])).toBeUndefined(); // other request numbers
    expect(read(HASH, [1])).toBeUndefined(); // other count
    put({ ...good, episode_hash: "2".repeat(64) });
    expect(read(HASH, [1, 3])).toBeUndefined();
    put({ ...good, source_hash: "f".repeat(64) });
    expect(read(HASH, [1, 3])).toBeUndefined();
    put({ ...good, blocks: [block] }); // block count differs from the request count
    expect(read(HASH, [1, 3])).toBeUndefined();
    put({ ...good, blocks: "x" });
    expect(read(HASH, [1, 3])).toBeUndefined();
    put({ ...good, request_seqs: "x" });
    expect(read(HASH, [1, 3])).toBeUndefined();
    put({ ...good, blocks: [block, sample("oracle-diagnostic")] });
    expect(read(HASH, [1, 3])).toBeUndefined();
    put({ ...good, blocks: [block, { ...block, source: { ...block.source, hash: "f".repeat(64) } }] });
    expect(read(HASH, [1, 3])).toBeUndefined();
    put({ ...good, blocks: [block, { ...block, slots: "x" }] }); // fails the reader rules
    expect(read(HASH, [1, 3])).toBeUndefined();
    put("{not json");
    expect(read(HASH, [1, 3])).toBeUndefined();
    expect(read("3".repeat(64), [1, 3])).toBeUndefined(); // no file
  });
});
