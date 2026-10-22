import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { jcs, type Json } from "../src/canonical-json.ts";
import { checkStringLength, ChunkedFileWriter, readFileLines } from "../src/chunked-file.ts";
import { goldenBattle, syntheticFixtureHashes, syntheticTeam } from "../src/golden.ts";
import { type BattleRecord, type Episode, episodeFromRecord, loadEngine, loadTrainingSet, nodeSha256, stampTeam, writeTrainingStream } from "../src/index.ts";
import type { PrivateTeamSpec } from "@areneon/schema/types/v1/private-team-spec.ts";

let dir: string;
beforeAll(() => {
  dir = mkdtempSync(join(tmpdir(), "areneon-chunked-file-"));
});
afterAll(() => rmSync(dir, { recursive: true, force: true }));

describe("ChunkedFileWriter and readFileLines", () => {
  const text = ["alpha", "bêta ünïcode 🎲", "", "x".repeat(300), "last"].map((line) => `${line}\n`).join("");
  it("writes the bytes of the joined text for every chunk size, and counts them", () => {
    for (const chunk_chars of [1, 7, 64, 100000]) {
      const path = join(dir, `w-${chunk_chars}`);
      const writer = new ChunkedFileWriter(path, { chunk_chars });
      for (const line of text.split("\n").slice(0, -1)) writer.write(`${line}\n`);
      writer.finish();
      expect(readFileSync(path, "utf8")).toBe(text);
      expect(writer.bytes).toBe(Buffer.byteLength(text));
    }
  });
  it("refuses a piece or a chunk over the string limit", () => {
    const path = join(dir, "limit");
    const writer = new ChunkedFileWriter(path, { chunk_chars: 100, max_string_chars: 10 });
    expect(() => writer.write("x".repeat(11))).toThrow(RangeError);
    const joined = new ChunkedFileWriter(path, { chunk_chars: 100, max_string_chars: 10 });
    joined.write("x".repeat(6));
    expect(() => joined.write("x".repeat(6))).not.toThrow(); // under the chunk size: kept
    expect(() => joined.finish()).toThrow(RangeError); // the 12-character chunk
    writer.close();
    expect(() => checkStringLength("abc", {}, "x")).not.toThrow();
  });
  it("reads lines back across read boundaries, multibyte characters included, and a last line without a newline", () => {
    const path = join(dir, "r");
    writeFileSync(path, text + "no newline");
    for (const read_bytes of [1, 3, 5, 4096]) expect([...readFileLines(path, read_bytes)]).toEqual([...text.split("\n").slice(0, -1), "no newline"]);
    writeFileSync(path, "");
    expect([...readFileLines(path)]).toEqual([]);
  });
});

describe("the training stream writer against the in-memory loader (committed fixtures)", () => {
  const KNOWN = syntheticFixtureHashes(loadEngine());
  const team = stampTeam(syntheticTeam() as PrivateTeamSpec["team"], KNOWN);
  let record: BattleRecord;
  let episodes: Episode[];
  beforeAll(async () => {
    record = await goldenBattle(loadEngine());
    episodes = await Promise.all(
      (["p1", "p2"] as const).map((player) =>
        episodeFromRecord(record, player, {
          team,
          opponent_team: team,
          known_synthetic: KNOWN,
          behaviour_policy_id: "first-legal",
          collection: { kind: "league", run_id: "league-1", seed: record.seed, opponent_policy_id: "first-legal" },
        }),
      ),
    );
  });
  const input = (eps: Episode[]) => ({ episodes: eps, evaluation_manifests: [], known_synthetic: KNOWN, behaviour_policy_id: "first-legal" });

  it("gives the bytes, the digest and the TrainingSet the one-string writer gave", async () => {
    // The old writer, as it was: every line joined into one string, hashed as one.
    const { examples, set } = await loadTrainingSet(input(episodes));
    const old = examples.map((example) => `${jcs(example as unknown as Json)}\n`).join("");
    expect(set.stream_sha256).toBe(await nodeSha256(new TextEncoder().encode(old)));
    for (const chunk_chars of [1, 1000, 1 << 20]) {
      const path = join(dir, `s-${chunk_chars}`);
      const written = await writeTrainingStream(input(episodes), { path, chunk_chars });
      expect(readFileSync(path, "utf8")).toBe(old);
      expect(written).toEqual(set);
    }
  });

  it("writes a stream far over the string limit it is given, where the in-memory loader refuses", async () => {
    const { stream } = await loadTrainingSet(input(episodes));
    const longest = Math.max(...stream.split("\n").map((line) => line.length + 1));
    const limits = { chunk_chars: longest, max_string_chars: longest * 2 };
    expect(stream.length).toBeGreaterThan(limits.max_string_chars * 2);
    await expect(loadTrainingSet(input(episodes), limits)).rejects.toThrow(/writeTrainingStream/);
    const path = join(dir, "big");
    const set = await writeTrainingStream(input(episodes), { path, ...limits });
    expect(readFileSync(path, "utf8")).toBe(stream);
    expect(set.stream_sha256).toBe(await nodeSha256(new TextEncoder().encode(stream)));
    // A single example over the limit is refused, never split.
    await expect(writeTrainingStream(input(episodes), { path, chunk_chars: 10, max_string_chars: 10 })).rejects.toThrow(RangeError);
  });
});
