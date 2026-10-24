import { mkdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { canonicalJson, readTrainingRunConfig, resolveFormat, teamVocabularyHash } from "@areneon/showdown-bridge";
import { loadFixtureInputs } from "@areneon/team-dataset";
import { it } from "vitest";
import { writeLocalCopy } from "../src/local-copy.ts";
import { readPrivateSnapshot } from "../src/snapshot-reader.ts";
import { writePrivateSnapshot } from "../src/snapshot-writer.ts";
import { engine, FORMAT_ID, input, records, ROOT, standIn, workDir } from "./snapshot-support.ts";

it("diag", async () => {
  const out: string[] = [];
  const step = async (name: string, run: () => unknown): Promise<unknown> => {
    try {
      const v = await run();
      out.push(`${name} ok`);
      return v;
    } catch (error) {
      out.push(`${name} ERR ${(error as Error).name} ${(error as { code?: string }).code ?? ""} ${(error as Error).message.slice(0, 160)}`);
      return undefined;
    }
  };
  const config = readTrainingRunConfig(JSON.parse(readFileSync(join(ROOT, "fixtures/contracts/v1/training-run-config/valid/tiny.json"), "utf8")));
  const inputs = (await step("inputs", () => loadFixtureInputs(engine, config.format_id))) as ReturnType<typeof loadFixtureInputs>;
  const vocabularyHash = teamVocabularyHash(inputs.vocabulary);
  const w = await writePrivateSnapshot(input({ records: records(12), vocabularyHash }, standIn()));
  const dir = join(dirname(workDir()), "snapshots");
  mkdirSync(dir, { recursive: true, mode: 0o700 });
  await step("copy", () => writeLocalCopy(dir, w.snapshotHash, new TextEncoder().encode(canonicalJson(w.manifest as never)), w.shards));
  const manifest = JSON.parse(readFileSync(join(dir, w.snapshotHash, "manifest.json"), "utf8"));
  await step("read", () =>
    readPrivateSnapshot({
      engine,
      manifest,
      shard: (index) => readFileSync(join(dir, w.snapshotHash, "shards", `${index}.parquet`)),
      labelSet: inputs.labelSet,
      expectedSnapshotHash: w.snapshotHash,
      expected: { manifest: resolveFormat(engine, FORMAT_ID).manifest, vocabularyHash },
    }),
  );
  throw new Error(`DIAG ${out.join(" | ")} shards=${w.shards.length}`);
});
