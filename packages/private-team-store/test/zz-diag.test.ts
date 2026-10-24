import { readFileSync } from "node:fs";
import { join } from "node:path";
import { readTrainingRunConfig, resolveFormat, teamVocabularyHash, canonicalJson } from "@areneon/showdown-bridge";
import { loadFixtureInputs } from "@areneon/team-dataset";
import { it } from "vitest";
import { readPrivateSnapshot } from "../src/snapshot-reader.ts";
import { writePrivateSnapshot } from "../src/snapshot-writer.ts";
import { engine, FORMAT_ID, input, records, ROOT, standIn } from "./snapshot-support.ts";

it("diag", async () => {
  const out: string[] = [];
  try {
    const tiny = JSON.parse(readFileSync(join(ROOT, "fixtures/contracts/v1/training-run-config/valid/tiny.json"), "utf8"));
    const config = readTrainingRunConfig(tiny);
    out.push(`format ${config.format_id}`);
    const inputs = loadFixtureInputs(engine, FORMAT_ID);
    const vocabularyHash = teamVocabularyHash(inputs.vocabulary);
    const w = await writePrivateSnapshot(input({ records: records(12), vocabularyHash }, standIn()));
    out.push(`written ${w.snapshotHash.slice(0, 8)} ${canonicalJson(w.manifest.manifest as never) === canonicalJson(resolveFormat(engine, config.format_id).manifest as never)}`);
    await readPrivateSnapshot({ engine, manifest: w.manifest, shard: (i) => w.shards[i] as Uint8Array, labelSet: inputs.labelSet, expectedSnapshotHash: w.snapshotHash, expected: { manifest: resolveFormat(engine, config.format_id).manifest, vocabularyHash } });
    out.push("read ok");
  } catch (error) {
    out.push(`ERR ${(error as Error).name} ${(error as { code?: string }).code ?? ""} ${(error as Error).message.slice(0, 200)}`);
  }
  throw new Error(`DIAG ${out.join(" | ")}`);
});
