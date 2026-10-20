// `scripts/play-generation.ts`: one generation of A1 play for the Python driver
// (ml/areneon_ml/loop). It takes everything on the command line, starts the
// Python policy worker (CPU torch, stdio only) with the frozen pair, plays the
// generation, writes its files and prints one canonical JSON line, the
// generation's `generation.json`. Exit 0 ok, 1 a generation that failed, 2 a
// usage or configuration error. Nothing is retried.

import { readFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { engineDataHash, readEngineData } from "./engine-data.ts";
import { loadEngine } from "./engine.ts";
import { GenerationError, playGeneration, trainingTeamPool, writeGeneration } from "./league-generation.ts";
import { type Json, jcs } from "./canonical-json.ts";
import { resolveFormat } from "./formats.ts";
import { PolicyWorkerClient } from "./policy-worker.ts";
import { readTrainingRunConfig, runConfigHash } from "./training-run-config.ts";

export class UsageError extends Error {
  override name = "UsageError";
}

const FLAGS = {
  config: { type: "string" },
  generation: { type: "string" },
  "policy-id": { type: "string" },
  preview: { type: "string" },
  battle: { type: "string" },
  "run-dir": { type: "string" },
  "engine-data": { type: "string" },
  "encoder-version": { type: "string" },
  "worker-python": { type: "string" },
  out: { type: "string" },
} as const;

/** Run one generation; returns the process exit code and writes its one JSON line with `print`. */
export async function runGenerationCli(argv: readonly string[], print: (line: string) => void = console.log): Promise<number> {
  let values: Record<keyof typeof FLAGS, string | undefined>;
  try {
    values = parseArgs({ args: [...argv], options: FLAGS, strict: true }).values as typeof values;
  } catch (error) {
    throw new UsageError((error as Error).message);
  }
  const need = (name: keyof typeof FLAGS): string => {
    const value = values[name];
    if (value === undefined) throw new UsageError(`--${name} is required`);
    return value;
  };
  for (const name of Object.keys(FLAGS) as (keyof typeof FLAGS)[]) need(name); // refuse a missing flag before anything starts
  const generation = Number(need("generation"));
  if (!Number.isInteger(generation) || generation < 1) throw new UsageError("--generation must be a positive integer");
  const config = readTrainingRunConfig(JSON.parse(readFileSync(need("config"), "utf8")));
  const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
  const dataFile = need("engine-data");
  const dataHash = engineDataHash(readEngineData(JSON.parse(readFileSync(dataFile, "utf8"))));
  const { max_line_bytes, n_parallel, handshake_timeout_ms } = config.generation;
  const preview = need("preview");
  const battle = need("battle");
  const client = new PolicyWorkerClient({
    command: need("worker-python"),
    args: [
      "-I", "-m", "areneon_ml.worker", "--run-dir", need("run-dir"), "--preview", preview, "--battle", battle,
      "--engine-data", dataFile, "--max-line-bytes", String(max_line_bytes),
    ],
    env: {},
    max_line_bytes,
    n_parallel,
    handshake_timeout_ms,
    expect: {
      encoder_version: need("encoder-version"),
      engine_data_hash: dataHash,
      checkpoints: [
        { role: "battle", checkpoint_id: battle },
        { role: "preview", checkpoint_id: preview },
      ],
    },
  });
  try {
    const pool = trainingTeamPool(engine, config.generation.excluded_abilities);
    await client.start();
    const result = await playGeneration({ engine, config, generation, policy_id: need("policy-id"), client, pool });
    writeGeneration(need("out"), result);
    print(jcs({ generation, run_config_hash: await runConfigHash(config), training_set_hash: result.training_set_hash, examples: result.set.examples }));
    return 0;
  } catch (error) {
    if (error instanceof GenerationError) {
      console.error(`play-generation: ${error.message}`);
      return 1;
    }
    throw error;
  } finally {
    await client.close();
  }
}

/** `scripts/format-manifest.ts --format <id>`: print the version manifest of a supported format as one canonical JSON line. */
export function runManifestCli(argv: readonly string[], print: (line: string) => void = console.log): number {
  let format: string | undefined;
  try {
    format = parseArgs({ args: [...argv], options: { format: { type: "string" } }, strict: true }).values.format;
  } catch (error) {
    throw new UsageError((error as Error).message);
  }
  if (format === undefined) throw new UsageError("--format is required");
  print(jcs(resolveFormat(loadEngine(process.env.SHOWDOWN_DIR || undefined), format).manifest as unknown as Json));
  return 0;
}
