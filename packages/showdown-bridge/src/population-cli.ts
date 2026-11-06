// The population branch of `scripts/play-generation.ts` (ADR-021 slice 4): one policy worker per member, one
// generation of population play, the members' files and the top-level files written, one canonical JSON line
// printed. The Python driver (slice 5) will pass `--member` for each member; until then the loop driver refuses
// a population config, so this is reached only from the script.

import { mkdirSync } from "node:fs";
import type { Engine } from "./engine.ts";
import { jcs, type Json } from "./canonical-json.ts";
import { GenerationError, type TrainingPool } from "./league-generation.ts";
import type { BeliefWorldsFactory } from "./damage-aware-worlds.ts";
import { playCalibration, writeCalibration } from "./population-calibration.ts";
import { calibrationAnchors } from "./population-plan.ts";
import { checkPopulationConfig, type PopulationMember, playPopulationGeneration, writePopulationGeneration } from "./population-generation.ts";
import { PolicyWorkerClient } from "./policy-worker.ts";
import { runConfigHash, type TrainingRunConfig } from "./training-run-config.ts";

/** One `--member m<k>,PREVIEW,BATTLE,POLICY_ID` flag: the member's frozen pair (a policy id holds a colon, so the fields are comma separated). */
export interface MemberFlag {
  member: string;
  preview: string;
  battle: string;
  policy_id: string;
}

export function parseMemberFlag(text: string): MemberFlag {
  const match = /^(m(?:0|[1-9][0-9]?)),([0-9a-f]{64}),([0-9a-f]{64}),(policy:[0-9a-f]{64})$/.exec(text);
  if (match === null) throw new GenerationError("--member is m<k>,PREVIEW,BATTLE,POLICY_ID (two 64-hex checkpoint ids and a policy: id)");
  return { member: match[1] as string, preview: match[2] as string, battle: match[3] as string, policy_id: match[4] as string };
}

export interface PopulationCliInput {
  engine: Engine;
  config: TrainingRunConfig;
  generation: number;
  pool: TrainingPool;
  members: readonly MemberFlag[];
  out: string;
  run_dir: string;
  worker_python: string;
  engine_data: string;
  engine_data_hash: string;
  encoder_version: string;
  belief_worlds?: BeliefWorldsFactory | undefined;
  /** A test seam: the client of a member (default: a `PolicyWorkerClient` on the Python worker). */
  make_client?: (entry: MemberFlag) => MemberClient;
}

/** What the CLI needs of a member's worker client. */
export type MemberClient = PopulationMember["client"] & { start: () => Promise<void>; close: () => Promise<void> };

/** Run one population generation; returns the exit code (0 ok, 1 a failed generation) and prints the summary line. */
export async function runPopulationCli(input: PopulationCliInput, print: (line: string) => void): Promise<number> {
  const { config, generation } = input;
  const { max_line_bytes, n_parallel, handshake_timeout_ms } = config.generation;
  // A refused config costs no worker: each session is about 445 MB.
  try {
    checkPopulationConfig(config, input.members, generation);
  } catch (error) {
    if (!(error instanceof GenerationError)) throw error;
    console.error(`play-generation: ${error.message}`);
    return 1;
  }
  const clients: MemberClient[] = input.members.map(
    (entry) =>
      input.make_client?.(entry) ??
      new PolicyWorkerClient({
        command: input.worker_python,
        args: ["-I", "-m", "areneon_ml.worker", "--run-dir", input.run_dir, "--preview", entry.preview, "--battle", entry.battle, "--engine-data", input.engine_data, "--max-line-bytes", String(max_line_bytes)],
        env: {},
        max_line_bytes,
        n_parallel,
        handshake_timeout_ms,
        expect: {
          encoder_version: input.encoder_version,
          engine_data_hash: input.engine_data_hash,
          checkpoints: [
            { role: "battle", checkpoint_id: entry.battle },
            { role: "preview", checkpoint_id: entry.preview },
          ],
        },
      }),
  );
  try {
    mkdirSync(input.out, { recursive: true });
    await Promise.all(clients.map((client) => client.start()));
    const members: PopulationMember[] = input.members.map((entry, index) => ({ member: entry.member, policy_id: entry.policy_id, checkpoints: { preview: entry.preview, battle: entry.battle }, client: clients[index] as MemberClient }));
    const result = await playPopulationGeneration({ engine: input.engine, config, generation, members, pool: input.pool, out_dir: input.out, belief_worlds: input.belief_worlds });
    writePopulationGeneration(input.out, result);
    print(jcs({ generation, run_config_hash: await runConfigHash(config), members: result.members.map((entry) => ({ member: entry.member, training_set_hash: entry.training_set_hash, examples: entry.set.examples })) } as unknown as Json));
    return 0;
  } catch (error) {
    if (error instanceof GenerationError) {
      console.error(`play-generation: ${error.message}`);
      return 1;
    }
    throw error;
  } finally {
    await Promise.all(clients.map((client) => client.close()));
  }
}

export interface CalibrationCliInput {
  engine: Engine;
  config: TrainingRunConfig;
  pool: TrainingPool;
  out: string;
  belief_worlds?: BeliefWorldsFactory | undefined;
}

/**
 * Play the anchor calibration of a population run (generation 0, ADR-021 section 4) and write `calibration.json`; no
 * worker starts. Returns the exit code (0 ok, 1 a failed calibration) and prints one canonical JSON line.
 */
export async function runCalibrationCli(input: CalibrationCliInput, print: (line: string) => void): Promise<number> {
  const { config } = input;
  try {
    if (config.population === undefined) throw new GenerationError("population: the config has no population section");
    calibrationAnchors(config);
    mkdirSync(input.out, { recursive: true });
    const result = await playCalibration({ engine: input.engine, config, pool: input.pool, belief_worlds: input.belief_worlds });
    writeCalibration(input.out, result);
    print(jcs({ generation: 0, run_config_hash: result.run_config_hash, anchors: result.anchors, games: result.games.length } as unknown as Json));
    return 0;
  } catch (error) {
    if (error instanceof GenerationError) {
      console.error(`play-generation: ${error.message}`);
      return 1;
    }
    throw error;
  }
}
