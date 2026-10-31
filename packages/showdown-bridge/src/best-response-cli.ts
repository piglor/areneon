// `scripts/best-response.ts`: the E0-P best-response evaluation for the Python command (ml/areneon_ml/progress/best_response.py).
// It starts the policy worker with the frozen pair, plays the four arms of every validation pair from both seats and writes
// `best-response.json` into `--out`. It prints one canonical JSON line (the record hash, the games and the wall-clock). Exit 0
// ok, 1 an evaluation that failed or was refused, 2 a usage or configuration error. Nothing is retried or committed here: the
// Python side makes the directory real and logs the look.
//
// Evaluation-only: the responder has white-box access to the evaluated pair's weights (ADR-029 section 2). This is the one entry
// point of `best-response.ts`; nothing else imports it (a source scan).

import { closeSync, fsyncSync, mkdirSync, openSync, readFileSync, writeSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { bestResponseRecordHash } from "./best-response-record.ts";
import { BestResponseError, playBestResponse, type PlayMode, type ResponderSettings, type ResponderTeamPort } from "./best-response.ts";
import { type Json, jcs } from "./canonical-json.ts";
import { engineDataHash, readEngineDataAny } from "./engine-data.ts";
import { EVAL_DATASET_FLAGS, readEvalDataset, withEvalDataset } from "./eval-dataset.ts";
import { loadEngine } from "./engine.ts";
import type { Engine } from "./engine.ts";
import { UsageError } from "./generation-cli.ts";
import { goldenTeamVocabulary } from "./golden.ts";
import { committedPlanSeeds, frozenPairClient, pairFeatureSetup, type ProgressPorts, type ProgressPoolSource, readTeamSet } from "./progress-cli.ts";
import { ProgressError } from "./progress-evaluation.ts";
import { readTrainingRunConfig } from "./training-run-config.ts";

export { UsageError };

/** The ports a composition root fills: the belief's full-team draw and the run's own teams (the bridge imports neither the opponent model nor the dataset reader). */
export interface BestResponsePorts extends Pick<ProgressPorts, "run_teams"> {
  teams?: (engine: Engine, formatId: string) => ResponderTeamPort;
}

/** The record the command writes into `--out`. */
export const BEST_RESPONSE_FILE = "best-response.json";

const FLAGS = {
  config: { type: "string" },
  generation: { type: "string" },
  preview: { type: "string" },
  battle: { type: "string" },
  mode: { type: "string" },
  "policy-id": { type: "string" },
  "generation-manifest-sha256": { type: "string" },
  "team-set": { type: "string" },
  "run-dir": { type: "string" },
  "engine-data": { type: "string" },
  "encoder-version": { type: "string" },
  "worker-python": { type: "string" },
  pairs: { type: "string" },
  worlds: { type: "string" },
  "k-own": { type: "string" },
  "own-rule": { type: "string" },
  leaf: { type: "string" },
  "preview-budget-ms": { type: "string" },
  out: { type: "string" },
  ...EVAL_DATASET_FLAGS,
  "evaluation-key": { type: "string" },
} as const;
const OPTIONAL = new Set<string>(["eval-dataset-dir", "eval-dataset-hash", "evaluation-key"]);

const integer = (name: string, text: string, least: number): number => {
  const value = Number(text);
  if (!Number.isSafeInteger(value) || value < least || !/^\d+$/.test(text)) throw new UsageError(`--${name} must be an integer of at least ${least}`);
  return value;
};

/** `first-request` or `rollout:<turns>`. */
export function parseLeaf(text: string): ResponderSettings["leaf"] {
  if (text === "first-request") return { kind: "first-request" };
  const match = /^rollout:(\d+)$/.exec(text);
  if (match === null || Number(match[1]) < 1) throw new UsageError("--leaf must be first-request or rollout:<turns> with at least one turn");
  return { kind: "rollout", turns: Number(match[1]) };
}

/** Run one best-response evaluation; returns the process exit code and writes its one JSON line with `print`. */
export async function runBestResponseCli(argv: readonly string[], print: (line: string) => void, poolSource: ProgressPoolSource, ports: BestResponsePorts = {}): Promise<number> {
  let values: Record<keyof typeof FLAGS, string | undefined>;
  try {
    values = parseArgs({ args: [...argv], options: FLAGS, strict: true }).values as typeof values;
  } catch (error) {
    throw new UsageError((error as Error).message);
  }
  const need = (name: keyof typeof FLAGS): string => values[name] ?? ((): never => { throw new UsageError(`--${name} is required`); })();
  for (const name of Object.keys(FLAGS) as (keyof typeof FLAGS)[]) if (!OPTIONAL.has(name)) need(name);
  const mode = need("mode");
  if (mode !== "argmax" && mode !== "sample") throw new UsageError("--mode must be argmax or sample");
  const ownRule = need("own-rule");
  if (ownRule !== "prior-top-k" && ownRule !== "distinct-leads") throw new UsageError("--own-rule must be prior-top-k or distinct-leads");
  const settings: ResponderSettings = { worlds: integer("worlds", need("worlds"), 1), k_own: integer("k-own", need("k-own"), 1), own_rule: ownRule, leaf: parseLeaf(need("leaf")) };
  const pairs = integer("pairs", need("pairs"), 1);
  const generation = integer("generation", need("generation"), 0);
  const previewBudget = integer("preview-budget-ms", need("preview-budget-ms"), 1);
  if (ports.teams === undefined || ports.run_teams === undefined) throw new UsageError("this command has no belief port or reader of the run's own teams: it needs the team-dataset script");
  const config = readTrainingRunConfig(JSON.parse(readFileSync(need("config"), "utf8")));
  if (config.seeds.run_key.startsWith("CHANGE-ME")) throw new UsageError("seeds.run_key is still the CHANGE-ME placeholder: give the run its own name");
  if (config.team_dataset === undefined) {
    console.error("best-response: the run config names no team dataset: an evaluation needs the validation split of one");
    return 1;
  }
  const evalDataset = readEvalDataset(values);
  if (evalDataset !== undefined && evalDataset.dataset_hash === config.team_dataset.dataset_hash) throw new UsageError("--eval-dataset-hash is the run's own training dataset: leave the option out to evaluate on its validation split");
  const teamSet = readTeamSet(need("team-set"));
  const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
  if (pairFeatureSetup(engine, config) !== undefined) throw new UsageError("features_pair: a pair that reads candidate features is not supported by the responder");
  const runTeams = evalDataset === undefined ? new Set<string>() : ports.run_teams(engine, config);
  const dataFile = need("engine-data");
  const dataHash = engineDataHash(readEngineDataAny(JSON.parse(readFileSync(dataFile, "utf8"))));
  const { max_line_bytes, handshake_timeout_ms } = config.generation;
  const preview = need("preview");
  const battle = need("battle");
  const client = frozenPairClient({ python: need("worker-python"), runDir: need("run-dir"), pair: { preview, battle }, encoderVersion: need("encoder-version"), engineDataFile: dataFile, engineDataHash: dataHash, max_line_bytes, handshake_timeout_ms });
  const started = performance.now();
  try {
    const pool = poolSource(engine, withEvalDataset(config, evalDataset));
    await client.start();
    const record = await playBestResponse({
      engine,
      config,
      pool,
      pairs,
      evaluated: { generation, generation_manifest_sha256: need("generation-manifest-sha256"), checkpoints: { preview, battle }, mode: mode as PlayMode, policy_id: need("policy-id"), client },
      responder: { settings, teams: ports.teams(engine, config.format_id), vocabulary: goldenTeamVocabulary(), preview_budget_ms: previewBudget },
      training_teams: new Set([...teamSet, ...runTeams]),
      league_seeds: committedPlanSeeds(need("run-dir")),
      ...(values["evaluation-key"] === undefined ? {} : { evaluation_key: values["evaluation-key"] }),
    });
    mkdirSync(need("out"), { recursive: true });
    const descriptor = openSync(join(need("out"), BEST_RESPONSE_FILE), "wx");
    try {
      writeSync(descriptor, `${jcs(record as unknown as Json)}\n`);
      fsyncSync(descriptor);
    } finally {
      closeSync(descriptor);
    }
    print(jcs({ record_hash: bestResponseRecordHash(record), games: record.plan.games, wall_ms: Math.round(performance.now() - started) } as unknown as Json));
    return 0;
  } catch (error) {
    if (error instanceof BestResponseError || error instanceof ProgressError) {
      console.error(`best-response: ${error.message}`);
      return 1;
    }
    throw error;
  } finally {
    await client.close();
  }
}
