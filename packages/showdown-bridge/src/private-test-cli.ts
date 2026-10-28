// `scripts/private-test.ts`' logic: the Node child of the owner's private test
// command (`python -m areneon_ml.loop private-test`, ADR-015 erratum E1, item
// 2). It takes everything on the command line, reads the teams through a
// source port, starts the policy worker (CPU torch, stdio only) with the frozen
// pair, plays the matches and prints ONE canonical JSON line: the
// PrivateTestRecord and the wall-clock per match. It writes no file. The
// Python parent writes the record under the private root.
//
// Privacy: the owner-session guard's child check runs first (environment only:
// the parent holds the terminal check). Every failure is a fixed code and, for
// a team, a 1-based position in the source's order; no paste, species, hash,
// file name, validator text, engine text or stack ever reaches an output.
// Exit 0 ok, 1 a refusal or failure, 2 a usage error.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { type Json, jcs } from "./canonical-json.ts";
import { engineDataHash, readEngineData } from "./engine-data.ts";
import type { Engine } from "./engine.ts";
import { loadEngine } from "./engine.ts";
import { resolveFormat } from "./formats.ts";
import { damageAwareSetup } from "./league-generation.ts";
import { assertGuardedChild, OwnerSessionError } from "./owner-session.ts";
import { frozenPairClient, type FrozenPairClientOptions, pairFeatureSetup } from "./progress-cli.ts";
import type { PolicyClient } from "./policy-worker.ts";
import type { ProgressPool } from "./progress-evaluation.ts";
import { pairsAvailable, type PrivateTestTeam, playPrivateTest } from "./private-test.ts";
import { PasteDirectorySource } from "./private-team-source.ts";
import { privateTestAllowed } from "./private-test-record.ts";
import { checkSalt } from "./private-test-seed.ts";
import { importPaste, validateTeam } from "./teams.ts";
import { canonicalTeamHash } from "./team-records.ts";
import { readTrainingRunConfig, type TrainingRunConfig } from "./training-run-config.ts";
import { syntheticTeamHashes, teamHash, type SyntheticTeamHashes } from "./team-class.ts";

/** Every refusal of the private test, with its fixed words. A refusal may add a 1-based position, never text. */
export const PRIVATE_TEST_REFUSALS = {
  usage: "usage error",
  salt_unreadable: "the salt file cannot be read or is not 64 hex digits",
  source_unreadable: "the team source cannot be read",
  source_empty: "the team source holds no team",
  team_import_failed: "a team cannot be imported (position among the source's teams)",
  team_invalid: "a team is not legal in the format (position among the source's teams)",
  team_repeated: "a team repeats an earlier one (position among the source's teams)",
  snapshot_unverified: "the snapshot does not verify against the pinned engine and rules",
  too_few_teams: "the test set needs at least two teams",
  private_pair_source: "a pair with private ancestry is tested on hold-out teams of a snapshot or on a public dataset only",
  too_many_pairs: "more pairs asked for than the set gives",
  worker_failed: "the policy worker failed",
  play_failed: "the test could not be played",
} as const;
export type PrivateTestRefusalCode = keyof typeof PRIVATE_TEST_REFUSALS;

export class PrivateTestRefusal extends Error {
  readonly code: PrivateTestRefusalCode;
  readonly position: number | undefined;
  constructor(code: PrivateTestRefusalCode, position?: number) {
    super(`private-test: ${code}${position === undefined ? "" : ` ${position}`}`);
    this.name = "PrivateTestRefusal";
    this.code = code;
    this.position = position;
  }
}

/**
 * Teams from pastes, in the source's order: each through `importPaste` and `validateTeam` on a copy, then hashed.
 * A refusal names a code and the paste's 1-based position, never what the engine said.
 */
export function privateTestTeams(engine: Engine, formatId: string, pastes: Iterable<string>): PrivateTestTeam[] {
  const format = resolveFormat(engine, formatId);
  const teams: PrivateTestTeam[] = [];
  const seen = new Set<string>();
  let position = 0;
  for (const paste of pastes) {
    position++;
    let sets;
    try {
      sets = importPaste(engine, paste);
    } catch {
      throw new PrivateTestRefusal("team_import_failed", position);
    }
    // validateTeam normalises the sets in place, as every pool does.
    if (!validateTeam(engine, format, sets).valid) throw new PrivateTestRefusal("team_invalid", position);
    const canonical_team_hash = canonicalTeamHash(sets);
    if (seen.has(canonical_team_hash)) throw new PrivateTestRefusal("team_repeated", position);
    seen.add(canonical_team_hash);
    teams.push({ canonical_team_hash, sets });
  }
  return teams;
}

/** The staged directory (`PasteDirectorySource`) as teams; an unreadable directory is `source_unreadable`. */
export async function stagedTeams(engine: Engine, formatId: string, teamsDir: string): Promise<PrivateTestTeam[]> {
  const pastes: string[] = [];
  try {
    for await (const entry of new PasteDirectorySource(teamsDir).entries()) pastes.push(entry.paste);
  } catch {
    throw new PrivateTestRefusal("source_unreadable");
  }
  return privateTestTeams(engine, formatId, pastes);
}

/** What a team source hands the test. */
export interface TestTeams {
  teams: PrivateTestTeam[];
  source: { kind: "staging" } | { kind: "snapshot"; snapshot_hash: string } | { kind: "dataset"; dataset_hash: string };
  subset: "all" | "holdout";
  known_synthetic?: SyntheticTeamHashes;
}

/** The team sources outside the bridge (a snapshot's local copy, a public dataset): ports the scripts fill. */
export interface PrivateTestSources {
  snapshot: (args: { engine: Engine; config: TrainingRunConfig; snapshotsDir: string; snapshotHash: string; subset: "all" | "holdout" }) => Promise<TestTeams>;
  dataset: (args: { engine: Engine; config: TrainingRunConfig }) => ProgressPool;
}

/** Test seams: the default starts the real policy worker. */
export interface PrivateTestSeams {
  clientFor?: (args: FrozenPairClientOptions) => WorkerClient;
}
type WorkerClient = PolicyClient & { beginGame?: () => Promise<void>; start: () => Promise<unknown>; close: () => Promise<unknown> };

const FLAGS = {
  config: { type: "string" },
  generation: { type: "string" },
  preview: { type: "string" },
  battle: { type: "string" },
  "policy-id-sample": { type: "string" },
  "policy-id-argmax": { type: "string" },
  "generation-manifest-sha256": { type: "string" },
  "candidate-data-class": { type: "string" },
  "run-dir": { type: "string" },
  "engine-data": { type: "string" },
  "encoder-version": { type: "string" },
  "worker-python": { type: "string" },
  "salt-file": { type: "string" },
  source: { type: "string" },
  subset: { type: "string" },
  "teams-dir": { type: "string" },
  "snapshots-dir": { type: "string" },
  "snapshot-hash": { type: "string" },
  "max-pairs": { type: "string" },
  "damage-aware-budget-ms": { type: "string" },
  // Another frozen pair as an opponent: `--opponent-pair <preview>,<battle>,<policy-id-sample>,<policy-id-argmax>`
  // with its run directory (read-only: its checkpoints are the worker's) and encoder version.
  "opponent-pair": { type: "string" },
  "opponent-run-dir": { type: "string" },
  "opponent-encoder-version": { type: "string" },
} as const;
const OPTIONAL = new Set<string>(["teams-dir", "snapshots-dir", "snapshot-hash", "max-pairs", "damage-aware-budget-ms", "opponent-pair", "opponent-run-dir", "opponent-encoder-version"]);
const HEX64 = /^[0-9a-f]{64}$/;

const usage = (): never => {
  throw new PrivateTestRefusal("usage");
};
const integer = (text: string | undefined, least: number): number => {
  const value = Number(text);
  return text === undefined || !/^\d+$/.test(text) || !Number.isSafeInteger(value) || value < least ? usage() : value;
};

/** Run the private test; prints one JSON line with `print` and returns the process exit code. */
export async function runPrivateTestCli(argv: readonly string[], print: (line: string) => void, sources: PrivateTestSources, report: (line: string) => void, seams: PrivateTestSeams = {}): Promise<number> {
  try {
    assertGuardedChild();
    return await run(argv, print, sources, seams);
  } catch (error) {
    if (error instanceof PrivateTestRefusal) {
      report(error.message);
      return error.code === "usage" ? 2 : 1;
    }
    if (error instanceof OwnerSessionError) {
      report(error.message);
      return 1;
    }
    // Anything else may quote a team (an engine, parser or stack text): only its absence of detail is shown.
    report("private-test: failed");
    return 1;
  }
}

async function run(argv: readonly string[], print: (line: string) => void, sources: PrivateTestSources, seams: PrivateTestSeams): Promise<number> {
  let values: Record<keyof typeof FLAGS, string | undefined>;
  try {
    values = parseArgs({ args: [...argv], options: FLAGS, strict: true }).values as typeof values;
  } catch {
    return usage();
  }
  const need = (name: keyof typeof FLAGS): string => values[name] ?? usage();
  for (const name of Object.keys(FLAGS) as (keyof typeof FLAGS)[]) if (!OPTIONAL.has(name)) need(name);
  const pairFlags = ["opponent-pair", "opponent-run-dir", "opponent-encoder-version"] as const;
  const pairGiven = pairFlags.filter((name) => values[name] !== undefined);
  if (pairGiven.length !== 0 && pairGiven.length !== pairFlags.length) usage();
  const pairParts = values["opponent-pair"]?.split(",");
  if (pairParts !== undefined && (pairParts.length !== 4 || !HEX64.test(pairParts[0] as string) || !HEX64.test(pairParts[1] as string))) usage();
  const generation = integer(values.generation, 0);
  const maxPairs = values["max-pairs"] === undefined ? undefined : integer(values["max-pairs"], 1);
  const damageBudget = values["damage-aware-budget-ms"] === undefined ? undefined : integer(values["damage-aware-budget-ms"], 1);
  const subset = need("subset");
  const sourceKind = need("source");
  const candidateClass = need("candidate-data-class");
  if (subset !== "all" && subset !== "holdout") return usage();
  if (candidateClass !== "synthetic" && candidateClass !== "private") return usage();
  if (!["staging", "snapshot", "dataset"].includes(sourceKind)) usage();
  if (candidateClass === "private" && !privateTestAllowed(sourceKind, subset)) throw new PrivateTestRefusal("private_pair_source");

  let salt: string;
  try {
    salt = readFileSync(need("salt-file"), "utf8").trim();
    checkSalt(salt);
  } catch {
    throw new PrivateTestRefusal("salt_unreadable");
  }

  let config: TrainingRunConfig;
  let engine: Engine;
  let data: ReturnType<typeof readEngineData>;
  try {
    config = readTrainingRunConfig(JSON.parse(readFileSync(need("config"), "utf8")));
    engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
    data = readEngineData(JSON.parse(readFileSync(need("engine-data"), "utf8")));
  } catch {
    throw new PrivateTestRefusal("usage");
  }

  // The owner-session guard has passed (the parent holds the terminal check): the teams are read now.
  let loaded: TestTeams;
  if (sourceKind === "staging") {
    if (subset !== "all") usage();
    loaded = { teams: await stagedTeams(engine, config.format_id, values["teams-dir"] ?? usage()), source: { kind: "staging" }, subset: "all" };
  } else if (sourceKind === "snapshot") {
    const snapshotHash = values["snapshot-hash"] ?? usage();
    if (!HEX64.test(snapshotHash)) usage();
    loaded = await sources.snapshot({ engine, config, snapshotsDir: values["snapshots-dir"] ?? usage(), snapshotHash, subset });
  } else {
    if (subset !== "all") usage();
    let pool: ProgressPool;
    try {
      pool = sources.dataset({ engine, config });
    } catch {
      throw new PrivateTestRefusal("source_unreadable");
    }
    loaded = {
      teams: pool.teams.map((team) => ({ canonical_team_hash: team.canonical_team_hash, sets: team.sets })),
      source: { kind: "dataset", dataset_hash: pool.dataset_hash },
      subset: "all",
      known_synthetic: syntheticTeamHashes(pool.teams.map((team) => teamHash(team.sets))),
    };
  }
  if (loaded.teams.length < 2) throw new PrivateTestRefusal("too_few_teams");
  if (maxPairs !== undefined && maxPairs > pairsAvailable(loaded.teams.length)) throw new PrivateTestRefusal("too_many_pairs");

  const dataFile = need("engine-data");
  const dataHash = engineDataHash(data);
  const { max_line_bytes, handshake_timeout_ms } = config.generation;
  // The candidate features of the pair (and of the opponent pair, from its own run config); none: nothing changes.
  let features: ReturnType<typeof pairFeatureSetup>;
  let opponentFeatures: ReturnType<typeof pairFeatureSetup>;
  try {
    features = pairFeatureSetup(engine, config);
    if (pairParts !== undefined) opponentFeatures = pairFeatureSetup(engine, readTrainingRunConfig(JSON.parse(readFileSync(join(need("opponent-run-dir"), "run-config.json"), "utf8"))));
  } catch {
    throw new PrivateTestRefusal("worker_failed");
  }
  const clientFor = (pair: { preview: string; battle: string }, runDir: string, encoderVersion: string, setup: NonNullable<typeof features> | null = features ?? null): WorkerClient => {
    const asked = { python: need("worker-python"), runDir, pair, encoderVersion, engineDataFile: dataFile, engineDataHash: dataHash, max_line_bytes, handshake_timeout_ms, ...(setup === null ? {} : { features: setup.statement }) };
    return seams.clientFor === undefined ? frozenPairClient(asked) : seams.clientFor(asked);
  };
  const checkpoints = { preview: need("preview"), battle: need("battle") };
  const client = clientFor(checkpoints, need("run-dir"), need("encoder-version"));
  const opponentCheckpoints = pairParts === undefined ? undefined : { preview: pairParts[0] as string, battle: pairParts[1] as string };
  const opponentClient = opponentCheckpoints === undefined ? undefined : clientFor(opponentCheckpoints, need("opponent-run-dir"), need("opponent-encoder-version"), opponentFeatures ?? null);
  try {
    try {
      await client.start();
      await opponentClient?.start();
    } catch {
      throw new PrivateTestRefusal("worker_failed");
    }
    let result;
    try {
      result = await playPrivateTest({
        engine,
        config,
        generation,
        generation_manifest_sha256: need("generation-manifest-sha256"),
        checkpoints,
        policy_ids: { sample: need("policy-id-sample"), argmax: need("policy-id-argmax") },
        candidate_data_class: candidateClass,
        client,
        ...(features === undefined ? {} : { features: features.configs }),
        ...(opponentClient === undefined || opponentCheckpoints === undefined || pairParts === undefined
          ? {}
          : { opponent_pair: { checkpoints: opponentCheckpoints, policy_ids: { sample: pairParts[2] as string, argmax: pairParts[3] as string }, client: opponentClient, ...(opponentFeatures === undefined ? {} : { features: opponentFeatures.configs }) } }),
        ...(damageBudget === undefined ? {} : { damage_aware: { setup: damageAwareSetup(engine, config.format_id), match_budget_ms: damageBudget } }),
        teams: loaded.teams,
        source: loaded.source,
        subset: loaded.subset,
        salt,
        ...(maxPairs === undefined ? {} : { max_pairs: maxPairs }),
        ...(loaded.known_synthetic === undefined ? {} : { known_synthetic: loaded.known_synthetic }),
      });
    } catch {
      throw new PrivateTestRefusal("play_failed"); // an engine, worker or harness text may quote a team: only the code leaves
    }
    print(jcs({ record: result.record, wall_ms: result.wall_ms } as unknown as Json));
    return 0;
  } finally {
    await Promise.allSettled([client.close(), ...(opponentClient === undefined ? [] : [opponentClient.close()])]);
  }
}
