// The composition root of the differential harness at level 2, the battle step (ADR-030 groundwork G2, ADR-028 section 5,
// #1462): it joins the engine-free core (`@areneon/differential`: the port, the comparison, the entity rule, the interaction
// classes, the tally) with the bridge's engine side (`recordBattleTrace` on the unmodified engine, `showdownBattleBackend`, the
// committed battles), which does not import the core. It also draws the fresh corpus: seeded battles on generator-pool teams
// (dev and training splits, never validation) across the `uniform-all`, `random-legal` and `pending-first` mixes, seeded by
// G1's fresh-corpus rule. Evaluation code never imports this module.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { freshSeed, sha256Jcs } from "@areneon/differential";
import {
  addBattles,
  type BattleBackend,
  type BattleCorpus,
  type BattleTrace,
  type InteractionClasses,
  newBattleCorpus,
  readBattleTrace,
  readInteractionClasses,
} from "@areneon/differential/battle";
import {
  type Engine,
  type GenerationTeamPool,
  leagueGameStreams,
  type PokemonSet,
  recordBattleTrace,
  REPLAY_FORMAT,
  showdownBattleBackend,
  type TraceSpec,
} from "@areneon/showdown-bridge";
import { type SweepMix, sweepActors } from "./reducer-sweep.ts";

const ROOT = fileURLToPath(new URL("../../..", import.meta.url));

/** The reviewed interaction classes (config/differential/battle-interaction-classes.v1.json). */
export const CLASSES_FILE = join(ROOT, "config", "differential", "battle-interaction-classes.v1.json");

export const loadInteractionClasses = (): InteractionClasses => readInteractionClasses(JSON.parse(readFileSync(CLASSES_FILE, "utf8")));

/** The pinned Showdown as a battle backend of the core's port (the bridge declares the same shape structurally). */
export const battleBackend = (engine: Engine): BattleBackend => showdownBattleBackend(engine);

/** Record a spec and read the trace back through the core's reader (schema and reader rules). */
export async function traceOf(engine: Engine, spec: TraceSpec): Promise<BattleTrace> {
  const { trace } = await recordBattleTrace(engine, spec);
  return readBattleTrace(JSON.parse(JSON.stringify(trace)));
}

/** The fresh corpus's actor mixes (ADR-028 section 5), in the order battle `i` takes them (`i mod 3`). */
export const FRESH_MIXES: readonly SweepMix[] = ["uniform-all", "random-legal", "pending-first"];

/** The fresh seed of the Showdown battle backend at this engine (G1's rule: no fresh corpus before the backend version exists). */
export const battleFreshSeed = (engine: Engine): string => freshSeed({ backend_commit: engine.commit, backend_format: REPLAY_FORMAT });

/**
 * Battle `index` of the fresh corpus: its mix, then seed, teams and actor seeds from `leagueGameStreams(<fresh seed>/<mix>, 1,
 * index)` (the reducer sweep's draw), on `pool` (the loop's own pool function: dev and training splits only).
 */
export function freshSpec(seed: string, pool: GenerationTeamPool, index: number, formatId: string, maxTurns: number): TraceSpec {
  const mix = FRESH_MIXES[index % FRESH_MIXES.length] as SweepMix;
  const streams = leagueGameStreams(`${seed}/${mix}`, 1, index);
  const teams = {
    p1: pool.teams[streams.team_draws.p1 % pool.teams.length] as PokemonSet[],
    p2: pool.teams[streams.team_draws.p2 % pool.teams.length] as PokemonSet[],
  };
  const actors = sweepActors(mix, { p1: streams.baseline_seed, p2: (streams.baseline_seed ^ 0x9e3779b9) >>> 0 });
  return { battle_id: `fresh/${mix}/${index}`, format_id: formatId, seed: streams.seed, teams, actors: { p1: actors.p1.actor, p2: actors.p2.actor }, max_turns: maxTurns };
}

/** The identity of a pool: SHA-256 over the JCS of its team hashes, in pool order. */
export const poolHash = (pool: GenerationTeamPool): string => sha256Jcs([...pool.hashes]);

/** Compare `traces` with `backend` into `corpus`, in batches of `batchSize` battles. */
export function compareTraces(classes: InteractionClasses, traces: readonly BattleTrace[], corpus: BattleCorpus, backend: BattleBackend, batchSize = 16): BattleCorpus {
  addBattles(corpus, classes, backend, traces, batchSize);
  return corpus;
}

export { newBattleCorpus };
