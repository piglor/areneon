// What the search-actor tests share: a stand-in policy worker (no network, no Python) whose prior and value are fixed pure
// functions of the observation, a game runner that plays a searching seat against a first-legal seat, and the real opponent
// model behind the search's port. The value is the hit-point differential of the observation's own log read by the shared
// reader, so it moves with the damage a world's sets deal: a test of the search has teeth only if the leaf does.
import { createHash } from "node:crypto";
import { type PublicState, interpretChannel, readerSha256 } from "../../public-state/src/index.ts";
import {
  battleClass,
  type EvaluationClient,
  type Engine,
  goldenWorldFactsData,
  type PolicyClient,
  type PokemonSet,
  type PriorKind,
  resolveFormat,
  runBattle,
  type Actor,
  searchActor,
  type SearchActor,
  type SearchActorOptions,
  searchPolicy,
  searchSetup,
  type SearchSetup,
  type SearchSetupInput,
  stampTeam,
  syntheticTeamHashes,
  teamHash,
  type WorldSamplePort,
  loadRulesSnapshot,
} from "@areneon/showdown-bridge";
import { firstLegal, goldenTeamVocabulary } from "../../showdown-bridge/src/golden.ts";
import type { RandomBattleFile } from "../../showdown-bridge/src/random-corpus.ts";
import { createSearchWorlds } from "../src/search-worlds.ts";
import { engine, FORMAT } from "./support.ts";

export const vocabulary = goldenTeamVocabulary();
export const snapshot = loadRulesSnapshot(engine.commit);
export const manifest = resolveFormat(engine, FORMAT, snapshot).manifest;

const sum = (state: PublicState, side: "own" | "opponent"): number => {
  const roster = state[side].roster;
  return roster.reduce((total, mon) => total + (mon.hp_num !== null && mon.hp_den ? Math.floor((mon.hp_num * 1_000_000) / mon.hp_den) : mon.fainted ? 0 : 1_000_000), 0) / Math.max(1, roster.length);
};

/** The stand-in worker: a prior from a hash of each choice, a value from the hit-point difference in the observation's own log. */
export class StandInWorker implements PolicyClient, EvaluationClient {
  readonly featureSet = null;
  readonly queries: { kind: string; log_lines: number }[] = [];
  async ask(_battle: string, query: Parameters<PolicyClient["ask"]>[1]): ReturnType<PolicyClient["ask"]> {
    const open = query.candidates.filter((candidate) => !candidate.pending);
    const choice = (open[0] ?? query.candidates[0]).choice;
    this.queries.push({ kind: "ask", log_lines: 0 });
    return { choice } as Awaited<ReturnType<PolicyClient["ask"]>>;
  }
  async evaluate(_battle: string, query: Parameters<EvaluationClient["evaluate"]>[1]): ReturnType<EvaluationClient["evaluate"]> {
    const observation = query.observation as { player: string; log?: readonly string[] };
    this.queries.push({ kind: "evaluate", log_lines: observation.log?.length ?? 0 });
    const prior_ppm = query.candidates.map(({ choice }) => 1 + (createHash("sha256").update(choice).digest().readUInt16BE(0) % 1000));
    let value_ppm = 500_000;
    if (observation.log !== undefined) {
      const state = interpretChannel(observation.player, observation.log).frames.at(-1)?.state;
      if (state !== undefined) value_ppm = Math.round(500_000 + (sum(state, "own") - sum(state, "opponent")) / 2);
    }
    return { identity: query.identity, prior_ppm, value_ppm } as Awaited<ReturnType<EvaluationClient["evaluate"]>>;
  }
}

export const portFor = (arm: PriorKind = "generator-matched"): WorldSamplePort => createSearchWorlds({ engine, vocabulary, reader_sha256: readerSha256(), manifest, arm });

export const setupFor = (port: WorldSamplePort = portFor(), change: Partial<SearchSetupInput> = {}): SearchSetup =>
  searchSetup({ engine, format_id: FORMAT, vocabulary, port, facts_data: goldenWorldFactsData(), data_class: "synthetic", snapshot, ...change });

export interface SearchGame {
  record: Awaited<ReturnType<typeof runBattle>>;
  actor: SearchActor;
  worker: StandInWorker;
}

/** The searching seat (p1 unless `seat` says p2) against a first-legal seat on the corpus battle's seed and teams (the opponent's team can be changed). */
export async function playSearch(
  battle: RandomBattleFile,
  change: {
    seat?: "p1" | "p2";
    p2?: PokemonSet[];
    p1?: PokemonSet[];
    seed?: string;
    setup?: SearchSetup;
    maxTurns?: number;
    options?: Partial<SearchActorOptions>;
    worker?: StandInWorker;
    /** Wraps the searching actor (a test that changes what it is asked). */
    wrap?: (actor: SearchActor) => Actor;
  } = {},
): Promise<SearchGame> {
  const seat = change.seat ?? "p1";
  const teams = { p1: change.p1 ?? battle.teams.p1, p2: change.p2 ?? battle.teams.p2 };
  const seed = change.seed ?? battle.seed;
  const setup = change.setup ?? setupFor();
  const worker = change.worker ?? new StandInWorker();
  const known = syntheticTeamHashes([teamHash(teams.p1), teamHash(teams.p2)]);
  const pair = { preview: "a".repeat(64), battle: "b".repeat(64) };
  const { policy_id, identity_hash } = searchPolicy(setup, manifest, pair, "argmax");
  const actor = searchActor({
    setup,
    client: worker,
    battle: `game-0-${seat}`,
    player: seat,
    team: stampTeam(teams[seat] as never, known),
    manifest,
    channel_class: battleClass({ p1: teamHash(teams.p1), p2: teamHash(teams.p2) }, known),
    seed,
    game: 0,
    policy_id,
    identity_hash,
    sampling: "argmax",
    ...change.options,
  });
  const mine = change.wrap === undefined ? actor : change.wrap(actor);
  const actors = seat === "p1" ? { p1: mine, p2: firstLegal } : { p1: firstLegal, p2: mine };
  const record = await runBattle({ engine: engine as Engine, formatId: FORMAT, seed, teams, actors, maxTurns: change.maxTurns ?? 3 });
  return { record, actor, worker };
}
