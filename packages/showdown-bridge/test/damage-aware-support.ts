// Shared setup for the damage-aware baseline's tests: the pinned engine, the committed config records,
// real requests captured from seeded battles, and a probe stand-in whose answers a test controls.
import { interpretChannel, type PublicState } from "@areneon/public-state";
import { type DamageAnswer, type SpeedOrderAnswer } from "../src/damage-aware-score.ts";
import { damageAwareActor, type DamageAwareOptions } from "../src/damage-aware-actor.ts";
import { loadDamageAwareConfig } from "../src/damage-aware-config.ts";
import type { Engine, PokemonSet } from "../src/engine.ts";
import { GOLDEN_FORMAT, goldenTeamVocabulary } from "../src/golden.ts";
import { loadEngine, loadRulesSnapshot, resolveFormat } from "../src/index.ts";
import { loadMechanicsConfig, mechanicsConfigHash } from "../src/mechanics-config.ts";
import type { MechanicsContext, ProbeOutcome } from "../src/mechanics-probe.ts";
import { mechanicsQueryHash, type ParsedMechanicsQuery } from "../src/mechanics-query.ts";
import { buildResult, type Outcome } from "../src/mechanics-result.ts";
import { loadRandomCorpus, type RandomBattleFile } from "../src/random-corpus.ts";
import { randomLegalActor } from "../src/random-legal.ts";
import { type Actor, type ActorContext, type Player, type RequestRecord, runBattle } from "../src/runner.ts";

export const engine: Engine = loadEngine();
export const vocabulary = goldenTeamVocabulary();
export const config = loadDamageAwareConfig("v1", vocabulary);
export const mechanics = loadMechanicsConfig(GOLDEN_FORMAT);
export const snapshot = loadRulesSnapshot(engine.commit);
export const corpus: RandomBattleFile[] = loadRandomCorpus();
const manifest = resolveFormat(engine, GOLDEN_FORMAT, snapshot).manifest;

export const options = (change: Partial<DamageAwareOptions> = {}): DamageAwareOptions => ({ engine, vocabulary, config, mechanics, snapshot, data_class: "synthetic", ...change });
export const newActor = (change: Partial<DamageAwareOptions> = {}) => damageAwareActor(options(change));

export interface Captured {
  request: RequestRecord;
  context: ActorContext;
  battle: RandomBattleFile;
}

/** Every request `player` is asked in a seeded battle played by random-legal, with the context it would get. */
export async function capture(battle: RandomBattleFile, player: Player = "p1", maxTurns = 12): Promise<Captured[]> {
  const out: Captured[] = [];
  const recorder = (seed: number): Actor => {
    const play = randomLegalActor(seed);
    return (request, context) => {
      out.push({ request: structuredClone(request), context: { ...(context as ActorContext), channel: [...(context as ActorContext).channel] }, battle });
      return play(request, context);
    };
  };
  const other = randomLegalActor(battle.actor_seeds[player === "p1" ? "p2" : "p1"]);
  const actors = player === "p1" ? { p1: recorder(battle.actor_seeds.p1), p2: other } : { p1: other, p2: recorder(battle.actor_seeds.p2) };
  await runBattle({ engine, formatId: GOLDEN_FORMAT, seed: battle.seed, teams: battle.teams, actors, maxTurns });
  return out;
}

export const stateOf = (captured: Captured): PublicState => {
  const run = interpretChannel(captured.context.player, captured.context.channel);
  return (run.frames[run.frames.length - 1] as { state: PublicState }).state;
};

const rolls = (value: number): DamageAnswer["no_crit"] => Array.from({ length: 16 }, () => value) as DamageAnswer["no_crit"];

export interface StubOptions {
  /** A constant roll for each move id (default 0); the target has 100 hit points, so a roll below 100 is `roll * 10,000` ppm. */
  values?: Record<string, number>;
  /** Whether the foe's action precedes the actor's in a `speed_order` query. */
  foeFirst?: boolean;
  /** Make the probe answer these query kinds with this outcome instead. */
  override?: Partial<Record<"damage" | "stats" | "speed_order", Outcome>>;
  log?: ParsedMechanicsQuery[];
  /** Make the probe throw this for these query kinds. */
  throwOn?: Partial<Record<"damage" | "stats" | "speed_order", Error>>;
}

/** A probe stand-in: damage answers a constant roll per move, every set has 100 hit points, and the order is the test's. */
export function stubProbe(stub: StubOptions = {}): (context: MechanicsContext, query: ParsedMechanicsQuery) => ProbeOutcome {
  return (_context, query) => {
    stub.log?.push(query);
    const kind = query.query.kind;
    const failure = stub.throwOn?.[kind];
    if (failure !== undefined) throw failure;
    const identity = { data_class: query.data_class, query_hash: mechanicsQueryHash(query), manifest, mechanics_config_hash: mechanicsConfigHash(mechanics), gap_ids: [] };
    const forced = stub.override?.[kind];
    let outcome: Outcome;
    if (forced !== undefined) outcome = forced;
    else if (query.query.kind === "damage") {
      const roll = stub.values?.[query.query.move] ?? 0;
      outcome = {
        kind: "computed",
        answer: { kind: "damage", no_crit: rolls(roll), crit: rolls(roll), crit_chance: { kind: "never" }, hp_current: 100, hp_max: 100, ko: { kind: "unsupported", reason: "ko_state_change" } },
      };
    } else if (query.query.kind === "stats") {
      outcome = { kind: "computed", answer: { kind: "stats", stats: { hp: 100, atk: 50, def: 50, spa: 50, spd: 50, spe: 50 } } };
    } else {
      const actions = query.query.actions.map((action) => ({ ...action, order: 0, priority: 0, speed: 100 })) as unknown as SpeedOrderAnswer["actions"];
      outcome = { kind: "computed", answer: { kind: "speed_order", order_basis: "turn_start", actions, groups: stub.foeFirst === true ? [[1], [0]] : [[0], [1]] } };
    }
    return { result: buildResult(identity, outcome), diagnostics: [] };
  };
}

export const toId = (value: string): string => value.toLowerCase().replace(/[^a-z0-9]/g, "");
export type { PokemonSet };
