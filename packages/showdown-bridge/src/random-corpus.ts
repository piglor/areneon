// The random-play channel corpus (ADR-010 §3, R10): seeded random-legal vs
// random-legal battles on random synthetic teams, each player's own channel
// kept as a PlayerChannelLog. It exists so the Python log-to-state reducer is
// tried on the lines the engine emits in ordinary play, not only on the one
// golden battle. CI replays each committed spec and compares (the teams are
// found once by `scripts/capture-random-corpus.ts`, like the choice corpus).
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type { PlayerChannelLog } from "@areneon/schema/types/v1/player-channel-log.ts";
import type { Engine, PokemonSet } from "./engine.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "./golden.ts";
import { NoCandidateError } from "./actor-rule.ts";
import { randomLegal, randomLegalActor } from "./random-legal.ts";
import { type Actor, type BattleRecord, type Player, runBattle } from "./runner.ts";

export const RANDOM_DIR = join(GOLDEN_DIR, "random");
export const RANDOM_MANIFEST = "manifest.json";
/**
 * Battles in which a player submitted a pending candidate the engine refused as
 * `[Unavailable choice]` (ADR-006 rule 5): `|error|` lines in a player's own
 * channel, which random-legal play never reaches.
 */
export const UNAVAILABLE_DIR = join(GOLDEN_DIR, "unavailable");

/** Battles are cut here (a forced tie) so the committed channels stay small. */
export const RANDOM_MAX_TURNS = 30;

export const RANDOM_PROVENANCE =
  "synthetic: random-legal vs random-legal on random teams drawn from the pinned engine's learnsets";

/** What a battle needs to be replayed. */
export interface RandomBattleSpec {
  id: string;
  seed: string;
  actor_seeds: Record<Player, number>;
  teams: Record<Player, PokemonSet[]>;
}

export interface RandomBattleFile extends RandomBattleSpec {
  schema_version: 1;
  data_class: "synthetic";
  provenance: string;
  winner: Player | null;
  turns: number;
  channels: Record<Player, PlayerChannelLog>;
}

/**
 * Like `randomLegalActor`, but plays a pending candidate whenever the set has
 * one (the opposite preference), so the engine may refuse it as
 * `[Unavailable choice]`. Same seeding rules as `randomLegalActor`.
 */
export function pendingFirstActor(seed: number): Actor {
  const fallback = randomLegalActor(seed);
  let state = seed;
  return (request, context) => {
    const legal = request.legal;
    if (legal === undefined) {
      throw new NoCandidateError(`${request.player}: request ${request.request_seq} has no legal choice set`);
    }
    if (legal.pending.length === 0) return fallback(request, context);
    const pick = randomLegal({ choices: [], pending: legal.pending }, state);
    state = pick.state;
    return pick.choice;
  };
}

export function playRandomBattle(
  engine: Engine,
  spec: RandomBattleSpec,
  actorFor: (seed: number) => Actor = randomLegalActor,
): Promise<BattleRecord> {
  return runBattle({
    engine,
    formatId: GOLDEN_FORMAT,
    seed: spec.seed,
    teams: spec.teams,
    actors: {
      p1: actorFor(spec.actor_seeds.p1),
      p2: actorFor(spec.actor_seeds.p2),
    },
    maxTurns: RANDOM_MAX_TURNS,
  });
}

/** The committed file of one played battle. Its teams are synthetic by construction. */
export function randomBattleFile(spec: RandomBattleSpec, record: BattleRecord): RandomBattleFile {
  const channel = (player: Player): PlayerChannelLog => ({
    schema_version: 1,
    data_class: "synthetic",
    player,
    lines: record.channels[player],
  });
  return {
    schema_version: 1,
    data_class: "synthetic",
    provenance: RANDOM_PROVENANCE,
    ...spec,
    winner: record.winner,
    turns: record.turns,
    channels: { p1: channel("p1"), p2: channel("p2") },
  };
}

/** Every committed battle file, in file-name order. */
export function loadRandomCorpus(dir: string = RANDOM_DIR): RandomBattleFile[] {
  return readdirSync(dir)
    .filter((name) => /^b\d+\.json$/.test(name))
    .sort()
    .map((name) => JSON.parse(readFileSync(join(dir, name), "utf8")) as RandomBattleFile);
}
