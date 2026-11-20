// The committed battles the battle-level differential records (ADR-028 section 5 "Corpora", #1462): each as a `TraceSpec` the
// recorder can play, with the actors its corpus was played by. The golden battle (first legal choice, 300-turn cap), the
// random-play corpus and the singles (random-legal), the `unavailable/` corpus (pending-first) and the targeted
// `search-fidelity/` corpus (each seat's recorded actor kind); all but the golden battle are cut at 30 turns as captured.
// `transform/battle.json` is not here: it was cut from a league game and holds no choices, so it cannot be played again.
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type { TraceSpec } from "./battle-trace.ts";
import { firstLegal, GOLDEN_DIR, GOLDEN_FORMAT, GOLDEN_SEED, syntheticTeam } from "./golden.ts";
import { loadRandomCorpus, pendingFirstActor, RANDOM_DIR, RANDOM_MAX_TURNS, type RandomBattleFile, UNAVAILABLE_DIR } from "./random-corpus.ts";
import { randomLegalActor } from "./random-legal.ts";
import type { Actor, Player } from "./runner.ts";
import { FIDELITY_DIR } from "./search-fidelity-corpus.ts";

/** One-battle corpora beside the random one, each `<dir>/battle.json`, played random-legal against random-legal. */
export const SINGLE_BATTLES = ["baton-pass", "baton-pass-volatiles", "fling", "illusion", "revival", "side-guard"] as const;

const ACTORS: Record<string, (seed: number) => Actor> = { "random-legal": randomLegalActor, "pending-first": pendingFirstActor };

/** A committed battle: its corpus and the spec to record it with. */
export interface CommittedBattle {
  corpus: string;
  spec: TraceSpec;
}

const specOf = (corpus: string, b: RandomBattleFile & { actors?: Record<Player, string> }, kind: (p: Player) => string): CommittedBattle => ({
  corpus,
  spec: {
    battle_id: `${corpus}/${b.id}`,
    format_id: GOLDEN_FORMAT,
    seed: b.seed,
    teams: b.teams,
    actors: { p1: (ACTORS[kind("p1")] as (seed: number) => Actor)(b.actor_seeds.p1), p2: (ACTORS[kind("p2")] as (seed: number) => Actor)(b.actor_seeds.p2) },
    max_turns: RANDOM_MAX_TURNS,
  },
});

/** The committed corpora, in a fixed order: golden, random, unavailable, the singles, search-fidelity. Actors are fresh per call. */
export function committedBattles(corpora: readonly string[] = ["golden", "random", "unavailable", "singles", "search-fidelity"]): CommittedBattle[] {
  const out: CommittedBattle[] = [];
  const team = syntheticTeam();
  for (const corpus of corpora) {
    if (corpus === "golden") out.push({ corpus, spec: { battle_id: "golden", format_id: GOLDEN_FORMAT, seed: GOLDEN_SEED, teams: { p1: team, p2: structuredClone(team) }, actors: { p1: firstLegal, p2: firstLegal }, max_turns: 300 } });
    else if (corpus === "random") out.push(...loadRandomCorpus(RANDOM_DIR).map((b) => specOf(corpus, b, () => "random-legal")));
    else if (corpus === "unavailable") out.push(...loadRandomCorpus(UNAVAILABLE_DIR).map((b) => specOf(corpus, b, () => "pending-first")));
    else if (corpus === "singles") {
      for (const dir of SINGLE_BATTLES.filter((d) => existsSync(join(GOLDEN_DIR, d, "battle.json")))) {
        out.push(specOf(dir, JSON.parse(readFileSync(join(GOLDEN_DIR, dir, "battle.json"), "utf8")) as RandomBattleFile, () => "random-legal"));
      }
    } else if (corpus === "search-fidelity") {
      for (const b of loadRandomCorpus(FIDELITY_DIR) as (RandomBattleFile & { actors: Record<Player, string> })[]) out.push(specOf(corpus, b, (p) => b.actors[p]));
    } else throw new Error(`no committed corpus named ${corpus}`);
  }
  return out;
}
