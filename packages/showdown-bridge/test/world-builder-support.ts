import { interpretChannel, ownChannel, type PublicFlag, worldFactsAtRequests } from "@areneon/public-state";
import type { EngineRequest } from "@areneon/schema/types/v1/engine-request.ts";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type { PokemonSet } from "../src/engine.ts";
import { GOLDEN_DIR } from "../src/golden.ts";
import { loadRandomCorpus } from "../src/random-corpus.ts";
import { readWorldFactsData } from "./world-facts-data.ts";
import type { WorldInput, WorldOpponentMember } from "../src/world-builder.ts";

export const toId = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");

export interface Case {
  id: string;
  input: WorldInput;
}

/**
 * Every own move request of the committed random-corpus channels, for both seats. The opponent's TRUE brought team
 * is the test's input (read from the opponent's own first move request): a sampled world would replace it.
 */
export function cases(): Case[] {
  return casesOf(loadRandomCorpus());
}

/** The committed single-run battles that hold the refusals and the trapping flags: `unavailable/` and `search-fidelity/`. */
export function extraBattles(): BattleLike[] {
  const root = join(GOLDEN_DIR);
  return ["unavailable", "search-fidelity"].flatMap((dir) =>
    readdirSync(join(root, dir)).filter((f) => /^b\d+\.json$/.test(f)).sort().map((f) => {
      const file = JSON.parse(readFileSync(join(root, dir, f), "utf8")) as BattleLike;
      return { id: `${dir}/${f.slice(0, -5)}`, teams: file.teams, channels: file.channels };
    }),
  );
}

export interface BattleLike {
  id: string;
  teams: Record<"p1" | "p2", PokemonSet[]>;
  channels: Record<"p1" | "p2", { lines: string[] }>;
}

export function casesOf(battles: readonly BattleLike[]): Case[] {
  const out: Case[] = [];
  const factsData = readWorldFactsData();
  for (const battle of battles) {
    for (const player of ["p1", "p2"] as const) {
      const foe = player === "p1" ? "p2" : "p1";
      const foeRun = interpretChannel(foe, battle.channels[foe].lines, { terrains: [...factsData.terrains] });
      const foeFirst = foeRun.requests.find((r) => (r as EngineRequest).active !== undefined) as EngineRequest;
      const brought = foeFirst.side.pokemon.map((entry) => entry.ident.slice(entry.ident.indexOf(": ") + 2));
      const trueSets = brought.map((name) => (battle.teams[foe] as PokemonSet[]).find((set) => (set.name === "" ? set.species : set.name) === name) as PokemonSet);
      const run = interpretChannel(player, battle.channels[player].lines, { terrains: [...factsData.terrains] });
      const facts = new Map(worldFactsAtRequests({ channel: ownChannel(player, battle.channels[player].lines), data: factsData }).map((r) => [r.line, r.facts]));
      for (const frame of run.frames) {
        if (frame.kind !== "request") continue;
        const request = run.requests[frame.request - 1] as EngineRequest;
        if (request.active === undefined || request.forceSwitch !== undefined) continue;
        const members: WorldOpponentMember[] = trueSets.map((set) => {
          const index = frame.state.opponent.roster.findIndex((mon) => mon.base_species === toId(set.species) || mon.species === toId(set.species));
          return { roster_index: index, set };
        });
        out.push({
          id: `${battle.id}/${player}/${frame.index}`,
          input: { request, team: battle.teams[player] as PokemonSet[], state: frame.state, flags: run.flags.filter((flag: PublicFlag) => flag.index <= frame.index), facts: facts.get(frame.index) ?? null, opponent: { members } },
        });
      }
    }
  }
  return out;
}


/**
 * The world input at the LAST request of an own channel (the lines `runBattle` hands an actor), with the
 * opponent's true brought sets as the test's input. Null when the last line is no move request.
 */
export function inputAtLastRequest(player: "p1" | "p2", lines: readonly string[], team: readonly PokemonSet[], foeSets: readonly PokemonSet[]): WorldInput | null {
  const factsData = readWorldFactsData();
  const run = interpretChannel(player, lines, { terrains: [...factsData.terrains] });
  const frame = run.frames.at(-1);
  if (frame === undefined || frame.kind !== "request") return null;
  const request = run.requests[frame.request - 1] as EngineRequest;
  if (request.active === undefined || request.forceSwitch !== undefined) return null;
  const facts = worldFactsAtRequests({ channel: ownChannel(player, lines), data: factsData }).find((r) => r.line === frame.index)?.facts ?? null;
  const members: WorldOpponentMember[] = foeSets.map((set) => ({
    roster_index: frame.state.opponent.roster.findIndex((mon) => mon.base_species === toId(set.species) || mon.species === toId(set.species)),
    set,
  }));
  return { request, team, state: frame.state, flags: run.flags.filter((flag: PublicFlag) => flag.index <= frame.index), facts, opponent: { members } };
}
