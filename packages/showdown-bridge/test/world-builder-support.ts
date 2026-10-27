import { interpretChannel, type PublicFlag } from "@areneon/public-state";
import type { EngineRequest } from "@areneon/schema/types/v1/engine-request.ts";
import type { PokemonSet } from "../src/engine.ts";
import { loadRandomCorpus } from "../src/random-corpus.ts";
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
  const out: Case[] = [];
  for (const battle of loadRandomCorpus()) {
    for (const player of ["p1", "p2"] as const) {
      const foe = player === "p1" ? "p2" : "p1";
      const foeRun = interpretChannel(foe, battle.channels[foe].lines);
      const foeFirst = foeRun.requests.find((r) => (r as EngineRequest).active !== undefined) as EngineRequest;
      const brought = foeFirst.side.pokemon.map((entry) => entry.ident.slice(entry.ident.indexOf(": ") + 2));
      const trueSets = brought.map((name) => (battle.teams[foe] as PokemonSet[]).find((set) => (set.name === "" ? set.species : set.name) === name) as PokemonSet);
      const run = interpretChannel(player, battle.channels[player].lines);
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
          input: { request, team: battle.teams[player] as PokemonSet[], state: frame.state, flags: run.flags.filter((flag: PublicFlag) => flag.index <= frame.index), opponent: { members } },
        });
      }
    }
  }
  return out;
}

