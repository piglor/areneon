// The player channel of each committed choice-corpus case: the player's own stream up to the request the
// case's candidates answer, as the engine sends it (the secret half of the player's lines, the public half of
// the other's). Derived by replaying the committed log on the pinned engine; the fixture
// `choice-channels/channels.json` is compared with a fresh replay by `test/choice-channels.test.ts`.
import { join } from "node:path";
import { caseSpec, CHOICES_DIR, loadCorpus } from "./choice-corpus.ts";
import type { Engine } from "./engine.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "./golden.ts";
import { playerView } from "./search-turn.ts";

export const CHOICE_CHANNELS_FILE = join(GOLDEN_DIR, "choice-channels", "channels.json");

export interface CaseChannel {
  player: string;
  lines: string[];
}

interface ReplayBattle {
  setPlayer(player: string, options: { name: string; team: string }): void;
  choose(player: string, choice: string): boolean;
  sendUpdates(): void;
}

/** Replay every case of the corpus in `dir` and return its player channel, by file name. */
export function choiceChannels(engine: Engine, dir: string = CHOICES_DIR): Record<string, CaseChannel> {
  const out: Record<string, CaseChannel> = {};
  for (const [file, found] of Object.entries(loadCorpus(dir))) {
    const spec = caseSpec(found);
    const seat = spec.player === "p1" ? 1 : 2;
    const lines: string[] = [];
    const battle = new engine.sim.Battle({
      formatid: GOLDEN_FORMAT,
      seed: spec.seed,
      send: (type: string, data: string | string[]) => {
        if (type === "update") lines.push(...playerView((data as string[]).join("\n"), seat));
        if (type === "sideupdate") {
          const text = data as string;
          const at = text.indexOf("\n");
          if (text.slice(0, at) === spec.player) lines.push(text.slice(at + 1));
        }
      },
    } as never) as unknown as ReplayBattle;
    for (const player of ["p1", "p2"] as const) {
      battle.setPlayer(player, { name: player, team: engine.sim.Teams.pack(spec.teams[player]) });
    }
    battle.sendUpdates();
    for (const { player, choice } of spec.log) {
      if (!battle.choose(player, choice)) throw new Error(`${spec.id}: the engine refused a logged choice`);
      battle.sendUpdates();
    }
    out[file] = { player: spec.player, lines };
  }
  return out;
}

/** The fixture's bytes. */
export function choiceChannelsText(channels: Record<string, CaseChannel>): string {
  return `${JSON.stringify(channels, null, 1)}\n`;
}
