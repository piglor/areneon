// Print each committed choice-corpus case's player channel (the player's own stream up to the request
// the case's candidates answer) as JSON, `{<file>: {player, lines}}`. Test tooling for the Python encoder
// tests (`tests/ml/model/test_target.py`): it replays the committed log on the pinned engine, as CI does for
// the corpus, and keeps the engine's own player view (the secret half of the player's lines, the public half
// of the other's). Nothing is written.
//   node packages/showdown-bridge/scripts/choice-channels.ts
import { caseSpec, loadCorpus } from "../src/choice-corpus.ts";
import { loadEngine } from "../src/engine.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { playerView } from "../src/search-turn.ts";

interface ReplayBattle {
  setPlayer(player: string, options: { name: string; team: string }): void;
  choose(player: string, choice: string): boolean;
  sendUpdates(): void;
}

const engine = loadEngine();
const out: Record<string, { player: string; lines: string[] }> = {};
for (const [file, found] of Object.entries(loadCorpus())) {
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
process.stdout.write(JSON.stringify(out));
