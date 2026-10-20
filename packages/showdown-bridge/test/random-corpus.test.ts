import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { type Engine, loadEngine } from "../src/index.ts";
import { playerChannel } from "../src/player-channel.ts";
import {
  loadRandomCorpus,
  playRandomBattle,
  RANDOM_DIR,
  RANDOM_MANIFEST,
  randomBattleFile,
} from "../src/random-corpus.ts";

let engine: Engine;
beforeAll(() => {
  // SHOWDOWN_DIR: a worktree without its own build may point at another checkout.
  engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
});

const corpus = loadRandomCorpus();

describe("the random-play channel corpus", () => {
  it("is the committed set its manifest names", () => {
    const manifest = JSON.parse(readFileSync(join(RANDOM_DIR, RANDOM_MANIFEST), "utf8")) as {
      format: string;
      battles: { id: string; seed: string; turns: number; winner: string | null }[];
    };
    expect(manifest.format).toBe(GOLDEN_FORMAT);
    expect(manifest.battles).toEqual(
      corpus.map((b) => ({ id: b.id, seed: b.seed, turns: b.turns, winner: b.winner })),
    );
    expect(corpus.length).toBeGreaterThanOrEqual(40);
  });

  it("re-plays to the same channels in a fresh process (determinism)", async () => {
    for (const committed of corpus) {
      const { id, seed, actor_seeds, teams } = committed;
      const spec = { id, seed, actor_seeds, teams };
      expect(randomBattleFile(spec, await playRandomBattle(engine, spec)), id).toEqual(committed);
    }
  }, 120_000);

  it("holds only player-perspective channels", () => {
    for (const battle of corpus) {
      for (const player of ["p1", "p2"] as const) {
        expect(battle.channels[player].player).toBe(player);
        expect(() => playerChannel(battle.channels[player]), `${battle.id} ${player}`).not.toThrow();
      }
    }
  });

  it("shows the situations ordinary play reaches", () => {
    const text = corpus.map((b) => b.channels.p1.lines.join("\n")).join("\n");
    for (const marker of ["|-mega|", "|faint|", "|-weather|", "|-sidestart|", "|-status|", "|-boost|"]) {
      expect(text, marker).toContain(marker);
    }
    expect(new Set(corpus.map((b) => b.winner)).size).toBeGreaterThan(1);
  });
});
