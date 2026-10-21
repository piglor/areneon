import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { type Engine, loadEngine } from "../src/index.ts";
import { playerChannel } from "../src/player-channel.ts";
import { NoCandidateError } from "../src/actor-rule.ts";
import {
  loadRandomCorpus,
  pendingFirstActor,
  playRandomBattle,
  RANDOM_DIR,
  RANDOM_MANIFEST,
  randomBattleFile,
  UNAVAILABLE_DIR,
} from "../src/random-corpus.ts";
import type { RequestRecord } from "../src/runner.ts";

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

describe("the [Unavailable choice] channel corpus", () => {
  const unavailable = loadRandomCorpus(UNAVAILABLE_DIR);

  it("is the committed set its manifest names", () => {
    const manifest = JSON.parse(readFileSync(join(UNAVAILABLE_DIR, RANDOM_MANIFEST), "utf8")) as {
      battles: { id: string; seed: string; turns: number; winner: string | null }[];
    };
    expect(manifest.battles).toEqual(
      unavailable.map((b) => ({ id: b.id, seed: b.seed, turns: b.turns, winner: b.winner })),
    );
    expect(unavailable.length).toBeGreaterThanOrEqual(2);
  });

  it("re-plays to the same channels, each holding a refusal that the engine sent to one player", async () => {
    for (const committed of unavailable) {
      const { id, seed, actor_seeds, teams } = committed;
      const spec = { id, seed, actor_seeds, teams };
      const replayed = randomBattleFile(spec, await playRandomBattle(engine, spec, pendingFirstActor));
      expect(replayed, id).toEqual(committed);
      const errors = [...replayed.channels.p1.lines, ...replayed.channels.p2.lines].filter((l) =>
        l.startsWith("|error|"),
      );
      expect(errors.length, id).toBeGreaterThan(0);
      expect(errors.every((l) => l.startsWith("|error|[Unavailable choice] "))).toBe(true);
      for (const player of ["p1", "p2"] as const) {
        expect(() => playerChannel(replayed.channels[player]), `${id} ${player}`).not.toThrow();
      }
    }
  }, 120_000);
});

describe("pendingFirstActor", () => {
  const request = (legal: RequestRecord["legal"]): RequestRecord => ({ player: "p1", request_seq: 1, legal }) as RequestRecord;

  it("plays a pending candidate when there is one, else a non-pending one", () => {
    const actor = pendingFirstActor(7);
    const mixed = { choices: ["a"], pending: ["p", "q"] } as unknown as RequestRecord["legal"];
    for (let i = 0; i < 10; i++) expect(["p", "q"]).toContain(actor(request(mixed)));
    const plain = { choices: ["a", "b"], pending: [] } as unknown as RequestRecord["legal"];
    expect(["a", "b"]).toContain(actor(request(plain)));
  });

  it("refuses a request with no legal choice set", () => {
    expect(() => pendingFirstActor(7)(request(undefined))).toThrow(NoCandidateError);
  });
});
