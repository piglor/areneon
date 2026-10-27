// The belief on the teams the committed `generator-matched` prior is derived for (#1169): seeded random-legal games on teams drawn
// from the verified `fixtures-v1` dataset (the loop's own pool function, Illusion excluded as the run configs do), both seats, every
// request. The prior's support is the generator's, so the belief must exist at every request (no `evidence_has_no_support`, no reader
// flag) and the opponent's real sets and brought set must be possible under it (the oracle of `opponent-model/test/corpus.ts`, which
// reads the real team; the belief is never given it). `opponent-model` is imported by path only, as in `generator-prior.test.ts`.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { readTrainingRunConfig } from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { BeliefError, belief } from "../../opponent-model/src/index.ts";
import { type Game, type Set6, hiding, points, sound } from "../../opponent-model/test/corpus.ts";
import { realVocabulary } from "../../opponent-model/test/real.ts";
import { verifiedDatasetHash } from "../src/dataset-hash.ts";
import { DATASETS_DIR } from "../src/fixtures.ts";
import { datasetTeamPool } from "../src/loop-pool.ts";
import { playSweepBattle } from "../src/reducer-sweep.ts";
import { engine, FORMAT } from "./support.ts";

const ROOT = fileURLToPath(new URL("../../..", import.meta.url));
const SOURCE = join(DATASETS_DIR(FORMAT), "fixtures-v1");
const prior = JSON.parse(readFileSync(join(ROOT, "config", "opponent-model", FORMAT, "generator-matched.uniform-v1.json"), "utf8"));
const raw = JSON.parse(readFileSync(join(ROOT, "fixtures/contracts/v1/training-run-config/valid/tiny.json"), "utf8")) as Record<string, unknown>;
(raw.generation as Record<string, unknown>).excluded_abilities = ["Illusion"];
raw.team_dataset = { path: SOURCE, dataset_hash: verifiedDatasetHash(engine, SOURCE, FORMAT) };
const config = readTrainingRunConfig(raw);

describe("the belief on generator teams", () => {
  it("exists at every request and holds the real team and brought set possible", { timeout: 300_000 }, async () => {
    const pool = datasetTeamPool(engine, config);
    let requests = 0;
    for (let index = 0; index < 8; index++) {
      const battle = await playSweepBattle(engine, FORMAT, pool, { run_key: "generator-belief", mix: "random-legal", index, max_turns: 100 });
      expect(battle.failure).toBeUndefined();
      const game: Game = { id: `g${index}`, teams: battle.teams as unknown as Record<"p1" | "p2", Set6[]>, channels: battle.channels };
      for (const point of points(game)) {
        const where = `${game.id} ${point.seat} request ${point.request}`;
        let b;
        try {
          b = belief({ prior, vocabulary: realVocabulary, config: hiding, opponent: point.state.opponent, flags: point.flags });
        } catch (e) {
          if (e instanceof BeliefError) throw new Error(`${where}: ${e.code} (${e.message})`, { cause: e });
          throw e;
        }
        const { truth } = point;
        if (truth === null) throw new Error(`${where}: the reader holds an Illusion twin on a team without Illusion`);
        expect(sound(b, { ...point, truth }).length, where).toBeGreaterThan(0);
        requests += 1;
      }
    }
    expect(requests).toBeGreaterThan(200);
  });
});
