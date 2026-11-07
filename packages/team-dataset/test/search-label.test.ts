// The label search (ADR-026 erratum E2 decision 2, SD-b) with the real opponent model, the pinned engine and the stand-in worker:
// the actor under the committed label config searches the slot-cover set, the set is a function of the request and the prior alone,
// a same input gives the same scores, a changed hidden opponent changes neither, and every score carries a q_ppm in range.
import { enumerateCandidates, loadSearchConfig, searchedIndices, type Actor, type SearchActor } from "@areneon/showdown-bridge";
import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import { loadRandomCorpus, type RandomBattleFile } from "../../showdown-bridge/src/random-corpus.ts";
import { comparable, mutated } from "../../showdown-bridge/test/damage-aware-invariance-support.ts";
import { FORMAT } from "./support.ts";
import { playSearch, portFor, setupFor } from "./search-actor-support.ts";

const battles = loadRandomCorpus().slice(0, 2);
const label = loadSearchConfig(FORMAT, "rerank-1-label.v1");
// The stand-in worker's prior, recomputed here: a hash of each choice.
const priorOf = (choice: string): number => 1 + (createHash("sha256").update(choice).digest().readUInt16BE(0) % 1000);

interface Seen {
  request_seq: number;
  scores: readonly { choice: string; sum: number; q_ppm: number }[];
}

async function labelled(battle: RandomBattleFile, change: { p2?: RandomBattleFile["teams"]["p2"] } = {}) {
  const seen: Seen[] = [];
  const expected = new Map<number, string[]>();
  const game = await playSearch(battle, {
    maxTurns: 3,
    ...change,
    setup: setupFor(portFor(), { config: label }),
    options: { on_decision: (e) => seen.push({ request_seq: e.request_seq, scores: e.scores }) },
    wrap: (actor: SearchActor): Actor => async (request, context) => {
      if (request.phase === "move") {
        const { candidates } = enumerateCandidates(request);
        const prior = candidates.map(({ choice }) => priorOf(choice));
        expected.set(request.request_seq, searchedIndices(label, candidates, prior).map((i) => candidates[i]?.choice as string));
      }
      return actor(request, context);
    },
  });
  return { ...game, seen, expected };
}

describe("the actor under the label config", () => {
  it("searches the slot-cover set, scores every searched candidate in range, and is deterministic", async () => {
    const battle = battles[0] as RandomBattleFile;
    const a = await labelled(battle);
    const b = await labelled(battle);
    expect(a.seen.length).toBeGreaterThan(0);
    for (const { request_seq, scores } of a.seen) {
      expect(scores.map((s) => s.choice)).toEqual(a.expected.get(request_seq));
      expect(scores.length).toBeLessThanOrEqual(label.k_max as number);
      for (const s of scores) {
        expect(Number.isInteger(s.q_ppm)).toBe(true);
        expect(s.q_ppm).toBeGreaterThanOrEqual(0);
        expect(s.q_ppm).toBeLessThanOrEqual(1_000_000);
      }
    }
    expect(b.seen).toEqual(a.seen);
    expect(b.record.choices).toEqual(a.record.choices);
    expect(b.actor.report().counts).toEqual(a.actor.report().counts);
    // the label search is wider than the evaluation search's k = 4 wherever the slots offer more than four distinct candidates
    expect(a.seen.some(({ scores }) => scores.length > 4)).toBe(true);
  }, 300_000);

  it("labels from the player's view alone: a hidden-information change in the opponent's team changes no choice and no score", async () => {
    let searched = 0;
    for (const battle of battles) {
      const a = await labelled(battle);
      const b = await labelled(battle, { p2: mutated(battle.teams.p2) });
      for (const { seq, first, second } of comparable(a.record, b.record)) {
        expect(second, `${battle.id} request ${seq}`).toBe(first);
        const sa = a.seen.find((s) => s.request_seq === seq);
        const sb = b.seen.find((s) => s.request_seq === seq);
        expect(sb, `${battle.id} request ${seq} scores`).toEqual(sa);
        if (sa !== undefined) searched += 1;
      }
    }
    expect(searched).toBeGreaterThanOrEqual(2);
  }, 300_000);
});
