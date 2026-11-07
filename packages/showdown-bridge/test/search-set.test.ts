// The searched set of a decision (ADR-026 erratum E2 decision 2, SD-b): prior-top-k as the actor always searched it, and
// slot-cover, which reaches every move and switch of each active slot. Hand-worked sets, then the committed corpora: a stalling
// move (the engine's list, read from the reviewed data file, never named here) is in the set whenever a slot offers it.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { enumerateCandidates, type Candidate } from "../src/choices.ts";
import { decodeRequest } from "../src/request.ts";
import { loadRandomCorpus, UNAVAILABLE_DIR } from "../src/random-corpus.ts";
import { searchedIndices, slotCover } from "../src/search-set.ts";
import { STALL_MOVES_DIR, stallMovesFile } from "../src/stall-moves.ts";

const FORMAT = "gen9championsvgc2026regmc";
const cand = (choice: string, pending = false): Candidate => ({ choice, pending });

describe("slotCover, worked by hand", () => {
  const candidates = [
    cand("move 1 1, move 1 2"), // 0: c*
    cand("move 1 1, move 2 1"), // 1
    cand("move 2 1, move 1 2"), // 2: same move 2 as 3, lower prior
    cand("move 2 2, move 1 2"), // 3: the representative of move 2 in slot 0 (fixes the target)
    cand("switch 3, move 1 2"), // 4
    cand("move 1 1, switch 3"), // 5
    cand("move 2 2, switch 4", true), // 6: pending, never searched
    cand("switch 4, switch 3"), // 7: no candidate with the other slot equal to c*'s
  ];
  const prior = [50, 40, 30, 35, 20, 10, 99, 1];

  it("adds one candidate per slot action, with the other slot as the prior's best has it, in prior order", () => {
    expect(slotCover(candidates, prior, 16)).toEqual([0, 1, 3, 4, 5, 7]);
  });

  it("cuts by the prior order of the slot actions at k_max", () => {
    expect(slotCover(candidates, prior, 4)).toEqual([0, 1, 3, 4]);
    expect(slotCover(candidates, prior, 2)).toEqual([0, 1]);
  });

  it("covers a single slot, skips pass, and ties fall in canonical order", () => {
    const one = [cand("move 1 1"), cand("move 2"), cand("switch 3"), cand("switch 4")];
    expect(slotCover(one, [5, 5, 5, 5], 16)).toEqual([0, 1, 2, 3]);
    const passed = [cand("pass, move 1 1"), cand("pass, move 2")];
    expect(slotCover(passed, [1, 2], 16)).toEqual([1, 0]);
  });

  it("is empty with nothing to search", () => {
    expect(slotCover([cand("move 1", true)], [1], 16)).toEqual([]);
    expect(slotCover([], [], 16)).toEqual([]);
  });

  it("is chosen by the config's rule; the other rules keep the prior's top k", () => {
    const base = { top_k: 3, k_max: 16 };
    expect(searchedIndices({ ...base, top_k_rule: "prior-top-k" }, candidates, prior)).toEqual([0, 1, 3]);
    expect(searchedIndices({ ...base, top_k_rule: "distinct-slot" }, candidates, prior)).toEqual([0, 1, 3]);
    expect(searchedIndices({ ...base, top_k_rule: "slot-cover" }, candidates, prior)).toEqual([0, 1, 3, 4, 5, 7]);
  });
});

describe("slot-cover over the committed corpora", () => {
  const stalling = new Set((JSON.parse(readFileSync(join(STALL_MOVES_DIR, stallMovesFile(FORMAT)), "utf8")) as { stall_moves: string[] }).stall_moves);
  // A deterministic prior that favours the later candidates, and a flat one: the set must not depend on where the prior puts its mass.
  const priors: ((n: number) => number[])[] = [(n) => Array.from({ length: n }, (_, i) => 1 + i), (n) => Array.from({ length: n }, () => 7), (n) => Array.from({ length: n }, (_, i) => 1 + ((i * 37) % 11))];

  it("holds every offered stalling move of each slot", () => {
    let offered = 0;
    let requests = 0;
    for (const battle of [...loadRandomCorpus(), ...loadRandomCorpus(UNAVAILABLE_DIR)]) {
      for (const player of ["p1", "p2"] as const) {
        for (const line of battle.channels[player].lines) {
          if (!line.startsWith("|request|")) continue;
          const body = line.slice("|request|".length);
          const decoded = decodeRequest(body);
          if (decoded.phase !== "move") continue;
          const { active } = JSON.parse(body) as { active: { moves: { id: string }[] }[] };
          const { candidates } = enumerateCandidates(decoded);
          requests += 1;
          for (const priorOf of priors) {
            const set = new Set(slotCover(candidates, priorOf(candidates.length), 16).map((i) => (candidates[i] as Candidate).choice));
            candidates.forEach((candidate, index) => {
              if (candidate.pending) return;
              candidate.choice.split(", ").forEach((part, slot) => {
                const [verb, number] = part.split(" ");
                if (verb !== "move" || !stalling.has(active[slot]?.moves[Number(number) - 1]?.id ?? "")) return;
                offered += 1;
                const covered = [...set].some((choice) => choice.split(", ")[slot]?.split(" ").slice(0, 2).join(" ") === `move ${number}`);
                expect(covered, `${battle.id} ${player} candidate ${index} slot ${slot}`).toBe(true);
              });
            });
          }
        }
      }
    }
    expect(requests).toBeGreaterThan(100);
    expect(offered).toBeGreaterThan(0);
  });
});
