// The searched set of a decision (ADR-026 section 2; erratum E2 decision 2).
//
// Pure: it reads only the enumerator's candidates in canonical order and the pair's temperature-1 prior (one entry per candidate).
// `prior-top-k` (and `distinct-slot`, which the actor has always treated the same way) takes the k highest non-pending candidates,
// equal priors in canonical order. `slot-cover` searches every move and every switch of each active slot once, so a move of a slot is
// reached whenever it is offered, without naming any move. No game rule is written here: a slot action is the verb and number of a
// part of the canonical choice string.

import type { Candidate } from "./choices.ts";
import type { SearchConfig } from "./search-config.ts";

/** Non-pending candidates by prior descending; equal priors in canonical order (a stable sort on the enumerator's order). Indices into `candidates`. */
const rankedIndices = (candidates: readonly Candidate[], prior_ppm: readonly number[]): number[] =>
  candidates
    .map((candidate, index) => ({ candidate, index, ppm: prior_ppm[index] as number }))
    .filter(({ candidate }) => !candidate.pending)
    .sort((a, b) => b.ppm - a.ppm || a.index - b.index)
    .map(({ index }) => index);

const partsOf = (choice: string): string[] => choice.split(", ");

/** The slot action a part is: `move N` or `switch N` (target and Mega flag dropped); null for `pass` (a slot with nothing to do). */
const actionOf = (part: string): string | null => {
  const [verb, number] = part.split(" ");
  return verb === "move" || verb === "switch" ? `${verb} ${number}` : null;
};

/**
 * The slot-cover set, as indices into `candidates` in prior order. Let c* be the prior's argmax over the non-pending candidates. For
 * each slot s and each distinct slot action a of s, the representative is the highest-prior non-pending candidate holding a in s (it
 * fixes the target and the Mega flag); add the highest-prior non-pending candidate whose slot s equals the representative's part and
 * whose other slots equal c*'s, else the highest-prior one with that part in slot s. Duplicates are removed; at most `k_max` are kept,
 * cut by the prior order of the representatives (ties: slot, then canonical order).
 */
export function slotCover(candidates: readonly Candidate[], prior_ppm: readonly number[], k_max: number): number[] {
  const ranked = rankedIndices(candidates, prior_ppm);
  if (ranked.length === 0) return [];
  const parts = candidates.map((candidate) => partsOf(candidate.choice));
  const star = parts[ranked[0] as number] as string[];
  const reps: { rank: number; slot: number; part: string }[] = [];
  const seen = new Set<string>();
  const slots = Math.max(...ranked.map((i) => (parts[i] as string[]).length));
  for (let slot = 0; slot < slots; slot += 1) {
    ranked.forEach((index, rank) => {
      const part = (parts[index] as string[])[slot];
      const action = part === undefined ? null : actionOf(part);
      if (part === undefined || action === null || seen.has(`${slot}:${action}`)) return;
      seen.add(`${slot}:${action}`);
      reps.push({ rank, slot, part });
    });
  }
  reps.sort((a, b) => a.rank - b.rank || a.slot - b.slot);
  const chosen = new Set<number>();
  for (const { slot, part } of reps) {
    const holders = ranked.filter((index) => (parts[index] as string[])[slot] === part);
    const companion = holders.find((index) => (parts[index] as string[]).every((p, s) => s === slot || p === star[s])) ?? (holders[0] as number);
    chosen.add(companion);
    if (chosen.size >= k_max) break;
  }
  return ranked.filter((index) => chosen.has(index));
}

/** The indices of the candidates the config's rule searches, in prior rank. */
export function searchedIndices(config: Pick<SearchConfig, "top_k" | "top_k_rule" | "k_max">, candidates: readonly Candidate[], prior_ppm: readonly number[]): number[] {
  if (config.top_k_rule === "slot-cover") return slotCover(candidates, prior_ppm, config.k_max as number);
  return rankedIndices(candidates, prior_ppm).slice(0, config.top_k);
}
