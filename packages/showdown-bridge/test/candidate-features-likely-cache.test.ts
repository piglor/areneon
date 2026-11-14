import { beforeAll, describe, expect, it } from "vitest";
import { toId, type PublicMon } from "@areneon/public-state";
import { candidateFeaturesLikely, type CandidateFeaturesLikely } from "../src/candidate-features-likely.ts";
import { createFeatureMemo, type FeatureConfigs, type FeatureMemo } from "../src/candidate-features.ts";
import { jcs, type Json } from "../src/canonical-json.ts";
import type { PokemonSet } from "../src/engine.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { type LikelySetup, loadLikelySetup, type MoveList } from "../src/likely-threats-config.ts";
import { withMoveOutcome } from "../src/mechanics-config.ts";
import { loadOutcomeConfig } from "../src/outcome-config.ts";
import type { PlayerChannel } from "../src/observation.ts";
import { playerChannel } from "../src/player-channel.ts";
import { loadRandomCorpus } from "../src/random-corpus.ts";
import { FIDELITY_DIR } from "../src/search-fidelity-corpus.ts";
import { capture, type Captured, config, corpus, engine, mechanics, snapshot, stateOf, vocabulary } from "./damage-aware-support.ts";

// ADR-022 erratum E-22.7.4 and E-22.7.6 (cache equality, required): every likely row the per-game roll cache gives equals the value of one
// direct, uncached probe query at the same state, over the requests of the committed random-play and search-fidelity corpora, both seats,
// with the cache kept across the requests of a game (so a row really comes from an earlier request's rolls). Negative controls: a key
// without stat stages answers a boosted state with the value of the neutral one, and a record that lacks the hit-point handler of a move
// that reads them does the same at another hit-point value.

const base = loadLikelySetup(GOLDEN_FORMAT, "probe-outcome-likely.v1.json");
const configs = (likely: LikelySetup): FeatureConfigs => ({ vocabulary, config, mechanics: withMoveOutcome(mechanics), outcome: { config: loadOutcomeConfig(GOLDEN_FORMAT) }, likely, snapshot, data_class: "synthetic" });
const dex = engine.sim.Dex.mod("champions") as unknown as { moves: { get(id: string): { exists: boolean; category: string; target: string } } };

const FULL = process.env.ARENEON_LIKELY_SWEEP === "full";
const RANDOM_BATTLES = FULL ? 40 : 4;
const FIDELITY_STEP = FULL ? 10 : 67;
const MAX_TURNS = 10;

const teamOf = (c: Captured): PokemonSet[] => c.context.team as PokemonSet[];
const candidatesOf = (c: Captured) => {
  const legal = c.request.legal as NonNullable<typeof c.request.legal>;
  return [...legal.choices, ...legal.pending].map((choice) => ({ choice, pending: legal.pending.includes(choice) }));
};
const channelOf = (c: Captured, lines: readonly string[] = c.context.channel): PlayerChannel => playerChannel({ schema_version: 1, data_class: "synthetic", player: c.context.player, lines: [...lines] });
const build = (c: Captured, likely: LikelySetup, memo: FeatureMemo = createFeatureMemo(), lines?: readonly string[]): CandidateFeaturesLikely => candidateFeaturesLikely(engine, configs(likely), channelOf(c, lines), teamOf(c), candidatesOf(c), memo);
const likelyBytes = (f: CandidateFeaturesLikely): string => jcs({ belief_threats: f.belief_threats, refused: f.belief_threats_refused ?? null } as unknown as Json);
const foesOf = (c: Captured): PublicMon[] => Object.values(stateOf(c).opponent.slots).filter((i) => i !== null).map((i) => stateOf(c).opponent.roster[i as number] as PublicMon);
const moveRequests = (cs: Captured[]): Captured[] => cs.filter((c) => c.request.phase === "move" && (c.request.legal?.choices.length ?? 0) >= 2);

const games: Captured[][] = [];
beforeAll(async () => {
  const fidelity = loadRandomCorpus(FIDELITY_DIR);
  const battles = [...corpus.slice(0, RANDOM_BATTLES), ...fidelity.filter((_, i) => i % FIDELITY_STEP === 0)];
  for (const battle of battles) for (const player of ["p1", "p2"] as const) games.push(await capture(battle, player, MAX_TURNS));
}, FULL ? 7_200_000 : 600_000);

describe("the roll cache equals a direct probe query", () => {
  it("at every move request of the sampled corpora, with the cache kept across a game's requests", () => {
    const direct: LikelySetup = { ...base, direct: true };
    const total = { requests: 0, rows: 0, hits: 0, ko: 0, refused: 0, ranked: 0 };
    for (const game of games) {
      const memo = createFeatureMemo();
      for (const c of moveRequests(game)) {
        let cached: CandidateFeaturesLikely;
        try {
          cached = build(c, base, memo);
        } catch (error) {
          // a refusal is a function of the player's view: the direct build refuses alike
          expect(() => build(c, direct)).toThrow();
          total.refused += 1;
          void error;
          continue;
        }
        const fresh = build(c, direct);
        expect(likelyBytes(cached), `${c.battle.id} ${c.context.player} request ${c.request.request_seq}`).toBe(likelyBytes(fresh));
        total.requests += 1;
        total.rows += cached.belief_threats.length;
        total.ko += cached.belief_threats.filter((r) => r.kind === "computed" && (r.ohko_rolls ?? 0) > 0).length;
      }
      const counts = memo.likely?.counts;
      total.hits += (counts?.full_hits ?? 0) + (counts?.base_hits ?? 0);
      total.ranked += counts?.ranked_foes ?? 0;
    }
    expect(total.requests, JSON.stringify(total)).toBeGreaterThan(30);
    expect(total.rows, JSON.stringify(total)).toBeGreaterThan(100);
    // not vacuous: rows really came from the cache, a ranking was needed, and some rows reach the KO
    expect(total.hits, JSON.stringify(total)).toBeGreaterThan(0);
    expect(total.ranked, JSON.stringify(total)).toBeGreaterThan(0);
  }, FULL ? 14_400_000 : 1_800_000);

  it("is the bytes of a cold cache and of a fresh memo", () => {
    const warm = createFeatureMemo();
    let compared = 0;
    for (const c of moveRequests(games[0] as Captured[])) {
      const first = build(c, base, warm);
      expect(likelyBytes(build(c, base, warm))).toBe(likelyBytes(first)); // asked again at once: all from the cache
      expect(likelyBytes(build(c, base))).toBe(likelyBytes(first));
      compared += 1;
    }
    expect(compared).toBeGreaterThan(0);
  }, 600_000);
});

/** A foe boosted by the channel: `|-boost|` lines inserted before the request. */
function boosted(c: Captured): string[] | null {
  const foe = foesOf(c)[0];
  const letter = Object.keys(stateOf(c).opponent.slots).sort()[0];
  if (foe === undefined || foe.name === null || letter === undefined) return null;
  const lines = [...c.context.channel];
  const request = lines.length - 1;
  const ident = `${stateOf(c).opponent.player}${letter}: ${foe.name}`;
  lines.splice(request, 0, `|-boost|${ident}|atk|2`, `|-boost|${ident}|spa|2`);
  return lines;
}

describe("negative controls: a key without a part of the state answers with another state's value", () => {
  const pick = (): Captured => {
    const found = games.flatMap((game) => moveRequests(game)).find((c) => boosted(c) !== null && foesOf(c).length > 0 && build(c, base).belief_threats.some((r) => r.kind === "computed" && r.max_ppm > 0));
    return found as Captured;
  };

  it("the stat stages are part of the key: after a boost the cache equals the direct value, and a key without them does not", () => {
    const c = pick();
    const lines = boosted(c) as string[];
    const direct = likelyBytes(build(c, { ...base, direct: true }, createFeatureMemo(), lines));
    // with the stages in the key: warm at the neutral state, then ask at the boosted one
    const memo = createFeatureMemo();
    build(c, base, memo);
    expect(likelyBytes(build(c, base, memo, lines))).toBe(direct);
    expect(direct).not.toBe(likelyBytes(build(c, base)));
    // without: the neutral state's rolls answer the boosted one
    const poisoned = createFeatureMemo();
    const loose: LikelySetup = { ...base, drop_key: "stages" };
    build(c, loose, poisoned);
    expect(likelyBytes(build(c, loose, poisoned, lines))).not.toBe(direct);
  }, 600_000);

  it("the attacker's hit points are part of the key for a move whose power reads them, and a record without its handler is flagged", () => {
    // each foe's table list is its first hit-point-reading move if it has one (Reversal, Flail, ...), else its first damaging pool move
    const readers = ["eruption", "waterspout", "reversal", "flail", "wringout", "crushgrip"];
    const listFor = (mon: PublicMon): string | undefined => {
      const pool = (vocabulary.roster.find((entry) => entry.id === toId(mon.species ?? mon.base_species))?.moves ?? []).filter((m) => dex.moves.get(m).exists && dex.moves.get(m).category !== "Status" && dex.moves.get(m).target !== "self");
      return pool.find((m) => readers.includes(m)) ?? undefined;
    };
    const c = games.flatMap((game) => moveRequests(game)).find((r) => foesOf(r).length > 0 && foesOf(r).every((mon) => listFor(mon) !== undefined) && boostedHp(r) !== null) as Captured | undefined;
    expect(c, "a request whose foes all have a hit-point-reading move").toBeDefined();
    if (c === undefined) return;
    const table = new Map<string, MoveList>();
    for (const mon of stateOf(c).opponent.roster) {
      const move = listFor(mon);
      table.set(toId(mon.species ?? mon.base_species), { slots: 4, total: 1_000_000, masses: new Map(move === undefined ? [] : [[move, 250_000]]) });
    }
    const lines = boostedHp(c) as string[];
    const right: LikelySetup = { config: base.config, table };
    const direct = likelyBytes(build(c, { ...right, direct: true }, createFeatureMemo(), lines));
    const memo = createFeatureMemo();
    build(c, right, memo);
    expect(likelyBytes(build(c, right, memo, lines))).toBe(direct);
    // a record whose handler list lacks the callback: the first state's rolls answer the second
    const missing: LikelySetup = { config: { ...base.config, hp_key_handlers: ["onModifyDamage"] }, table };
    const stale = createFeatureMemo();
    build(c, missing, stale);
    expect(likelyBytes(build(c, missing, stale, lines))).not.toBe(direct);
  }, 600_000);
});

/** The first foe shown at half hit points (a `-damage` line before the request). */
function boostedHp(c: Captured): string[] | null {
  const foe = foesOf(c)[0];
  const letter = Object.keys(stateOf(c).opponent.slots).sort()[0];
  if (foe === undefined || foe.name === null || letter === undefined) return null;
  const lines = [...c.context.channel];
  lines.splice(lines.length - 1, 0, `|-damage|${stateOf(c).opponent.player}${letter}: ${foe.name}|40/100`);
  return lines;
}
