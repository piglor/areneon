// Invariance of the belief and the worlds under an open sheet (ADR-025 erratum E1, #1155): the sheet shows the team as built and
// holds no Stat Points, so two opponents that differ only in Stat Points give byte-identical beliefs and worlds at every request
// with an equal player channel, while a difference the sheet does show (an item) changes the belief (the test has teeth).
// The engine is imported by path for this test only: the package itself is engine-free.

import { beforeAll, describe, expect, it } from "vitest";
import { interpretChannel } from "@areneon/public-state";
import { loadEngine } from "../../showdown-bridge/src/index.ts";
import type { Engine, PokemonSet } from "../../showdown-bridge/src/engine.ts";
import { OPEN_SHEET_FORMAT, loadOpenSheetCorpus } from "../../showdown-bridge/src/open-sheet-corpus.ts";
import type { RandomBattleFile } from "../../showdown-bridge/src/random-corpus.ts";
import { createSetLegality } from "../../showdown-bridge/src/set-legality.ts";
import { type BattleRecord, type RequestRecord, runBattle } from "../../showdown-bridge/src/runner.ts";
import { actorRule } from "../../showdown-bridge/src/actor-rule.ts";
import { type SpreadSampler, beliefDigest, createSampler, worldsDigest } from "../src/index.ts";
import { beliefAt, skewedPrior } from "./corpus.ts";
import { realVocabulary } from "./real.ts";

const MAX_TURNS = 5;
let engine: Engine;
let battles: RandomBattleFile[];
beforeAll(() => {
  engine = loadEngine();
  battles = loadOpenSheetCorpus().slice(0, 3);
});

const spreads: SpreadSampler = {
  other: (_species, _forme, below) => ({
    nature: (realVocabulary.natures[Number(below(BigInt(realVocabulary.natures.length)))] as { id: string }).id,
    stat_points: { hp: Number(below(33n)), atk: Number(below(33n)), def: 0, spa: 0, spd: 0, spe: Number(below(2n)) },
  }),
};
const stream = (k: number): Uint8Array => Uint8Array.from({ length: 32 }, (_, i) => (k * 29 + i * 11) & 255);
const firstLegal = (request: RequestRecord): string => {
  if (request.legal === undefined) throw new Error("no legal set");
  return actorRule(request.legal);
};

const STATS = ["hp", "atk", "def", "spa", "spd", "spe"] as const;
/** The same team with every set's Stat Points rotated two places over the stats: the total, each value and the legality are unchanged. */
const restated = (team: PokemonSet[]): PokemonSet[] =>
  team.map((set) => ({ ...structuredClone(set), evs: Object.fromEntries(STATS.map((stat, k) => [stat, set.evs[STATS[(k + 2) % 6] as (typeof STATS)[number]]])) as PokemonSet["evs"] }));

const play = (battle: RandomBattleFile, p2: PokemonSet[]): Promise<BattleRecord> =>
  runBattle({ engine, formatId: OPEN_SHEET_FORMAT, seed: battle.seed, teams: { p1: battle.teams.p1, p2 }, actors: { p1: firstLegal, p2: firstLegal }, maxTurns: MAX_TURNS });

/** The p1 channel prefixes up to and including each request, while the two battles' channels agree. */
function equalPrefixes(a: BattleRecord, b: BattleRecord): [string[], string[]][] {
  const cuts = (record: BattleRecord): string[][] => record.channels.p1.flatMap((_, i, lines) => (lines[i]?.startsWith("|request|") ? [lines.slice(0, i + 1)] : []));
  const left = cuts(a);
  const right = cuts(b);
  const out: [string[], string[]][] = [];
  for (let k = 0; k < Math.min(left.length, right.length); k++) {
    if (JSON.stringify(left[k]) !== JSON.stringify(right[k])) break;
    out.push([left[k] as string[], right[k] as string[]]);
  }
  return out;
}

const view = (lines: string[]) => {
  const run = interpretChannel("p1", lines);
  const state = run.frames[run.frames.length - 1]!.state;
  return { state, point: { state, flags: run.flags, game: "", seat: "p1" as const, request: 0, truth: null } };
};

describe("an open sheet: the belief does not depend on the opponent's Stat Points", () => {
  it("is byte-identical, belief and worlds, at every request with an equal player channel", async () => {
    const sampler = createSampler({ legality: createSetLegality({ engine, vocabulary: realVocabulary }), spreads });
    let compared = 0;
    let beyondPreview = 0;
    let differs = false;
    for (const [n, battle] of battles.entries()) {
      const a = await play(battle, battle.teams.p2);
      const b = await play(battle, restated(battle.teams.p2));
      differs ||= JSON.stringify(restated(battle.teams.p2)) !== JSON.stringify(battle.teams.p2);
      for (const [mine, yours] of equalPrefixes(a, b)) {
        const one = view(mine);
        const two = view(yours);
        expect(one.state.opponent.sheet).toHaveLength(6);
        const prior = skewedPrior(one.state.opponent.roster.map((m) => m.base_species));
        const left = beliefAt(one.point, prior);
        const right = beliefAt(two.point, prior);
        expect(beliefDigest(right)).toBe(beliefDigest(left));
        const s = stream(compared + n);
        expect(worldsDigest(sampler.sampleWorlds(right, s, 3))).toBe(worldsDigest(sampler.sampleWorlds(left, s, 3)));
        compared += 1;
        if (one.state.turn > 0) beyondPreview += 1;
      }
    }
    expect(differs).toBe(true); // the pair really differs in Stat Points
    expect(compared).toBeGreaterThanOrEqual(6);
    expect(beyondPreview).toBeGreaterThanOrEqual(3);
  }, 240_000);

  it("has teeth: opponents that differ in a held item give different beliefs, because the sheet shows it", async () => {
    let differing = 0;
    for (const battle of battles) {
      const theirs = battle.teams.p2;
      const other = structuredClone(theirs);
      other.forEach((set, i) => {
        set.item = (theirs[(i + 1) % theirs.length] as PokemonSet).item;
      });
      const validator = engine.sim.TeamValidator.get(OPEN_SHEET_FORMAT);
      if ((validator.validateTeam(structuredClone(other)) ?? []).length > 0) continue; // an Item Clause or stone clash: not a legal pair
      const a = await play(battle, theirs);
      const b = await play(battle, other);
      // the sheet comes before the first request, so the channels differ from their first line on: compare the two Team Previews
      const preview = (record: BattleRecord): string[] => record.channels.p1.slice(0, record.channels.p1.findIndex((line) => line.startsWith("|request|")) + 1);
      const mine = view(preview(a));
      const yours = view(preview(b));
      const prior = skewedPrior(mine.state.opponent.roster.map((m) => m.base_species));
      if (beliefDigest(beliefAt(mine.point, prior)) !== beliefDigest(beliefAt(yours.point, prior))) differing += 1;
    }
    expect(differing).toBeGreaterThan(0);
  }, 240_000);
});
