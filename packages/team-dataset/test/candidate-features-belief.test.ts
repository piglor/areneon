// `probe-damage-belief/1` with the real opponent model, the pinned validator, the committed prior and the real probe
// (ADR-025 section 6 item 2, section 8; ADR-022 section 3 and erratum E-22.2): the rows are a function of the player's
// own view alone (hidden-information invariance with a negative control and a canary), the rows made at play time with a
// game-long memo equal those recomputed alone, and the set identity names the belief.
import {
  type BeliefWorldsContext,
  type CandidateFeaturesBelief,
  candidateFeaturesBelief,
  createFeatureMemo,
  FEATURE_SET_BELIEF,
  type FeatureConfigs,
  featureSetHash,
  featureSetIdentity,
  loadDamageAwareConfig,
  loadDamageAwareConfigV2,
  loadMechanicsConfig,
  loadRulesSnapshot,
  resolveFormat,
  runBattle,
} from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { readerSha256 } from "../../public-state/src/index.ts";
import { jcs, type Json } from "../../showdown-bridge/src/canonical-json.ts";
import type { PokemonSet } from "../../showdown-bridge/src/engine.ts";
import { type Actor } from "../../showdown-bridge/src/runner.ts";
import { firstLegal, goldenTeamVocabulary } from "../../showdown-bridge/src/golden.ts";
import { playerChannel } from "../../showdown-bridge/src/player-channel.ts";
import { loadRandomCorpus, type RandomBattleFile } from "../../showdown-bridge/src/random-corpus.ts";
import { mutated } from "../../showdown-bridge/test/damage-aware-invariance-support.ts";
import { createBeliefWorlds } from "../src/damage-aware-worlds.ts";
import { engine, FORMAT } from "./support.ts";

const vocabulary = goldenTeamVocabulary();
const record = loadDamageAwareConfigV2(vocabulary);
const snapshot = loadRulesSnapshot(engine.commit);
const manifest = resolveFormat(engine, FORMAT, snapshot).manifest;
const context: BeliefWorldsContext = { engine, vocabulary, config: record, reader_sha256: readerSha256(), manifest };
const WORLDS = 4; // the committed record says 8; four keep the real-probe tests short, the rule is the same
const MAX_TURNS = 10;
const corpus = loadRandomCorpus().slice(0, 8);
const toId = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");

const configs = (change: Partial<FeatureConfigs> = {}): FeatureConfigs => ({
  vocabulary,
  config: loadDamageAwareConfig("v1", vocabulary),
  mechanics: loadMechanicsConfig(FORMAT),
  snapshot,
  data_class: "synthetic",
  feature_set: FEATURE_SET_BELIEF,
  belief: { port: createBeliefWorlds(context), worlds: WORLDS },
  ...change,
});

interface Point {
  seq: number;
  lines: string[];
  candidates: { choice: string; pending: boolean }[];
  team: PokemonSet[];
}

/** p1's move requests of one battle (p1 plays the first legal choice, so the pair differs only in what p2 holds). */
async function points(battle: RandomBattleFile, p2: PokemonSet[]): Promise<Point[]> {
  const out: Point[] = [];
  const p1: Actor = (request, ctx) => {
    if (request.phase === "move" && request.legal !== undefined && ctx?.team !== undefined) {
      const legal = request.legal;
      out.push({ seq: request.request_seq, lines: [...ctx.channel], candidates: [...legal.choices, ...legal.pending].map((choice) => ({ choice, pending: legal.pending.includes(choice) })), team: ctx.team as PokemonSet[] });
    }
    return firstLegal(request, ctx as never);
  };
  await runBattle({ engine, formatId: FORMAT, seed: battle.seed, teams: { p1: battle.teams.p1, p2 }, actors: { p1, p2: firstLegal }, maxTurns: MAX_TURNS });
  return out;
}

const featuresAt = (point: Point, change: Partial<FeatureConfigs> = {}, memo = createFeatureMemo()): CandidateFeaturesBelief =>
  candidateFeaturesBelief(engine, configs(change), playerChannel({ schema_version: 1, data_class: "synthetic", player: "p1", lines: point.lines }), point.team, point.candidates, memo);
const bytes = (f: CandidateFeaturesBelief): string => jcs(f as unknown as Json);

/** Request points of a pair whose p1 channels are equal (the prefix up to and including the request). */
function equalPoints(a: Point[], b: Point[]): { a: Point; b: Point }[] {
  const out: { a: Point; b: Point }[] = [];
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    const [left, right] = [a[i] as Point, b[i] as Point];
    if (left.seq !== right.seq || jcs(left.lines as unknown as Json) !== jcs(right.lines as unknown as Json)) break;
    out.push({ a: left, b: right });
  }
  return out;
}

/** The negative control's port: every world is the opponent's REAL sets, as a belief built from the real team would be. */
function cheatingPort(real: PokemonSet[]) {
  const inner = createBeliefWorlds(context);
  return {
    ...inner,
    sample: (input: Parameters<typeof inner.sample>[0]) => {
      const world = {
        slots: input.opponent.roster.map((mon, index) => {
          const set = real.find((s) => [mon.base_species, mon.species].includes(toId(s.species))) ?? (real[0] as PokemonSet);
          return {
            index,
            brought: true,
            set: { species: mon.base_species, forme: mon.base_species, item: set.item === "" ? null : toId(set.item), ability: toId(set.ability), nature: toId(set.nature), stat_points: set.evs, moves: set.moves.map(toId).sort(), level: 50, gender: null },
          };
        }),
      };
      return { worlds: Array.from({ length: input.n }, () => ({ world })), redraws: 0 };
    },
  };
}

describe("hidden-information invariance of the belief features (ADR-022 section 3, ADR-025 section 8)", () => {
  it("gives byte-identical rows and feature_set_hash at every request with an equal player channel", async () => {
    let total = 0;
    let withRows = 0;
    for (const battle of corpus) {
      const other = mutated(battle.teams.p2);
      expect(JSON.stringify(other)).not.toBe(JSON.stringify(battle.teams.p2));
      for (const { a, b } of equalPoints(await points(battle, battle.teams.p2), await points(battle, other))) {
        const [left, right] = [featuresAt(a), featuresAt(b)];
        expect(bytes(right), `${battle.id} request ${a.seq}`).toBe(bytes(left));
        total += 1;
        if (left.rows.length > 0) withRows += 1;
      }
    }
    expect(total).toBeGreaterThanOrEqual(8);
    expect(withRows).toBeGreaterThanOrEqual(6);
  }, 280_000);

  it("has teeth: a belief handed the real opponent team gives different rows at such a request", async () => {
    let compared = 0;
    let differing = 0;
    for (const battle of corpus) {
      const other = mutated(battle.teams.p2);
      for (const pair of equalPoints(await points(battle, battle.teams.p2), await points(battle, other))) {
        compared += 1;
        const left = featuresAt(pair.a, { belief: { port: cheatingPort(battle.teams.p2), worlds: WORLDS } });
        const right = featuresAt(pair.b, { belief: { port: cheatingPort(other), worlds: WORLDS } });
        if (bytes(left) !== bytes(right)) differing += 1;
      }
    }
    expect(compared).toBeGreaterThanOrEqual(8);
    expect(differing).toBeGreaterThan(0);
  }, 280_000);

  it("shows no never-public field of the opponent in the bytes of any row set (canary)", async () => {
    let scanned = 0;
    for (const battle of corpus.slice(0, 2)) {
      for (const team of [battle.teams.p2, mutated(battle.teams.p2)]) {
        for (const point of await points(battle, team)) {
          const text = bytes(featuresAt(point)).toLowerCase();
          const channel = point.lines.join("\n").toLowerCase();
          const hidden = team.flatMap((set) => [set.item, set.nature, ...set.moves.slice(1)]).filter((name) => name !== "" && !channel.includes(name.toLowerCase())).map((name) => name.toLowerCase());
          expect(hidden.length).toBeGreaterThan(0);
          for (const marker of hidden) expect(text.includes(marker), `${battle.id} request ${point.seq}`).toBe(false);
          scanned += 1;
        }
      }
    }
    expect(scanned).toBeGreaterThanOrEqual(8);
    expect(jcs({ rows: [], marker: "Choice Scarf" } as unknown as Json).toLowerCase().includes("choice scarf")).toBe(true); // the scan sees a marker when one is there
  }, 280_000);
});

describe("play time and stream time (ADR-022 section 3)", () => {
  it("gives the rows of a decision the same bytes with a game-long memo (play) and a memo of its own (recomputation)", async () => {
    const battle = corpus[1] as RandomBattleFile;
    const all = await points(battle, battle.teams.p2);
    const game = createFeatureMemo();
    let compared = 0;
    for (const point of all) {
      expect(bytes(featuresAt(point, {}, createFeatureMemo())), `request ${point.seq}`).toBe(bytes(featuresAt(point, {}, game)));
      compared += 1;
    }
    expect(compared).toBeGreaterThanOrEqual(3);
    expect(game.belief?.decisions).toBe(compared);
  }, 120_000);
});

describe("the set identity names the belief", () => {
  it("is another hash than probe-damage/1 and moves with the world count and the belief", () => {
    const hash = (change: Partial<FeatureConfigs>, set: "probe-damage/1" | typeof FEATURE_SET_BELIEF) => featureSetHash(featureSetIdentity(engine, configs(change), set));
    const belief = hash({}, FEATURE_SET_BELIEF);
    expect(belief).not.toBe(hash({}, "probe-damage/1"));
    expect(hash({ belief: { port: createBeliefWorlds(context), worlds: WORLDS + 1 } }, FEATURE_SET_BELIEF)).not.toBe(belief);
    expect(hash({ belief: { port: { ...createBeliefWorlds(context), identity_hash: "0".repeat(64) }, worlds: WORLDS } }, FEATURE_SET_BELIEF)).not.toBe(belief);
    // the real port's identity is ADR-025's belief identity, and it is in the preimage
    expect(featureSetIdentity(engine, configs(), FEATURE_SET_BELIEF).belief_identity_hash).toBe(createBeliefWorlds(context).identity_hash);
  });
});
