// `probe-damage/3` with the real opponent model (ADR-022, ADR-025 section 3): the composition root's `moveInclusions` and the
// builder's `belief_threats` through the real port and the real probe. Privacy: battles whose player channels are equal up to a request
// but whose opponents hold different unrevealed sets give byte-identical features (rows, arrays, belief threats and hash); a port
// handed the real opponent team gives different ones (negative control); a real unrevealed move the belief does not propose appears in
// no belief threat of the real port and does in the leaking one (canary).
import {
  BELIEF_THREATS_PER_FOE,
  type BeliefThreatsPort,
  candidateFeatures3,
  decodeRequest,
  type FeatureConfigs,
  enumerateCandidates,
  FEATURE_SET_3,
  loadMechanicsConfig,
  loadRulesSnapshot,
  playerChannel,
  resolveFormat,
} from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { interpretChannel, readerSha256, toId } from "../../public-state/src/index.ts";
import { actorRule } from "../../showdown-bridge/src/actor-rule.ts";
import { jcs } from "../../showdown-bridge/src/canonical-json.ts";
import { loadDamageAwareConfig } from "../../showdown-bridge/src/damage-aware-config.ts";
import type { PokemonSet } from "../../showdown-bridge/src/engine.ts";
import { goldenTeamVocabulary } from "../../showdown-bridge/src/golden.ts";
import { loadRandomCorpus, type RandomBattleFile } from "../../showdown-bridge/src/random-corpus.ts";
import { type BattleRecord, type RequestRecord, runBattle } from "../../showdown-bridge/src/runner.ts";
import { evidenceKey, createBeliefFeatures } from "../src/belief-features.ts";
import { engine, FORMAT } from "./support.ts";

const vocabulary = goldenTeamVocabulary();
const manifest = resolveFormat(engine, FORMAT, loadRulesSnapshot(engine.commit)).manifest;
const port = createBeliefFeatures({ engine, vocabulary, reader_sha256: readerSha256(), manifest });
const MAX_TURNS = 5;
const battles: RandomBattleFile[] = loadRandomCorpus().slice(0, 3);
const configs = (threats: BeliefThreatsPort = port): FeatureConfigs => ({
  vocabulary,
  config: loadDamageAwareConfig("v1", vocabulary),
  mechanics: loadMechanicsConfig(FORMAT),
  snapshot: loadRulesSnapshot(engine.commit),
  data_class: "synthetic",
  belief_threats: { port: threats, per_foe: BELIEF_THREATS_PER_FOE },
});

describe("the port's moveInclusions", () => {
  it("shares the belief identity with the sheet and refuses (null) where the model refuses", () => {
    const opponent = interpretChannel("p1", ["|player|p1|Me|1", "|player|p2|Foe|1", "|gametype|doubles"]).frames.at(-1)?.state.opponent;
    expect(port.identity_hash).toMatch(/^[0-9a-f]{64}$/);
    expect(opponent).toBeDefined();
    // Not a full Team Preview: the model refuses, and the port says so with null instead of throwing.
    expect(port.moveInclusions({ opponent: opponent as never, flags: [] })).toEqual({ refused: "team_size" });
  });

  it("answers a real request like the model does, and the memo gives the same answer twice", async () => {
    const record = await play(battles[0] as RandomBattleFile, (battles[0] as RandomBattleFile).teams.p2, "keep");
    const prefix = cuts(record)[2] as string[];
    const state = (interpretChannel("p1", prefix).frames.at(-1) as { state: ReturnType<typeof interpretChannel>["frames"][number]["state"] }).state;
    const first = port.moveInclusions({ opponent: state.opponent, flags: [] });
    expect(first).not.toHaveProperty("refused");
    expect(JSON.stringify(port.moveInclusions({ opponent: state.opponent, flags: [] }))).toBe(JSON.stringify(first));
    const lists = (first as { slots: { id: string; inclusion_ppm: number }[][] }).slots;
    expect(lists.length).toBe(state.opponent.roster.length);
    // sorted by inclusion then id, no revealed move
    for (const [position, list] of lists.entries()) {
      const shown = (state.opponent.roster[position] as { moves_revealed: string[] }).moves_revealed;
      for (const m of list) expect(shown).not.toContain(m.id);
      for (const [a, b] of list.slice(1).map((m, i) => [list[i] as { id: string; inclusion_ppm: number }, m] as const)) expect(a.inclusion_ppm > b.inclusion_ppm || (a.inclusion_ppm === b.inclusion_ppm && a.id < b.id)).toBe(true);
    }
    // a flagged line is a refusal of the model
    expect(port.moveInclusions({ opponent: state.opponent, flags: [{ index: 0, line: "|x|", reason: "r" } as never] })).toEqual({ refused: "reader_flagged" });
    expect(evidenceKey(state.opponent, [])).toBe(evidenceKey(state.opponent, []));
  }, 120_000);
});

const withGenders = (team: PokemonSet[]): PokemonSet[] =>
  team.map((set) => {
    const fixed = (engine.sim.Dex.mod("champions").species.get(set.species) as unknown as { gender: string }).gender;
    return { ...structuredClone(set), gender: set.gender !== "" ? set.gender : fixed !== "" ? fixed : "M" };
  });

/** p2's team with fields nobody can see before they act changed: nature, Stat Points, the held items (rotated) and moves 2 to 4. */
function mutated(team: PokemonSet[]): PokemonSet[] {
  const validator = engine.sim.TeamValidator.get(FORMAT);
  const natures = vocabulary.natures.map((n) => n.name);
  const items = team.map((set) => set.item);
  const out = team.map((set, index) => {
    const copy = structuredClone(set);
    copy.nature = natures[(natures.indexOf(set.nature) + 7 + index) % natures.length] as string;
    copy.evs = { ...set.evs, hp: set.evs.spe, spe: set.evs.hp, atk: set.evs.def, def: set.evs.atk };
    copy.item = items[(index + 1) % items.length] as string;
    return copy;
  });
  out.forEach((set) => {
    const pool = vocabulary.roster.find((e) => e.id === toId(set.species))?.moves ?? [];
    for (let slot = 1; slot < set.moves.length; slot++) {
      const replacement = pool.find((id) => !set.moves.map(toId).includes(id));
      if (replacement === undefined) continue;
      const before = set.moves[slot] as string;
      set.moves[slot] = (engine.sim.Dex.mod("champions").moves.get(replacement) as { name: string }).name;
      if ((validator.validateTeam(structuredClone(out)) ?? []).length > 0) set.moves[slot] = before;
    }
  });
  return out;
}

const firstLegal = (request: RequestRecord): string => {
  if (request.legal === undefined) throw new Error("no legal set");
  return actorRule(request.legal);
};
const play = (battle: RandomBattleFile, p2: PokemonSet[], preview: "keep" | "other-back"): Promise<BattleRecord> =>
  runBattle({
    engine,
    formatId: FORMAT,
    seed: battle.seed,
    teams: { p1: withGenders(battle.teams.p1), p2 },
    actors: {
      p1: firstLegal,
      p2: (request) => {
        const first = firstLegal(request);
        return preview === "other-back" && first.startsWith("team ") && request.legal?.choices.includes("team 1256") ? "team 1256" : first;
      },
    },
    maxTurns: MAX_TURNS,
  });

/** The p1 channel prefixes ending in a request. */
const cuts = (record: BattleRecord): string[][] => record.channels.p1.flatMap((line, i) => (line.startsWith("|request|") ? [record.channels.p1.slice(0, i + 1)] : []));

/** The prefixes at which both battles' p1 channels are identical. */
function equalPrefixes(a: BattleRecord, b: BattleRecord): [string[], string[]][] {
  const left = cuts(a);
  const right = cuts(b);
  const out: [string[], string[]][] = [];
  for (let k = 0; k < Math.min(left.length, right.length); k++) {
    if (JSON.stringify(left[k]) !== JSON.stringify(right[k])) break;
    out.push([left[k] as string[], right[k] as string[]]);
  }
  return out;
}

/** The `/3` features of a prefix ending in a move or forced-switch request, or null (preview, wait). */
function featuresAt(lines: string[], team: PokemonSet[], threats: BeliefThreatsPort = port): ReturnType<typeof candidateFeatures3> | null {
  const decoded = decodeRequest((lines.at(-1) as string).slice("|request|".length));
  if (decoded.phase !== "move" && decoded.phase !== "forceSwitch") return null;
  const candidates = enumerateCandidates(decoded).candidates.map(({ choice, pending }) => ({ choice, pending }));
  return candidateFeatures3(engine, configs(threats), playerChannel({ schema_version: 1, data_class: "synthetic", player: "p1", lines }), team as never, candidates);
}

describe("invariance through the real port and the real probe", () => {
  it("features are byte-identical at every request with an equal player channel, whatever the opponent holds unrevealed", async () => {
    let compared = 0;
    let withBelief = 0;
    for (const battle of battles) {
      const theirs = withGenders(battle.teams.p2);
      const other = mutated(theirs);
      expect(JSON.stringify(other)).not.toBe(JSON.stringify(theirs));
      const a = await play(battle, theirs, "keep");
      const b = await play(battle, other, "other-back");
      for (const [mine, yours] of equalPrefixes(a, b)) {
        const one = featuresAt(mine, withGenders(battle.teams.p1));
        const two = featuresAt(yours, withGenders(battle.teams.p1));
        expect(two === null ? null : jcs(two as never)).toBe(one === null ? null : jcs(one as never));
        if (one === null) continue;
        expect(one.feature_set).toBe(FEATURE_SET_3);
        compared += 1;
        if (one.belief_threats.length > 0) withBelief += 1;
      }
    }
    expect(compared).toBeGreaterThanOrEqual(4);
    expect(withBelief).toBeGreaterThan(0);
  }, 400_000);

  it("has teeth, and a real unrevealed move appears only when the port is handed the real team", async () => {
    let real: PokemonSet[] = [];
    // The leak: the belief's inclusions replaced by the real team's moves (what a builder given the opponent's team would say).
    const leaking: BeliefThreatsPort = {
      identity_hash: port.identity_hash,
      moveInclusions: ({ opponent }) => ({
        slots: opponent.roster.map((mon) => {
          const set = real.find((s) => [mon.species, mon.base_species].includes(toId(s.species)));
          return (set?.moves ?? []).map(toId).filter((id) => !mon.moves_revealed.includes(id)).map((id) => ({ id, inclusion_ppm: 1_000_000 }));
        }),
      }),
    };
    let differing = 0;
    let canaryHits = 0;
    let canaryChecked = 0;
    for (const battle of battles) {
      const theirs = withGenders(battle.teams.p2);
      const other = mutated(theirs);
      const a = await play(battle, theirs, "keep");
      const b = await play(battle, other, "other-back");
      for (const [mine, yours] of equalPrefixes(a, b)) {
        real = theirs;
        const one = featuresAt(mine, withGenders(battle.teams.p1), leaking);
        const honest = featuresAt(mine, withGenders(battle.teams.p1));
        real = other;
        const two = featuresAt(yours, withGenders(battle.teams.p1), leaking);
        if (one === null || two === null || honest === null) continue;
        if (jcs(one as never) !== jcs(two as never)) differing += 1;
        // canary: a real move the honest belief does not propose never shows in its rows, and does in the leaking one
        const proposed = new Set(honest.belief_threats.map((r) => r.move));
        const state = (interpretChannel("p1", mine).frames.at(-1) as { state: ReturnType<typeof interpretChannel>["frames"][number]["state"] }).state;
        const realMoves = new Set(theirs.flatMap((s) => s.moves.map(toId)).filter((m) => !state.opponent.roster.some((mon) => mon.moves_revealed.includes(m))));
        for (const move of realMoves) {
          if (proposed.has(move)) continue;
          canaryChecked += 1;
          expect(JSON.stringify(honest.belief_threats).includes(`"move":"${move}"`)).toBe(false);
          if (one.belief_threats.some((r) => r.move === move)) canaryHits += 1;
        }
      }
    }
    expect(differing).toBeGreaterThan(0);
    expect(canaryChecked).toBeGreaterThan(0);
    expect(canaryHits).toBeGreaterThan(0);
  }, 400_000);
});
