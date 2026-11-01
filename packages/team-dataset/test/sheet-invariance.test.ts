// ADR-025 erratum E4 point 3: the opponent sheet block is a pure function of the player's own view. Through the real port, the real
// trapping check of the player's own move request and `sheetForChannel` (the function the seat and the loader call): battles whose
// player channels are equal up to a request but whose opponents hold different unrevealed sets give byte-identical blocks; a builder
// handed the real opponent team differs (the negative control); marker values placed only in unrevealed fields never appear.
import { createSheetBuilder, loadMechanicsConfig, loadRulesSnapshot, resolveFormat, sheetForChannel, type BeliefFeaturesPort } from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { interpretChannel, readerSha256, toId } from "../../public-state/src/index.ts";
import { actorRule } from "../../showdown-bridge/src/actor-rule.ts";
import { jcs } from "../../showdown-bridge/src/canonical-json.ts";
import type { PokemonSet } from "../../showdown-bridge/src/engine.ts";
import { goldenTeamVocabulary } from "../../showdown-bridge/src/golden.ts";
import { checkScenario } from "../../showdown-bridge/src/mechanics-scenario.ts";
import { loadRandomCorpus, type RandomBattleFile } from "../../showdown-bridge/src/random-corpus.ts";
import { type BattleRecord, type RequestRecord, runBattle } from "../../showdown-bridge/src/runner.ts";
import { createBeliefFeatures } from "../src/belief-features.ts";
import { engine, FORMAT } from "./support.ts";

const vocabulary = goldenTeamVocabulary();
const manifest = resolveFormat(engine, FORMAT, loadRulesSnapshot(engine.commit)).manifest;
const port = createBeliefFeatures({ engine, vocabulary, reader_sha256: readerSha256(), manifest });
const scenario_id = checkScenario(engine, loadMechanicsConfig(FORMAT));
const MAX_TURNS = 5;
const battles: RandomBattleFile[] = loadRandomCorpus().slice(0, 3);

/** The port with a spy on the trapping check it is given (a check that observes something is what the memo must not cover). */
function spied(): { port: BeliefFeaturesPort; informative: () => number; checks: () => number } {
  let informative = 0;
  let checks = 0;
  return {
    port: {
      ...port,
      sheet(input) {
        if (input.trapping !== null && input.trapping !== undefined) {
          checks += 1;
          if (input.trapping.observed.length > 0) informative += 1;
        }
        return port.sheet(input);
      },
    },
    informative: () => informative,
    checks: () => checks,
  };
}

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

/** The p1 channel prefixes ending in a request at which both battles' channels are identical, as each battle's own lines. */
function equalPrefixes(a: BattleRecord, b: BattleRecord): [string[], string[]][] {
  const cuts = (record: BattleRecord): string[][] => record.channels.p1.flatMap((line, i) => (line.startsWith("|request|") ? [record.channels.p1.slice(0, i + 1)] : []));
  const left = cuts(a);
  const right = cuts(b);
  const out: [string[], string[]][] = [];
  for (let k = 0; k < Math.min(left.length, right.length); k++) {
    if (JSON.stringify(left[k]) !== JSON.stringify(right[k])) break;
    out.push([left[k] as string[], right[k] as string[]]);
  }
  return out;
}

describe("invariance through the real port, the real trapping check and sheetForChannel", () => {
  it("blocks are byte-identical at every request with an equal player channel, whatever the opponent holds unrevealed", async () => {
    const spy = spied();
    const builder = createSheetBuilder({ engine, vocabulary, port: spy.port, scenario_id });
    let compared = 0;
    let beyondPreview = 0;
    let revealed = 0;
    for (const battle of battles) {
      const theirs = withGenders(battle.teams.p2);
      const other = mutated(theirs);
      expect(JSON.stringify(other)).not.toBe(JSON.stringify(theirs));
      const a = await play(battle, theirs, "keep");
      const b = await play(battle, other, "other-back");
      expect(JSON.stringify(a.requests.p2[0])).not.toBe(JSON.stringify(b.requests.p2[0])); // the opponents really differ
      for (const [mine, yours] of equalPrefixes(a, b)) {
        const one = jcs(sheetForChannel(builder, { player: "p1", lines: mine, team: withGenders(battle.teams.p1) }) as never);
        const two = jcs(sheetForChannel(builder, { player: "p1", lines: yours, team: withGenders(battle.teams.p1) }) as never);
        expect(two).toBe(one);
        compared += 1;
        if ((interpretChannel("p1", mine).frames.at(-1)?.state.turn ?? 0) > 0) beyondPreview += 1;
        if (one.includes('"brought_source":"revealed"')) revealed += 1;
      }
    }
    expect(compared).toBeGreaterThanOrEqual(6);
    expect(beyondPreview).toBeGreaterThanOrEqual(3);
    expect(revealed).toBeGreaterThan(0);
    expect(spy.checks()).toBeGreaterThan(0); // the real trapping check was given to the port at move requests
  }, 240_000);

  it("has teeth: a port handed the opponent's real team gives different blocks for the two teams", async () => {
    let real: PokemonSet[] = [];
    const leaking: BeliefFeaturesPort = {
      ...port,
      sheet(input) {
        const block = port.sheet(input);
        const item = toId((real[0] as PokemonSet).item);
        // The leak: the first slot's item written from the real set.
        return { ...block, slots: block.slots.map((s, i) => (i === 0 ? { ...s, item: { source: "predicted" as const, entries: [{ id: item === "" ? null : item, ppm: 1_000_000 }], other_ppm: 0 } } : s)) } as typeof block;
      },
    };
    const builder = createSheetBuilder({ engine, vocabulary, port: leaking, scenario_id });
    let differing = 0;
    for (const battle of battles) {
      const theirs = withGenders(battle.teams.p2);
      const other = mutated(theirs);
      const a = await play(battle, theirs, "keep");
      const b = await play(battle, other, "other-back");
      for (const [mine, yours] of equalPrefixes(a, b)) {
        real = theirs;
        const one = jcs(sheetForChannel(builder, { player: "p1", lines: mine, team: withGenders(battle.teams.p1) }) as never);
        real = other;
        const two = jcs(sheetForChannel(builder, { player: "p1", lines: yours, team: withGenders(battle.teams.p1) }) as never);
        if (one !== two) differing += 1;
      }
    }
    expect(differing).toBeGreaterThan(0);
  }, 240_000);
});
