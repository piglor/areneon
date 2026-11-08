// The uninformative belief mode (#1309, ADR-025 erratum E4; EXP-B1's same-shape control arm) with the real model: the opponent sheet
// block and the `probe-damage/3` move inclusions are built from the prior alone, so they are equal at every request of a game whatever
// the channel reveals, they carry the mode's own hashes, and the informative port keeps its bytes.
import { EVIDENCE_NONE, SHEET_K, sheetFeatureSetHash, uninformativeIdentityHash, UNINFORMATIVE_FORMAT, uninformativeOpponent } from "@areneon/opponent-model";
import { createSheetBuilder, loadRulesSnapshot, readOpponentSheet, resolveFormat } from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { readerSha256 } from "../../public-state/src/index.ts";
import { goldenTeamVocabulary } from "../../showdown-bridge/src/golden.ts";
import { loadMechanicsConfig } from "../../showdown-bridge/src/mechanics-config.ts";
import { checkScenario } from "../../showdown-bridge/src/mechanics-scenario.ts";
import { cases } from "../../showdown-bridge/test/world-builder-support.ts";
import { createBeliefFeatures } from "../src/belief-features.ts";
import { engine, FORMAT } from "./support.ts";

const vocabulary = goldenTeamVocabulary();
const manifest = resolveFormat(engine, FORMAT, loadRulesSnapshot(engine.commit)).manifest;
const context = { engine, vocabulary, reader_sha256: readerSha256(), manifest };
const informative = createBeliefFeatures(context);
const blind = createBeliefFeatures({ ...context, evidence: EVIDENCE_NONE });
const scenario_id = checkScenario(engine, loadMechanicsConfig(FORMAT));

// The decisions of one seat of one battle, in game order (the id is `battle/player/frame`).
const games = new Map<string, ReturnType<typeof cases>>();
for (const c of cases()) {
  const key = c.id.slice(0, c.id.lastIndexOf("/"));
  games.set(key, [...(games.get(key) ?? []), c]);
}
const longGames = [...games.values()].filter((g) => g.length >= 4).slice(0, 8);
type Case = ReturnType<typeof cases>[number];
/** The first decision of the first long game. */
const firstCase = (): Case => {
  const found = longGames[0]?.[0];
  if (found === undefined) throw new Error("the corpus has no long game");
  return found;
};
const rosterMon = (state: Case["input"]["state"], i = 0): Case["input"]["state"]["opponent"]["roster"][number] => {
  const mon = state.opponent.roster[i];
  if (mon === undefined) throw new Error("no roster entry");
  return mon;
};

describe("identity", () => {
  it("the mode has its own belief identity, sheet source hash and format; the informative port is unchanged", () => {
    expect(blind.identity_hash).toBe(uninformativeIdentityHash(informative.identity_hash));
    expect(blind.identity_hash).not.toBe(informative.identity_hash);
    expect(blind.features_hash).toBe(sheetFeatureSetHash(blind.identity_hash, SHEET_K));
    expect(blind.features_hash).not.toBe(informative.features_hash);
    expect(informative.features_hash).toBe(sheetFeatureSetHash(informative.identity_hash, SHEET_K));
    expect(blind.prior_hash).toBe(informative.prior_hash);
    expect(UNINFORMATIVE_FORMAT).toBe("belief-uninformative/1");
    expect(() => uninformativeIdentityHash("nope")).toThrow(RangeError);
  });
});

describe("the blank opponent", () => {
  it("keeps the preview species and nothing else, and drops a phantom entry", () => {
    const state = firstCase().input.state;
    const opponent = structuredClone(state.opponent);
    opponent.roster.push({ ...structuredClone(rosterMon(state)), phantom: true });
    const before = JSON.stringify(opponent);
    const blank = uninformativeOpponent(opponent);
    expect(JSON.stringify(opponent)).toBe(before);
    expect(blank.roster.map((m) => m.base_species)).toEqual(state.opponent.roster.map((m) => m.base_species));
    for (const mon of blank.roster) {
      expect(mon).toMatchObject({ name: null, species: null, level: null, gender: null, brought: null, fainted: false, active_slot: null, item: null, ability: null, mega: false, moves_revealed: [], volatiles: [] });
    }
    expect(blank.slots).toEqual({});
  });
});

describe("a game under the uninformative mode", () => {
  it("every sheet block and every move inclusion list is the same at every request, whatever the channel reveals", () => {
    expect(longGames.length).toBeGreaterThan(3);
    const builder = createSheetBuilder({ engine, vocabulary, port: blind, scenario_id });
    let revealed = 0;
    for (const game of longGames) {
      const blocks = game.map((c) => builder.build({ request: c.input.request, team: c.input.team, state: c.input.state, flags: c.input.flags }));
      const inclusions = game.map((c) => JSON.stringify(blind.moveInclusions({ opponent: c.input.state.opponent, flags: c.input.flags })));
      blocks.forEach((b, i) => {
        expect(() => readOpponentSheet(b)).not.toThrow();
        expect(b.source).toEqual({ kind: "belief", hash: blind.features_hash });
        expect(JSON.stringify(b), game[i]?.id).toBe(JSON.stringify(blocks[0]));
        expect(inclusions[i], game[i]?.id).toBe(inclusions[0]);
      });
      // No fact is revealed in any slot, and no move is known.
      for (const slot of (blocks[0]?.slots ?? [])) {
        expect(slot.brought_source).toBe("predicted");
        expect(slot.item.source).not.toBe("revealed");
        expect(slot.ability.source).not.toBe("revealed");
        expect(slot.moves.slots_known).toBe(0);
      }
      // The channel does reveal things later in the game (the control would be vacuous otherwise).
      revealed += game.filter((c) => c.input.state.opponent.roster.some((m) => m.moves_revealed.length > 0 || m.brought === true || m.active_slot !== null)).length;
    }
    expect(revealed).toBeGreaterThan(0);
  });

  it("negative control: the informative port differs along the same games", () => {
    const builder = createSheetBuilder({ engine, vocabulary, port: informative, scenario_id });
    let differing = 0;
    for (const game of longGames) {
      const blocks = game.map((c) => JSON.stringify(builder.build({ request: c.input.request, team: c.input.team, state: c.input.state, flags: c.input.flags }).slots));
      if (new Set(blocks).size > 1) differing++;
    }
    expect(differing).toBeGreaterThan(0);
  });

  it("a flag, a phantom entry and a trapping check change nothing", () => {
    const c = firstCase();
    const builder = createSheetBuilder({ engine, vocabulary, port: blind, scenario_id });
    const plain = JSON.stringify(builder.build({ request: c.input.request, team: c.input.team, state: c.input.state, flags: [] }));
    const flagged = JSON.stringify(builder.build({ request: c.input.request, team: c.input.team, state: c.input.state, flags: [{ index: 1, line: "x", reason: "x" }] as never }));
    expect(flagged).toBe(plain);
    const withPhantom = structuredClone(c.input.state);
    withPhantom.opponent.roster.push({ ...structuredClone(rosterMon(withPhantom)), phantom: true });
    expect(JSON.stringify(blind.sheet({ opponent: withPhantom.opponent, flags: [] }).slots)).toBe(JSON.stringify(blind.sheet({ opponent: c.input.state.opponent, flags: [] }).slots));
    expect(blind.sheet({ opponent: c.input.state.opponent, flags: [], trapping: { observed: [1], shows: () => [0] } as never }).source).toEqual({ kind: "belief", hash: blind.features_hash });
  });
});
