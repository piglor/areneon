// The uninformative belief mode (#1309, ADR-025 erratum E4; EXP-B1's same-shape control arm) with the real model: the opponent sheet
// block and the `probe-damage/3` move inclusions are built from the prior alone, so they are equal at every request of a game whatever
// the channel reveals, they carry the mode's own hashes, and the informative port keeps its bytes.
import { EVIDENCE_NONE, SHEET_K, sheetFeatureSetHash, uninformativeIdentityHash, UNINFORMATIVE_FORMAT, uninformativeOpponent } from "@areneon/opponent-model";
import {
  BELIEF_THREATS_PER_FOE,
  type BeliefThreatsPort,
  candidateFeatures3,
  createSheetBuilder,
  decodeRequest,
  enumerateCandidates,
  type FeatureConfigs,
  loadRulesSnapshot,
  playerChannel,
  readOpponentSheet,
  resolveFormat,
} from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { interpretChannel, readerSha256 } from "../../public-state/src/index.ts";
import { jcs } from "../../showdown-bridge/src/canonical-json.ts";
import { loadDamageAwareConfig } from "../../showdown-bridge/src/damage-aware-config.ts";
import type { PokemonSet } from "../../showdown-bridge/src/engine.ts";
import { goldenTeamVocabulary } from "../../showdown-bridge/src/golden.ts";
import { loadRandomCorpus } from "../../showdown-bridge/src/random-corpus.ts";
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

  it("refuses where the informative belief refuses for a reader reason (a flag, a phantom entry) and reads no other evidence, a trapping check included", () => {
    const c = firstCase();
    const builder = createSheetBuilder({ engine, vocabulary, port: blind, scenario_id });
    const plain = JSON.stringify(blind.sheet({ opponent: c.input.state.opponent, flags: [] }));
    const flag = [{ index: 1, line: "x", reason: "x" }] as never;
    const flagged = builder.build({ request: c.input.request, team: c.input.team, state: c.input.state, flags: flag });
    expect(flagged.source).toEqual({ kind: "belief", hash: blind.features_hash, refusal: "reader_flagged" });
    expect(flagged.slots.length).toBe(c.input.state.opponent.roster.length);
    const informativeFlagged = createSheetBuilder({ engine, vocabulary, port: informative, scenario_id }).build({ request: c.input.request, team: c.input.team, state: c.input.state, flags: flag });
    expect(informativeFlagged.source).toMatchObject({ refusal: "reader_flagged" });
    const withPhantom = structuredClone(c.input.state);
    withPhantom.opponent.roster.push({ ...structuredClone(rosterMon(withPhantom)), phantom: true });
    expect(blind.sheet({ opponent: withPhantom.opponent, flags: [] }).source).toMatchObject({ refusal: "reader_flagged" });
    expect(informative.sheet({ opponent: withPhantom.opponent, flags: [] }).source).toMatchObject({ refusal: "reader_flagged" });
    expect(blind.moveInclusions({ opponent: withPhantom.opponent, flags: [] })).toEqual({ refused: "reader_flagged" });
    expect(blind.moveInclusions({ opponent: c.input.state.opponent, flags: flag })).toEqual({ refused: "reader_flagged" });
    // no other evidence: a trapping check that observes something changes nothing
    const trapped = JSON.stringify(blind.sheet({ opponent: c.input.state.opponent, flags: [], trapping: { observed: [1], shows: () => [0] } as never }));
    expect(trapped).toBe(plain);
  });
});

// ---- probe-damage/3 over a game: which parts are constant -------------------------------------------------------------------------

const configsFor = (threats: BeliefThreatsPort): FeatureConfigs => ({
  vocabulary,
  config: loadDamageAwareConfig("v1", vocabulary),
  mechanics: loadMechanicsConfig(FORMAT),
  snapshot: loadRulesSnapshot(engine.commit),
  data_class: "synthetic",
  belief_threats: { port: threats, per_foe: BELIEF_THREATS_PER_FOE },
});

const withGenders = (team: PokemonSet[]): PokemonSet[] =>
  team.map((set) => {
    const fixed = (engine.sim.Dex.mod("champions").species.get(set.species) as unknown as { gender: string }).gender;
    return { ...structuredClone(set), gender: set.gender !== "" ? set.gender : fixed !== "" ? fixed : "M" };
  });

/** The `/3` features at every move or forced-switch request of a channel, in game order. */
function featuresOf(lines: readonly string[], team: PokemonSet[], threats: BeliefThreatsPort): NonNullable<ReturnType<typeof candidateFeatures3>>[] {
  const out: NonNullable<ReturnType<typeof candidateFeatures3>>[] = [];
  lines.forEach((line, i) => {
    if (!line.startsWith("|request|")) return;
    const decoded = decodeRequest(line.slice("|request|".length));
    if (decoded.phase !== "move" && decoded.phase !== "forceSwitch") return;
    const candidates = enumerateCandidates(decoded).candidates.map(({ choice, pending }) => ({ choice, pending }));
    out.push(candidateFeatures3(engine, configsFor(threats), playerChannel({ schema_version: 1, data_class: "synthetic", player: "p1", lines: lines.slice(0, i + 1) as string[] }), team as never, candidates));
  });
  return out;
}

describe("probe-damage/3 under the uninformative belief is partly informative by design", () => {
  const battles = loadRandomCorpus().slice(0, 3);
  const runs = battles.map((battle) => featuresOf(battle.channels.p1.lines, withGenders(battle.teams.p1), blind));
  const informativeRuns = battles.map((battle) => featuresOf(battle.channels.p1.lines, withGenders(battle.teams.p1), informative));
  const part = (key: "belief_threats" | "threats" | "speeds" | "own_priorities" | "foe_priorities") => (f: (typeof runs)[number][number]) => jcs(f[key] as never);

  it("the per-roster inclusion lists are constant; the rows built from them, the own state and the /2 arrays are not", () => {
    expect(runs.every((r) => r.length >= 4)).toBe(true);
    for (const [index, battle] of battles.entries()) {
      const first = (battle.channels.p1.lines.map((l, i) => (l.startsWith("|request|") ? i : -1)).find((i) => i >= 0) ?? 0) + 1;
      const state = interpretChannel("p1", battle.channels.p1.lines.slice(0, first)).frames.at(-1)?.state;
      const lists = (blind.moveInclusions({ opponent: (state as NonNullable<typeof state>).opponent, flags: [] }) as { slots: { id: string; inclusion_ppm: number }[][] }).slots;
      const allowed = new Set(lists.flat().map((m) => `${m.id}:${m.inclusion_ppm}`));
      // CONSTANT: every belief row's (move, inclusion) is one of the roster's fixed inclusion entries (the same at every request of the game)
      for (const f of runs[index] ?? []) for (const row of f.belief_threats) expect(allowed.has(`${row.move}:${row.inclusion_ppm}`)).toBe(true);
    }
    // NOT constant: across the games, the belief rows, the revealed-move arrays and the speed/priority arrays change along a game
    const varies = (key: Parameters<typeof part>[0]) => runs.some((r) => new Set(r.map(part(key))).size > 1);
    for (const key of ["belief_threats", "threats", "speeds", "own_priorities", "foe_priorities"] as const) expect(varies(key), key).toBe(true);
  }, 300_000);

  it("the two arms differ only in the belief: the /2 arrays and the rows are equal where both arms have no belief threats, and the hashes differ", () => {
    for (const [i, r] of runs.entries()) {
      for (const [k, f] of r.entries()) {
        const g = (informativeRuns[i] ?? [])[k] as (typeof r)[number];
        for (const key of ["threats", "speeds", "own_priorities", "foe_priorities"] as const) expect(part(key)(f)).toBe(part(key)(g));
        expect(jcs(f.rows as never)).toBe(jcs(g.rows as never));
      }
    }
    expect(runs.some((r, i) => r.some((f, k) => jcs(f.belief_threats as never) !== jcs((informativeRuns[i] ?? [])[k]?.belief_threats as never))).valueOf()).toBe(true);
  }, 300_000);

  it("a roster with an Illusion twin (a phantom entry) from an earlier request neither crashes nor leaks: the arm refuses, and a port that does not refuse is skipped for the phantom foe", () => {
    // A twin is forged into a real channel: the first p2a switch-in shown again in slot b, before the third request; features are asked at the fifth.
    // A port without the refusal (slots for the non-phantom entries only, empty lists) reaches the phantom foe in the builder.
    const lenient: BeliefThreatsPort = { identity_hash: blind.identity_hash, moveInclusions: ({ opponent }) => ({ slots: opponent.roster.filter((m) => m.phantom !== true).map(() => []) }) };
    let withPhantom = 0;
    let refused = 0;
    let skipped = 0;
    for (const battle of loadRandomCorpus().slice(0, 6)) {
      const lines = battle.channels.p1.lines;
      const requests = lines.map((l, i) => (l.startsWith("|request|") ? i : -1)).filter((i) => i >= 0);
      const at = requests[2];
      const end = requests[4];
      const twin = lines.find((l) => l.startsWith("|switch|p2a:"));
      if (at === undefined || end === undefined || twin === undefined) continue;
      const forged = [...lines.slice(0, at), twin.replace("|switch|p2a:", "|switch|p2b:"), ...lines.slice(at, end + 1)];
      const decoded = decodeRequest((forged.at(-1) as string).slice("|request|".length));
      if (decoded.phase !== "move") continue;
      const state = interpretChannel("p1", forged).frames.at(-1)?.state;
      if (state === undefined || !state.opponent.roster.some((m) => m.phantom === true)) continue;
      withPhantom += 1;
      const candidates = enumerateCandidates(decoded).candidates.map(({ choice, pending }) => ({ choice, pending }));
      const channel = playerChannel({ schema_version: 1, data_class: "synthetic", player: "p1", lines: forged });
      const team = withGenders(battle.teams.p1);
      const viaBlind = candidateFeatures3(engine, configsFor(blind), channel, team as never, candidates);
      expect(viaBlind.belief_threats).toEqual([]);
      expect(viaBlind.belief_threats_refused).toBe("reader_flagged");
      refused += 1;
      // the informative arm refuses the same request the same way (refusal parity)
      expect(candidateFeatures3(engine, configsFor(informative), channel, team as never, candidates).belief_threats_refused).toBe("reader_flagged");
      const viaLenient = candidateFeatures3(engine, configsFor(lenient), channel, team as never, candidates);
      expect(viaLenient.belief_threats_refused).toBeUndefined();
      skipped += 1;
    }
    expect(withPhantom).toBeGreaterThan(0);
    expect(refused).toBe(withPhantom);
    expect(skipped).toBe(withPhantom);
  }, 300_000);
});
