// Hidden-information tests of the belief and the sampler on seeded engine battles (ADR-025 section 8; the shape of
// ADR-018 section 3's invariance test): the INVARIANCE of beliefs and sampled worlds, with a negative control that hands the
// filter the opponent's real sets, and the CANARY (marker values placed only in the opponent's unrevealed fields never appear
// in a belief or a world), with a negative control. The bridge and the engine are imported by path for this test only: the
// package itself is engine-free.

import { beforeAll, describe, expect, it } from "vitest";
import { type PublicSide, interpretChannel, toId } from "@areneon/public-state";
import type { UsagePrior } from "@areneon/schema/types/v1/usage-prior.ts";
import { loadEngine } from "../../showdown-bridge/src/index.ts";
import type { Engine, PokemonSet } from "../../showdown-bridge/src/engine.ts";
import { loadRandomCorpus, type RandomBattleFile } from "../../showdown-bridge/src/random-corpus.ts";
import { createSetLegality } from "../../showdown-bridge/src/set-legality.ts";
import { type BattleRecord, type RequestRecord, runBattle } from "../../showdown-bridge/src/runner.ts";
import { actorRule } from "../../showdown-bridge/src/actor-rule.ts";
import { type OpponentBelief, type SetLegality, type SpreadSampler, beliefDigest, createSampler, worldsDigest } from "../src/index.ts";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { readBeliefConfig } from "../../showdown-bridge/src/belief-config.ts";
import { loadDamageAwareConfig } from "../../showdown-bridge/src/damage-aware-config.ts";
import { createDamageCheck } from "../../showdown-bridge/src/damage-check.ts";
import { loadMechanicsConfig } from "../../showdown-bridge/src/mechanics-config.ts";
import { beliefAt, hiding, skewedPrior } from "./corpus.ts";
import { ROOT, realVocabulary } from "./real.ts";
import { text } from "./support.ts";

const FORMAT = realVocabulary.format_id;
const MAX_TURNS = 5;
let engine: Engine;
let battles: RandomBattleFile[];
beforeAll(() => {
  engine = loadEngine();
  battles = loadRandomCorpus().slice(0, 3);
});

// The prior's other-spread entry, drawn from the world's own stream.
const spreads: SpreadSampler = {
  other: (_species, _forme, below) => ({
    nature: (realVocabulary.natures[Number(below(BigInt(realVocabulary.natures.length)))] as { id: string }).id,
    stat_points: { hp: Number(below(33n)), atk: Number(below(33n)), def: 0, spa: 0, spd: 0, spe: Number(below(2n)) },
  }),
};
const stream = (k: number): Uint8Array => Uint8Array.from({ length: 32 }, (_, i) => (k * 31 + i * 7) & 255);

/** The same team with every gender stated: the engine draws an unstated one from the battle seed and the preview line shows it. */
const withGenders = (team: PokemonSet[]): PokemonSet[] =>
  team.map((set) => {
    const fixed = (engine.sim.Dex.mod("champions").species.get(set.species) as unknown as { gender: string }).gender;
    return { ...structuredClone(set), gender: set.gender !== "" ? set.gender : fixed !== "" ? fixed : "M" };
  });

/** p2's team with fields nobody can see before they act changed: nature, Stat Points, the held items (rotated) and moves 2 to 4. */
function mutated(team: PokemonSet[]): PokemonSet[] {
  const validator = engine.sim.TeamValidator.get(FORMAT);
  const natures = realVocabulary.natures.map((n) => n.name);
  const items = team.map((set) => set.item);
  const out = team.map((set, index) => {
    const copy = structuredClone(set);
    copy.nature = natures[(natures.indexOf(set.nature) + 7 + index) % natures.length] as string;
    copy.evs = { ...set.evs, hp: set.evs.spe, spe: set.evs.hp, atk: set.evs.def, def: set.evs.atk };
    copy.item = items[(index + 1) % items.length] as string;
    return copy;
  });
  out.forEach((set) => {
    const pool = realVocabulary.roster.find((e) => e.id === toId(set.species))?.moves ?? [];
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

type Choose = "keep" | "other-back";
/** p2 plays the first legal choice, except that at Team Preview it may keep the same two leads and bring other two Pokemon behind them. */
const p2Actor = (preview: Choose) => (request: RequestRecord): string => {
  const legal = request.legal;
  if (legal === undefined) throw new Error("no legal set");
  const first = actorRule(legal);
  if (preview === "keep" || !first.startsWith("team ")) return first;
  const other = "team 1256";
  return legal.choices.includes(other) ? other : first;
};

const firstLegal = (request: RequestRecord): string => {
  if (request.legal === undefined) throw new Error("no legal set");
  return actorRule(request.legal);
};

function play(battle: RandomBattleFile, p2: PokemonSet[], preview: Choose): Promise<BattleRecord> {
  return runBattle({ engine, formatId: FORMAT, seed: battle.seed, teams: { p1: withGenders(battle.teams.p1), p2 }, actors: { p1: firstLegal, p2: p2Actor(preview) }, maxTurns: MAX_TURNS });
}

/** The p1 requests at which both battles' p1 channels are identical up to and including the request: their prefixes. */
function equalPrefixes(a: BattleRecord, b: BattleRecord): [string[], string[]][] {
  const out: [string[], string[]][] = [];
  const cuts = (record: BattleRecord): string[][] => {
    const lines = record.channels.p1;
    const list: string[][] = [];
    lines.forEach((line, i) => {
      if (line.startsWith("|request|")) list.push(lines.slice(0, i + 1));
    });
    return list;
  };
  const left = cuts(a);
  const right = cuts(b);
  for (let k = 0; k < Math.min(left.length, right.length); k++) {
    if (JSON.stringify(left[k]) !== JSON.stringify(right[k])) break;
    out.push([left[k] as string[], right[k] as string[]]);
  }
  return out;
}

/** The input a leak would make: the opponent half of the state with the real team's unrevealed fields written into it. */
function leaked(opponent: PublicSide, team: readonly PokemonSet[]): PublicSide {
  const copy = structuredClone(opponent);
  copy.roster.forEach((mon, i) => {
    const set = team.find((t) => [mon.species, mon.base_species].includes(toId(t.species))) ?? (team[i] as PokemonSet);
    mon.item = toId(set.item);
    mon.item_known = true;
    mon.moves_revealed = set.moves.map(toId).sort();
  });
  return copy;
}

describe("invariance: the belief and the worlds do not depend on what the opponent holds unrevealed", () => {
  it("are byte-identical at every request with an equal player channel, whatever the opponent's items, natures, Stat Points, moves 2 to 4 and back Pokemon", async () => {
    const legality: SetLegality = createSetLegality({ engine, vocabulary: realVocabulary });
    const sampler = createSampler({ legality, spreads });
    let compared = 0;
    let beyondPreview = 0;
    for (const [n, battle] of battles.entries()) {
      const theirs = withGenders(battle.teams.p2);
      const other = mutated(theirs);
      expect(JSON.stringify(other)).not.toBe(JSON.stringify(theirs)); // the pair really differs in what p2 holds
      const a = await play(battle, theirs, "keep");
      const b = await play(battle, other, "other-back");
      expect(JSON.stringify(a.requests.p2[0])).not.toBe(JSON.stringify(b.requests.p2[0]));
      for (const [mine, yours] of equalPrefixes(a, b)) {
        // each battle's own p1 channel, read by the reader on its own, and the belief of each (one module, interleaved)
        const view = (lines: string[]) => {
          const run = interpretChannel("p1", lines);
          const state = run.frames[run.frames.length - 1]!.state;
          return beliefAt({ state, flags: run.flags, game: "", seat: "p1", request: 0, truth: null }, skewedPrior(state.opponent.roster.map((m) => m.base_species)));
        };
        const one = view(mine);
        const two = view(yours);
        const state = interpretChannel("p1", mine).frames.at(-1)!.state;
        expect(beliefDigest(two)).toBe(beliefDigest(one));
        const s = stream(compared + n);
        expect(worldsDigest(sampler.sampleWorlds(two, s, 3))).toBe(worldsDigest(sampler.sampleWorlds(one, s, 3)));
        compared += 1;
        if (state.turn > 0) beyondPreview += 1;
      }
    }
    expect(compared).toBeGreaterThanOrEqual(6);
    expect(beyondPreview).toBeGreaterThanOrEqual(3);
  }, 240_000);

  it("has teeth: a filter handed the opponent's real sets gives different beliefs for the two teams", async () => {
    let compared = 0;
    let differing = 0;
    for (const battle of battles) {
      const theirs = withGenders(battle.teams.p2);
      const other = mutated(theirs);
      const a = await play(battle, theirs, "keep");
      const b = await play(battle, other, "other-back");
      for (const [lines] of equalPrefixes(a, b)) {
        const run = interpretChannel("p1", lines);
        const state = run.frames[run.frames.length - 1]!.state;
        const prior = skewedPrior(state.opponent.roster.map((m) => m.base_species));
        const mk = (team: PokemonSet[]): OpponentBelief => beliefAt({ state, flags: run.flags, game: "", seat: "p1", request: 0, truth: null }, prior, { opponent: leaked(state.opponent, team) });
        const left = mk(theirs);
        const right = mk(other);
        compared += 1;
        if (beliefDigest(left) !== beliefDigest(right)) differing += 1;
      }
    }
    expect(compared).toBeGreaterThanOrEqual(3);
    expect(differing).toBeGreaterThan(0);
  }, 240_000);
});

describe("invariance with the damage evidence on", () => {
  const damageConfig = readBeliefConfig(JSON.parse(readFileSync(join(ROOT, "config", "opponent-model", FORMAT, "belief.damage-v1.json"), "utf8")), realVocabulary);
  const config = { ...hiding, damage_evidence: damageConfig.damage_evidence! };

  /** The belief of p1 at a request with the damage port built from p1's own lines and team, or from a port the caller supplies. */
  function at(lines: string[], team: PokemonSet[], port?: (events: { id: string; slot: number; role: "attacker" | "defender" }[]) => { events: { id: string; slot: number; role: "attacker" | "defender" }[]; check: (e: never, c: never) => boolean | null }) {
    const run = interpretChannel("p1", lines);
    const state = run.frames[run.frames.length - 1]!.state;
    const real = createDamageCheck({ engine, vocabulary: realVocabulary, config: loadDamageAwareConfig("v1", realVocabulary), mechanics: loadMechanicsConfig(FORMAT), belief: damageConfig, data_class: "synthetic" }, { player: "p1", lines, team, run });
    const damage = port === undefined ? real : real === null ? null : port([...real.events]);
    return beliefAt({ state, flags: run.flags, game: "", seat: "p1", request: 0, truth: null }, skewedPrior(state.opponent.roster.map((m) => m.base_species)), { config, damage: damage as never });
  }

  it("beliefs and worlds are byte-identical at every request with an equal player channel, with events asked of the real probe", async () => {
    const sampler = createSampler({ legality: createSetLegality({ engine, vocabulary: realVocabulary }), spreads });
    let compared = 0;
    let events = 0;
    for (const [n, battle] of battles.entries()) {
      const theirs = withGenders(battle.teams.p2);
      const other = mutated(theirs);
      const a = await play(battle, theirs, "keep");
      const b = await play(battle, other, "other-back");
      for (const [mine, yours] of equalPrefixes(a, b)) {
        const one = at(mine, withGenders(battle.teams.p1));
        const two = at(yours, withGenders(battle.teams.p1));
        expect(beliefDigest(two)).toBe(beliefDigest(one));
        const s = stream(compared + n);
        expect(worldsDigest(sampler.sampleWorlds(two, s, 3))).toBe(worldsDigest(sampler.sampleWorlds(one, s, 3)));
        events += one.damage.events;
        compared += 1;
      }
    }
    expect(compared).toBeGreaterThanOrEqual(6);
    expect(events).toBeGreaterThan(0);
  }, 240_000);

  it("has teeth: a port that consults the real opponent team gives different beliefs for the two teams", async () => {
    let differing = 0;
    for (const battle of battles) {
      const theirs = withGenders(battle.teams.p2);
      const other = mutated(theirs);
      const a = await play(battle, theirs, "keep");
      const b = await play(battle, other, "other-back");
      for (const [mine, yours] of equalPrefixes(a, b)) {
        // The leak: the verdict is any function of the real item (here a residue of its name's character codes).
        const leaking = (team: PokemonSet[]) => (events: { id: string; slot: number; role: "attacker" | "defender" }[]) => ({
          events,
          check: (_e: never, c: { species: string; item: string | null }) => {
            const set = team.find((t) => toId(t.species) === c.species);
            const code = (x: string): number => [...x].reduce((n, ch) => n + ch.charCodeAt(0), 0) % 3;
            return set === undefined ? null : code(c.item ?? "") === code(toId(set.item));
          },
        });
        const left = at(mine, withGenders(battle.teams.p1), leaking(theirs) as never);
        const right = at(yours, withGenders(battle.teams.p1), leaking(other) as never);
        if (beliefDigest(left) !== beliefDigest(right)) differing += 1;
      }
    }
    expect(differing).toBeGreaterThan(0);
  }, 240_000);
});

// ---- the canary ----

type Forme = UsagePrior["species"][number]["formes"][number];

describe("canary: marker values in the opponent's unrevealed fields never reach a belief or a world", () => {
  /** A prior that names every option except the markers and keeps no residual, so a marker can enter only through a leak. */
  function closedPrior(ids: readonly string[], markers: { items: ReadonlySet<string>; moves: ReadonlySet<string>; natures: ReadonlySet<string> }): UsagePrior {
    const natures = realVocabulary.natures.map((n) => n.id).filter((n) => !markers.natures.has(n));
    const items = realVocabulary.items.filter((i) => !markers.items.has(i));
    const species = [...ids].sort().map((id) => {
      const formes = realVocabulary.roster.filter((r) => r.id === id).map((r): Forme => {
        const pool = r.moves.filter((m) => !markers.moves.has(m));
        return {
          id: r.id,
          weight: 1,
          abilities: { denominator: r.abilities.length, entries: r.abilities.map((a) => ({ id: a, weight: 1 })), residual: 0, unmapped: 0 },
          items: { denominator: items.length + 1, entries: items.map((i) => ({ id: i, weight: 1 })), no_item: 1, residual: 0, unmapped: 0 },
          spreads: { denominator: natures.length, entries: natures.map((n) => ({ nature: n, stat_points: { hp: 4, atk: 0, def: 0, spa: 0, spd: 0, spe: 4 }, weight: 1 })), residual: 0, unmapped: 0 },
          moves: { denominator: pool.length, slots: 4, entries: pool.map((m) => ({ id: m, inclusion: 4 })), empty_slots: 0, residual: 0, unmapped: 0 },
        };
      });
      return { id, denominator: formes.length, formes: formes as [Forme, ...Forme[]] };
    });
    return { ...skewedPrior(ids), species: species as UsagePrior["species"] };
  }

  it("never appear, though the real team holds them; a prior that names them (a leak) is flagged by the same scan", async () => {
    const legality: SetLegality = createSetLegality({ engine, vocabulary: realVocabulary });
    const sampler = createSampler({ legality });
    const battle = battles[0] as RandomBattleFile;
    const theirs = withGenders(battle.teams.p2);
    const real = mutated(theirs);
    // The markers: what `mutated` put in the unrevealed fields, plus the nature everyone holds.
    const markerNature = real[0]!.nature;
    const markers = {
      items: new Set(real.map((s) => toId(s.item)).filter((i) => i !== "")),
      moves: new Set(real.flatMap((s, i) => s.moves.slice(1).filter((m, k) => toId(m) !== toId(theirs[i]!.moves[k + 1] as string)).map(toId))),
      natures: new Set(real.map((s) => toId(s.nature))),
    };
    expect(markerNature).toBeTruthy();
    expect(markers.items.size).toBeGreaterThan(3);
    expect(markers.moves.size).toBeGreaterThan(3);
    const record = await play(battle, real, "keep");
    const prefixes = (() => {
      const lines = record.channels.p1;
      return lines.flatMap((line, i) => (line.startsWith("|request|") ? [lines.slice(0, i + 1)] : []));
    })();

    /** The markers found as whole JSON tokens in a belief and worlds, that a revealed fact does not explain. */
    const found = (prior: (ids: string[]) => UsagePrior): string[] => {
      const hits = new Set<string>();
      for (const lines of prefixes) {
        const run = interpretChannel("p1", lines);
        const state = run.frames[run.frames.length - 1]!.state;
        const revealed = new Set(state.opponent.roster.flatMap((m) => [...m.moves_revealed, ...(m.item_known && m.item !== null ? [m.item] : [])]));
        const b = beliefAt({ state, flags: run.flags, game: "", seat: "p1", request: 0, truth: null }, prior(state.opponent.roster.map((m) => m.base_species)));
        const out = text({ b, w: sampler.sampleWorlds(b, stream(prefixes.indexOf(lines)), 3) });
        for (const marker of [...markers.items, ...markers.moves, ...markers.natures]) {
          if (out.includes(`"${marker}"`) && !revealed.has(marker)) hits.add(marker);
        }
      }
      return [...hits];
    };

    expect(found((ids) => closedPrior(ids, markers))).toEqual([]);
    // Negative control: a prior built from the real team (the markers named) is flagged.
    const leaky = (ids: string[]) => closedPrior(ids, { items: new Set(), moves: new Set(), natures: new Set() });
    expect(found(leaky).length).toBeGreaterThan(0);
  }, 240_000);
});
