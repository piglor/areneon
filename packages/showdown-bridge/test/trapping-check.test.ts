/* eslint-disable @typescript-eslint/no-explicit-any */
import { beforeAll, describe, expect, it } from "vitest";
import type { PublicMon } from "@areneon/public-state";
import { GOLDEN_FORMAT, goldenTeamVocabulary } from "../src/golden.ts";
import { type Engine, loadEngine, type PokemonSet } from "../src/engine.ts";
import { loadMechanicsConfig } from "../src/mechanics-config.ts";
import { checkScenario } from "../src/mechanics-scenario.ts";
import { loadRandomCorpus, playRandomBattle, type RandomBattleFile } from "../src/random-corpus.ts";
import { randomLegalActor } from "../src/random-legal.ts";
import type { Actor } from "../src/runner.ts";
import type { TeamVocabulary } from "../src/team-vocabulary.ts";
import { createTrappingCheck, type TrapLevel, type TrappingCheckContext } from "../src/trapping-check.ts";
import { cases, casesOf, extraBattles, inputAtLastRequest, toId } from "./world-builder-support.ts";

// The trapping check (ticket #1127) against the engine. The oracle is the LIVE battle: at every own move request of
// the committed seeded battles the check is built from the player's own view only (request, own team, reader state)
// and asked about the foes' TRUE actives (read from the live battle, a test-only read); the larger of their levels
// must be exactly what the request showed. So the check reproduces the engine's flags on real states, and a
// candidate the real world cannot have is never excluded.

let engine: Engine;
let context: TrappingCheckContext;
const vocabulary = goldenTeamVocabulary();
beforeAll(() => {
  engine = loadEngine();
  context = { engine, vocabulary, scenario_id: checkScenario(engine, loadMechanicsConfig(GOLDEN_FORMAT)) };
});

const tally = { requests: 0, checked: 0, declined: 0, informative: 0, seen: 0, unseen: 0, mismatched: 0 };

async function replay(spec: RandomBattleFile): Promise<void> {
  let captured: any = null;
  const Base = engine.sim.BattleStream as unknown as new () => { battle: unknown };
  const capture = (stream: unknown): void => {
    captured = stream;
  };
  class Spy extends Base {
    constructor() {
      super();
      capture(this);
    }
  }
  const spy: Engine = { ...engine, sim: { ...engine.sim, BattleStream: Spy as never } };
  const actorFor = (seed: number, player: "p1" | "p2"): Actor => {
    const inner = randomLegalActor(seed);
    return (request, ctx) => {
      if (ctx !== undefined && request.phase === "move" && captured?.battle != null) {
        const live = captured.battle as any;
        const foe = player === "p1" ? "p2" : "p1";
        const seat = player === "p1" ? 0 : 1;
        const names = new Set(live.sides[1 - seat].pokemon.map((m: any) => m.name));
        const foeSets = (spec.teams[foe] as PokemonSet[]).filter((set) => names.has(set.name === "" ? set.species : set.name));
        const input = inputAtLastRequest(player, ctx.channel, spec.teams[player] as PokemonSet[], foeSets);
        if (input !== null) {
          tally.requests += 1;
          const check = createTrappingCheck(context, { request: input.request, team: input.team, state: input.state });
          if (check === null) tally.declined += 1;
          else {
            tally.checked += 1;
            const foes = input.state.opponent.roster.filter((mon: PublicMon) => mon.active_slot !== null && !mon.fainted);
            const eligible = foes.every((mon: PublicMon) => !mon.ability_received && !mon.ability_suppressed && !mon.volatiles.includes("transform"));
            const levels = foes.map((mon: PublicMon): readonly TrapLevel[] => {
              const liveMon = live.sides[1 - seat].pokemon.find((m: any) => m.name === mon.name);
              const item = vocabulary.roster.find((r) => r.id === (mon.temp_species ?? mon.species))?.required_item ?? null;
              return check.shows({ species: (mon.temp_species ?? mon.species) as string, ability: liveMon.ability, item });
            });
            // The upper bound must hold for every foe, whatever the foe's ability history.
            for (const shown of levels) shown.forEach((level, k) => expect(level, `${spec.id}: a real foe shows more than the request`).toBeLessThanOrEqual(check.observed[k] as number));
            if (eligible && foes.length > 0) {
              const merged = check.observed.map((_, k) => Math.max(...levels.map((l) => l[k] as number)));
              if (merged.join() !== check.observed.join()) tally.mismatched += 1;
            }
            tally.informative += check.observed.length;
            if (check.observed.some((level) => level >= 1)) tally.seen += 1;
            else tally.unseen += 1;
          }
        }
      }
      return inner(request, ctx);
    };
  };
  await playRandomBattle(spy, spec, (seed) => actorFor(seed, spec.actor_seeds.p1 === seed ? "p1" : "p2"));
}

describe("the check reproduces the engine's trapping flags on real states", () => {
  it("over the committed seeded battles, the larger level of the real foes is what the request showed", async () => {
    for (const spec of loadRandomCorpus()) await replay(spec);
    console.log(`trapping check live: ${JSON.stringify(tally)}`);
    expect(tally.mismatched).toBe(0);
    expect(tally.checked).toBeGreaterThan(1000);
    expect(tally.unseen).toBeGreaterThan(1000);
  }, 400_000);
});

// `unavailable/` and `search-fidelity/` hold the committed requests that show maybeTrapped (a foe's Mega Gengar next
// to a last active) and trapped: true raised by a foe's hidden trap on an earlier active. The real foes are the
// case's true sets; a Mega Evolved foe holds its Mega forme's ability.
describe("the check on the committed requests that show the flags", () => {
  it("never asks for more than the request showed, and reproduces the last active's maybeTrapped", () => {
    let flagged = 0;
    let lastActiveSeen = 0;
    for (const c of casesOf(extraBattles())) {
      const check = createTrappingCheck(context, { request: c.input.request, team: c.input.team, state: c.input.state });
      if (check === null) continue;
      const foes = c.input.state.opponent.roster.filter((mon: PublicMon) => mon.active_slot !== null && !mon.fainted);
      const levels = foes.map((mon: PublicMon) => {
        const species = (mon.temp_species ?? mon.species) as string;
        const entry = vocabulary.roster.find((r) => r.id === species);
        const set = c.input.opponent.members.find((m) => m.roster_index === c.input.state.opponent.roster.indexOf(mon))?.set;
        const ability = entry?.required_item !== undefined ? (entry.abilities[0] as string) : toId(set?.ability ?? "");
        return check.shows({ species, ability, item: entry?.required_item ?? null });
      });
      levels.forEach((shown) => shown.forEach((level, k) => expect(level, `${c.id}: a real foe shows more than the request`).toBeLessThanOrEqual(check.observed[k] as number)));
      if (check.observed.some((level) => level === 1)) {
        flagged += 1;
        const merged = check.observed.map((_, k) => Math.max(...levels.map((l) => l[k] as number)));
        expect(merged, `${c.id}: the real foes show the maybeTrapped`).toEqual([...check.observed]);
        lastActiveSeen += 1;
      }
    }
    expect(flagged).toBeGreaterThan(0);
    expect(lastActiveSeen).toBe(flagged);
  }, 200_000);
});

// The only species of the format's roster that lists a trapping ability is Mega Gengar. The engine's rule is also
// exercised with a species outside the roster, added to a copy of the vocabulary: the engine decides which species
// and abilities these are.
describe("the check reads the engine's maybeTrapped rule", () => {
  function trapperSpecies(): { species: string; trapper: string; plain: string[] } {
    const dex = engine.sim.Dex.mod(engine.sim.Dex.formats.get(GOLDEN_FORMAT).mod) as unknown as {
      abilities: { all(): { id: string; onFoeMaybeTrapPokemon?: unknown }[] };
      species: { all(): { id: string; abilities: Record<string, string> }[] };
    };
    const trappers = new Set(dex.abilities.all().filter((a) => a.onFoeMaybeTrapPokemon !== undefined).map((a) => a.id));
    const found = dex.species.all().find((sp) => {
      const ids = Object.values(sp.abilities).map(toId);
      return ids.some((id) => trappers.has(id)) && ids.some((id) => !trappers.has(id));
    });
    if (found === undefined) throw new Error("the engine lists no species with a trapping ability beside another");
    const ids = [...new Set(Object.values(found.abilities).map(toId))];
    return { species: found.id, trapper: ids.find((id) => trappers.has(id)) as string, plain: ids.filter((id) => !trappers.has(id)) };
  }

  it("a foe whose species lists a trapping ability shows maybeTrapped to the last active; the one that holds it also traps an earlier active", () => {
    const { species, trapper, plain } = trapperSpecies();
    const entry = { id: species, abilities: [trapper, ...plain], moves: [(vocabulary.roster[0] as { moves: string[] }).moves[0] as string] };
    const extended: TeamVocabulary = { ...vocabulary, roster: [...vocabulary.roster, entry] } as TeamVocabulary;
    const ctx: TrappingCheckContext = { ...context, vocabulary: extended };
    let hits = 0;
    let trapped = 0;
    for (const c of cases()) {
      const check = createTrappingCheck(ctx, { request: c.input.request, team: c.input.team, state: c.input.state });
      if (check === null) continue;
      const candidate = (ability: string) => check.shows({ species, ability, item: null });
      const noticed = candidate(plain[0] as string);
      if (!noticed.some((level) => level === 1)) continue;
      hits += 1;
      // The engine's trap loop compares an ability's display name with the holder's ability id, so it never skips the held slot: maybeTrapped is the species' flag whichever ability the foe holds. The held trap works, though: an earlier active shows trapped: true.
      const held = candidate(trapper);
      held.forEach((level, k) => expect(level, `${c.id}: holding the trapping ability shows at least what not holding it shows`).toBeGreaterThanOrEqual(noticed[k] as number));
      if (held.some((level) => level === 2)) trapped += 1;
      for (const ability of plain) expect(candidate(ability), `${c.id}: any other ability shows the same`).toEqual(noticed);
      if (hits >= 3) break;
    }
    expect(hits).toBeGreaterThan(0);
    expect(trapped).toBeGreaterThan(0);
  });

  it("an unknown candidate species constrains nothing (it shows what was observed)", () => {
    const c = cases().map((x) => ({ x, check: createTrappingCheck(context, { request: x.input.request, team: x.input.team, state: x.input.state }) })).find((y) => y.check !== null);
    const check = (c as { check: NonNullable<typeof c>["check"] }).check as NonNullable<ReturnType<typeof createTrappingCheck>>;
    expect(check.shows({ species: "notaspecies", ability: "nothing", item: null })).toEqual(check.observed);
  });

  it("declines when the request is no move request, a field condition is in play, the team does not match, or nothing is informative", () => {
    const [c] = cases() as [ReturnType<typeof cases>[number]];
    const base = { request: c.input.request, team: c.input.team, state: c.input.state };
    expect(createTrappingCheck(context, base)).not.toBeNull();
    expect(createTrappingCheck(context, { ...base, request: { ...c.input.request, wait: true } as typeof c.input.request })).toBeNull();
    expect(createTrappingCheck(context, { ...base, request: { ...c.input.request, active: undefined } as unknown as typeof c.input.request })).toBeNull();
    expect(createTrappingCheck(context, { ...base, state: { ...c.input.state, field_conditions: ["gravity"] } })).toBeNull();
    expect(createTrappingCheck(context, { ...base, state: { ...c.input.state, player: c.input.state.player === "p1" ? "p2" : "p1" } })).toBeNull();
    const first = c.input.request.side.pokemon[0]?.ident.slice(c.input.request.side.pokemon[0].ident.indexOf(": ") + 2);
    expect(createTrappingCheck(context, { ...base, team: c.input.team.filter((set) => (set.name === "" ? set.species : set.name) !== first) })).toBeNull();
    expect(createTrappingCheck(context, { ...base, team: [...c.input.team, ...c.input.team] })).toBeNull();
    // every own active carries a volatile or trapped: true: no information
    const trapped = { ...c.input.request, active: c.input.request.active?.map((slot) => ({ ...slot, trapped: true })) } as typeof c.input.request;
    expect(createTrappingCheck(context, { ...base, request: trapped })).toBeNull();
    const busy = { ...c.input.state, own: { ...c.input.state.own, roster: c.input.state.own.roster.map((mon) => ({ ...mon, volatiles: ["x"] })) } };
    expect(createTrappingCheck(context, { ...base, state: busy })).toBeNull();
  });
});
