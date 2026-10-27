// Shared by the hidden-information invariance tests of the damage-aware baseline (ADR-018 section 3, ADR-025 section 6).
import type { PokemonSet } from "../src/engine.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { jcs, type Json } from "../src/canonical-json.ts";
import type { BattleRecord } from "../src/runner.ts";
import { engine, toId, vocabulary } from "./damage-aware-support.ts";

const validator = () => engine.sim.TeamValidator.get(GOLDEN_FORMAT);

/** p2's team with fields the baseline cannot see before they act changed: nature, Stat Points, the held items (rotated, so none is lost or doubled) and moves 2 to 4. */
export function mutated(team: PokemonSet[]): PokemonSet[] {
  const natures = vocabulary.natures.map((nature) => nature.name);
  const items = team.map((set) => set.item);
  const out = team.map((set, index) => {
    const copy = structuredClone(set);
    copy.nature = natures[(natures.indexOf(set.nature) + 7 + index) % natures.length] as string;
    copy.evs = { ...set.evs, hp: set.evs.spe, spe: set.evs.hp, atk: set.evs.def, def: set.evs.atk };
    copy.item = items[(index + 1) % items.length] as string;
    return copy;
  });
  // moves 2 to 4: another move of the species' pool, kept only while the whole team stays legal
  out.forEach((set, index) => {
    const pool = vocabulary.roster.find((entry) => entry.id === toId(set.species) || entry.id === toId(set.species.split("-")[0] ?? ""))?.moves ?? [];
    for (let slot = 1; slot < set.moves.length; slot++) {
      const replacement = pool.find((id) => !set.moves.map(toId).includes(id) && !(out[index] as PokemonSet).moves.map(toId).includes(id));
      if (replacement === undefined) continue;
      const before = (out[index] as PokemonSet).moves[slot] as string;
      (out[index] as PokemonSet).moves[slot] = (engine.sim.Dex.mod("champions").moves.get(replacement) as { name: string }).name;
      if ((validator().validateTeam(structuredClone(out)) ?? []).length > 0) (out[index] as PokemonSet).moves[slot] = before;
    }
  });
  return out;
}

/** A team with every gender stated: the engine draws an unstated one from the battle seed, and the Team Preview line shows it. */
export function withGenders(team: PokemonSet[]): PokemonSet[] {
  return team.map((set) => {
    const fixed = (engine.sim.Dex.mod("champions").species.get(set.species) as unknown as { gender: string }).gender;
    return { ...structuredClone(set), gender: set.gender !== "" ? set.gender : fixed !== "" ? fixed : "M" };
  });
}


/** The p1 requests where both battles' channels are identical up to and including the request, and the two choices made there. */
export function comparable(a: BattleRecord, b: BattleRecord): { seq: number; first: string; second: string }[] {
  const out: { seq: number; first: string; second: string }[] = [];
  const cut = (record: BattleRecord, seq: number): string[] => {
    const lines = record.channels.p1;
    let seen = 0;
    for (let i = 0; i < lines.length; i++) {
      if ((lines[i] as string).startsWith("|request|") && ++seen === seq) return lines.slice(0, i + 1);
    }
    return [];
  };
  const mine = (record: BattleRecord, seq: number) => record.choices.find((c) => c.player === "p1" && c.request_seq === seq)?.choice;
  const count = Math.min(a.requests.p1.length, b.requests.p1.length);
  for (let seq = 1; seq <= count; seq++) {
    const left = cut(a, seq);
    const right = cut(b, seq);
    // the request lines carry no rqid on A1, so equal prefixes are equal player views
    if (left.length === 0 || jcs(left as unknown as Json) !== jcs(right as unknown as Json)) break;
    const first = mine(a, seq);
    const second = mine(b, seq);
    if (first !== undefined && second !== undefined) out.push({ seq, first, second });
  }
  return out;
}

