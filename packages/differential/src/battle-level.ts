// The battle-step level's entity and exercising rule (ADR-028 section 5, reviewed; ADR-026 slice F's carrier rule): what a
// compared step holds and what it read, from the trace (the oracle) alone. No list of game content is written here: every id is
// one the trace's teams or log lines name.
//
// Present at step k: every species, ability, item and move of both teams as brought, plus every entity an earlier or this step
// exercised. Exercised at step k (from its omniscient lines): the move of each `|move|` line; the shown species, the ability and
// the item of every Pokemon that acted (`|move|`, `|cant|`: the carrier rule); the species a switch, drag, replace or forme
// change shows; the ability, item, effect, status, weather, field or side condition a line names directly or in a `[from]` tag.
import type { BattleTrace, PokemonSet } from "@areneon/schema/types/v1/battle-trace.ts";
import { SEATS } from "./battle-backend.ts";

/** Lower case, letters and digits only (how the engine names ids). */
export const toId = (name: string): string => name.toLowerCase().replace(/[^a-z0-9]/g, "");

const IDENT = /^(p[12])[a-c]?: (.+)$/;
const PREFIX = /^(move|ability|item): (.+)$/;

/** `<kind>:<id>` for an effect name: `move: X`, `ability: X` and `item: X` keep their kind, anything else is a volatile. */
function effect(name: string): string | null {
  const m = PREFIX.exec(name.trim());
  const id = toId(m === null ? name : (m[2] as string));
  if (id === "") return null;
  return `${m === null ? "volatile" : m[1]}:${id}`;
}

const setsOf = (trace: Pick<BattleTrace, "teams">): Map<string, PokemonSet> => {
  const out = new Map<string, PokemonSet>();
  for (const seat of SEATS) for (const set of trace.teams[seat]) out.set(`${seat}:${set.name === "" ? set.species : set.name}`, set);
  return out;
};

/** The entities of both teams as brought. */
export function teamEntities(trace: Pick<BattleTrace, "teams">): string[] {
  const out = new Set<string>();
  for (const set of setsOf(trace).values()) {
    out.add(`species:${toId(set.species)}`);
    if (set.ability !== "") out.add(`ability:${toId(set.ability)}`);
    if (set.item !== "") out.add(`item:${toId(set.item)}`);
    for (const move of set.moves) out.add(`move:${toId(move)}`);
  }
  return [...out];
}

/** Per step of a trace: the sorted entities present and the sorted entities its lines exercised. */
export function stepFacts(trace: Pick<BattleTrace, "teams" | "steps">): { entities: string[]; exercised: string[] }[] {
  const sets = setsOf(trace);
  const shown = new Map<string, string>();
  const present = new Set(teamEntities(trace));
  return trace.steps.map((step) => {
    const exercised = new Set<string>();
    const add = (id: string | null): void => {
      if (id !== null) exercised.add(id);
    };
    const carrier = (ident: string): void => {
      const m = IDENT.exec(ident);
      if (m === null) return;
      const key = `${m[1]}:${m[2]}`;
      const set = sets.get(key);
      add(`species:${toId(shown.get(key) ?? set?.species ?? "")}`);
      if (set !== undefined && set.ability !== "") add(`ability:${toId(set.ability)}`);
      if (set !== undefined && set.item !== "") add(`item:${toId(set.item)}`);
    };
    for (const line of step.omniscient) {
      const [type, ...args] = line.slice(1).split("|");
      const arg = (i: number): string => args[i] ?? "";
      switch (type) {
        case "move":
          add(effect(`move: ${arg(1)}`));
          carrier(arg(0));
          break;
        case "cant":
          carrier(arg(0));
          break;
        case "switch":
        case "drag":
        case "replace":
        case "detailschange":
        case "-formechange": {
          const species = arg(1).split(",")[0] as string;
          const m = IDENT.exec(arg(0));
          if (m !== null) shown.set(`${m[1]}:${m[2]}`, species);
          add(`species:${toId(species)}`);
          break;
        }
        case "-mega":
          add(effect(`item: ${arg(2)}`));
          break;
        case "-ability":
          add(effect(`ability: ${arg(1)}`));
          break;
        case "-item":
        case "-enditem":
          add(effect(`item: ${arg(1)}`));
          break;
        case "-start":
        case "-end":
        case "-activate":
        case "-singleturn":
        case "-singlemove":
          add(effect(arg(1)));
          break;
        case "-sidestart":
        case "-sideend":
          add(effect(arg(1))?.replace(/^(move|volatile):/, "side:") ?? null);
          break;
        case "-fieldstart":
        case "-fieldend":
          add(effect(arg(0))?.replace(/^(move|volatile):/, "field:") ?? null);
          break;
        case "-weather": {
          const weather = toId(arg(0));
          if (weather !== "" && weather !== "none") add(`weather:${weather}`);
          break;
        }
        case "-status":
        case "-curestatus":
          add(`status:${toId(arg(1))}`);
          break;
      }
      for (const field of args) if (field.startsWith("[from] ")) add(effect(field.slice("[from] ".length)));
    }
    exercised.delete("species:");
    for (const id of exercised) present.add(id);
    return { entities: [...present].sort(), exercised: [...exercised].sort() };
  });
}
