// Spike #682 (throwaway): enumerate candidate joint choices from one
// player's own `|request|` JSON. It proposes candidates only; the pinned
// engine decides which are legal (see oracle.ts). Doubles only.
// Not implemented in the spike: rejecting unknown request fields that can
// change legality (e.g. `reviving`); ADR-006 rule 3 requires it in the real
// enumerator.

type Json = null | boolean | number | string | Json[] | { [key: string]: Json };
type Obj = { [key: string]: Json };

const ACTIVE_SLOTS = 2;

/** Request fields whose truth the engine hides from the player. */
export interface Uncertainty {
  maybeTrapped: boolean;
  maybeDisabled: boolean;
  maybeLocked: boolean;
}

export interface Enumeration {
  phase: "preview" | "move" | "forceSwitch" | "wait";
  candidates: string[];
  uncertainty: Uncertainty;
}

export class EnumerationError extends Error {
  override name = "EnumerationError";
}

const asObj = (value: Json | undefined): Obj => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new EnumerationError("expected an object");
  }
  return value;
};
const asArr = (value: Json | undefined): Json[] => {
  if (!Array.isArray(value)) throw new EnumerationError("expected an array");
  return value;
};

/** Ordered k-permutations of 1..n, as digit strings. */
function permutations(n: number, k: number): string[] {
  const out: string[] = [];
  const walk = (prefix: number[]): void => {
    if (prefix.length === k) {
      out.push(prefix.join(""));
      return;
    }
    for (let i = 1; i <= n; i++) if (!prefix.includes(i)) walk([...prefix, i]);
  };
  walk([]);
  return out;
}

/** Target locations a doubles move may name, from its request `target`. */
function targetsFor(target: string | undefined, slot: number): number[] {
  const self = -(slot + 1);
  const ally = -(2 - slot);
  switch (target) {
    case "normal":
    case "any":
      return [1, 2, ally];
    case "adjacentFoe":
      return [1, 2];
    case "adjacentAlly":
      return [ally];
    case "adjacentAllyOrSelf":
      return [ally, self];
    default:
      return []; // self, spread, field and locked moves take no target
  }
}

const fainted = (mon: Obj): boolean => String(mon.condition).endsWith(" fnt");

/** Bench slots (1-based) a Pokémon could switch to. */
function benchSlots(pokemon: Obj[]): number[] {
  const out: number[] = [];
  pokemon.forEach((mon, i) => {
    if (i >= ACTIVE_SLOTS && mon.active !== true && !fainted(mon)) out.push(i + 1);
  });
  return out;
}

interface SlotOption {
  text: string;
  switchTo?: number;
  mega?: boolean;
}

function moveSlotOptions(active: Obj, mon: Obj, slot: number, bench: number[]): SlotOption[] {
  if (fainted(mon) || mon.commanding === true) return [{ text: "pass" }];
  const options: SlotOption[] = [];
  const events: string[] = [""];
  if (active.canMegaEvo === true) events.push("mega");
  if (active.canMegaEvoX === true) events.push("megax");
  if (active.canMegaEvoY === true) events.push("megay");
  if (active.canTerastallize !== undefined && active.canTerastallize !== false) {
    events.push("terastallize");
  }
  asArr(active.moves).forEach((raw, j) => {
    const move = asObj(raw);
    if (move.disabled !== undefined && move.disabled !== false) return;
    const targets = targetsFor(move.target as string | undefined, slot);
    for (const event of events) {
      const suffix = event ? ` ${event}` : "";
      const mega = event.startsWith("mega");
      if (targets.length === 0) options.push({ text: `move ${j + 1}${suffix}`, mega });
      for (const t of targets) options.push({ text: `move ${j + 1} ${t}${suffix}`, mega });
    }
  });
  if (active.trapped !== true) {
    for (const b of bench) options.push({ text: `switch ${b}`, switchTo: b });
  }
  return options;
}

function joint(a: SlotOption[], b: SlotOption[]): string[] {
  const out: string[] = [];
  for (const x of a) {
    for (const y of b) {
      if (x.switchTo !== undefined && x.switchTo === y.switchTo) continue;
      if (x.mega === true && y.mega === true) continue;
      out.push(`${x.text}, ${y.text}`);
    }
  }
  return out;
}

export function enumerate(request: Obj): Enumeration {
  const side = asObj(request.side);
  const pokemon = asArr(side.pokemon).map(asObj);
  const uncertainty: Uncertainty = { maybeTrapped: false, maybeDisabled: false, maybeLocked: false };

  if (request.teamPreview === true) {
    const k = Number(request.maxChosenTeamSize ?? pokemon.length);
    const candidates = permutations(pokemon.length, k).map((p) => `team ${p}`);
    return { phase: "preview", candidates, uncertainty };
  }
  if (request.wait === true) return { phase: "wait", candidates: [], uncertainty };

  const bench = benchSlots(pokemon);
  if (Array.isArray(request.forceSwitch)) {
    const flags = request.forceSwitch.map((f) => f === true);
    const mustSwitch = Math.min(flags.filter(Boolean).length, bench.length);
    const perSlot = flags.map((flag): SlotOption[] =>
      flag
        ? [...bench.map((b) => ({ text: `switch ${b}`, switchTo: b })), { text: "pass" }]
        : [{ text: "pass" }],
    );
    const switches = (s: string): number => s.split(", ").filter((x) => x.startsWith("switch")).length;
    const candidates = joint(perSlot[0] ?? [], perSlot[1] ?? []).filter((c) => switches(c) === mustSwitch);
    return { phase: "forceSwitch", candidates, uncertainty };
  }

  const actives = asArr(request.active).map(asObj);
  for (const a of actives) {
    uncertainty.maybeTrapped ||= a.maybeTrapped === true;
    uncertainty.maybeDisabled ||= a.maybeDisabled === true;
    uncertainty.maybeLocked ||= a.maybeLocked === true;
  }
  const perSlot = [0, 1].map((slot) => {
    const active = actives[slot];
    const mon = pokemon[slot];
    if (active === undefined || mon === undefined) return [{ text: "pass" }];
    return moveSlotOptions(active, mon, slot, bench);
  });
  return { phase: "move", candidates: joint(perSlot[0] ?? [], perSlot[1] ?? []), uncertainty };
}
