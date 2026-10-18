import type { DecodedRequest, JsonObject, RequestPhase } from "./request.ts";
import type { Json } from "./request-identity.ts";

/**
 * One candidate joint choice (ADR-006). The enumerator proposes it; only the
 * pinned engine decides whether it is legal.
 */
export interface Candidate {
  /** Canonical engine syntax, e.g. `move 1 2 mega, switch 3`. */
  choice: string;
  /**
   * Its legality can depend on information the engine hides from the player
   * (`maybeTrapped`, `maybeDisabled`, `maybeLocked`). A pre-submission
   * verdict on it would leak that information, so none is given.
   */
  pending: boolean;
}

export interface CandidateSet {
  phase: RequestPhase;
  /** Deterministic ADR-006 order; empty for `wait`. */
  candidates: Candidate[];
}

/** The request holds something the enumerator does not understand. */
export class EnumerationError extends Error {
  override name = "EnumerationError";
}

const ACTIVE_SLOTS = 2;

// Request fields the enumerator understands, per pinned Showdown's request
// types (`sim/side.ts`). Anything else could change legality, so it fails
// closed (ADR-006 rule 3). Z-Moves, Dynamax, Ultra Burst and multi battles
// (`ally`) are not part of any supported format and fail closed too.
const REQUEST_KEYS = new Set([
  "active",
  "forceSwitch",
  "maxChosenTeamSize",
  "noCancel",
  "rqid",
  "side",
  "teamPreview",
  "update",
  "wait",
]);
const SIDE_KEYS = new Set(["id", "name", "noCancel", "pokemon"]);
const POKEMON_KEYS = new Set([
  "ability",
  "active",
  "baseAbility",
  "commanding",
  "condition",
  "details",
  "ident",
  "item",
  "moves",
  "pokeball",
  "reviving",
  "stats",
  "teraType",
  "terastallized",
]);
const ACTIVE_KEYS = new Set([
  "canMegaEvo",
  "canMegaEvoX",
  "canMegaEvoY",
  "canTerastallize",
  "maybeDisabled",
  "maybeLocked",
  "maybeTrapped",
  "moves",
  "trapped",
]);
const MOVE_KEYS = new Set(["disabled", "disabledSource", "id", "maxpp", "move", "pp", "target"]);

/** Move targets that take no target location in the choice string. */
const UNTARGETED = new Set([
  "all",
  "allAdjacent",
  "allAdjacentFoes",
  "allies",
  "allySide",
  "allyTeam",
  "foeSide",
  "randomNormal",
  "scripted",
  "self",
]);

/** Mega Evolution and Terastallization, in ADR-006 event order. */
const EVENTS = [
  ["canMegaEvo", "mega"],
  ["canMegaEvoX", "megax"],
  ["canMegaEvoY", "megay"],
  ["canTerastallize", "terastallize"],
] as const;

function object(value: Json | undefined, what: string, keys: Set<string>): JsonObject {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new EnumerationError(`${what} is not an object`);
  }
  const unknown = Object.keys(value).filter((key) => !keys.has(key));
  if (unknown.length > 0) {
    throw new EnumerationError(`${what} has unsupported field(s): ${unknown.join(", ")}`);
  }
  return value;
}

function array(value: Json | undefined, what: string): Json[] {
  if (!Array.isArray(value)) throw new EnumerationError(`${what} is not an array`);
  return value;
}

/** Target locations a move may name from active slot `slot` (0 or 1). */
function targetsFor(target: Json | undefined, slot: number): number[] {
  const ally = -(ACTIVE_SLOTS - slot);
  const self = -(slot + 1);
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
  }
  // A locked move (e.g. Outrage, Recharge) is sent without a target type.
  if (target === undefined || (typeof target === "string" && UNTARGETED.has(target))) return [];
  throw new EnumerationError(`unsupported move target ${JSON.stringify(target)}`);
}

/** One slot's part of a joint choice. */
interface SlotOption {
  text: string;
  pending: boolean;
  switchTo?: number;
  /** Mega Evolution or Terastallization: at most one per side per choice. */
  event?: "mega" | "tera";
}

const PASS: SlotOption = { text: "pass", pending: false };

function joint(first: SlotOption[], second: SlotOption[]): Candidate[] {
  const out: Candidate[] = [];
  for (const a of first) {
    for (const b of second) {
      if (a.switchTo !== undefined && a.switchTo === b.switchTo) continue;
      if (a.event !== undefined && a.event === b.event) continue;
      out.push({ choice: `${a.text}, ${b.text}`, pending: a.pending || b.pending });
    }
  }
  return out;
}

interface Mon {
  fainted: boolean;
  commanding: boolean;
}

function readSide(request: JsonObject): Mon[] {
  const side = object(request.side, "request.side", SIDE_KEYS);
  return array(side.pokemon, "side.pokemon").map((raw, i) => {
    const mon = object(raw, `side.pokemon[${i}]`, POKEMON_KEYS);
    if (mon.reviving === true) {
      throw new EnumerationError("Revival Blessing (`reviving`) is not supported");
    }
    return {
      fainted: typeof mon.condition === "string" && mon.condition.endsWith(" fnt"),
      commanding: mon.commanding === true,
    };
  });
}

/** Bench slots (1-based, ascending) that can be switched in. */
function benchSlots(team: Mon[]): number[] {
  const out: number[] = [];
  team.forEach((mon, i) => {
    if (i >= ACTIVE_SLOTS && !mon.fainted) out.push(i + 1);
  });
  return out;
}

/** Ordered k-permutations of 1..n as digit strings, lexicographic. */
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

function moveSlot(raw: Json, mon: Mon | undefined, slot: number, bench: number[]): SlotOption[] {
  const active = object(raw, `active[${slot}]`, ACTIVE_KEYS);
  if (mon === undefined) throw new EnumerationError(`no side.pokemon for active slot ${slot + 1}`);
  if (mon.fainted || mon.commanding) return [PASS];

  // Where the engine hides the truth, legality is pending (ADR-006 rule 5).
  const movesPending = active.maybeDisabled === true || active.maybeLocked === true;
  const events = EVENTS.filter(([flag]) => {
    const value = active[flag];
    return value !== undefined && value !== false && value !== null;
  });
  const options: SlotOption[] = [];
  array(active.moves, `active[${slot}].moves`).forEach((rawMove, j) => {
    const move = object(rawMove, `active[${slot}].moves[${j}]`, MOVE_KEYS);
    if (move.disabled !== undefined && move.disabled !== false) return;
    const targets = targetsFor(move.target, slot);
    for (const target of targets.length === 0 ? [undefined] : targets) {
      const base = target === undefined ? `move ${j + 1}` : `move ${j + 1} ${target}`;
      options.push({ text: base, pending: movesPending });
      for (const [, event] of events) {
        options.push({
          text: `${base} ${event}`,
          pending: movesPending,
          event: event === "terastallize" ? "tera" : "mega",
        });
      }
    }
  });
  if (active.trapped !== true) {
    const switchPending = active.maybeTrapped === true;
    for (const b of bench) options.push({ text: `switch ${b}`, pending: switchPending, switchTo: b });
  }
  return options;
}

function previewCandidates(request: JsonObject, team: Mon[]): Candidate[] {
  const k = request.maxChosenTeamSize;
  if (typeof k !== "number" || !Number.isInteger(k) || k < 1 || k > team.length) {
    throw new EnumerationError(`unsupported maxChosenTeamSize ${JSON.stringify(k)}`);
  }
  return permutations(team.length, k).map((p) => ({ choice: `team ${p}`, pending: false }));
}

function forceSwitchCandidates(request: JsonObject, team: Mon[]): Candidate[] {
  const flags = array(request.forceSwitch, "forceSwitch");
  if (flags.length !== ACTIVE_SLOTS || flags.some((f) => typeof f !== "boolean")) {
    throw new EnumerationError("forceSwitch must hold one boolean per doubles slot");
  }
  const bench = benchSlots(team);
  const forced = Math.min(flags.filter(Boolean).length, bench.length);
  const perSlot = flags.map((flag): SlotOption[] =>
    flag ? [...bench.map((b) => ({ text: `switch ${b}`, pending: false, switchTo: b })), PASS] : [PASS],
  );
  const switches = (c: Candidate): number =>
    c.choice.split(", ").filter((part) => part.startsWith("switch")).length;
  return joint(perSlot[0] as SlotOption[], perSlot[1] as SlotOption[]).filter(
    (c) => switches(c) === forced,
  );
}

/**
 * Every engine-distinct joint choice for the player's own request, in
 * ADR-006 order. A pure function of the request: it proposes, the engine
 * judges (`legalChoiceSet`). Throws `EnumerationError` on any request field
 * it does not understand rather than guessing. Doubles only.
 */
export function enumerateCandidates(decoded: DecodedRequest): CandidateSet {
  const request = object(decoded.request, "request", REQUEST_KEYS);
  const team = readSide(request);
  const { phase } = decoded;
  if (phase === "wait") return { phase, candidates: [] };
  if (phase === "preview") return { phase, candidates: previewCandidates(request, team) };
  if (phase === "forceSwitch") return { phase, candidates: forceSwitchCandidates(request, team) };

  const active = array(request.active, "active");
  if (active.length !== ACTIVE_SLOTS) {
    throw new EnumerationError(`expected ${ACTIVE_SLOTS} active slots, got ${active.length}`);
  }
  const bench = benchSlots(team);
  const [first, second] = active.map((raw, slot) => moveSlot(raw, team[slot], slot, bench));
  return { phase, candidates: joint(first as SlotOption[], second as SlotOption[]) };
}
