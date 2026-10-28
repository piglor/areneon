// The world facts (ADR-026 section 1, slice L, finding N-H). The scenario worlds of decision-time search
// are written by reviewed writers (config/search/world-writers.v1.json), and those writers need facts the
// shared reader (`interpreter.cjs`) does not track: turns since the switch-in, the Protect streak, the last
// move, faints per turn, the start turn of every timed condition, hazard layers and the Toxic stage. This
// module derives them from ONE player's own channel and nothing else.
//
// It is a module of its own with its own format id and file hash. The shared reader, its hash
// (`readerSha256`) and ADR-018's baseline identity are not touched, so no replay page and no baseline
// identity moves. It imports nothing (a test enforces it): a hand-written line splitter reads only what it
// needs, so a change to the reader's grammar cannot move a fact silently, and the facts are checked against
// the reader and against the omniscient log's turn structure in the tests.
//
// What it reads is public: every line comes from the player's own channel. The input type is a branded
// `OwnChannel` that only `ownChannel` makes (it refuses a `|split|` line, the marker of a secret half), so
// an omniscient log, a team or a seed is a compile error and a run-time refusal. Nothing here hashes, draws
// or reads a file at run time.
//
// Semantics follow the pinned engine (`sim/battle.ts` `nextTurn`, `data/conditions.ts` `stall`, `tox`):
// - `active_turns`: the engine increments `activeTurns` of every Pokemon in play once per `|turn|` line, so
//   it is the number of `|turn|` lines since the `|switch|` or `|drag|` that sent the Pokemon in (a lead is
//   1 at turn 1, a Pokemon sent in during turn N is 1 at turn N+1).
// - `protect_streak`: the number of consecutive turns of successful stall-move uses (the `stall` volatile
//   survives the turn after a use, so a use two turns back no longer counts); a failed use, a switch and a
//   faint end it. Which moves stall is data the caller passes (`WorldFactsData.stall_moves`).
// - `last_move`: the last `|move|` line of the Pokemon without a `[from] move/ability/item` tag (a called
//   move or a reflected one is not what the Pokemon chose); `[from]lockedmove` is its own move.
// - Toxic: the number of the status's residual damage (or Poison Heal) lines since the status or the
//   switch-in, the stage the engine reaches (at most `toxic_stage_max`).
// - Faints: `total_fainted` is the side's `|faint|` count, `fainted_this_turn` / `fainted_last_turn` say
//   whether the side lost a Pokemon in the current / the previous turn (the engine's `faintedLastTurn`).
//
// Fail closed: a line it cannot attribute, a Pokemon whose Toxic tick was not shown (Magic Guard, Poison
// Heal at full health), a stall move whose outcome is not shown, and an Illusion `|replace|` make the
// facts of that channel a typed `WorldFactsError`, from that line on. Facts from before are still read.

/** The format of the derivation: bump when how a fact is computed from the lines changes (part of the search identity, ADR-026 section 5). */
export const WORLD_FACTS_FORMAT = "world-facts/1";

export type FactsPlayer = "p1" | "p2";

/** Reviewed data from the pinned engine, passed by the caller; this module holds no game list. */
export interface WorldFactsData {
  /** Ids of the moves that add the engine's `stall` volatile (the Protect family, Endure, Quick Guard, Wide Guard). */
  readonly stall_moves: readonly string[];
  /** Ids of the terrains (`-fieldstart` of one of them is the terrain, any other id is a pseudo-weather). */
  readonly terrains: readonly string[];
  /** Ids of the side conditions Court Change swaps (`-swapsideconditions` moves only these, the engine's own list). */
  readonly swap_side_conditions: readonly string[];
  /** The highest Toxic stage the engine reaches (`data/conditions.ts` tox: the stage stops there). */
  readonly toxic_stage_max: number;
}

declare const OWN: unique symbol;

/** One player's own channel, checked. Only `ownChannel` makes one. */
export interface OwnChannel {
  readonly [OWN]: true;
  readonly player: FactsPlayer;
  readonly lines: readonly string[];
}

export const WORLD_FACTS_REFUSALS = [
  "bad_player",
  "secret_marker",
  "malformed_line",
  "unattributable_line",
  "illusion_replace",
  "stall_unresolved",
  "toxic_tick_unobserved",
] as const;
export type WorldFactsRefusal = (typeof WORLD_FACTS_REFUSALS)[number];

/** A refusal: a code and the 0-based index of the line, in a fixed message that quotes no line. */
export class WorldFactsError extends Error {
  override name = "WorldFactsError";
  readonly code: WorldFactsRefusal;
  readonly line: number | null;
  constructor(code: WorldFactsRefusal, line: number | null) {
    super(`world-facts: ${code}${line === null ? "" : ` at line ${line}`}`);
    this.code = code;
    this.line = line;
  }
}

/** Checks one player's own channel lines: a player, lines that start with `|`, and no `|split|` marker. */
export function ownChannel(player: string, lines: readonly string[]): OwnChannel {
  if (player !== "p1" && player !== "p2") throw new WorldFactsError("bad_player", null);
  lines.forEach((line, index) => {
    if (!line.startsWith("|")) throw new WorldFactsError("malformed_line", index);
    if (line === "|split" || line.startsWith("|split|")) throw new WorldFactsError("secret_marker", index);
  });
  return { player, lines } as unknown as OwnChannel;
}

export interface ActiveFacts {
  /** The nickname as the channel shows it. */
  name: string;
  active_turns: number;
  last_move: string | null;
  protect_streak: number;
  /** The Toxic stage reached (residual lines since the status or the switch-in), null when the Pokemon is not badly poisoned. */
  toxic_ticks: number | null;
}

export interface ConditionFacts {
  /** The turn counter when the condition began (0 before `|turn|1`). */
  started_turn: number;
  /** `-sidestart` lines since it began: the layers of Spikes and Toxic Spikes, 1 for a condition that does not stack. */
  layers: number;
}

export interface SideFacts {
  total_fainted: number;
  fainted_this_turn: boolean;
  fainted_last_turn: boolean;
  /** Per position letter that has held a Pokemon; null when the position is empty now. */
  active: Record<string, ActiveFacts | null>;
  conditions: Record<string, ConditionFacts>;
}

export interface WorldFacts {
  world_facts_format: string;
  turn: number;
  weather: { id: string; started_turn: number } | null;
  terrain: { id: string; started_turn: number } | null;
  field_conditions: Record<string, { started_turn: number }>;
  own: SideFacts;
  opponent: SideFacts;
}

interface Occupant {
  name: string;
  switch_turn: number;
  last_move: string | null;
  stall_count: number;
  stall_turn: number;
}

interface MonStatus {
  status: string;
  ticks: number;
  /** Line index of the `-status`. */
  since: number;
  /** Line index of the latest tick, -1 when none. */
  tick_at: number;
}

interface SideState {
  active: Map<string, Occupant | null>;
  conditions: Map<string, ConditionFacts>;
  total_fainted: number;
  faints: Map<number, number>;
}

interface Line {
  kind: string;
  args: string[];
  tags: Map<string, string>;
}

const TAG = /^\[([A-Za-z_]+)\]\s?(.*)$/;
const IDENT = /^(p[12])([a-c])?: (.+)$/;

const toId = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]+/g, "");

function split(raw: string, index: number): Line {
  if (!raw.startsWith("|")) throw new WorldFactsError("malformed_line", index);
  const parts = raw.slice(1).split("|");
  const args: string[] = [];
  const tags = new Map<string, string>();
  for (const part of parts.slice(1)) {
    const tag = TAG.exec(part);
    if (tag === null) args.push(part);
    else tags.set(tag[1] as string, tag[2] as string);
  }
  return { kind: parts[0] as string, args, tags };
}

/** The effect name of an id-bearing field: `move: Reflect` and `Reflect` both give `reflect`. */
const effectId = (text: string): string => toId(text.replace(/^[a-z ]+: /, ""));

/** Kinds that end a stall move's own lines: whatever follows is another action. */
const BOUNDARY = new Set(["move", "cant", "switch", "drag", "faint", "turn", "upkeep", "", "replace", "swap", "win", "tie"]);

/**
 * The incremental reader: `feed` the lines of one own channel in order, `facts()` at any point (for
 * instance at each `|request|`). A refusal is remembered and thrown by `facts()` from that line on.
 */
export class WorldFactsReader {
  readonly #player: FactsPlayer;
  readonly #stall: ReadonlySet<string>;
  readonly #terrains: ReadonlySet<string>;
  readonly #swappable: ReadonlySet<string>;
  readonly #toxicMax: number;
  readonly #sides: Record<FactsPlayer, SideState> = { p1: newSide(), p2: newSide() };
  readonly #mons = new Map<string, MonStatus>();
  #turn = 0;
  #index = -1;
  #blank = -1;
  #weather: { id: string; started_turn: number } | null = null;
  #terrain: { id: string; started_turn: number } | null = null;
  #pseudo = new Map<string, { started_turn: number }>();
  #pending: { side: FactsPlayer; letter: string } | null = null;
  #failure: WorldFactsError | null = null;

  constructor(player: FactsPlayer, data: WorldFactsData) {
    this.#player = player;
    this.#stall = new Set(data.stall_moves);
    this.#terrains = new Set(data.terrains);
    this.#swappable = new Set(data.swap_side_conditions);
    this.#toxicMax = data.toxic_stage_max;
  }

  /** Reads one line. Never throws: a refusal is kept for `facts()`. */
  feed(raw: string): void {
    this.#index += 1;
    if (this.#failure !== null) return;
    try {
      this.#line(split(raw, this.#index));
    } catch (error) {
      this.#failure = error instanceof WorldFactsError ? error : new WorldFactsError("malformed_line", this.#index);
    }
  }

  /** The facts after the lines fed so far, or the refusal. */
  facts(): WorldFacts {
    if (this.#failure !== null) throw this.#failure;
    if (this.#pending !== null) throw new WorldFactsError("stall_unresolved", this.#index);
    const opposing = this.#player === "p1" ? "p2" : "p1";
    return {
      world_facts_format: WORLD_FACTS_FORMAT,
      turn: this.#turn,
      weather: this.#weather === null ? null : { ...this.#weather },
      terrain: this.#terrain === null ? null : { ...this.#terrain },
      field_conditions: Object.fromEntries([...this.#pseudo].sort(byKey).map(([id, c]) => [id, { ...c }])),
      own: this.#sideFacts(this.#player),
      opponent: this.#sideFacts(opposing),
    };
  }

  #sideFacts(player: FactsPlayer): SideFacts {
    const side = this.#sides[player];
    const active: Record<string, ActiveFacts | null> = {};
    for (const [letter, occupant] of [...side.active].sort(byKey)) {
      if (occupant === null) {
        active[letter] = null;
        continue;
      }
      const status = this.#mons.get(`${player}:${occupant.name}`);
      active[letter] = {
        name: occupant.name,
        active_turns: this.#turn - occupant.switch_turn,
        last_move: occupant.last_move,
        protect_streak: occupant.stall_turn >= this.#turn - 1 ? occupant.stall_count : 0,
        toxic_ticks: status !== undefined && status.status === "tox" ? status.ticks : null,
      };
    }
    return {
      total_fainted: side.total_fainted,
      fainted_this_turn: (side.faints.get(this.#turn) ?? 0) > 0,
      fainted_last_turn: (side.faints.get(this.#turn - 1) ?? 0) > 0,
      active,
      conditions: Object.fromEntries([...side.conditions].sort(byKey).map(([id, c]) => [id, { ...c }])),
    };
  }

  #ident(text: string | undefined): { side: FactsPlayer; letter: string | null; name: string } {
    const match = text === undefined ? null : IDENT.exec(text);
    if (match === null) throw new WorldFactsError("unattributable_line", this.#index);
    return { side: match[1] as FactsPlayer, letter: match[2] ?? null, name: match[3] as string };
  }

  #occupant(text: string | undefined): { side: FactsPlayer; occupant: Occupant; letter: string } {
    const ref = this.#ident(text);
    const occupant = ref.letter === null ? null : this.#sides[ref.side].active.get(ref.letter);
    if (ref.letter === null || occupant === null || occupant === undefined) throw new WorldFactsError("unattributable_line", this.#index);
    return { side: ref.side, occupant, letter: ref.letter };
  }

  #resolveStall(line: Line): void {
    const pending = this.#pending;
    if (pending === null) return;
    const mine = line.args[0]?.startsWith(`${pending.side}${pending.letter}:`) === true;
    const occupant = this.#sides[pending.side].active.get(pending.letter);
    if ((line.kind === "-singleturn" || line.kind === "-fail") && mine && occupant) {
      this.#pending = null;
      if (line.kind === "-fail") {
        occupant.stall_count = 0;
        occupant.stall_turn = -1;
      } else {
        occupant.stall_count = occupant.stall_turn >= this.#turn - 1 ? occupant.stall_count + 1 : 1;
        occupant.stall_turn = this.#turn;
      }
    } else if (BOUNDARY.has(line.kind)) {
      throw new WorldFactsError("stall_unresolved", this.#index);
    }
  }

  #line(line: Line): void {
    this.#resolveStall(line);
    const a = line.args;
    switch (line.kind) {
      case "": this.#blank = this.#index; break;
      case "turn": {
        const n = Number(a[0]);
        if (!Number.isInteger(n) || n < this.#turn) throw new WorldFactsError("malformed_line", this.#index);
        this.#turn = n;
        break;
      }
      case "switch":
      case "drag": this.#switchIn(a[0]); break;
      case "swap": this.#swap(a[0], a[1]); break;
      case "replace": throw new WorldFactsError("illusion_replace", this.#index);
      case "faint": this.#faint(a[0]); break;
      case "move": this.#move(line); break;
      case "-status": this.#status(a[0], a[1]); break;
      case "-curestatus": this.#cure(a[0]); break;
      case "-cureteam": this.#cureTeam(a[0]); break;
      case "-damage":
      case "-heal": this.#hp(line); break;
      case "upkeep": this.#upkeep(); break;
      case "-weather": this.#setWeather(line); break;
      case "-fieldstart": this.#fieldStart(a[0]); break;
      case "-fieldend": this.#fieldEnd(a[0]); break;
      case "-sidestart": this.#sideStart(a[0], a[1]); break;
      case "-sideend": this.#sides[this.#ident(a[0]).side].conditions.delete(effectId(a[1] ?? "")); break;
      case "-swapsideconditions": {
        const [one, two] = [this.#sides.p1.conditions, this.#sides.p2.conditions];
        for (const id of this.#swappable) {
          const a = one.get(id);
          const b = two.get(id);
          one.delete(id);
          two.delete(id);
          if (b !== undefined) one.set(id, b);
          if (a !== undefined) two.set(id, a);
        }
        break;
      }
      default: break;
    }
  }

  #switchIn(text: string | undefined): void {
    const ref = this.#ident(text);
    if (ref.letter === null) throw new WorldFactsError("unattributable_line", this.#index);
    this.#sides[ref.side].active.set(ref.letter, { name: ref.name, switch_turn: this.#turn, last_move: null, stall_count: 0, stall_turn: -1 });
    const status = this.#mons.get(`${ref.side}:${ref.name}`);
    if (status !== undefined && status.status === "tox") {
      status.ticks = 0;
      status.tick_at = -1;
    }
  }

  #swap(text: string | undefined, target: string | undefined): void {
    const ref = this.#ident(text);
    const to = String.fromCharCode(97 + Number(target));
    const active = this.#sides[ref.side].active;
    if (ref.letter === null || !active.has(ref.letter) || !active.has(to) || !Number.isInteger(Number(target))) {
      throw new WorldFactsError("unattributable_line", this.#index);
    }
    const one = active.get(ref.letter) as Occupant | null;
    active.set(ref.letter, active.get(to) as Occupant | null);
    active.set(to, one);
  }

  #faint(text: string | undefined): void {
    const ref = this.#ident(text);
    const side = this.#sides[ref.side];
    if (ref.letter === null || !side.active.has(ref.letter)) throw new WorldFactsError("unattributable_line", this.#index);
    side.active.set(ref.letter, null);
    side.total_fainted += 1;
    side.faints.set(this.#turn, (side.faints.get(this.#turn) ?? 0) + 1);
    this.#mons.delete(`${ref.side}:${ref.name}`);
  }

  #move(line: Line): void {
    const { side, occupant, letter } = this.#occupant(line.args[0]);
    const move = toId(line.args[1] ?? "");
    if (move === "") throw new WorldFactsError("malformed_line", this.#index);
    const from = line.tags.get("from");
    // A move another effect made the Pokemon use (called, reflected, copied) is not its own choice.
    if (from !== undefined && /^(move|ability|item):/.test(from)) return;
    occupant.last_move = move;
    if (this.#stall.has(move)) this.#pending = { side, letter };
  }

  #status(text: string | undefined, status: string | undefined): void {
    const ref = this.#ident(text);
    this.#mons.set(`${ref.side}:${ref.name}`, { status: status ?? "", ticks: 0, since: this.#index, tick_at: -1 });
  }

  #cure(text: string | undefined): void {
    const ref = this.#ident(text);
    this.#mons.delete(`${ref.side}:${ref.name}`);
  }

  #cureTeam(text: string | undefined): void {
    const side = this.#ident(text).side;
    for (const key of [...this.#mons.keys()]) if (key.startsWith(`${side}:`)) this.#mons.delete(key);
  }

  #hp(line: Line): void {
    const from = line.tags.get("from");
    const poison = line.kind === "-damage" ? from === "psn" : from === "ability: Poison Heal";
    if (!poison) return;
    const ref = this.#ident(line.args[0]);
    const status = this.#mons.get(`${ref.side}:${ref.name}`);
    if (status !== undefined && status.status === "tox") {
      status.ticks = Math.min(this.#toxicMax, status.ticks + 1);
      status.tick_at = this.#index;
    }
  }

  /** The residual section ended: every badly poisoned Pokemon in play must have shown its tick. */
  #upkeep(): void {
    for (const player of ["p1", "p2"] as const) {
      for (const occupant of this.#sides[player].active.values()) {
        if (occupant === null) continue;
        const status = this.#mons.get(`${player}:${occupant.name}`);
        if (status !== undefined && status.status === "tox" && status.since < this.#blank && status.tick_at < this.#blank) {
          throw new WorldFactsError("toxic_tick_unobserved", this.#index);
        }
      }
    }
  }

  #setWeather(line: Line): void {
    if (line.tags.has("upkeep")) return;
    const id = toId(line.args[0] ?? "");
    this.#weather = id === "none" || id === "" ? null : { id, started_turn: this.#turn };
  }

  #fieldStart(text: string | undefined): void {
    const id = effectId(text ?? "");
    if (id === "") throw new WorldFactsError("malformed_line", this.#index);
    if (this.#terrains.has(id)) this.#terrain = { id, started_turn: this.#turn };
    else this.#pseudo.set(id, { started_turn: this.#turn });
  }

  #fieldEnd(text: string | undefined): void {
    const id = effectId(text ?? "");
    if (this.#terrains.has(id)) {
      if (this.#terrain?.id === id) this.#terrain = null;
    } else this.#pseudo.delete(id);
  }

  #sideStart(text: string | undefined, name: string | undefined): void {
    const id = effectId(name ?? "");
    if (id === "") throw new WorldFactsError("malformed_line", this.#index);
    const conditions = this.#sides[this.#ident(text).side].conditions;
    const known = conditions.get(id);
    if (known === undefined) conditions.set(id, { started_turn: this.#turn, layers: 1 });
    else known.layers += 1;
  }
}

function newSide(): SideState {
  return { active: new Map(), conditions: new Map(), total_fainted: 0, faints: new Map() };
}

function byKey<T>(x: [string, T], y: [string, T]): number {
  return x[0] < y[0] ? -1 : x[0] > y[0] ? 1 : 0;
}

/** The facts after all the lines of the channel, or the refusal. */
export function worldFacts(input: { readonly channel: OwnChannel; readonly data: WorldFactsData }): WorldFacts {
  const reader = new WorldFactsReader(input.channel.player, input.data);
  for (const line of input.channel.lines) reader.feed(line);
  return reader.facts();
}

export interface RequestFacts {
  /** Index of the `|request|` line in the channel. */
  line: number;
  facts: WorldFacts | null;
  /** The refusal code when `facts` is null. */
  refusal: WorldFactsRefusal | null;
}

/** The facts at every `|request|` line of the channel (the facts before the request line), with a refusal counted where one stands. */
export function worldFactsAtRequests(input: { readonly channel: OwnChannel; readonly data: WorldFactsData }): RequestFacts[] {
  const reader = new WorldFactsReader(input.channel.player, input.data);
  const out: RequestFacts[] = [];
  input.channel.lines.forEach((line, index) => {
    if (line.startsWith("|request|")) {
      try {
        out.push({ line: index, facts: reader.facts(), refusal: null });
      } catch (error) {
        out.push({ line: index, facts: null, refusal: (error as WorldFactsError).code });
      }
    }
    reader.feed(line);
  });
  return out;
}
