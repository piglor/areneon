// The history facts (ADR-022 erratum E-22.5.2, slice H1). The engine reads battle history that neither the
// shared reader nor the world facts track: `activeMoveActions` (Fake Out), `volatiles.stall.counter` (Protect),
// `timesAttacked` (Rage Fist), `hurtThisTurn` and `attackedBy` (Assurance, Comeuppance), `moveLastTurnResult`
// (Stomping Tantrum), `statsRaisedThisTurn` / `statsLoweredThisTurn` (Lash Out, Alluring Voice), `ateBerry` and
// `lastItem` (Belch, Recycle), `moveSlots[].used` (Last Resort), `side.totalFainted` (Last Respects),
// `battle.lastMove` (Copycat) and the pending slot conditions (Wish, Future Sight). This module derives each one
// from ONE player's own channel and nothing else.
//
// It is a module of its own with its own format id and file hash (`historyFactsSha256()` in `index.ts`). The shared
// reader (`interpreter.cjs`, `readerSha256`) and the world facts (`world-facts.ts`, `WORLD_FACTS_FORMAT`) are not
// touched, so no reader, baseline, belief, feature-set or search identity moves. It imports nothing (a test
// enforces it), and it holds no game list: the stall moves, the counter constants, the move-line sources that the
// engine runs as actions of their own, the slot conditions with their span and the volatile copy table are
// engine data the caller passes (`HistoryFactsData`, read from the pinned engine by the bridge). Only protocol
// wire tokens appear here (`[from]`, `[eat]`, `[of]`, `-singleturn`, `-fail`, ...).
//
// What it reads is public: the player's own stream, never the secret half of a `|split|` and never the other
// player's requests. The input type is the branded `OwnChannel` that only `ownChannel` makes. Nothing here
// hashes, draws or reads a file at run time.
//
// Definitions, each held equal to the live engine's field by `history-facts-parity.test.ts`:
// - `move_actions` is `activeMoveActions`: the world facts' rule (own `|move|` lines without a call tag,
//   `[from]lockedmove`, tags in `action_sources`, `|cant|` lines and the confusion self-hit) since the switch-in.
// - `stall_counter` is `volatiles.stall.counter`, or null without the volatile: it follows consecutive successful
//   stalling uses (`-singleturn` of the user after its stalling move; `-fail` ends the chain; the volatile lasts
//   the turn after a use), the counter starting at `counter_start`, multiplied by `counter_factor` on each further
//   use until it reaches `counter_max`. A Baton Pass hands it over unless the volatile copy table says it is not copied.
// - `times_attacked` is `timesAttacked`: one per `-damage` (no `[from]`) that a move
//   action dealt to another Pokemon than its user (a hit a Substitute takes is no number damage, so it counts none). A `-transform` copies the target's count. It resets when the
//   Pokemon leaves play (the champions mod's `clearVolatile`).
// - `hurt_this_turn` is `hurtThisTurn !== null`: a `-damage` that went through `Battle.damage` (a direct damage
//   line is the user's own `-damage` with no tag in its own move action, or `[from] recoil`).
//   It resets at a turn rollover (not at turn 1) and when the Pokemon leaves play.
// - `attackers_this_turn` is the positions (`own:a`, `opponent:b`) of the sources of the move actions that dealt a
//   counted hit this turn (`attackedBy` entries of this turn with a numeric damage value); it resets at every `|turn|`.
// - `last_move_failed` is `moveLastTurnResult === false`: the result of the Pokemon's action of the previous turn.
//   An action fails when it ends in `|cant|`, or its lines show a failure (`-fail`, `-notarget` of the user, `-miss`
//   of the user, `-immune`) and no effect on a target.
// - `stats_raised_this_turn` / `stats_lowered_this_turn` are `statsRaisedThisTurn` / `statsLoweredThisTurn`: a
//   `-boost` / `-unboost` line on the Pokemon, reset as `hurt_this_turn`.
// - `ate_berry` and `last_item` (per Pokemon of the roster, never reset on a switch): `-enditem` with a tag in `eat_tags` (`[eat]`, `[weaken]`) sets
//   both for the Pokemon; `[from] stealeat` sets `ate_berry` for the `[of]` Pokemon; a plain `-enditem` (or
//   `[from] gem`, or a `[from]` in `item_use_sources` (Fling) sets `last_item`; any other `[from]` is a theft or a removal that sets neither.
// - `moves_used` (own Pokemon only) is `moveSlots[].used`: the ids of the own moves the Pokemon's own move lines
//   used since its switch-in, among the moves its own request lists.
// - `total_fainted` (per side) is `side.totalFainted`; `field_last_move` is `battle.lastMove` (the latest `|move|` line,
//   whatever its tag); `pending_slot` is the timed slot conditions of `slot_conditions` per side and position, from
//   the creating move line (a `-fail` cancels it) for `residual_turns` turn residuals; `volatile_started` is the turn
//   each `-start` volatile began (the world facts' rule).
//
// Illusion: a `|replace|` marks the Pokemon at that position `unknown` until it leaves play (no fact is guessed).
// A line the module cannot place is a typed `HistoryFactsError`; the facts before it stand.

/** The format of the derivation: bump when how a fact is computed from the lines changes. */
export const HISTORY_FACTS_FORMAT = "history-facts/1";

export type HistoryPlayer = "p1" | "p2";

/** Reviewed data from the pinned engine, passed by the caller; this module holds no game list. */
export interface HistoryFactsData {
  /** Ids of the moves that add the engine's `stall` volatile. */
  readonly stall_moves: readonly string[];
  /** Ids of the stall moves whose engine target is the user's own side: they print no line when the side's condition is already up. */
  readonly ally_side_moves?: readonly string[];
  /** The stall condition's counter: set to `start` on the first use, multiplied by `factor` on each further use while below `max`. */
  readonly stall_counter: { readonly start: number; readonly factor: number; readonly max: number };
  /** The `[from]` texts of the move lines the engine ran as move actions of their own (Round, Instruct, Dancer). */
  readonly action_sources: readonly string[];
  /** Those of `action_sources` that run with the engine's `externalMove` (Dancer): no PP is deducted for them, so they set no `moveSlots[].used`. Absent: none. */
  readonly external_action_sources?: readonly string[];
  /** The tags of `-enditem` lines that mark an eaten berry (`eat`, and the one the damage-weakening berries print). */
  readonly eat_tags: readonly string[];
  /** The `[from]` tags of `-enditem` lines of a berry the move's target ate (`stealeat`): the `[of]` Pokemon ate it. */
  readonly steal_eat_tags: readonly string[];
  /** The `[from]` tags of `-enditem` lines of a consumed gem: the user's `lastItem` is set. */
  readonly gem_tags: readonly string[];
  /** Ids of the items whose `-enditem` line is printed where the item is cleared directly, not used: no `lastItem` is set (the balloon that pops). */
  readonly direct_end_items: readonly string[];
  /** Ids of the berries with an `onEat` effect: one thrown at a target is eaten by it (`ateBerry`), with no line of its own. */
  readonly eaten_berries: readonly string[];
  /** The `[from]` texts of `-item` lines that give a Pokemon back its `lastItem` and clear it (Recycle, Harvest). */
  readonly last_item_restorers: readonly string[];
  /** The `[from]` texts of `-item` lines (Pickup) that take another Pokemon's `lastItem`: which one is not named, so every active holder of that item is unsettled. */
  readonly last_item_takers: readonly string[];
  /** The `[from]` texts of `-enditem` lines (`move: Fling`) whose effect sets `lastItem` on the Pokemon the line names. */
  readonly item_use_sources: readonly string[];
  /** The timed slot conditions: the moves whose use starts one at a position and the turn residuals it lasts. */
  readonly slot_conditions: readonly { readonly id: string; readonly moves: readonly string[]; readonly residual_turns: number }[];
  /** The reasons of `|cant|` lines whose `BeforeMove` handler returns null (the Pokemon had no option: recharge): the action does not count as failed. */
  readonly null_cant_reasons: readonly string[];
  /** Ids of the moves that break a Protect-like volatile (`breaksProtect`): the engine prints `-activate|TARGET|move: NAME` when one does and deletes the target's `stall` volatile. */
  readonly protect_breaker_moves: readonly string[];
  /** The volatiles a Baton Pass does not hand over (`volatile-copy`), and those a Shed Tail does. */
  readonly volatile_not_copied: readonly string[];
  readonly shed_tail_copies: readonly string[];
}

declare const OWN: unique symbol;

/** One player's own channel, checked. Only `ownChannel` makes one. */
export interface OwnHistoryChannel {
  readonly [OWN]: true;
  readonly player: HistoryPlayer;
  readonly lines: readonly string[];
}

export const HISTORY_FACTS_REFUSALS = ["bad_player", "secret_marker", "malformed_line", "unattributable_line", "stall_unresolved"] as const;
export type HistoryFactsRefusal = (typeof HISTORY_FACTS_REFUSALS)[number];

/** A refusal: a code and the 0-based index of the line, in a fixed message that quotes no line. */
export class HistoryFactsError extends Error {
  override name = "HistoryFactsError";
  readonly code: HistoryFactsRefusal;
  readonly line: number | null;
  constructor(code: HistoryFactsRefusal, line: number | null) {
    super(`history-facts: ${code}${line === null ? "" : ` at line ${line}`}`);
    this.code = code;
    this.line = line;
  }
}

/** Checks one player's own channel lines: a player, lines that start with `|`, and no `|split|` marker. */
export function ownHistoryChannel(player: string, lines: readonly string[]): OwnHistoryChannel {
  if (player !== "p1" && player !== "p2") throw new HistoryFactsError("bad_player", null);
  lines.forEach((line, index) => {
    if (!line.startsWith("|")) throw new HistoryFactsError("malformed_line", index);
    if (line === "|split" || line.startsWith("|split|")) throw new HistoryFactsError("secret_marker", index);
  });
  return { player, lines } as unknown as OwnHistoryChannel;
}

/** The facts of one Pokemon in a position. `unknown` (Illusion) carries no fact. */
export type PokemonHistory =
  | {
      name: string;
      unknown: false;
      move_actions: number;
      /** The stall volatile's counter, null without the volatile. */
      stall_counter: number | null;
      times_attacked: number;
      hurt_this_turn: boolean;
      /** Positions (`own:a`, `opponent:b`) that hit it this turn, sorted; null when a delayed attack's user has left play. */
      attackers_this_turn: string[] | null;
      /** `moveLastTurnResult === false`; null when the lines of the action cannot settle it (a called or reflected move). */
      last_move_failed: boolean | null;
      stats_raised_this_turn: boolean;
      stats_lowered_this_turn: boolean;
      /** Own Pokemon only: the own move ids used since the switch-in, sorted. Null for an opponent. */
      moves_used: string[] | null;
      volatile_started: Record<string, { started_turn: number; after_residual: boolean }>;
    }
  | { name: string; unknown: true };

/** The `last_item` of a Pokemon whose last item another Pokemon's Pickup may have taken: the channel cannot say. */
export const UNSETTLED = "unsettled";

export interface RosterHistory {
  /** Null when the Pokemon's facts are unknown (Illusion). */
  ate_berry: boolean | null;
  /** The id of the last item the Pokemon used or ate, null when none; `UNSETTLED` after a Pickup that may have taken it. */
  last_item: string | null;
}

export interface SideHistory {
  total_fainted: number;
  /** Per position letter that has held a Pokemon; null when the position is empty now. */
  active: Record<string, PokemonHistory | null>;
  /** Per position letter, the pending timed slot conditions by id. */
  pending_slot: Record<string, Record<string, { started_turn: number; after_residual: boolean; remaining: number }>>;
  /** Per Pokemon of the roster seen so far, by shown nickname. */
  roster: Record<string, RosterHistory>;
}

export interface HistoryFacts {
  history_facts_format: string;
  turn: number;
  /** The id of the battle's last move (`battle.lastMove`), null before any. */
  field_last_move: string | null;
  own: SideHistory;
  opponent: SideHistory;
}

interface Occupant {
  name: string;
  unknown: boolean;
  move_actions: number;
  stall_count: number;
  stall_turn: number;
  times_attacked: number;
  hurt: boolean;
  attackers: Set<string>;
  /** A hit this turn came from a Pokemon the channel cannot place (a delayed attack whose user left play). */
  attackers_unsettled: boolean;
  /** The result of this turn's action: "ok", "failed" or "none" (no action yet); the previous turn's is `last_failed`. */
  result: "ok" | "failed" | "unknown" | "none";
  last_failed: boolean | null;
  raised: boolean;
  lowered: boolean;
  used: Set<string>;
  volatiles: Map<string, { started_turn: number; after_residual: boolean }>;
}

interface Slot {
  id: string;
  started_turn: number;
  after_residual: boolean;
  /** The Pokemon whose move started it (the attacker of a Future Sight that lands) and its side. */
  source: Occupant;
  source_side: HistoryPlayer;
}

interface SideState {
  active: Map<string, Occupant | null>;
  total_fainted: number;
  slots: Map<string, Map<string, Slot>>;
  roster: Map<string, { ate: boolean; last_item: string; unknown: boolean }>;
}

interface Line {
  kind: string;
  args: string[];
  tags: Map<string, string>;
}

/** The move action being read: its user, the target slot of a creating move, and what its lines showed. */
interface Action {
  side: HistoryPlayer;
  letter: string;
  occupant: Occupant;
  /** The position the move line names as its target (null when it names none or a Pokemon not in a position). */
  target: { side: HistoryPlayer; letter: string } | null;
  /** A counted hit: a `-damage` with no `[from]` on another Pokemon. */
  hit: boolean;
  /** `-miss` or `-immune` lines. */
  misses: number;
  /** An effect line on the user of a self-targeted move. */
  selfEffect: boolean;
  /** Any other line that shows the move did something or failed in a way the lines do not settle (`-fail`, an effect on another Pokemon, a zero boost). */
  other: boolean;
  /** The landing of a delayed attack: its hits count, but it is no action of the Pokemon that used it. */
  delayed: boolean;
  /** A move another effect called or reflected ran inside this action: its result is that nested move's, which the lines do not separate. */
  nested: boolean;
  /** The move's own target is its user (a self-targeted move: an effect line on the user is its success). */
  selfTarget: boolean;
  /** A slot condition this action would start unless a `-fail` follows. */
  slot: { side: HistoryPlayer; letter: string; id: string; move: string } | null;
}

const TAG = /^\[([A-Za-z_]+)\]\s?(.*)$/;
const IDENT = /^(p[12])([a-c])?: (.+)$/;
const toId = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]+/g, "");
const effectId = (text: string): string => toId(text.replace(/^[a-z ]+: /, ""));

function split(raw: string, index: number): Line {
  if (!raw.startsWith("|")) throw new HistoryFactsError("malformed_line", index);
  const parts = raw.slice(1).split("|");
  const args: string[] = [];
  const tags = new Map<string, string>();
  const body = parts[0] === "request" ? [parts.slice(1).join("|")] : parts.slice(1);
  for (const part of body) {
    const tag = parts[0] === "request" ? null : TAG.exec(part);
    if (tag === null) args.push(part);
    else tags.set(tag[1] as string, tag[2] as string);
  }
  return { kind: parts[0] as string, args, tags };
}

/** Kinds that end a move action's own lines: whatever follows is another action. */
const BOUNDARY = new Set(["move", "cant", "switch", "drag", "faint", "turn", "upkeep", "", "replace", "swap", "win", "tie"]);
/** Kinds that end the hits a move action dealt (a `faint` comes after the hits it follows, so it is not one). */
const ACTION_END = new Set(["move", "cant", "switch", "drag", "turn", "upkeep", "", "replace", "swap", "win", "tie"]);
/** Lines on a target that show a move reached it. */
const EFFECT = new Set(["-damage", "-heal", "-boost", "-unboost", "-status", "-start", "-sidestart", "-fieldstart", "-weather", "-singleturn", "-singlemove", "-curestatus", "-setboost", "-item", "-ability", "-clearallboost", "-clearboost", "-clearnegativeboost", "-clearpositiveboost", "-copyboost", "-swapboost", "-invertboost", "-cureteam", "-sethp", "-transform", "-formechange", "-prepare", "-mustrecharge", "-sideend", "-fieldend", "-end", "-enditem"]);

export class HistoryFactsReader {
  readonly #player: HistoryPlayer;
  readonly #data: HistoryFactsData;
  readonly #stall: ReadonlySet<string>;
  readonly #allySide: ReadonlySet<string>;
  readonly #actionSources: ReadonlySet<string>;
  readonly #externalSources: ReadonlySet<string>;
  readonly #restorers: ReadonlySet<string>;
  readonly #takers: ReadonlySet<string>;
  readonly #berries: ReadonlySet<string>;
  readonly #eatTags: ReadonlySet<string>;
  readonly #stealEat: ReadonlySet<string>;
  readonly #gem: ReadonlySet<string>;
  readonly #directEnd: ReadonlySet<string>;
  readonly #itemUse: ReadonlySet<string>;
  readonly #nullCant: ReadonlySet<string>;
  readonly #breakers: ReadonlySet<string>;
  readonly #notCopied: ReadonlySet<string>;
  readonly #shedTail: ReadonlySet<string>;
  readonly #sides: Record<HistoryPlayer, SideState> = { p1: newSide(), p2: newSide() };
  /** The own move ids of the Pokemon in each own position, from the latest request (the position, not the nickname: an Illusion shows another name). */
  readonly #requestMoves = new Map<string, Set<string>>();
  /** `side:move` of the side-condition stall moves whose `-singleturn` was printed this turn. */
  readonly #guards = new Set<string>();
  /** The position each Pokemon of each side last held, by nickname (a fainted target is named without one). */
  readonly #lastLetter = new Map<string, string>();
  #turn = 0;
  #index = -1;
  #upkeepAt = -1;
  #lastMove: string | null = null;
  #action: Action | null = null;
  /** The Pokemon whose `|cant|` was the previous line: a sleep-usable move follows it in the same action (one move action, not two). */
  #lastCant: Occupant | null = null;
  #pending: { side: HistoryPlayer; letter: string; move: string } | null = null;
  #failure: HistoryFactsError | null = null;

  constructor(player: HistoryPlayer, data: HistoryFactsData) {
    this.#player = player;
    this.#data = data;
    this.#stall = new Set(data.stall_moves);
    this.#allySide = new Set(data.ally_side_moves ?? []);
    this.#actionSources = new Set(data.action_sources.map(toId));
    this.#externalSources = new Set((data.external_action_sources ?? []).map(toId));
    this.#restorers = new Set(data.last_item_restorers.map(toId));
    this.#takers = new Set(data.last_item_takers.map(toId));
    this.#berries = new Set(data.eaten_berries);
    this.#eatTags = new Set(data.eat_tags);
    this.#stealEat = new Set(data.steal_eat_tags);
    this.#gem = new Set(data.gem_tags);
    this.#directEnd = new Set(data.direct_end_items);
    this.#itemUse = new Set(data.item_use_sources.map(toId));
    this.#nullCant = new Set(data.null_cant_reasons);
    this.#breakers = new Set(data.protect_breaker_moves);
    this.#notCopied = new Set(data.volatile_not_copied);
    this.#shedTail = new Set(data.shed_tail_copies);
  }

  /** Reads one line. Never throws: a refusal is kept for `facts()`. */
  feed(raw: string): void {
    this.#index += 1;
    if (this.#failure !== null) return;
    try {
      this.#line(split(raw, this.#index));
    } catch (error) {
      this.#failure = error instanceof HistoryFactsError ? error : new HistoryFactsError("malformed_line", this.#index);
    }
  }

  /** The facts after the lines fed so far, or the refusal. */
  facts(): HistoryFacts {
    if (this.#failure !== null) throw this.#failure;
    if (this.#pending !== null) throw new HistoryFactsError("stall_unresolved", this.#index);
    const opposing = this.#player === "p1" ? "p2" : "p1";
    return {
      history_facts_format: HISTORY_FACTS_FORMAT,
      turn: this.#turn,
      field_last_move: this.#lastMove,
      own: this.#sideFacts(this.#player),
      opponent: this.#sideFacts(opposing),
    };
  }

  #counter(occupant: Occupant): number | null {
    // The volatile lasts through the turn after a use, until that turn's residual.
    if (occupant.stall_count === 0 || occupant.stall_turn < this.#turn - (this.#upkeepAt === this.#turn ? 0 : 1)) return null;
    const { start, factor, max } = this.#data.stall_counter;
    let counter = start;
    for (let use = 1; use < occupant.stall_count; use++) if (counter < max) counter *= factor;
    return counter;
  }

  #sideFacts(player: HistoryPlayer): SideHistory {
    const side = this.#sides[player];
    const own = player === this.#player;
    const active: Record<string, PokemonHistory | null> = {};
    for (const [letter, occupant] of [...side.active].sort(byKey)) {
      if (occupant === null) {
        active[letter] = null;
      } else if (occupant.unknown) {
        active[letter] = { name: occupant.name, unknown: true };
      } else {
        active[letter] = {
          name: occupant.name,
          unknown: false,
          move_actions: occupant.move_actions,
          stall_counter: this.#counter(occupant),
          times_attacked: occupant.times_attacked,
          hurt_this_turn: occupant.hurt,
          attackers_this_turn: occupant.attackers_unsettled ? null : [...occupant.attackers].sort(),
          last_move_failed: occupant.last_failed,
          stats_raised_this_turn: occupant.raised,
          stats_lowered_this_turn: occupant.lowered,
          moves_used: own ? [...occupant.used].sort() : null,
          volatile_started: Object.fromEntries([...occupant.volatiles].sort(byKey).map(([id, began]) => [id, { ...began }])),
        };
      }
    }
    const pending: SideHistory["pending_slot"] = {};
    for (const [letter, slots] of [...side.slots].sort(byKey)) {
      const live = Object.fromEntries(
        [...slots]
          .sort(byKey)
          .map(([id, s]) => [id, { started_turn: s.started_turn, after_residual: s.after_residual, remaining: this.#remaining(s) }] as const)
          .filter(([, s]) => s.remaining >= 1),
      );
      if (Object.keys(live).length > 0) pending[letter] = live;
    }
    return {
      total_fainted: side.total_fainted,
      active,
      pending_slot: pending,
      roster: Object.fromEntries([...side.roster].sort(byKey).map(([name, r]) => [name, r.unknown ? { ate_berry: null, last_item: null } : { ate_berry: r.ate, last_item: r.last_item === "" ? null : r.last_item === "?" ? UNSETTLED : r.last_item }])),
    };
  }

  /** Residuals left: the span minus the residuals that have run since it began (the residual of the turn it began in counts unless it began after it). */
  #remaining(slot: Slot): number {
    const span = this.#data.slot_conditions.find((c) => c.id === slot.id)?.residual_turns ?? 0;
    return span - (this.#turn - slot.started_turn - (slot.after_residual ? 1 : 0) + (this.#upkeepAt === this.#turn ? 1 : 0));
  }

  #ident(text: string | undefined): { side: HistoryPlayer; letter: string | null; name: string } {
    const match = text === undefined ? null : IDENT.exec(text);
    if (match === null) throw new HistoryFactsError("unattributable_line", this.#index);
    return { side: match[1] as HistoryPlayer, letter: match[2] ?? null, name: match[3] as string };
  }

  /** The Pokemon a position field names, or null when the position is empty or the field names none. */
  #at(text: string | undefined): { side: HistoryPlayer; letter: string; occupant: Occupant } | null {
    const match = text === undefined ? null : IDENT.exec(text);
    if (match === null || match[2] === undefined) return null;
    const occupant = this.#sides[match[1] as HistoryPlayer].active.get(match[2]);
    return occupant === undefined || occupant === null ? null : { side: match[1] as HistoryPlayer, letter: match[2], occupant };
  }

  #occupant(text: string | undefined): { side: HistoryPlayer; letter: string; occupant: Occupant } {
    const found = this.#at(text);
    if (found === null) throw new HistoryFactsError("unattributable_line", this.#index);
    return found;
  }

  #seen(side: HistoryPlayer, letter: string): string {
    return `${side === this.#player ? "own" : "opponent"}:${letter}`;
  }

  #residualRan(): boolean {
    return this.#turn === 0 || this.#upkeepAt === this.#turn;
  }

  #resolveStall(line: Line): void {
    if (line.kind === "-singleturn") {
      const side = IDENT.exec(line.args[0] ?? "")?.[1];
      const shown = toId((line.args[1] ?? "").replace(/^move: /, ""));
      if (side !== undefined && this.#allySide.has(shown)) this.#guards.add(`${side}:${shown}`);
    }
    const pending = this.#pending;
    if (pending === null) return;
    const mine = line.args[0]?.startsWith(`${pending.side}${pending.letter}:`) === true;
    const occupant = this.#sides[pending.side].active.get(pending.letter);
    if ((line.kind === "-singleturn" || line.kind === "-fail") && mine && occupant) {
      this.#pending = null;
      if (line.kind === "-fail") {
        occupant.stall_count = 0;
        occupant.stall_turn = -1;
      } else this.#stallSucceeded(occupant);
    } else if (BOUNDARY.has(line.kind)) {
      if (occupant && this.#allySide.has(pending.move) && this.#guards.has(`${pending.side}:${pending.move}`)) {
        this.#pending = null;
        this.#stallSucceeded(occupant);
      } else throw new HistoryFactsError("stall_unresolved", this.#index);
    }
  }

  #stallSucceeded(occupant: Occupant): void {
    occupant.stall_count = occupant.stall_turn >= this.#turn - 1 ? occupant.stall_count + 1 : 1;
    occupant.stall_turn = this.#turn;
  }

  /** The action ends where another begins: its result is recorded for `last_move_failed`, and a creating move that was not cancelled starts its slot condition. */
  #endAction(): void {
    const action = this.#action;
    if (action === null) return;
    this.#action = null;
    if (action.delayed) return;
    action.occupant.result = action.nested ? "unknown" : action.hit || (action.selfTarget && action.selfEffect && action.misses === 0 && !action.other) ? "ok" : action.misses > 0 && !action.selfEffect && !action.other ? "failed" : "unknown";
    if (action.slot !== null) this.#startSlot(action.slot, action.occupant, action.side);
  }

  #startSlot(slot: { side: HistoryPlayer; letter: string; id: string }, source: Occupant, user: HistoryPlayer): void {
    const slots = this.#sides[slot.side].slots.get(slot.letter) ?? new Map<string, Slot>();
    const known = slots.get(slot.id);
    if (known === undefined || this.#remaining(known) < 1) slots.set(slot.id, { id: slot.id, started_turn: this.#turn, after_residual: this.#residualRan(), source, source_side: user });
    this.#sides[slot.side].slots.set(slot.letter, slots);
  }

  #line(line: Line): void {
    const justCant = this.#lastCant;
    this.#lastCant = null;
    this.#resolveStall(line);
    if (ACTION_END.has(line.kind) && this.#action !== null && !this.#nestedMove(line)) this.#endAction();
    const a = line.args;
    switch (line.kind) {
      case "request": this.#request(a[0]); break;
      case "turn": {
        const n = Number(a[0]);
        if (!Number.isInteger(n) || n < this.#turn) throw new HistoryFactsError("malformed_line", this.#index);
        this.#turn = n;
        this.#guards.clear();
        this.#rollover(n);
        break;
      }
      case "upkeep": this.#upkeepAt = this.#turn; break;
      case "switch":
      case "drag": this.#switchIn(a[0], line); break;
      case "swap": this.#swap(a[0], a[1]); break;
      case "replace": this.#replace(a[0]); break;
      case "faint": this.#faint(a[0]); break;
      case "move": this.#move(line, justCant); break;
      case "cant": this.#cant(a[0], a[1]); break;
      case "-fail": this.#fail(a[0]); break;
      case "-notarget": this.#fail(a[0]); break;
      case "-miss": if (this.#action !== null) this.#action.misses += 1; break;
      case "-immune": if (this.#action !== null) this.#action.misses += 1; break;
      case "-damage": this.#damage(line); break;
      case "-boost": this.#stat(a[0], true); break;
      case "-unboost": this.#stat(a[0], false); break;
      case "-enditem": this.#endItem(line); break;
      case "-item": this.#item(line); break;
      case "-transform": this.#transform(a[0], a[1]); break;
      case "-start": this.#volatileStart(a[0], a[1]); break;
      case "-end": this.#volatileEnd(a[0], a[1]); this.#slotLands(a[0], a[1]); break;
      case "-activate": this.#activate(line); break;
      default: break;
    }
    if (this.#action !== null && EFFECT.has(line.kind) && line.kind !== "-damage") this.#effect(line);
  }

  /** An effect line inside an action: on the user of a self-targeted move it is that move's success, anywhere else it is left unsettled. */
  #effect(line: Line): void {
    const action = this.#action as Action;
    const zero = (line.kind === "-boost" || line.kind === "-unboost") && Number(line.args[2]) === 0;
    if (!zero && action.selfTarget && this.#selfLine(line.args[0])) action.selfEffect = true;
    else action.other = true;
  }

  /** A `move` line another effect made the Pokemon run (called, copied, reflected) that is not an action of its own: it belongs to the action that is open. */
  #nestedMove(line: Line): boolean {
    const from = line.tags.get("from");
    return line.kind === "move" && from !== undefined && /^(move|ability|item):/.test(from) && !this.#actionSources.has(toId(from));
  }

  /** Whether a line is on the action's own user (an effect on the user alone is no sign the move reached a target). */
  #selfLine(text: string | undefined): boolean {
    const action = this.#action;
    return action !== null && text !== undefined && text.startsWith(`${action.side}${action.letter}:`);
  }

  /** The own request: the moves each own position lists with a PP count (for `moves_used`; Struggle and a locked move's continuation carry none). */
  #request(body: string | undefined): void {
    let parsed: unknown;
    try {
      parsed = JSON.parse(body ?? "");
    } catch {
      return;
    }
    const request = parsed as { active?: { moves?: { id?: unknown; maxpp?: unknown }[] }[] } | null;
    const active = request?.active;
    if (!Array.isArray(active)) return;
    active.forEach((entry, i) => {
      if (!Array.isArray(entry?.moves)) return;
      this.#requestMoves.set(String.fromCharCode(97 + i), new Set(entry.moves.map((m) => (typeof m?.id === "string" && typeof m.maxpp === "number" ? m.id : "")).filter((id) => id !== "")));
    });
  }

  #rollover(n: number): void {
    for (const side of [this.#sides.p1, this.#sides.p2]) {
      for (const occupant of side.active.values()) {
        if (occupant === null) continue;
        occupant.last_failed = occupant.result === "unknown" ? null : occupant.result === "failed";
        occupant.result = "none";
        occupant.attackers.clear();
        occupant.attackers_unsettled = false;
        if (n !== 1) {
          occupant.hurt = false;
          occupant.raised = false;
          occupant.lowered = false;
        }
      }
    }
  }

  #switchIn(text: string | undefined, line: Line): void {
    const ref = this.#ident(text);
    if (ref.letter === null) throw new HistoryFactsError("unattributable_line", this.#index);
    const side = this.#sides[ref.side];
    const out = side.active.get(ref.letter) ?? null;
    const occupant: Occupant = { name: ref.name, unknown: false, move_actions: 0, stall_count: 0, stall_turn: -1, times_attacked: 0, hurt: false, attackers: new Set(), attackers_unsettled: false, result: "none", last_failed: false, raised: false, lowered: false, used: new Set(), volatiles: new Map() };
    // A Baton Pass hands over the volatiles the copy table says (the stall counter among them); a Shed Tail only its own list.
    const from = line.tags.get("from");
    if (out !== null && !out.unknown && (from === "Baton Pass" || from === "Shed Tail")) {
      const copied = from === "Baton Pass" ? !this.#notCopied.has("stall") : this.#shedTail.has("stall");
      if (copied) {
        occupant.stall_count = out.stall_count;
        occupant.stall_turn = out.stall_turn;
      }
    }
    side.active.set(ref.letter, occupant);
    this.#lastLetter.set(`${ref.side}:${ref.name}`, ref.letter);
    // The Pokemon that came in has not been listed by a request yet: it cannot act before the next one.
    if (ref.side === this.#player) this.#requestMoves.delete(ref.letter);
    if (!side.roster.has(ref.name)) side.roster.set(ref.name, { ate: false, last_item: "", unknown: false });
  }

  #swap(text: string | undefined, target: string | undefined): void {
    const ref = this.#ident(text);
    const to = String.fromCharCode(97 + Number(target));
    const active = this.#sides[ref.side].active;
    if (ref.letter === null || !active.has(ref.letter) || !active.has(to) || !Number.isInteger(Number(target))) throw new HistoryFactsError("unattributable_line", this.#index);
    const one = active.get(ref.letter) as Occupant | null;
    active.set(ref.letter, active.get(to) as Occupant | null);
    active.set(to, one);
    if (one !== null) this.#lastLetter.set(`${ref.side}:${one.name}`, to);
    const moved = active.get(ref.letter);
    if (moved !== null && moved !== undefined) this.#lastLetter.set(`${ref.side}:${moved.name}`, ref.letter);
    if (ref.side === this.#player) {
      const moves = this.#requestMoves.get(ref.letter);
      const other = this.#requestMoves.get(to);
      if (other === undefined) this.#requestMoves.delete(ref.letter);
      else this.#requestMoves.set(ref.letter, other);
      if (moves === undefined) this.#requestMoves.delete(to);
      else this.#requestMoves.set(to, moves);
    }
  }

  /** An Illusion ends: from here until it leaves play the Pokemon's facts are not settled by the channel. */
  #replace(text: string | undefined): void {
    const ref = this.#ident(text);
    const occupant = ref.letter === null ? null : this.#sides[ref.side].active.get(ref.letter);
    if (occupant === undefined || occupant === null) throw new HistoryFactsError("unattributable_line", this.#index);
    const side = this.#sides[ref.side];
    const shown = side.roster.get(occupant.name);
    if (shown !== undefined) shown.unknown = true;
    occupant.unknown = true;
    occupant.name = ref.name;
    side.roster.set(ref.name, { ate: false, last_item: "", unknown: true });
  }

  #faint(text: string | undefined): void {
    const ref = this.#ident(text);
    const side = this.#sides[ref.side];
    if (ref.letter === null || !side.active.has(ref.letter)) throw new HistoryFactsError("unattributable_line", this.#index);
    side.active.set(ref.letter, null);
    side.total_fainted += 1;
  }

  #move(line: Line, justCant: Occupant | null): void {
    const { side, letter, occupant } = this.#occupant(line.args[0]);
    const move = toId(line.args[1] ?? "");
    if (move === "") throw new HistoryFactsError("malformed_line", this.#index);
    this.#lastMove = move;
    const from = line.tags.get("from");
    const called = from !== undefined && /^(move|ability|item):/.test(from);
    const target = IDENT.exec(line.args[2] ?? "");
    const slotLetter = target === null ? undefined : (target[2] ?? this.#lastLetter.get(`${target[1]}:${target[3]}`));
    let slot: Action["slot"] = null;
    const creating = this.#data.slot_conditions.find((c) => c.moves.includes(move));
    if (creating !== undefined && !called && target !== null && slotLetter !== undefined) slot = { side: target[1] as HistoryPlayer, letter: slotLetter, id: creating.id, move };
    if (this.#action !== null && called && !this.#actionSources.has(toId(from))) {
      this.#action.nested = true;
      // A move another effect made this Pokemon run (a reflected move, a called one) overwrites its own last result.
      if (this.#action.occupant !== occupant) occupant.result = "unknown";
      if (this.#stall.has(move)) this.#pending = { side, letter, move };
      return;
    }
    this.#action = { side, letter, occupant, target: target !== null && target[2] !== undefined ? { side: target[1] as HistoryPlayer, letter: target[2] } : null, hit: false, misses: 0, selfEffect: false, other: false, delayed: false, nested: false, selfTarget: target !== null && target[1] === side && target[2] === letter, slot };
    if (called && !this.#actionSources.has(toId(from))) {
      if (this.#stall.has(move)) this.#pending = { side, letter, move };
      return;
    }
    if (justCant !== occupant) occupant.move_actions += 1;
    // An action the engine runs through `runMove` deducts PP, which sets `used`, unless it is an external one (Dancer).
    if (side === this.#player && !(called && this.#externalSources.has(toId(from))) && this.#requestMoves.get(letter)?.has(move) === true) occupant.used.add(move);
    if (called) return;
    if (this.#stall.has(move)) this.#pending = { side, letter, move };
  }

  /** A `|cant|` ends the action (BeforeMove failed): it counts as a move action that failed, and no move line follows. */
  #cant(text: string | undefined, reason: string | undefined): void {
    const found = this.#at(text);
    if (found === null) return;
    found.occupant.move_actions += 1;
    found.occupant.result = this.#nullCant.has(reason ?? "") ? "ok" : "failed";
    this.#lastCant = found.occupant;
    this.#action = null;
  }

  #fail(text: string | undefined): void {
    const action = this.#action;
    if (action === null) return;
    if (text !== undefined && text.startsWith(`${action.side}${action.letter}:`)) action.slot = null;
    action.other = true;
  }

  #damage(line: Line): void {
    const target = this.#at(line.args[0]);
    if (target === null) return;
    const action = this.#action;
    const from = line.tags.get("from");
    const own = action !== null && action.occupant === target.occupant;
    // `Battle.directDamage` (a user paying a cost, Struggle's recoil) does not set `hurtThisTurn`; confusion's hit goes through `Battle.damage`.
    const direct = from === "recoil" || (from === undefined && own);
    if (from === "confusion") {
      target.occupant.move_actions += 1;
      target.occupant.result = "failed";
    }
    if (!direct) target.occupant.hurt = true;
    if (from === undefined && action !== null && !own) {
      this.#hit(action, target);
      action.hit = true;
    }
  }

  /** A hit of the action's move on `target`: `timesAttacked` and `attackedBy`. */
  #hit(action: Action, target: { side: HistoryPlayer; letter: string; occupant: Occupant }): void {
    target.occupant.times_attacked += 1;
    if (action.letter === "?") target.occupant.attackers_unsettled = true;
    else target.occupant.attackers.add(this.#seen(action.side, action.letter));
  }

  #stat(text: string | undefined, raise: boolean): void {
    const found = this.#at(text);
    if (found === null) return;
    if (raise) found.occupant.raised = true;
    else found.occupant.lowered = true;
  }

  #endItem(line: Line): void {
    const ref = this.#ident(line.args[0]);
    const side = this.#sides[ref.side];
    const item = toId(line.args[1] ?? "");
    const from = line.tags.get("from");
    const entry = side.roster.get(ref.name);
    if ([...line.tags.keys()].some((tag) => this.#eatTags.has(tag))) {
      if (entry !== undefined) {
        entry.ate = true;
        entry.last_item = item;
      }
    } else if (from !== undefined && this.#stealEat.has(from)) {
      const of = line.tags.get("of");
      const eater = of === undefined ? null : IDENT.exec(of);
      const roster = eater === null ? undefined : this.#sides[eater[1] as HistoryPlayer].roster.get(eater[3] as string);
      if (roster !== undefined) roster.ate = true;
    } else if (from === undefined || this.#gem.has(from) || this.#itemUse.has(toId(from))) {
      if (entry !== undefined && !this.#directEnd.has(item)) entry.last_item = item;
      // A berry thrown at a target is eaten by it, which the engine records without a line of the target's own.
      const target = this.#action?.target;
      if (from !== undefined && !this.#gem.has(from) && target !== null && target !== undefined && this.#berries.has(item)) {
        const eater = this.#sides[target.side].active.get(target.letter);
        const roster = eater === undefined || eater === null ? undefined : this.#sides[target.side].roster.get(eater.name);
        if (roster !== undefined) roster.ate = true;
      }
    }
  }

  /** `-item` with a `[from]` that restores the Pokemon's `lastItem` (it is cleared) or takes another's (an unsettled one). */
  #item(line: Line): void {
    const from = line.tags.get("from");
    if (from === undefined) return;
    const ref = this.#ident(line.args[0]);
    if (this.#restorers.has(toId(from))) {
      const entry = this.#sides[ref.side].roster.get(ref.name);
      if (entry !== undefined) entry.last_item = "";
    } else if (this.#takers.has(toId(from))) {
      const item = toId(line.args[1] ?? "");
      for (const side of [this.#sides.p1, this.#sides.p2]) for (const entry of side.roster.values()) if (entry.last_item === item) entry.last_item = "?";
    }
  }

  /** `-transform|USER|TARGET`: the copy takes the target's `timesAttacked` and a fresh `moveSlots`. */
  #transform(user: string | undefined, target: string | undefined): void {
    const one = this.#occupant(user).occupant;
    const two = this.#at(target)?.occupant;
    if (two === undefined) throw new HistoryFactsError("unattributable_line", this.#index);
    one.times_attacked = two.times_attacked;
    one.used.clear();
  }

  /**
   * `-end|TARGET|move: NAME` of a slot-condition move is the attack landing at the end of its delay: the lines that follow
   * until the next boundary are that move's hit, by the Pokemon that used it (`attackedBy` records the source's slot now; a
   * source that has left play has a slot the channel cannot place, so the target's attackers this turn are unsettled, but the hit still counts).
   */
  #slotLands(text: string | undefined, name: string | undefined): void {
    const target = this.#ident(text);
    if (target.letter === null || !(name ?? "").startsWith("move: ")) return;
    const move = effectId(name as string);
    const condition = this.#data.slot_conditions.find((c) => c.moves.includes(move));
    const slot = condition === undefined ? undefined : this.#sides[target.side].slots.get(target.letter)?.get(condition.id);
    if (slot === undefined) return;
    // The user's position now; a user that has left play has none the channel can place (letter "?").
    let found: { side: HistoryPlayer; letter: string } = { side: slot.source_side, letter: "?" };
    for (const side of ["p1", "p2"] as const) {
      for (const [letter, occupant] of this.#sides[side].active) if (occupant === slot.source) found = { side, letter };
    }
    this.#action = { side: found.side, letter: found.letter, occupant: slot.source, target: null, hit: false, misses: 0, selfEffect: false, other: false, delayed: true, nested: false, selfTarget: false, slot: null };
  }

  /** `-activate|TARGET|move: NAME` of a move that breaks Protect-like volatiles: the target's `stall` volatile is deleted. */
  #activate(line: Line): void {
    if (!(line.args[1] ?? "").startsWith("move: ") || !this.#breakers.has(effectId(line.args[1] as string))) return;
    const found = this.#at(line.args[0]);
    if (found !== null) {
      found.occupant.stall_count = 0;
      found.occupant.stall_turn = -1;
    }
  }

  #volatileStart(text: string | undefined, name: string | undefined): void {
    const found = this.#at(text);
    const id = effectId(name ?? "");
    if (id === "") throw new HistoryFactsError("malformed_line", this.#index);
    if (found !== null && !found.occupant.volatiles.has(id)) found.occupant.volatiles.set(id, { started_turn: this.#turn, after_residual: this.#residualRan() });
  }

  #volatileEnd(text: string | undefined, name: string | undefined): void {
    const found = this.#at(text);
    if (found !== null) found.occupant.volatiles.delete(effectId(name ?? ""));
  }
}

function newSide(): SideState {
  return { active: new Map(), total_fainted: 0, slots: new Map(), roster: new Map() };
}

function byKey<T>(x: [string, T], y: [string, T]): number {
  return x[0] < y[0] ? -1 : x[0] > y[0] ? 1 : 0;
}

/** The facts after all the lines of the channel, or the refusal. */
export function historyFacts(input: { readonly channel: OwnHistoryChannel; readonly data: HistoryFactsData }): HistoryFacts {
  const reader = new HistoryFactsReader(input.channel.player, input.data);
  for (const line of input.channel.lines) reader.feed(line);
  return reader.facts();
}

/** The facts before each `|request|` line of the channel (a refusal is counted where it stands, as `worldFactsAtRequests` does). */
export function historyFactsAtRequests(input: { readonly channel: OwnHistoryChannel; readonly data: HistoryFactsData }): { line: number; facts: HistoryFacts | null; refusal: HistoryFactsRefusal | null }[] {
  const reader = new HistoryFactsReader(input.channel.player, input.data);
  const out: { line: number; facts: HistoryFacts | null; refusal: HistoryFactsRefusal | null }[] = [];
  input.channel.lines.forEach((line, index) => {
    if (line.startsWith("|request|")) {
      try {
        out.push({ line: index, facts: reader.facts(), refusal: null });
      } catch (error) {
        out.push({ line: index, facts: null, refusal: error instanceof HistoryFactsError ? error.code : "malformed_line" });
      }
    }
    reader.feed(line);
  });
  return out;
}
