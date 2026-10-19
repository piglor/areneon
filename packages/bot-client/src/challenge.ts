import { type Engine, resolveFormat, type RulesSnapshot, UnsupportedFormatError } from "@areneon/showdown-bridge";
import { GLOBAL_ROOM, type RoomEvent } from "./protocol.ts";

/** Showdown's `toID` (sim/dex-data.ts): lower case, letters and digits only. */
export const toId = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]+/g, "");

/**
 * The bare user name in an identity the pinned server sends: a group symbol
 * (a space for a regular user), then the name (server/users.ts:544-562,
 * `getIdentity`), then `@!` when the user is not online, in `|updateuser|`
 * (:563-567, `getIdentityWithStatus`; :975, `getUpdateuserText`). Drops the
 * first character, then a trailing `@!`. The basic name filter keeps `@` out
 * of every name (server/chat.ts:1620-1637, `namefilter`), so the suffix is
 * never part of the name.
 */
export const userName = (identity: string): string => {
  const name = identity.slice(1);
  return name.endsWith("@!") ? name.slice(0, -2) : name;
};

/** A battle challenge to the bot, as the pinned server sends it. */
export interface Challenge {
  /** The challenger's bare name (`userName` of the sender's identity). */
  from: string;
  /** The format id exactly as the server holds it, custom rules (`@@@...`) included. */
  format: string;
}

/** Why a challenge was refused. */
export type ChallengeRefusal =
  /** The sender is not the human named at launch (ADR-005 §4). */
  | "challenger"
  /** The format is not one the rules snapshot accepts at the pin (ADR-005 §2, ADR-007). */
  | "format"
  /** Not a plain battle challenge: a game challenge, an invite, or a message or buttons attached. */
  | "shape";

export type ChallengeDecision =
  | { kind: "accept"; challenge: Challenge; command: string }
  | { kind: "reject"; challenge: Challenge; reason: ChallengeRefusal; detail: string; command: string };

/** A `/challenge` message the pinned server would not send. Never accepted. */
export class ChallengeError extends Error {
  override name = "ChallengeError";
}

export interface ChallengeFilterOptions {
  engine: Engine;
  /** The rules snapshot for the pin, loaded once (`loadRulesSnapshot`). */
  snapshot: RulesSnapshot;
  /** The bot's own name. */
  botName: string;
  /** The human's name given at launch: the only challenger accepted. */
  challenger: string;
}

const CHALLENGE = "/challenge";

/**
 * Parses a challenge update addressed to `botName`. The pinned server sends
 * challenges as private messages, not as the `|updatechallenges|` JSON that
 * PROTOCOL.md still documents (the pinned server code never writes it):
 * `|pm|SENDER|RECEIVER|/challenge FORMAT|TEAMBUILDERFORMAT|MESSAGE|ACCEPTBUTTON|REJECTBUTTON`
 * (server/ladders-challenges.ts:220-238, `getUpdate`, `update`, `send`;
 * :239-262, `updateFor`, on connect). SENDER and RECEIVER are identities, a
 * group symbol then the name (server/users.ts:544-561, `getIdentity`); the
 * challenger is the sender (`update` orders the pair `from`, `to`).
 *
 * Returns `null` for anything else: another line or room, another private
 * message (`/log X wants to battle!`, server/ladders.ts:257), a bare `/challenge`
 * (the challenge was removed), or a challenge the bot sent or that is not
 * addressed to it. Throws `ChallengeError` for a `/challenge` with a body
 * of the wrong arity.
 */
export function parseChallenge(event: RoomEvent, botName: string): (Challenge & { plain: boolean }) | null {
  if (event.kind !== "line" || event.room !== GLOBAL_ROOM || !event.line.startsWith("|pm|")) return null;
  const [sender = "", receiver = "", ...message] = event.line.slice("|pm|".length).split("|");
  const text = message.join("|");
  if (!text.startsWith(`${CHALLENGE} `)) return null;
  const bot = toId(botName);
  const from = userName(sender);
  if (toId(userName(receiver)) !== bot || toId(from) === bot) return null;
  const fields = text.slice(CHALLENGE.length + 1).split("|");
  if (fields.length !== 5) throw new ChallengeError(`a /challenge body has 5 fields, not ${fields.length}: ${text}`);
  const [format, teambuilderFormat, ...extras] = fields as [string, string, string, string, string];
  // A BattleChallenge from `/challenge` has a team for the same format and no
  // message or buttons (server/ladders.ts:256, `new BattleChallenge(user.id, targetUser.id, ready)`).
  const plain = teambuilderFormat === format && extras.every((field) => field === "");
  return { from, format, plain };
}

/**
 * The challenge filter of ADR-005 §2 and §4. It accepts a challenge only
 * when all of these hold, and rejects it otherwise:
 *
 * - it is a plain battle challenge (`parseChallenge`);
 * - the sender's user id equals the challenger's given at launch;
 * - its format id, exactly as the server holds it, resolves against the
 *   rules snapshot and the pinned engine (`resolveFormat`). The server keeps
 *   custom rules in the id (sim/dex-formats.ts:694-723, `validate`: a Bo3
 *   challenge is `<format>@@@Best Of = 3`), and the snapshot names no such
 *   id, so Best-of-3 and every custom rule are refused.
 *
 * `/accept` and `/reject` take the user id (server/chat-commands/core.ts
 * `accept` :1587-1608 and `reject` :1610-1625, both through `splitUser`). Returns `null`
 * when the event is no challenge to the bot.
 */
export function decideChallenge(event: RoomEvent, options: ChallengeFilterOptions): ChallengeDecision | null {
  const parsed = parseChallenge(event, options.botName);
  if (parsed === null) return null;
  const { plain, ...challenge } = parsed;
  const id = toId(challenge.from);
  const reject = (reason: ChallengeRefusal, detail: string): ChallengeDecision => ({
    kind: "reject",
    challenge,
    reason,
    detail,
    command: `|/reject ${id}`,
  });
  if (!plain) return reject("shape", "not a plain battle challenge");
  if (id !== toId(options.challenger)) return reject("challenger", `${challenge.from} is not ${options.challenger}`);
  if (!Object.hasOwn(options.snapshot.formats, challenge.format)) {
    return reject("format", `${challenge.format} is not in the rules snapshot`);
  }
  try {
    resolveFormat(options.engine, challenge.format, options.snapshot);
  } catch (error) {
    if (error instanceof UnsupportedFormatError) return reject("format", error.message);
    throw error;
  }
  return { kind: "accept", challenge, command: `|/accept ${id}` };
}

/**
 * The `/timer on` line for a battle room the bot has just joined, or `null`.
 * The server sends `|init|battle` to each connection that joins a battle
 * room (server/rooms.ts:2011-2012, `onConnect`). The bot's own request keeps
 * the VGC Timer on even if the human turns theirs off
 * (server/room-battle.ts:215-258: `start` adds the requester, and `stop` by
 * a player removes only that player's request). A second request changes
 * little: on a rejoin it changes nothing (:217); with `forcetimer` already on
 * it only adds the bot as a requester (:222-226).
 */
export function timerOn(event: RoomEvent): string | null {
  return event.kind === "line" && event.line === "|init|battle" && event.room !== GLOBAL_ROOM
    ? `${event.room}|/timer on`
    : null;
}
