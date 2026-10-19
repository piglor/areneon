// The server-only line filter for the A2 PlayerChannel adapter (ADR-005 §5),
// with no imports at all so that any adapter can bundle it, like
// canonical-json.ts. A test keeps it import-free.
//
// A player's connection to a battle room on the local Showdown server carries
// the engine's battle protocol for that player plus server and social lines
// that the in-process sim (A1) never writes. Dropping the latter leaves a
// channel that equals the A1 channel of the same battle, so an observation is
// built from the same lines on A1 and A2. A line the A1 channel holds is never
// dropped: the golden channels pass through unchanged (a test checks it).
//
// Kept on purpose:
// - `|error|`: the engine's choice refusals (`[Unavailable choice]`,
//   `[Invalid choice]`). The A1 runner records them in the channel too, and
//   the client classifies them (ADR-005 §1, ADR-006).
// - `|tier|`, `|rule|`, `|gametype|`, `|player|` and every other line the sim
//   writes, including `|message|`, `|-message|`, `|bigerror|` and the empty
//   `|` spacer.
// - `|uhtml|otsrequest|` and `|uhtmlchange|otsrequest|`: the Open Team Sheets
//   prompt that the engine's own `openteamsheets` rule writes into each
//   player's half of the log (data/rulesets.ts at the pin). It is in the A1
//   channel. Every other `|uhtml|` box comes from the server.
// - The empty line, which the sim stream also yields.

/** Line kinds only the server, never the sim, sends to a player. */
const SERVER_ONLY_KINDS: ReadonlySet<string> = new Set([
  // Joins, leaves and renames (lower case: unregistered or quiet).
  "j",
  "J",
  "join",
  "l",
  "L",
  "leave",
  "n",
  "N",
  "name",
  // Chat: untrusted free text that never reaches an observation (ADR-005 §5).
  "c",
  "c:",
  "chat",
  "pm",
  // Server HTML and notices.
  "raw",
  "html",
  "uhtml",
  "uhtmlchange",
  "notify",
  "tempnotify",
  "tempnotifyoff",
  "popup",
  "hidelines",
  "unlink",
  // Battle timer. Timer state feeds only `decision_context` (ADR-005 §5).
  "inactive",
  "inactiveoff",
  // Wall-clock stamps, which A1 drops as well.
  "t:",
  ":",
  "timestamp",
  // Room and session state.
  "init",
  "title",
  "deinit",
  "noinit",
  "expire",
  "usercount",
  "users",
  "customgroups",
  "challstr",
  "updateuser",
  "updatesearch",
  "updatechallenges",
  "nametaken",
  "formats",
  "queryresponse",
  // The server re-sends a choice already made with the current request
  // (ADR-005 §1). The client reads it from the raw frame; the sim never
  // writes it, so it is not part of the channel.
  "sentchoice",
]);

/** The engine's own `|uhtml|` box: the Open Team Sheets prompt. */
const ENGINE_UHTML: readonly string[] = ["|uhtml|otsrequest|", "|uhtmlchange|otsrequest|"];

/**
 * True if `line` from a battle room connection is a server-only or social
 * line that an A1 PlayerChannel never holds:
 * - a `>room` header;
 * - a plain-text notice (a non-empty line not starting with `|`);
 * - `|request|null`, which the server sends when a player leaves; it is not
 *   a request and never reaches `decodeRequest` (ADR-005 §1);
 * - a line whose kind is in the server-only list above, except the engine's
 *   own `|uhtml|` boxes.
 */
export function serverOnlyLine(line: string): boolean {
  if (line === "") return false;
  if (!line.startsWith("|")) return true;
  if (line === "|request|null") return true;
  const end = line.indexOf("|", 1);
  const kind = end < 0 ? line.slice(1) : line.slice(1, end);
  if (!SERVER_ONLY_KINDS.has(kind)) return false;
  return !ENGINE_UHTML.some((prefix) => line.startsWith(prefix));
}

/** The lines of a battle room stream that belong in a PlayerChannel, in order. */
export function stripServerLines(lines: readonly string[]): string[] {
  return lines.filter((line) => !serverOnlyLine(line));
}
