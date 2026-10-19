// The server-only line filter for the A2 PlayerChannel adapter (ADR-005 §5),
// with no imports at all so that any adapter can bundle it, like
// canonical-json.ts. A test keeps it import-free.
//
// A player's connection to a battle room on the local Showdown server carries
// the engine's battle protocol for that player plus server and social lines
// that the in-process sim (A1) never writes. `stripServerLines` keeps only the
// former, so the channel equals the A1 channel of the same battle up to
// server-layer `|error|` refusals ("nothing to choose", "too late"), which
// only A2 produces (ADR-005 §1). The golden A1 channels pass through
// unchanged (a test checks it). A2 equality for `|player|` lines is
// confirmed by the A1/A2 equality test (#764), not here: no A2 capture
// exists yet, and the rule below is read from the pinned server's source.
//
// It is an allowlist. A line is kept only if its kind is one the sim writes
// (SIM_KINDS). Known server kinds are dropped silently; any other kind is
// dropped and reported, so the session loop can log it and a sim kind this
// list lacks shows up instead of vanishing.
//
// Three kinds come from both the sim and the server, and only their position
// in the stream tells them apart:
// - `|uhtml|otsrequest|`: the sim's own Open Team Sheets prompt, written at
//   Team Preview (data/rulesets.ts `openteamsheets`). Kept before `|start`.
// - `|uhtmlchange|otsrequest|`: the sim writes it once at `|start`; the
//   server also sends it during Team Preview to a player who accepts or
//   declines the sheets (server/chat-commands/core.ts). Kept after `|start`
//   only, so a bot that declines still holds the sim's single copy.
// - `|player|`: see `stripServerLines`.
// Every other `|uhtml|` / `|uhtmlchange|` box comes from the server.

/**
 * Line kinds the sim writes into a player's stream: SIM-PROTOCOL.md at the
 * pin, the kinds the pinned sim's source writes that the document omits, and
 * every kind in the golden channels. `""` is the bare `|` spacer.
 */
const SIM_KINDS: ReadonlySet<string> = new Set([
  "",
  // Battle initialisation and progress.
  "gametype",
  "player",
  "teamsize",
  "gen",
  "tier",
  "rated",
  "rule",
  "clearpoke",
  "poke",
  "teampreview",
  "updatepoke",
  "showteam",
  "start",
  "request",
  "error",
  "turn",
  "upkeep",
  "win",
  "tie",
  // Major actions.
  "move",
  "switch",
  "drag",
  "detailschange",
  "replace",
  "swap",
  "cant",
  "faint",
  // Minor actions.
  "-fail",
  "-block",
  "-notarget",
  "-miss",
  "-damage",
  "-heal",
  "-sethp",
  "-status",
  "-curestatus",
  "-cureteam",
  "-boost",
  "-unboost",
  "-setboost",
  "-swapboost",
  "-invertboost",
  "-clearboost",
  "-clearallboost",
  "-clearpositiveboost",
  "-clearnegativeboost",
  "-copyboost",
  "-weather",
  "-fieldstart",
  "-fieldend",
  "-fieldactivate",
  "-sidestart",
  "-sideend",
  "-swapsideconditions",
  "-start",
  "-end",
  "-crit",
  "-supereffective",
  "-resisted",
  "-immune",
  "-item",
  "-enditem",
  "-ability",
  "-endability",
  "-transform",
  "-mega",
  "-primal",
  "-burst",
  "-zpower",
  "-zbroken",
  "-terastallize",
  "-candynamax",
  "-activate",
  "-hint",
  "-center",
  "-message",
  "-combine",
  "-waiting",
  "-prepare",
  "-mustrecharge",
  "-nothing",
  "-hitcount",
  "-singlemove",
  "-singleturn",
  "-formechange",
  "-ohko",
  "-anim",
  // Sim messages outside SIM-PROTOCOL.md's lists (sim/battle.ts).
  "message",
  "bigerror",
  "debug",
]);

/**
 * Line kinds the server sends, plus the sim kinds no supported format writes
 * (`|raw|`, `|html|`, `|chat|`, `|j|`, `|l|`) and `|t:|`, which A1 drops
 * too. Dropped without a report.
 */
const SERVER_KINDS: ReadonlySet<string> = new Set([
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
  // Wall-clock stamps.
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
  "tournament",
  "battle",
  // The server re-sends a choice already made with the current request
  // (ADR-005 §1). The client reads it from the raw frame; the sim never
  // writes it, so it is not part of the channel.
  "sentchoice",
]);

const OTS_PROMPT = "|uhtml|otsrequest|";
const OTS_CLEAR = "|uhtmlchange|otsrequest|";

function kindOf(line: string): string {
  const end = line.indexOf("|", 1);
  return end < 0 ? line.slice(1) : line.slice(1, end);
}

/**
 * True if `line` is a server-only line wherever it appears in the stream:
 * - a `>room` header;
 * - a plain-text notice (a non-empty line not starting with `|`), or `||`
 *   followed by text (the bare `|` spacer is the sim's);
 * - `|request|null`, which the server sends when a player leaves; it is not
 *   a request and never reaches `decodeRequest` (ADR-005 §1);
 * - `|player|<slot>|` with no name, which the server sends when a player
 *   leaves (the sim always names a side);
 * - a line of a known server kind, except the sim's Open Team Sheets lines,
 *   whose place in the stream decides (`stripServerLines`).
 */
export function serverOnlyLine(line: string): boolean {
  if (line === "") return false;
  if (!line.startsWith("|") || (line.startsWith("||") && line.length > 2)) return true;
  if (line === "|request|null") return true;
  if (/^\|player\|[^|]*\|$/.test(line)) return true;
  if (line.startsWith(OTS_PROMPT) || line.startsWith(OTS_CLEAR)) return false;
  return SERVER_KINDS.has(kindOf(line));
}

/** The lines of a battle room stream that belong in a PlayerChannel. */
export interface StrippedLines {
  /** The kept lines, in order. */
  lines: string[];
  /** Dropped lines of a kind neither the sim list nor the server list knows. */
  dropped: string[];
}

/**
 * Filter one player's battle room stream, from its first line, down to the
 * lines that belong in a PlayerChannel. The result carries the dropped
 * lines of unknown kind rather than calling back, so the filter stays pure
 * and the caller decides how to log them. The rules that depend on position
 * need the whole stream, so the adapter filters everything received so far
 * each time it builds a channel.
 *
 * `|player|`: the server's RoomBattle writes `>player <slot>` to the sim and
 * at once adds its own `|player|<slot>|<name>|<avatar>|` to the room log
 * (server/room-battle.ts `addPlayer`), and again when a player rejoins
 * (`onJoin`). The sim writes its line from `setPlayer`, after the
 * `|gametype|` its constructor writes first (sim/battle.ts). So the sim's
 * line for a slot is the first `|player|` line for that slot after
 * `|gametype|`: lines before `|gametype|` are the server's, later lines for
 * the same slot are rejoins, and lines after `|start` are dropped as well.
 */
export function stripServerLines(stream: readonly string[]): StrippedLines {
  const lines: string[] = [];
  const dropped: string[] = [];
  const named = new Set<string>();
  let gametype = false;
  let started = false;
  for (const line of stream) {
    if (serverOnlyLine(line)) continue;
    const kind = kindOf(line);
    let keep: boolean;
    if (line.startsWith(OTS_PROMPT)) keep = !started;
    else if (line.startsWith(OTS_CLEAR)) keep = started;
    else if (kind === "player") {
      const slot = line.split("|")[2] as string;
      keep = gametype && !started && !named.has(slot);
      if (keep) named.add(slot);
    } else if (SIM_KINDS.has(kind)) {
      keep = true;
      if (kind === "gametype") gametype = true;
      if (kind === "start") started = true;
    } else {
      keep = false;
      dropped.push(line);
    }
    if (keep) lines.push(line);
  }
  return { lines, dropped };
}
