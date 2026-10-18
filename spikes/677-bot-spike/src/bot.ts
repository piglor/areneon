// SPIKE (#677): throwaway bot client for runtime A2. It plays the legal
// heuristic (an integration fixture, not AI) as a normal player on a
// LOCAL Showdown server. It must never be pointed at the public server.
//
// Player-perspective only: the bot sees its own `|request|` and the public
// room log it is sent as a player. It never sees the opponent's request.
import {
  decodeRequest,
  type JsonObject,
  loadEngine,
  type PokemonSet,
} from "../../../packages/showdown-bridge/src/index.ts";
import { forceSwitchChoice, moveChoice, previewChoice } from "./heuristic.ts";

export const FORMAT_ID = "gen9championsvgc2026regmc";
const LOCAL_HOSTS = new Set(["127.0.0.1", "localhost", "[::1]"]);

/** One decision, timed from receipt of the request to the choice being sent. */
export interface DecisionRecord {
  bot: string;
  battle: string;
  request_seq: number;
  rqid?: number;
  request_hash: string;
  phase: string;
  choice: string;
  decision_ms: number;
  /** VGC Timer state the server reported for this decision (seconds). */
  timer_turn_left_s?: number;
  timer_bank_left_s?: number;
  timer_grace_s?: number;
  fallback?: string;
}

export interface BotOptions {
  url: string;
  name: string;
  team: PokemonSet[];
  /** Challenge this user once logged in; otherwise wait for challenges. */
  challenge?: string;
  onDecision?: (record: DecisionRecord) => void;
  onLine?: (bot: string, room: string, line: string) => void;
}

export interface BattleResult {
  bot: string;
  battle: string;
  winner: string;
  turns: number;
  otsRejectedBy: string[];
  timerOn: boolean;
  decisions: DecisionRecord[];
}

const toId = (s: string): string => s.toLowerCase().replace(/[^a-z0-9]/g, "");

export function assertLocal(url: string): void {
  const host = new URL(url).hostname;
  if (!LOCAL_HOSTS.has(host)) throw new Error(`refusing non-local server ${host}`);
}

/** Run one bot until its first battle ends. */
export function runBot(options: BotOptions): Promise<BattleResult> {
  assertLocal(options.url);
  const packedTeam = loadEngine().sim.Teams.pack(options.team);
  const ws = new WebSocket(options.url);
  const send = (msg: string): void => ws.send(msg);
  const decisions: DecisionRecord[] = [];
  let requestSeq = 0;
  let battle = "";
  let turns = 0;
  let timerOn = false;
  let otsSent = false;
  const otsRejectedBy: string[] = [];
  let pending: DecisionRecord | undefined;
  let lastRequest: { rqid?: number } = {};

  const choose = (choice: string, rqid?: number): void => {
    send(`${battle}|/choose ${choice}${rqid === undefined ? "" : `|${rqid}`}`);
  };

  const onRequest = (body: string): void => {
    const received = performance.now();
    const decoded = decodeRequest(body);
    lastRequest = decoded.rqid === undefined ? {} : { rqid: decoded.rqid };
    if (decoded.phase === "wait") return;
    if (decoded.phase === "preview" && !otsSent) {
      otsSent = true;
      send(`${battle}|/rejectopenteamsheets`); // CTS: decline before choosing
    }
    const req = decoded.request as JsonObject;
    const choice =
      decoded.phase === "preview"
        ? previewChoice(decoded.preview?.maxChosenTeamSize ?? 4)
        : decoded.phase === "forceSwitch"
          ? forceSwitchChoice(req)
          : moveChoice(req);
    choose(choice, decoded.rqid);
    pending = {
      bot: options.name,
      battle,
      request_seq: ++requestSeq,
      request_hash: decoded.request_hash,
      phase: decoded.phase,
      choice,
      decision_ms: Number((performance.now() - received).toFixed(3)),
    };
    if (decoded.rqid !== undefined) pending.rqid = decoded.rqid;
    decisions.push(pending);
    options.onDecision?.(pending);
  };

  return new Promise<BattleResult>((resolve, reject) => {
    ws.addEventListener("error", () => reject(new Error(`${options.name}: websocket error`)));
    ws.addEventListener("message", (event: MessageEvent<string>) => {
      let lines = event.data.split("\n");
      let room = "";
      if (lines[0]?.startsWith(">")) {
        room = lines[0].slice(1);
        lines = lines.slice(1);
      }
      for (const line of lines) {
        options.onLine?.(options.name, room, line);
        const parts = line.split("|");
        const kind = parts[1];
        if (kind === "challstr") {
          // --no-security server: guest-style rename without an assertion.
          send(`|/trn ${options.name},0,`);
        } else if (kind === "updateuser" && toId(parts[2] ?? "") === toId(options.name)) {
          send(`|/utm ${packedTeam}`);
          if (options.challenge) send(`|/challenge ${options.challenge}, ${FORMAT_ID}`);
        } else if (kind === "pm" && (parts[4] ?? "").startsWith("/challenge ")) {
          const from = (parts[2] ?? "").slice(1);
          if (toId(from) === toId(options.name)) continue; // our own outgoing challenge
          const format = (parts[4] ?? "").slice("/challenge ".length).split("|")[0];
          // Exact format only: custom rules (`@@@`) could disable OTS or the timer.
          send(format === FORMAT_ID ? `|/accept ${from}` : `|/reject ${from}`);
        } else if (kind === "init" && parts[2] === "battle") {
          battle = room;
          send(`${battle}|/timer on`);
        } else if (room !== battle || battle === "") {
          continue;
        } else if (kind === "request" && parts.length > 2) {
          const body = parts.slice(2).join("|");
          if (body) onRequest(body);
        } else if (kind === "inactive") {
          const text = parts.slice(2).join("|");
          if (text.startsWith("Battle timer is ON")) timerOn = true;
          const m = /Time left: (\d+) sec this turn \| (\d+) sec total(?: \| (\d+) sec grace)?/.exec(text);
          if (m && pending && pending.timer_turn_left_s === undefined) {
            pending.timer_turn_left_s = Number(m[1]);
            pending.timer_bank_left_s = Number(m[2]);
            if (m[3]) pending.timer_grace_s = Number(m[3]);
          }
        } else if (kind === "error" && /\[(Invalid|Unavailable) choice\]/.test(parts[2] ?? "")) {
          if (pending) pending.fallback = parts[2] ?? "";
          choose("default", lastRequest.rqid);
        } else if (kind === "turn") {
          turns = Number(parts[2]);
        } else if (!line.startsWith("|")) {
          // Plain room text, e.g. "<name> rejected open team sheets."
          const ots = /^(.+) rejected open team sheets\.$/.exec(line);
          if (ots?.[1]) otsRejectedBy.push(ots[1]);
        } else if (kind === "win" || kind === "tie") {
          const winner = kind === "win" ? (parts[2] ?? "") : "(tie)";
          send(`${battle}|/leave`);
          ws.close();
          resolve({ bot: options.name, battle, winner, turns, otsRejectedBy, timerOn, decisions });
        }
      }
    });
  });
}
