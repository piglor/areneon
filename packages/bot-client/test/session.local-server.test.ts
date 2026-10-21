import {
  continueChannel,
  loadEngine,
  loadRulesSnapshot,
  type PokemonSet,
  resolveFormat,
  stampTeam,
  type StampedTeam,
  teamHash,
} from "@areneon/showdown-bridge";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import {
  type A2BattleRecord,
  type A2BotOptions,
  a2Episode,
  type A2SessionResult,
  connectWebSocket,
  runA2Bot,
  type SessionLog,
  timerRulesOf,
  type Transport,
} from "../src/index.ts";
import { fixtureTeam, FORMAT, startLocalServer } from "./local-battle.ts";

// Bot vs bot on the pinned server, launched on 127.0.0.1 (ADR-005 §4): one
// runA2Bot challenges (the harness-only `challenge` option), the other is the
// sparring bot with that challenger named at launch. Both play the shared
// actor rule; the engine judges every choice at submission.

const engine = loadEngine();
const snapshot = loadRulesSnapshot(engine.commit);
const own = fixtureTeam("legal-synthetic.txt");
const opponent = fixtureTeam("legal-opponent.txt");

let local: Awaited<ReturnType<typeof startLocalServer>>;
beforeAll(async () => {
  local = await startLocalServer();
}, 60_000);
afterAll(async () => {
  await local?.stop();
});

interface Bot {
  sent: string[];
  logs: SessionLog[];
  result: Promise<A2SessionResult>;
  loggedIn: Promise<void>;
}

/** A bot whose every connection goes through the real WebSocket transport, its sent lines recorded. */
function bot(
  options: Pick<A2BotOptions, "name" | "challenger" | "team"> & Partial<A2BotOptions>,
  wrap: (transport: Transport, index: number) => Transport = (t) => t,
): Bot {
  const sent: string[] = [];
  const logs: SessionLog[] = [];
  let onLogin: () => void = () => {};
  const loggedIn = new Promise<void>((done) => (onLogin = done));
  let connections = 0;
  const result = runA2Bot({
    engine,
    snapshot,
    url: local.ws,
    format: FORMAT,
    data_class: "synthetic",
    battles: 1,
    reconnect: { attempts: 5, delayMs: 100 },
    onLog(entry) {
      logs.push(entry);
      if (entry.event === "logged_in") onLogin();
    },
    async connect(url) {
      const transport = await connectWebSocket(url);
      const recorded: Transport = {
        ...transport,
        send(line) {
          transport.send(line);
          sent.push(line);
        },
      };
      return wrap(recorded, connections++);
    },
    ...options,
  });
  return { sent, logs, result, loggedIn };
}

/** Every `/choose` line a bot sent was a candidate of the request its rqid names. */
function expectCandidateChoices(sent: string[], record: A2BattleRecord): number {
  const chosen = sent.filter((line) => line.startsWith(`${record.room}|/choose `));
  for (const line of chosen) {
    const [, body = "", rqid = ""] = line.split("|");
    const choice = body.slice("/choose ".length);
    const decision = record.decisions.find((d) => d.rqid === Number(rqid));
    expect(decision, line).toBeDefined();
    expect([...(decision?.legal.choices ?? []), ...(decision?.legal.pending ?? [])], line).toContain(choice);
    expect(decision?.legal.rqid).toBe(Number(rqid));
  }
  return chosen.length;
}

/** The opponent's preview in each channel: species and details only (ADR-003 CTS, rules/gaps/cts-invariant.json). */
function expectClosedPreview(record: A2BattleRecord, team: PokemonSet[]): void {
  const foe = record.player === "p1" ? "p2" : "p1";
  for (const channel of record.channels) {
    expect(channel.lines.filter((line) => line.startsWith("|showteam|"))).toEqual([]);
    // Team Preview: everything before the battle starts, in a rejoin's replayed log too.
    const preview = channel.lines.slice(0, channel.lines.indexOf("|start"));
    expect(preview.length).toBeGreaterThan(0);
    const pokes = preview.filter((line) => line.startsWith(`|poke|${foe}|`));
    // The champions mod keeps the shiny flag (cts-invariant); nothing else of the set.
    for (const line of pokes) expect(line).toMatch(new RegExp(`^\\|poke\\|${foe}\\|[^,|]+, L50(, [MF])?(, shiny)?\\|$`));
    expect(pokes.map((line) => line.split("|")[3]?.split(",")[0])).toEqual(team.map((set) => set.species));
    const view = preview.join("\n");
    const sheet = team.flatMap((set) => [set.name === set.species ? "" : set.name, set.item, set.ability, ...set.moves]);
    for (const entry of sheet.filter(Boolean)) expect(view, entry).not.toContain(`|${entry}|`);
  }
}

function expectOwnNumbering(record: A2BattleRecord): void {
  expect(record.decisions.length).toBeGreaterThan(0);
  for (const decision of record.decisions) expect(decision.channel_request_seq).toBe(decision.request_seq);
}

/** After a reconnect the decisions are numbered on the joined channel: strictly rising, though the connection restarts its own count. */
function expectJoinedNumbering(record: A2BattleRecord): void {
  const numbers = record.decisions.map((decision) => decision.channel_request_seq);
  expect(numbers.length).toBeGreaterThan(0);
  expect(numbers).toEqual([...numbers].sort((a, b) => a - b));
  expect(new Set(numbers).size).toBe(numbers.length);
  const joined = record.channels.reduce(continueChannel);
  const requests = joined.lines.filter((line) => line.startsWith("|request|")).length;
  expect(Math.max(...numbers)).toBeLessThanOrEqual(requests);
}

describe("runA2Bot on the pinned local server", () => {
  it(
    "plays one Bo1 to the end with open team sheets declined, choosing only candidates",
    async () => {
      const suffix = local.server.session.bot_name_suffix;
      const sparring = bot({ name: `Areneon-bot-${suffix}`, challenger: `Areneon-ch-${suffix}`, team: own });
      await sparring.loggedIn;
      const challenger = bot({
        name: `Areneon-ch-${suffix}`,
        challenger: `Areneon-bot-${suffix}`,
        challenge: `Areneon-bot-${suffix}`,
        team: opponent,
      });
      const [a, b] = await Promise.all([sparring.result, challenger.result]);
      expect(sparring.logs).toContainEqual({ event: "challenge_accepted", from: `Areneon-ch-${suffix}`, format: FORMAT });

      const [mine] = a.battles as [A2BattleRecord];
      const [theirs] = b.battles as [A2BattleRecord];
      expect(a.battles).toHaveLength(1);
      expect(b.battles).toHaveLength(1);
      expect(mine.room).toBe(theirs.room);
      expect([mine.result, theirs.result].sort()).toEqual(
        mine.result === "tie" ? ["tie", "tie"] : ["loss", "win"],
      );
      for (const [run, record, foeTeam] of [
        [sparring, mine, opponent],
        [challenger, theirs, own],
      ] as const) {
        expect(record.ots).toBe("confirmed");
        expect(record.ots_failure).toBeNull();
        expect(record.defect).toBe(false);
        expect(run.sent).toContain(`${record.room}|/rejectopenteamsheets`);
        expect(run.sent).toContain(`${record.room}|/timer on`);
        expect(run.sent.some((line) => line.includes("acceptopenteamsheets"))).toBe(false);
        expect(expectCandidateChoices(run.sent, record)).toBe(record.decisions.length);
        expectOwnNumbering(record);
        expectClosedPreview(record, foeTeam);
        // Every new request's budget came from the server's own timer snapshot.
        for (const decision of record.decisions.filter((d) => !d.inherited)) {
          expect(decision.timer?.timer_source).toBe("server");
          expect(decision.budget.submit_within_ms).toBeGreaterThan(0);
        }
        expect(record.decisions[0]?.phase).toBe("preview");
        expect(record.channels).toHaveLength(1);
        // Its own channel only: no SECRET line of the opponent, and its own requests alone.
        for (const line of record.channels[0]?.lines ?? []) {
          if (line.startsWith("|request|")) expect(line).toContain(`"id":"${record.player}"`);
        }
      }
    },
    60_000,
  );

  it(
    "reconnects mid-battle, keeps one decline per room and plays on to the end",
    async () => {
      const suffix = local.server.session.bot_name_suffix;
      let dropped = false;
      // Drop the sparring bot's first connection right after its second move choice.
      const sparring = bot(
        { name: `Areneon-b2-${suffix}`, challenger: `Areneon-c2-${suffix}`, team: own },
        (transport, index) =>
          index > 0
            ? transport
            : {
                ...transport,
                send(line) {
                  transport.send(line);
                  if (!dropped && line.includes("|/choose move") && sparring.sent.filter((l) => l.includes("|/choose move")).length === 2) {
                    dropped = true;
                    setTimeout(() => transport.close(), 0);
                  }
                },
              },
      );
      await sparring.loggedIn;
      const challenger = bot({
        name: `Areneon-c2-${suffix}`,
        challenger: `Areneon-b2-${suffix}`,
        challenge: `Areneon-b2-${suffix}`,
        team: opponent,
      });
      const [a] = await Promise.all([sparring.result, challenger.result]);
      const [record] = a.battles as [A2BattleRecord];
      expect(dropped).toBe(true);
      expect(sparring.logs.filter((entry) => entry.event === "logged_in")).toHaveLength(2);
      expect(record.result).not.toBe("unfinished");
      expect(record.ots).toBe("confirmed");
      expect(record.channels).toHaveLength(2);
      expect(sparring.sent.filter((line) => line.endsWith("|/rejectopenteamsheets"))).toHaveLength(1);
      expect(expectCandidateChoices(sparring.sent, record)).toBeGreaterThanOrEqual(record.decisions.length);
      // Each connection numbers its own requests from 1; a decision is numbered on the joined channel.
      expectJoinedNumbering(record);
      // The record of a reconnected battle is one Episode on the one joined channel (ADR-008 §5),
      // unless a request first decided on the rejoin lost its timer snapshot to the old connection:
      // which of the two happens depends on where the drop fell, and each is pinned by the fixtures.
      const known = new Set([teamHash(own), teamHash(opponent)]);
      const written = a2Episode(record, {
        team: stampTeam(own as StampedTeam["team"], known),
        known_synthetic: known,
        manifest: resolveFormat(engine, FORMAT, snapshot).manifest,
        timer_rules: timerRulesOf(engine, FORMAT, 5),
        behaviour_policy_id: "actor-rule@a2",
        collection: { kind: "sparring", session_id: "reconnect" },
      });
      if (record.decisions.some((decision) => decision.timer === null)) {
        await expect(written).rejects.toThrow(/no timer snapshot/);
      } else {
        const episode = await written;
        expect(episode.channel).toEqual(record.channels.reduce(continueChannel).lines);
        expect(episode.decisions.map((d) => d.identity.request_seq)).toEqual(record.decisions.map((d) => d.channel_request_seq));
      }
      expectClosedPreview(record, opponent);
    },
    60_000,
  );
});
