import { loadEngine, loadRulesSnapshot, type RulesSnapshot } from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import {
  type ChallengeDecision,
  ChallengeError,
  type ChallengeFilterOptions,
  decideChallenge,
  memoryTransport,
  parseChallenge,
  ProtocolReader,
  timerOn,
  toId,
} from "../src/index.ts";
import { battle, serverFrame } from "./frames.ts";

// Challenge frames, written from the pinned server source: a challenge
// update is a global private message (no `>room` header),
// `|pm|SENDER|RECEIVER|/challenge FORMAT|TEAMBUILDERFORMAT|MESSAGE|ACCEPTBUTTON|REJECTBUTTON`
// (server/ladders-challenges.ts:220-238), with identities `<group><name>`
// (server/users.ts:544-561; a regular user's group is a space). On connect
// the pending challenges come as one frame of such lines (:240-260).

const engine = loadEngine();
const snapshot = loadRulesSnapshot(engine.commit);
const FORMAT = "gen9championsvgc2026regmc";
const BOT = "Areneon-bot-0000";
const HUMAN = "Sparring Human";
const OPTIONS: ChallengeFilterOptions = { engine, snapshot, botName: BOT, challenger: HUMAN };

/** What `Dex.formats.validate` makes of a challenge's format string (sim/dex-formats.ts:694-723). */
const validate = (name: string): string =>
  (engine.sim.Dex.formats as unknown as { validate(name: string): string }).validate(name);

const challengeLine = (from: string, format: string, to = BOT, teambuilder = format, extras = "||"): string =>
  `|pm| ${from}| ${to}|/challenge ${format}|${teambuilder}|${extras}`;

const decide = (frame: string, options: ChallengeFilterOptions = OPTIONS): (ChallengeDecision | null)[] =>
  new ProtocolReader().read(frame).map((event) => decideChallenge(event, options));

describe("challenge filter (ADR-005 §2 and §4)", () => {
  it("accepts the exact rules-snapshot format from the launch challenger", () => {
    expect(validate(FORMAT)).toBe(FORMAT);
    expect(decide(challengeLine(HUMAN, FORMAT))).toEqual([
      { kind: "accept", challenge: { from: HUMAN, format: FORMAT }, command: "|/accept sparringhuman" },
    ]);
  });

  it("matches the challenger by user id, as the server does", () => {
    const [decision] = decide(challengeLine("sparring-HUMAN", FORMAT));
    expect(decision).toMatchObject({ kind: "accept", command: "|/accept sparringhuman" });
  });

  it("refuses Best-of-3, which the server holds as a custom-rule format id", () => {
    const bo3 = validate(`${FORMAT}@@@Best Of = 3`);
    expect(bo3).toBe(`${FORMAT}@@@Best Of = 3`);
    const [decision] = decide(challengeLine(HUMAN, bo3));
    expect(decision).toMatchObject({ kind: "reject", reason: "format", command: "|/reject sparringhuman" });
  });

  it("refuses any other custom rule, an unknown format and a prototype key", () => {
    for (const format of [`${FORMAT}@@@-Open Team Sheets`, "gen9ou", "gen9championsvgc2026regmcbo3", "__proto__"]) {
      const [decision] = decide(challengeLine(HUMAN, format));
      expect(decision).toMatchObject({ kind: "reject", reason: "format" });
    }
  });

  it("refuses a snapshot format the pinned engine does not resolve", () => {
    const stale: RulesSnapshot = { ...snapshot, showdown_commit: "0".repeat(40) };
    const [decision] = decide(challengeLine(HUMAN, FORMAT), { ...OPTIONS, snapshot: stale });
    expect(decision).toMatchObject({ kind: "reject", reason: "format", detail: expect.stringMatching(/built for/) });
  });

  it("rethrows a resolver failure that is not a refusal", () => {
    const spec = snapshot.formats[FORMAT];
    const broken = { ...snapshot, formats: { [FORMAT]: { ...spec, forbidden_rules: null } } } as unknown as RulesSnapshot;
    expect(() => decide(challengeLine(HUMAN, FORMAT), { ...OPTIONS, snapshot: broken })).toThrow(TypeError);
  });

  it("refuses a challenge from anyone but the launch challenger", () => {
    const [decision] = decide(challengeLine("Stranger", FORMAT));
    expect(decision).toMatchObject({ kind: "reject", reason: "challenger", command: "|/reject stranger" });
  });

  it("refuses what is not a plain battle challenge", () => {
    const shapes = [
      challengeLine(HUMAN, FORMAT, BOT, ""), // a game challenge has no team format
      challengeLine(HUMAN, FORMAT, BOT, FORMAT, "hi||"),
      challengeLine(HUMAN, FORMAT, BOT, FORMAT, "|Join|"),
    ];
    for (const line of shapes) expect(decide(line)[0]).toMatchObject({ kind: "reject", reason: "shape" });
  });

  it("throws on a /challenge body the server would not send", () => {
    expect(() => decide(`|pm| ${HUMAN}| ${BOT}|/challenge ${FORMAT}`)).toThrow(ChallengeError);
    expect(() => decide(challengeLine(HUMAN, FORMAT, BOT, FORMAT, "|||"))).toThrow(ChallengeError);
  });

  it("ignores what is no challenge to the bot", () => {
    const frame = [
      `|pm| ${HUMAN}| ${BOT}|/challenge`, // removed (ladders-challenges.ts:221)
      `|pm| ${HUMAN}| ${BOT}|/log ${HUMAN} wants to battle!`, // server/ladders.ts:257
      `|pm| ${HUMAN}| ${BOT}|hello`,
      challengeLine(BOT, FORMAT, HUMAN), // the bot's own challenge
      challengeLine(HUMAN, FORMAT, "Someone Else"),
      '|updatechallenges|{"challengesFrom":{"sparringhuman":"gen9championsvgc2026regmc"},"challengeTo":null}',
      serverFrame("global_updateuser"),
    ].join("\n");
    expect(decide(frame)).toEqual(Array(7).fill(null));
    // A private-message line inside a room is no challenge either.
    expect(decide(`>${battle.room}\n${challengeLine(HUMAN, FORMAT)}`)).toEqual([null]);
    expect(parseChallenge({ kind: "error", room: "", error: "x" }, BOT)).toBeNull();
  });

  it("decides each challenge of a reconnect frame, and the transport carries the commands", () => {
    const transport = memoryTransport();
    const reader = new ProtocolReader();
    transport.onFrame((frame) => {
      for (const event of reader.read(frame)) {
        const decision = decideChallenge(event, OPTIONS);
        if (decision !== null) transport.send(decision.command);
      }
    });
    transport.push(`${challengeLine("Stranger", FORMAT)}\n${challengeLine(HUMAN, FORMAT)}\n`);
    expect(transport.sent).toEqual(["|/reject stranger", "|/accept sparringhuman"]);
  });

  it("uses Showdown's toID", () => {
    expect(toId(" Sparring-Human 1!")).toBe("sparringhuman1");
  });
});

describe("timer on at battle start (ADR-005 decision driver 1, §4)", () => {
  it("asks for the timer when the bot joins its battle room", () => {
    // server/rooms.ts:2011-2012: `|init|battle` then `|title|` on connect.
    const frame = `>${battle.room}\n|init|battle\n|title|${HUMAN} vs. ${BOT}\n|j|☆${BOT}`;
    const transport = memoryTransport();
    const reader = new ProtocolReader();
    transport.onFrame((f) => {
      for (const event of reader.read(f)) {
        const line = timerOn(event);
        if (line !== null) transport.send(line);
      }
    });
    transport.push(frame);
    expect(transport.sent).toEqual([`${battle.room}|/timer on`]);
  });

  it("asks for nothing on other lines, or outside a room", () => {
    const events = new ProtocolReader().read("|init|battle");
    expect(events.map(timerOn)).toEqual([null]);
    expect(new ProtocolReader().read(`>lobby\n|init|chat`).map(timerOn)).toEqual([null]);
    expect(timerOn({ kind: "error", room: battle.room, error: "|init|battle" })).toBeNull();
  });
});
