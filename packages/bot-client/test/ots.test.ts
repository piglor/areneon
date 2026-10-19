import { describe, expect, it } from "vitest";
import {
  type Budget,
  type MemoryTransport,
  memoryTransport,
  OtsDecline,
  OtsDeclineError,
  ProtocolReader,
  type RoomEvent,
  TransportClosedError,
} from "../src/index.ts";
import { battle, frameIndex } from "./frames.ts";

// Frames the pinned server sends around a rejection, written from its source:
// - `/rejectopenteamsheets` (server/chat-commands/core.ts:980-1002) first
//   sends `|uhtmlchange|otsrequest|` to every player through `sendRoom`
//   (server/room-game.ts:91-93, users.ts:526-528 adds the `>room` header),
//   then adds `<name> rejected open team sheets.` to the room, which
//   `update` broadcasts as one `>room` frame (server/rooms.ts:346-347,
//   :379-381, :434-441; chat.ts `update`, :1027-1029);
// - a refused command comes back as `|error|<text>` to the sender's
//   connection in that room (chat.ts:484-485 and :870-876, `errorReply`);
// - `|showteam|<side>|<packed team>` is written by the sim only
//   (sim/battle.ts:3220, `showOpenTeamSheets`, run on `>show-openteamsheets`,
//   sim/battle-stream.ts:209-210, which the server writes once every player
//   accepted, core.ts:973-975).

const ROOM = battle.room;
const BOT = "Areneon-bot-0000";
const HUMAN = "Sparring Human";
const BUDGET: Budget = { submit_within_ms: 60_000, policy_budget_ms: 57_000 };
const REJECT = `${ROOM}|/rejectopenteamsheets`;
const FORFEIT = `${ROOM}|/forfeit`;
const ACCEPT = "/acceptopenteamsheets";
const PREVIEW_CHOICE = `${ROOM}|/choose team 4561|1`;

const uhtmlchange = `>${ROOM}\n|uhtmlchange|otsrequest|`;
const rejected = (name: string): string => `>${ROOM}\n${name} rejected open team sheets.`;
const showteam = `>${ROOM}\n|showteam|p2|Gengar||focussash|cursedbody|shadowball,sludgebomb,protect,trick||||||50|`;
const previewIndex = frameIndex('|request|{"teamPreview":true');

/** A memory transport, a reader and a decline wired as the session loop (#757) will wire them. */
function session(options: { serverConfirms?: boolean; now?: () => number } = {}) {
  const { serverConfirms = true, now = () => 0 } = options;
  const reader = new ProtocolReader();
  const transport: MemoryTransport = memoryTransport({
    onSend: (line, peer) => {
      // The scripted server answers the rejection as core.ts:999-1002 does.
      if (line === REJECT && serverConfirms) {
        peer.push(uhtmlchange);
        peer.push(rejected(BOT));
      }
    },
  });
  const ots = new OtsDecline({ room: ROOM, botName: BOT, send: (line) => transport.send(line) });
  const errors: unknown[] = [];
  transport.onFrame((frame) => {
    for (const event of reader.read(frame)) {
      try {
        ots.onEvent(event, now(), event.kind === "request" && event.request.phase === "preview" ? BUDGET : undefined);
      } catch (error) {
        errors.push(error);
      }
    }
  });
  const toPreview = (): void => {
    for (const frame of battle.frames.slice(0, previewIndex + 1)) transport.push(frame);
  };
  return { transport, ots, errors, toPreview };
}

describe("OTS decline (ADR-005 §2)", () => {
  it("sends /rejectopenteamsheets on the preview request, before the preview choice", () => {
    const { transport, ots, toPreview } = session();
    toPreview();
    expect(ots.status).toBe("confirmed");
    ots.submitPreview(PREVIEW_CHOICE);
    expect(transport.sent).toEqual([REJECT, PREVIEW_CHOICE]);
  });

  it("holds a preview choice submitted before the confirmation, and sends it on the bot's own line only", () => {
    const { transport, ots, toPreview } = session({ serverConfirms: false });
    toPreview();
    ots.submitPreview(PREVIEW_CHOICE);
    transport.push(uhtmlchange);
    transport.push(rejected(HUMAN));
    expect(ots.status).toBe("awaiting");
    expect(transport.sent).toEqual([REJECT]);
    transport.push(rejected(BOT));
    expect(ots.status).toBe("confirmed");
    expect(transport.sent).toEqual([REJECT, PREVIEW_CHOICE]);
  });

  it("never sends /acceptopenteamsheets, and sends the rejection once, over a whole recorded battle", () => {
    const { transport, ots, errors } = session();
    for (const frame of battle.frames) transport.push(frame);
    // A rejoin re-sends the preview request: still no second rejection.
    transport.push(battle.frames[previewIndex] as string);
    expect(errors).toEqual([]);
    expect(ots.status).toBe("confirmed");
    expect(transport.sent).toEqual([REJECT]);
    expect(transport.sent.some((line) => line.includes(ACCEPT))).toBe(false);
  });

  it("ignores other rooms and everything before the preview request", () => {
    const { transport, ots, errors } = session();
    transport.push(`>battle-gen9championsvgc2026regmc-2\n|showteam|p2|x`);
    transport.push(`>${ROOM}\n|error|[Invalid choice] There's nothing to choose`);
    transport.push(rejected(BOT));
    expect(errors).toEqual([]);
    expect(ots.status).toBe("idle");
    expect(transport.sent).toEqual([]);
  });

  it("refuses a preview choice before the preview request, and a preview request without its budget", () => {
    const sent: string[] = [];
    const ots = new OtsDecline({ room: ROOM, botName: BOT, send: (line) => sent.push(line) });
    expect(() => ots.submitPreview(PREVIEW_CHOICE)).toThrow(/no Team Preview request/);
    const [preview] = new ProtocolReader().read(battle.frames[previewIndex] as string);
    expect(preview?.kind).toBe("request");
    expect(() => ots.onEvent(preview as RoomEvent, 0)).toThrow(TypeError);
    expect(sent).toEqual([]);
    expect(ots.status).toBe("idle");
  });
});

describe("fail-closed forfeit (ADR-005 §2)", () => {
  it("forfeits when the rejection line does not arrive within the preview policy budget", () => {
    let clock = 1_000;
    const { transport, ots, toPreview } = session({ serverConfirms: false, now: () => clock });
    toPreview();
    ots.submitPreview(PREVIEW_CHOICE);
    ots.tick(clock + BUDGET.policy_budget_ms - 1);
    expect(ots.status).toBe("awaiting");
    clock += BUDGET.policy_budget_ms;
    let thrown: unknown;
    try {
      ots.tick(clock);
    } catch (error) {
      thrown = error;
    }
    expect(thrown).toBeInstanceOf(OtsDeclineError);
    expect(thrown).toMatchObject({ reason: "not_confirmed", forfeitSent: true });
    expect(ots.status).toBe("forfeited");
    expect(transport.sent).toEqual([REJECT, FORFEIT]);
    // A late confirmation releases nothing, and the choice can no longer go out.
    transport.push(rejected(BOT));
    ots.tick(clock + 1);
    expect(() => ots.submitPreview(PREVIEW_CHOICE)).toThrow(OtsDeclineError);
    expect(transport.sent).toEqual([REJECT, FORFEIT]);
  });

  it("forfeits when the rejection line arrives after the budget ran out", () => {
    let clock = 0;
    const { transport, ots, errors, toPreview } = session({ serverConfirms: false, now: () => clock });
    toPreview();
    ots.submitPreview(PREVIEW_CHOICE);
    clock = BUDGET.policy_budget_ms;
    transport.push(rejected(BOT));
    expect(errors[0]).toMatchObject({ reason: "not_confirmed", forfeitSent: true });
    expect(transport.sent).toEqual([REJECT, FORFEIT]);
  });

  it("forfeits at once when the server refuses the rejection", () => {
    const { transport, ots, errors, toPreview } = session({ serverConfirms: false });
    toPreview();
    ots.submitPreview(PREVIEW_CHOICE);
    const text = "You have already made your decision about agreeing to open team sheets.";
    transport.push(`>${ROOM}\n|error|${text}`);
    expect(errors).toHaveLength(1);
    expect(errors[0]).toMatchObject({ name: "OtsDeclineError", reason: "refused", detail: text });
    expect(transport.sent).toEqual([REJECT, FORFEIT]);
  });

  it("forfeits whenever a |showteam| line arrives, even after a confirmed decline", () => {
    const { transport, ots, errors, toPreview } = session();
    toPreview();
    ots.submitPreview(PREVIEW_CHOICE);
    transport.push(battle.frames[previewIndex + 1] as string);
    transport.push(showteam);
    expect(errors).toHaveLength(1);
    expect(errors[0]).toMatchObject({ reason: "showteam", forfeitSent: true });
    expect(transport.sent).toEqual([REJECT, PREVIEW_CHOICE, FORFEIT]);
    // Forfeited once: later lines change nothing, and a choice gets the same failure.
    transport.push(showteam);
    expect(errors).toHaveLength(1);
    expect(ots.status).toBe("forfeited");
    expect(() => ots.submitPreview(PREVIEW_CHOICE)).toThrow(errors[0] as Error);
  });

  it("forfeits on |showteam| before the preview request too", () => {
    const { transport, errors } = session();
    transport.push(showteam);
    expect(errors[0]).toMatchObject({ reason: "showteam" });
    expect(transport.sent).toEqual([FORFEIT]);
  });

  it("reports a forfeit the closed transport could not send", () => {
    const { transport, ots, toPreview } = session({ serverConfirms: false });
    toPreview();
    transport.end(1006);
    let thrown: unknown;
    try {
      ots.tick(BUDGET.policy_budget_ms);
    } catch (error) {
      thrown = error;
    }
    expect(thrown).toMatchObject({ name: "OtsDeclineError", reason: "not_confirmed", forfeitSent: false });
    expect((thrown as Error).cause).toBeInstanceOf(TransportClosedError);
  });
});
