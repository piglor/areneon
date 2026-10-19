import { EnumeratorDefectError, type LegalChoiceSet } from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import {
  classifyRefusal,
  type CurrentRequest,
  ProtocolReader,
  type RefusalInput,
  RefusalError,
  type RoomEvent,
  type SequencedRequest,
} from "../src/index.ts";
import { INVALID_CHOICE, OUT_OF_TURN_CHOICE } from "../testing/server-frames.ts";
import { battle, frameIndex, legalFor, sentAfter, serverFrame } from "./frames.ts";

/** Reads the battle up to (not including) frame `end`; returns the last request and the reader. */
function readTo(end: number): { reader: ProtocolReader; request: SequencedRequest } {
  const reader = new ProtocolReader();
  const events = battle.frames.slice(0, end).flatMap((frame) => reader.read(frame));
  const request = events.flatMap((e) => (e.kind === "request" ? [e.request] : [])).at(-1);
  if (request === undefined) throw new Error("no request before that frame");
  return { reader, request };
}

/** The `|error|` text of one recorded frame. */
function errorOf(frame: string): string {
  const event = new ProtocolReader().read(frame)[0] as RoomEvent;
  if (event.kind !== "error") throw new Error("not an error frame");
  return event.error;
}

const current = (request: SequencedRequest, legal: LegalChoiceSet | null = legalFor(request)): CurrentRequest => ({
  request_seq: request.request_seq,
  ...(request.rqid === undefined ? {} : { rqid: request.rqid }),
  legal,
});

describe("[Unavailable choice] (ADR-005 §1, ADR-006 rule 5)", () => {
  const index = frameIndex("|error|[Unavailable choice]");
  const { reader, request } = readTo(index);
  const choice = sentAfter(index - 1);

  it("is unavailable for the pending candidate p1 sent, and a new request follows", () => {
    const legal = legalFor(request);
    expect(legal.pending).toContain(choice);
    const refusal = classifyRefusal({
      error: errorOf(battle.frames[index] as string),
      submission: { request_seq: request.request_seq, choice },
      current: current(request, legal),
      mode: "harness",
    });
    expect(refusal).toEqual({ kind: "unavailable", choice });
    reader.read(battle.frames[index] as string);
    const next = reader.read(battle.frames[index + 1] as string)[0];
    expect(next).toMatchObject({ kind: "request", resend: false, request: { request_seq: request.request_seq + 1 } });
  });

  it("is a defect for a non-pending candidate (ADR-006 rule 6)", () => {
    const legal = legalFor(request);
    const asCandidate = { ...legal, choices: [choice], pending: [] };
    expect(() =>
      classifyRefusal({
        error: errorOf(battle.frames[index] as string),
        submission: { request_seq: request.request_seq, choice },
        current: current(request, asCandidate),
        mode: "harness",
      }),
    ).toThrow(EnumeratorDefectError);
  });

  it("in a live match, marks a non-pending candidate's defect and sends nothing: the updated request is new", () => {
    const legal = legalFor(request);
    const asCandidate = { ...legal, choices: [choice, ...legal.choices], pending: [] };
    const refusal = classifyRefusal({
      error: errorOf(battle.frames[index] as string),
      submission: { request_seq: request.request_seq, choice },
      current: current(request, asCandidate),
      mode: "live_match",
    });
    expect(refusal).toMatchObject({ kind: "defect", defect: true });
    expect(refusal.kind === "defect" && refusal.error).toBeInstanceOf(EnumeratorDefectError);
    expect("next" in refusal).toBe(false);
    // The engine's updated request follows, with a new rqid and request_seq.
    const updated = new ProtocolReader();
    battle.frames.slice(0, index + 2).forEach((frame) => updated.read(frame));
    expect(updated.state(battle.room).last).toMatchObject({
      request_seq: request.request_seq + 1,
      request: { update: true },
    });
    expect(updated.state(battle.room).last?.rqid).not.toBe(request.rqid);
  });
});

describe("a sim [Invalid choice] of a current, non-pending candidate", () => {
  // The engine's real refusal of INVALID_CHOICE stands in for an enumerator
  // defect: the test's legal set offers it as a non-pending candidate.
  const index = frameIndex("|error|[Invalid choice] Can't move");
  const { request } = readTo(index);
  const error = errorOf(battle.frames[index] as string);
  const legal = legalFor(request);
  const defective: LegalChoiceSet = { ...legal, choices: [INVALID_CHOICE, ...legal.choices] };
  const input = (overrides: Partial<RefusalInput> = {}): RefusalInput => ({
    error,
    submission: { request_seq: request.request_seq, choice: INVALID_CHOICE },
    current: current(request, defective),
    mode: "harness",
    ...overrides,
  });

  it("was refused on the request p1 answered", () => {
    expect(sentAfter(index - 1)).toBe(INVALID_CHOICE);
    expect(request.request.update).toBe(true);
  });

  it("throws EnumeratorDefectError in a harness: every test and harness run fails", () => {
    expect(() => classifyRefusal(input())).toThrow(EnumeratorDefectError);
    try {
      classifyRefusal(input());
    } catch (defect) {
      expect((defect as EnumeratorDefectError).refused).toEqual([{ choice: INVALID_CHOICE, legal: false, error }]);
    }
  });

  it("in a live match, marks the match and sends the next non-pending candidate with the current rqid", () => {
    const refusal = classifyRefusal(input({ mode: "live_match" }));
    expect(refusal).toMatchObject({
      kind: "defect",
      defect: true,
      next: { choice: legal.choices[0], rqid: request.rqid },
    });
    expect(refusal.kind === "defect" && refusal.error).toBeInstanceOf(EnumeratorDefectError);
  });

  it("skips candidates already refused, then finally sends default", () => {
    const [first, second] = legal.choices as [string, string];
    const skipped = classifyRefusal(input({ mode: "live_match", refused: [first] }));
    expect(skipped.kind === "defect" && skipped.next?.choice).toBe(second);
    const last = legal.choices.at(-1) as string;
    const final = classifyRefusal(
      input({ mode: "live_match", submission: { request_seq: request.request_seq, choice: last } }),
    );
    expect(final.kind === "defect" && final.next).toEqual({ choice: "default", rqid: request.rqid });
  });

  it("stops after default: a refused default is not classified", () => {
    const submission = { request_seq: request.request_seq, choice: "default" };
    expect(() => classifyRefusal(input({ mode: "live_match", submission }))).toThrow(RefusalError);
  });

  it("sends no rqid where the request had none (A1)", () => {
    const { rqid, ...simRequest } = request;
    const refusal = classifyRefusal(input({ mode: "live_match", current: current(simRequest, defective) }));
    expect(refusal.kind === "defect" && refusal.next).toEqual({ choice: legal.choices[0] });
  });

  it("is not classified for a pending candidate or a choice outside the set", () => {
    const pending = { ...defective, choices: legal.choices, pending: [INVALID_CHOICE] };
    expect(() => classifyRefusal(input({ current: current(request, pending) }))).toThrow(RefusalError);
    expect(() => classifyRefusal(input({ current: current(request, legal) }))).toThrow(RefusalError);
  });

  it("refuses a legal set built for another request", () => {
    const stale = { ...defective, request_seq: request.request_seq - 1 };
    expect(() => classifyRefusal(input({ current: current(request, stale) }))).toThrow(/legal set is for request/);
  });
});

describe("stale_choice: refusals that are never defects (ADR-005 §1)", () => {
  const answered = frameIndex('|request|{"active"');
  const { request } = readTo(answered + 1);
  const submission = { request_seq: request.request_seq, choice: sentAfter(answered) };

  it.each(["nothing_to_choose", "too_late"])("the server layer's %s on the current request", (name) => {
    const refusal = classifyRefusal({
      error: errorOf(serverFrame(name)),
      submission,
      current: current(request),
      mode: "harness",
    });
    expect(refusal).toEqual({ kind: "stale_choice", choice: submission.choice, reason: "refused_as_stale" });
  });

  it("a choice for a request that is no longer current", () => {
    const { request: later } = readTo(frameIndex("|error|[Unavailable choice]"));
    const refusal = classifyRefusal({
      error: errorOf(serverFrame("too_late")),
      submission,
      current: current(later),
      mode: "harness",
    });
    expect(refusal).toMatchObject({ kind: "stale_choice", reason: "not_current" });
  });

  it("the server's answer to a choice sent while p1 holds a wait request", () => {
    const index = frameIndex("|error|[Invalid choice] Can't do anything");
    const { request: wait } = readTo(index);
    expect(wait.phase).toBe("wait");
    const waiting = { request_seq: wait.request_seq, choice: OUT_OF_TURN_CHOICE };
    const refusal = classifyRefusal({
      error: errorOf(serverFrame("nothing_to_choose")),
      submission: waiting,
      current: current(wait, null),
      mode: "live_match",
    });
    expect(refusal).toEqual({ kind: "stale_choice", choice: OUT_OF_TURN_CHOICE, reason: "no_request" });
  });

  // Sim-only (the server answers a wait itself, above): a probe of the engine's text.
  it("the sim's not-your-turn text, probed on the in-process engine", () => {
    const index = frameIndex("|error|[Invalid choice] Can't do anything");
    const { request: wait } = readTo(index);
    expect(wait.phase).toBe("wait");
    expect(sentAfter(index - 1)).toBe(OUT_OF_TURN_CHOICE);
    const error = errorOf(battle.frames[index] as string);
    const waiting = { request_seq: wait.request_seq, choice: OUT_OF_TURN_CHOICE };
    expect(classifyRefusal({ error, submission: waiting, current: current(wait, null), mode: "harness" })).toEqual({
      kind: "stale_choice",
      choice: OUT_OF_TURN_CHOICE,
      reason: "no_request",
    });
    // The same text on an open request is still not a defect.
    expect(classifyRefusal({ error, submission, current: current(request), mode: "harness" })).toMatchObject({
      reason: "refused_as_stale",
    });
  });

  it("any refusal after |request|null withdrew the request", () => {
    const reader = new ProtocolReader();
    reader.read(serverFrame("request_null"));
    expect(reader.state(battle.room).open).toBe(false);
    const refusal = classifyRefusal({
      error: errorOf(serverFrame("nothing_to_choose")),
      submission,
      current: null,
      mode: "live_match",
    });
    expect(refusal).toMatchObject({ kind: "stale_choice", reason: "no_request" });
  });
});

it("refuses to classify an |error| that is not a choice refusal", () => {
  expect(() =>
    classifyRefusal({
      error: "/challenge - Access denied.",
      submission: { request_seq: 1, choice: "default" },
      current: null,
      mode: "harness",
    }),
  ).toThrow(RefusalError);
});
