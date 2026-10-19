import { afterEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_SNAPSHOT_WAIT_MS, DeadlineWatchdog, type WatchdogFire } from "../src/index.ts";

// The watchdog on a fake clock: `now` is the local clock and vitest's fake
// timers run the deadlines, advanced together by `advance`.

let now = 0;
const fires: WatchdogFire[] = [];

function make(): DeadlineWatchdog {
  return new DeadlineWatchdog({ clock: () => now, onFire: (fire) => fires.push(fire) });
}

function advance(ms: number): void {
  now += ms;
  vi.advanceTimersByTime(ms);
}

vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
afterEach(() => {
  now = 0;
  fires.length = 0;
  vi.clearAllTimers();
});

describe("overrun", () => {
  it("fires at received_at + policy_budget_ms, before submit_within", () => {
    const dog = make();
    // submit_within 20 s, a 1 s fallback reserve: the policy has 19 s.
    expect(dog.armOverrun(3, 5_000, { submit_within_ms: 20_000, policy_budget_ms: 19_000 })).toBe(true);
    expect(dog.armed).toBe(3);
    advance(23_999);
    expect(fires).toEqual([]);
    advance(1);
    expect(fires).toEqual([{ rqid: 3, reason: "overrun", fire_at: 24_000 }]);
    // The fallback still has the reserve before submit_within (25 s).
    expect(25_000 - now).toBe(1_000);
    expect(dog.armed).toBeUndefined();
  });

  it("fires at once for a budget of 0, and for a deadline already past", () => {
    const dog = make();
    dog.armOverrun(1, 0, { submit_within_ms: 500, policy_budget_ms: 0 });
    advance(0);
    expect(fires).toHaveLength(1);
    now = 10_000;
    dog.armOverrun(2, 0, { submit_within_ms: 3_000, policy_budget_ms: 2_000 });
    advance(0);
    expect(fires).toHaveLength(2);
  });

  it("does not fire for a request settled before its deadline", () => {
    const dog = make();
    dog.armOverrun(3, 0, { submit_within_ms: 2_000, policy_budget_ms: 1_000 });
    dog.settle(3);
    advance(10_000);
    expect(fires).toEqual([]);
    expect(dog.armed).toBeUndefined();
  });

  it("settling another request leaves the armed one", () => {
    const dog = make();
    dog.armOverrun(3, 0, { submit_within_ms: 2_000, policy_budget_ms: 1_000 });
    dog.settle(2);
    advance(1_000);
    expect(fires).toHaveLength(1);
  });
});

describe("one fire per request, never for a stale one", () => {
  it("refuses to arm a request that fired or settled", () => {
    const dog = make();
    dog.armSnapshotWait(1, 0, 10);
    advance(10);
    expect(fires).toHaveLength(1);
    expect(dog.armSnapshotWait(1, now, 10)).toBe(false);
    expect(dog.armOverrun(1, now, { submit_within_ms: 10, policy_budget_ms: 0 })).toBe(false);
    dog.settle(2);
    expect(dog.armSnapshotWait(2, now, 10)).toBe(false);
    advance(1_000);
    expect(fires).toHaveLength(1);
  });

  it("arming a newer request drops the older deadline", () => {
    const dog = make();
    dog.armSnapshotWait(1, 0, 100);
    dog.armSnapshotWait(2, 50, 100);
    advance(100);
    expect(fires).toEqual([]);
    advance(50);
    expect(fires).toEqual([{ rqid: 2, reason: "snapshot_missing", fire_at: 150 }]);
  });

  it("cancel and dispose drop the deadline without settling the request", () => {
    const dog = make();
    dog.armSnapshotWait(1, 0, 100);
    dog.cancel();
    advance(1_000);
    expect(fires).toEqual([]);
    expect(dog.armSnapshotWait(1, now, 100)).toBe(true);
    dog.dispose();
    advance(1_000);
    expect(fires).toEqual([]);
    dog.cancel();
  });

  it("a new request drops the deadline of the one before", () => {
    const dog = make();
    dog.armSnapshotWait(1, 0, 100);
    dog.onRequest(2, 2, "move");
    advance(1_000);
    expect(fires).toEqual([]);
  });
});

describe("snapshot wait", () => {
  it("fires after the wait from the receipt", () => {
    const dog = make();
    dog.armSnapshotWait(4, 200, DEFAULT_SNAPSHOT_WAIT_MS);
    advance(1_199);
    expect(fires).toEqual([]);
    advance(1);
    expect(fires).toEqual([{ rqid: 4, reason: "snapshot_missing", fire_at: 1_200 }]);
  });

  it("refuses a wait that is not a whole number of ms >= 0", () => {
    const dog = make();
    expect(() => dog.armSnapshotWait(1, 0, -1)).toThrow(RangeError);
    expect(() => dog.armSnapshotWait(1, 0, 1.5)).toThrow(RangeError);
    expect(dog.armSnapshotWait(1, 0, 0)).toBe(true);
  });
});

describe("inherited deadline", () => {
  it("moves a still-armed deadline to the follow-up request without restarting it", () => {
    const dog = make();
    dog.armOverrun(3, 0, { submit_within_ms: 20_000, policy_budget_ms: 19_000 });
    advance(10_000);
    expect(dog.inherit(7)).toBe(true);
    expect(dog.armed).toBe(7);
    advance(8_999);
    expect(fires).toEqual([]);
    advance(1);
    expect(fires).toEqual([{ rqid: 7, reason: "overrun", fire_at: 19_000 }]);
  });

  it("keeps the original deadline after the first request was answered", () => {
    const dog = make();
    dog.armOverrun(3, 0, { submit_within_ms: 20_000, policy_budget_ms: 19_000 });
    advance(4_000);
    dog.settle(3);
    expect(dog.inherit(7)).toBe(true);
    advance(14_999);
    expect(fires).toEqual([]);
    advance(1);
    expect(fires).toEqual([{ rqid: 7, reason: "overrun", fire_at: 19_000 }]);
  });

  it("fires at once when the original deadline has passed, and only once", () => {
    const dog = make();
    dog.armSnapshotWait(3, 0, 1_000);
    advance(1_000);
    expect(fires).toHaveLength(1);
    expect(dog.inherit(7)).toBe(true);
    advance(0);
    expect(fires).toHaveLength(2);
    expect(fires[1]).toEqual({ rqid: 7, reason: "snapshot_missing", fire_at: 1_000 });
    expect(dog.inherit(7)).toBe(false);
  });

  it("has nothing to inherit before any deadline", () => {
    expect(make().inherit(1)).toBe(false);
  });
});

describe("auto-choose detector", () => {
  it("reports an unanswered request when a newer one arrives", () => {
    const dog = make();
    expect(dog.onRequest(1, 1, "preview")).toBeNull();
    dog.onAnswered(1);
    expect(dog.onRequest(3, 2, "move")).toBeNull();
    expect(dog.onRequest(5, 3, "move")).toEqual({ rqid: 3, request_seq: 2, phase: "move" });
    // Reported once: the newer request is now the open one.
    dog.onAnswered(5);
    expect(dog.onRequest(7, 4, "forceSwitch")).toBeNull();
    expect(dog.onRequest(8, 5, "wait")).toEqual({ rqid: 7, request_seq: 4, phase: "forceSwitch" });
  });

  it("asks nothing of a wait request, and ignores an answer for another request", () => {
    const dog = make();
    dog.onRequest(1, 1, "wait");
    expect(dog.onRequest(2, 2, "move")).toBeNull();
    dog.onAnswered(1);
    expect(dog.onRequest(3, 3, "move")).toEqual({ rqid: 2, request_seq: 2, phase: "move" });
  });
});
