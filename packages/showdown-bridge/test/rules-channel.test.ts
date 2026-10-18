import { describe, expect, it } from "vitest";
import { checkChannel } from "../src/index.ts";

const PIN = "a".repeat(40);
const OLD_PIN = "b".repeat(40);
const H1 = "1".repeat(64);
const H2 = "2".repeat(64);
const H3 = "3".repeat(64);
const allowed = new Map([
  [H1, PIN],
  [H2, PIN],
  [H3, OLD_PIN],
]);
const channel = (seq: number, hash: string) => ({ channel: "stable", seq, hash });

describe("checkChannel", () => {
  it("accepts a new channel at seq 1", () => {
    expect(checkChannel({ head: channel(1, H1), base: undefined, allowed, pin: PIN })).toEqual([]);
  });

  it("rejects a new channel that does not start at seq 1", () => {
    expect(checkChannel({ head: channel(2, H1), base: undefined, allowed, pin: PIN })).toEqual([
      "a new channel starts at seq 1, not 2",
    ]);
  });

  it("accepts an unchanged channel", () => {
    expect(checkChannel({ head: channel(4, H1), base: channel(4, H1), allowed, pin: PIN })).toEqual([]);
  });

  it("rejects a seq change without a new hash", () => {
    expect(checkChannel({ head: channel(5, H1), base: channel(4, H1), allowed, pin: PIN })).toEqual([
      "seq changed from 4 without a new hash",
    ]);
  });

  it("accepts activation or rollback with seq + 1", () => {
    expect(checkChannel({ head: channel(5, H2), base: channel(4, H1), allowed, pin: PIN })).toEqual([]);
  });

  it.each([4, 6])("rejects a new hash with seq %i instead of 5", (seq) => {
    expect(checkChannel({ head: channel(seq, H2), base: channel(4, H1), allowed, pin: PIN })).toEqual([
      `seq must go from 4 to 5, not ${seq}`,
    ]);
  });

  it("rejects a hash no allowed build produced", () => {
    const problems = checkChannel({ head: channel(5, "9".repeat(64)), base: channel(4, H1), allowed, pin: PIN });
    expect(problems).toEqual([`${"9".repeat(64)} is not a snapshot built from this commit or an earlier main commit`]);
  });

  // Rollback is bounded by the engine pin (ADR-007): an older snapshot built
  // for a different pin is refused even though it was once published.
  it("rejects a snapshot built for another pin", () => {
    expect(checkChannel({ head: channel(5, H3), base: channel(4, H1), allowed, pin: PIN })).toEqual([
      `${H3} was built for Showdown ${OLD_PIN}, not the pinned ${PIN}`,
    ]);
  });

  // After a pin bump an unchanged channel names an old-pin snapshot, so the
  // pin-bump PR must also move the channel to its own build.
  it("rejects an unchanged channel left on the old pin", () => {
    expect(checkChannel({ head: channel(4, H3), base: channel(4, H3), allowed, pin: PIN })).toEqual([
      `${H3} was built for Showdown ${OLD_PIN}, not the pinned ${PIN}`,
    ]);
  });
});
