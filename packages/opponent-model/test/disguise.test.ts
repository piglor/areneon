import { describe, expect, it } from "vitest";
import { enumerate, hasMatching } from "../src/disguise.ts";

describe("hasMatching (Hall's condition)", () => {
  it("is true for no positions and for positions with room, false when a position has no option", () => {
    expect(hasMatching([])).toBe(true);
    expect(hasMatching([[0, 9], [1, 9]])).toBe(true);
    expect(hasMatching([[0], []])).toBe(false);
  });
  it("is false when more positions than slots compete for the same slot, true after an augmenting path", () => {
    expect(hasMatching([[9], [9]])).toBe(false);
    expect(hasMatching([[9, 8], [9]])).toBe(true);
  });
});

describe("enumerate", () => {
  const hiding = new Set([9]);
  it("lists the assignments in option order and never uses a hiding slot twice", () => {
    expect(enumerate([[0, 9], [1, 9]], hiding, 10)).toEqual([[0, 1], [0, 9], [9, 1]]);
  });
  it("returns the list at exactly the limit and null above it", () => {
    expect(enumerate([[0, 9], [1, 9]], hiding, 3)).toHaveLength(3);
    expect(enumerate([[0, 9], [1, 9]], hiding, 2)).toBeNull();
  });
  it("has one empty assignment for no positions", () => {
    expect(enumerate([], hiding, 1)).toEqual([[]]);
  });
});
