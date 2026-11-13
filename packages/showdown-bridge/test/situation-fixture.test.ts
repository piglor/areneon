import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { GOLDEN_DIR } from "../src/golden.ts";
import { classifySlot } from "../src/situation.ts";

// The fixture the Python twin of `classifySlot` (ml/areneon_ml/learner/situation.py, ADR-010 erratum A11.2) is held equal to
// (tests/ml/learner/test_situation.py). Here the stored verdicts must be what `classifySlot` says of the stored rows, so a change
// of the TS definition fails until `scripts/export-situation-fixture.ts` is rerun and the Python twin follows.
interface Entry {
  id: string;
  request: { active?: unknown[]; side: { pokemon: { active: boolean; condition: string }[] } };
  rows: { threats: unknown[]; speeds: unknown[]; foe_priorities: unknown[] };
  slots: { position: number; situation: string }[];
  threatened: boolean;
}
const fixture = JSON.parse(readFileSync(join(GOLDEN_DIR, "..", "..", "situation", "rows.json"), "utf8")) as { entries: Entry[] };

describe("situation fixture", () => {
  it("holds classifySlot's verdict for every living active slot of a move request", () => {
    expect(fixture.entries.length).toBeGreaterThan(100);
    const seen = new Set<string>();
    for (const entry of fixture.entries) {
      const expected = entry.request.active === undefined ? [] : entry.request.side.pokemon.flatMap((p, position) => (p.active && !/ fnt$/.test(p.condition) ? [{ position, situation: classifySlot(entry.rows as never, position).situation }] : []));
      expect(entry.slots, entry.id).toEqual(expected);
      expect(entry.threatened, entry.id).toBe(expected.some((s) => s.situation === "threatened"));
      for (const s of expected) seen.add(s.situation);
    }
    expect([...seen].sort()).toEqual(["exposed_slower", "safe", "threatened"]);
  });
});
