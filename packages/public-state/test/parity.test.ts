import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { FORMAT_DIR, ROOT, fixtures, fullDex, loadInterpreter, readJson } from "./support.ts";

const api = loadInterpreter();
const dex = fullDex();
const python = process.env.ARENEON_PYTHON ?? join(ROOT, ".venv", "bin", "python");
const here = dirname(fileURLToPath(import.meta.url));

/** The reducer's own fields, compared as they are. `hp_fraction` is derived and the page does not carry it; `item_received` and `ability_received` are the interpreter's (ADR-025), the reducer has no such field; `types` (the type a `typechange` gave, ADR-010 erratum A4 item 7) is the reducer's, the interpreter keeps the volatile alone; so are `recent` (per Pokemon), `condition_turns` (per side) and `elapsed` (the field), the recent-history fields of #1013 (state schema 2), which the interpreter does not carry (the world facts hold its own `active_turns`, `last_move`, `protect_streak` and condition start turns); and `history` (state schema 3, ADR-022 E-22.5.2), the history facts, which `history-facts.ts` derives and `history-facts-parity.test.ts` holds equal to the reducer's. */
function reducerShape(state: any): any {
  const strip = (side: any) => ({
    ...side,
    condition_turns: undefined,
    roster: side.roster.map((mon: any) => {
      const { hp_fraction: _fraction, own: _own, item_received: _ir, ability_received: _ar, types: _types, recent: _recent, ...rest } = mon;
      return rest;
    }),
  });
  const { schema_version: _version, labels: _labels, elapsed: _elapsed, history: _history, ...rest } = state;
  return { ...rest, own: strip(state.own), opponent: strip(state.opponent) };
}

const noMovesOf = (side: any) => ({ ...side, roster: side.roster.map(({ moves_revealed: _moves, ...rest }: any) => rest) });

/** The one known, labelled limit: an Illusion twin shown in two positions is not reconciled. */
const TWIN = "a Pokemon shown in two positions (an Illusion twin) is not reconciled";
const unexpected = (flags: { reason: string }[]) => flags.map((f) => f.reason).filter((r) => r !== TWIN);

describe("the interpreter against the reducer's committed golden states", () => {
  for (const [name, file] of [
    ["golden/p1", "p1.json"],
    ["golden/p2", "p2.json"],
  ] as const) {
    it(`${name} ends in the reducer's state`, () => {
      const fixture = fixtures().find((f) => f.id === name)!;
      const golden = readJson(join(FORMAT_DIR, "reduced", file)).state;
      const run = api.interpretChannel(fixture.player, fixture.lines, dex);
      expect(run.flags).toEqual([]);
      expect(reducerShape(run.frames.at(-1)!.state)).toEqual(reducerShape(golden));
    });
  }

  for (const [id, file] of [
    ["unavailable/b00/p1", "unavailable-b00-p1.json"],
    ["unavailable/b00/p2", "unavailable-b00-p2.json"],
    ["unavailable/b01/p1", "unavailable-b01-p1.json"],
    ["unavailable/b01/p2", "unavailable-b01-p2.json"],
    ["illusion/battle/p1", "illusion-p1.json"],
    ["illusion/battle/p2", "illusion-p2.json"],
    ["baton-pass/battle/p1", "baton-pass-p1.json"],
    ["baton-pass/battle/p2", "baton-pass-p2.json"],
    ["baton-pass-volatiles/battle/p1", "baton-pass-volatiles-p1.json"],
    ["baton-pass-volatiles/battle/p2", "baton-pass-volatiles-p2.json"],
    ["fling/battle/p1", "fling-p1.json"],
    ["fling/battle/p2", "fling-p2.json"],
  ] as const) {
    it(`${id} ends in the reducer's state`, () => {
      const fixture = fixtures().find((f) => f.id === id)!;
      const golden = readJson(join(FORMAT_DIR, "reduced", file)).state;
      const run = api.interpretChannel(fixture.player, fixture.lines, dex);
      const last = reducerShape(run.frames.at(-1)!.state);
      const expected = reducerShape(golden);
      if (id.startsWith("illusion")) {
        // The page does not reconcile every Illusion twin: the active slots must still agree.
        for (const side of ["own", "opponent"] as const) {
          const slots = (s: any) =>
            Object.entries(s[side].slots).map(([k, v]) => [k, v === null ? null : s[side].roster[v as number].species]);
          expect(slots(last)).toEqual(slots(expected));
        }
        expect({ ...last, own: null, opponent: null }).toEqual({ ...expected, own: null, opponent: null });
      } else {
        expect(last).toEqual(expected);
      }
    });
  }
});

describe.skipIf(!existsSync(python))("the interpreter against the reducer at request points", () => {
  // The Transform battle (#938) is the channel that stopped a run on a boost past the cap once the target's stages were not copied (#1375).
  const transform = ["p1", "p2"].map((p) => {
    const doc = readJson(join(FORMAT_DIR, "transform", "battle.json"));
    return { id: `transform/battle/${p}`, player: p, lines: doc.channels[p].lines as string[], team: doc.teams[p] as unknown[] };
  });
  // A channel that boosts after a Transform: the golden battle up to its first turn line, then the opponent's active is dropped by two Speed stages,
  // the own active transforms into it and is dropped by four more (a capped delta, which left -6..6 before the stages were copied).
  const golden = fixtures().find((f) => f.id === "golden/p1")!;
  const turnAt = golden.lines.map((l, i) => (l.startsWith("|turn|") ? i : -1)).filter((i) => i >= 0)[0]!;
  const head = golden.lines.slice(0, turnAt + 1);
  const state = api.interpretChannel(golden.player, head, dex).frames.at(-1)!.state;
  const slotName = (side: any, letter: string) => side.roster[side.slots[letter]].name as string;
  const own = `p1a: ${slotName(state.own, "a")}`;
  const foe = `p2a: ${slotName(state.opponent, "a")}`;
  const boosted = {
    id: "transform/stages/p1",
    player: "p1",
    lines: [...head, `|-unboost|${foe}|spd|2`, `|-transform|${own}|${foe}`, `|-unboost|${own}|spd|4`],
    team: golden.team,
  };
  const all = [...fixtures(), ...transform, boosted];
  const cutsOf = (lines: string[], id: string): number[] => {
    const requests = lines.map((l, i) => (l.startsWith("|request|") ? i + 1 : 0)).filter((i) => i > 0);
    const wanted = id.startsWith("random/")
      ? [0, 1, 2].map((k) => requests[Math.floor((k * requests.length) / 3)]!)
      : requests;
    return [...new Set([...wanted, lines.length])].sort((a, b) => a - b);
  };
  const reference = (() => {
    // The describe body runs at collection even when the suite is skipped: spawn only with an interpreter.
    if (!existsSync(python)) return {} as Record<string, Record<string, any>>;
    const jobs = all.map((f) => ({ id: f.id, player: f.player, lines: f.lines, team: f.team, cuts: cutsOf(f.lines, f.id) }));
    const run = spawnSync(python, [join(here, "reduce_points.py")], {
      input: JSON.stringify({ engine_data: join(FORMAT_DIR, "engine-data.json"), jobs }),
      maxBuffer: 1 << 30,
      encoding: "utf8",
    });
    if (run.status !== 0) throw new Error(run.stderr);
    return JSON.parse(run.stdout) as Record<string, Record<string, any>>;
  })();

  for (const fixture of all) {
    it(`${fixture.id}`, () => {
      const run = api.interpretChannel(fixture.player, fixture.lines, dex);
      expect(unexpected(run.flags)).toEqual([]);
      for (const [cut, expected] of Object.entries(reference[fixture.id]!)) {
        if (expected.refused !== undefined) continue;
        const got = reducerShape(run.frames[Number(cut)]!.state);
        const twin = run.flags.some((f) => f.reason === TWIN && f.index < Number(cut));
        if (twin) {
          // Past an unreconciled twin the rosters differ by design; the field state must still agree.
          const field = ({ own: _o, opponent: _p, ...rest }: any) => rest;
          expect(field(got), `${fixture.id} cut ${cut} field`).toEqual(field(reducerShape(expected)));
        } else if (fixture.id.startsWith("illusion")) {
          for (const side of ["own", "opponent"] as const) {
            const slots = (s: any) =>
              Object.entries(s[side].slots).map(([k, v]) => [k, v === null ? null : s[side].roster[v as number].species]);
            expect(slots(got), `${fixture.id} cut ${cut} ${side} slots`).toEqual(slots(reducerShape(expected)));
          }
        } else if (fixture.id.startsWith("transform/")) {
          // The interpreter ports the Transform stat-stage copy (#1375), not the reducer's replacement of the revealed move list (#938, `moves_revealed`).
          const noMoves = (state: any) => ({ ...state, own: noMovesOf(state.own), opponent: noMovesOf(state.opponent) });
          expect(noMoves(got), `${fixture.id} cut ${cut}`).toEqual(noMoves(reducerShape(expected)));
        } else {
          expect(got, `${fixture.id} cut ${cut}`).toEqual(reducerShape(expected));
        }
      }
    });
  }
});
