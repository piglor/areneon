import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { FORMAT_DIR, ROOT, fixtures, fullDex, loadInterpreter, readJson } from "./support.ts";

const api = loadInterpreter();
const dex = fullDex();
const python = join(ROOT, ".venv", "bin", "python");
const here = dirname(fileURLToPath(import.meta.url));

/** The reducer's own fields, compared as they are. `hp_fraction` is derived and the page does not carry it. */
function reducerShape(state: any): any {
  const strip = (side: any) => ({
    ...side,
    roster: side.roster.map((mon: any) => {
      const { hp_fraction: _fraction, own: _own, ...rest } = mon;
      return rest;
    }),
  });
  const { schema_version: _version, labels: _labels, ...rest } = state;
  return { ...rest, own: strip(state.own), opponent: strip(state.opponent) };
}

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
  const all = fixtures();
  const cutsOf = (lines: string[], id: string): number[] => {
    const requests = lines.map((l, i) => (l.startsWith("|request|") ? i + 1 : 0)).filter((i) => i > 0);
    const wanted = id.startsWith("random/")
      ? [0, 1, 2].map((k) => requests[Math.floor((k * requests.length) / 3)]!)
      : requests;
    return [...new Set([...wanted, lines.length])].sort((a, b) => a - b);
  };
  const reference = (() => {
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
        } else {
          expect(got, `${fixture.id} cut ${cut}`).toEqual(reducerShape(expected));
        }
      }
    });
  }
});
