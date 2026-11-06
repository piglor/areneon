import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { type Engine, formatRulesHash, loadEngine, resolveFormat } from "../src/index.ts";
import {
  loadOpenSheetCorpus,
  OPEN_SHEET_BATTLES,
  OPEN_SHEET_DIR,
  OPEN_SHEET_FORMAT,
  OPEN_SHEET_PROVENANCE,
  openSheetSpecs,
} from "../src/open-sheet-corpus.ts";
import { playerChannel } from "../src/player-channel.ts";
import { loadRandomCorpus, playRandomBattle, RANDOM_MANIFEST, randomBattleFile } from "../src/random-corpus.ts";

// The open-regime channel corpus (scripts/capture-open-regime-corpus.ts):
// seeded A1 battles of the engine's open-sheet format, one game, no series.

const toId = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]+/g, "");

let engine: Engine;
beforeAll(() => {
  engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
});

const corpus = loadOpenSheetCorpus();

describe("the open-regime channel corpus (ADR-027 slice 1)", () => {
  it("is the committed set its manifest names", () => {
    const manifest = JSON.parse(readFileSync(join(OPEN_SHEET_DIR, RANDOM_MANIFEST), "utf8")) as {
      format: string;
      battles: { id: string; seed: string; turns: number; winner: string | null }[];
    };
    expect(manifest.format).toBe(OPEN_SHEET_FORMAT);
    expect(manifest.battles).toEqual(corpus.map((b) => ({ id: b.id, seed: b.seed, turns: b.turns, winner: b.winner })));
    expect(corpus).toHaveLength(OPEN_SHEET_BATTLES);
    expect(openSheetSpecs().map((s) => s.id)).toEqual(corpus.map((b) => b.id));
  });

  it("re-plays to the same channels (determinism, A1, one game)", async () => {
    for (const committed of corpus) {
      const { id, seed, actor_seeds, teams } = committed;
      const spec = { id, seed, actor_seeds, teams };
      const record = await playRandomBattle(engine, spec, undefined, OPEN_SHEET_FORMAT);
      expect(record.forfeit, id).toBeUndefined();
      expect(randomBattleFile(spec, record, OPEN_SHEET_PROVENANCE), id).toEqual(committed);
    }
  }, 120_000);

  it("runs the format through runBattle with its own version manifest and a legal set at every request", async () => {
    const { id, seed, actor_seeds, teams } = corpus[0] as (typeof corpus)[number];
    const record = await playRandomBattle(engine, { id, seed, actor_seeds, teams }, undefined, OPEN_SHEET_FORMAT);
    expect(record.manifest).toEqual(resolveFormat(engine, OPEN_SHEET_FORMAT).manifest);
    expect(record.manifest.format_id).toBe(OPEN_SHEET_FORMAT);
    for (const player of ["p1", "p2"] as const) {
      expect(record.requests[player].length).toBeGreaterThan(2);
      for (const r of record.requests[player]) if (r.phase !== "wait") expect(r.legal?.choices.length).toBeGreaterThan(0);
    }
  });

  it("plays the specs the capture script derives from the random corpus", () => {
    const specs = openSheetSpecs();
    for (const [i, spec] of specs.entries()) {
      const { id, seed, actor_seeds, teams } = corpus[i] as (typeof corpus)[number];
      expect({ id, seed, actor_seeds, teams }).toEqual(spec);
    }
    expect(new Set(specs.map((s) => s.seed)).size).toBe(specs.length);
    const randomSeeds = new Set(loadRandomCorpus().map((b) => b.seed));
    expect(specs.some((s) => randomSeeds.has(s.seed))).toBe(false);
  });

  it("holds player-perspective channels that carry the engine's sheets before the lead choice", () => {
    for (const battle of corpus) {
      for (const player of ["p1", "p2"] as const) {
        const { lines } = battle.channels[player];
        expect(() => playerChannel(battle.channels[player]), `${battle.id} ${player}`).not.toThrow();
        const sheets = lines.map((line, i) => ({ line, i })).filter(({ line }) => line.startsWith("|showteam|"));
        expect(sheets.map(({ line }) => line.split("|")[2]), `${battle.id} ${player}`).toEqual(["p1", "p2"]);
        const firstRequest = lines.findIndex((line) => line.startsWith("|request|"));
        expect(Math.max(...sheets.map(({ i }) => i))).toBeLessThan(firstRequest);
      }
    }
  });

  // The pinned engine's sheet: species, item, ability, moves, nature, gender and level;
  // no nickname, Stat Points or IVs (gap row open-team-sheet-contents).
  it("shows each side's sheet with the nature and no nickname, Stat Points or IVs", () => {
    for (const battle of corpus) {
      const lines = battle.channels.p1.lines.filter((line) => line.startsWith("|showteam|"));
      for (const line of lines) {
        const [, , side, ...rest] = line.split("|") as [string, string, "p1" | "p2", ...string[]];
        const packed = rest.join("|");
        const mons = packed.split("]").map((mon) => mon.split("|"));
        const team = battle.teams[side];
        expect(mons).toHaveLength(team.length);
        for (const [k, fields] of mons.entries()) {
          const set = team[k] as (typeof team)[number];
          const [name, species, item, ability, moves, nature, evs, gender, ivs, , level] = fields as string[];
          // The engine packs an empty nickname as the species name (and then leaves the species field empty).
          expect(toId((species as string) || (name as string))).toBe(toId(set.species));
          expect(toId(name as string)).toBe(toId(set.species));
          expect(evs).toBe("");
          expect(ivs).toBe("");
          expect(nature).toBe(set.nature);
          expect(toId(item as string)).toBe(toId(set.item));
          expect(toId(ability as string)).toBe(toId(set.ability));
          expect((moves as string).split(",").map(toId)).toEqual(set.moves.map(toId));
          expect(level).toBe(String(set.level));
          expect(["M", "F", ""]).toContain(gender);
          if (set.gender === "M" || set.gender === "F") expect(gender).toBe(set.gender);
        }
      }
    }
  });

  it("binds the open-sheet format's own replay identity, unlike the closed corpus", () => {
    const open = resolveFormat(engine, OPEN_SHEET_FORMAT);
    const closed = resolveFormat(engine, "gen9championsvgc2026regmc");
    expect(open.manifest.format_rules_hash).toBe(formatRulesHash(open.rules));
    expect(open.manifest.format_rules_hash).not.toBe(closed.manifest.format_rules_hash);
    const closedText = loadRandomCorpus().map((b) => b.channels.p1.lines.join("\n")).join("\n");
    expect(closedText).not.toContain("|showteam|");
  });
});
