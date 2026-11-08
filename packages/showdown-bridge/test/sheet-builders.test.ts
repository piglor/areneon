// The engine-sheet/1 and revealed-only/1 builders (ADR-027 slice 3, #1151): values against the true teams, the
// open-regime and closed-regime invariance with negative controls, A1/A2 equality, the type rule and a source scan.
import { readFileSync } from "node:fs";
import { interpretChannel, type PublicState } from "@areneon/public-state";
import { beforeAll, describe, expect, it } from "vitest";
import { SAMPLES_DIR } from "../src/samples.ts";
import {
  type BattleRecord,
  type Engine,
  loadEngine,
  type OmniscientLog,
  type PlayerChannel,
  playerChannel,
  randomLegalActor,
  readOpponentSheet,
  readTrainingExample,
  revealedOnlySheet,
  REVEALED_ONLY_HASH,
  ENGINE_SHEET_HASH,
  SheetBuildError,
  sheetFromChannel,
  type StampedTeam,
} from "../src/index.ts";
import { loadOpenSheetCorpus, OPEN_SHEET_FORMAT } from "../src/open-sheet-corpus.ts";
import { loadRandomCorpus, playRandomBattle, type RandomBattleFile } from "../src/random-corpus.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "../src/golden.ts";

type Json = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any
const toId = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]+/g, "");
declare const channel: PlayerChannel;
declare const team: StampedTeam;
declare const log: OmniscientLog;
declare const record: BattleRecord;
const OPEN = { team_sheets: "open" as const };
const CLOSED = {};

let engine: Engine;
beforeAll(() => {
  engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
});

/** Every prefix of a channel that ends in a `|request|` line. */
function requestPrefixes(file: RandomBattleFile, player: "p1" | "p2", regime: "open" | "closed"): PlayerChannel[] {
  const log = file.channels[player];
  const out: PlayerChannel[] = [];
  log.lines.forEach((line, i) => {
    if (line.startsWith("|request|") && !line.startsWith("|request|null")) out.push(playerChannel({ ...log, lines: log.lines.slice(0, i + 1) }, regime));
  });
  return out;
}

const open = loadOpenSheetCorpus();
const closed = loadRandomCorpus().slice(0, 6);
/** The reader's state at the end of a channel prefix. */
function stateAt(player: "p1" | "p2", lines: readonly string[]): PublicState {
  const frames = interpretChannel(player, [...lines]).frames;
  return (frames[frames.length - 1] as { state: PublicState }).state;
}
const other = (p: "p1" | "p2"): "p1" | "p2" => (p === "p1" ? "p2" : "p1");

describe("sheetFromChannel (engine-sheet/1)", () => {
  it("equals the opponent's true team field for field at every request, brought unknown", () => {
    let checked = 0;
    for (const battle of open) {
      for (const player of ["p1", "p2"] as const) {
        const truth = battle.teams[other(player)];
        for (const channel of requestPrefixes(battle, player, "open")) {
          const block = readOpponentSheet(sheetFromChannel(channel, OPEN));
          expect(block.source).toEqual({ kind: "engine-sheet", hash: ENGINE_SHEET_HASH });
          expect(block.slots).toHaveLength(truth.length);
          block.slots.forEach((slot, i) => {
            const set = truth[i] as (typeof truth)[number];
            expect(slot.brought_source).toBe("unknown");
            expect(slot.brought_ppm).toBe(0);
            // A Mega forme's set names the forme the validator rewrites: the sheet is of the engine's own version of it.
            if (/-Mega/.test(set.species)) return;
            expect(slot.item.entries).toEqual([{ id: set.item ? toId(set.item) : null, ppm: 1000000 }]);
            expect(slot.ability.entries).toEqual([{ id: toId(set.ability), ppm: 1000000 }]);
            expect(slot.nature.entries).toEqual([{ id: toId(set.nature ?? ""), ppm: 1000000 }]);
            expect(slot.moves.entries.map((e) => e.id)).toEqual(set.moves.map(toId).sort());
            expect(slot.moves.slots_known).toBe(set.moves.length);
            checked++;
          });
        }
      }
    }
    expect(checked).toBeGreaterThan(100);
  });

  it("refuses a closed-format record, a channel with no sheet and a channel that ends in no request", () => {
    const battle = open[0] as RandomBattleFile;
    const [first] = requestPrefixes(battle, "p1", "open") as [PlayerChannel];
    expect(() => sheetFromChannel(first, CLOSED)).toThrow(SheetBuildError);
    expect(() => sheetFromChannel(first, { team_sheets: "closed" })).toThrow("open-sheet format only");
    const log = battle.channels.p1;
    const stripped = playerChannel({ ...log, lines: log.lines.slice(0, log.lines.findIndex((l) => l.startsWith("|request|")) + 1).filter((l) => !l.startsWith("|showteam|")) }, "closed");
    expect(() => sheetFromChannel(stripped, OPEN)).toThrow("holds no opponent team sheet");
    const noRequest = playerChannel({ ...log, lines: log.lines.slice(0, log.lines.findIndex((l) => l.startsWith("|request|"))) }, "open");
    expect(() => sheetFromChannel(noRequest, OPEN)).toThrow("does not end in a request");
  });

  it("fails closed on a sheet it cannot give a slot for (no nature, a repeated move, five moves, a short sheet, a stray line)", () => {
    const battle = open[0] as RandomBattleFile;
    const [first] = requestPrefixes(battle, "p1", "open") as [PlayerChannel];
    const at = first.lines.findIndex((l) => l.startsWith("|showteam|p2|"));
    const withSheet = (change: (mons: string[][]) => string[][]): PlayerChannel => {
      const [head, ...rest] = (first.lines[at] as string).split("|showteam|p2|") as [string, ...string[]];
      const mons = change(rest.join("").split("]").map((m) => m.split("|")));
      const lines = [...first.lines];
      lines[at] = `${head}|showteam|p2|${mons.map((m) => m.join("|")).join("]")}`;
      return { ...first, lines } as PlayerChannel;
    };
    // fields: NAME|SPECIES|ITEM|ABILITY|MOVES|NATURE|...
    expect(() => sheetFromChannel(withSheet((m) => m.map((x) => ((x[5] = ""), x))), OPEN)).toThrow("no nature");
    expect(() => sheetFromChannel(withSheet((m) => m.map((x) => ((x[4] = `${(x[4] as string).split(",")[0]},${x[4]}`), x))), OPEN)).toThrow("one to four distinct moves");
    expect(() => sheetFromChannel(withSheet((m) => m.map((x) => ((x[4] = `${x[4]},zz1,zz2,zz3,zz4`), x))), OPEN)).toThrow("one to four distinct moves");
    expect(() => sheetFromChannel(withSheet((m) => m.slice(1)), OPEN)).toThrow("flagged");
    const flagged = [...first.lines];
    flagged.splice(at, 0, "|-damage|p9a: Nobody|1/2");
    expect(() => sheetFromChannel({ ...first, lines: flagged } as PlayerChannel, OPEN)).toThrow("flagged");
  });

  it("is invariant to the opponent's Stat Points and reads the sheet (negative control: the nature)", async () => {
    const spec = open[0] as RandomBattleFile;
    const rest = (): { teams: RandomBattleFile["teams"] } => ({ teams: structuredClone(spec.teams) });
    const swapped = rest();
    for (const set of swapped.teams.p2) {
      const evs = { ...(set.evs ?? {}) } as Record<string, number>;
      [evs.atk, evs.def] = [evs.def ?? 0, evs.atk ?? 0];
      set.evs = evs as never;
    }
    const natured = rest();
    for (const set of natured.teams.p2) set.nature = set.nature === "Timid" ? "Jolly" : "Timid";
    const first = async (teams: RandomBattleFile["teams"]): Promise<string> => {
      const record = await playRandomBattle(engine, { ...spec, teams }, randomLegalActor, OPEN_SHEET_FORMAT);
      const lines = record.channels.p1;
      const upto = lines.findIndex((l) => l.startsWith("|request|")) + 1;
      return JSON.stringify(sheetFromChannel(playerChannel({ schema_version: 1, data_class: "synthetic", player: "p1", lines: lines.slice(0, upto) }, "open"), OPEN));
    };
    const base = await first(spec.teams);
    expect(await first(swapped.teams)).toBe(base);
    expect(await first(natured.teams)).not.toBe(base);
  });
});

describe("revealedOnlySheet (revealed-only/1)", () => {
  it("is a valid block whose known fields are the opponent's true ones and everything else unknown", () => {
    let revealed = 0;
    for (const battle of closed) {
      for (const player of ["p1", "p2"] as const) {
        const truth = battle.teams[other(player)];
        for (const channel of requestPrefixes(battle, player, "closed")) {
          const state = stateAt(channel.player, channel.lines);
          const block = readOpponentSheet(revealedOnlySheet(state.opponent));
          expect(block.source).toEqual({ kind: "revealed-only", hash: REVEALED_ONLY_HASH });
          expect(block.slots).toHaveLength(state.opponent.roster.length);
          for (const slot of block.slots) {
            expect(slot.brought_source).toBe("unknown");
            expect(slot.nature.source).toBe("unknown");
            const set = truth.find((t) => toId(t.species) === slot.species || toId(t.name ?? "") === slot.species);
            if (set === undefined || /-Mega/.test(set.species)) continue;
            if (slot.item.source === "revealed") {
              expect(slot.item.entries[0]?.id).toBe(toId(set.item ?? ""));
              revealed++;
            }
            if (slot.ability.source === "revealed") expect(slot.ability.entries[0]?.id).toBe(toId(set.ability));
            if (slot.moves.source === "revealed") expect(slot.moves.entries.map((e) => e.id)).toEqual(set.moves.map(toId).sort());
          }
        }
      }
    }
    expect(revealed).toBeGreaterThan(0);
  });

  it("is invariant to the opponent's unrevealed fields and a peeking builder differs (negative control)", async () => {
    const spec = closed[0] as RandomBattleFile;
    const changed = structuredClone(spec.teams);
    for (const set of changed.p2) {
      set.nature = set.nature === "Timid" ? "Jolly" : "Timid";
      const evs = { ...(set.evs ?? {}) } as Record<string, number>;
      [evs.atk, evs.def] = [evs.def ?? 0, evs.atk ?? 0];
      set.evs = evs as never;
    }
    const record = await playRandomBattle(engine, { ...spec, teams: changed }, randomLegalActor, GOLDEN_FORMAT);
    const mine = { ...spec.channels.p1 };
    const theirs = { ...spec.channels.p1, lines: record.channels.p1 };
    const blockAt = (lines: readonly string[], n: number): string => {
      return JSON.stringify(revealedOnlySheet(stateAt("p1", lines.slice(0, n)).opponent));
    };
    let compared = 0;
    mine.lines.forEach((line, i) => {
      if (!line.startsWith("|request|") || theirs.lines.slice(0, i + 1).some((l, j) => l !== mine.lines[j])) return;
      expect(blockAt(theirs.lines, i + 1)).toBe(blockAt(mine.lines, i + 1));
      compared++;
    });
    expect(compared).toBeGreaterThan(0);
    // A builder handed a real opponent field that the channel does not show gives another block.
    const real = stateAt("p1", mine.lines.slice(0, mine.lines.findIndex((l) => l.startsWith("|request|")) + 1)).opponent;
    const peeked = structuredClone(real);
    (peeked.roster[0] as { ability: string | null }).ability = toId((spec.teams.p2[0] as { ability: string }).ability);
    expect(JSON.stringify(revealedOnlySheet(peeked))).not.toBe(JSON.stringify(revealedOnlySheet(real)));
  });

  it("gives equal blocks for the A1 and A2 forms of every committed decision", () => {
    const pairs = JSON.parse(readFileSync(`${GOLDEN_DIR}/a2-pairs.json`, "utf8")) as {
      streams: { a1: string[][]; a2: string[][] };
      decisions: { seat: "p1" | "p2"; a1: { stream: number; length: number }; a2: { stream: number; length: number } }[];
    };
    const block = (lines: string[], seat: "p1" | "p2"): string => {
      return JSON.stringify(revealedOnlySheet(stateAt(seat, lines).opponent));
    };
    expect(pairs.decisions.length).toBeGreaterThan(10);
    for (const d of pairs.decisions) {
      const a1 = (pairs.streams.a1[d.a1.stream] as string[]).slice(0, d.a1.length);
      const a2 = (pairs.streams.a2[d.a2.stream] as string[]).slice(0, d.a2.length);
      expect(block(a2, d.seat)).toBe(block(a1, d.seat));
    }
  });

  it("refuses nothing it is given: a partial move list stays unknown, four revealed moves are revealed", () => {
    const base = { name: null, base_species: "x", species: null, volatiles: [] as string[], item_known: false, item: null, item_received: false, mega: false, mega_stone: null, ability: null, ability_received: false, moves_revealed: [] as string[] };
    const sheet = revealedOnlySheet({ roster: [{ ...base, moves_revealed: ["a", "b", "c"] }, { ...base, moves_revealed: ["d", "c", "b", "a"], item_known: true, item: "ix", ability: "ab" }, { ...base, volatiles: ["transform"], moves_revealed: ["a", "b", "c", "d"] }, { ...base, mega: true, mega_stone: "stone" }] as never });
    expect(sheet.slots[0]?.moves.source).toBe("unknown");
    expect(sheet.slots[1]?.moves.entries.map((e) => e.id)).toEqual(["a", "b", "c", "d"]);
    expect(sheet.slots[1]?.item.entries[0]?.id).toBe("ix");
    expect(sheet.slots[2]?.moves.source).toBe("unknown");
    expect(sheet.slots[3]?.item.entries[0]?.id).toBe("stone");
    // an empty roster is a block with no slot (the schema, not this builder, says six)
    expect(revealedOnlySheet({ roster: [] }).slots).toEqual([]);
  });
});

describe("what the builders can see", () => {
  it("takes a channel (or the reader's opponent half) and nothing of the opponent's team, the omniscient log, the record or a seed", () => {
    const never = (): void => {
      sheetFromChannel(channel, OPEN);
      // @ts-expect-error a third argument (the opponent's team) has no parameter
      sheetFromChannel(channel, OPEN, team);
      // @ts-expect-error an omniscient log is not a player's channel
      sheetFromChannel(log, OPEN);
      // @ts-expect-error a battle record is not a player's channel
      sheetFromChannel(record, OPEN);
      // @ts-expect-error a seed or a hidden verdict has no parameter
      sheetFromChannel(channel, OPEN, "sodium,00", record.hidden_verdicts);
      // @ts-expect-error the opponent half alone: no team
      revealedOnlySheet({ roster: [] }, team);
    };
    expect(typeof never).toBe("function");
  });

  it("imports no engine, runner, omniscient or seed module", () => {
    const source = readFileSync(new URL("../src/sheet-builders.ts", import.meta.url), "utf8");
    const imports = [...source.matchAll(/^import .* from "([^"]+)";$/gm)].map((m) => m[1]);
    expect(imports.sort()).toEqual(["./canonical-json.ts", "./observation.ts", "./rules.ts", "@areneon/public-state", "@areneon/schema/types/v1/opponent-sheet.ts", "node:crypto"].sort());
  });

  it("a TrainingExample carrying an oracle block is refused by the reader", () => {
    const example = JSON.parse(readFileSync(`${SAMPLES_DIR}/v1/training-example/invalid/opponent-sheet-oracle-kind.json`, "utf8")) as Json;
    expect(() => readTrainingExample(example)).toThrow();
  });
});
