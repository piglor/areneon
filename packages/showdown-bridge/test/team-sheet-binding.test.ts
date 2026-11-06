/* eslint-disable @typescript-eslint/no-non-null-assertion -- test fixtures: the committed corpus is known to be filled */
import { readdirSync, readFileSync } from "node:fs";
import { join, relative } from "node:path";
import type { PrivateTeamSpec } from "@areneon/schema/types/v1/private-team-spec.ts";
import { beforeAll, describe, expect, it } from "vitest";
import { goldenBattle, syntheticFixtureHashes } from "../src/golden.ts";
import {
  assertSheetBinding,
  type BattleRecord,
  type Episode,
  EpisodeError,
  episodeFromRecord,
  loadEngine,
  loadTrainingSet,
  ObservationError,
  playerChannel,
  readEpisode,
  recordedChannel,
  sheetRegimeIn,
  sheetRegimeOf,
  stampTeam,
  syntheticTeamHashes,
  teamHash,
  TrainingLoaderError,
} from "../src/index.ts";
import { loadOpenSheetCorpus, OPEN_SHEET_FORMAT } from "../src/open-sheet-corpus.ts";
import { loadRandomCorpus, playRandomBattle } from "../src/random-corpus.ts";
import { loadRulesSnapshot } from "../src/rules-load.ts";

const CLOSED_FORMAT = "gen9championsvgc2026regmc";
const toId = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]+/g, "");

const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const open = loadOpenSheetCorpus();
const closed = loadRandomCorpus();
const log = (player: "p1" | "p2", lines: readonly string[]) => ({ schema_version: 1, data_class: "synthetic", player, lines: [...lines] });

const SHEET = (side: string) => `|showteam|${side}|Altaria||Altarianite|NaturalCure|Substitute|Serious||F|||50|`;
const FIRST_REQUEST = `|request|{"teamPreview":true,"maxChosenTeamSize":4,"side":{"name":"p1","id":"p1","pokemon":[{"ident":"p1: Altaria","details":"Altaria, L50, F","condition":"154/154","active":true,"stats":{"atk":90,"def":110,"spa":90,"spd":125,"spe":80},"moves":["substitute"],"baseAbility":"naturalcure","item":"altarianite","pokeball":"pokeball","ability":"naturalcure"}]}}`;

describe("the format binding of |showteam| (ADR-027 section 5)", () => {
  it("reads the regime from the rules snapshot, never from an id written here", () => {
    const snapshot = loadRulesSnapshot(engine.commit);
    expect(sheetRegimeIn(snapshot, OPEN_SHEET_FORMAT)).toBe("open");
    expect(sheetRegimeIn(snapshot, CLOSED_FORMAT)).toBe("closed");
    expect(sheetRegimeIn(snapshot, "gen9nonexistent")).toBe("closed");
    expect(sheetRegimeOf(engine.commit, OPEN_SHEET_FORMAT)).toBe("open");
    expect(sheetRegimeOf(engine.commit, CLOSED_FORMAT)).toBe("closed");
    expect(sheetRegimeOf(engine.commit, "gen9nonexistent")).toBe("closed");
  });

  it("refuses every |showteam| line in a closed-format channel", () => {
    expect(() => assertSheetBinding(["|gametype|doubles", SHEET("p1")], "closed")).toThrow(/closed-sheet format's channel/);
    expect(() => assertSheetBinding(["|showteam"], "closed")).toThrow(ObservationError);
    expect(() => assertSheetBinding(["|gametype|doubles", "|start"], "closed")).not.toThrow();
    expect(() => assertSheetBinding(["|showteamy|p1"], "closed")).not.toThrow();
  });

  it("requires, in an open-format channel, exactly one sheet per side before Team Preview", () => {
    const ok = ["|gametype|doubles", SHEET("p1"), SHEET("p2"), FIRST_REQUEST];
    expect(() => assertSheetBinding(ok, "open")).not.toThrow();
    expect(() => assertSheetBinding(["|gametype|doubles", SHEET("p2"), SHEET("p1"), FIRST_REQUEST], "open")).not.toThrow();
    expect(() => assertSheetBinding(["|gametype|doubles", SHEET("p1"), FIRST_REQUEST], "open")).toThrow(/one \|showteam\| line per side/);
    expect(() => assertSheetBinding(["|gametype|doubles", FIRST_REQUEST], "open")).toThrow(/one \|showteam\| line per side/);
    expect(() => assertSheetBinding(["|gametype|doubles", SHEET("p1"), SHEET("p1"), FIRST_REQUEST], "open")).toThrow(/second \|showteam\| line/);
    expect(() => assertSheetBinding(["|gametype|doubles", SHEET("p3"), FIRST_REQUEST], "open")).toThrow(/names no side/);
    expect(() => assertSheetBinding(["|showteam", FIRST_REQUEST], "open")).toThrow(/names no side/);
    expect(() => assertSheetBinding(["|gametype|doubles", SHEET("p1"), SHEET("p2"), "|start", SHEET("p1")], "open")).toThrow(/second/);
    expect(() => assertSheetBinding(["|gametype|doubles", SHEET("p1"), "|start", SHEET("p2")], "open")).toThrow(/not before Team Preview/);
    expect(() => assertSheetBinding(["|gametype|doubles", SHEET("p1"), FIRST_REQUEST, SHEET("p2")], "open")).toThrow(/not before Team Preview/);
  });

  it("does not judge a cut that holds no request yet beyond the line rules", () => {
    expect(() => assertSheetBinding(["|gametype|doubles", SHEET("p1")], "open")).not.toThrow();
    expect(() => assertSheetBinding(["|gametype|doubles"], "open")).not.toThrow();
  });

  it("accepts the committed open-regime channels as open, and refuses them as closed (the default)", () => {
    for (const battle of open) {
      for (const player of ["p1", "p2"] as const) {
        expect(() => playerChannel(battle.channels[player], "open"), `${battle.id} ${player}`).not.toThrow();
        expect(() => playerChannel(battle.channels[player]), `${battle.id} ${player}`).toThrow(/closed-sheet format/);
        expect(() => playerChannel(battle.channels[player], "closed"), `${battle.id} ${player}`).toThrow(ObservationError);
      }
    }
  });

  it("refuses an open-format channel read as a closed game: a sheet missing, doubled or late", () => {
    const lines = open[0]!.channels.p1.lines;
    const sheets = lines.filter((line) => line.startsWith("|showteam|"));
    const without = (drop: string) => lines.filter((line) => line !== drop);
    expect(() => playerChannel(log("p1", without(sheets[1]!)), "open")).toThrow(/one \|showteam\| line per side/);
    expect(() => playerChannel(log("p1", [...lines.slice(0, 22), sheets[0]!, ...lines.slice(22)]), "open")).toThrow(/second \|showteam\| line/);
    expect(() => playerChannel(log("p1", [...without(sheets[1]!), sheets[1]!]), "open")).toThrow(/not before Team Preview/);
  });
});

describe("canaries: no |showteam| in a closed format", () => {
  it("the committed closed channels (golden, random-play) hold none", async () => {
    const golden = await goldenBattle(engine);
    for (const player of ["p1", "p2"] as const) {
      expect(golden.channels[player].some((line) => line.startsWith("|showteam"))).toBe(false);
    }
    for (const battle of closed) {
      for (const player of ["p1", "p2"] as const) {
        expect(battle.channels[player].lines.some((line) => line.startsWith("|showteam")), `${battle.id} ${player}`).toBe(false);
        expect(() => playerChannel(battle.channels[player], "closed")).not.toThrow();
      }
    }
  });

  it("a freshly played seeded closed-format game holds none, and its recorded channel binds the closed regime", async () => {
    const { id, seed, actor_seeds, teams } = closed[0]!;
    const record = await playRandomBattle(engine, { id, seed, actor_seeds, teams });
    expect(record.manifest.format_id).toBe(CLOSED_FORMAT);
    for (const player of ["p1", "p2"] as const) {
      expect(record.channels[player].some((line) => line.startsWith("|showteam"))).toBe(false);
      expect(() => recordedChannel(record, player, 1, "synthetic" as never)).not.toThrow();
    }
  }, 60_000);

  it("a spliced |showteam| line is refused by playerChannel", () => {
    const lines = [...closed[0]!.channels.p1.lines];
    const at = lines.findIndex((line) => line.startsWith("|request|"));
    lines.splice(at, 0, open[0]!.channels.p1.lines.find((line) => line.startsWith("|showteam|p2"))!);
    expect(() => playerChannel(log("p1", lines), "closed")).toThrow(/closed-sheet format's channel/);
    expect(() => playerChannel(log("p1", lines))).toThrow(/closed-sheet format's channel/);
  });
});

describe("harness only: the engine's sheet is the true team without Stat Points, IVs and nickname (N3)", () => {
  // This test reads both teams, as the omniscient tests do; no consumer does.
  it("equals the true team field for field, in team order, on every open-regime channel", () => {
    expect(open).toHaveLength(8);
    for (const battle of open) {
      for (const player of ["p1", "p2"] as const) {
        const lines = battle.channels[player].lines;
        for (const side of ["p1", "p2"] as const) {
          const line = lines.find((l) => l.startsWith(`|showteam|${side}|`))!;
          const packed = line.split("|").slice(3).join("|");
          const shown = packed.split("]").map((mon) => mon.split("|"));
          // The details of the preview (`|poke|`) give each Pokemon's gender, in preview order, which is team order.
          const preview = lines.filter((l) => l.startsWith(`|poke|${side}|`)).map((l) => l.split("|")[3]!.split(", "));
          const team = battle.teams[side];
          expect(shown, `${battle.id} ${player} ${side}`).toHaveLength(team.length);
          expect(preview.map((p) => toId(p[0]!))).toEqual(team.map((set) => toId(set.species)));
          for (const [k, set] of team.entries()) {
            const [name, species, item, ability, moves, nature, points, gender, ivs, shiny, level, extras] = shown[k]!;
            const details = preview[k]!;
            expect({
              name: toId(name!),
              species,
              item: toId(item!),
              ability: toId(ability!),
              moves: moves!.split(",").map(toId),
              nature,
              points,
              gender,
              ivs,
              shiny,
              level,
              extras,
            }).toEqual({
              name: toId(set.species),
              species: "", // an empty nickname is printed as the species name, leaving this field blank
              item: toId(set.item),
              ability: toId(set.ability),
              moves: set.moves.map(toId),
              nature: set.nature,
              points: "", // no Stat Points on the sheet
              gender: details.find((d) => d === "M" || d === "F") ?? "",
              ivs: "", // no IVs
              shiny: "",
              level: String(set.level),
              extras: "",
            });
            if (set.gender === "M" || set.gender === "F") expect(gender).toBe(set.gender);
          }
        }
      }
    }
  });
});

describe("the Episode and the training loader bind the format's regime", () => {
  let record: BattleRecord;
  let episode: Episode;
  let known: ReturnType<typeof syntheticTeamHashes>;
  const league = () => ({ kind: "league", run_id: "league-1", seed: record.seed, opponent_policy_id: "first-legal" }) as const;
  beforeAll(async () => {
    const { id, seed, actor_seeds, teams } = open[0]!;
    record = await playRandomBattle(engine, { id, seed, actor_seeds, teams }, undefined, OPEN_SHEET_FORMAT);
    known = syntheticTeamHashes([teamHash(teams.p1), teamHash(teams.p2)]);
    episode = await episodeFromRecord(record, "p1", {
      team: stampTeam(teams.p1 as PrivateTeamSpec["team"], known),
      opponent_team: stampTeam(teams.p2 as PrivateTeamSpec["team"], known),
      known_synthetic: known,
      behaviour_policy_id: "first-legal",
      collection: league(),
    });
  }, 120_000);

  const mutate = (change: (copy: Episode) => void): Episode => {
    const copy = structuredClone(episode);
    change(copy);
    return copy;
  };
  const sheetLines = (e: Episode) => e.channel.filter((line) => line.startsWith("|showteam|"));

  it("reads an open-format Episode that holds one sheet per side, and the loader trains on it", async () => {
    expect(episode.provenance.manifest.format_id).toBe(OPEN_SHEET_FORMAT);
    expect(sheetLines(episode)).toHaveLength(2);
    await expect(readEpisode(episode, { known_synthetic: known, engine })).resolves.toMatchObject({ player: "p1" });
    const { examples } = await loadTrainingSet({ episodes: [episode], evaluation_manifests: [], known_synthetic: known, behaviour_policy_id: "first-legal" });
    expect(examples.length).toBe(episode.decisions.length);
    expect(examples[0]!.observation.log.some((line) => line.startsWith("|showteam|"))).toBe(true);
  });

  it("refuses an open-format Episode with a sheet missing, doubled, or labelled a closed format", async () => {
    const second = sheetLines(episode)[1]!;
    const missing = mutate((e) => {
      e.channel = e.channel.filter((line) => line !== second) as Episode["channel"];
    });
    await expect(readEpisode(missing, { known_synthetic: known, engine })).rejects.toThrow(EpisodeError);
    await expect(readEpisode(missing, { known_synthetic: known, engine })).rejects.toThrow(/one \|showteam\| line per side/);
    const doubled = mutate((e) => {
      e.channel.splice(e.channel.indexOf(second), 0, second);
    });
    await expect(readEpisode(doubled, { known_synthetic: known, engine })).rejects.toThrow(/second \|showteam\| line/);
    const mislabelled = mutate((e) => {
      e.provenance.manifest.format_id = CLOSED_FORMAT;
    });
    await expect(readEpisode(mislabelled, { known_synthetic: known, engine })).rejects.toThrow(/closed-sheet format's channel/);
  });

  it("refuses a closed-format Episode with a spliced |showteam| line, in readEpisode and in the loader", async () => {
    const golden = await goldenBattle(engine);
    const goldenKnown = syntheticFixtureHashes(engine);
    const teams = (await import("../src/golden.ts")).syntheticTeam() as PrivateTeamSpec["team"];
    const stamped = stampTeam(teams, goldenKnown);
    const made = await episodeFromRecord(golden, "p1", {
      team: stamped,
      opponent_team: stamped,
      known_synthetic: goldenKnown,
      behaviour_policy_id: "first-legal",
      collection: { kind: "league", run_id: "league-1", seed: golden.seed, opponent_policy_id: "first-legal" },
    });
    await expect(readEpisode(made, { known_synthetic: goldenKnown, engine })).resolves.toBeDefined();
    const spliced = structuredClone(made);
    const at = spliced.channel.findIndex((line) => line.startsWith("|request|"));
    spliced.channel.splice(at, 0, sheetLines(episode)[0]!);
    await expect(readEpisode(spliced, { known_synthetic: goldenKnown, engine })).rejects.toThrow(/closed-sheet format's channel/);
    await expect(
      loadTrainingSet({ episodes: [spliced], evaluation_manifests: [], known_synthetic: goldenKnown, behaviour_policy_id: "first-legal" }),
    ).rejects.toThrow(TrainingLoaderError);
  });
});

describe("the true-team sheet test is reachable only from test code", () => {
  const root = join(import.meta.dirname, "..", "..", "..");
  const walk = (dir: string): string[] =>
    readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
      if (entry.name === "node_modules" || entry.name === "vendor" || entry.name === ".git") return [];
      const path = join(dir, entry.name);
      return entry.isDirectory() ? walk(path) : [path];
    });
  const shipped = [join(root, "packages"), join(root, "ml"), join(root, "scripts")]
    .flatMap(walk)
    .filter((path) => /\.(ts|cjs|py)$/.test(path) && !/(^|\/)(test|tests)\//.test(relative(root, path)) && !/\.test\./.test(path));

  it("no shipped source imports or names the test that reads both teams", () => {
    expect(shipped.length).toBeGreaterThan(100);
    for (const path of shipped) {
      expect(readFileSync(path, "utf8"), relative(root, path)).not.toMatch(/team-sheet-binding\.test|harness only: the engine's sheet/);
    }
  });

  it("the shipped binding module reads no team, corpus or true sheet: its imports are the rules, the observation error and the request decoder", () => {
    const text = readFileSync(join(root, "packages", "showdown-bridge", "src", "team-sheet-binding.ts"), "utf8");
    const imports = [...text.matchAll(/^import .* from "(.+)";$/gm)].map((m) => m[1]).sort();
    expect(imports).toEqual(["./observation.ts", "./request.ts", "./rules-load.ts", "./rules.ts"]);
  });
});
