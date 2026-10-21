import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { PrivateTeamSpec } from "@areneon/schema/types/v1/private-team-spec.ts";
import { beforeAll, describe, expect, it, vi } from "vitest";
import { type ChannelFixture, GOLDEN_DIR, GOLDEN_FORMAT, goldenBattle, syntheticFixtureHashes, syntheticTeam } from "../src/golden.ts";
import {
  assertOwnTeamStats,
  buildObservation,
  type Engine,
  EpisodeError,
  episodeFromRecord,
  loadEngine,
  nodeSha256,
  ObservationError,
  OwnTeamStatsError,
  playerChannel,
  readEpisode,
  stampTeam,
  type StampedTeam,
  teamHash,
} from "../src/index.ts";
import { resolveFormat } from "../src/formats.ts";
import { syntheticTeamHashes } from "../src/team-class.ts";
import { loadRandomCorpus } from "../src/random-corpus.ts";
import type { PokemonSet } from "../src/engine.ts";

let engine: Engine;
beforeAll(() => {
  engine = loadEngine();
});

const spec = (team: PokemonSet[]): PrivateTeamSpec => ({
  schema_version: 1,
  data_class: "synthetic",
  team: team as PrivateTeamSpec["team"],
});
const channelOf = (player: "p1" | "p2") =>
  playerChannel((JSON.parse(readFileSync(join(GOLDEN_DIR, "channels", `${player}.json`), "utf8")) as ChannelFixture).channel);

/** The golden team with one set changed. */
function mutated(change: (set: PokemonSet) => void, at = 2): PokemonSet[] {
  const team = structuredClone(syntheticTeam());
  change(team[at] as PokemonSet);
  return team;
}

describe("assertOwnTeamStats", () => {
  it("accepts the golden channels with the golden team, and every random-corpus channel with its own team", () => {
    for (const player of ["p1", "p2"] as const) {
      expect(() => assertOwnTeamStats(engine, GOLDEN_FORMAT, spec(syntheticTeam()), channelOf(player))).not.toThrow();
    }
    const corpus = loadRandomCorpus();
    expect(corpus.length).toBeGreaterThanOrEqual(40);
    for (const battle of corpus) {
      for (const player of ["p1", "p2"] as const) {
        expect(
          () => assertOwnTeamStats(engine, GOLDEN_FORMAT, spec(battle.teams[player]), playerChannel(battle.channels[player])),
          `${battle.id} ${player}`,
        ).not.toThrow();
      }
    }
  }, 120_000);

  // Each mirror differs from the golden team in one number only, so
  // `buildObservation` (species, item, ability, moves) still takes it: the
  // negative control that the old check cannot see it.
  const mirrors: [string, (set: PokemonSet) => void][] = [
    ["a Stat Point", (set) => void (set.evs.atk += 1)],
    ["the nature", (set) => void (set.nature = set.nature === "Timid" ? "Adamant" : "Timid")],
    ["the level", (set) => void (set.level = 49)],
  ];
  it.each(mirrors)("refuses a mirror team that differs in %s, naming only its position", async (_what, change) => {
    const team = mutated(change);
    const full = channelOf("p1");
    const channel = playerChannel({ ...full, lines: full.lines.slice(0, full.lines.findIndex((line) => line.startsWith("|request|")) + 1) });
    await expect(
      buildObservation({ channel, team: spec(team), manifest: resolveFormat(engine, GOLDEN_FORMAT).manifest, sha256: nodeSha256 }),
    ).resolves.toMatchObject({ kind: "preview" });
    let error: unknown;
    try {
      assertOwnTeamStats(engine, GOLDEN_FORMAT, spec(team), channel);
    } catch (caught) {
      error = caught;
    }
    expect(error).toBeInstanceOf(OwnTeamStatsError);
    expect(error).toBeInstanceOf(ObservationError);
    expect((error as OwnTeamStatsError).position).toBe(2);
    expect((error as Error).message).toBe(
      "the stats of the team's Pokémon at position 2 are not those of the player's Team Preview request",
    );
  });

  it.each(["hp", "atk", "def", "spa", "spd", "spe"] as const)(
    "does not see an IV (%s): the Champions stat formula ignores IVs, so the request cannot show one",
    (stat) => {
      const team = mutated((set) => void (set.ivs[stat] = 0));
      expect(() => assertOwnTeamStats(engine, GOLDEN_FORMAT, spec(team), channelOf("p1"))).not.toThrow();
    },
  );

  it("refuses the same team in another order, and the other player's team", () => {
    const reordered = [...syntheticTeam().slice(1), syntheticTeam()[0] as PokemonSet];
    expect(() => assertOwnTeamStats(engine, GOLDEN_FORMAT, spec(reordered), channelOf("p1"))).toThrow(OwnTeamStatsError);
    const other = loadRandomCorpus()[0] as ReturnType<typeof loadRandomCorpus>[number];
    expect(() => assertOwnTeamStats(engine, GOLDEN_FORMAT, spec(other.teams.p1), channelOf("p1"))).toThrow(OwnTeamStatsError);
  });

  it("refuses a team of another size, a channel past Team Preview and a team the engine cannot start", () => {
    expect(() => assertOwnTeamStats(engine, GOLDEN_FORMAT, spec(syntheticTeam().slice(0, 4)), channelOf("p1"))).toThrow(
      "differ in size",
    );
    const channel = channelOf("p1");
    const second = channel.lines.findIndex((line, i) => i > 0 && line.startsWith("|request|") && !line.includes("teamPreview"));
    const late = playerChannel({ ...channel, lines: channel.lines.slice(second) });
    expect(() => assertOwnTeamStats(engine, GOLDEN_FORMAT, spec(syntheticTeam()), late)).toThrow("does not start at Team Preview");
    const broken = syntheticTeam().map((set) => ({ ...set, moves: undefined })) as unknown as PokemonSet[];
    expect(() => assertOwnTeamStats(engine, GOLDEN_FORMAT, spec(broken), channelOf("p1"))).toThrow(OwnTeamStatsError);
    expect(() => assertOwnTeamStats(engine, GOLDEN_FORMAT, spec(syntheticTeam()), playerChannel({ ...channel, lines: [] }))).toThrow(
      "does not start at Team Preview",
    );
    const noPreview = { ...engine, sim: { ...engine.sim, Battle: class { sides = []; setPlayer(): void {} } } } as unknown as Engine;
    expect(() => assertOwnTeamStats(noPreview, GOLDEN_FORMAT, spec(syntheticTeam()), channel)).toThrow("sent no Team Preview");
  });

  it("never echoes the team: the message is fixed and holds a position only", () => {
    const names = syntheticTeam().flatMap((set) => [set.species, set.item, set.ability, ...set.moves]);
    for (const team of [mutated((set) => void (set.evs.def += 1)), syntheticTeam().slice(0, 3)]) {
      try {
        assertOwnTeamStats(engine, GOLDEN_FORMAT, spec(team), channelOf("p1"));
        expect.unreachable();
      } catch (error) {
        for (const name of names) expect((error as Error).message).not.toContain(name);
      }
    }
  });
});

describe("the pairing points", () => {
  it("readEpisode refuses a stored Episode whose team gives other stats", async () => {
    const record = await goldenBattle(engine);
    const known = syntheticFixtureHashes(engine);
    const team = stampTeam(syntheticTeam() as PrivateTeamSpec["team"], known);
    const episode = await episodeFromRecord(record, "p1", {
      team,
      opponent_team: team,
      known_synthetic: known,
      behaviour_policy_id: "first-legal@test",
    });
    await expect(readEpisode(episode, { known_synthetic: known })).resolves.toBeDefined();
    // Another engine object is used when given (a stand-in that must be asked).
    const asked: string[] = [];
    const spy = {
      ...engine,
      sim: {
        ...engine.sim,
        Battle: class extends engine.sim.Battle {
          constructor(options: { formatid: string; seed: string }) {
            super(options);
            asked.push(options.formatid);
          }
        },
      },
    } as Engine;
    await readEpisode(episode, { known_synthetic: known, engine: spy });
    expect(asked).toEqual([GOLDEN_FORMAT]);
    // The mirror team, known to be synthetic so only the stats can refuse it.
    const other = mutated((set) => void (set.evs.spa += 1)) as PrivateTeamSpec["team"];
    const forged = { ...structuredClone(episode), team: { ...episode.team, team: other } } as typeof episode;
    const knownMirror = syntheticTeamHashes([teamHash(other)]);
    await expect(readEpisode(forged, { known_synthetic: knownMirror })).rejects.toThrow(EpisodeError);
    await expect(readEpisode(forged, { known_synthetic: knownMirror })).rejects.toThrow("at position 2");
    expect(team satisfies StampedTeam).toBeDefined();
  });

  it("refuses an Episode of a format the repo does not support, before the stats check", async () => {
    const record = await goldenBattle(engine);
    const known = syntheticFixtureHashes(engine);
    const team = stampTeam(syntheticTeam() as PrivateTeamSpec["team"], known);
    const episode = await episodeFromRecord(record, "p1", { team, opponent_team: team, known_synthetic: known, behaviour_policy_id: "x" });
    const other = structuredClone(episode);
    other.provenance.manifest.format_id = "gen9ou";
    const asked: string[] = [];
    const spy = { ...engine, sim: { ...engine.sim, Battle: class extends engine.sim.Battle {
      constructor(options: { formatid: string; seed: string }) { super(options); asked.push(options.formatid); }
    } } } as Engine;
    await expect(readEpisode(other, { known_synthetic: known, engine: spy })).rejects.toThrow(
      "the Episode's format is not a supported format",
    );
    expect(asked).toEqual([]);
  });

  it("reports an unavailable engine as an infrastructure fault with a fixed message", async () => {
    vi.resetModules();
    vi.doMock("../src/engine.ts", async (original) => ({
      ...(await original<typeof import("../src/engine.ts")>()),
      loadEngine: () => {
        throw new Error("Showdown checkout is at /secret/path, expected pinned abc123");
      },
    }));
    try {
      const fresh = await import("../src/own-team-stats.ts");
      expect(() => fresh.pinnedEngine()).toThrow("the pinned Showdown engine is not available");
      const record = await goldenBattle(engine);
      const known = syntheticFixtureHashes(engine);
      const team = stampTeam(syntheticTeam() as PrivateTeamSpec["team"], known);
      const episode = await episodeFromRecord(record, "p1", { team, opponent_team: team, known_synthetic: known, behaviour_policy_id: "x" });
      const reader = await import("../src/episode-io.ts");
      const failure = await reader.readEpisode(episode, { known_synthetic: known }).catch((error: Error) => error);
      expect(failure).toBeInstanceOf(Error);
      expect((failure as Error).name).toBe("EngineError");
      expect((failure as Error).message).toBe("the pinned Showdown engine is not available");
    } finally {
      vi.doUnmock("../src/engine.ts");
      vi.resetModules();
    }
  });

  it("pins that the validator allows no Hidden Power, so a bare one cannot reach the move match", () => {
    const validator = engine.sim.TeamValidator.get(GOLDEN_FORMAT);
    for (const move of ["Hidden Power", "Hidden Power Fire"]) {
      const team = mutated((set) => void (set.moves = [move, "Protect"]), 0);
      const problems = validator.validateTeam(team) ?? [];
      expect(problems.some((problem) => problem.includes("does not exist in Gen 9")), move).toBe(true);
    }
  });
});
