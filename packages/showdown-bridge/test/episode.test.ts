import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { PrivateTeamSpec } from "@areneon/schema/types/v1/private-team-spec.ts";
import { beforeAll, describe, expect, it } from "vitest";
import { jcs, type Json } from "../src/canonical-json.ts";
import { firstLegal, GOLDEN_DIR, GOLDEN_FORMAT, goldenBattle, syntheticFixtureHashes, syntheticTeam } from "../src/golden.ts";
import {
  answersRequest,
  battleClass,
  type CollectionInput,
  type BattleRecord,
  buildObservation,
  type Episode,
  continueChannel,
  EpisodeError,
  episodeFromRecord,
  episodeHash,
  importPaste,
  loadEngine,
  nodeSha256,
  playerChannel,
  readEpisode,
  recordedChannel,
  type Recommendation,
  rightsFor,
  runBattle,
  stampTeam,
  type StampedTeam,
  teamHash,
  webSha256,
} from "../src/index.ts";

// Hand-built, so cast: a test of the run-time checks needs a label stampTeam
// would not give (a lie, or a class chosen by the test).
const spec = (team: PrivateTeamSpec["team"], data_class: PrivateTeamSpec["data_class"] = "synthetic"): StampedTeam =>
  ({
    schema_version: 1,
    data_class,
    team,
  }) as StampedTeam;
const synthetic = (): StampedTeam => spec(syntheticTeam() as PrivateTeamSpec["team"]);
const POLICY = "first-legal@test";

const KNOWN = syntheticFixtureHashes(loadEngine());
const both = (team: StampedTeam, opponent_team: StampedTeam = team) => ({
  team,
  opponent_team,
  known_synthetic: KNOWN,
  behaviour_policy_id: POLICY,
});
let golden: BattleRecord;
let episodes: Record<"p1" | "p2", Episode>;
beforeAll(async () => {
  golden = await goldenBattle(loadEngine());
  episodes = {
    p1: await episodeFromRecord(golden, "p1", both(synthetic())),
    p2: await episodeFromRecord(golden, "p2", both(synthetic())),
  };
});

describe("episodeFromRecord (A1)", () => {
  it("records one player's own channel, team and decisions, with provenance and rights", () => {
    for (const player of ["p1", "p2"] as const) {
      const episode = episodes[player];
      expect(episode).toMatchObject({
        schema_version: 1,
        data_class: "synthetic",
        rights: "cc0-synthetic",
        provenance: { runtime: "A1", behaviour_policy_id: POLICY, manifest: golden.manifest },
        player,
        result: { winner: golden.winner, turns: golden.turns },
      });
      expect(episode.channel).toEqual(golden.channels[player]);
      expect(episode.decisions.map((d) => d.choice)).toEqual(
        golden.choices.filter((c) => c.player === player).map((c) => c.choice),
      );
      expect(episode.decisions.every((d) => d.deadline === undefined && d.source_policy_id === POLICY)).toBe(true);
    }
  });

  it("rebuilds every decision's observation from the stored channel, with the decision's identity", async () => {
    const episode = episodes.p1;
    for (const decision of episode.decisions) {
      let seen = 0;
      const end = episode.channel.findIndex((l) => l.startsWith("|request|") && ++seen === decision.identity.request_seq);
      const observation = await buildObservation({
        channel: playerChannel({ schema_version: 1, data_class: "synthetic", player: "p1", lines: episode.channel.slice(0, end + 1) }),
        team: episode.team,
        manifest: episode.provenance.manifest,
        sha256: nodeSha256,
      });
      expect(observation.identity).toEqual(decision.identity);
    }
  });

  it("keeps a refused pending candidate's refusal, and makes a private team's Episode private", async () => {
    const record = structuredClone(golden);
    const choice = record.choices.find((c) => c.player === "p1" && c.request_seq > 1) as BattleRecord["choices"][number];
    choice.refused = "[Unavailable choice] Can't switch: The active Pokémon is trapped";
    const episode = await episodeFromRecord(
      record,
      "p1",
      both(spec(syntheticTeam() as PrivateTeamSpec["team"], "private"), synthetic()),
    );
    expect(episode.decisions.find((d) => d.identity.request_seq === choice.request_seq)?.refused).toBe(choice.refused);
    expect([episode.data_class, episode.rights]).toEqual(["private", "owner-private"]);
  });

  it("holds no secret of the opponent", async () => {
    const engine = loadEngine();
    const opponent = importPaste(
      engine,
      readFileSync(join(GOLDEN_DIR, "../../teams", GOLDEN_FORMAT, "legal-opponent.txt"), "utf8"),
    );
    const opponentSpec = spec(structuredClone(opponent) as PrivateTeamSpec["team"]);
    const record = await runBattle({
      engine,
      formatId: GOLDEN_FORMAT,
      seed: "sodium,0f0e0d0c0b0a09080706050403020100",
      teams: { p1: syntheticTeam(), p2: opponent },
      actors: { p1: firstLegal, p2: firstLegal },
    });
    const episode = await episodeFromRecord(record, "p1", both(synthetic(), opponentSpec));
    const bytes = jcs(episode as unknown as Json);
    const seen = record.channels.p1.join("\n");
    const secrets = opponentSpec.team.flatMap((set) => [
      jcs(set as unknown as Json),
      ...[set.item, ...set.moves].filter((name) => name !== "" && !seen.includes(name)),
    ]);
    expect(secrets.length).toBeGreaterThan(6);
    for (const secret of secrets) {
      expect(bytes.includes(secret) || bytes.includes(JSON.stringify(secret).slice(1, -1)), secret).toBe(false);
    }
    // And the opponent's team cannot be written into p1's Episode.
    await expect(episodeFromRecord(record, "p1", both(opponentSpec, synthetic()))).rejects.toThrow(
      /not the one p1 played/,
    );
  });
});

describe("episodeHash", () => {
  it("is the SHA-256 of the JCS bytes, the same through either port and on every run", async () => {
    const hash = await episodeHash(episodes.p1, nodeSha256);
    expect(hash).toMatch(/^[0-9a-f]{64}$/);
    expect(await episodeHash(episodes.p1, webSha256)).toBe(hash);
    expect(await episodeHash(structuredClone(episodes.p1), nodeSha256)).toBe(hash);
    expect(await episodeHash(episodes.p2, nodeSha256)).not.toBe(hash);
  });
});

describe("readEpisode", () => {
  const stored = (): Record<string, unknown> => JSON.parse(JSON.stringify(episodes.p1)) as Record<string, unknown>;

  it("reads back what was written", async () => {
    expect(await readEpisode(stored(), { known_synthetic: KNOWN })).toEqual(episodes.p1);
  });

  it.each<[string, (e: Record<string, unknown>) => unknown, RegExp]>([
    ["an unknown schema_version", (e) => ({ ...e, schema_version: 2 }), /no reader for an Episode with schema_version 2/],
    ["no schema_version", (e) => ({ ...e, schema_version: undefined }), /no reader for an Episode with schema_version undefined/],
    ["an unknown field", (e) => ({ ...e, omniscient_log: [] }), /invalid Episode/],
    ["rights that do not go with the class", (e) => ({ ...e, rights: "owner-private" }), /synthetic data cannot carry owner-private/],
    [
      "a private team in a synthetic Episode",
      (e) => ({ ...e, team: { ...(e.team as object), data_class: "private" } }),
      /an Episode with a private team is private/,
    ],
    [
      "a decision for a request the channel does not hold",
      (e) => ({ ...e, decisions: [{ identity: { request_seq: 999, request_hash: "a".repeat(64) }, choice: "x", source_policy_id: "p" }] }),
      /decision 999 does not answer that request/,
    ],
    [
      "a decision whose hash names another request",
      (e) => {
        const [first, ...rest] = e.decisions as { identity: { request_hash: string } }[];
        return { ...e, decisions: [{ ...first, identity: { ...first?.identity, request_hash: "b".repeat(64) } }, ...rest] };
      },
      /does not answer that request/,
    ],
    [
      "a decision at a wait request",
      (e) => {
        const channel = e.channel as string[];
        const waits = channel.filter((l) => l.startsWith("|request|"));
        const seq = waits.findIndex((l) => l.includes('"wait":true')) + 1;
        return { ...e, decisions: [{ identity: { request_seq: seq, request_hash: "c".repeat(64) }, choice: "x", source_policy_id: "p" }] };
      },
      /the last request is a wait request/,
    ],
    ["a broken request line", (e) => ({ ...e, channel: [...(e.channel as string[]), "|request|{bad"] }), /invalid Episode/],
    ["another player's channel", (e) => ({ ...e, channel: episodes.p2.channel }), /p1's channel holds a request for p2/],
    [
      "an unresolved split line",
      (e) => ({ ...e, channel: [...(e.channel as string[]).slice(0, 30), "|split|p2", ...(e.channel as string[]).slice(30)] }),
      /a line only the opponent may see: \|split\|p2/,
    ],
    [
      "the opponent's exact HP",
      (e) => {
        const channel = [...(e.channel as string[])];
        channel.splice(30, 0, "|switch|p2a: Snorlax|Snorlax, L50, M|236/236");
        return { ...e, channel };
      },
      /a line only the opponent may see/,
    ],
    [
      "the opponent's team, with no decisions",
      (e) => ({ ...e, decisions: [], team: { ...(e.team as object), team: (e.team as { team: { name: string }[] }).team.map((s) => ({ ...s, name: "Imposter" })) } }),
      /the team is not this player's/,
    ],
    [
      "decisions out of order",
      (e) => ({ ...e, decisions: [...(e.decisions as unknown[])].reverse() }),
      /out of request order/,
    ],
    [
      "a decision twice",
      (e) => ({ ...e, decisions: [(e.decisions as unknown[])[0], ...(e.decisions as unknown[])] }),
      /out of request order/,
    ],
    ["a result the channel does not show", (e) => ({ ...e, result: { winner: "p1", turns: 9999 } }), /is not the channel's/],
    [
      "a winner who is not one of the channel's players",
      (e) => ({ ...e, channel: (e.channel as string[]).map((l) => (l.startsWith("|win|") ? "|win|Mallory" : l)) }),
      /the channel's winner Mallory is not one of its players/,
    ],
    [
      "a channel cut before the battle ends",
      (e) => ({ ...e, channel: (e.channel as string[]).slice(0, 40), decisions: [] }),
      /the channel does not end the battle/,
    ],
    [
      "an A2 decision without its deadline record",
      (e) => ({ ...e, provenance: { ...(e.provenance as object), runtime: "A2" } }),
      /A2 decision \d+ lacks its deadline record or timer context/,
    ],
  ])("refuses %s", async (_label, tamper, error) => {
    await expect(readEpisode(tamper(stored()), { known_synthetic: KNOWN })).rejects.toThrow(error);
    await expect(readEpisode(tamper(stored()), { known_synthetic: KNOWN })).rejects.toBeInstanceOf(EpisodeError);
  });

  it("refuses a channel with no request", async () => {
    const e = stored();
    const lines = (e.channel as string[]).filter((l) => !l.startsWith("|request|"));
    await expect(readEpisode({ ...e, channel: lines, decisions: [] }, { known_synthetic: KNOWN })).rejects.toThrow(/holds no request/);
  });

  it("refuses a non-object, or one that is not plain data", async () => {
    await expect(readEpisode(null, { known_synthetic: KNOWN })).rejects.toThrow(EpisodeError);
    await expect(readEpisode({ ...stored(), toJSON: () => ({}) }, { known_synthetic: KNOWN })).rejects.toThrow(EpisodeError);
  });

  it("accepts an A2 Episode whose decisions carry their deadline evidence", async () => {
    const context = {
      timer_source: "server",
      timer_rules: { starting: 420, grace: 90, add_per_turn: 0, max_per_turn: 55, max_first_turn: 90, tick: 5 },
      turn_seconds_left: 55,
      bank_seconds_left: 420,
      grace_seconds_left: 0,
      safety_margin_ms: 3000,
      submit_within_ms: 20000,
      policy_budget_ms: 19000,
    };
    const deadline = {
      deadline_outcome: "in_budget",
      elapsed_ms: 1,
      submitted_after_ms: 2,
      bank_seconds_after: 420,
      server_auto_choose_detected: false,
    };
    const e = stored();
    const a2 = {
      ...e,
      provenance: { ...(e.provenance as object), runtime: "A2" },
      decisions: (e.decisions as object[]).map((d) => ({ ...d, decision_context: context, deadline })),
    };
    expect((await readEpisode(a2, { known_synthetic: KNOWN })).provenance.runtime).toBe("A2");
  });

  it("stores a battle with a reconnect as one continuous channel: the joined connections read, the rejoin channel alone does not (#813)", async () => {
    const e = stored();
    const lines = e.channel as string[];
    const requests = lines.flatMap((line, i) => (line.startsWith("|request|") ? [i] : []));
    const cut = requests[2] as number;
    const wrap = (l: string[]) => playerChannel({ schema_version: 1, data_class: "synthetic", player: "p1", lines: l });
    const withRqids = (l: string[]) => {
      let n = 0;
      return l.map((line) => (line.startsWith("|request|") ? `|request|${JSON.stringify({ ...JSON.parse(line.slice(9)), rqid: ++n })}` : line));
    };
    // Connection 1 ends at the third request; the rejoin replays the room log and sends that request again.
    const before = withRqids(lines.slice(0, cut + 1));
    const again = before.at(-1) as string;
    const rest = lines.slice(cut + 1).map((line) => (line.startsWith("|request|") ? `|request|${JSON.stringify({ ...JSON.parse(line.slice(9)), rqid: 99 })}` : line));
    const rejoin = [...before.slice(0, -1).filter((line) => !line.startsWith("|request|")), again, ...rest];
    const joined = continueChannel(wrap(before), wrap(rejoin)).lines;
    expect(joined.filter((line) => !line.startsWith("|request|"))).toEqual(lines.filter((line) => !line.startsWith("|request|")));
    expect(joined.filter((line) => line.startsWith("|request|"))).toHaveLength(requests.length);
    expect((await readEpisode({ ...e, channel: joined }, { known_synthetic: KNOWN })).channel).toEqual(joined);
    await expect(readEpisode({ ...e, channel: rejoin }, { known_synthetic: KNOWN })).rejects.toThrow(EpisodeError);
  });

  it("returns a copy: changing the record or the input later changes no Episode", async () => {
    const record = structuredClone(golden);
    const episode = await episodeFromRecord(record, "p1", both(synthetic()));
    const before = await episodeHash(episode, nodeSha256);
    record.channels.p1.push("|tampered");
    record.manifest.format_id = "tampered";
    expect(await episodeHash(episode, nodeSha256)).toBe(before);
    const input = stored();
    const read = await readEpisode(input, { known_synthetic: KNOWN });
    (input.channel as string[]).push("|tampered");
    expect(read.channel).not.toContain("|tampered");
  });
});

describe("answersRequest (stale invalidation, ADR-008 §4)", () => {
  const identity = { request_seq: 3, request_hash: "d".repeat(64) };
  const recommendation = (extra: Partial<Recommendation["identity"]> = {}): Recommendation => ({
    schema_version: 1,
    identity: { ...identity, ...extra },
    choice: "move 1 1, move 1 2",
    source_policy_id: "first-legal@test",
    elapsed_ms: 12,
    deadline_outcome: "in_budget",
  });

  it.each<[string, Recommendation, Recommendation["identity"], boolean]>([
    ["the same request", recommendation(), identity, true],
    ["a later request_seq", recommendation(), { ...identity, request_seq: 4 }, false],
    ["another request_hash", recommendation(), { ...identity, request_hash: "e".repeat(64) }, false],
    ["the same rqid on both", recommendation({ rqid: 7 }), { ...identity, rqid: 7 }, true],
    ["another rqid", recommendation({ rqid: 7 }), { ...identity, rqid: 8 }, false],
    ["an rqid on one side only", recommendation({ rqid: 7 }), identity, true],
    ["an rqid only on the current request", recommendation(), { ...identity, rqid: 7 }, true],
  ])("%s", (_label, rec, current, answers) => {
    expect(answersRequest(rec, current)).toBe(answers);
  });

  it("maps each data class to its rights", () => {
    expect([rightsFor("synthetic"), rightsFor("private")]).toEqual(["cc0-synthetic", "owner-private"]);
  });
});

describe("an Episode takes the higher data class of both teams (#817)", () => {
  const engine = loadEngine();
  const realOpponent = () =>
    importPaste(engine, readFileSync(join(GOLDEN_DIR, "../../teams", GOLDEN_FORMAT, "legal-opponent.txt"), "utf8"));
  const play = (p2: ReturnType<typeof realOpponent>) =>
    runBattle({
      engine,
      formatId: GOLDEN_FORMAT,
      seed: "sodium,0f0e0d0c0b0a09080706050403020100",
      teams: { p1: syntheticTeam(), p2 },
      actors: { p1: firstLegal, p2: firstLegal },
    });
  // A team of the owner's that is in no synthetic set: its sets differ from every fixture.
  const privateOpponent = () => realOpponent().map((set, i) => (i === 0 ? { ...set, nature: "Quirky" } : set));

  it("is private, with owner-private rights, when only the opponent's team is", async () => {
    const record = await play(privateOpponent());
    expect(KNOWN.has(teamHash(privateOpponent()))).toBe(false);
    const opponent_team = stampTeam(privateOpponent() as PrivateTeamSpec["team"], KNOWN);
    expect(opponent_team.data_class).toBe("private");
    for (const player of ["p1", "p2"] as const) {
      const own = player === "p1" ? synthetic() : opponent_team;
      const rival = player === "p1" ? opponent_team : synthetic();
      const episode = await episodeFromRecord(record, player, both(own, rival));
      expect([episode.data_class, episode.rights]).toEqual(["private", "owner-private"]);
    }
  });

  it("is synthetic only when both teams are", async () => {
    const record = await play(realOpponent());
    const opponent_team = stampTeam(realOpponent() as PrivateTeamSpec["team"], KNOWN);
    expect(opponent_team.data_class).toBe("synthetic");
    const episode = await episodeFromRecord(record, "p1", both(synthetic(), opponent_team));
    expect([episode.data_class, episode.rights]).toEqual(["synthetic", "cc0-synthetic"]);
  });

  it("refuses a private team labelled synthetic by the caller", async () => {
    const record = await play(privateOpponent());
    const lie = spec(privateOpponent() as PrivateTeamSpec["team"], "synthetic");
    await expect(episodeFromRecord(record, "p1", both(synthetic(), lie))).rejects.toThrow(/not a known synthetic team/);
    await expect(episodeFromRecord(record, "p2", both(lie, synthetic()))).rejects.toThrow(/not a known synthetic team/);
  });

  it("takes only a stamped team: a hand-built spec does not compile", () => {
    const handBuilt: PrivateTeamSpec = { schema_version: 1, data_class: "synthetic", team: syntheticTeam() as PrivateTeamSpec["team"] };
    // @ts-expect-error a PrivateTeamSpec is not a StampedTeam: only stampTeam returns one
    const unstamped: StampedTeam = handBuilt;
    expect(unstamped).toBe(handBuilt);
  });

  it("refuses a team that is not the one the battle ran with", async () => {
    const record = await play(realOpponent());
    await expect(episodeFromRecord(record, "p1", both(synthetic(), synthetic()))).rejects.toThrow(
      /not the one p2 played/,
    );
  });

  it("records each seat's team hash, of the set array alone", async () => {
    const record = await play(realOpponent());
    expect(record.team_hashes).toEqual({ p1: teamHash(syntheticTeam()), p2: teamHash(realOpponent()) });
    expect(teamHash(spec(syntheticTeam() as PrivateTeamSpec["team"], "private").team)).toBe(record.team_hashes.p1);
  });

  it("labels the channel and its observations by the battle's class", async () => {
    const record = await play(privateOpponent());
    const known = battleClass(record.team_hashes, KNOWN);
    expect(known).toBe("private");
    expect(battleClass(golden.team_hashes, KNOWN)).toBe("synthetic");
    expect(battleClass({ p1: "unknown", p2: golden.team_hashes.p2 }, KNOWN)).toBe("private");
    const channel = recordedChannel(record, "p1", 1, known);
    expect(channel.data_class).toBe("private");
    const observation = await buildObservation({
      channel,
      team: synthetic(),
      manifest: record.manifest,
      sha256: nodeSha256,
    });
    expect(observation.data_class).toBe("private");
  });

  it("readEpisode refuses a synthetic label the known set does not prove", async () => {
    const stored = JSON.parse(JSON.stringify(episodes.p1)) as Record<string, unknown>;
    await expect(readEpisode(stored, { known_synthetic: KNOWN })).resolves.toBeDefined();
    await expect(readEpisode(stored, { known_synthetic: new Set() })).rejects.toThrow(/not a known synthetic team/);
  });
});

describe("the caller's team after a battle (#817)", () => {
  it("still hashes to the record, so the same object can go to episodeFromRecord", async () => {
    const engine = loadEngine();
    const own = syntheticTeam();
    const rival = importPaste(
      engine,
      readFileSync(join(GOLDEN_DIR, "../../teams", GOLDEN_FORMAT, "legal-opponent.txt"), "utf8"),
    );
    const record = await runBattle({
      engine,
      formatId: GOLDEN_FORMAT,
      seed: "sodium,0f0e0d0c0b0a09080706050403020100",
      teams: { p1: own, p2: rival },
      actors: { p1: firstLegal, p2: firstLegal },
    });
    expect(teamHash(own)).toBe(record.team_hashes.p1);
    expect(teamHash(rival)).toBe(record.team_hashes.p2);
    const episode = await episodeFromRecord(
      record,
      "p1",
      both(spec(own as PrivateTeamSpec["team"]), spec(rival as PrivateTeamSpec["team"])),
    );
    expect(episode.data_class).toBe("synthetic");
  });
});

describe("provenance.collection (ADR-009 §4)", () => {
  const LEAGUE = (): CollectionInput => ({ kind: "league", run_id: "run-1", seed: golden.seed, opponent_policy_id: "first-legal@test" });
  const EVALUATION = (): CollectionInput => ({ kind: "evaluation", run_id: "eval-1", sealed: true, seed: golden.seed });
  // The stored form: the writer adds the opponent's class.
  const STORED = (input: CollectionInput, opponent_team_data_class: "synthetic" | "private" = "synthetic") => ({
    ...input,
    opponent_team_data_class,
  });
  const withCollection = (
    collection: unknown,
    runtime: "A1" | "A2" = "A1",
    base: Episode = episodes.p1,
  ): Record<string, unknown> => {
    const e = JSON.parse(JSON.stringify(base)) as Record<string, unknown>;
    e.provenance = { ...(e.provenance as object), runtime, collection };
    if (runtime === "A2") e.decisions = [];
    return e;
  };

  it("is optional: an Episode without it stays valid and carries none", () => {
    expect(episodes.p1.provenance).not.toHaveProperty("collection");
  });

  it.each([["league", LEAGUE], ["evaluation", EVALUATION]])("is written on a %s Episode and read back", async (_kind, make) => {
    const episode = await episodeFromRecord(golden, "p1", { ...both(synthetic()), collection: make() });
    expect(episode.provenance.collection).toEqual(STORED(make()));
    expect(await readEpisode(JSON.parse(JSON.stringify(episode)), { known_synthetic: KNOWN })).toEqual(episode);
  });

  it("is refused when its seed is not the battle's own", async () => {
    await expect(
      episodeFromRecord(golden, "p1", { ...both(synthetic()), collection: { ...LEAGUE(), seed: "sodium,00" + "0".repeat(30) } as CollectionInput }),
    ).rejects.toThrow("the collection record's seed is not the seed the battle ran with");
  });

  it("takes a private sparring record without a seed, and refuses a synthetic one", async () => {
    const sparring = { kind: "sparring", session_id: "s-1" } as const;
    const episode = await episodeFromRecord(golden, "p1", {
      ...both(spec(syntheticTeam() as PrivateTeamSpec["team"], "private"), synthetic()),
      collection: sparring,
    });
    expect(episode.provenance.collection).toEqual(sparring);
    expect(episode.data_class).toBe("private");
    expect((await readEpisode(withCollection(sparring, "A1", episode), { known_synthetic: KNOWN })).provenance.collection).toEqual(sparring);
    await expect(readEpisode(withCollection(sparring), { known_synthetic: KNOWN })).rejects.toThrow(
      "a sparring Episode is private: it records a battle against a human",
    );
  });

  const C = { opponent_team_data_class: "synthetic" };
  it.each<[string, unknown, RegExp]>([
    ["a league record without its run id", { ...C, kind: "league", seed: "sodium,ab", opponent_policy_id: "p" }, /invalid Episode.*run_id/s],
    ["an evaluation record without its run id", { ...C, kind: "evaluation", sealed: false, seed: "sodium,ab" }, /invalid Episode.*run_id/s],
    ["an evaluation record without its sealed flag", { ...C, kind: "evaluation", run_id: "r", seed: "sodium,ab" }, /invalid Episode.*sealed/s],
    ["sealed on a league record", { ...C, kind: "league", run_id: "r", seed: "sodium,ab", opponent_policy_id: "p", sealed: true }, /invalid Episode.*sealed/s],
    ["a league record without the opponent's class", { kind: "league", run_id: "r", seed: "sodium,ab", opponent_policy_id: "p" }, /invalid Episode.*opponent_team_data_class/s],
    ["an evaluation record without the opponent's class", { kind: "evaluation", run_id: "r", sealed: true, seed: "sodium,ab" }, /invalid Episode.*opponent_team_data_class/s],
    ["an opponent class that is not a class", { ...C, kind: "league", run_id: "r", seed: "sodium,ab", opponent_policy_id: "p", opponent_team_data_class: "public" }, /invalid Episode.*opponent_team_data_class/s],
    ["a sparring record without its session id", { kind: "sparring" }, /invalid Episode.*session_id/s],
    ["an unknown kind", { kind: "training", run_id: "r" }, /invalid Episode/],
    ["a seed that is not an engine seed", { ...C, kind: "league", run_id: "r", seed: "1,2", opponent_policy_id: "p" }, /invalid Episode.*pattern/s],
  ])("refuses %s", async (_name, collection, message) => {
    await expect(readEpisode(withCollection(collection), { known_synthetic: KNOWN })).rejects.toThrow(message);
  });

  it.each(["league", "evaluation"] as const)("refuses an A1 %s record with a null seed", async (kind) => {
    const base = kind === "league" ? LEAGUE() : EVALUATION();
    await expect(readEpisode(withCollection({ ...STORED(base), seed: null }), { known_synthetic: KNOWN })).rejects.toThrow(
      `an A1 ${kind} collection record needs the seed the battle ran with`,
    );
  });

  it.each(["league", "evaluation"] as const)("refuses an A2 %s record with a seed, and accepts none", async (kind) => {
    const base = kind === "league" ? LEAGUE() : EVALUATION();
    await expect(readEpisode(withCollection(STORED(base), "A2"), { known_synthetic: KNOWN })).rejects.toThrow(
      `an A2 ${kind} collection record has no seed: a challenge cannot be seeded`,
    );
    expect((await readEpisode(withCollection({ ...STORED(base), seed: null }, "A2"), { known_synthetic: KNOWN })).provenance.collection).toEqual({
      ...STORED(base),
      seed: null,
    });
  });
});

describe("the opponent's class on the collection record (ADR-009 §3, §4, #826)", () => {
  const engine = loadEngine();
  const realOpponent = () =>
    importPaste(engine, readFileSync(join(GOLDEN_DIR, "../../teams", GOLDEN_FORMAT, "legal-opponent.txt"), "utf8"));
  const privateOpponent = () => realOpponent().map((set, i) => (i === 0 ? { ...set, nature: "Quirky" } : set));
  const play = (p2: ReturnType<typeof realOpponent>) =>
    runBattle({
      engine,
      formatId: GOLDEN_FORMAT,
      seed: "sodium,0f0e0d0c0b0a09080706050403020100",
      teams: { p1: syntheticTeam(), p2 },
      actors: { p1: firstLegal, p2: firstLegal },
    });
  const league = (record: BattleRecord): CollectionInput => ({
    kind: "league",
    run_id: "run-1",
    seed: record.seed,
    opponent_policy_id: "first-legal@test",
  });
  const stampedOpponent = () => stampTeam(privateOpponent() as PrivateTeamSpec["team"], KNOWN);
  const stored = (episode: Episode) => JSON.parse(JSON.stringify(episode)) as Record<string, unknown>;

  it.each(["league", "evaluation"] as const)(
    "is set by the writer from the opponent's stamped team on a %s record",
    async (kind) => {
      const record = await play(privateOpponent());
      const collection: CollectionInput =
        kind === "league" ? league(record) : { kind, run_id: "e", sealed: false, seed: record.seed };
      const priv = await episodeFromRecord(record, "p1", { ...both(synthetic(), stampedOpponent()), collection });
      expect(priv.provenance.collection).toMatchObject({ opponent_team_data_class: "private" });
      expect(priv.data_class).toBe("private");
      const plain = await play(realOpponent());
      const syn = await episodeFromRecord(plain, "p1", {
        ...both(synthetic(), stampTeam(realOpponent() as PrivateTeamSpec["team"], KNOWN)),
        collection: { ...collection, seed: plain.seed } as CollectionInput,
      });
      expect(syn.provenance.collection).toMatchObject({ opponent_team_data_class: "synthetic" });
      expect(syn.data_class).toBe("synthetic");
    },
  );

  it("refuses a stored private-opponent Episode relabelled synthetic (the case the reader could not catch before)", async () => {
    const record = await play(privateOpponent());
    const episode = await episodeFromRecord(record, "p1", {
      ...both(synthetic(), stampedOpponent()),
      collection: league(record),
    });
    expect(episode.data_class).toBe("private");
    await expect(readEpisode(stored(episode), { known_synthetic: KNOWN })).resolves.toBeDefined();
    const relabelled = { ...stored(episode), data_class: "synthetic", rights: "cc0-synthetic" };
    await expect(readEpisode(relabelled, { known_synthetic: KNOWN })).rejects.toThrow(
      "the Episode is labelled synthetic, but its own team is synthetic and its opponent's team is private: the class must be private",
    );
  });

  it("refuses an Episode whose class is not the higher of the two, above it too", async () => {
    const record = await play(realOpponent());
    const episode = await episodeFromRecord(record, "p1", {
      ...both(synthetic(), stampTeam(realOpponent() as PrivateTeamSpec["team"], KNOWN)),
      collection: league(record),
    });
    const above = { ...stored(episode), data_class: "private", rights: "owner-private" };
    await expect(readEpisode(above, { known_synthetic: KNOWN })).rejects.toThrow(
      "the Episode is labelled private, but its own team is synthetic and its opponent's team is synthetic: the class must be synthetic",
    );
  });

  it("an observation of a mixed battle is private for either seat, at a preview and at a move request", async () => {
    const record = await play(privateOpponent());
    const class_ = battleClass(record.team_hashes, KNOWN);
    for (const player of ["p1", "p2"] as const) {
      const own = player === "p1" ? synthetic() : stampedOpponent();
      expect(record.requests[player][0]?.phase).toBe("preview");
      for (const request of [record.requests[player][0], record.requests[player].find((r) => r.phase === "move")]) {
        const observation = await buildObservation({
          channel: recordedChannel(record, player, (request as { request_seq: number }).request_seq, class_),
          team: own,
          manifest: record.manifest,
          sha256: nodeSha256,
        });
        expect(observation.data_class).toBe("private");
      }
    }
  });
});
