import {
  type CollectionInput,
  EpisodeError,
  loadEngine,
  loadRulesSnapshot,
  readEpisode,
  resolveFormat,
  stampTeam,
  type StampedTeam,
  teamHash,
} from "@areneon/showdown-bridge";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import {
  type A2BattleRecord,
  a2Episode,
  A2EpisodeError,
  type A2EpisodeOptions,
  type DecisionRecord,
  timerRulesOf,
} from "../src/index.ts";
import { fixtureTeam, FORMAT, playRecordedBattle, startLocalServer } from "./local-battle.ts";

// The A2 Episode writer (ADR-009 slice f) on a real battle: the pinned local
// server, bot against bot, the bot's own record out of runA2Bot. Failure cases
// edit a copy of that record.

const engine = loadEngine();
const snapshot = loadRulesSnapshot(engine.commit);
const own = fixtureTeam("legal-synthetic.txt");
const rival = fixtureTeam("legal-opponent.txt");
const KNOWN = new Set([teamHash(own), teamHash(rival)]);
const stamp = (team: typeof own, known = KNOWN): StampedTeam => stampTeam(team as StampedTeam["team"], known);
const manifest = resolveFormat(engine, FORMAT, snapshot).manifest;
const SPARRING: CollectionInput = { kind: "sparring", session_id: "a2-test" };
const LEAGUE: CollectionInput = { kind: "league", run_id: "run-1", seed: null, opponent_policy_id: "actor-rule@a2" };

const options = (extra: Partial<A2EpisodeOptions> = {}): A2EpisodeOptions => ({
  team: stamp(own),
  known_synthetic: KNOWN,
  manifest,
  timer_rules: timerRulesOf(engine, FORMAT, 5),
  behaviour_policy_id: "actor-rule@a2",
  collection: SPARRING,
  ...extra,
});

let local: Awaited<ReturnType<typeof startLocalServer>>;
let mine: A2BattleRecord;
let theirs: A2BattleRecord;

beforeAll(async () => {
  local = await startLocalServer();
  ({ mine, theirs } = await playRecordedBattle(local));
}, 90_000);
afterAll(async () => {
  await local?.stop();
});

/** A copy of the bot's record, edited. */
const edited = (change: (record: A2BattleRecord) => void): A2BattleRecord => {
  const copy = structuredClone(mine);
  change(copy);
  return copy;
};
const decision = (record: A2BattleRecord, i = 0): DecisionRecord => record.decisions[i] as DecisionRecord;

describe("a2Episode on a battle of the local server", () => {
  it("writes the bot's own Episode of a sparring battle: private, with every decision's deadline record", async () => {
    const episode = await a2Episode(mine, options());
    expect(episode).toMatchObject({
      schema_version: 1,
      data_class: "private",
      rights: "owner-private",
      player: mine.player,
      provenance: { runtime: "A2", behaviour_policy_id: "actor-rule@a2", manifest, collection: SPARRING },
    });
    expect(episode.decisions).toHaveLength(mine.decisions.length);
    expect(episode.channel).toEqual(mine.channels[0]?.lines);
    mine.decisions.forEach((d, i) => {
      const written = episode.decisions[i];
      expect(written?.identity).toEqual({ request_seq: d.channel_request_seq, request_hash: d.request_hash, rqid: d.rqid });
      expect(written?.choice).toBe(d.choice);
      expect(written?.deadline).toMatchObject({ deadline_outcome: "in_budget", server_auto_choose_detected: false });
      expect(written?.decision_context).toMatchObject({
        timer_source: "server",
        safety_margin_ms: 3000,
        submit_within_ms: d.budget.submit_within_ms,
        policy_budget_ms: d.budget.policy_budget_ms,
      });
    });
    expect(episode.decisions.at(-1)?.deadline?.bank_seconds_after).toBeNull();
    const result = mine.result === "tie" ? null : mine.result === "win" ? mine.player : mine.player === "p1" ? "p2" : "p1";
    expect(episode.result.winner).toBe(result);
    // A stored copy reads back as it was written (the team under the names the server gave it),
    // and the other bot's Episode is its own.
    const played = new Set([...KNOWN, teamHash(episode.team.team)]);
    expect(await readEpisode(JSON.parse(JSON.stringify(episode)), { known_synthetic: played })).toEqual(episode);
    const other = await a2Episode(theirs, options({ team: stamp(rival) }));
    expect(other.player).not.toBe(episode.player);
    expect(other.team.team.map((set) => set.species)).toEqual(rival.map((set) => set.species));
    expect(JSON.stringify(episode)).not.toContain(rival.flatMap((set) => set.moves).find((m) => !episode.channel.join().includes(m)) ?? "\u0000");
  });

  describe("nicknames (pinned independent of the server's behaviour)", () => {
    const NICKS = ["Alpha", "Bravo", "Cinder", "Delta", "Echo", "Foxtrot"];
    const nicknamed = own.map((set, i) => ({ ...set, name: NICKS[i] as string }));
    /** The bot's record with its Team Preview request listing the team under `names`; no decisions (their hashes name the original request). */
    const named = (names: readonly string[]): A2BattleRecord =>
      edited((r) => {
        r.decisions = [];
        const lines = (r.channels[0] as { lines: string[] }).lines;
        const at = lines.findIndex((line) => line.startsWith("|request|"));
        const request = JSON.parse((lines[at] as string).slice("|request|".length)) as { side: { pokemon: { ident: string }[] } };
        request.side.pokemon.forEach((mon, i) => (mon.ident = `${mine.player}: ${names[i]}`));
        lines[at] = `|request|${JSON.stringify(request)}`;
      });
    const species = own.map((set) => set.species);
    const known = new Set([teamHash(nicknamed)]);
    const write = (record: A2BattleRecord) =>
      a2Episode(record, options({ team: stamp(nicknamed, known), known_synthetic: known, collection: LEAGUE, opponent_team: stamp(rival) }));

    it("takes the species names the channel shows when the server removed the nicknames, and keeps the proof", async () => {
      const episode = await write(named(species));
      expect(episode.team.team.map((set) => set.name)).toEqual(species);
      expect(episode.data_class).toBe("synthetic");
      const stored = JSON.parse(JSON.stringify(episode));
      // The played form is a different set of sets: a reader needs its hash too.
      await expect(readEpisode(stored, { known_synthetic: known })).rejects.toThrow(/not a known synthetic team/);
      expect(await readEpisode(stored, { known_synthetic: new Set([...known, teamHash(episode.team.team)]) })).toEqual(episode);
    });

    it("keeps the nicknames the channel shows when the server kept them", async () => {
      const episode = await write(named(NICKS));
      expect(episode.team.team.map((set) => set.name)).toEqual(NICKS);
      expect(episode.data_class).toBe("synthetic");
      expect(await readEpisode(JSON.parse(JSON.stringify(episode)), { known_synthetic: known })).toEqual(episode);
    });

    it("does not carry a class over for a paste that is not known synthetic", async () => {
      await expect(
        a2Episode(named(species), options({ team: stamp(nicknamed, known), known_synthetic: new Set(), collection: LEAGUE })),
      ).rejects.toThrow(/not a known synthetic team/);
    });
  });

  it("a league Episode of two known synthetic teams is synthetic, and records the opponent's class", async () => {
    const episode = await a2Episode(mine, options({ collection: LEAGUE, opponent_team: stamp(rival) }));
    expect(episode).toMatchObject({ data_class: "synthetic", rights: "cc0-synthetic" });
    expect(episode.provenance.collection).toEqual({ ...LEAGUE, opponent_team_data_class: "synthetic" });
  });

  it("writes an A2 league or evaluation Episode as private when no opponent class is derivable", async () => {
    for (const collection of [LEAGUE, { kind: "evaluation", run_id: "ev-1", sealed: false, seed: null }] as const) {
      const episode = await a2Episode(mine, options({ collection }));
      expect(episode).toMatchObject({ data_class: "private", rights: "owner-private" });
      expect(episode.provenance.collection).toMatchObject({ opponent_team_data_class: "private" });
    }
  });

  it("is private when the opponent's team is private, and when its own team is", async () => {
    const privateRival = stamp(rival, new Set());
    expect((await a2Episode(mine, options({ collection: LEAGUE, opponent_team: privateRival }))).data_class).toBe("private");
    const none = new Set<string>();
    expect((await a2Episode(mine, options({ team: stamp(own, none), known_synthetic: none, collection: LEAGUE, opponent_team: stamp(rival) }))).data_class).toBe("private");
  });

  it("refuses what the reader refuses: a label that is not proven, a seed on A2, another team", async () => {
    await expect(a2Episode(mine, options({ known_synthetic: new Set() }))).rejects.toThrow(EpisodeError);
    await expect(a2Episode(mine, options({ collection: { ...LEAGUE, seed: "sodium,00ff" } }))).rejects.toThrow(/no seed/);
    await expect(a2Episode(mine, options({ team: stamp(rival) }))).rejects.toThrow(EpisodeError);
    await expect(a2Episode(edited((r) => (r.result = "tie")), options())).rejects.toThrow(/not the channel's/);
    await expect(a2Episode(edited((r) => ((r.channels[0] as { lines: string[] }).lines = ["|gametype|doubles"])), options())).rejects.toThrow(/holds no request/);
    await expect(a2Episode(edited((r) => (r.result = r.result === "win" ? "loss" : "win")), options())).rejects.toThrow(/not the channel's/);
  });

  it("refuses a channel whose first request does not list the bot's team", async () => {
    const broken = (first: string) =>
      edited((r) => {
        const lines = (r.channels[0] as { lines: string[] }).lines;
        lines[lines.findIndex((line) => line.startsWith("|request|"))] = first;
      });
    await expect(a2Episode(broken("|request|{not json"), options())).rejects.toThrow(/does not list the bot's team/);
    await expect(a2Episode(broken('|request|{"wait":true}'), options())).rejects.toThrow(/does not list the bot's team/);
    await expect(a2Episode(mine, options({ team: stamp(own.slice(0, 5)) }))).rejects.toThrow(/does not list the bot's team/);
  });

  it("refuses a sparring opponent team: a human's team is never known", async () => {
    await expect(a2Episode(mine, options({ opponent_team: stamp(rival) }))).rejects.toThrow(A2EpisodeError);
  });

  it.each([
    ["a fatal end", (r: A2BattleRecord) => (r.fatal = { error: "SessionError", channel_complete: true }), /fatal/],
    ["an unfinished battle", (r: A2BattleRecord) => (r.result = "unfinished"), /did not finish/],
    ["an unconfirmed decline", (r: A2BattleRecord) => (r.ots = "forfeited"), /Open Team Sheets/],
    ["an OTS failure", (r: A2BattleRecord) => (r.ots_failure = { reason: "refused", detail: "x" }), /Open Team Sheets/],
    ["a defect", (r: A2BattleRecord) => (r.defect = true), /defect/],
    ["a server auto-choose", (r: A2BattleRecord) => r.server_auto_chooses.push({ rqid: 1, reason: "x" } as never), /auto-chose/],
    ["no seat", (r: A2BattleRecord) => (r.player = null), /no player seat/],
    ["no channel", (r: A2BattleRecord) => (r.channels = []), /no channel/],
    ["a second channel that does not continue the first", (r: A2BattleRecord) => r.channels.push(r.channels[0] as never), /2 connections' channels do not join/],
    ["another player's channel", (r: A2BattleRecord) => ((r.channels[0] as { player: string }).player = r.player === "p1" ? "p2" : "p1"), /not the bot's own/],
    ["a decision with no timer snapshot", (r: A2BattleRecord) => (decision(r).timer = null), /no timer snapshot/],
    ["an overrun with no fallback", (r: A2BattleRecord) => (decision(r, 1).elapsed_ms = decision(r, 1).budget.submit_within_ms + 1), /overran/],
  ])("refuses %s", async (_name, change, message) => {
    await expect(a2Episode(edited(change), options())).rejects.toThrow(message);
    await expect(a2Episode(edited(change), options())).rejects.toBeInstanceOf(A2EpisodeError);
  });

  it("maps the session's deadline evidence: watchdog, zero budget, rounded times, the next fresh snapshot", async () => {
    const episode = await a2Episode(
      edited((r) => {
        decision(r, 1).watchdog = "overrun";
        decision(r, 2).budget = { ...decision(r, 2).budget, policy_budget_ms: 0 };
        decision(r, 3).elapsed_ms = 12.2;
        decision(r, 0).elapsed_ms = -0.5;
        decision(r, 4).inherited = true;
      }),
      options(),
    );
    const outcomes = episode.decisions.map((d) => d.deadline?.deadline_outcome);
    expect(outcomes.slice(0, 3)).toEqual(["in_budget", "fallback_timeout", "skipped_low_budget"]);
    expect(episode.decisions[3]?.deadline).toMatchObject({ elapsed_ms: 13, submitted_after_ms: 13 });
    expect(episode.decisions[0]?.deadline?.elapsed_ms).toBe(0);
    expect(episode.decisions[0]?.deadline?.bank_seconds_after).toBe(decision(mine, 1).timer?.bank_seconds_left);
    // The next request is an update that inherits its timer: no fresh snapshot, no bank.
    expect(episode.decisions[3]?.deadline?.bank_seconds_after).toBeNull();
    expect(episode.decisions[4]?.decision_context?.remaining_decisions_estimate).toBe(decision(mine, 4).remaining_decisions_estimate);
  });

  it("omits R-hat where the session has none (Team Preview) and uses the given safety margin", async () => {
    const episode = await a2Episode(mine, options({ budget: { safety_margin_ms: 2500 } }));
    expect(episode.decisions[0]?.decision_context).not.toHaveProperty("remaining_decisions_estimate");
    expect(episode.decisions[0]?.decision_context?.safety_margin_ms).toBe(2500);
    const next = edited((r) => (decision(r, 1).timer = decision(r, 1).timer && null));
    await expect(a2Episode(next, options())).rejects.toThrow(/no timer snapshot/);
  });
});

describe("timerRulesOf", () => {
  it("reads the VGC Timer settings from the pinned engine's rule table", () => {
    expect(timerRulesOf(engine, FORMAT, 5)).toEqual({ starting: 420, grace: 90, add_per_turn: 0, max_per_turn: 55, max_first_turn: 90, tick: 5 });
  });

  it("refuses a format with no timer rules, and a tick that is not a whole number of seconds", () => {
    expect(() => timerRulesOf(engine, "gen9ou", 5)).toThrow(A2EpisodeError);
    expect(() => timerRulesOf(engine, FORMAT, 0)).toThrow(/tick/);
    expect(() => timerRulesOf(engine, FORMAT, 2.5)).toThrow(/tick/);
  });
});
