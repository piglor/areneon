import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import type { PrivateTeamSpec } from "@areneon/schema/types/v1/private-team-spec.ts";
import { afterEach, describe, expect, it } from "vitest";
import { actorRule, NoCandidateError } from "../src/actor-rule.ts";
import { enumerateCandidates } from "../src/choices.ts";
import { guardActor, type GuardOptions, playGame, PolicyWorkerError, UnsupportedCandidateError } from "../src/evaluation-harness.ts";
import { GOLDEN_FORMAT, GOLDEN_SEED, syntheticFixtureHashes, syntheticTeam } from "../src/golden.ts";
import { type Actor, type ActorContext, type BattleRecord, loadEngine, type Player, type RequestRecord, runBattle } from "../src/index.ts";
import { recordedChannel } from "../src/player-channel.ts";
import { type PolicyActorOptions, policyActor, policyGame, sampleU } from "../src/policy-actor.ts";
import { PolicyProtocolError, type PolicyQueryInput } from "../src/policy-protocol.ts";
import { type PolicyClient, PolicyReplyError, PolicyWorkerClient } from "../src/policy-worker.ts";
import { resolveFormat } from "../src/formats.ts";
import { battleClass, stampTeam, teamHash } from "../src/team-class.ts";
import { CHECKPOINTS, ENCODER_VERSION, ENGINE_DATA_HASH, MAX_LINE_BYTES } from "./policy-fake.ts";

const FAKE = fileURLToPath(new URL("./fake-policy-worker.ts", import.meta.url));
const engine = loadEngine();
const known = syntheticFixtureHashes(engine);
const sets = syntheticTeam();
const team = stampTeam(sets as PrivateTeamSpec["team"], known);
const manifest = resolveFormat(engine, GOLDEN_FORMAT).manifest;
const channel_class = battleClass({ p1: teamHash(sets), p2: teamHash(sets) }, known);
// SHA-256 of the JCS above, first 53 bits (computed independently of sampleU).
const KNOWN_U = 3_764_479_066_356_454;
const budget = { preview_ms: 5_000, battle_ms: 5_000 };
const policy: GuardOptions = { role: "policy", latency_budget: budget };

const optionsFor = (client: PolicyClient, player: Player, change: Partial<PolicyActorOptions> = {}): PolicyActorOptions => ({
  client,
  battle: `game-${player}`,
  player,
  team,
  manifest,
  channel_class,
  seed: GOLDEN_SEED,
  game: 0,
  policy_id: `policy:${"a".repeat(64)}`,
  sampling: "sample",
  ...change,
});

/** A client that records each query and plays the shared actor rule on its candidates. */
function recorder(): PolicyClient & { queries: { battle: string; query: PolicyQueryInput }[] } {
  const queries: { battle: string; query: PolicyQueryInput }[] = [];
  return {
    queries,
    ask(battle, query) {
      queries.push({ battle, query });
      const open = query.candidates.filter((c) => !c.pending).map((c) => c.choice);
      const pending = query.candidates.filter((c) => c.pending).map((c) => c.choice);
      return Promise.resolve({
        schema_version: 1,
        identity: query.identity,
        choice: actorRule({ choices: open, pending }),
        source_policy_id: query.policy_id,
        elapsed_ms: 0,
        deadline_outcome: "in_budget",
      });
    },
  };
}

const play = (actors: { p1: Actor; p2: Actor }, seed = GOLDEN_SEED) =>
  playGame({ engine, formatId: GOLDEN_FORMAT, seed, teams: { p1: sets, p2: sets }, actors });

const both = (client: PolicyClient, change: Partial<PolicyActorOptions> = {}) => ({
  p1: guardActor(policyActor(optionsFor(client, "p1", change)), policy),
  p2: guardActor(policyActor(optionsFor(client, "p2", change)), policy),
});

describe("the sampler (ADR-010 section 2)", () => {
  const input = { seed: GOLDEN_SEED, game: 0, player: "p1" as const, policy_id: "policy:x", request_seq: 1 };

  it("is the first 53 bits of SHA-256 over the JCS of its five inputs", () => {
    const jcs = `{"game":0,"player":"p1","policy_id":"policy:x","request_seq":1,"seed":"${GOLDEN_SEED}"}`;
    const bits = BigInt(`0x${createHash("sha256").update(jcs).digest("hex").slice(0, 16)}`) >> 11n;
    expect(sampleU(input)).toBe(Number(bits));
    expect(Number.isSafeInteger(sampleU(input))).toBe(true);
    // A known answer, so a change of the construction is noticed.
    expect(sampleU(input)).toBe(KNOWN_U);
  });

  it("stays in [0, 2^53 - 1] and separates every input", () => {
    const draws = [
      input,
      { ...input, seed: "sodium,ffffffffffffffffffffffffffffffff" },
      { ...input, game: 1 },
      { ...input, game: 2 },
      { ...input, player: "p2" as const },
      { ...input, policy_id: "policy:y" },
      { ...input, request_seq: 2 },
    ].map(sampleU);
    expect(new Set(draws).size).toBe(draws.length);
    for (const u of draws) {
      expect(u).toBeGreaterThanOrEqual(0);
      expect(u).toBeLessThanOrEqual(2 ** 53 - 1);
    }
  });
});
describe("the A1 online observation adapter", () => {
  it("asks the worker one query per decision, built from the player's own stream and candidates", async () => {
    const client = recorder();
    const { outcome, record } = await play(both(client));
    expect(outcome.kind).toBe("played");
    const requests = (record as BattleRecord).requests;
    expect(client.queries.length).toBeGreaterThan(4);
    for (const { query } of client.queries) {
      const { observation } = query;
      const mine = requests[observation.player].find((r) => r.request_seq === observation.identity.request_seq) as RequestRecord;
      // The observation is the player's own channel up to that request, built the A1 way.
      const channel = recordedChannel(record as BattleRecord, observation.player, mine.request_seq, channel_class);
      expect(observation.log).toEqual(channel.lines);
      expect(query.identity).toEqual(observation.identity);
      expect(query.identity).toEqual({ request_seq: mine.request_seq, request_hash: mine.request_hash });
      // The candidates are the enumerator's, in its order, each with only choice and pending.
      expect(query.candidates).toEqual(enumerateCandidates(mine).candidates);
      expect(query.sampling.kind).toBe("sample");
    }
  });

  it("sends the worker no seed, no omniscient line and no verdict", async () => {
    const client = recorder();
    await play(both(client));
    const sent = JSON.stringify(client.queries.map((q) => q.query));
    expect(sent).not.toContain("000102030405060708090a0b0c0d0e0f");
    expect(sent).not.toContain("|split|");
    expect(sent).not.toContain("hidden");
    expect(sent).not.toContain("verdict");
    // The pending flag is the whole of what it learns about a pending candidate.
    const keys = new Set(client.queries.flatMap((q) => q.query.candidates.flatMap((c) => Object.keys(c))));
    expect([...keys].sort()).toEqual(["choice", "pending"]);
  });

  it("draws u from the game index: the same battle seed gives games 1 and 3 their own streams", async () => {
    const [first, third, again] = [recorder(), recorder(), recorder()];
    await play(both(first, { game: 1 }));
    await play(both(third, { game: 3 }));
    await play(both(again, { game: 1 }));
    const us = (r: ReturnType<typeof recorder>) => r.queries.map((q) => (q.query.sampling.kind === "sample" ? q.query.sampling.u : -1));
    expect(us(first)).toEqual(us(again));
    expect(us(first)).not.toEqual(us(third));
    expect(us(first)[0]).toBe(sampleU({ seed: GOLDEN_SEED, game: 1, player: "p1", policy_id: `policy:${"a".repeat(64)}`, request_seq: 1 }));
  });

  it("asks for argmax without a u", async () => {
    const client = recorder();
    await play(both(client, { sampling: "argmax" }));
    expect(new Set(client.queries.map((q) => JSON.stringify(q.query.sampling)))).toEqual(new Set(['{"kind":"argmax"}']));
  });

  it("names the game to the client, one per seat", async () => {
    const client = recorder();
    await play(both(client));
    expect(new Set(client.queries.map((q) => q.battle))).toEqual(new Set(["game-p1", "game-p2"]));
  });
});

describe("runBattle hands an actor its own stream", () => {
  it("passes the player and a copy of their lines up to the request, without timestamps", async () => {
    const seen: { request: RequestRecord; context: ActorContext | undefined }[] = [];
    const watch: Actor = (request, context) => {
      seen.push({ request, context });
      return actorRule(request.legal as NonNullable<RequestRecord["legal"]>);
    };
    const record = await runBattle({ engine, formatId: GOLDEN_FORMAT, seed: GOLDEN_SEED, teams: { p1: sets, p2: sets }, actors: { p1: watch, p2: watch } });
    expect(seen.length).toBeGreaterThan(4);
    for (const { request, context } of seen) {
      const { player, channel } = context as ActorContext;
      expect(player).toBe(request.player);
      expect(channel).toEqual(recordedChannel(record, player, request.request_seq, channel_class).lines);
      expect(channel.some((line) => line.startsWith("|t:|"))).toBe(false);
    }
  });
});

describe("failures reach guardActor classified (ADR-010 section 2)", () => {
  const failing = (error: Error): PolicyClient => ({ ask: () => Promise.reject(error) });

  it("a dead worker forfeits the seat that asked, with the worker's cause", async () => {
    const { outcome } = await play(both(failing(new PolicyWorkerError("out_of_memory", "worker killed"))));
    expect(outcome).toMatchObject({ kind: "forfeit", cause: "out_of_memory", detail: "worker killed" });
  });

  it("an invalid reply forfeits as invalid_reply", async () => {
    const { outcome } = await play(both(failing(new PolicyReplyError("the reply answers another request"))));
    expect(outcome).toMatchObject({ kind: "forfeit", cause: "invalid_reply", detail: "the reply answers another request" });
  });

  it("an unsupported candidate is an incident, never a forfeit", async () => {
    const { outcome } = await play(both(failing(new UnsupportedCandidateError("shape"))));
    expect(outcome).toMatchObject({ kind: "incident", incident: "unsupported_candidate" });
  });

  it("a query the harness cannot send is an incident", async () => {
    const { outcome } = await play(both(failing(new PolicyProtocolError("1025 candidates"))));
    expect(outcome).toMatchObject({ kind: "incident", incident: "policy_protocol" });
  });

  it("a worker that is too slow forfeits as latency_overrun, and the next game is not charged for it", async () => {
    const worker = new PolicyWorkerClient({
      command: process.execPath,
      args: [FAKE, "slow"],
      env: {},
      max_line_bytes: MAX_LINE_BYTES,
      n_parallel: 2,
      handshake_timeout_ms: 10_000,
      expect: { encoder_version: ENCODER_VERSION, engine_data_hash: ENGINE_DATA_HASH, checkpoints: CHECKPOINTS },
    });
    await worker.start();
    try {
      const seat = { client: worker, policy_id: `policy:${"a".repeat(64)}`, sampling: "sample" as const };
      const game = (ms: number, index: number) =>
        policyGame({
          engine, formatId: GOLDEN_FORMAT, seed: GOLDEN_SEED, game: index, teams: { p1: sets, p2: sets }, known_synthetic: known,
          seats: { p1: seat, p2: seat }, latency_budget: { preview_ms: ms, battle_ms: ms }, maxTurns: 1,
        });
      await worker.beginGame();
      expect((await game(100, 0)).outcome).toMatchObject({ kind: "forfeit", cause: "latency_overrun" });
      // The abandoned query would keep the worker busy; beginGame gives the next game a clean one.
      await worker.beginGame();
      expect((await game(5_000, 1)).outcome.kind).toBe("played");
    } finally {
      await worker.close();
    }
  }, 30_000);

  describe("the adapter's own refusals", () => {
    const request = async (): Promise<RequestRecord> => {
      let first: RequestRecord | undefined;
      await runBattle({
        engine,
        formatId: GOLDEN_FORMAT,
        seed: GOLDEN_SEED,
        teams: { p1: sets, p2: sets },
        actors: {
          p1: (r) => {
            first ??= r;
            return actorRule(r.legal as NonNullable<RequestRecord["legal"]>);
          },
          p2: (r) => actorRule(r.legal as NonNullable<RequestRecord["legal"]>),
        },
        maxTurns: 1,
      });
      return first as RequestRecord;
    };
    const contextFor = async (): Promise<ActorContext> => {
      const record = await runBattle({ engine, formatId: GOLDEN_FORMAT, seed: GOLDEN_SEED, teams: { p1: sets, p2: sets }, actors: { p1: (r) => actorRule(r.legal as never), p2: (r) => actorRule(r.legal as never) }, maxTurns: 1 });
      return { player: "p1", channel: recordedChannel(record, "p1", 1, channel_class).lines };
    };

    it("needs the context of its own player", async () => {
      const actor = policyActor(optionsFor(recorder(), "p1"));
      await expect(actor(await request())).rejects.toThrow(PolicyProtocolError);
      await expect(actor(await request(), { player: "p2", channel: [] })).rejects.toThrow(/own player's channel/);
    });

    it("refuses a request whose observation is of another request", async () => {
      const actor = policyActor(optionsFor(recorder(), "p1"));
      const asked = await request();
      await expect(actor({ ...asked, request_hash: "0".repeat(64) }, await contextFor())).rejects.toThrow(/another request/);
    });

    it("refuses a channel or team that cannot be observed, as a harness fault", async () => {
      const asked = await request();
      const context = await contextFor();
      await expect(policyActor(optionsFor(recorder(), "p1"))(asked, { ...context, channel: ["|nonsense"] })).rejects.toThrow(/cannot observe/);
      const foreign = { ...team, team: team.team.slice(1) } as unknown as PrivateTeamSpec;
      await expect(policyActor(optionsFor(recorder(), "p1", { team: foreign }))(asked, context)).rejects.toThrow(PolicyProtocolError);
    });

    it("refuses a team whose stats are not the Team Preview request's, as a harness fault, and asks the given engine", async () => {
      const asked = await request();
      const context = await contextFor();
      const mirror = structuredClone(team) as { team: { evs: { atk: number } }[] };
      (mirror.team[2] as { evs: { atk: number } }).evs.atk += 1;
      await expect(policyActor(optionsFor(recorder(), "p1", { team: mirror as unknown as PrivateTeamSpec }))(asked, context)).rejects.toThrow(
        /cannot observe the request: the stats of the team's Pokémon at position 2/,
      );
      const noPreview = { ...engine, sim: { ...engine.sim, Battle: class { sides = []; setPlayer(): void {} } } } as unknown as typeof engine;
      await expect(policyActor(optionsFor(recorder(), "p1", { engine: noPreview }))(asked, context)).rejects.toThrow(/sent no Team Preview/);
    });

    it("reports a request the enumerator does not know as an unsupported candidate", async () => {
      const asked = await request();
      const odd = { ...asked, request: { ...(asked.request as object), mysteryField: 1 } } as RequestRecord;
      await expect(policyActor(optionsFor(recorder(), "p1"))(odd, await contextFor())).rejects.toThrow(UnsupportedCandidateError);
    });

    it("has nothing to ask about when a request has no candidate", async () => {
      const wait = { ...(await request()), phase: "wait" } as RequestRecord;
      await expect(policyActor(optionsFor(recorder(), "p1"))(wait, await contextFor())).rejects.toThrow(NoCandidateError);
    });
  });
});

describe("a whole A1 battle through the worker process (stub model)", () => {
  const clients: PolicyWorkerClient[] = [];
  afterEach(async () => {
    await Promise.all(clients.splice(0).map((c) => c.close()));
  });
  const open = async (mode: string): Promise<PolicyWorkerClient> => {
    const worker = new PolicyWorkerClient({
      command: process.execPath,
      args: [FAKE, mode],
      env: {},
      max_line_bytes: MAX_LINE_BYTES,
      n_parallel: 2,
      handshake_timeout_ms: 10_000,
      expect: { encoder_version: ENCODER_VERSION, engine_data_hash: ENGINE_DATA_HASH, checkpoints: CHECKPOINTS },
    });
    clients.push(worker);
    await worker.start();
    return worker;
  };

  const run = (worker: PolicyWorkerClient, game = 0) => {
    const seat = { client: worker, policy_id: `policy:${"a".repeat(64)}`, sampling: "sample" as const };
    return policyGame({
      engine, formatId: GOLDEN_FORMAT, seed: GOLDEN_SEED, game, teams: { p1: sets, p2: sets }, known_synthetic: known,
      seats: { p1: seat, p2: seat }, latency_budget: budget,
    });
  };

  it("takes the seed once: the battle and the sampler run on the same one", async () => {
    const client = recorder();
    const seat = { client, policy_id: `policy:${"a".repeat(64)}`, sampling: "sample" as const };
    const { record } = await policyGame({
      engine, formatId: GOLDEN_FORMAT, seed: GOLDEN_SEED, game: 2, teams: { p1: sets, p2: sets }, known_synthetic: known,
      seats: { p1: seat, p2: seat }, latency_budget: budget,
    });
    expect((record as BattleRecord).seed).toBe(GOLDEN_SEED);
    for (const { query } of client.queries) {
      const sampling = query.sampling as { u: number };
      expect(sampling.u).toBe(sampleU({ seed: GOLDEN_SEED, game: 2, player: query.observation.player, policy_id: seat.policy_id, request_seq: query.identity.request_seq }));
    }
  });

  it("leaves a seat that is already an actor as it is", async () => {
    const worker = await open("ok");
    const seat = { client: worker, policy_id: `policy:${"a".repeat(64)}`, sampling: "argmax" as const };
    const { outcome } = await policyGame({
      engine, formatId: GOLDEN_FORMAT, seed: GOLDEN_SEED, game: 0, teams: { p1: sets, p2: sets }, known_synthetic: known,
      seats: { p1: seat, p2: guardActor((r) => actorRule(r.legal as NonNullable<RequestRecord["legal"]>), policy) }, latency_budget: budget, maxTurns: 3,
    });
    expect(outcome.kind).toBe("played");
  });

  it("plays to a result, and the same commit, seed and game reproduce the log", async () => {
    const worker = await open("ok");
    const first = await run(worker);
    const second = await run(worker);
    expect(first.outcome.kind).toBe("played");
    expect((second.record as BattleRecord).omniscient_log).toEqual((first.record as BattleRecord).omniscient_log);
    expect((first.record as BattleRecord).choices.length).toBeGreaterThan(4);
  });

  it("a worker that crashes mid-game forfeits it, and is restarted between games", async () => {
    const worker = await open("crash");
    const lost = await run(worker);
    expect(lost.outcome).toMatchObject({ kind: "forfeit", cause: "crash" });
    await worker.beginGame();
    expect((await run(worker)).outcome).toMatchObject({ kind: "forfeit", cause: "crash" });
  });

  it("a worker that answers outside the candidates forfeits as invalid_reply", async () => {
    const worker = await open("wrong-choice");
    expect((await run(worker)).outcome).toMatchObject({ kind: "forfeit", cause: "invalid_reply" });
  });
});
