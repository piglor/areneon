// The search actor (ADR-026 section 2, slice R part 2) with the real opponent model, the pinned engine and a stand-in worker
// (no Python): determinism, what is searched and what falls back, the leaves, the drops, the privacy invariance with its negative
// control, and that nothing searched reaches an Episode or a TrainingExample.
import {
  type DropReason,
  episodeFromRecord,
  loadSearchConfig,
  loadTrainingSet,
  ReplyModelError,
  type ReplyModel,
  replayActor,
  runBattle,
  SearchError,
  searchPolicy,
  searchSetup,
  goldenWorldFactsData,
  stampTeam,
  syntheticTeamHashes,
  teamHash,
  type WorldSamplePort,
  createReplyModel,
  loadMechanicsConfig,
  loadDamageAwareConfig,
} from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { firstLegal } from "../../showdown-bridge/src/golden.ts";
import { loadRandomCorpus, type RandomBattleFile } from "../../showdown-bridge/src/random-corpus.ts";
import { comparable, mutated, withGenders } from "../../showdown-bridge/test/damage-aware-invariance-support.ts";
import { engine, FORMAT } from "./support.ts";
import { manifest, playSearch, portFor, setupFor, snapshot, StandInWorker, vocabulary } from "./search-actor-support.ts";

const corpus = loadRandomCorpus();
const battles = corpus.slice(0, 3);
const toId = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");

describe("the setup", () => {
  it("refuses a config for another format, another belief, another prior arm and a prior of another format", () => {
    const port = portFor();
    const config = loadSearchConfig(FORMAT);
    expect(() => setupFor(port, { config: { ...config, format_id: "gen9championsvgc2026regmd" } })).toThrow(SearchError);
    expect(() => setupFor(port, { config: { ...config, belief_identity_hash: "0".repeat(64) } })).toThrow("belief identity");
    const arm = portFor("role-sets");
    expect(() => setupFor(arm, { config: { ...config, belief_identity_hash: arm.identity_hash } })).toThrow("prior arm");
    expect(() => setupFor({ ...port, prior_format_id: "gen9other" })).toThrow("another format");
  });

  it("names the search by an identity that moves with the leaf and the prior arm, and a policy id that is not the pair's own", () => {
    const a = setupFor();
    const b = setupFor(portFor(), { config: { ...a.config, leaf: { kind: "hp-ko-differential" } } });
    const pair = { preview: "a".repeat(64), battle: "b".repeat(64) };
    const one = searchPolicyOf(a, pair);
    expect(searchPolicyOf(b, pair).identity_hash).not.toBe(one.identity_hash);
    expect(one.policy_id).not.toBe(searchPolicyOf(a, pair, "sample").policy_id);
    expect(one.policy_id).toMatch(/^policy:[0-9a-f]{64}$/);
  });
});
const searchPolicyOf = (setup: ReturnType<typeof setupFor>, pair: { preview: string; battle: string }, mode: "argmax" | "sample" = "argmax") => searchPolicy(setup, manifest, pair, mode);

describe("a searched game", () => {
  it("searches move requests, falls back for Team Preview, and counts what it did", async () => {
    let searched = 0;
    for (const battle of battles) {
      const { actor, record, worker } = await playSearch(battle);
      const { counts } = actor.report();
      searched += counts.searched_decisions;
      expect(counts.decisions).toBe(record.choices.filter((c) => c.player === "p1").length);
      expect(counts.fallbacks.team_preview).toBe(1);
      const fellBack = Object.values(counts.fallbacks).reduce((total, n) => total + n, 0);
      expect(counts.searched_decisions + fellBack).toBe(counts.decisions);
      expect(worker.queries.filter((q) => q.kind === "ask")).toHaveLength(fellBack);
      expect(counts.worlds_built + counts.worlds_refused).toBe(counts.worlds_asked);
      if (counts.searched_decisions > 0) {
        // k candidates x N worlds x M replies per searched decision, less what the memo answered
        expect(counts.turns).toBeGreaterThan(0);
        const leaves = counts.leaf_queries + counts.leaf_memo_hits + counts.leaf_game_end;
        expect(leaves).toBeGreaterThan(0);
        expect(leaves).toBeLessThanOrEqual(counts.turns);
        expect(counts.worlds_scored).toBeGreaterThanOrEqual(4 * counts.searched_decisions);
        expect(counts.replies_max_damage).toBeGreaterThan(0);
        expect(counts.replies_uniform).toBe(2 * counts.replies_max_damage);
      }
      expect(actor.report().prior_label).toContain("an upper bound for this evaluation");
    }
    expect(searched).toBeGreaterThan(0);
  }, 300_000);

  it("gives the same choices, log and counts from two runs", async () => {
    const battle = battles[0] as RandomBattleFile;
    const [a, b] = [await playSearch(battle), await playSearch(battle)];
    expect(b.record.choices).toEqual(a.record.choices);
    expect(b.record.omniscient_log).toEqual(a.record.omniscient_log);
    expect(JSON.stringify(b.actor.report())).toBe(JSON.stringify(a.actor.report()));
    expect(a.actor.report().counts.searched_decisions).toBeGreaterThan(0);
  }, 300_000);

  it("plays the pair's own choice at a request the search skips, and a skipped decision equals the pair's decision", async () => {
    const battle = battles[0] as RandomBattleFile;
    // The A2 clock below the floor skips every move request: the game is the pair's game.
    const skipped = await playSearch(battle, { options: { bank_ms: () => 0 } });
    const counts = skipped.actor.report().counts;
    expect(counts.searched_decisions).toBe(0);
    expect(counts.fallbacks.clock_floor).toBeGreaterThan(0);
    expect(skipped.worker.queries.every((q) => q.kind === "ask")).toBe(true);
    // Unknown bank, or plenty: searched.
    const unknown = await playSearch(battle, { options: { bank_ms: () => null } });
    const plenty = await playSearch(battle, { options: { bank_ms: () => 10 ** 9 } });
    expect(unknown.actor.report().counts.searched_decisions).toBeGreaterThan(0);
    expect(plenty.record.choices).toEqual(unknown.record.choices);
  }, 300_000);

  it("does not search a forced switch, and counts it", async () => {
    let forced = 0;
    for (const battle of corpus.slice(0, 8)) {
      const { actor } = await playSearch(battle, { maxTurns: 6 });
      forced += actor.report().counts.fallbacks.forced_switch;
      if (forced > 0) break;
    }
    expect(forced).toBeGreaterThan(0);
  }, 300_000);

  it("scores a game's end as the game score and searches with the non-learned leaf on request", async () => {
    const config = { ...loadSearchConfig(FORMAT), leaf: { kind: "hp-ko-differential" as const } };
    const { actor, worker } = await playSearch(battles[0] as RandomBattleFile, { setup: setupFor(portFor(), { config }) });
    const { counts } = actor.report();
    expect(counts.searched_decisions).toBeGreaterThan(0);
    expect(counts.leaf_hp_ko).toBeGreaterThan(0);
    expect(counts.leaf_queries).toBe(0);
    // The prior is still the worker's (one evaluate per searched decision); no value query.
    expect(worker.queries.filter((q) => q.kind === "evaluate")).toHaveLength(counts.searched_decisions + counts.fallbacks.below_min_valid);
  }, 300_000);

  it("plays the secondary prior arm through the same sampler", async () => {
    const config = { ...loadSearchConfig(FORMAT), belief_identity_hash: portFor("role-sets").identity_hash, worlds: { ...loadSearchConfig(FORMAT).worlds, prior: { kind: "role-sets" as const, label: portFor("role-sets").prior_label } } };
    const { actor } = await playSearch(battles[0] as RandomBattleFile, { setup: setupFor(portFor("role-sets"), { config }) });
    expect(actor.report().prior_label).toBe(portFor("role-sets").prior_label);
    expect(actor.report().counts.decisions).toBeGreaterThan(0);
  }, 300_000);
});

describe("play modes and reply weights", () => {
  const battle = battles[0] as RandomBattleFile;
  const withReplies = (replies: { max_damage_ppm: number; uniform_draws: number }) => setupFor(portFor(), { config: { ...loadSearchConfig(FORMAT), replies } });

  it("searches as the second seat, with the pair's sampled play in the fallbacks and no snapshot given to the setup", async () => {
    const setup = searchSetup({ engine, format_id: FORMAT, vocabulary, port: portFor(), facts_data: goldenWorldFactsData(), data_class: "synthetic" });
    const game = await playSearch(battle, { seat: "p2", setup, options: { sampling: "sample", play: { temperature_ppm: 500_000 } } });
    expect(game.actor.report().counts.decisions).toBeGreaterThan(0);
    expect(game.actor.report().counts.searched_decisions).toBeGreaterThan(0);
  }, 300_000);

  it("takes the whole weight on the max-damage reply with no uniform draw, and only uniform replies with no max-damage weight (the ablation)", async () => {
    const onlyMax = await playSearch(battle, { setup: withReplies({ max_damage_ppm: 1_000_000, uniform_draws: 0 }) });
    const a = onlyMax.actor.report().counts;
    expect(a.replies_uniform).toBe(0);
    expect(a.replies_max_damage).toBeGreaterThan(0);
    const uniform = await playSearch(battle, { setup: withReplies({ max_damage_ppm: 0, uniform_draws: 2 }) });
    const b = uniform.actor.report().counts;
    expect(b.replies_max_damage).toBe(0);
    expect(b.replies_uniform).toBeGreaterThan(0);
  }, 300_000);

  it("plays a finished game through, scoring a turn that ends it as the game score", async () => {
    const game = await playSearch(corpus[1] as RandomBattleFile, { maxTurns: 40 });
    expect(game.record.winner).not.toBeNull();
    expect(game.actor.report().counts.leaf_game_end).toBeGreaterThan(0);
  }, 600_000);
});

describe("refusals and drops", () => {
  const battle = battles[0] as RandomBattleFile;
  const real = (): ReplyModel =>
    createReplyModel({ engine, vocabulary, config: loadDamageAwareConfig("v1", vocabulary), mechanics: loadMechanicsConfig(FORMAT), snapshot, data_class: "synthetic" });

  it("refuses a pair that reads candidate features, a context without the team and another player's context", async () => {
    const worker = new StandInWorker();
    const withFeatures = Object.assign(Object.create(worker) as StandInWorker, { featureSet: { feature_set: "probe-damage/1", feature_set_hash: "c".repeat(64) } });
    await expect(playSearch(battle, { worker: withFeatures })).rejects.toThrow(SearchError);
    const { actor } = await playSearch(battle, { maxTurns: 1 });
    const request = { phase: "move", request_seq: 1 } as never;
    await expect(actor(request, undefined)).rejects.toThrow("own player's channel");
    await expect(actor(request, { player: "p2", channel: [] })).rejects.toThrow("own player's channel");
    await expect(actor(request, { player: "p1", channel: [] })).rejects.toThrow(SearchError);
  });

  it("drops a world whose reply model fails or whose reply the engine refuses, counts it, and plays the pair when too few worlds remain", async () => {
    const failing: ReplyModel = { ...real(), reply: () => { throw new ReplyModelError("no reply"); } };
    const a = await playSearch(battle, { options: { reply_model: failing } });
    const dropped = a.actor.report().counts.worlds_dropped;
    expect(dropped.reply_model).toBeGreaterThan(0);
    expect(a.actor.report().counts.searched_decisions).toBe(0);
    const illegal: ReplyModel = { ...real(), reply: () => ({ choice: "move 9 9, move 9 9", kind: "max_damage", value: 1 }) };
    const b = await playSearch(battle, { options: { reply_model: illegal } });
    expect(b.actor.report().counts.worlds_dropped.choice_refused satisfies number).toBeGreaterThan(0);
    expect(b.actor.report().counts.fallbacks.below_min_valid).toBeGreaterThan(0);
    const keys: DropReason[] = ["reply_model", "choice_refused"];
    expect(Object.keys(b.actor.report().counts.worlds_dropped)).toEqual(keys);
  }, 300_000);

  it("plays the pair when too few candidates are left, and fails on a request or channel that is not the one asked", async () => {
    // The searching seat is handed a request body with one move per active and nothing on the bench: one candidate, nothing to rank.
    let single = 0;
    let wrong: unknown;
    let garbled: unknown;
    const game = await playSearch(battle, {
      maxTurns: 2,
      wrap: (inner) => async (request, context) => {
        const legal = (request.legal?.choices[0] ?? "default") as string;
        if (request.phase !== "move") return inner(request, context);
        const body = JSON.parse(JSON.stringify(request.request)) as { active: { moves: object[] }[]; side: { pokemon: { condition: string }[] } };
        const agility = { move: "Agility", id: "agility", pp: 20, maxpp: 20, target: "self", disabled: false };
        for (const active of body.active) active.moves = [agility];
        body.side.pokemon.forEach((mon, index) => (index >= body.active.length ? (mon.condition = "0 fnt") : undefined));
        await inner({ ...request, request: body as never }, context);
        single += 1;
        await Promise.resolve(inner({ ...request, request_seq: request.request_seq + 1 }, context)).catch((error: unknown) => (wrong = error));
        await Promise.resolve(inner(request, { ...(context as object), channel: ["|bogus"] } as never)).catch((error: unknown) => (garbled = error));
        return legal;
      },
    });
    expect(single).toBeGreaterThan(0);
    expect(game.actor.report().counts.fallbacks.too_few_candidates).toBeGreaterThanOrEqual(single);
    expect((wrong as Error).message).toContain("another request than the one asked");
    expect((garbled as Error).message).toContain("cannot observe the request");
  }, 300_000);

  it("lets a defect of the reply model through and does not mistake it for a dropped world", async () => {
    const broken: ReplyModel = { ...real(), reply: () => { throw new TypeError("a defect"); } };
    await expect(playSearch(battle, { options: { reply_model: broken } })).rejects.toThrow("a defect");
  }, 300_000);

  it("plays the pair when the belief refuses the request", async () => {
    const port: WorldSamplePort = { ...portFor(), open: () => ({ kind: "refused", code: "evidence_has_no_support" }) };
    const { actor } = await playSearch(battle, { setup: setupFor(port, { config: loadSearchConfig(FORMAT) }) });
    const { counts } = actor.report();
    expect(counts.fallbacks.belief_refused).toBeGreaterThan(0);
    expect(counts.belief_refusals).toEqual({ evidence_has_no_support: counts.fallbacks.belief_refused });
  }, 300_000);
});

// ---- privacy ----

const MAX_TURNS = 4;
type Scores = Map<number, readonly { choice: string; sum: number }[]>;
/** Plays a game and keeps every searched decision's scores by request. */
const scored = async (battle: RandomBattleFile, change: Parameters<typeof playSearch>[1] = {}) => {
  const scores: Scores = new Map();
  const game = await playSearch(battle, { maxTurns: MAX_TURNS, ...change, options: { ...change.options, on_decision: (e) => scores.set(e.request_seq, e.scores) } });
  return { ...game, scores };
};
const validatorTeams = (b: RandomBattleFile) => b.teams;

describe("hidden-information invariance with the belief in the path (ADR-026 section 7)", () => {
  it("makes the same choices, and gives the same scores, at every request whose player channel is the same, whatever the opponent holds unrevealed", async () => {
    let compared = 0;
    let searchedAndCompared = 0;
    for (const battle of battles.slice(0, 2)) {
      const other = mutated(validatorTeams(battle).p2);
      expect(JSON.stringify(other)).not.toBe(JSON.stringify(battle.teams.p2));
      const a = await scored(battle);
      const b = await scored(battle, { p2: other });
      const pairs = comparable(a.record, b.record);
      for (const { seq, first, second } of pairs) {
        expect(second, `${battle.id} request ${seq}`).toBe(first);
        // not only the choice: the same worlds were searched, so every candidate's score is the same
        expect(b.scores.get(seq), `${battle.id} request ${seq} scores`).toEqual(a.scores.get(seq));
        if (a.scores.has(seq)) searchedAndCompared += 1;
      }
      compared += pairs.length;
    }
    expect(compared).toBeGreaterThanOrEqual(2);
    expect(searchedAndCompared).toBeGreaterThanOrEqual(2);
  }, 300_000);

  it("has teeth: a belief handed the opponent's real team gives different choices", async () => {
    const cheat = (real: RandomBattleFile["teams"]["p2"]): WorldSamplePort => {
      const inner = portFor();
      return {
        ...inner,
        open: ({ opponent }) => {
          const brought = real.slice(0, 4).map((set) => toId(set.species));
          const world = {
            slots: opponent.roster.map((mon, index) => {
              const set = real.slice(0, 4).find((s) => [mon.base_species, mon.species].includes(toId(s.species)));
              if (set === undefined || !brought.includes(toId(set.species))) return { index, brought: false, set: null };
              return {
                index,
                brought: true,
                set: { species: mon.base_species, forme: mon.base_species, item: set.item === "" ? null : toId(set.item), ability: toId(set.ability), nature: toId(set.nature), stat_points: set.evs, moves: set.moves.map(toId).sort(), level: 50, gender: null },
              };
            }),
          };
          return { kind: "source", source: { draw: () => ({ world: world as never, redraws: 0 }) } };
        },
      };
    };
    // two opponents that look alike until something is revealed and differ starkly in bulk and power
    const skew = (team: RandomBattleFile["teams"]["p2"], bulky: boolean) => team.map((set) => ({ ...structuredClone(set), evs: bulky ? { hp: 32, atk: 0, def: 32, spa: 0, spd: 2, spe: 0 } : { hp: 0, atk: 32, def: 0, spa: 32, spd: 0, spe: 2 } }));
    // The first move request: the player's channel is the same whatever the opponent holds, so an honest belief must answer the same.
    // Four opponents with the same species and different hidden sets (bulk, power, moves and items); two battles where they matter.
    let differing = 0;
    let scoresDiffering = 0;
    let pairsCompared = 0;
    for (const battle of [corpus[7], corpus[8]] as RandomBattleFile[]) {
      const teams = [battle.teams.p2, skew(battle.teams.p2, true), skew(battle.teams.p2, false), mutated(battle.teams.p2)];
      const games = [];
      for (const team of teams) games.push(await scored(battle, { p2: team, maxTurns: 2, setup: setupFor(cheat(team)) }));
      for (const [i, a] of games.entries()) {
        for (const b of games.slice(i + 1)) {
          expect(comparable(a.record, b.record).some((pair) => pair.seq === 2)).toBe(true);
          pairsCompared++;
          if (JSON.stringify(a.scores.get(2)) !== JSON.stringify(b.scores.get(2))) scoresDiffering++;
          if (a.record.choices.find((c) => c.player === "p1" && c.request_seq === 2)?.choice !== b.record.choices.find((c) => c.player === "p1" && c.request_seq === 2)?.choice) differing++;
        }
      }
    }
    expect(pairsCompared).toBe(12);
    // with the real team in the worlds the scores follow the opponent's hidden sets, and the choice with them at some request
    expect(scoresDiffering).toBe(pairsCompared);
    expect(differing).toBeGreaterThan(0);
  }, 300_000);

  it("depends on the battle seed only through the harness's search stream: the same stream is the same scores, another seed another worlds", async () => {
    const battle = battles[0] as RandomBattleFile;
    const [mine, theirs] = [withGenders(battle.teams.p1), withGenders(battle.teams.p2)];
    const a = await scored(battle, { p1: mine, p2: theirs });
    const again = await scored(battle, { p1: mine, p2: theirs });
    const other = await scored(battle, { p1: mine, p2: theirs, seed: "sodium,ffffffffffffffffffffffffffffffff" });
    expect(again.scores).toEqual(a.scores);
    expect(a.scores.size).toBeGreaterThan(0);
    expect(other.record.seed).not.toBe(a.record.seed);
    // The search stream is hashed from the seed (`search/1`), so other worlds are drawn: the scores move.
    expect(JSON.stringify([...other.scores])).not.toBe(JSON.stringify([...a.scores]));
  }, 300_000);
});

describe("nothing searched is stored", () => {
  it("gives the Episode and the training examples of a search game that a replay of its choices gives", async () => {
    const battle = battles[0] as RandomBattleFile;
    const game = await playSearch(battle, { maxTurns: MAX_TURNS });
    expect(game.actor.report().counts.searched_decisions).toBeGreaterThan(0);
    const replay = await runBattle({
      engine,
      formatId: FORMAT,
      seed: battle.seed,
      teams: battle.teams,
      actors: { p1: replayActor(game.record.choices, "p1"), p2: firstLegal },
      maxTurns: MAX_TURNS,
    });
    expect(replay.omniscient_log).toEqual(game.record.omniscient_log);
    const known = syntheticTeamHashes([teamHash(battle.teams.p1), teamHash(battle.teams.p2)]);
    const behaviour_policy_id = game.actor.report().policy_id;
    const episodeOf = (record: typeof replay) =>
      episodeFromRecord(record, "p1", {
        team: stampTeam(battle.teams.p1 as never, known),
        opponent_team: stampTeam(battle.teams.p2 as never, known),
        known_synthetic: known,
        behaviour_policy_id,
        collection: { kind: "league", run_id: "search-1", seed: record.seed, opponent_policy_id: "first-legal@golden" },
      });
    const [mine, theirs] = [await episodeOf(game.record), await episodeOf(replay)];
    expect(JSON.stringify(mine)).toBe(JSON.stringify(theirs));
    const keys = new Set<string>();
    const walk = (value: unknown): void => {
      if (Array.isArray(value)) value.forEach(walk);
      else if (typeof value === "object" && value !== null) for (const [key, inner] of Object.entries(value)) {
        keys.add(key);
        walk(inner);
      }
    };
    walk(mine);
    for (const key of ["search", "worlds", "world", "scores", "prior_ppm", "value_ppm", "top"]) expect(keys.has(key), key).toBe(false);
    const load = (episode: typeof mine) => loadTrainingSet({ episodes: [episode], evaluation_manifests: [], known_synthetic: known, behaviour_policy_id });
    expect(JSON.stringify(await load(mine))).toBe(JSON.stringify(await load(theirs)));
  }, 300_000);
});

