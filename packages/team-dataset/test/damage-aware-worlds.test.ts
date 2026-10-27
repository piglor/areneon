// The composition root of damage-aware@v2 (ADR-025 section 6) with the real opponent model, the pinned
// validator and the committed prior: the port, its refusals, the baseline on real belief worlds
// (hidden-information invariance with a negative control, determinism) and v2 as a training opponent.
import { cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { beliefIdentityHash } from "@areneon/opponent-model";
import {
  actorRule,
  type BeliefWorldsContext,
  damageAwareActor,
  type DamageAwareOptions,
  loadDamageAwareConfigV2,
  loadMechanicsConfig,
  loadRulesSnapshot,
  playGeneration,
  type PolicyClient,
  readTrainingRunConfig,
  resolveFormat,
  runBattle,
  teamVocabularyHash,
  usagePriorHash,
  worldsStream,
} from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it } from "vitest";
import { interpretChannel, readerSha256 } from "../../public-state/src/index.ts";
import { firstLegal, goldenTeamVocabulary } from "../../showdown-bridge/src/golden.ts";
import { loadRandomCorpus, type RandomBattleFile } from "../../showdown-bridge/src/random-corpus.ts";
import { comparable, mutated, withGenders } from "../../showdown-bridge/test/damage-aware-invariance-support.ts";
import { mechanicsQueryHash } from "../../showdown-bridge/src/mechanics-query.ts";
import { probeMechanics } from "../../showdown-bridge/src/mechanics-probe.ts";
import { datasetTeamPool } from "../src/loop-pool.ts";
import { type BeliefWorldsFiles, beliefWorldsFactory, beliefWorldsFiles, createBeliefWorlds } from "../src/damage-aware-worlds.ts";
import { DATASETS_DIR } from "../src/fixtures.ts";
import { verifiedDatasetHash } from "../src/dataset-hash.ts";
import { engine, FORMAT } from "./support.ts";

const ROOT = fileURLToPath(new URL("../../..", import.meta.url));
const vocabulary = goldenTeamVocabulary();
const config = loadDamageAwareConfigV2(vocabulary);
const snapshot = loadRulesSnapshot(engine.commit);
const manifest = resolveFormat(engine, FORMAT, snapshot).manifest;
const context: BeliefWorldsContext = { engine, vocabulary, config, reader_sha256: readerSha256(), manifest };
const files = beliefWorldsFiles(FORMAT);
const scratch = mkdtempSync(join(tmpdir(), "v2-worlds-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));

const options = (change: Partial<DamageAwareOptions> = {}): DamageAwareOptions => ({
  engine,
  vocabulary,
  config,
  mechanics: loadMechanicsConfig(FORMAT),
  snapshot,
  data_class: "synthetic",
  worlds: createBeliefWorlds(context),
  ...change,
});

describe("the port of the committed record", () => {
  const port = createBeliefWorlds(context);

  it("is the record's prior and belief config, for the record's format, with the belief identity of ADR-025 section 5", () => {
    expect(port.prior_hash).toBe(config.assumed_set.prior_hash);
    expect(port.belief_config_hash).toBe(config.assumed_set.belief_config_hash);
    expect(port.prior_format_id).toBe(config.format_id);
    expect(port.identity_hash).toBe(
      beliefIdentityHash({ prior_hash: port.prior_hash, belief_config_hash: port.belief_config_hash, vocabulary_hash: teamVocabularyHash(vocabulary), reader_sha256: readerSha256(), version_manifest: manifest }),
    );
    expect(beliefWorldsFactory(context).identity_hash).toBe(port.identity_hash);
  });

  const preview = ["abomasnow", "charizard", "ditto", "gallade", "aerodactyl", "garchomp"];
  const names = new Map(Object.entries({ abomasnow: "Abomasnow", charizard: "Charizard", ditto: "Ditto", gallade: "Gallade", aerodactyl: "Aerodactyl", garchomp: "Garchomp" }));
  const lines = ["|gametype|doubles", "|player|p1|Alice|", "|player|p2|Bob|", "|clearpoke", ...preview.map((id) => `|poke|p2|${names.get(id)}, L50, M|`)];
  const opponent = (() => {
    const run = interpretChannel("p1", lines);
    return (run.frames[run.frames.length - 1] as NonNullable<(typeof run.frames)[number]>).state.opponent;
  })();
  const request = { request_hash: "ab".repeat(32), request_seq: 1, player: "p1" };

  it("samples the record's number of worlds from the stream, the same worlds for the same stream and others for another", () => {
    const a = port.sample({ opponent, flags: [], stream: worldsStream(request), n: 8 });
    expect(a.worlds).toHaveLength(8);
    expect(a.worlds.every((w) => w.world.slots.length === 6 && w.world.slots.filter((s) => s.brought).length === 4)).toBe(true);
    const again = port.sample({ opponent, flags: [], stream: worldsStream(request), n: 8 });
    expect(JSON.stringify(again)).toBe(JSON.stringify(a));
    const other = port.sample({ opponent, flags: [], stream: worldsStream({ ...request, request_seq: 2 }), n: 8 });
    expect(JSON.stringify(other)).not.toBe(JSON.stringify(a));
  });

  it("refuses by the model's own code: a flagged channel is a reader_flagged BeliefError", () => {
    expect(() => port.sample({ opponent, flags: [{ index: 0, line: "|x|", reason: "unknown" }], stream: worldsStream(request), n: 1 })).toThrow(expect.objectContaining({ name: "BeliefError", code: "reader_flagged" }));
  });
});

describe("what it refuses to build", () => {
  const copy = (name: string, change: (dir: string) => void): BeliefWorldsFiles => {
    const dir = join(scratch, name);
    mkdirSync(dir);
    cpSync(files.model_dir, dir, { recursive: true });
    change(dir);
    return { ...files, model_dir: dir };
  };
  const edit = (dir: string, name: string, change: (value: Record<string, unknown>) => void): void => {
    const file = join(dir, name);
    const value = JSON.parse(readFileSync(file, "utf8")) as Record<string, unknown>;
    change(value);
    writeFileSync(file, JSON.stringify(value));
  };

  it("a usage prior other than the one the record names, and a belief config other than the one it names", () => {
    const prior = copy("prior", (dir) => edit(dir, "generator-matched.uniform-v1.json", (v) => ((v.derivation as Record<string, unknown>).script_version = "generator-prior.v0")));
    expect(() => createBeliefWorlds(context, prior)).toThrow("not the one the record names");
    const belief = copy("belief", (dir) => edit(dir, "belief.v1.json", (v) => (v.max_legality_redraws = 33)));
    expect(() => createBeliefWorlds(context, belief)).toThrow("belief config is not the one the record names");
  });

  it("a prior for another format than the record's: never used silently (review finding N10)", () => {
    expect(() => createBeliefWorlds({ ...context, config: { ...config, format_id: "gen9championsvgc2026regmd" } }, files)).toThrow("another format");
  });

  it("a prior that was not derived from the committed generator config", () => {
    const generator = join(scratch, "generator.json");
    const value = JSON.parse(readFileSync(files.generator_file, "utf8")) as Record<string, unknown>;
    value.moves_per_set = (value.moves_per_set as number) - 1;
    writeFileSync(generator, JSON.stringify(value));
    expect(() => createBeliefWorlds(context, { ...files, generator_file: generator })).toThrow("not derived from the committed generator config");
  });

  it("the prior file's hash is the record's (the committed pair agree)", () => {
    const prior = JSON.parse(readFileSync(join(files.model_dir, "generator-matched.uniform-v1.json"), "utf8"));
    expect(usagePriorHash(prior)).toBe(config.assumed_set.prior_hash);
  });
});

// ---- the baseline on real belief worlds ----

const MAX_TURNS = 6;
const corpus = loadRandomCorpus().slice(0, 3);
const battles = corpus.slice(0, 2);
const validatorTeams = (battle: RandomBattleFile) => battle.teams;

/** The opponent's team with its Stat Points put where they make it bulky or frail (still a legal team). */
const skewed = (team: RandomBattleFile["teams"]["p2"], bulky: boolean): RandomBattleFile["teams"]["p2"] =>
  team.map((set) => ({ ...structuredClone(set), evs: bulky ? { hp: 32, atk: 0, def: 32, spa: 0, spd: 2, spe: 0 } : { hp: 0, atk: 32, def: 0, spa: 32, spd: 0, spe: 2 } }));

/** What the probe was asked at each request of the baseline's seat (hashes of the queries, in order), beside the battle. */
async function play(battle: RandomBattleFile, p2 = battle.teams.p2, change: Partial<DamageAwareOptions> = {}, seed = battle.seed, p1 = battle.teams.p1) {
  let current: string[] = [];
  const asked = new Map<number, string[]>();
  const actor = damageAwareActor(
    options({
      ...change,
      probe: (context, query) => {
        current.push(mechanicsQueryHash(query));
        return probeMechanics(context, query);
      },
    }),
  );
  const logged: typeof actor = Object.assign(
    (request: Parameters<typeof actor>[0], ctx?: Parameters<typeof actor>[1]) => {
      current = [];
      asked.set(request.request_seq, current);
      return actor(request, ctx);
    },
    { report: actor.report },
  );
  const record = await runBattle({ engine, formatId: FORMAT, seed, teams: { p1, p2 }, actors: { p1: logged, p2: firstLegal }, maxTurns: MAX_TURNS });
  return { record, report: actor.report(), asked };
}

describe("hidden-information invariance with the belief in the path (ADR-018 section 3, ADR-025 section 8)", () => {
  it("makes the same choices at every request whose player channel is the same, whatever the opponent holds unrevealed", async () => {
    let compared = 0;
    let scored = 0;
    for (const battle of battles) {
      const other = mutated(validatorTeams(battle).p2);
      expect(JSON.stringify(other)).not.toBe(JSON.stringify(battle.teams.p2));
      const a = await play(battle);
      const b = await play(battle, other);
      const pairs = comparable(a.record, b.record);
      for (const { seq, first, second } of pairs) {
        expect(second, `${battle.id} request ${seq}`).toBe(first);
        // not only the choice: the probe is asked the very same questions (the worlds depend on the channel alone)
        expect(b.asked.get(seq), `${battle.id} request ${seq} queries`).toEqual(a.asked.get(seq));
      }
      compared += pairs.length;
      scored += a.report.counts.scored_decisions;
      expect(a.report.belief?.decisions).toBeGreaterThan(0);
    }
    expect(compared).toBeGreaterThanOrEqual(4);
    expect(scored).toBeGreaterThanOrEqual(3);
  }, 240_000);

  it("has teeth: a belief given the opponent's real team (every world is the real team) makes the probe be asked different questions at such a request", async () => {
    // the cheating port: all eight worlds are the opponent's real sets
    const cheat = (real: RandomBattleFile["teams"]["p2"]) => {
      const inner = createBeliefWorlds(context);
      return {
        ...inner,
        sample: (input: Parameters<typeof inner.sample>[0]) => {
          const toId = (t: string): string => t.toLowerCase().replace(/[^a-z0-9]/g, "");
          const world = {
            slots: input.opponent.roster.map((mon, index) => {
              const set = real.find((s) => [mon.base_species, mon.species].includes(toId(s.species))) ?? (real[0] as (typeof real)[number]);
              return {
                index,
                brought: true,
                set: {
                  species: mon.base_species,
                  forme: mon.base_species,
                  item: set.item === "" ? null : toId(set.item),
                  ability: toId(set.ability),
                  nature: toId(set.nature),
                  stat_points: set.evs,
                  moves: set.moves.map(toId).sort(),
                  level: 50,
                  gender: null,
                },
              };
            }),
          };
          return { worlds: Array.from({ length: input.n }, () => ({ world })), redraws: 0 };
        },
      };
    };
    let differing = 0;
    let compared = 0;
    for (const battle of corpus) {
      // two opponents that look alike until something is revealed and differ starkly in bulk: all Stat Points in hit points and defences, or in attack and speed
      const [bulky, frail] = [skewed(battle.teams.p2, true), skewed(battle.teams.p2, false)];
      const a = await play(battle, bulky, { worlds: cheat(bulky) });
      const b = await play(battle, frail, { worlds: cheat(frail) });
      for (const { seq } of comparable(a.record, b.record)) {
        compared++;
        if (JSON.stringify(a.asked.get(seq)) !== JSON.stringify(b.asked.get(seq))) differing++;
      }
    }
    expect(compared).toBeGreaterThanOrEqual(6);
    // the probe is asked about the real team: the questions differ at requests where the channels are the same
    expect(differing).toBeGreaterThan(0);
  }, 240_000);

  it("does not change when the battle seed changes, at the requests that are the same", async () => {
    const battle = battles[0] as RandomBattleFile;
    const [mine, theirs] = [withGenders(battle.teams.p1), withGenders(battle.teams.p2)];
    const a = await play(battle, theirs, {}, battle.seed, mine);
    const b = await play(battle, theirs, {}, "sodium,ffffffffffffffffffffffffffffffff", mine);
    expect(a.record.seed).not.toBe(b.record.seed);
    const pairs = comparable(a.record, b.record);
    for (const { seq, first, second } of pairs) expect(second, `${battle.id} request ${seq}`).toBe(first);
    expect(pairs.length).toBeGreaterThanOrEqual(2);
  }, 120_000);

  it("does not change with the actor's own stream seed while epsilon_ppm is 0", async () => {
    const battle = battles[0] as RandomBattleFile;
    const a = await play(battle, battle.teams.p2, { seed: 1 });
    const b = await play(battle, battle.teams.p2, { seed: 0xdeadbeef });
    expect(b.record.choices).toEqual(a.record.choices);
    expect(b.report).toEqual(a.report);
  }, 120_000);
});

describe("determinism", () => {
  it("gives the same choices, counts, belief counts and identity from two runs", async () => {
    const battle = battles[1] as RandomBattleFile;
    const a = await play(battle);
    const b = await play(battle);
    expect(b.record.choices).toEqual(a.record.choices);
    expect(b.record.omniscient_log).toEqual(a.record.omniscient_log);
    const strip = (r: typeof a.report) => ({ ...r, counts: { ...r.counts, probe_runs: 0 } });
    expect(JSON.stringify(strip(b.report))).toBe(JSON.stringify(strip(a.report)));
    expect(a.report.policy_id).toBe("damage-aware@v2");
    expect(a.report.identity.belief_identity_hash).toBe(createBeliefWorlds(context).identity_hash);
    expect(a.report.belief).toMatchObject({ worlds: 8, refusals: {} });
    expect(a.report.belief?.worlds_sampled).toBe(8 * (a.report.belief?.decisions ?? 0));
  }, 120_000);
});

describe("damage-aware@v2 as a training opponent", () => {
  it("is played by the composition root's factory and recorded under its own policy id and identity", async () => {
    const dir = join(DATASETS_DIR(FORMAT), "fixtures-v1");
    const raw = JSON.parse(readFileSync(join(ROOT, "fixtures/contracts/v1/training-run-config/valid/tiny.json"), "utf8")) as Record<string, unknown>;
    (raw.generation as Record<string, unknown>).excluded_abilities = [];
    raw.team_dataset = { path: dir, dataset_hash: verifiedDatasetHash(engine, dir, FORMAT) };
    (raw.generation as Record<string, unknown>).games = 1;
    (raw.generation as Record<string, unknown>).baseline_games = [{ actor: "damage-aware", policy_id: "damage-aware@v2", games: 2 }];
    const run = readTrainingRunConfig(raw);
    const pool = datasetTeamPool(engine, run);
    const client = {
      beginGame: () => Promise.resolve(),
      ask: (_battle, query) =>
        Promise.resolve({
          schema_version: 1 as const,
          identity: query.identity,
          choice: actorRule({ choices: query.candidates.filter((c) => !c.pending).map((c) => c.choice), pending: query.candidates.filter((c) => c.pending).map((c) => c.choice) }),
          source_policy_id: query.policy_id,
          elapsed_ms: 0,
          deadline_outcome: "in_budget" as const,
        }),
    } satisfies PolicyClient & { beginGame: () => Promise<void> };
    const result = await playGeneration({ engine, config: run, generation: 1, policy_id: `policy:${"c".repeat(64)}`, client, pool, belief_worlds: beliefWorldsFactory });
    const entry = (result.run_record as { baselines?: { policy_id: string; identity_hash: string; games: number }[] }).baselines ?? [];
    expect(entry).toHaveLength(1);
    expect(entry[0]).toMatchObject({ policy_id: "damage-aware@v2", games: 2 });
    expect(result.games.every((g) => g.outcome === "played")).toBe(true);
  }, 240_000);
});
