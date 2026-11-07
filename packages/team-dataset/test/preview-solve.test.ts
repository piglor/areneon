// The Team Preview solver as a policy seat (ADR-029 sections 3, 4, 5 and 7; ticket #1259) with the real engine, the real opponent
// model, the real validator and the real integer solver, and a stand-in worker (no network, no Python). The stand-in's Preview prior
// reads the observation's OWN TEAM (as a Preview network does) and its value is a hash of the hypothetical channel, so a world's
// hidden sets move the table: a test of the privacy path has teeth only if the numbers depend on what the worlds hold.
import { createHash } from "node:crypto";
import { readerSha256 } from "../../public-state/src/index.ts";
import {
  type Actor,
  battleClass,
  episodeFromRecord,
  type EvaluationClient,
  HypotheticalError,
  jcs,
  type Json,
  loadPreviewSolveConfig,
  loadTrainingSet,
  type PayoffClient,
  PayoffError,
  previewSolveBudgetMs,
  UnsupportedCandidateError,
  startPreviewWorlds,
  PolicyProtocolError,
  policyActor,
  PolicyReplyError,
  policyId,
  type PokemonSet,
  type PreviewSolveConfig,
  previewSolveActor,
  type PreviewSolveActorOptions,
  PreviewSolveError,
  previewSolvePolicy,
  previewSolveSetup,
  type PreviewSolveSetup,
  type PreviewSolved,
  type PreviewTeamPort,
  readPreviewChannel,
  replayActor,
  resolveFormat,
  runBattle,
  solvePreview,
  stampTeam,
  syntheticTeamHashes,
  teamHash,
  TrainingLoaderError,
  loadRulesSnapshot,
} from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { firstLegal, goldenTeamVocabulary } from "../../showdown-bridge/src/golden.ts";
import { loadRandomCorpus, type RandomBattleFile } from "../../showdown-bridge/src/random-corpus.ts";
import { mutated, withGenders } from "../../showdown-bridge/test/damage-aware-invariance-support.ts";
import { createPreviewSolvePorts } from "../src/preview-solver-port.ts";
import { StandInWorker } from "./search-actor-support.ts";
import { engine, FORMAT } from "./support.ts";

const vocabulary = goldenTeamVocabulary();
const snapshot = loadRulesSnapshot(engine.commit);
const manifest = resolveFormat(engine, FORMAT, snapshot).manifest;
const battles = loadRandomCorpus().slice(0, 3);
const pair = { preview: "a".repeat(64), battle: "b".repeat(64) };
const toId = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");

/** The stand-in: a Preview prior that reads the observation's own team, and a value that hashes the hypothetical channel. */
class Worker extends StandInWorker implements PayoffClient {
  fault: "none" | "reply" | "engine" | "protocol" | "unsupported" | "prior" = "none";
  override async evaluate(_battle: string, query: Parameters<EvaluationClient["evaluate"]>[1]): ReturnType<EvaluationClient["evaluate"]> {
    if (this.fault === "reply") throw new PolicyReplyError("the stand-in's reply is invalid");
    if (this.fault === "engine") throw new HypotheticalError("the stand-in's engine refuses");
    if (this.fault === "unsupported") throw new UnsupportedCandidateError("the stand-in cannot parse a candidate");
    if (this.fault === "protocol") throw new PolicyProtocolError("the stand-in's worker is gone");
    const observation = query.observation as unknown as { kind: string; team: unknown; log?: readonly string[] };
    this.queries.push({ kind: "evaluate", log_lines: observation.log?.length ?? 0, candidates: query.candidates.length, rows: false });
    const digest = (text: string): number => createHash("sha256").update(text).digest().readUInt32BE(0);
    const own = jcs(observation.team as Json);
    const prior_ppm = query.candidates.map(({ choice }) => 1 + (digest(`${observation.kind}|${own}|${choice}`) % 1000));
    if (this.fault === "prior") prior_ppm.pop();
    const value_ppm = digest(`value|${observation.log?.join("\n") ?? ""}`) % 1_000_001;
    return { identity: query.identity, prior_ppm, value_ppm } as Awaited<ReturnType<EvaluationClient["evaluate"]>>;
  }
}

/** The committed config at test size: two worlds, two rows, two columns (the committed values are the ADR's defaults). */
const smallConfig = (change: (c: PreviewSolveConfig) => void = () => {}): PreviewSolveConfig => {
  const c = structuredClone(loadPreviewSolveConfig(FORMAT));
  c.k_own = 2;
  c.k_opp = 2;
  c.worlds.n = 2;
  c.worlds.min_valid = 2;
  c.solver.iterations = 256;
  change(c);
  return c;
};

const ports = (): { teams: PreviewTeamPort & ReturnType<typeof createPreviewSolvePorts>["teams"]; solver: PreviewSolveSetup["solver"] } =>
  createPreviewSolvePorts({ engine, vocabulary, reader_sha256: readerSha256(), manifest, arm: "generator-matched" });

const setupFor = (config: PreviewSolveConfig = smallConfig(), p: { teams: PreviewTeamPort; solver: PreviewSolveSetup["solver"] } = ports()): PreviewSolveSetup =>
  previewSolveSetup({ engine, format_id: FORMAT, vocabulary, config, teams: p.teams, solver: p.solver, data_class: "synthetic" });

const MAX_TURNS = 2;
/** The opponent's other Team Preview: the last legal choice, so its real four and order differ. */
const lastLegal: Actor = (request) => (request.legal === undefined ? firstLegal(request) : (request.legal.choices.at(-1) ?? firstLegal(request)));

interface Game {
  record: Awaited<ReturnType<typeof runBattle>>;
  actor: ReturnType<typeof previewSolveActor>;
  worker: Worker;
  solved: { request_seq: number; choice: string; solved: PreviewSolved }[];
  policy_id: string;
}

async function play(
  battle: RandomBattleFile,
  change: { p2?: PokemonSet[]; p2Actor?: Actor; setup?: PreviewSolveSetup; worker?: Worker; bank_ms?: () => number | null; seed?: string; options?: Partial<PreviewSolveActorOptions> } = {},
): Promise<Game> {
  const teams = { p1: withGenders(battle.teams.p1), p2: change.p2 ?? withGenders(battle.teams.p2) };
  const seed = change.seed ?? battle.seed;
  const setup = change.setup ?? setupFor();
  const worker = change.worker ?? new Worker();
  const known = syntheticTeamHashes([teamHash(teams.p1), teamHash(teams.p2)]);
  const channel_class = battleClass({ p1: teamHash(teams.p1), p2: teamHash(teams.p2) }, known);
  const { policy_id, identity_hash } = previewSolvePolicy(setup, manifest, pair, "argmax");
  const base = policyActor({ client: worker, battle: "game-0-p1", player: "p1", team: stampTeam(teams.p1 as never, known), manifest, channel_class, seed, game: 0, policy_id, sampling: "argmax", engine });
  const solved: Game["solved"] = [];
  const actor = previewSolveActor({
    setup,
    client: worker,
    battle: "game-0-p1",
    player: "p1",
    policy_id,
    identity_hash,
    seed,
    game: 0,
    channel_class,
    base,
    on_solve: (decision) => solved.push(decision),
    ...(change.bank_ms === undefined ? {} : { bank_ms: change.bank_ms }),
    ...change.options,
  });
  const record = await runBattle({ engine, formatId: FORMAT, seed, teams, actors: { p1: actor, p2: change.p2Actor ?? firstLegal }, maxTurns: MAX_TURNS });
  return { record, actor, worker, solved, policy_id };
}

const previewChoice = (g: Game): string => g.record.choices.find((c) => c.player === "p1" && c.request_seq === 1)?.choice as string;
const tableOf = (g: Game) => {
  const { solved } = g.solved[0] as Game["solved"][number];
  return { own: solved.table.own.map((s) => s.key), opp: solved.table.opp.map((o) => o.leads.join("")), entries: solved.table.entries, row: solved.row, col: solved.col };
};

describe("the committed config and the setup", () => {
  it("names the belief the composition root builds, and the setup refuses what disagrees", () => {
    const committed = loadPreviewSolveConfig(FORMAT);
    expect(() => setupFor(committed)).not.toThrow();
    const p = ports();
    const bad = (change: (c: PreviewSolveConfig) => void, code: string, message: RegExp): void => {
      try {
        setupFor(smallConfig(change), p);
        throw new Error("no refusal");
      } catch (error) {
        expect(error).toBeInstanceOf(PreviewSolveError);
        expect((error as PreviewSolveError).code).toBe(code);
        expect((error as Error).message).toMatch(message);
      }
    };
    bad((c) => (c.format_id = "gen9other"), "setup_mismatch", /another format/);
    bad((c) => (c.belief_identity_hash = "1".repeat(64)), "setup_mismatch", /belief identity/);
    bad((c) => (c.worlds.prior = { kind: "role-sets", label: "x" }), "setup_mismatch", /prior arm/);
    bad((c) => (c.solver = { kind: "exact-support" }), "solver_unbuilt", /not built/);
    expect(() => previewSolveSetup({ ...setupFor(), teams: { ...p.teams, prior_format_id: "gen9other" } })).toThrow(/prior is of another format/);
  });
});

describe("a seeded game with the solver", () => {
  it("solves the Team Preview once, draws from its mixture, and delegates every other request", async () => {
    const battle = battles[0] as RandomBattleFile;
    const g = await play(battle);
    const report = g.actor.report();
    expect(report.counts).toMatchObject({ previews: 1, solved: 1, fallbacks: { belief_refusal: 0, too_few_worlds: 0, worker_failure: 0, engine_failure: 0, clock_floor: 0 }, worlds_asked: 2, worlds_refused: 0, worlds_unstarted: 0 });
    expect(report.counts.decisions).toBeGreaterThan(report.counts.previews);
    expect(report.counts.payoff).toMatchObject({ worlds_used: 2, worlds_dropped: 0, missing_entries: 0 });
    const shape = (g.solved[0] as Game["solved"][number]).solved.table;
    expect(report.counts.payoff.entries).toBe(2 * shape.own.length * shape.opp.length);
    const decision = g.solved[0] as Game["solved"][number];
    expect(decision.request_seq).toBe(1);
    expect(previewChoice(g)).toBe(decision.choice);
    const { solved } = decision;
    expect(solved.row.reduce((a, b) => a + b, 0)).toBe(1_000_000);
    expect(solved.col.reduce((a, b) => a + b, 0)).toBe(1_000_000);
    expect(solved.table.own.map((s) => s.order)).toContain(decision.choice);
    expect(solved.table.complete).toBe(true);
    expect(report.counts.exploitability_ppm).toBe(solved.exploitability_ppm);
    expect(report.counts.support_size).toBe(solved.row.filter((p) => p > 0).length);
    // Counts and the report are integers and labels only.
    const walk = (value: unknown): void => {
      if (typeof value === "number") expect(Number.isInteger(value)).toBe(true);
      else if (typeof value === "object" && value !== null) Object.values(value).forEach(walk);
    };
    walk(report.counts);
    expect(report.prior_label).toMatch(/upper bound/);
  }, 300_000);

  it("is deterministic: the same choices, log, counts and table", async () => {
    const battle = battles[0] as RandomBattleFile;
    const [a, b] = [await play(battle), await play(battle)];
    expect(b.record.omniscient_log).toEqual(a.record.omniscient_log);
    expect(b.record.choices).toEqual(a.record.choices);
    expect(b.actor.report()).toEqual(a.actor.report());
    expect(tableOf(b)).toEqual(tableOf(a));
  }, 300_000);

  it("gives the Episode and the training examples of a solved game that a replay of its choices gives, and the loader refuses them as the pair's own", async () => {
    const battle = battles[0] as RandomBattleFile;
    const g = await play(battle);
    const teams = { p1: withGenders(battle.teams.p1), p2: withGenders(battle.teams.p2) };
    const replay = await runBattle({ engine, formatId: FORMAT, seed: battle.seed, teams, actors: { p1: replayActor(g.record.choices, "p1"), p2: firstLegal }, maxTurns: MAX_TURNS });
    expect(replay.omniscient_log).toEqual(g.record.omniscient_log);
    const known = syntheticTeamHashes([teamHash(teams.p1), teamHash(teams.p2)]);
    const episodeOf = (record: typeof replay, behaviour_policy_id: string) =>
      episodeFromRecord(record, "p1", {
        team: stampTeam(teams.p1 as never, known),
        opponent_team: stampTeam(teams.p2 as never, known),
        known_synthetic: known,
        behaviour_policy_id,
        collection: { kind: "league", run_id: "preview-solve-1", seed: record.seed, opponent_policy_id: "first-legal@golden" },
      });
    const [mine, theirs] = [await episodeOf(g.record, g.policy_id), await episodeOf(replay, g.policy_id)];
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
    for (const key of ["preview_solve", "worlds", "world", "table", "matrix", "prior_ppm", "value_ppm", "mixture"]) expect(keys.has(key), key).toBe(false);
    const load = (episode: typeof mine, behaviour_policy_id: string) => loadTrainingSet({ episodes: [episode], evaluation_manifests: [], known_synthetic: known, behaviour_policy_id });
    expect(JSON.stringify(await load(mine, g.policy_id))).toBe(JSON.stringify(await load(theirs, g.policy_id)));
    // N12: the solver's policy id is not the frozen pair's, so the pair's own training load refuses the solved seat.
    const frozenId = policyId(pair, "argmax");
    expect(g.policy_id).not.toBe(frozenId);
    await expect(load(mine, frozenId)).rejects.toThrow(TrainingLoaderError);
    await expect(load(mine, frozenId)).rejects.toThrow(/only the trained policy's own seats train/);
  }, 300_000);
});

describe("hidden information", () => {
  /** The opponent's real sets as a port would hand them as worlds (the leak this test must catch). */
  const realWorlds = (team: PokemonSet[]) =>
    team.map((set) => ({
      species: toId(set.species),
      forme: toId(set.species),
      item: set.item === "" ? null : toId(set.item),
      ability: toId(set.ability),
      nature: toId(set.nature),
      stat_points: set.evs,
      moves: set.moves.map(toId),
      level: set.level,
      gender: set.gender === "" ? null : set.gender,
    }));
  const leaking = (team: PokemonSet[]): PreviewTeamPort => {
    const p = ports();
    return { ...p.teams, open: () => ({ kind: "source", source: { draw: () => ({ kind: "team", team: { sets: realWorlds(team) as never }, redraws: 0 }) } }) };
  };

  it("gives equal tables, mixtures and choices for opponents that show the same Team Preview and hold other hidden sets and another four; the leaking port does not", { timeout: 900_000 }, async () => {
    let compared = 0;
    for (const battle of battles) {
      const p2a = withGenders(battle.teams.p2);
      const p2b = withGenders(mutated(battle.teams.p2));
      const [a, b] = [await play(battle, { p2: p2a }), await play(battle, { p2: p2b, p2Actor: lastLegal })];
      // Equal public Team Preview: the channels agree up to and including the request.
      const cut = (g: Game): string[] => g.record.channels.p1.slice(0, g.record.channels.p1.findIndex((l) => l.startsWith("|request|")) + 1);
      if (jcs(cut(a) as unknown as Json) !== jcs(cut(b) as unknown as Json)) continue;
      compared += 1;
      expect(tableOf(b)).toEqual(tableOf(a));
      expect(previewChoice(b)).toBe(previewChoice(a));
      // The later requests differ with the other opponent's play (a game of another length); the solve's counts do not.
      expect({ ...b.actor.report().counts, decisions: 0 }).toEqual({ ...a.actor.report().counts, decisions: 0 });
      // Negative control: a port whose worlds are the opponent's real sets sees the hidden sets, and the table moves with them.
      const [ra, rb] = [await play(battle, { p2: p2a, setup: setupFor(smallConfig(), { teams: leaking(p2a), solver: ports().solver }) }), await play(battle, { p2: p2b, setup: setupFor(smallConfig(), { teams: leaking(p2b), solver: ports().solver }) })];
      expect(jcs(tableOf(ra) as unknown as Json)).not.toBe(jcs(tableOf(rb) as unknown as Json));
      expect(jcs(tableOf(ra) as unknown as Json)).not.toBe(jcs(tableOf(a) as unknown as Json));
      if (compared >= 2) break;
    }
    expect(compared).toBeGreaterThanOrEqual(1);
  });

  it("takes only the own view: a type test on what a solve may read", () => {
    type Input = Parameters<typeof solvePreview>[1];
    const ok: Input = { format_id: FORMAT, seat: "p1", team: [], channel: [], stream: new Uint8Array(32) };
    expect(Object.keys(ok).sort()).toEqual(["channel", "format_id", "seat", "stream", "team"]);
    // @ts-expect-error a running battle is not an input
    const _battle: Input = { ...ok, battle: {} };
    // @ts-expect-error a BattleRecord is not an input
    const _record: Input = { ...ok, record: {} };
    // @ts-expect-error the omniscient log is not an input
    const _omniscient: Input = { ...ok, omniscient_log: [] };
    // @ts-expect-error the battle seed is not an input
    const _seed: Input = { ...ok, seed: "sodium,00" };
    // @ts-expect-error the opponent's team is not an input
    const _opponent: Input = { ...ok, opponent_team: [] };
    // @ts-expect-error hidden verdicts are not an input
    const _verdicts: Input = { ...ok, hidden_verdicts: [] };
    expect([_battle, _record, _omniscient, _seed, _opponent, _verdicts]).toHaveLength(6);
  });
});

describe("fallbacks and refusals", () => {
  const battle = battles[0] as RandomBattleFile;
  const refusing = (code: string): PreviewTeamPort => ({ ...ports().teams, open: () => ({ kind: "refused", code }) });

  /** The own Team Preview view of a game, for the direct calls. */
  async function view() {
    const g = await play(battle);
    const channel = g.record.channels.p1.slice(0, g.record.channels.p1.findIndex((l) => l.startsWith("|request|")) + 1);
    return { channel, team: withGenders(battle.teams.p1) };
  }

  const deps = (setup: PreviewSolveSetup, worker = new Worker()) => ({ engine, vocabulary, config: setup.config, teams: setup.teams, solver: setup.solver, client: worker, battle: "direct", policy_id: `policy:${"c".repeat(64)}`, channel_class: "synthetic" as const });

  it("plays the pair's own Team Preview and counts a belief refusal by code", async () => {
    const setup = setupFor(smallConfig(), { teams: refusing("species_not_in_prior"), solver: ports().solver });
    const g = await play(battle, { setup });
    expect(g.solved).toHaveLength(0);
    expect(g.actor.report().counts).toMatchObject({ solved: 0, fallbacks: { belief_refusal: 1 }, belief_refusals: { species_not_in_prior: 1 } });
    // The stand-in worker is the pair: it plays its first non-pending candidate.
    const free = await play(battle);
    expect(previewChoice(g)).toMatch(/^team \d+$/);
    expect(free.solved).toHaveLength(1);
  }, 300_000);

  it("treats the belief's run-setup defects as defects, not fallbacks", async () => {
    const { channel, team } = await view();
    const input = { format_id: FORMAT, seat: "p1" as const, team, channel, stream: new Uint8Array(32) };
    for (const code of ["format_mismatch", "item_clause_unsupported", "spread_sampler_required"]) {
      await expect(solvePreview(deps(setupFor(smallConfig(), { teams: refusing(code), solver: ports().solver })), input)).rejects.toThrow(PreviewSolveError);
    }
    const drawn: PreviewTeamPort = { ...ports().teams, open: () => ({ kind: "source", source: { draw: () => ({ kind: "refused", code: "spread_sampler_required" }) } }) };
    await expect(solvePreview(deps(setupFor(smallConfig(), { teams: drawn, solver: ports().solver })), input)).rejects.toMatchObject({ code: "belief_defect" });
  }, 300_000);

  it("falls back for too few worlds, and for a belief that refused every draw", async () => {
    const { channel, team } = await view();
    const input = { format_id: FORMAT, seat: "p1" as const, team, channel, stream: new Uint8Array(32) };
    const real = ports();
    const half: PreviewTeamPort = { ...real.teams, open: (o) => {
      const opened = real.teams.open(o);
      if (opened.kind === "refused") return opened;
      return { kind: "source", source: { draw: (stream, index) => (index === 0 ? { kind: "refused", code: "legality_redraws_exhausted" } : opened.source.draw(stream, index)) } };
    } };
    const few = await solvePreview(deps(setupFor(smallConfig(), { teams: half, solver: real.solver })), input);
    expect(few).toMatchObject({ kind: "fallback", reason: "too_few_worlds", worlds: { asked: 2, refused: 1 } });
    const none: PreviewTeamPort = { ...real.teams, open: () => ({ kind: "source", source: { draw: () => ({ kind: "refused", code: "legality_redraws_exhausted" }) } }) };
    const all = await solvePreview(deps(setupFor(smallConfig(), { teams: none, solver: real.solver })), input);
    expect(all).toMatchObject({ kind: "fallback", reason: "belief_refusal", code: "legality_redraws_exhausted", worlds: { asked: 2, refused: 2 } });
  }, 300_000);

  it("counts an invalid worker reply as a fallback and keeps the work it counted; every other failure is an incident", async () => {
    const reply = new Worker();
    reply.fault = "reply";
    const failed = await play(battle, { worker: reply });
    expect(failed.actor.report().counts.fallbacks).toMatchObject({ worker_failure: 1, engine_failure: 0 });
    // The failing query is the first one; the work the table counted before it failed is kept.
    expect(failed.actor.report().counts.payoff.worlds_asked).toBe(2);
    // Defects throw: the engine's hypothetical refusal of a query, a candidate shape the worker cannot parse, a prior that does not
    // match its candidates (a PayoffError), and a dead worker.
    for (const [fault, error] of [["engine", HypotheticalError], ["unsupported", UnsupportedCandidateError], ["prior", PayoffError], ["protocol", PolicyProtocolError]] as const) {
      const worker = new Worker();
      worker.fault = fault;
      await expect(play(battle, { worker }), fault).rejects.toThrow(error);
    }
  }, 300_000);

  it("throws, and does not count an unstarted world, when our own request or team does not match the hypothetical battle", async () => {
    const { channel, team } = await view();
    const base = { format_id: FORMAT, seat: "p1" as const, channel, stream: new Uint8Array(32) };
    const setup = setupFor();
    await expect(solvePreview(deps(setup), { ...base, team: team.slice(0, 5) })).rejects.toThrow(HypotheticalError);
    // A team the validator refuses as a whole is the one counted case.
    const source = (setup.teams.open({ opponent: readPreviewChannel("p1", channel).opponent, flags: [] }) as { source: Parameters<typeof startPreviewWorlds>[0]["source"] }).source;
    const view_ = readPreviewChannel("p1", channel);
    const input = { engine, vocabulary, format_id: FORMAT, source, count: 1, stream: new Uint8Array(32), seat: "p1" as const, own_team: team, channel_class: "synthetic" as const, own_request: view_.request };
    expect(startPreviewWorlds({ ...input, source: { draw: () => ({ kind: "team", redraws: 0, team: { sets: [] } }) } }).unstarted).toBe(1);
    expect(() => startPreviewWorlds({ ...input, own_team: team.slice(0, 5) })).toThrow(HypotheticalError);
  }, 300_000);

  it("exposes the config's budget for the evaluation wiring", () => {
    const setup = setupFor();
    expect(previewSolveBudgetMs(setup)).toBe(setup.config.budget_ms);
    expect(previewSolveBudgetMs(setupFor(smallConfig((c) => (c.budget_ms = 1234))))).toBe(1234);
  });

  it("skips the solve below the A2 clock floor and solves above it", async () => {
    const low = await play(battle, { bank_ms: () => 100 });
    expect(low.actor.report().counts).toMatchObject({ solved: 0, fallbacks: { clock_floor: 1 } });
    expect(low.worker.queries.some((q) => q.kind === "evaluate")).toBe(false);
    const enough = await play(battle, { bank_ms: () => 60_000 });
    expect(enough.actor.report().counts.solved).toBe(1);
    const unknown = await play(battle, { bank_ms: () => null });
    expect(unknown.actor.report().counts.solved).toBe(1);
  }, 300_000);

  it("values the entries after a rollout when the config says so", async () => {
    const setup = setupFor(smallConfig((c) => (c.leaf = { kind: "rollout", turns: 1 })));
    const g = await play(battle, { setup });
    expect(g.actor.report().counts.solved).toBe(1);
    expect(g.actor.report().counts.payoff.rollout_choices).toBeGreaterThan(0);
    // Another leaf is another solver identity and another policy id.
    expect(previewSolvePolicy(setup, manifest, pair, "argmax").policy_id).not.toBe(previewSolvePolicy(setupFor(), manifest, pair, "argmax").policy_id);
  }, 600_000);

  it("refuses a pair that reads candidate features or the opponent sheet, and a missing own view", async () => {
    const g = await play(battle);
    const options = (client: unknown): PreviewSolveActorOptions => ({ setup: setupFor(), client: client as PayoffClient, battle: "b", player: "p1", policy_id: g.policy_id, identity_hash: "d".repeat(64), seed: battle.seed, game: 0, channel_class: "synthetic", base: firstLegal });
    const featured = Object.assign(new Worker(), { featureSet: { feature_set: "probe-damage/1", feature_set_hash: "e".repeat(64) } });
    expect(() => previewSolveActor(options(featured))).toThrow(/candidate features/);
    const sheet = Object.assign(new Worker(), { sheetSource: "belief" });
    expect(() => previewSolveActor(options(sheet))).toThrow(/opponent sheet/);
    const actor = previewSolveActor(options(new Worker()));
    const request = { phase: "preview", request_seq: 1, player: "p1" } as never;
    await expect(actor(request, undefined)).rejects.toThrow(PolicyProtocolError);
    await expect(actor(request, { player: "p2", channel: [] } as never)).rejects.toThrow(PolicyProtocolError);
    await expect(actor(request, { player: "p1", channel: [] } as never)).rejects.toMatchObject({ code: "missing_context" });
    expect(() => readPreviewChannel("p1", [])).toThrow(/no line or no request/);
  }, 300_000);
});
