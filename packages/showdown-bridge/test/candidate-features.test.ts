import { readFileSync } from "node:fs";
import { beforeAll, describe, expect, it } from "vitest";
import { type CandidateFeatures, candidateFeatures, createFeatureMemo, FEATURE_SET, FeatureBuildError, type FeatureConfigs, type FeatureRow, featureSetHash, featureSetIdentity } from "../src/candidate-features.ts";
import { parseJointChoice } from "../src/damage-aware-choice.ts";
import { damageAwareConfigHash } from "../src/damage-aware-config.ts";
import { foesHit } from "../src/damage-aware-view.ts";
import { rollsValue } from "../src/damage-aware-score.ts";
import { mechanicsConfigHash } from "../src/mechanics-config.ts";
import { type ParsedMechanicsQuery } from "../src/mechanics-query.ts";
import { MechanicsProbeError, type MechanicsContext, probeMechanics } from "../src/mechanics-probe.ts";
import { MECHANICS_ADAPTER_FORMAT } from "../src/mechanics-result.ts";
import { playerChannel } from "../src/player-channel.ts";
import { readerSha256 } from "@areneon/public-state";
import { type BattleRecord, type HiddenVerdicts, type OmniscientLog, type PlayerChannel } from "../src/index.ts";
import { capture, type Captured, config, corpus, engine, mechanics, snapshot, stateOf, stubProbe, vocabulary } from "./damage-aware-support.ts";

const configs = (change: Partial<FeatureConfigs> = {}): FeatureConfigs => ({ vocabulary, config, mechanics, snapshot, data_class: "synthetic", probe: stubProbe(), ...change });

let captured: Captured[] = [];
beforeAll(async () => {
  captured = [...(await capture(corpus[0] as never)), ...(await capture(corpus[1] as never)), ...(await capture(corpus[3] as never))];
}, 120_000); // seeded real battles on the pinned engine: seconds on a loaded machine

const channelOf = (c: Captured): PlayerChannel => playerChannel({ schema_version: 1, data_class: "synthetic", player: c.context.player, lines: c.context.channel });
const candidatesOf = (c: Captured) => {
  const legal = c.request.legal as NonNullable<typeof c.request.legal>;
  return [...legal.choices, ...legal.pending].map((choice) => ({ choice, pending: legal.pending.includes(choice) }));
};
const build = (c: Captured, change: Partial<FeatureConfigs> = {}, memo?: ReturnType<typeof createFeatureMemo>, candidates = candidatesOf(c)): CandidateFeatures =>
  candidateFeatures(engine, configs(change), channelOf(c), c.context.team as never, candidates, memo);
const bytes = (features: CandidateFeatures): string => JSON.stringify(features);

type Active = { moves: { id: string; target?: string }[] }[];
const activeOf = (c: Captured): Active | undefined => (c.request.request as unknown as { active?: Active }).active;
const category = (id: string): string => (engine.sim.Dex.mod("champions").moves.get(id) as unknown as { category: string }).category;
const moveRequests = (): Captured[] => captured.filter((c) => c.request.phase === "move" && activeOf(c) !== undefined && (c.request.legal?.choices.length ?? 0) >= 2);

/** The test's own reading of which rows a request needs: every damaging move part of any candidate, against each foe it can hit. */
function expectedKeys(c: Captured, candidates = candidatesOf(c)): string[] {
  const state = stateOf(c);
  const letters = Object.keys(state.opponent.slots).sort();
  const alive = letters.flatMap((letter, position) => {
    const mon = state.opponent.roster[state.opponent.slots[letter] as number];
    return mon === undefined || mon.fainted || mon.hp_num === 0 ? [] : [letters.length - position];
  });
  const keys = new Set<string>();
  for (const { choice } of candidates) {
    parseJointChoice(choice).forEach((part, slot) => {
      if (part.kind !== "move") return;
      const move = (activeOf(c) as Active)[slot]?.moves[part.index - 1] as Active[number]["moves"][number];
      if (category(move.id) === "Status" || move.target === undefined) return;
      const hit = move.target === "allAdjacentFoes" || move.target === "allAdjacent" || move.target === "randomNormal" ? alive : alive.filter((loc) => loc === part.target);
      for (const loc of hit) keys.add(`${slot}/${part.index}/${loc}/${part.mega}`);
    });
  }
  return [...keys].sort();
}
const keyOf = (row: FeatureRow): string => `${row.slot}/${row.move}/${row.target}/${row.mega}`;
const sorted = (rows: FeatureRow[]): string[] => rows.map(keyOf).sort();

describe("rows (ADR-022 section 3)", () => {
  it("cover exactly the distinct damaging move parts of the candidates, never one per joint candidate", () => {
    let spread = 0;
    let mega = 0;
    let status = 0;
    let switches = 0;
    let checked = 0;
    for (const c of moveRequests()) {
      const features = build(c);
      expect(sorted(features.rows), `request ${c.request.request_seq}`).toEqual(expectedKeys(c));
      expect(new Set(features.rows.map(keyOf)).size).toBe(features.rows.length);
      expect(features.rows.length).toBeLessThanOrEqual(candidatesOf(c).length * 2);
      const parts = candidatesOf(c).flatMap((candidate) => parseJointChoice(candidate.choice));
      const targets = (c.request.request as unknown as { active: Active }).active.flatMap((a) => a.moves.map((m) => m.target));
      if (targets.some((t) => t === "allAdjacentFoes" || t === "allAdjacent") && features.rows.some((row) => features.rows.some((other) => other !== row && other.slot === row.slot && other.move === row.move && other.target !== row.target))) spread += 1;
      if (parts.some((p) => p.kind === "move" && p.mega) && features.rows.some((row) => row.mega)) mega += 1;
      if (targets.length > 0 && (c.request.request as unknown as { active: Active }).active.some((a) => a.moves.some((m) => category(m.id) === "Status"))) status += 1;
      if (parts.some((p) => p.kind === "switch")) switches += 1;
      checked += 1;
    }
    expect(checked).toBeGreaterThan(10);
    // the recorded battles hold a spread move into two foes, a Mega candidate, a status move and a switch
    expect([spread > 0, mega > 0, status > 0, switches > 0]).toEqual([true, true, true, true]);
  });

  it("holds no row for a status move, a switch or a pass", () => {
    for (const c of moveRequests()) {
      const active = (c.request.request as unknown as { active: Active }).active;
      for (const row of build(c).rows) expect(category((active[row.slot] as Active[number]).moves[row.move - 1]?.id as string)).not.toBe("Status");
    }
  });

  it("is in canonical order: slot, move position, target position, Mega flag", () => {
    for (const c of moveRequests()) {
      const rows = build(c).rows;
      const order = (r: FeatureRow): number[] => [r.slot, r.move, r.target, Number(r.mega)];
      expect([...rows].sort((a, b) => order(a).reduce((d, x, i) => d || x - (order(b)[i] as number), 0))).toEqual(rows);
    }
  });

  it("has integer values in the stand-in's known arithmetic", () => {
    const values: Record<string, number> = {};
    for (const c of moveRequests()) for (const a of activeOf(c) as Active) for (const m of a.moves) values[m.id] = 1 + ([...m.id].reduce((s, ch) => s + ch.charCodeAt(0), 0) % 90);
    const c = moveRequests()[0] as Captured;
    const features = build(c, { probe: stubProbe({ values }) });
    expect(features.rows.length).toBeGreaterThan(0);
    for (const row of features.rows) {
      const id = (activeOf(c) as Active)[row.slot]?.moves[row.move - 1]?.id as string;
      expect(row).toEqual({ slot: row.slot, move: row.move, target: row.target, mega: row.mega, kind: "computed", mean_ppm: (values[id] as number) * 10_000, max_ppm: (values[id] as number) * 10_000, min_ppm: (values[id] as number) * 10_000, ohko_rolls: null });
    }
  });

  it("clamps each roll at the target's hit points and takes the KO count from the probe", () => {
    const c = moveRequests()[0] as Captured;
    const rolls = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160];
    const probe = stubProbe({
      override: { damage: { kind: "computed", answer: { kind: "damage", no_crit: rolls, crit: rolls, crit_chance: { kind: "never" }, hp_current: 100, hp_max: 100, ko: { kind: "counts", ohko_no_crit: 7, ohko_crit: 9, thko_no_crit: 200 } } as never } },
    });
    const row = build(c, { probe }).rows[0] as Extract<FeatureRow, { kind: "computed" }>;
    expect(row).toMatchObject({ kind: "computed", mean_ppm: rollsValue(rolls, 100, 0), max_ppm: 1_000_000, min_ppm: 100_000, ohko_rolls: 7 });
    expect(row.mean_ppm).toBe(Math.floor(([10, 20, 30, 40, 50, 60, 70, 80, 90].reduce((a, b) => a + b, 0) + 100 * 7) * 1_000_000 / 1600));
  });

  it("turns an unsupported probe answer into an unsupported row with its reason, never a guess", () => {
    const c = moveRequests()[0] as Captured;
    const features = build(c, { probe: stubProbe({ override: { damage: { kind: "unsupported", reason: "random_effect", position: null } } }) });
    expect(features.rows.length).toBeGreaterThan(0);
    for (const row of features.rows) expect(row).toMatchObject({ kind: "unsupported", reason: "random_effect" });
    expect(features.rows.every((row) => !("mean_ppm" in row))).toBe(true);
  });

  it("has no rows for Team Preview, a forced switch or a wait", () => {
    const others = captured.filter((c) => c.request.phase !== "move");
    expect(others.length).toBeGreaterThan(0);
    for (const c of others) expect(build(c).rows).toEqual([]);
  });
});

describe("the declared state is the request's and the reader's", () => {
  const withRequest = (c: Captured, edit: (body: { side: { pokemon: { condition: string }[] } }) => void, extra: string[] = []): PlayerChannel => {
    const lines = [...c.context.channel];
    const request = JSON.parse((lines[lines.length - 1] as string).slice("|request|".length));
    edit(request);
    return playerChannel({ schema_version: 1, data_class: "synthetic", player: c.context.player, lines: [...lines.slice(0, -1), ...extra, `|request|${JSON.stringify(request)}`] });
  };

  it("declares an own Pokemon's current hit points and status from its request", () => {
    const c = moveRequests()[0] as Captured;
    const log: ParsedMechanicsQuery[] = [];
    const channel = withRequest(c, (body) => {
      (body.side.pokemon[0] as { condition: string }).condition = "37/100 par";
    });
    candidateFeatures(engine, configs({ probe: stubProbe({ log }) }), channel, c.context.team as never, candidatesOf(c));
    const attackers = log.flatMap((q) => (q.query.kind === "damage" ? [q.query.attacker] : []));
    expect(attackers.some((a) => a.hp !== null)).toBe(true);
    expect(attackers.filter((a) => a.status === "par").length).toBeGreaterThan(0);
  });

  it("leaves a fainted opposing active without rows", () => {
    const c = moveRequests().find((candidate) => Object.keys(stateOf(candidate).opponent.slots).length === 2 && build(candidate).rows.some((row) => row.target === 2)) as Captured;
    const state = stateOf(c);
    const mon = state.opponent.roster[state.opponent.slots[Object.keys(state.opponent.slots).sort()[0] as string] as number] as { name: string };
    const channel = withRequest(c, () => undefined, [`|faint|p${c.context.player === "p1" ? 2 : 1}a: ${mon.name}`]);
    const features = candidateFeatures(engine, configs(), channel, c.context.team as never, candidatesOf(c));
    expect(features.rows.some((row) => row.target === 2)).toBe(false);
    expect(features.rows.length).toBeLessThan(build(c).rows.length);
  });

  it("uses the pinned rules snapshot and the real probe when none is given", () => {
    const c = moveRequests()[0] as Captured;
    const { snapshot: _snapshot, probe: _probe, ...rest } = configs();
    const first = candidateFeatures(engine, rest, channelOf(c), c.context.team as never, candidatesOf(c).slice(0, 4));
    const again = candidateFeatures(engine, { ...rest, snapshot, probe: probeMechanics }, channelOf(c), c.context.team as never, candidatesOf(c).slice(0, 4));
    expect(bytes(first)).toBe(bytes(again));
  }, 60_000); // the real probe: 33 scenario clones per query
});

describe("a row equals a direct probe query on the same assumed sets", () => {
  it("asks one damage query per row and the rows are the real probe's own numbers", () => {
    const c = moveRequests().find((candidate) => build(candidate).rows.length >= 3) as Captured;
    const asked: { query: Parameters<typeof probeMechanics>[1]; result: ReturnType<typeof probeMechanics>["result"] }[] = [];
    const probe = (context: MechanicsContext, query: Parameters<typeof probeMechanics>[1]) => {
      const outcome = probeMechanics(context, query);
      asked.push({ query, result: outcome.result });
      return outcome;
    };
    const features = build(c, { probe });
    const damage = asked.filter((entry) => entry.query.query.kind === "damage");
    expect(damage.length).toBe(features.rows.length); // one damage query per row
    const direct = damage.map((entry) => probeMechanics({ engine, config: mechanics, snapshot }, entry.query).result);
    expect(direct).toEqual(damage.map((entry) => entry.result));
    const rows = features.rows.map((row) => (row.kind === "computed" ? row.mean_ppm : -1));
    const expected = direct.map((result) => {
      if (result.kind !== "computed" || result.answer.kind !== "damage") return -1;
      const { no_crit, hp_current } = result.answer;
      return Math.floor((no_crit.reduce((sum, roll) => sum + Math.min(roll, hp_current), 0) * 1_000_000) / (16 * hp_current));
    });
    // the queries come in canonical-row order only up to the first candidate's order: compare as multisets
    expect([...rows].sort((a, b) => a - b)).toEqual([...expected].sort((a, b) => a - b));
    const unsupported = features.rows.filter((row) => row.kind === "unsupported").length;
    expect(unsupported).toBe(direct.filter((result) => result.kind === "unsupported").length);
  }, 60_000); // the real probe: 33 scenario clones per query
});

describe("determinism", () => {
  it("gives the same bytes twice, with and without the memo, and in shuffled candidate order", () => {
    for (const c of moveRequests().slice(0, 12)) {
      const plain = bytes(build(c));
      expect(bytes(build(c))).toBe(plain);
      const memo = createFeatureMemo();
      expect(bytes(build(c, {}, memo))).toBe(plain);
      expect(bytes(build(c, {}, memo))).toBe(plain); // a warm memo
      const candidates = candidatesOf(c);
      expect(bytes(build(c, {}, undefined, [...candidates].reverse()))).toBe(plain);
      expect(bytes(build(c, {}, undefined, [...candidates.slice(3), ...candidates.slice(0, 3)]))).toBe(plain);
    }
  });

  it("answers a repeated query from the memo, keyed by the query hash", () => {
    const c = moveRequests()[0] as Captured;
    const memo = createFeatureMemo();
    build(c, {}, memo);
    const first = { ...memo.counts };
    expect(first.probe_runs).toBe(first.queries);
    build(c, {}, memo);
    expect(memo.counts.queries).toBe(first.queries * 2);
    expect(memo.counts.probe_runs).toBe(first.probe_runs); // nothing asked of the probe again
  });

  it("keeps the defender's hit points in the key: another shown HP is another query, not a memo hit", () => {
    const c = moveRequests().find((candidate) => Object.keys(stateOf(candidate).opponent.slots).length === 1) ?? (moveRequests()[0] as Captured);
    const state = stateOf(c);
    const letter = Object.keys(state.opponent.slots).sort()[0] as string;
    const mon = state.opponent.roster[state.opponent.slots[letter] as number] as { name: string };
    const hurt = [...c.context.channel];
    hurt.splice(hurt.length - 1, 0, `|-damage|p${c.context.player === "p1" ? 2 : 1}${letter}: ${mon.name}|1/100`);
    const memo = createFeatureMemo();
    build(c, {}, memo);
    const warm = memo.counts.probe_runs;
    candidateFeatures(engine, configs(), playerChannel({ schema_version: 1, data_class: "synthetic", player: c.context.player, lines: hurt }), c.context.team as never, candidatesOf(c), memo);
    expect(memo.counts.probe_runs).toBeGreaterThan(warm);
  });
});

describe("feature_set_hash", () => {
  const identity = () => featureSetIdentity(engine, configs());
  it("is the SHA-256 of the JCS of the six inputs, each from its own source", () => {
    const id = identity();
    expect(Object.keys(id).sort()).toEqual(["adapter_format", "assumed_set_config_hash", "feature_set", "mechanics_config_hash", "reader_sha256", "version_manifest"]);
    expect(id).toMatchObject({ feature_set: FEATURE_SET, mechanics_config_hash: mechanicsConfigHash(mechanics), assumed_set_config_hash: damageAwareConfigHash(config), adapter_format: MECHANICS_ADAPTER_FORMAT, reader_sha256: readerSha256() });
    expect(build(moveRequests()[0] as Captured).feature_set_hash).toBe(featureSetHash(id));
    expect(FEATURE_SET).toBe("probe-damage/1");
  });

  it("changes when any of its six inputs changes", () => {
    const base = featureSetHash(identity());
    const variants = {
      feature_set: { ...identity(), feature_set: "probe-damage/2" as never },
      mechanics_config_hash: featureSetIdentity(engine, configs({ mechanics: { ...mechanics, ko_blockers: { ...mechanics.ko_blockers, items: [] } } })),
      assumed_set_config_hash: featureSetIdentity(engine, configs({ config: { ...config, epsilon_ppm: config.epsilon_ppm + 1 } })),
      adapter_format: { ...identity(), adapter_format: MECHANICS_ADAPTER_FORMAT + 1 },
      reader_sha256: { ...identity(), reader_sha256: "0".repeat(64) },
      version_manifest: { ...identity(), version_manifest: { ...identity().version_manifest, showdown_commit: "1".repeat(40) } },
    };
    const hashes = Object.entries(variants).map(([name, id]) => {
      expect(featureSetHash(id), name).not.toBe(base);
      return featureSetHash(id);
    });
    expect(new Set(hashes).size).toBe(hashes.length);
  });
});

describe("foesHit", () => {
  it("builds the foes only for a target type that hits one", () => {
    const boom = (): never => {
      throw new Error("the foes were built");
    };
    for (const type of [undefined, "self", "adjacentAlly", "allySide", "all"]) expect(foesHit(type, null, boom)).toEqual([]);
    expect(() => foesHit("normal", null, () => [])).toThrow(/needs a target/);
    expect(() => foesHit("mystery", null, boom)).toThrow(/does not know/);
    expect(() => foesHit("allAdjacentFoes", null, boom)).toThrow(/foes were built/);
  });
});

describe("refusals", () => {
  const first = (): Captured => moveRequests()[0] as Captured;
  const code = (run: () => unknown): string | undefined => {
    try {
      run();
    } catch (error) {
      return error instanceof FeatureBuildError ? error.code : `other:${String(error)}`;
    }
    return undefined;
  };

  it("needs the channel to end in the request being answered", () => {
    const c = first();
    const lines = c.context.channel.slice(0, -1);
    const channel = playerChannel({ schema_version: 1, data_class: "synthetic", player: c.context.player, lines });
    expect(code(() => candidateFeatures(engine, configs(), channel, c.context.team as never, candidatesOf(c)))).toBe("no_request");
  });

  it("refuses when the reader flagged a line since the previous request", () => {
    const c = first();
    const lines = [...c.context.channel];
    lines.splice(lines.length - 1, 0, "|-notarealline|p1a: X|y");
    const channel = playerChannel({ schema_version: 1, data_class: "synthetic", player: c.context.player, lines });
    expect(code(() => candidateFeatures(engine, configs(), channel, c.context.team as never, candidatesOf(c)))).toBe("reader_flagged");
  });

  it("refuses when the probe refuses a declared state", () => {
    const probe = stubProbe({ throwOn: { damage: new MechanicsProbeError("the engine does not take that boost") } });
    expect(code(() => build(first(), { probe }))).toBe("probe_refused");
  });

  it("refuses when no assumed set can be built", () => {
    const sparse = { ...vocabulary, roster: [] } as never;
    expect(code(() => build(first(), { vocabulary: sparse }))).toBe("assumed_set");
  });

  it("lets a defect propagate, not a FeatureBuildError", () => {
    const c = first();
    expect(code(() => build(c, {}, undefined, [{ choice: "move one", pending: false }]))).toMatch(/^other:/);
    expect(code(() => build(c, {}, undefined, [{ choice: "pass", pending: false }]))).toMatch(/^other:.*different number of parts/);
    const stats = { kind: "computed", answer: { kind: "stats", stats: { hp: 1, atk: 1, def: 1, spa: 1, spd: 1, spe: 1 } } } as never;
    expect(code(() => build(c, { probe: stubProbe({ override: { damage: stats } }) }))).toMatch(/^other:.*no damage answer/);
  });
});

describe("the type rule (ADR-022 section 3, as for buildObservation)", () => {
  it("takes no opponent team, omniscient log, hidden verdicts, battle record or seed", () => {
    // Compile-time checks: `pnpm typecheck` fails if any of these compiles.
    const neverRun = (log: OmniscientLog, verdicts: HiddenVerdicts[], record: BattleRecord, seed: string, raw: { player: "p1"; lines: string[] }): void => {
      const team = [] as never;
      // @ts-expect-error: the omniscient log is not a PlayerChannel
      void candidateFeatures(engine, configs(), log, team, []);
      // @ts-expect-error: nor hidden verdicts
      void candidateFeatures(engine, configs(), verdicts, team, []);
      // @ts-expect-error: nor a battle record
      void candidateFeatures(engine, configs(), record, team, []);
      // @ts-expect-error: nor a seed
      void candidateFeatures(engine, configs(), seed, team, []);
      // @ts-expect-error: nor an unchecked log; only an adapter brands a channel
      void candidateFeatures(engine, configs(), raw, team, []);
      // @ts-expect-error: the configs hold no opponent team
      void candidateFeatures(engine, { vocabulary, config, mechanics, data_class: "synthetic", opponent_team: [] }, undefined as never, team, []);
    };
    expect(neverRun).toBeTypeOf("function");
  });

  it("imports no runner, omniscient or seed module at run time", () => {
    const source = readFileSync(new URL("../src/candidate-features.ts", import.meta.url), "utf8");
    const imports = [...source.matchAll(/^import .* from "(.*)";$/gm)].map((m) => m[1] as string);
    expect(imports.filter((path) => /runner|omniscient|hidden|league-seed|progress-seed/.test(path))).toEqual([]);
  });
});
