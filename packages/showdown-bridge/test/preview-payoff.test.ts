// The Team Preview payoff table (ADR-029 section 3): strategies dedupe to 180, the table on fixed hypothetical worlds equals a
// table worked out here from the teams alone, the first-request leaf cannot see a back pair and the rollout leaf can, a table is a
// function of its inputs, and the counts say what happened.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { GOLDEN_DIR, GOLDEN_FORMAT, syntheticTeam, TEAMS_DIR } from "../src/golden.ts";
import {
  backDraw,
  BACK_DRAW_PURPOSE,
  buildPayoffTable,
  type Engine,
  evaluateHypothetical,
  HypotheticalError,
  type HypotheticalPreview,
  hypotheticalCandidates,
  hypotheticalRequest,
  importPaste,
  loadEngine,
  PAYOFF_LEAF_FORMAT,
  type PayoffConfig,
  PayoffError,
  type PayoffClient,
  type PokemonSet,
  type PreviewStrategy,
  selectOwn,
  startHypothetical,
  strategiesOf,
} from "../src/index.ts";
import { decodeRequest } from "../src/request.ts";

const STREAM = new Uint8Array(32).fill(9);
let engine: Engine;
let own: PokemonSet[];
let worlds: PokemonSet[][];
/** Our REAL Team Preview request: the golden battle's first request for p1, whose team is the synthetic team. */
let REAL: string;
beforeAll(() => {
  const log = JSON.parse(readFileSync(join(GOLDEN_DIR, "channels/p1.json"), "utf8")) as { channel: { lines: string[] } };
  REAL = (log.channel.lines.find((line) => line.startsWith("|request|")) as string).slice("|request|".length);
  engine = loadEngine();
  own = syntheticTeam();
  const base = importPaste(engine, readFileSync(join(TEAMS_DIR, "legal-opponent.txt"), "utf8"));
  const named = base.map((set, i) => ({ ...set, name: ["Alpha", "Bravo", "Chuck", "Delta", "Echo", "Foxtrot"][i] as string }));
  worlds = [base, named];
});

const start = (world: number, team: PokemonSet[] = worlds[world] as PokemonSet[], overrides: { own_team?: PokemonSet[] } = {}): { index: number; preview: HypotheticalPreview } => ({
  index: world,
  preview: startHypothetical({ engine, format_id: GOLDEN_FORMAT, seat: "p1", own_team: overrides.own_team ?? own, world_team: team, stream: STREAM, world, data_class: "synthetic", own_request: REAL }),
});

/** The stand-in Team Preview prior: a fixed integer weight per candidate index. */
const previewWeight = (i: number): number => ((i * 37) % 11) + 1;
/** The stand-in value: the characters of the `|switch|` lines' nicknames and species, so it moves with the leads and the world. */
function valueOf(log: readonly string[]): number {
  let sum = 0;
  for (const line of log) {
    if (!line.startsWith("|switch|")) continue;
    const [, , ident, details] = line.split("|");
    const nick = (ident as string).split(": ")[1] as string;
    for (const ch of `${nick}|${(details as string).split(",")[0]}`) sum += ch.charCodeAt(0);
  }
  return (sum % 1_200_000) - 100_000;
}

function standIn(options: { battlePrior?: (n: number) => number[]; value?: (log: readonly string[]) => number } = {}): PayoffClient & { queries: number } {
  const client = {
    queries: 0,
    featureSet: null,
    async evaluate(_battle: string, query: Parameters<PayoffClient["evaluate"]>[1]) {
      client.queries += 1;
      const n = query.candidates.length;
      const preview = query.observation.kind === "preview";
      const prior = preview ? query.candidates.map((_c, i) => previewWeight(i)) : (options.battlePrior?.(n) ?? query.candidates.map(() => 1));
      return { identity: query.identity, prior_ppm: prior as [number, ...number[]], value_ppm: (options.value ?? valueOf)(query.observation.log) };
    },
  };
  return client as unknown as PayoffClient & { queries: number };
}

const config = (change: Partial<PayoffConfig> = {}): PayoffConfig => ({ k_own: 3, k_opp: 3, own_rule: "distinct-leads", leaf: { kind: "first-request" }, ...change });
const input = (client: PayoffClient, worldList: { index: number; preview: HypotheticalPreview }[], cfg: PayoffConfig) => ({ config: cfg, worlds: worldList, client, battle: "b", policy_id: "test@policy", own_request: REAL, stream: STREAM });

/** Every 4-of-6 team order, the way the engine lists them: lexicographic over the position digits. */
function orders(): string[] {
  const out: string[] = [];
  for (let a = 1; a <= 6; a += 1) for (let b = 1; b <= 6; b += 1) for (let c = 1; c <= 6; c += 1) for (let d = 1; d <= 6; d += 1) if (new Set([a, b, c, d]).size === 4) out.push(`team ${a}${b}${c}${d}`);
  return out;
}

describe("strategies", () => {
  it("dedupe 360 orders of four of six to 180 strategies, probability the sum of the two orders, played as the likelier order", () => {
    const preview = start(0).preview;
    const candidates = hypotheticalCandidates(preview.own_channel);
    expect(candidates.map((c) => c.order)).toEqual(orders());
    const strategies = strategiesOf(candidates, candidates.map((_c, i) => previewWeight(i)));
    expect(strategies).toHaveLength(180);
    expect(new Set(strategies.map((s) => s.key)).size).toBe(180);
    expect(strategies.reduce((sum, s) => sum + s.ppm, 0)).toBe(candidates.reduce((sum, _c, i) => sum + previewWeight(i), 0));
    const lead = strategies.find((s) => s.key === "12|34") as PreviewStrategy;
    const a = orders().indexOf("team 1234");
    const b = orders().indexOf("team 1243");
    expect(lead.ppm).toBe(previewWeight(a) + previewWeight(b));
    expect(lead.back).toEqual([3, 4]);
    expect(lead.order).toBe(previewWeight(b) > previewWeight(a) ? "team 1243" : "team 1234");
  });

  it("plays the first order on a tie and refuses shapes it does not know", () => {
    const tie = strategiesOf([{ order: "team 1234" }, { order: "team 1243" }], [5, 5]);
    expect(tie).toHaveLength(1);
    expect(tie[0]?.order).toBe("team 1234");
    expect(() => strategiesOf([{ order: "team 1234" }], [])).toThrow(PayoffError);
    expect(() => strategiesOf([{ order: "default" }], [1])).toThrow(/not a Team Preview order/);
    expect(() => strategiesOf([{ order: "team 12" }], [1])).toThrow(/not a Team Preview order/);
    expect(() => strategiesOf([{ order: "team 1123" }], [1])).toThrow(/repeats/);
  });

  it("selects our top k by probability, distinct leads skipping a repeated ordered lead pair", () => {
    const strategies = strategiesOf(
      [{ order: "team 1234" }, { order: "team 1256" }, { order: "team 2134" }, { order: "team 1243" }, { order: "team 3456" }],
      [9, 8, 7, 6, 1],
    );
    // 1234 and 1243 are one strategy (leads 12, back 34): 15; 1256: 8; 2134: 7; 3456: 1.
    expect(selectOwn(strategies, 3, "prior-top-k").map((s) => s.key)).toEqual(["12|34", "12|56", "21|34"]);
    expect(selectOwn(strategies, 3, "distinct-leads").map((s) => s.key)).toEqual(["12|34", "21|34", "34|56"]);
    expect(selectOwn(strategies, 9, "distinct-leads")).toHaveLength(3);
  });
});

describe("the table", () => {
  it("equals the table worked out from the teams: floor mean over worlds of the clamped value, columns by summed lead mass", async () => {
    const client = standIn();
    const ws = [start(0), start(1)];
    const table = await buildPayoffTable(input(client, ws, config()));

    // Our rows, worked out here: the 180 strategies by weight, distinct leads, top 3.
    const all = orders();
    const weightOf = new Map<string, number>();
    const rankOf = new Map<string, number>();
    all.forEach((order, i) => {
      const d = order.slice(5);
      const key = `${d.slice(0, 2)}|${[d[2], d[3]].sort().join("")}`;
      weightOf.set(key, (weightOf.get(key) ?? 0) + previewWeight(i));
      if (!rankOf.has(key)) rankOf.set(key, i);
    });
    const ranked = [...weightOf.keys()].sort((a, b) => (weightOf.get(b) as number) - (weightOf.get(a) as number) || (rankOf.get(a) as number) - (rankOf.get(b) as number));
    const seen = new Set<string>();
    const expectedOwn: string[] = [];
    for (const key of ranked) {
      if (seen.has(key.slice(0, 2))) continue;
      seen.add(key.slice(0, 2));
      expectedOwn.push(key);
      if (expectedOwn.length === 3) break;
    }
    expect(table.own.map((s) => s.key)).toEqual(expectedOwn);

    // Their columns: the summed weight by ordered lead pair (the same prior in both worlds), top 3.
    const leadSum = new Map<string, number>();
    const leadRank = new Map<string, number>();
    all.forEach((order, i) => {
      const lead = order.slice(5, 7);
      leadSum.set(lead, (leadSum.get(lead) ?? 0) + previewWeight(i));
      if (!leadRank.has(lead)) leadRank.set(lead, i);
    });
    const leads = [...leadSum.keys()].sort((a, b) => (leadSum.get(b) as number) - (leadSum.get(a) as number) || (leadRank.get(a) as number) - (leadRank.get(b) as number)).slice(0, 3);
    expect(table.opp.map((o) => o.leads.join(""))).toEqual(leads);
    expect(table.opp.map((o) => o.mean_ppm)).toEqual(leads.map((l) => leadSum.get(l) as number));

    // Entries, from the species and nicknames of the four leads alone.
    const ident = (set: PokemonSet): string => `${set.name || set.species}|${set.species}`;
    const sum = (text: string): number => [...text].reduce((total, ch) => total + ch.charCodeAt(0), 0);
    let clamps = 0;
    const expected = table.own.map((s) =>
      leads.map((lead) => {
        const values = (worlds as PokemonSet[][]).map((team) => {
          const heads = [...s.leads.map((p) => own[p - 1] as PokemonSet), ...[...lead].map((p) => team[Number(p) - 1] as PokemonSet)];
          const raw = (heads.reduce((total, set) => total + sum(ident(set)), 0) % 1_200_000) - 100_000;
          if (raw < 0 || raw > 1_000_000) clamps += 1;
          return Math.min(1_000_000, Math.max(0, raw));
        });
        return Math.floor(values.reduce((a, b) => a + b, 0) / values.length);
      }),
    );
    expect(table.entries).toEqual(expected);
    expect(table.contributors).toEqual(table.own.map(() => leads.map(() => 2)));
    expect(table.complete).toBe(true);
    expect(table.matrix).toEqual(expected);
    expect(table.counts).toMatchObject({ worlds_asked: 2, worlds_used: 2, preview_queries: 3, entry_attempts: 18, entries: 18, value_queries: 18, clamped: clamps, missing_entries: 0, engine_refusals: 0, back_draws: 0 });
    expect(client.queries).toBe(3 + 18);
    expect(PAYOFF_LEAF_FORMAT).toBe(1);
  }, 120_000);

  it("is the same for the same inputs, rollout leaf included", async () => {
    const run = async () => buildPayoffTable(input(standIn({ battlePrior: (n) => Array.from({ length: n }, (_x, i) => (i * 5) % 7) }), [start(0), start(1)], config({ k_own: 2, k_opp: 2, leaf: { kind: "rollout", turns: 1 } })));
    const a = await run();
    const b = await run();
    expect(JSON.stringify(b)).toBe(JSON.stringify(a));
    expect(a.counts.back_draws).toBe(2 * 2 * 2);
    expect(a.counts.rollout_choices).toBeGreaterThan(0);
  }, 120_000);

  it("counts the clamps below and above, game ends, engine refusals, non-move requests and missing entries", async () => {
    const ws = [start(0)];
    const cfg = config({ k_own: 2, k_opp: 2 });
    const low = await buildPayoffTable(input(standIn({ value: () => -5 }), ws, cfg));
    expect(low.counts).toMatchObject({ clamped: 4, clamped_low: 4, clamped_high: 0 });
    expect(low.entries.flat()).toEqual([0, 0, 0, 0]);
    const high = await buildPayoffTable(input(standIn({ value: () => 1_200_000 }), ws, cfg));
    expect(high.counts).toMatchObject({ clamped: 4, clamped_high: 4 });
    expect(high.entries.flat()).toEqual([1_000_000, 1_000_000, 1_000_000, 1_000_000]);

    const real = (ws[0] as { preview: HypotheticalPreview }).preview;
    const wrap = (play: HypotheticalPreview["play"]) => [{ index: 0, preview: { ...real, play } as HypotheticalPreview }];
    const refused = await buildPayoffTable(input(standIn(), wrap(() => ({ kind: "refused", reason: "x" })), cfg));
    expect(refused.counts).toMatchObject({ engine_refusals: 1, worlds_dropped: 1, entries: 0, missing_entries: 4 });
    expect(refused.complete).toBe(false);
    expect(refused.matrix).toBeNull();
    expect(refused.entries.flat()).toEqual([null, null, null, null]);
    const ended = await buildPayoffTable(input(standIn(), wrap(() => ({ kind: "ended", outcome: "tie" })), cfg));
    expect(ended.counts).toMatchObject({ game_end_entries: 4, entries: 4 });
    expect(ended.entries.flat()).toEqual([500_000, 500_000, 500_000, 500_000]);
    const waiting = await buildPayoffTable(
      input(standIn(), wrap((o, f) => {
        const turn = real.play(o, f);
        return turn.kind === "request" ? { kind: "request", channel: turn.channel, request: { ...turn.request, phase: "wait" } as typeof turn.request } : turn;
      }), cfg),
    );
    expect(waiting.counts).toMatchObject({ not_move_request: 1, worlds_dropped: 1, missing_entries: 4 });
  }, 120_000);

  it("refuses bad configs, no worlds and a pair that reads candidate features", async () => {
    const ws = [start(0)];
    const client = standIn();
    await expect(buildPayoffTable(input(client, ws, config({ k_own: 0 })))).rejects.toThrow(PayoffError);
    await expect(buildPayoffTable(input(client, ws, config({ own_rule: "x" as "prior-top-k" })))).rejects.toThrow(/own rule/);
    await expect(buildPayoffTable(input(client, ws, config({ leaf: { kind: "rollout", turns: 0 } })))).rejects.toThrow(/at least one turn/);
    await expect(buildPayoffTable(input(client, ws, config({ leaf: { kind: "x" } as unknown as PayoffConfig["leaf"] })))).rejects.toThrow(/unknown leaf/);
    await expect(buildPayoffTable(input(client, [], config()))).rejects.toThrow(/no worlds/);
    const features = { ...client, featureSet: { feature_set: "probe-damage/1", feature_set_hash: "a".repeat(64) } } as unknown as PayoffClient;
    await expect(buildPayoffTable(input(features, ws, config()))).rejects.toThrow(HypotheticalError);
  });
});

describe("what a leaf can see of the opponent's back pair (B1)", () => {
  it("the first-request leaf gives the same observation for two back pairs with the same leads, the rollout leaf can differ", async () => {
    const preview = start(0).preview;
    const log = async (turn: ReturnType<HypotheticalPreview["play"]> | Awaited<ReturnType<HypotheticalPreview["rollout"]>>): Promise<readonly string[]> => {
      if (turn.kind !== "request") throw new Error(turn.kind);
      const client = standIn();
      const seen: string[] = [];
      const probe = { ...client, evaluate: async (b: string, q: Parameters<PayoffClient["evaluate"]>[1]) => (seen.push(...q.observation.log), client.evaluate(b, q)) } as PayoffClient;
      await evaluateHypothetical(probe, "b", await preview.observation(turn.channel), "p");
      return seen;
    };
    const mine = "team 1234";
    const a = await log(preview.play(mine, "team 1256"));
    const b = await log(preview.play(mine, "team 1265"));
    const c = await log(preview.play(mine, "team 1234"));
    expect(b).toEqual(a);
    expect(c).toEqual(a);

    // Both seats take the last listed candidate (a switch when one is offered): the foe's back pair then shows.
    const last = async (observation: Parameters<Parameters<HypotheticalPreview["rollout"]>[3]>[0]): Promise<string> => {
      const { candidates } = await evaluateHypothetical(standIn(), "b", observation, "p");
      return (candidates[candidates.length - 1] as { order: string }).order;
    };
    const r1 = await log(await preview.rollout(mine, "team 1256", 1, last));
    const r2 = await log(await preview.rollout(mine, "team 1234", 1, last));
    expect(r2).not.toEqual(r1);
  }, 120_000);

  it("a rollout plays on through waits and forced switches, and refuses fewer than one turn", async () => {
    const preview = start(0).preview;
    const choose = async (observation: Parameters<Parameters<HypotheticalPreview["rollout"]>[3]>[0]): Promise<string> => {
      const { candidates } = await evaluateHypothetical(standIn(), "b", observation, "p");
      return (candidates.find((c) => !c.pending) as { order: string }).order;
    };
    await expect(preview.rollout("team 1234", "team 1234", 0, choose)).rejects.toThrow(/at least one turn/);
    const turn = await preview.rollout("team 1234", "team 5612", 3, choose);
    expect(turn.kind).toBe("request");
    const refused = await preview.rollout("team 1234", "team 1111", 2, choose);
    expect(refused.kind).toBe("refused");
  }, 120_000);
});

describe("whole worlds, the step cap and our real request", () => {
  it("drops a world for every cell when one of its entries is refused, so all cells average the same worlds", async () => {
    const good = start(0);
    const bad = start(1);
    let calls = 0;
    const flaky = { index: 1, preview: { ...bad.preview, play: (o: string, f: string) => (++calls === 3 ? ({ kind: "refused", reason: "x" } as const) : bad.preview.play(o, f)) } as HypotheticalPreview };
    const table = await buildPayoffTable(input(standIn(), [good, flaky], config({ k_own: 2, k_opp: 2 })));
    expect(table.counts).toMatchObject({ engine_refusals: 1, worlds_dropped: 1, entries: 4 });
    expect(table.contributors).toEqual([[1, 1], [1, 1]]);
    const alone = await buildPayoffTable(input(standIn(), [good], config({ k_own: 2, k_opp: 2 })));
    expect(table.entries).toEqual(alone.entries);
  }, 120_000);

  it("values a rollout that hits the step cap at the request it reached, counted apart", async () => {
    const world = start(0);
    const cfg = config({ k_own: 1, k_opp: 1, leaf: { kind: "rollout", turns: 6 } });
    const capped = await buildPayoffTable({ ...input(standIn(), [world], cfg), max_rollout_steps: 1 });
    expect(capped.counts).toMatchObject({ rollout_capped: 1, value_queries: 1, entries: 1, engine_refusals: 0, worlds_dropped: 0 });
    expect(capped.complete).toBe(true);
    const value = capped.entries[0]?.[0] as number;
    expect(value).toBeGreaterThanOrEqual(0);
    expect(value).toBeLessThanOrEqual(1_000_000);
    const uncapped = await buildPayoffTable(input(standIn(), [world], cfg));
    expect(uncapped.counts.rollout_capped).toBe(0);
  }, 120_000);

  it("gives our side the real Team Preview request in every world whatever the seed would draw, and refuses a world that does not", async () => {
    const real = decodeRequest(REAL).request_hash;
    for (let w = 0; w < 4; w += 1) expect(hypotheticalRequest(start(w, worlds[w % 2]).preview.own_channel).request_hash).toBe(real);
    const unfixed = [0, 1, 2, 3].map((w) => startHypothetical({ engine, format_id: GOLDEN_FORMAT, seat: "p1", own_team: own, world_team: worlds[0] as PokemonSet[], stream: STREAM, world: w, data_class: "synthetic" }));
    expect(new Set(unfixed.map((p) => hypotheticalRequest(p.own_channel).request_hash)).size).toBeGreaterThan(1);
    await expect(buildPayoffTable(input(standIn(), [{ index: 0, preview: unfixed.find((p) => hypotheticalRequest(p.own_channel).request_hash !== real) as HypotheticalPreview }], config()))).rejects.toThrow(/another Team Preview request/);
    await expect(buildPayoffTable({ ...input(standIn(), [start(0)], config()), own_request: REAL.replace("L50", "L51") })).rejects.toThrow(/another Team Preview request/);
  }, 120_000);

  it("refuses an own request that is not a request or does not match our team", () => {
    const base = { engine, format_id: GOLDEN_FORMAT, seat: "p1" as const, own_team: own, world_team: worlds[0] as PokemonSet[], stream: STREAM, world: 0, data_class: "synthetic" as const };
    expect(() => startHypothetical({ ...base, own_request: "not json" })).toThrow(/not a request/);
    expect(() => startHypothetical({ ...base, own_request: "{}" })).toThrow(/does not match our team/);
    expect(() => startHypothetical({ ...base, own_request: JSON.stringify({ side: { pokemon: [{ details: "" }] } }) })).toThrow(/does not match our team/);
  });
});

describe("the back pair draw", () => {
  it("is SHA-256 over the JCS of {leads, purpose, stream, world} scaled to the total: vectors computed with Python hashlib", () => {
    // Python: h = hashlib.sha256(json.dumps({"leads": [1, 2], "purpose": "preview-opp-back/1", "stream": "09" * 32, "world": W},
    //   sort_keys=True, separators=(",", ":")).encode()).hexdigest(); (int(h[:16], 16) * TOTAL) >> 64
    expect(BACK_DRAW_PURPOSE).toBe("preview-opp-back/1");
    expect(backDraw(STREAM, 0, [1, 2], 1_000_000)).toBe(BACK_VECTORS[0]);
    expect(backDraw(STREAM, 3, [1, 2], 1_000_000)).toBe(BACK_VECTORS[1]);
    expect(backDraw(STREAM, 3, [2, 1], 77)).toBe(BACK_VECTORS[2]);
  });
});

const BACK_VECTORS = [382325, 131899, 10];
