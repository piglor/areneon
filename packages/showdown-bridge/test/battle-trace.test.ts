// The forced-draw trace recorder (ADR-028 section 5 F0, #1462) and the Showdown battle backend: the random sources against the
// engine's own generator, the channel split, gate 3 (a trace equals the runner's own record of the battle: omniscient log,
// channels, requests, legal sets, hidden verdicts, winner and turns), determinism, and the backend's self-replay and refusals.
import { beforeAll, describe, expect, it } from "vitest";
import { committedBattles } from "../src/battle-trace-corpus.ts";
import { addEmitted, agreesWithRecord, emitted, forcingSource, interceptedSource, maskOf, recordBattleTrace, recordingSource, stateHash, TRACE_FORMAT, type Draw, TraceRecordError } from "../src/battle-trace.ts";
import { ReplayError, showdownBattleBackend } from "../src/battle-replay.ts";
import { loadEngine } from "../src/engine.ts";
import { decodeRequest } from "../src/request.ts";
import { loadRulesSnapshot } from "../src/rules-load.ts";
import { type BattleRecord, ForfeitSignal } from "../src/runner.ts";
import { randomLegalActor } from "../src/random-legal.ts";

const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const PRNG = engine.sim.PRNG as unknown as new (seed: string) => Record<string, (...a: unknown[]) => unknown>;
const SEED = "sodium,000102030405060708090a0b0c0d0e0f";

describe("the random sources", () => {
  it("a recording source answers exactly what the engine's generator answers, and logs every call", () => {
    const natural = new PRNG(SEED);
    const draws: Draw[] = [];
    const source = recordingSource(new PRNG(SEED), () => draws);
    const items = ["a", "b", "c", "d"];
    const shuffled = [...items];
    const naturalShuffled = [...items];
    expect(source.random()).toBe(natural.random?.());
    expect(source.random(16)).toBe(natural.random?.(16));
    expect(source.random(85, 101)).toBe(natural.random?.(85, 101));
    expect(source.randomChance(1, 3)).toBe(natural.randomChance?.(1, 3));
    expect(source.sample(items)).toBe(natural.sample?.(items));
    source.shuffle(shuffled);
    natural.shuffle?.(naturalShuffled);
    expect(shuffled).toEqual(naturalShuffled);
    source.shuffle(shuffled, 1, 3);
    natural.shuffle?.(naturalShuffled, 1, 3);
    expect(shuffled).toEqual(naturalShuffled);
    expect(source.getSeed()).toBe(natural.getSeed?.());
    expect(typeof source.startingSeed).toBe("string");
    expect(source[Symbol.toStringTag as never]).toBeUndefined();
    expect(draws.map((d) => [d.method, d.args])).toEqual([
      ["random", []],
      ["random", ["16"]],
      ["random", ["85", "101"]],
      ["randomChance", ["1", "3"]],
      ["sample", ["4"]],
      ["shuffle", ["4"]],
      ["shuffle", ["4", "1", "3"]],
    ]);
    expect(draws.every((d) => d.result.every((r) => typeof r === "string"))).toBe(true);
  });

  it("records a member it does not know as a draw, with its arguments and result", () => {
    const draws: Draw[] = [];
    const source = recordingSource({ next: (n: number) => n + 1, getSeed: () => "s" }, () => draws);
    expect(source.next(41)).toBe(42);
    expect(draws).toEqual([{ method: "next", args: ["41"], result: ["42"] }]);
  });

  it("a forcing source answers from the loaded draws where the call matches, else from the generator, and lists every call", () => {
    const natural = new PRNG(SEED);
    const forcing = forcingSource(new PRNG(SEED));
    const items = ["a", "b", "c"];
    forcing.load([
      { method: "random", args: ["16"], result: ["15"] },
      { method: "randomChance", args: ["1", "2"], result: ["false"] },
      { method: "sample", args: ["3"], result: ["2"] },
      { method: "shuffle", args: ["3"], result: ["2", "1", "0"] },
      { method: "random", args: ["9"], result: ["3"] },
    ]);
    expect(forcing.source.random(16)).toBe(15);
    expect(forcing.source.randomChance(1, 2)).toBe(false);
    expect(forcing.source.sample(items)).toBe("c");
    const order = [...items];
    forcing.source.shuffle(order);
    expect(order).toEqual(["c", "b", "a"]);
    // Another call than the next forced one, and one past the end, get the generator's own answer.
    natural.random?.(16);
    natural.randomChance?.(1, 2);
    natural.sample?.([0, 1, 2]);
    natural.shuffle?.([0, 1, 2]);
    expect(forcing.source.random(10)).toBe(natural.random?.(10));
    expect(forcing.source.random(10)).toBe(natural.random?.(10));
    expect(forcing.requests().map((d) => [d.method, d.args])).toEqual([
      ["random", ["16"]],
      ["randomChance", ["1", "2"]],
      ["sample", ["3"]],
      ["shuffle", ["3"]],
      ["random", ["10"]],
      ["random", ["10"]],
    ]);
    // The generator advanced in lock step, so the serialised state is the natural one.
    expect(forcing.source.getSeed()).toBe(natural.getSeed?.());
    forcing.load([]);
    expect(forcing.requests()).toEqual([]);
  });
});

describe("the channel split", () => {
  it("gives each channel what the player streams give it", () => {
    const update = ["|move|p1a: X|Tackle|p2a: Y", "|split|p1", "|-damage|p1a: X|50/150", "|-damage|p1a: X|33/100", "|split|p2", "|-heal|p2a: Y|10/20", "", "|t:|123"].join("\n");
    expect(emitted("update", update)).toEqual({
      omniscient: ["|move|p1a: X|Tackle|p2a: Y", "|-damage|p1a: X|50/150", "|-heal|p2a: Y|10/20"],
      p1: ["|move|p1a: X|Tackle|p2a: Y", "|-damage|p1a: X|50/150"],
      p2: ["|move|p1a: X|Tackle|p2a: Y", "|-damage|p1a: X|33/100", "|-heal|p2a: Y|10/20"],
    });
    expect(emitted("update", "|split|p1\n|secret\n")).toEqual({ omniscient: ["|secret"], p1: ["|secret"], p2: [""] });
    expect(emitted("sideupdate", 'p2\n|request|{"wait":true}')).toEqual({ omniscient: [], p1: [], p2: ['|request|{"wait":true}'] });
    expect(emitted("sideupdate", "p1\n|error|[Invalid choice] x")).toEqual({ omniscient: [], p1: ["|error|[Invalid choice] x"], p2: [] });
    expect(emitted("end", "{}")).toEqual({ omniscient: [], p1: [], p2: [] });
    const into = { channels: { p1: [] as string[], p2: [] as string[] }, omniscient: [] as string[], bodies: { p1: [] as string[], p2: [] as string[] } };
    addEmitted(into, "sideupdate", 'p1\n|request|{"wait":true}');
    expect(into.bodies.p1).toEqual(['{"wait":true}']);
  });
});

const record = async (corpus: string, at: number[] = [0]) => Promise.all(committedBattles([corpus]).filter((_, i) => at.includes(i)).map(({ spec }) => recordBattleTrace(engine, spec)));
let golden: Awaited<ReturnType<typeof record>>;
let random: Awaited<ReturnType<typeof record>>;
let unavailable: Awaited<ReturnType<typeof record>>;
beforeAll(async () => {
  golden = await record("golden");
  // b15 reaches the 30-turn cap (the runner's forcetie).
  random = await record("random", [0, 1, 2, 15]);
  unavailable = await record("unavailable", [0, 1]);
}, 120_000);

describe("gate 3: a trace is the runner's own record of the battle", () => {
  it("holds the same omniscient log, channels, requests, legal sets, hidden verdicts, winner and turns", () => {
    for (const { trace, record: r } of [...golden, ...random, ...unavailable]) {
      expect(trace.trace_format).toBe(TRACE_FORMAT);
      expect(trace.steps.flatMap((s) => s.omniscient)).toEqual(r.omniscient_log);
      for (const seat of ["p1", "p2"] as const) {
        expect(trace.steps.flatMap((s) => s.channels[seat])).toEqual(r.channels[seat]);
        const requests = trace.steps.flatMap((s) => s.requests[seat]);
        expect(requests.map((q) => q.request_hash)).toEqual(r.requests[seat].map((q) => q.request_hash));
      }
      expect(trace.outcome).toEqual({ winner: r.winner, turns: r.turns });
      expect(trace.team_hashes).toEqual(r.team_hashes);
      expect(trace.steps[0]?.actions).toEqual([]);
      const choices = trace.steps.flatMap((s) => s.actions).filter((a) => a.kind === "choice");
      expect(choices.map((a) => (a.kind === "choice" ? [a.seat, a.choice] : null))).toEqual(r.choices.map((c) => [c.player, c.choice]));
    }
    expect(random.some(({ trace }) => trace.steps.some((s) => s.actions.some((a) => a.kind === "command")))).toBe(true);
    expect(unavailable.some(({ record: r }) => r.hidden_verdicts.length > 0)).toBe(true);
  });

  it("records the same bytes twice", async () => {
    const [again] = await record("random");
    expect(JSON.stringify(again?.trace)).toBe(JSON.stringify(random[0]?.trace));
  });

  it("takes a rules snapshot", async () => {
    const { spec } = committedBattles(["golden"])[0] as { spec: Parameters<typeof recordBattleTrace>[1] };
    const { trace } = await recordBattleTrace(engine, { ...spec, snapshot: loadRulesSnapshot(engine.commit) });
    expect(JSON.stringify(trace)).toBe(JSON.stringify(golden[0]?.trace));
  });

  it("refuses a battle an actor forfeits", async () => {
    const { spec } = committedBattles(["random"])[0] as { spec: Parameters<typeof recordBattleTrace>[1] };
    const forfeiting = randomLegalActor(1);
    let n = 0;
    const actors = { p1: spec.actors.p1, p2: ((request: never, ctx: never) => (++n > 3 ? Promise.reject(new ForfeitSignal("crash", "x")) : forfeiting(request, ctx))) as never };
    await expect(recordBattleTrace(engine, { ...spec, actors })).rejects.toThrow(TraceRecordError);
  });

  it("refuses a record that does not say what the trace says", () => {
    const { trace, record: r } = unavailable[0] as { trace: Parameters<typeof agreesWithRecord>[0]; record: BattleRecord };
    expect(() => agreesWithRecord(trace, r)).not.toThrow();
    const fewer = { ...r, requests: { ...r.requests, p1: r.requests.p1.slice(1) } };
    expect(() => agreesWithRecord(trace, fewer)).toThrow("another number of requests");
    const rehashed = structuredClone(r);
    (rehashed.requests.p2[0] as { request_hash: string }).request_hash = "0".repeat(64);
    expect(() => agreesWithRecord(trace, rehashed)).toThrow("does not match the line");
    const judged = r.hidden_verdicts[0] as BattleRecord["hidden_verdicts"][number];
    const flipped = structuredClone(r);
    const verdict = flipped.hidden_verdicts[0]?.verdicts[0] as { legal: boolean };
    verdict.legal = !verdict.legal;
    expect(() => agreesWithRecord(trace, flipped)).toThrow("legal set or hidden verdicts differ");
    const unjudged = structuredClone(r);
    delete (unjudged.requests[judged.player][judged.request_seq - 1] as { legal?: unknown }).legal;
    expect(() => agreesWithRecord(trace, unjudged)).not.toThrow();
  });

  it("a pending candidate with no hidden verdict cannot be masked", () => {
    const body = random[0]?.record.channels.p1.find((l) => l.startsWith("|request|") && l.includes('"active"'))?.slice("|request|".length) as string;
    expect(maskOf(decodeRequest(body), []).every((m) => m.verdict === "legal")).toBe(true);
    const trapped = unavailable.flatMap(({ record: r }) => r.hidden_verdicts.map((h) => r.channels[h.player].filter((l) => l.startsWith("|request|"))[h.request_seq - 1] as string))[0] as string;
    expect(() => maskOf(decodeRequest(trapped.slice("|request|".length)), [])).toThrow("no hidden verdict");
  });
});

describe("the committed battles", () => {
  it("lists every corpus once, each battle with its own name, and refuses an unknown corpus", () => {
    const all = committedBattles();
    const count = (c: string): number => all.filter((b) => b.corpus === c).length;
    expect([count("golden"), count("random"), count("unavailable"), count("search-fidelity")]).toEqual([1, 40, 2, 200]);
    expect(all.filter((b) => !["golden", "random", "unavailable", "search-fidelity"].includes(b.corpus)).map((b) => b.corpus)).toEqual(["baton-pass", "baton-pass-volatiles", "fling", "illusion", "revival", "side-guard"]);
    expect(new Set(all.map((b) => b.spec.battle_id)).size).toBe(all.length);
    expect(all.every((b) => b.spec.max_turns === (b.corpus === "golden" ? 300 : 30))).toBe(true);
    expect(() => committedBattles(["transform"])).toThrow("no committed corpus named transform");
  });
});

describe("the Showdown battle backend", () => {
  it("replays a trace with its draws forced and gives every step's output back, state included", () => {
    const backend = showdownBattleBackend(loadEngine(process.env.SHOWDOWN_DIR || undefined));
    for (const { trace } of [...golden, ...random, ...unavailable]) {
      const start = backend.start([{ battle_id: trace.battle_id, format_id: trace.format_id, seed: trace.seed, teams: trace.teams as never, draws: trace.steps[0]?.draws ?? [] }])[0];
      if (start?.ok !== true) throw new Error("the start was refused");
      const outputs = [start.result.output];
      for (const step of trace.steps.slice(1)) {
        const out = backend.step([{ handle: start.result.handle, actions: step.actions, draws: step.draws }])[0];
        if (out?.ok !== true) throw new Error("a step was refused");
        outputs.push(out.result);
      }
      outputs.forEach((o, k) => {
        const s = trace.steps[k] as (typeof trace.steps)[number];
        expect({ channels: o.channels, requests: o.requests, draws: o.draws, state: o.state_sha256 }).toEqual({ channels: s.channels, requests: s.requests, draws: s.draws, state: s.state_sha256 });
      });
      expect([outputs.at(-1)?.ended, outputs.at(-1)?.winner, outputs.at(-1)?.turns]).toEqual([true, trace.outcome.winner, trace.outcome.turns]);
      backend.release([start.result.handle]);
      expect(backend.step([{ handle: start.result.handle, actions: [], draws: [] }])[0]).toMatchObject({ ok: false, error: expect.any(ReplayError) });
    }
    expect(backend.id).toBe("showdown-battle");
    expect(backend.backend_commit).toBe(engine.commit);
  });

  it("refuses a team the validator refuses, and a write the engine fails on", () => {
    const { trace } = golden[0] as (typeof golden)[number];
    const backend = showdownBattleBackend(engine, { id: "other", snapshot: loadRulesSnapshot(engine.commit) });
    const bad = structuredClone(trace.teams) as never as { p1: { moves: string[] }[] };
    (bad.p1[0] as { moves: string[] }).moves = ["Not A Move"];
    expect(backend.start([{ battle_id: "x", format_id: trace.format_id, seed: trace.seed, teams: bad as never, draws: [] }])[0]).toMatchObject({ ok: false });
    const Base = engine.sim.BattleStream as unknown as new () => { _writeLine(t: string, m: string): void };
    class Throwing extends Base {
      override _writeLine(type: string, message: string): void {
        if (type === "p1" && message === "boom") throw new Error("boom");
        super._writeLine(type, message);
      }
    }
    const failing = showdownBattleBackend({ ...engine, sim: { ...engine.sim, BattleStream: Throwing as never } });
    const started = failing.start([{ battle_id: "y", format_id: trace.format_id, seed: trace.seed, teams: trace.teams as never, draws: [] }])[0];
    if (started?.ok !== true) throw new Error("the start was refused");
    expect(failing.step([{ handle: started.result.handle, actions: [{ kind: "choice", seat: "p1", choice: "boom" }], draws: [] }])[0]).toMatchObject({ ok: false, error: expect.any(ReplayError) });
    expect(backend.id).toBe("other");
  });

  it("hashes the engine state without its wall-clock lines", () => {
    const fake = (log: string[]) => ({ toJSON: () => ({ a: 1, log }) });
    expect(stateHash(fake(["|t:|1", "|x"]))).toBe(stateHash(fake(["|t:|2", "|x"])));
    expect(stateHash(fake(["|x"]))).not.toBe(stateHash(fake(["|y"])));
    expect(interceptedSource({ v: 1 }, () => []).v).toBe(1);
  });
});
