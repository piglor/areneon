// The hypothetical Team Preview battle (ADR-029 sections 3 and 7, findings N5 and N6): what the format shows the hypothetical
// opponent equals what a real battle of the same teams shows that seat; the brands keep every hypothetical channel and
// observation out of the real paths; a marker in a world's sets reaches the hypothetical channels and no real artifact;
// opponents with the same public Team Preview give our side the same hypothetical inputs.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, expectTypeOf, it } from "vitest";
import { jcs } from "../src/canonical-json.ts";
import { firstLegal, GOLDEN_FORMAT, syntheticFixtureHashes, syntheticTeam, TEAMS_DIR } from "../src/golden.ts";
import {
  battleClass,
  type BattleRecord,
  type Engine,
  type EvaluationClient,
  enumerateCandidates,
  episodeFromRecord,
  episodeHash,
  loadTrainingSet,
  teamHash,
  evaluateHypothetical,
  HYPOTHETICAL_BATTLE_FORMAT,
  HYPOTHETICAL_ENGINE_PURPOSE,
  HypotheticalError,
  type HypotheticalChannel,
  type HypotheticalObservation,
  type HypotheticalPreview,
  hypotheticalCandidates,
  type HypotheticalCandidate,
  hypotheticalEngineSeed,
  hypotheticalRequest,
  importPaste,
  InvalidTeamError,
  loadEngine,
  nodeSha256,
  type Observation,
  type PlayerChannel,
  playerChannel,
  type PokemonSet,
  type PolicyClient,
  type PreviewSolveInput,
  recordedChannel,
  runBattle,
  startHypothetical,
  type StampedTeam,
  stampTeam,
} from "../src/index.ts";
import { buildObservation, observe } from "../src/index.ts";

const STREAM = new Uint8Array(32).fill(7);
let engine: Engine;
let own: PokemonSet[];
let world: PokemonSet[];
const paste = (file: string): PokemonSet[] => importPaste(engine, readFileSync(join(TEAMS_DIR, file), "utf8"));
beforeAll(() => {
  engine = loadEngine();
  own = syntheticTeam();
  world = paste("legal-opponent.txt");
});

const start = (overrides: Partial<Parameters<typeof startHypothetical>[0]> = {}): HypotheticalPreview =>
  startHypothetical({ engine, format_id: GOLDEN_FORMAT, seat: "p1", own_team: own, world_team: world, stream: STREAM, world: 0, data_class: "synthetic", ...overrides });

/** A stand-in worker that records each evaluate query and answers with a flat prior. */
function recorder(): EvaluationClient & { queries: Parameters<EvaluationClient["evaluate"]>[1][] } {
  const queries: Parameters<EvaluationClient["evaluate"]>[1][] = [];
  return {
    queries,
    async evaluate(_battle, query) {
      queries.push(query);
      return { identity: query.identity, prior_ppm: query.candidates.map(() => 1) as [number, ...number[]], value_ppm: 500_000 };
    },
  };
}

const seen = async (preview: HypotheticalPreview, channel: HypotheticalChannel): Promise<Observation> => {
  const client = recorder();
  await evaluateHypothetical(client, "b", await preview.observation(channel), "test@policy");
  return client.queries[0]?.observation as Observation;
};

describe("the engine seed", () => {
  it("is SHA-256 over the JCS of {purpose, stream, world}: vectors computed with Python hashlib", () => {
    // Python: hashlib.sha256(json.dumps({"purpose": "preview-solve-engine/1", "stream": "07" * 32, "world": W},
    //   sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:32]
    expect(HYPOTHETICAL_ENGINE_PURPOSE).toBe("preview-solve-engine/1");
    expect(hypotheticalEngineSeed(STREAM, 3)).toBe("sodium,20563938ad631429936d6059ba9ee84a");
    expect(hypotheticalEngineSeed(STREAM, 4)).toBe("sodium,522847c080497e16ecad2be7dbbee25b");
    expect(HYPOTHETICAL_BATTLE_FORMAT).toBe(1);
  });
});

describe("what a real battle of the same teams gives", () => {
  let record: BattleRecord;
  beforeAll(async () => {
    record = await runBattle({ engine, formatId: GOLDEN_FORMAT, seed: hypotheticalEngineSeed(STREAM, 0), teams: { p1: own, p2: world }, actors: { p1: firstLegal, p2: firstLegal } });
  });
  const cls = () => battleClass(record.team_hashes, syntheticFixtureHashes(engine));

  it("the hypothetical opponent's Team Preview request and channel equal the real seat's (closed sheets)", async () => {
    const preview = start();
    const real = await observe({ channel: recordedChannel(record, "p2", 1, cls()), team: { schema_version: 1, data_class: "synthetic", team: world as never }, manifest: record.manifest });
    expect(await seen(preview, preview.foe_channel)).toEqual(real);
    expect(preview.foe_request.request_hash).toBe(record.requests.p2[0]?.request_hash);
    expect(preview.foe_request.phase).toBe("preview");
    const ownReal = await observe({ channel: recordedChannel(record, "p1", 1, cls()), team: { schema_version: 1, data_class: "synthetic", team: own as never }, manifest: record.manifest });
    expect(await seen(preview, preview.own_channel)).toEqual(ownReal);
    // The pinned rules snapshot has no open-sheet format, and an in-process battle never issues the server's `>show-openteamsheets`
    // (cts.test.ts), so both this battle and the real A1 battle compared above show closed sheets. A future open-sheet format would
    // need ADR-027's sheet reveal added to this module's start (the sheets shown to both seats before the request); until then
    // open-sheet equality is neither supported nor tested.
  });

  it("our hypothetical channel at the first move request equals the real one for the same orders and seed", async () => {
    const preview = start();
    const choice = (player: "p1" | "p2") => (record.choices.find((c) => c.player === player && c.request_seq === 1) as { choice: string }).choice;
    const turn = preview.play(choice("p1"), choice("p2"));
    expect(turn.kind).toBe("request");
    if (turn.kind !== "request") return;
    expect(turn.request.phase).toBe("move");
    const real = await observe({ channel: recordedChannel(record, "p1", 2, cls()), team: { schema_version: 1, data_class: "synthetic", team: own as never }, manifest: record.manifest });
    expect(await seen(preview, turn.channel)).toEqual(real);
    expect(hypotheticalRequest(turn.channel).request_hash).toBe(record.requests.p1[1]?.request_hash);
  });

  it("starts from either seat", async () => {
    const preview = start({ seat: "p2" });
    expect(preview.seat).toBe("p2");
    expect(preview.foe).toBe("p1");
    expect(preview.foe_request.player).toBe("p1");
  });
});

describe("plays and refusals", () => {
  it("plays every pair of orders on its own clone of the one started battle", () => {
    const preview = start();
    const own_ = hypotheticalCandidates(preview.own_channel).filter((c) => !c.pending);
    const foe_ = hypotheticalCandidates(preview.foe_channel).filter((c) => !c.pending);
    const a = preview.play((own_[0] as HypotheticalCandidate).order, (foe_[0] as HypotheticalCandidate).order);
    const b = preview.play((own_[1] as HypotheticalCandidate).order, (foe_[0] as HypotheticalCandidate).order);
    const again = preview.play((own_[0] as HypotheticalCandidate).order, (foe_[0] as HypotheticalCandidate).order);
    expect([a.kind, b.kind]).toEqual(["request", "request"]);
    expect(jcs(hypotheticalRequest((again as { channel: HypotheticalChannel }).channel) as never)).toBe(jcs(hypotheticalRequest((a as { channel: HypotheticalChannel }).channel) as never));
    expect(hypotheticalRequest((a as { channel: HypotheticalChannel }).channel).request_hash).not.toBe(hypotheticalRequest((b as { channel: HypotheticalChannel }).channel).request_hash);
  });

  it("reports a refused order, and refuses a team the validator refuses, a bad seat and a foreign object", async () => {
    const preview = start();
    expect(preview.play("team 9999", "default").kind).toBe("refused");
    expect(() => start({ world_team: paste("illegal-five-pokemon.txt") })).toThrow(InvalidTeamError);
    expect(() => start({ seat: "p3" as never })).toThrow(HypotheticalError);
    await expect(preview.observation({ player: "p1" } as unknown as HypotheticalChannel)).rejects.toThrow(HypotheticalError);
    await expect(evaluateHypothetical(recorder(), "b", { player: "p1", kind: "preview" } as unknown as HypotheticalObservation, "x")).rejects.toThrow(HypotheticalError);
    expect(() => hypotheticalRequest({ player: "p1" } as unknown as HypotheticalChannel)).toThrow(HypotheticalError);
  });
});

describe("the brands", () => {
  it("keep a hypothetical channel and observation out of every real path (type tests)", async () => {
    const preview = start();
    const channel = preview.foe_channel;
    const observation = await preview.observation(channel);
    expectTypeOf(channel).not.toMatchTypeOf<PlayerChannel>();
    expectTypeOf(observation).not.toMatchTypeOf<Observation>();
    // Compile-time only: the closure is never called.
    const never = (): void => {
      // @ts-expect-error a hypothetical channel is not a PlayerChannel: buildObservation refuses it
      void buildObservation({ channel, team: { schema_version: 1, data_class: "synthetic", team: world as never }, manifest: preview.manifest, sha256: nodeSha256 });
      // @ts-expect-error recordedChannel takes a BattleRecord, never a hypothetical channel
      void recordedChannel(channel, "p1", 1, "synthetic" as never);
      // @ts-expect-error the real policy path takes an Observation, never a hypothetical one
      const real: Parameters<PolicyClient["ask"]>[1]["observation"] = observation;
      void real;
      // @ts-expect-error a hypothetical request is not a DecodedRequest: the candidate enumerator refuses it
    void enumerateCandidates(preview.foe_request);
    // @ts-expect-error a hypothetical candidate is not a Candidate (no `choice`)
    const plain: import("../src/index.ts").Candidate = hypotheticalCandidates(channel)[0] as HypotheticalCandidate;
    void plain;
    // @ts-expect-error neither does an Episode record
      const inEpisode: Parameters<typeof episodeFromRecord>[0] = channel;
      void inEpisode;
    };
    void never;
    // The run-time side of the same wall: the real adapters see no lines in a brand.
    expect(() => playerChannel(channel)).toThrow();
  });

  it("PreviewSolveInput has no field for the running battle, a record, the omniscient log, hidden verdicts, the seed or the opponent's team", () => {
    expectTypeOf<keyof PreviewSolveInput>().toEqualTypeOf<"format_id" | "seat" | "team" | "channel" | "stream">();
    const base: PreviewSolveInput = { format_id: GOLDEN_FORMAT, seat: "p1", team: own, channel: [], stream: STREAM };
    void base;
    // @ts-expect-error no live battle
    const a: PreviewSolveInput = { ...base, battle: {} };
    // @ts-expect-error no BattleRecord
    const b: PreviewSolveInput = { ...base, record: {} as BattleRecord };
    // @ts-expect-error no omniscient log
    const c: PreviewSolveInput = { ...base, omniscient_log: [] };
    // @ts-expect-error no hidden verdicts
    const d: PreviewSolveInput = { ...base, hidden_verdicts: [] };
    // @ts-expect-error no battle seed
    const e: PreviewSolveInput = { ...base, seed: "sodium,00" };
    // @ts-expect-error no opponent team
    const f: PreviewSolveInput = { ...base, opponent_team: world };
    void [a, b, c, d, e, f];
  });
});

describe("canary: marker values of a world's sets (N6)", () => {
  const MARKERS = [0, 1, 2, 3, 4, 5].map((i) => `Zq9mark${i}`);
  const marked = (): PokemonSet[] => paste("legal-opponent.txt").map((set, i) => ({ ...set, name: MARKERS[i] as string }));
  const leaks = (value: unknown): string[] => {
    const text = JSON.stringify(value);
    return MARKERS.filter((marker) => text.includes(marker));
  };
  const league = (r: BattleRecord) => ({ kind: "league", run_id: "league-1", seed: r.seed, opponent_policy_id: "first-legal" }) as const;

  it("appear in the hypothetical channels, and in no Episode, training example, real observation or stored record of a real game made AFTER the marked solve on the same engine", async () => {
    // The marked solve first, in full (start, both seats' observations, a worker query, a played pair of orders), so any state it
    // left in the engine instance or the module would be there when the real game runs.
    const preview = start({ world_team: marked() });
    const client = recorder();
    expect(leaks(await seen(preview, preview.foe_channel)).length).toBeGreaterThan(0);
    const turn = preview.play(hypotheticalCandidates(preview.own_channel)[0]?.order as string, hypotheticalCandidates(preview.foe_channel)[0]?.order as string);
    expect(turn.kind).toBe("request");
    if (turn.kind === "request") await evaluateHypothetical(client, "b", await preview.observation(turn.channel), "x");

    // Now the real game, on the same engine instance.
    const record = await runBattle({ engine, formatId: GOLDEN_FORMAT, seed: hypotheticalEngineSeed(STREAM, 0), teams: { p1: own, p2: world }, actors: { p1: firstLegal, p2: firstLegal } });
    const known = syntheticFixtureHashes(engine);
    const cls = battleClass(record.team_hashes, known);
    const stamp = (team: PokemonSet[]): StampedTeam => stampTeam(team as never, known);
    const write = (player: "p1" | "p2") =>
      episodeFromRecord(record, player, { team: stamp(player === "p1" ? own : world), opponent_team: stamp(player === "p1" ? world : own), known_synthetic: known, behaviour_policy_id: "first-legal", collection: league(record) });
    const episodes = [await write("p1"), await write("p2")];
    const stream = await loadTrainingSet({ episodes, evaluation_manifests: [], known_synthetic: known, behaviour_policy_id: "first-legal" });
    const observation = await observe({ channel: recordedChannel(record, "p1", 2, cls), team: { schema_version: 1, data_class: "synthetic", team: own as never }, manifest: record.manifest });
    expect(leaks(record)).toEqual([]);
    expect(leaks(episodes)).toEqual([]);
    expect(leaks(stream)).toEqual([]);
    expect(stream.examples.length).toBeGreaterThan(0);
    expect(leaks(observation)).toEqual([]);
    // The stored record's hashes bind the real opponent, not the marked world.
    expect(record.team_hashes.p2).toBe(teamHash(world));
    expect(record.team_hashes.p2).not.toBe(teamHash(marked()));
    expect(await Promise.all(episodes.map((e) => episodeHash(e, nodeSha256)))).toHaveLength(2);
  });

  it("negative control: an unbranded route that writes hypothetical lines into real artefacts leaks the markers, and the scan catches it", async () => {
    const preview = start({ world_team: marked() });
    const client = recorder();
    await evaluateHypothetical(client, "b", await preview.observation(preview.foe_channel), "x");
    const lines = (client.queries[0]?.observation as { log: string[] }).log;
    // Route 1: the unchecked `playerChannel(unknown)` adapter, then a real observation.
    const viaAdapter = await observe({
      channel: playerChannel({ schema_version: 1, data_class: "synthetic", player: "p2", lines }),
      team: { schema_version: 1, data_class: "synthetic", team: marked() as never },
      manifest: preview.manifest,
    });
    expect(leaks(viaAdapter).length).toBeGreaterThan(0);
    // Route 2: the hypothetical lines pushed into a real record's channel, then the loader's stream.
    const record = await runBattle({ engine, formatId: GOLDEN_FORMAT, seed: hypotheticalEngineSeed(STREAM, 0), teams: { p1: own, p2: world }, actors: { p1: firstLegal, p2: firstLegal } });
    expect(leaks(record)).toEqual([]);
    const tainted = structuredClone(record);
    tainted.channels.p1.push(...lines);
    expect(leaks(tainted).length).toBeGreaterThan(0);
  });
});

describe("invariance: the same public Team Preview, other hidden sets", () => {
  const hidden = (): PokemonSet[] => {
    const sets = paste("legal-opponent.txt");
    return sets.map((set, i) => ({
      ...set,
      nature: (sets[(i + 1) % sets.length] as PokemonSet).nature,
      evs: { ...(sets[(i + 1) % sets.length] as PokemonSet).evs },
    }));
  };
  const orders = (p: HypotheticalPreview): [string, string] => [
    (hypotheticalCandidates(p.own_channel)[0] as HypotheticalCandidate).order,
    (hypotheticalCandidates(p.foe_channel)[0] as HypotheticalCandidate).order,
  ];

  for (const seat of ["p1", "p2"] as const) {
    it(`gives seat ${seat} an identical hypothetical Team Preview observation and, after play(), an identical first-move-request observation; the opponent's own view differs`, async () => {
      const a = start({ seat, world_team: world });
      const b = start({ seat, world_team: hidden() });
      expect(jcs((await seen(a, a.own_channel)) as never)).toBe(jcs((await seen(b, b.own_channel)) as never));
      expect(hypotheticalCandidates(a.own_channel)).toEqual(hypotheticalCandidates(b.own_channel));
      expect(jcs((await seen(a, a.foe_channel)) as never)).not.toBe(jcs((await seen(b, b.foe_channel)) as never));
      // The same orders on both: what our side sees at its first move request is the same too (the leads show the same public facts).
      const [own1, foe1] = orders(a);
      const ta = a.play(own1, foe1);
      const tb = b.play(own1, foe1);
      expect([ta.kind, tb.kind]).toEqual(["request", "request"]);
      if (ta.kind === "request" && tb.kind === "request") {
        expect(jcs((await seen(a, ta.channel)) as never)).toBe(jcs((await seen(b, tb.channel)) as never));
        expect(ta.request).toEqual(tb.request);
      }
    });
  }

  it("negative control: another public Team Preview (other species) changes it", async () => {
    const a = start({ world_team: world });
    const c = start({ world_team: own });
    expect(jcs((await seen(a, a.own_channel)) as never)).not.toBe(jcs((await seen(c, c.own_channel)) as never));
  });
});

describe("the worker entry", () => {
  it("refuses a pair that reads candidate features, and sends the module's own frozen copy of the observation", async () => {
    const preview = start();
    const observation = await preview.observation(preview.foe_channel);
    const featured = Object.assign(recorder(), { featureSet: { feature_set: "probe-damage/1", feature_set_hash: "a".repeat(64) } });
    await expect(evaluateHypothetical(featured, "b", observation, "x")).rejects.toThrow(/features_pair/);
    expect(featured.queries).toEqual([]);
    const plain = Object.assign(recorder(), { featureSet: null });
    const { candidates } = await evaluateHypothetical(plain, "b", observation, "x");
    expect(candidates.length).toBeGreaterThan(0);
    expect(Object.isFrozen(plain.queries[0]?.observation)).toBe(true);
    expect(Object.isFrozen((plain.queries[0]?.observation as { identity: object }).identity)).toBe(true);
  });
});

describe("an engine that gives no Team Preview, ends at once or sends no request (stand-in battle)", () => {
  type Sides = Record<"p1" | "p2", string>;
  /** The real battle's own request lines: the first (Team Preview) and the second (a move request) of each seat. */
  async function recorded(): Promise<{ preview: Sides; move: Sides }> {
    const record = await runBattle({ engine, formatId: GOLDEN_FORMAT, seed: hypotheticalEngineSeed(STREAM, 0), teams: { p1: own, p2: world }, actors: { p1: firstLegal, p2: firstLegal } });
    const nth = (p: "p1" | "p2", n: number): string => record.channels[p].filter((l) => l.startsWith("|request|"))[n] as string;
    return { preview: { p1: nth("p1", 0), p2: nth("p2", 0) }, move: { p1: nth("p1", 1), p2: nth("p2", 1) } };
  }

  /** A stand-in `Battle` that sends `requests` when started and plays a clone as `mode` says; everything else is the real engine's. */
  function fake(mode: "ended" | "silent", requests: Sides): Engine {
    class Fake {
      send: (type: string, data: string | string[]) => void;
      ended = mode === "ended";
      winner = "";
      log: string[] = [];
      sentLogPos = 0;
      prng: unknown;
      clone = false;
      constructor(options: { send?: (type: string, data: string | string[]) => void }) {
        this.send = options.send ?? (() => {});
      }
      setPlayer(): void {}
      sendUpdates(): void {
        if (this.clone) return;
        this.send("update", ["|gen|9"]);
        for (const side of ["p1", "p2"] as const) this.send("sideupdate", `${side}\n${requests[side]}`);
      }
      toJSON(): object {
        return {};
      }
      static fromJSON(): Fake {
        const clone = new Fake({});
        clone.clone = true;
        return clone;
      }
      restart(send: Fake["send"]): void {
        this.send = send;
      }
      choose(): boolean {
        return true;
      }
    }
    return { ...engine, sim: { ...engine.sim, Battle: Fake as never } } as Engine;
  }
  const startWith = (stand: Engine): HypotheticalPreview => startHypothetical({ engine: stand, format_id: GOLDEN_FORMAT, seat: "p1", own_team: own, world_team: world, stream: STREAM, world: 0, data_class: "synthetic" });

  it("refuses a format that asks for no Team Preview order", async () => {
    const { move } = await recorded();
    expect(() => startWith(fake("silent", move))).toThrow(HypotheticalError);
  });

  it("reports a game that ends at once, and an engine that sends the seat no request", async () => {
    const { preview } = await recorded();
    expect(startWith(fake("ended", preview)).play("team 1234", "team 1234")).toEqual({ kind: "ended", outcome: "tie" });
    expect(startWith(fake("silent", preview)).play("team 1234", "team 1234")).toEqual({ kind: "refused", reason: "the engine sent the seat no request" });
  });
});
