import { readFileSync } from "node:fs";
import { join } from "node:path";
import { ajvFor } from "@areneon/schema/ajv";
import battleSchema from "@areneon/schema/contracts/v1/battle-observation.schema.json" with { type: "json" };
import previewSchema from "@areneon/schema/contracts/v1/preview-observation.schema.json" with { type: "json" };
import type { PlayerChannelLog } from "@areneon/schema/types/v1/player-channel-log.ts";
import type { PrivateTeamSpec } from "@areneon/schema/types/v1/private-team-spec.ts";
import { beforeAll, describe, expect, it } from "vitest";
import { jcs } from "../src/canonical-json.ts";
import {
  type ChannelFixture,
  firstLegal,
  GOLDEN_CHANNELS,
  GOLDEN_DIR,
  GOLDEN_FORMAT,
  goldenBattle,
  syntheticTeam,
} from "../src/golden.ts";
import {
  type BattleRecord,
  buildObservation,
  decodeRequest,
  type Engine,
  importPaste,
  loadEngine,
  type Observation,
  ObservationError,
  nodeSha256,
  playerChannel,
  type Player,
  recordedChannel,
  runBattle,
  type SimBattle,
  webSha256,
} from "../src/index.ts";
import type { Json } from "../src/canonical-json.ts";

const contract = ajvFor("contract");
const validPreview = contract.compile(previewSchema);
const validBattle = contract.compile(battleSchema);
function expectContract(observation: Observation): void {
  const valid = observation.kind === "preview" ? validPreview : validBattle;
  expect(valid(observation), JSON.stringify(valid.errors)).toBe(true);
}

const synthetic = (team = syntheticTeam()): PrivateTeamSpec => ({
  schema_version: 1,
  data_class: "synthetic",
  team: team as PrivateTeamSpec["team"],
});
const fixture = (file: string): ChannelFixture =>
  JSON.parse(readFileSync(join(GOLDEN_DIR, file), "utf8")) as ChannelFixture;

/** Every non-wait request position of a channel: the decisions a player made. */
function decisionPrefixes(log: PlayerChannelLog): PlayerChannelLog[] {
  const out: PlayerChannelLog[] = [];
  log.lines.forEach((line, i) => {
    if (line.startsWith("|request|") && !(JSON.parse(line.slice(9)) as { wait?: boolean }).wait) {
      out.push({ ...log, lines: log.lines.slice(0, i + 1) });
    }
  });
  return out;
}

let engine: Engine;
let golden: BattleRecord;
beforeAll(async () => {
  engine = loadEngine();
  golden = await goldenBattle(engine);
});

describe("the ObservationBuilder module", () => {
  it.each(["observation.ts", "web-sha256.ts"])("%s imports no engine, runner or Node built-in", (file) => {
    const source = readFileSync(new URL(`../src/${file}`, import.meta.url), "utf8");
    const imports = source.split("\n").filter((line) => /^\s*(import|export)\b.*\bfrom\s/.test(line));
    for (const line of imports) {
      // Contracts (types only) and the import-free canonical JSON module.
      expect(line, line).toMatch(/^import type .* from "(@areneon\/schema\/types\/|\.\/observation\.ts)|from "\.\/canonical-json\.ts";$/);
    }
    expect(source).not.toMatch(/node:|engine\.ts|runner\.ts|\brequire\(/);
  });
});

describe("golden parsing (ADR-008 §5 adversarial 4)", () => {
  it("builds a contract-valid observation for every decision in the golden channels", async () => {
    let built = 0;
    for (const file of Object.keys(GOLDEN_CHANNELS)) {
      const { channel, manifest } = fixture(file);
      for (const prefix of decisionPrefixes(channel)) {
        const observation = await buildObservation({
          channel: playerChannel(prefix),
          team: synthetic(),
          manifest,
          sha256: nodeSha256,
        });
        expectContract(observation);
        const last = prefix.lines.at(-1) as string;
        const decoded = decodeRequest(last.slice("|request|".length));
        expect(observation.identity.request_hash).toBe(decoded.request_hash);
        expect(observation.kind).toBe(decoded.phase === "preview" ? "preview" : "battle");
        expect(observation.log).toEqual(prefix.lines);
        built++;
      }
    }
    expect(built).toBeGreaterThan(10);
  });

  it("gives the identity the runner gave each request (A1)", async () => {
    for (const player of ["p1", "p2"] as const) {
      for (const request of golden.requests[player].filter((r) => r.phase !== "wait")) {
        const observation = await buildObservation({
          channel: recordedChannel(golden, player, request.request_seq, "synthetic"),
          team: synthetic(),
          manifest: golden.manifest,
          sha256: nodeSha256,
        });
        expect(observation.identity).toEqual({ request_seq: request.request_seq, request_hash: request.request_hash });
        expect(observation.player).toBe(player);
      }
    }
  });
});

describe("the Sha256 ports", () => {
  it("agree between Node and WebCrypto on every golden request", async () => {
    expect(await webSha256(new Uint8Array())).toBe(
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    );
    for (const file of Object.keys(GOLDEN_CHANNELS)) {
      for (const line of fixture(file).channel.lines.filter((l) => l.startsWith("|request|"))) {
        const bytes = new TextEncoder().encode(line);
        expect(await webSha256(bytes)).toBe(await nodeSha256(bytes));
      }
    }
  });

  it("let the builder hash with either port to the same identity", async () => {
    const [prefix] = decisionPrefixes(fixture("channels/p1.json").channel);
    const input = { channel: playerChannel(prefix), team: synthetic(), manifest: golden.manifest };
    const viaNode = await buildObservation({ ...input, sha256: nodeSha256 });
    const viaWeb = await buildObservation({ ...input, sha256: webSha256 });
    expect(viaWeb).toEqual(viaNode);
  });
});

describe("the canary test (ADR-008 §5 adversarial 2)", () => {
  /** p1 brings the synthetic team; p2 a team sharing no species, item or move with it. */
  async function battle(): Promise<BattleRecord> {
    const opponent = importPaste(
      engine,
      readFileSync(join(GOLDEN_DIR, "../../teams", GOLDEN_FORMAT, "legal-opponent.txt"), "utf8"),
    );
    return runBattle({
      engine,
      formatId: GOLDEN_FORMAT,
      seed: "sodium,0f0e0d0c0b0a09080706050403020100",
      teams: { p1: syntheticTeam(), p2: opponent },
      actors: { p1: firstLegal, p2: firstLegal },
    });
  }

  /**
   * What `of` must never leak to the other player, as whole lines and JSON
   * objects (never bare numbers):
   * - its SECRET lines: omniscient lines `of`'s own channel carries but the
   *   other player's does not (the engine resolved the `|split|` there);
   * - the stats of every Pokémon in `of`'s own requests, which no public
   *   line ever reveals.
   */
  function canaries(record: BattleRecord, of: Player): { lines: string[]; stats: string[] } {
    const other: Player = of === "p1" ? "p2" : "p1";
    const seenByOther = new Set(record.channels[other]);
    const own = new Set(record.channels[of]);
    const lines = new Set(record.omniscient_log.filter((l) => own.has(l) && !seenByOther.has(l)));
    const stats = new Set<string>();
    for (const request of record.requests[of]) {
      const side = (request.request as { side: { pokemon: { stats: { [k: string]: Json } }[] } }).side;
      for (const mon of side.pokemon) stats.add(jcs(mon.stats));
    }
    return { lines: [...lines], stats: [...stats] };
  }

  /** A canary in an observation's JCS bytes, as itself or escaped inside a JSON string. */
  const leaks = (bytes: string, secret: string): boolean =>
    bytes.includes(secret) || bytes.includes(JSON.stringify(secret).slice(1, -1));

  async function observationsOf(record: BattleRecord, player: Player): Promise<Observation[]> {
    return Promise.all(
      record.requests[player]
        .filter((r) => r.phase !== "wait")
        .map((r) =>
          buildObservation({
            channel: recordedChannel(record, player, r.request_seq, "synthetic"),
            team: synthetic(),
            manifest: record.manifest,
            sha256: nodeSha256,
          }),
        ),
    );
  }

  it("finds no opponent secret in any observation of either player", async () => {
    const record = await battle();
    for (const [player, opponent] of [["p1", "p2"], ["p2", "p1"]] as const) {
      const { lines, stats } = canaries(record, opponent);
      expect(lines.length, `${opponent} secret lines`).toBeGreaterThan(0);
      expect(stats, `${opponent} stats`).toHaveLength(6);
      const secrets = [...lines, ...stats];
      const observations = await observationsOf(record, player);
      expect(observations.length).toBeGreaterThan(3);
      for (const observation of observations) {
        expectContract(observation);
        const bytes = jcs(observation as unknown as Json);
        for (const secret of secrets) expect(leaks(bytes, secret), secret).toBe(false);
      }
    }
  });

  it.each(["lines", "stats"] as const)("would catch a leak: a smuggled opponent canary (%s) is found", async (kind) => {
    const record = await battle();
    const secret = canaries(record, "p2")[kind][0] as string;
    const honest = recordedChannel(record, "p1", 1, "synthetic");
    const leaky = playerChannel({ ...honest, lines: [secret, ...honest.lines] });
    const observation = await buildObservation({
      channel: leaky,
      team: synthetic(),
      manifest: record.manifest,
      sha256: nodeSha256,
    });
    expect(leaks(jcs(observation as unknown as Json), secret)).toBe(true);
  });
});

describe("schema probes (ADR-008 §5 adversarial 1)", () => {
  let observation: Observation;
  beforeAll(async () => {
    const [prefix] = decisionPrefixes(fixture("channels/p1.json").channel);
    observation = await buildObservation({
      channel: playerChannel(prefix),
      team: synthetic(),
      manifest: golden.manifest,
      sha256: nodeSha256,
    });
  });

  it.each<[string, (o: Record<string, unknown>) => void]>([
    ["a smuggled omniscient log", (o) => (o.omniscient_log = ["|split|p2"])],
    ["an opponent team", (o) => (o.opponent_team = synthetic())],
    ["a hidden verdict", (o) => (o.hidden_verdicts = [])],
    ["an extra field on a set", (o) => (((o.team as { team: object[] }).team[0] as { secret?: string }).secret = "x")],
    ["a non-string log line", (o) => ((o.log as unknown[])[0] = { line: 1 })],
  ])("refuses %s", (_label, smuggle) => {
    const probe = structuredClone(observation) as unknown as Record<string, unknown>;
    smuggle(probe);
    expect(validPreview(probe)).toBe(false);
  });
});

describe("the type test (ADR-008 §5 adversarial 3)", () => {
  it("accepts only a branded PlayerChannel", () => {
    // Compile-time checks: `pnpm typecheck` fails if any of these compiles.
    const neverRun = (record: BattleRecord, battle: SimBattle, log: PlayerChannelLog): void => {
      const rest = { team: synthetic(), manifest: record.manifest, sha256: nodeSha256 };
      // @ts-expect-error: the omniscient log is not a PlayerChannel
      void buildObservation({ ...rest, channel: record.omniscient_log });
      // @ts-expect-error: nor is the engine's Battle
      void buildObservation({ ...rest, channel: battle });
      // @ts-expect-error: nor an unchecked log; only an adapter brands one
      void buildObservation({ ...rest, channel: log });
    };
    expect(neverRun).toBeTypeOf("function");
  });
});

describe("buildObservation and its adapter", () => {
  const p1 = (): PlayerChannelLog => fixture("channels/p1.json").channel;

  it("carries the server's rqid, the decision context, and a private class", async () => {
    const [prefix] = decisionPrefixes(p1()) as [PlayerChannelLog];
    const last = prefix.lines.at(-1) as string;
    const withRqid = { ...prefix, lines: [...prefix.lines.slice(0, -1), last.replace("{", '{"rqid":9,')] };
    const decision_context = {
      timer_source: "server" as const,
      timer_rules: { starting: 420, grace: 90, add_per_turn: 0, max_per_turn: 55, max_first_turn: 90, tick: 5 },
      turn_seconds_left: 90,
      bank_seconds_left: 420,
      grace_seconds_left: 90,
      safety_margin_ms: 3000,
      submit_within_ms: 60000,
      policy_budget_ms: 59000,
    };
    const observation = await buildObservation({
      channel: playerChannel(withRqid),
      team: { ...synthetic(), data_class: "private" },
      manifest: golden.manifest,
      sha256: nodeSha256,
      decision_context,
    });
    expect(observation.identity.rqid).toBe(9);
    expect(observation.decision_context).toEqual(decision_context);
    expect(observation.data_class).toBe("private");
    expectContract(observation);
    const plain = await buildObservation({
      channel: playerChannel({ ...prefix, data_class: "private" }),
      team: synthetic(),
      manifest: golden.manifest,
      sha256: nodeSha256,
    });
    expect(plain.data_class).toBe("private");
    expect(plain).not.toHaveProperty("decision_context");
    expect(plain.identity).not.toHaveProperty("rqid");
  });

  it("refuses a channel without a request, or one that ends waiting", async () => {
    const rest = { team: synthetic(), manifest: golden.manifest, sha256: nodeSha256 };
    const none = playerChannel({ ...p1(), lines: ["|gametype|doubles"] });
    await expect(buildObservation({ ...rest, channel: none })).rejects.toThrow(ObservationError);
    const side = { name: "p1", id: "p1", pokemon: [] };
    const waiting = playerChannel({ ...p1(), lines: [`|request|${JSON.stringify({ wait: true, side })}`] });
    await expect(buildObservation({ ...rest, channel: waiting })).rejects.toThrow(/wait request/);
  });

  it("brands only a valid channel of that player's own requests", () => {
    expect(() => playerChannel({ ...p1(), extra: 1 })).toThrow(/not a PlayerChannelLog/);
    expect(() => playerChannel({ ...p1(), player: "p2" })).toThrow(/p2's channel holds a request for p1/);
    expect(() => playerChannel({ ...p1(), lines: ['|request|{"side":{}}'] })).toThrow(/engine's request shape/);
    expect(() => recordedChannel(golden, "p1", 999, "synthetic")).toThrow(/no request 999/);
  });
});
