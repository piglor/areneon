import { readdirSync, readFileSync } from "node:fs";
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
  observe,
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
  // Every module specifier, however the import is written: static, multi-line,
  // side-effect, re-export, dynamic or require.
  const specifiers = (source: string): string[] =>
    [...source.matchAll(/(?:\bfrom|\bimport|\brequire)\s*\(?\s*["']([^"']+)["']/g)].map((m) => m[1] as string);

  it.each(["observation.ts", "web-sha256.ts"])("%s imports only contracts and canonical JSON", (file) => {
    const source = readFileSync(new URL(`../src/${file}`, import.meta.url), "utf8");
    const found = specifiers(source);
    expect(found.length).toBeGreaterThan(0);
    for (const specifier of found) {
      expect(specifier).toMatch(
        /^(@areneon\/schema\/types\/v1\/[a-z-]+\.ts|@areneon\/schema\/contracts\/v1\/[a-z-]+\.schema\.json|\.\/canonical-json\.ts|\.\/observation\.ts)$/,
      );
    }
  });

  it("is the only module, with its adapter, that brands a PlayerChannel", () => {
    for (const file of readdirSync(new URL("../src/", import.meta.url))) {
      if (file === "player-channel.ts") continue;
      const source = readFileSync(new URL(`../src/${file}`, import.meta.url), "utf8");
      expect(source, file).not.toMatch(/as PlayerChannel\b/);
    }
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
  /** The canary battle: p1 the synthetic team, p2 one sharing no species, item or move with it. */
  let record: BattleRecord;
  let teams: Record<Player, PrivateTeamSpec>;
  beforeAll(async () => {
    const opponent = importPaste(
      engine,
      readFileSync(join(GOLDEN_DIR, "../../teams", GOLDEN_FORMAT, "legal-opponent.txt"), "utf8"),
    );
    teams = { p1: synthetic(), p2: synthetic(opponent) };
    record = await runBattle({
      engine,
      formatId: GOLDEN_FORMAT,
      seed: "sodium,0f0e0d0c0b0a09080706050403020100",
      teams: { p1: syntheticTeam(), p2: opponent },
      actors: { p1: firstLegal, p2: firstLegal },
    });
  });
  const other = (player: Player): Player => (player === "p1" ? "p2" : "p1");

  /**
   * What `of` must never leak to the other player: anything the other
   * player's channel never shows. Whole lines and strings, never bare numbers.
   * - `lines`: `of`'s SECRET lines, omniscient lines its own channel carries
   *   and the other's does not (the engine resolved each `|split|` there).
   *   `log` is the observer's channel, so these guard regressions in the
   *   other fields rather than prove the split;
   * - `stats`: the stats of every Pokémon in `of`'s own requests;
   * - `sets`: each of `of`'s sets whole, and every item and move of them that
   *   the other player's channel never names (never revealed in battle).
   */
  function canaries(of: Player): { lines: string[]; stats: string[]; sets: string[] } {
    const seen = record.channels[other(of)];
    const seenBytes = seen.join("\n");
    const own = new Set(record.channels[of]);
    const lines = new Set(record.omniscient_log.filter((l) => own.has(l) && !seen.includes(l)));
    const stats = new Set<string>();
    for (const request of record.requests[of]) {
      const side = (request.request as { side: { pokemon: { stats: { [k: string]: Json } }[] } }).side;
      for (const mon of side.pokemon) stats.add(jcs(mon.stats));
    }
    const sets = new Set<string>();
    for (const set of teams[of].team) {
      sets.add(jcs(set as unknown as Json));
      for (const name of [set.item, ...set.moves]) if (name !== "" && !seenBytes.includes(name)) sets.add(name);
    }
    return { lines: [...lines], stats: [...stats], sets: [...sets] };
  }

  /** A canary in JCS bytes, as itself or escaped inside a JSON string. */
  const leaks = (bytes: string, secret: string): boolean =>
    bytes.includes(secret) || bytes.includes(JSON.stringify(secret).slice(1, -1));

  /** Each player's observations, each built with that player's own team. */
  const observationsOf = (player: Player): Promise<Observation[]> =>
    Promise.all(
      record.requests[player]
        .filter((r) => r.phase !== "wait")
        .map((r) =>
          observe({
            channel: recordedChannel(record, player, r.request_seq, "synthetic"),
            team: teams[player],
            manifest: record.manifest,
          }),
        ),
    );

  it("finds no opponent secret anywhere in any observation of either player", async () => {
    for (const player of ["p1", "p2"] as const) {
      const { lines, stats, sets } = canaries(other(player));
      expect(lines.length, "secret lines").toBeGreaterThan(0);
      expect(stats, "stats").toHaveLength(6);
      // Six whole sets, and items and moves never revealed in this battle.
      expect(sets.length, "sets").toBeGreaterThan(6 + 4);
      const observations = await observationsOf(player);
      expect(observations.length).toBeGreaterThan(3);
      for (const observation of observations) {
        const bytes = jcs(observation as unknown as Json);
        for (const secret of [...lines, ...stats, ...sets]) expect(leaks(bytes, secret), secret).toBe(false);
      }
    }
  });

  it.each(["lines", "stats", "sets"] as const)("would catch a leak: a smuggled opponent canary (%s) is found", async (kind) => {
    const secret = canaries("p2")[kind][0] as string;
    const honest = recordedChannel(record, "p1", 1, "synthetic");
    const leaky = playerChannel({ ...honest, lines: [secret, ...honest.lines] });
    const observation = await observe({ channel: leaky, team: teams.p1, manifest: record.manifest });
    expect(leaks(jcs(observation as unknown as Json), secret)).toBe(true);
  });

  it("would catch the opponent's team in an observation, which the builder refuses", async () => {
    const [honest] = await observationsOf("p1");
    const smuggled = { ...(honest as Observation), team: teams.p2 };
    const bytes = jcs(smuggled as unknown as Json);
    expect(canaries("p2").sets.some((secret) => leaks(bytes, secret))).toBe(true);
    const channel = recordedChannel(record, "p1", 1, "synthetic");
    await expect(observe({ channel, team: teams.p2, manifest: record.manifest })).rejects.toThrow(
      /the team is not this player's: no set matches Venusaur/,
    );
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
    const request = { ...(JSON.parse(last.slice("|request|".length)) as object), rqid: 9 };
    const withRqid = { ...prefix, lines: [...prefix.lines.slice(0, -1), `|request|${JSON.stringify(request)}`] };
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

  it("refuses a field the closed observation schema does not name, smuggled through an input", async () => {
    const [prefix] = decisionPrefixes(p1()) as [PlayerChannelLog];
    const rest = { channel: playerChannel(prefix), manifest: golden.manifest, sha256: nodeSha256 };
    const team = { ...synthetic(), opponent_team: synthetic() };
    await expect(buildObservation({ ...rest, team })).rejects.toThrow(/observation\/team holds opponent_team/);
    const sets = synthetic().team.map((set, i) => (i === 0 ? { ...set, secret: "x" } : set));
    await expect(buildObservation({ ...rest, team: { ...synthetic(), team: sets as PrivateTeamSpec["team"] } })).rejects.toThrow(
      /observation\/team\/team\/0 holds secret/,
    );
    const manifest = { ...golden.manifest, omniscient_log: [] };
    await expect(buildObservation({ ...rest, team: synthetic(), manifest })).rejects.toThrow(/observation\/manifest holds omniscient_log/);
    const { format_id, ...partial } = golden.manifest;
    await expect(buildObservation({ ...rest, team: synthetic(), manifest: partial as typeof golden.manifest })).rejects.toThrow(
      /observation\/manifest lacks format_id/,
    );
    const notObject = "7244fe8" as unknown as typeof golden.manifest;
    await expect(buildObservation({ ...rest, team: synthetic(), manifest: notObject })).rejects.toThrow(
      /observation\/manifest must be an object/,
    );
    const movesNotList = synthetic().team.map((set, i) => (i === 0 ? { ...set, moves: "Protect" } : set));
    await expect(
      buildObservation({ ...rest, team: { ...synthetic(), team: movesNotList as unknown as PrivateTeamSpec["team"] } }),
    ).rejects.toThrow(/observation\/team\/team\/0\/moves must be an array/);
  });

  it("checks leaf types and copies its inputs, so no later change reaches the observation", async () => {
    const [prefix] = decisionPrefixes(p1()) as [PlayerChannelLog];
    const rest = { channel: playerChannel(prefix), sha256: nodeSha256 };
    const regulation = { ...golden.manifest, regulation: 7 as unknown as string };
    await expect(buildObservation({ ...rest, team: synthetic(), manifest: regulation })).rejects.toThrow(
      /observation\/manifest\/regulation must be a string/,
    );
    const team = synthetic();
    const observation = await buildObservation({ ...rest, team, manifest: golden.manifest });
    (team.team[0] as { item: string }).item = "SECRET-ITEM";
    expect(JSON.stringify(observation)).not.toContain("SECRET-ITEM");
    const withToJson = { ...synthetic(), toJSON: () => ({}) } as unknown as PrivateTeamSpec;
    await expect(buildObservation({ ...rest, team: withToJson, manifest: golden.manifest })).rejects.toThrow(
      /an input is not plain data/,
    );
  });

  it("observe validates the whole observation against its schema", async () => {
    const [prefix] = decisionPrefixes(p1()) as [PlayerChannelLog];
    const bad = { ...golden.manifest, regulation: "" };
    await expect(observe({ channel: playerChannel(prefix), team: synthetic(), manifest: bad })).rejects.toThrow(
      /invalid observation/,
    );
  });

  describe("the own-team check, against the player's Team Preview request", () => {
    const stats = { atk: 1, def: 1, spa: 1, spd: 1, spe: 1 };
    const mon = (ident: string, details: string) => ({
      ident,
      details,
      condition: "100/100",
      active: false,
      stats,
      moves: ["hydropump"],
      baseAbility: "levitate",
      item: "sitrusberry",
      pokeball: "pokeball",
    });
    const preview = (...pokemon: object[]) =>
      `|request|${JSON.stringify({ teamPreview: true, maxChosenTeamSize: 1, side: { name: "p1", id: "p1", pokemon } })}`;
    const set = (name: string, item = "Sitrus Berry") => ({
      name,
      species: "Rotom-Wash",
      item,
      ability: "Levitate",
      moves: ["Hydro Pump"],
      nature: "Serious",
      gender: "N",
      evs: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 },
      ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 },
      level: 50,
    });
    const build = (lines: string[], sets: object[]) =>
      buildObservation({
        channel: playerChannel({ schema_version: 1, data_class: "synthetic", player: "p1", lines }),
        team: { schema_version: 1, data_class: "synthetic", team: sets as PrivateTeamSpec["team"] },
        manifest: golden.manifest,
        sha256: nodeSha256,
      });

    it("accepts an alternate forme without a nickname, which the engine names by base species", async () => {
      const observation = await build([preview(mon("p1: Rotom", "Rotom-Wash, L50"))], [set("")]);
      expect(observation.kind).toBe("preview");
    });

    it("refuses a mirror team: same species, another item", async () => {
      await expect(build([preview(mon("p1: Rotom", "Rotom-Wash, L50"))], [set("", "Leftovers")])).rejects.toThrow(
        /no set matches Rotom/,
      );
    });

    it("accepts the player's own set with its moves in another order", async () => {
      const two = { ...mon("p1: Rotom", "Rotom-Wash, L50"), moves: ["hydropump", "voltswitch"] };
      const reordered = { ...set(""), moves: ["Volt Switch", "Hydro Pump"] };
      expect((await build([preview(two)], [reordered])).kind).toBe("preview");
    });

    it("refuses a set whose nickname differs", async () => {
      await expect(build([preview(mon("p1: Rotom", "Rotom-Wash, L50"))], [set("Washer")])).rejects.toThrow(
        /no set matches Rotom/,
      );
    });

    it("refuses a channel that does not start at Team Preview", async () => {
      const side = { name: "p1", id: "p1", pokemon: [mon("p1: Rotom", "Rotom-Wash, L50")] };
      const move = `|request|${JSON.stringify({ active: [{ moves: [{ move: "Hydro Pump", id: "hydropump" }] }], side })}`;
      await expect(build([move], [set("")])).rejects.toThrow(/does not start at Team Preview/);
    });
  });

  it("brands only a valid channel of that player's own requests", () => {
    expect(() => playerChannel({ ...p1(), extra: 1 })).toThrow(/not a PlayerChannelLog/);
    expect(() => playerChannel({ ...p1(), player: "p2" })).toThrow(/p2's channel holds a request for p1/);
    expect(() => playerChannel({ ...p1(), lines: ['|request|{"side":{}}'] })).toThrow(/engine's request shape/);
    expect(() => recordedChannel(golden, "p1", 999, "synthetic")).toThrow(/no request 999/);
  });
});
