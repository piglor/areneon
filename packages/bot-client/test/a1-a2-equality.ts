import { readdirSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { ProtocolReader } from "@areneon/bot-client";
import type { HarnessFixture, Seat } from "@areneon/local-server/harness";
import {
  type BattleRecord,
  type Engine,
  jcs,
  type Observation,
  observe,
  type PlayerChannel,
  playerChannel,
  recordedChannel,
  replayActor,
  type RulesSnapshot,
  runBattle,
} from "@areneon/showdown-bridge";

/**
 * The A1/A2 equality check (ADR-005 gate 3, ADR-008 §4) over a recorded
 * bot-vs-bot fixture. `fixture.channels` is already the A2 adapter's output
 * (A2Channel + stripServerLines ran in the bot client that recorded it), so
 * it is the A2 side as it is; the input log only drives A1.
 */

export const SEATS: readonly Seat[] = ["p1", "p2"];
export const A2_FIXTURES_DIR = fileURLToPath(new URL("../../../fixtures/a2/", import.meta.url));

/** Every committed A2 fixture, as `<format>/<file>`. */
export function fixtureFiles(): string[] {
  return readdirSync(A2_FIXTURES_DIR, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .flatMap((dir) =>
      readdirSync(`${A2_FIXTURES_DIR}${dir.name}`)
        .filter((file) => file.endsWith(".json"))
        .map((file) => `${dir.name}/${file}`),
    );
}

export function readFixture(path: string): HarnessFixture {
  return JSON.parse(readFileSync(`${A2_FIXTURES_DIR}${path}`, "utf8")) as HarnessFixture;
}

/**
 * `engine` whose battle streams get the recorded names and avatars on their
 * `>player` lines (`runBattle` writes `name: p1` and no avatar). Every line
 * it writes to the sim is kept in `written`.
 */
export function withPlayers(engine: Engine, players: HarnessFixture["players"], written: string[]): Engine {
  const sim = new Proxy(engine.sim, {
    get(target, key, receiver) {
      if (key !== "getPlayerStreams") return Reflect.get(target, key, receiver) as unknown;
      return (stream: object) => {
        const streams = target.getPlayerStreams(stream);
        const write = streams.omniscient.write.bind(streams.omniscient);
        streams.omniscient.write = (chunk: string) => {
          const lines = chunk.split("\n").map((line) => {
            const match = /^>player (p[12]) (\{.*\})$/.exec(line);
            if (match === null) return line;
            const seat = match[1] as Seat;
            const { team } = JSON.parse(match[2] as string) as { team: string };
            return `>player ${seat} ${JSON.stringify({ name: players[seat].name, avatar: players[seat].avatar, team })}`;
          });
          written.push(...lines);
          write(lines.join("\n"));
        };
        return streams;
      };
    },
  });
  return { ...engine, sim };
}

const REQUEST = "|request|";

/** A request line without its `rqid` (ADR-008 §4), reserialised the same way on both sides. */
export function withoutRqid(line: string): string {
  if (!line.startsWith(REQUEST)) return line;
  const { rqid: _rqid, ...rest } = JSON.parse(line.slice(REQUEST.length)) as Record<string, unknown>;
  return REQUEST + JSON.stringify(rest);
}

/** An observation as ADR-008 §4 compares it: no `decision_context`, no `rqid` anywhere. */
export function comparable(observation: Observation): string {
  const { decision_context: _context, ...rest } = observation;
  const { rqid: _rqid, ...identity } = rest.identity;
  return jcs({ ...rest, identity, log: rest.log.map(withoutRqid) } as never);
}

/** Whether the request line is a wait request. */
const isWait = (line: string): boolean => (JSON.parse(line.slice(REQUEST.length)) as { wait?: true }).wait === true;

export interface EqualityResult {
  /** Every difference found; empty when A1 and A2 agree. */
  differences: string[];
  /** Per seat: requests compared (wait included), decisions compared (observations). */
  requests: Record<Seat, number>;
  decisions: Record<Seat, number>;
}

/**
 * Play `fixture` on A1, step by step, with `replayActor` over the choices of
 * its input log, and compare with the fixture's channels (the A2 side).
 * Never throws for a difference: it lists it.
 */
export async function compareA1A2(engine: Engine, snapshot: RulesSnapshot, fixture: HarnessFixture): Promise<EqualityResult> {
  const differences: string[] = [];
  const diff = (message: string): void => void differences.push(message);
  const written: string[] = [];

  // The A2 side: each seat's own stored channel, its requests through the
  // protocol reader (request_seq, request_hash) as the session loop reads them.
  const a2: Record<Seat, { channel: PlayerChannel; waits: boolean[]; reads: { seq: number; hash: string }[] }> = {
    p1: undefined as never,
    p2: undefined as never,
  };
  for (const seat of SEATS) {
    const channel = playerChannel(fixture.channels[seat]);
    const reader = new ProtocolReader();
    const room = "battle-equality";
    reader.read(`>${room}\n|init|battle`);
    const reads: { seq: number; hash: string }[] = [];
    for (const line of channel.lines.filter((l) => l.startsWith(REQUEST))) {
      for (const event of reader.read(`>${room}\n${line}`)) {
        if (event.kind === "request") reads.push({ seq: event.request.request_seq, hash: event.request.request_hash });
      }
    }
    a2[seat] = { channel, waits: channel.lines.filter((l) => l.startsWith(REQUEST)).map(isWait), reads };
  }

  // The choices of the input log, each assigned to the request it answers:
  // the n-th choice of a seat answers that seat's n-th non-wait request.
  const queued: Record<Seat, string[]> = { p1: [], p2: [] };
  for (const line of fixture.input_log.lines) {
    const match = /^>(p[12]) (.*)$/.exec(line);
    if (match !== null && !match[2]?.startsWith("{")) queued[match[1] as Seat].push(match[2] as string);
  }
  const choices: BattleRecord["choices"] = [];
  for (const seat of SEATS) {
    const answered = a2[seat].waits.flatMap((wait, i) => (wait ? [] : [i + 1]));
    if (answered.length !== queued[seat].length) {
      diff(`${seat}: ${queued[seat].length} choices in the input log for ${answered.length} decisions in the channel`);
    }
    answered.forEach((request_seq, i) => {
      const choice = queued[seat][i];
      if (choice !== undefined) choices.push({ player: seat, request_seq, choice });
    });
  }

  let record: BattleRecord;
  try {
    record = await runBattle({
      engine: withPlayers(engine, fixture.players, written),
      snapshot,
      formatId: fixture.format_id,
      seed: fixture.seed,
      teams: fixture.served_teams,
      actors: { p1: replayActor(choices, "p1"), p2: replayActor(choices, "p2") },
    });
  } catch (error) {
    diff(`A1 could not replay the fixture: ${(error as Error).message}`);
    return { differences, requests: { p1: 0, p2: 0 }, decisions: { p1: 0, p2: 0 } };
  }

  // A1 was started exactly as the server was (seed, names, avatars, packed teams).
  const logged = fixture.input_log.lines.slice(0, 3);
  // The server adds only `rating` to a `>player` line; name, avatar and team must be the same.
  const normal = (line: string): string => {
    const match = /^(>player p[12] )(\{.*\})$/.exec(line);
    if (match === null) return line;
    const { name, avatar, team } = JSON.parse(match[2] as string) as Record<string, unknown>;
    return (match[1] as string) + JSON.stringify({ name, avatar, team });
  };
  if (written.slice(0, 3).map(normal).join("\n") !== logged.map(normal).join("\n")) {
    diff("A1 was not started with the input log's >start and >player lines");
  }

  const result: EqualityResult = { differences, requests: { p1: 0, p2: 0 }, decisions: { p1: 0, p2: 0 } };
  for (const seat of SEATS) {
    const mine = a2[seat].channel.lines;
    const theirs = record.channels[seat];
    // Per-player channel lines, in order; only a request's rqid is left out.
    const width = Math.max(mine.length, theirs.length);
    for (let i = 0; i < width; i++) {
      const a = mine[i] === undefined ? undefined : withoutRqid(mine[i] as string);
      const b = theirs[i] === undefined ? undefined : withoutRqid(theirs[i] as string);
      if (a !== b) diff(`${seat} channel line ${i + 1} differs: A2 ${JSON.stringify(a)} vs A1 ${JSON.stringify(b)}`);
    }

    // Request identity at every request: A2's reader against A1's runner.
    const requests = record.requests[seat];
    result.requests[seat] = requests.length;
    if (a2[seat].reads.length !== requests.length) diff(`${seat}: A2 read ${a2[seat].reads.length} requests, A1 sent ${requests.length}`);
    requests.forEach((request, i) => {
      const read = a2[seat].reads[i];
      if (read === undefined) return;
      if (read.seq !== request.request_seq) diff(`${seat} request ${i + 1}: request_seq ${read.seq} vs ${request.request_seq}`);
      if (read.hash !== request.request_hash) diff(`${seat} request ${i + 1}: request_hash ${read.hash} vs ${request.request_hash}`);
    });

    // An observation at every decision, both sides, from the player's own
    // channel prefix and own team.
    const team = { schema_version: 1 as const, data_class: "synthetic" as const, team: fixture.served_teams[seat] as never };
    for (const request of requests.filter((r) => r.phase !== "wait")) {
      const seq = request.request_seq;
      let seen = 0;
      const end = mine.findIndex((l) => l.startsWith(REQUEST) && ++seen === seq);
      if (end < 0) {
        diff(`${seat} request ${seq}: not in the A2 channel`);
        continue;
      }
      const common = { team, manifest: fixture.manifest as never };
      const one = await observe({ ...common, channel: recordedChannel(record, seat, seq, "synthetic") });
      const two = await observe({
        ...common,
        channel: playerChannel({ ...a2[seat].channel, lines: mine.slice(0, end + 1) }),
      });
      result.decisions[seat] += 1;
      if (two.identity.request_hash !== one.identity.request_hash) {
        diff(`${seat} decision ${seq}: observation request_hash ${two.identity.request_hash} vs ${one.identity.request_hash}`);
      }
      if (comparable(one) !== comparable(two)) diff(`${seat} decision ${seq}: observations differ`);
    }
  }
  return result;
}
