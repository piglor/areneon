import { readdirSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { A2Channel, type A2ChannelOptions, ProtocolReader, splitFrame } from "@areneon/bot-client";
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
 * bot-vs-bot fixture. The A2 side starts from `fixture.frames`: the raw
 * server frames each bot received. They go through `ProtocolReader` and
 * `A2Channel` (with `stripServerLines`) here, as the session loop does, so an
 * adapter regression changes the result without any fixture changing. The
 * stored `channels` are not read; the input log only drives A1.
 */

/** The part of `A2Channel` the check uses; a test passes a broken one to prove the check notices. */
export interface Adapter {
  receive(frame: string): boolean;
  build(): { channel: PlayerChannel; dropped: string[]; resent: number };
}
export type AdapterFactory = (options: A2ChannelOptions) => Adapter;
const realAdapter: AdapterFactory = (options) => new A2Channel(options);

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
  /** Per seat, how much the adapter had to do: frames fed, lines in them, lines in the channel, re-sent requests it left out. */
  frames: Record<Seat, number>;
  framedLines: Record<Seat, number>;
  channelLines: Record<Seat, number>;
  resent: Record<Seat, number>;
}

const zeros = (): Record<Seat, number> => ({ p1: 0, p2: 0 });

/**
 * Play `fixture` on A1, step by step, with `replayActor` over the choices of
 * its input log, and compare with the fixture's channels (the A2 side).
 * Never throws for a difference: it lists it.
 */
export async function compareA1A2(
  engine: Engine,
  snapshot: RulesSnapshot,
  fixture: HarnessFixture,
  adapter: AdapterFactory = realAdapter,
): Promise<EqualityResult> {
  const differences: string[] = [];
  const diff = (message: string): void => void differences.push(message);
  const result: EqualityResult = {
    differences,
    requests: zeros(),
    decisions: zeros(),
    frames: zeros(),
    framedLines: zeros(),
    channelLines: zeros(),
    resent: zeros(),
  };

  // The A2 side: each seat's raw frames through the protocol reader
  // (request_seq, request_hash) and the A2 channel adapter, frame by frame as
  // the session loop feeds them. A re-sent request keeps its request_seq in
  // the reader and is left out of the channel by the adapter.
  const a2: Record<Seat, { channel: PlayerChannel; waits: boolean[]; reads: { seq: number; hash: string }[] }> = {
    p1: undefined as never,
    p2: undefined as never,
  };
  for (const seat of SEATS) {
    try {
      const reader = new ProtocolReader();
      const channel = adapter({ room: fixture.room, player: seat, data_class: "synthetic" });
      const reads: { seq: number; hash: string }[] = [];
      for (const frame of fixture.frames[seat]) {
        for (const event of reader.read(frame)) {
          if (event.kind === "request" && !event.resend) {
            reads.push({ seq: event.request.request_seq, hash: event.request.request_hash });
          }
        }
        channel.receive(frame);
        result.framedLines[seat] += splitFrame(frame).lines.length;
      }
      const built = channel.build();
      result.frames[seat] = fixture.frames[seat].length;
      result.channelLines[seat] = built.channel.lines.length;
      result.resent[seat] = built.resent;
      const requests = built.channel.lines.filter((l) => l.startsWith(REQUEST));
      a2[seat] = { channel: built.channel, waits: requests.map(isWait), reads };
    } catch (error) {
      diff(`${seat}: the A2 adapter refused the frames: ${(error as Error).message}`);
    }
  }
  if (differences.length > 0) return result;

  // Not handled: refused (`[Unavailable choice]`), `default` and auto-chosen
  // turns. The mapping assumes one logged choice per non-wait request. A
  // fixture with such a turn shows up as a "choices in the input log for"
  // count difference, or as "A1 could not replay" when a choice is illegal.
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
      engine,
      snapshot,
      formatId: fixture.format_id,
      seed: fixture.seed,
      teams: fixture.served_teams,
      players: {
        p1: { name: fixture.players.p1.name, avatar: fixture.players.p1.avatar },
        p2: { name: fixture.players.p2.name, avatar: fixture.players.p2.avatar },
      },
      actors: { p1: replayActor(choices, "p1"), p2: replayActor(choices, "p2") },
    });
  } catch (error) {
    diff(`A1 could not replay the fixture: ${(error as Error).message}`);
    return result;
  }

  // A1 was started as the server was: the input log's own seed, names,
  // avatars and packed teams against A1's first lines. The server adds only
  // `rating` to a `>player` line.
  const logged = fixture.input_log.lines.slice(0, 3);
  if (!(logged[0] ?? "").startsWith(">start ") || !(logged[1] ?? "").startsWith(">player p1 ") || !(logged[2] ?? "").startsWith(">player p2 ")) {
    diff("the input log does not start with >start, >player p1, >player p2");
  }
  const started = logged.flatMap((line) => {
    const match = /^>(start|player p[12]) (\{.*\})$/.exec(line);
    return match === null ? [] : [{ kind: match[1] as string, body: JSON.parse(match[2] as string) as Record<string, string> }];
  });
  // A1's own `>player` JSON (rating and any extra key) is not visible here:
  // name and avatar are read back from its `|player|` lines, and the team
  // side is recomputed from `served_teams` with the same pack.
  const expectedStart = started.flatMap(({ kind, body }) =>
    kind === "start"
      ? [`format ${body.formatid} seed ${body.seed}`]
      : [`${kind.slice(-2)} ${body.name} ${body.avatar ?? ""} ${body.team}`],
  );
  const a1Start = [
    `format ${fixture.format_id} seed ${record.seed}`,
    ...SEATS.map((seat) => {
      const line = record.omniscient_log.find((l) => l.startsWith(`|player|${seat}|`));
      const parts = line?.split("|");
      if (parts?.length !== 6) throw new Error(`A1 wrote no |player|${seat}| line`);
      const [, , , name, avatar] = parts;
      return `${seat} ${name} ${avatar} ${engine.sim.Teams.pack(fixture.served_teams[seat] as never)}`;
    }),
  ];
  if (a1Start.join("\n") !== expectedStart.join("\n")) {
    diff("A1 was not started with the input log's >start and >player lines");
  }

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
