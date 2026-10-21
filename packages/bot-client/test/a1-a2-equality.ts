import { readdirSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { A2Channel, type A2ChannelOptions, ProtocolReader, splitFrame } from "@areneon/bot-client";
import { type HarnessFixture, type HarnessFixtureV2, reconnectEvidence, type Seat, upgradeFixture } from "@areneon/local-server/harness";
import {
  battleClass,
  type BattleRecord,
  type Engine,
  jcs,
  type Observation,
  continueChannel,
  ObservationError,
  observe,
  type PlayerChannel,
  playerChannel,
  recordedChannel,
  replayActor,
  resolveFormat,
  type RulesSnapshot,
  runBattle,
  validateTeam,
} from "@areneon/showdown-bridge";

/**
 * The A1/A2 equality check (ADR-005 gate 3, ADR-008 §4) over a recorded
 * bot-vs-bot fixture. The A2 side starts from `fixture.frames`: the raw
 * server frames each bot received, one list per connection. Each connection
 * goes through its own `ProtocolReader` and `A2Channel` (with
 * `stripServerLines`) here, as the session does (it makes both anew for every
 * connection), so an adapter regression changes the result without any
 * fixture changing. The stored `channels` are not read; the input log only
 * drives A1.
 *
 * One connection per seat is the plain case: the A2 channel and the A1
 * channel are the same stream, line for line, and every observation is
 * compared with `decision_context` and `rqid` removed (ADR-008 §4), nothing else.
 *
 * A seat that reconnected has several connections. Each is its own channel
 * (it restarts at its own `|init|battle`; the server replays the room log to
 * it, and that log holds no `|request|` line, so the rejoin channel alone has
 * no Team Preview request and its `request_seq` counts from 1 again). The
 * check joins them with `continueChannel` (the rule of #813: a rejoin channel
 * carries the anchor of the connection it continues), connection by
 * connection, and compares the joined channel as the plain case does:
 *
 * - The joined channel of every connection is a prefix of A1's one continuous
 *   channel, and of the last one it is equal, line for line, with nothing
 *   normalised but `rqid` (ADR-008 §4).
 * - At every decision a connection received, the observation from the joined
 *   channel equals A1's FULL observation of that request (`request_seq`, log
 *   and team included), with `decision_context` and `rqid` removed.
 * - Across connections a request is identified by `rqid` + `request_hash`
 *   (ADR-008 §4): a later connection's first request that has the previous
 *   connection's last request's `rqid` and hash is that request sent again,
 *   one decision; the `rqid`s of the decisions rise; the `request_hash` of
 *   every decision equals A1's, whatever connection received it.
 */

/** The part of `A2Channel` the check uses; a test passes a broken one to prove the check notices. */
export interface Adapter {
  receive(frame: string): boolean;
  build(): { channel: PlayerChannel; dropped: string[]; resent: number };
}
/** Called once per connection of a seat, as the session makes an `A2Channel` for every connection. */
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

/** A committed fixture of any schema version still read, as the current one (`upgradeFixture`). */
export function readFixture(path: string): HarnessFixture {
  return upgradeFixture(JSON.parse(readFileSync(`${A2_FIXTURES_DIR}${path}`, "utf8")) as HarnessFixture | HarnessFixtureV2);
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

export interface EqualityResult {
  /** Every difference found; empty when A1 and A2 agree. */
  differences: string[];
  /** Per seat: A1's requests (wait included), and the decisions (non-wait requests) compared; a re-sent request is one. */
  requests: Record<Seat, number>;
  decisions: Record<Seat, number>;
  /** Per seat: observations compared (a decision seen by two connections is compared on both). */
  observations: Record<Seat, number>;
  /** Per seat: each connection's channel joined to the ones before it (`continueChannel`); the last is the whole battle. */
  continued: Record<Seat, PlayerChannel[]>;
  /** Per seat: connections, and requests that a later connection received again from the one before (same `rqid` and hash). */
  connections: Record<Seat, number>;
  resentAcross: Record<Seat, number>;
  /** Per seat, how much the adapter had to do: frames fed, lines in them, lines in the channels, re-sent requests it left out. */
  frames: Record<Seat, number>;
  framedLines: Record<Seat, number>;
  channelLines: Record<Seat, number>;
  resent: Record<Seat, number>;
}

const zeros = (): Record<Seat, number> => ({ p1: 0, p2: 0 });

/** The A1 and the A2 observation of one decision, as the check built them (`onPair`). */
export interface ObservationPair {
  seat: Seat;
  /** The connection of the seat that received the decision (0 for the first). */
  connection: number;
  /** The decision number among the seat's requests (A1's `request_seq`). */
  request_seq: number;
  a1: Observation;
  a2: Observation;
  /** The team the builder was given (the validator's defaults filled in). */
  team: unknown;
}

/** One request as a connection's `ProtocolReader` read it. */
interface Read {
  seq: number;
  rqid: number | undefined;
  hash: string;
  wait: boolean;
}

/** One connection of a seat as the A2 side rebuilt it. */
interface A2Connection {
  /** The channel of this connection joined to the ones before it. */
  channel: PlayerChannel;
  reads: Read[];
  /** Requests before this connection's first one: its k-th read is A1's request `base + k`. */
  base: number;
}

/**
 * Play `fixture` on A1, step by step, with `replayActor` over the choices of
 * its input log, and compare with the A2 side rebuilt from its frames (see
 * the top of this file for the rule). Never throws for a difference: it
 * lists it.
 */
export async function compareA1A2(
  engine: Engine,
  snapshot: RulesSnapshot,
  fixture: HarnessFixture,
  adapter: AdapterFactory = realAdapter,
  onPair?: (pair: ObservationPair) => void,
): Promise<EqualityResult> {
  const differences: string[] = [];
  const diff = (message: string): void => void differences.push(message);
  const result: EqualityResult = {
    differences,
    requests: zeros(),
    decisions: zeros(),
    observations: zeros(),
    continued: { p1: [], p2: [] },
    connections: zeros(),
    resentAcross: zeros(),
    frames: zeros(),
    framedLines: zeros(),
    channelLines: zeros(),
    resent: zeros(),
  };

  // The A2 side: each connection's raw frames through its own protocol reader
  // (request_seq, request_hash) and A2 channel adapter, frame by frame as the
  // session loop feeds them. A re-sent request keeps its request_seq in the
  // reader and is left out of the channel by the adapter. The decisions of a
  // seat are its requests in order, a request that a connection receives again
  // from the one before counted once: `logical` holds each once, by A1's number.
  const a2 = {} as Record<Seat, { connections: A2Connection[]; logical: Read[] }>;
  for (const seat of SEATS) {
    try {
      const connections: A2Connection[] = [];
      const logical: Read[] = [];
      for (const frames of fixture.frames[seat]) {
        const reader = new ProtocolReader();
        // synthetic by construction (recorded fixtures); derive it from team hashes (team-class.ts) when real teams can reach this path.
        const channel = adapter({ room: fixture.room, player: seat, data_class: "synthetic" });
        const reads: Read[] = [];
        for (const frame of frames) {
          for (const event of reader.read(frame)) {
            if (event.kind === "request" && !event.resend) {
              const { request_seq: seq, rqid, request_hash: hash, phase } = event.request;
              reads.push({ seq, rqid, hash, wait: phase === "wait" });
            }
          }
          channel.receive(frame);
          result.framedLines[seat] += splitFrame(frame).lines.length;
        }
        const built = channel.build();
        result.frames[seat] += frames.length;
        result.channelLines[seat] += built.channel.lines.length;
        result.resent[seat] += built.resent;
        const first = reads[0];
        const before = logical.at(-1);
        const again = first !== undefined && before !== undefined && first.rqid !== undefined && first.rqid === before.rqid && first.hash === before.hash;
        const base = logical.length - (again ? 1 : 0);
        if (again) result.resentAcross[seat] += 1;
        logical.push(...reads.slice(again ? 1 : 0));
        const earlier = connections.at(-1)?.channel;
        connections.push({ channel: earlier === undefined ? built.channel : continueChannel(earlier, built.channel), reads, base });
      }
      result.connections[seat] = connections.length;
      result.continued[seat] = connections.map((c) => c.channel);
      a2[seat] = { connections, logical };
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
  // A reconnect the fixture says it holds is the one it holds: the re-sent request is there, with
  // `|sentchoice|` exactly when its provenance says the cut came after the choice, and that choice is
  // the one the input log has for the decision.
  const claimed = /(right after it sent its choice for|when it received) its request/.exec(fixture.provenance)?.[1];
  for (const seat of SEATS.filter((s) => fixture.frames[s].length > 1)) {
    const evidence = reconnectEvidence(fixture.frames[seat]);
    if (evidence === undefined) {
      if (claimed !== undefined) diff(`${seat}: no request is re-sent on its second connection`);
    } else {
      if (claimed !== undefined && evidence.sentchoice !== claimed.startsWith("right after")) {
        diff(`${seat}: |sentchoice| ${evidence.sentchoice ? "is" : "is not"} there, the provenance says the cut came ${claimed}`);
      }
    }
  }
  const held: Partial<Record<Seat, { decision: number; choice: string }>> = {};
  for (const seat of SEATS.filter((s) => fixture.frames[s].length > 1)) {
    const evidence = reconnectEvidence(fixture.frames[seat]);
    if (evidence?.choice !== undefined) held[seat] = { decision: evidence.decision, choice: evidence.choice };
  }
  const choices: BattleRecord["choices"] = [];
  for (const seat of SEATS) {
    const answered = a2[seat].logical.flatMap((read, i) => (read.wait ? [] : [i + 1]));
    if (answered.length !== queued[seat].length) {
      diff(`${seat}: ${queued[seat].length} choices in the input log for ${answered.length} decisions in the channel`);
    }
    answered.forEach((request_seq, i) => {
      const choice = held[seat]?.decision === i + 1 ? held[seat]?.choice : queued[seat][i];
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
    const { connections, logical } = a2[seat];
    const theirs = record.channels[seat];
    const requests = record.requests[seat];
    result.requests[seat] = requests.length;

    // Request identity at every decision: A2's readers against A1's runner,
    // whichever connection received it. The rqids are the server's, so they rise.
    if (logical.length !== requests.length) diff(`${seat}: A2 read ${logical.length} requests, A1 sent ${requests.length}`);
    requests.forEach((request, i) => {
      const read = logical[i];
      if (read === undefined) return;
      if (read.hash !== request.request_hash) diff(`${seat} request ${i + 1}: request_hash ${read.hash} vs ${request.request_hash}`);
      const before = logical[i - 1];
      if (before !== undefined && (read.rqid ?? 0) <= (before.rqid ?? 0)) {
        diff(`${seat} request ${i + 1}: rqid ${String(read.rqid)} does not follow ${String(before.rqid)}`);
      }
    });

    // synthetic by construction (recorded fixtures); derive it from team hashes (team-class.ts) when real teams can reach this path.
    // The runner plays copies, so the served sets stay as recorded; the builder wants them with the
    // validator's defaults (gender, ...) filled in, as the engine hands them to the player.
    const filled = structuredClone(fixture.served_teams[seat]);
    validateTeam(engine, resolveFormat(engine, fixture.format_id, snapshot), filled);
    const team = { schema_version: 1 as const, data_class: "synthetic" as const, team: filled as never };
    const decided = new Set<number>();
    for (const [c, connection] of connections.entries()) {
      const who = connections.length === 1 ? seat : `${seat} connection ${c + 1}`;
      const mine = connection.channel.lines;
      // The joined channel is a prefix of A1's one continuous channel, the whole of it for the last.
      const expected = theirs;
      // Per-player channel lines, in order; only a request's rqid is left out.
      // A connection that was cut is a prefix of what follows it.
      const width = c === connections.length - 1 ? Math.max(mine.length, expected.length) : mine.length;
      for (let i = 0; i < width; i++) {
        const a = mine[i] === undefined ? undefined : withoutRqid(mine[i] as string);
        const b = expected[i] === undefined ? undefined : withoutRqid(expected[i] as string);
        if (a !== b) diff(`${who} channel line ${i + 1} differs: A2 ${JSON.stringify(a)} vs A1 ${JSON.stringify(b)}`);
      }

      // An observation at every decision this connection received, both
      // sides, from the player's own channel prefix and own team.
      for (const [k, read] of connection.reads.entries()) {
        const seq = k + 1;
        if (read.seq !== seq) diff(`${who} request ${seq}: request_seq ${read.seq}`);
        const request = requests[connection.base + seq - 1];
        if (read.wait || request === undefined) continue;
        let seen = 0;
        const end = mine.findIndex((l) => l.startsWith(REQUEST) && ++seen === connection.base + seq);
        if (end < 0) {
          diff(`${who} request ${seq}: not in the A2 channel`);
          continue;
        }
        const cut = recordedChannel(
          record,
          seat,
          request.request_seq,
          // The recorded fixtures' served teams are synthetic test data: their own hashes are the known set.
          battleClass(record.team_hashes, new Set(Object.values(record.team_hashes))),
        );
        const common = { team, manifest: fixture.manifest as never };
        let one: Observation;
        let two: Observation;
        try {
          one = await observe({ ...common, channel: cut });
          two = await observe({ ...common, channel: playerChannel({ ...connection.channel, lines: mine.slice(0, end + 1) }) });
        } catch (error) {
          if (!(error instanceof ObservationError)) throw error;
          diff(`${who} decision ${seq}: no observation could be built: ${error.message}`);
          continue;
        }
        decided.add(connection.base + seq);
        result.observations[seat] += 1;
        if (two.identity.request_hash !== request.request_hash) {
          diff(`${who} decision ${seq}: observation request_hash ${two.identity.request_hash} vs ${request.request_hash}`);
        }
        if (comparable(one) !== comparable(two)) diff(`${who} decision ${seq}: observations differ`);
        onPair?.({ seat, connection: c, request_seq: request.request_seq, a1: one, a2: two, team: team.team });
      }
    }
    result.decisions[seat] = decided.size;
  }
  return result;
}
