import { A2Channel, type A2ChannelOptions, splitFrame } from "@areneon/bot-client";
import { loadEngine, loadRulesSnapshot, observe, type PlayerChannel, playerChannel } from "@areneon/showdown-bridge";
import { channelFromFrames, type HarnessFixture, reconnectEvidence, verifyHarnessFixture } from "@areneon/local-server/harness";
import { describe, expect, it } from "vitest";
import { type Adapter, type AdapterFactory, compareA1A2, fixtureFiles, readFixture, SEATS } from "./a1-a2-equality.ts";

const engine = loadEngine();
const snapshot = loadRulesSnapshot(engine.commit);
const files = fixtureFiles();
const clone = (path: string) => structuredClone(readFixture(path));

type Seat = (typeof SEATS)[number];

/** How a seat's connection is named in a difference: just the seat when it has one. */
const who = (fixture: HarnessFixture, seat: Seat, connection = 0): string =>
  fixture.frames[seat].length === 1 ? seat : `${seat} connection ${connection + 1}`;

/** The seats that reconnected, with their connection counts. */
const reconnected = (fixture: HarnessFixture): Seat[] => SEATS.filter((seat) => fixture.frames[seat].length > 1);

/**
 * Rewrites the first (or last) line of a seat's frames that `pick` selects, in
 * its first connection unless `connection` says otherwise; returns its place.
 */
function editLine(
  fixture: HarnessFixture,
  seat: Seat,
  pick: (line: string) => boolean,
  edit: (line: string) => string,
  last = false,
  connection = 0,
): { frame: number; line: number } {
  const frames = fixture.frames[seat][connection] as string[];
  const places = frames.flatMap((frame, f) =>
    frame.split("\n").flatMap((line, l) => (pick(line) ? [{ frame: f, line: l }] : [])),
  );
  const at = last ? places.at(-1) : places[0];
  expect(at, "a line to edit").toBeDefined();
  const { frame, line } = at as { frame: number; line: number };
  const lines = (frames[frame] as string).split("\n");
  lines[line] = edit(lines[line] as string);
  frames[frame] = lines.join("\n");
  return at as { frame: number; line: number };
}

/** The request line of a seat's connection that arrives in a frame of its own with other lines, to send again. */
function resendAfter(fixture: HarnessFixture, seat: Seat, connection: number): void {
  const frames = fixture.frames[seat][connection] as string[];
  const at = frames.findIndex((frame) => frame.includes("|request|") && !frame.includes('"wait":true'));
  const request = (frames[at] as string).split("\n").find((l) => l.startsWith("|request|")) as string;
  // The server's onConnect re-send: the same request in a frame of its own.
  frames.splice(at + 1, 0, `>${fixture.room}\n${request}`);
}

/** An adapter that does not dedupe: each request gets a rqid of its own, so no copy equals the last one. */
function noDedupe(options: A2ChannelOptions): Adapter {
  const inner = new A2Channel(options);
  let n = 1_000_000;
  return {
    receive: (frame) =>
      inner.receive(
        frame
          .split("\n")
          .map((line) => (line.startsWith("|request|") ? line.replace(/"rqid":\d+/, `"rqid":${n++}`) : line))
          .join("\n"),
      ),
    build: () => inner.build(),
  };
}

/**
 * An adapter that does not start a new channel at the rejoin: one `A2Channel` per seat for all of
 * its connections, and the later connections' `|init|battle` taken out, so the replayed room log
 * lands after the old history.
 */
function brokenRejoin(): AdapterFactory {
  const channels = new Map<string, A2Channel>();
  return (options) => {
    const later = channels.has(options.player);
    const inner = channels.get(options.player) ?? new A2Channel(options);
    channels.set(options.player, inner);
    return {
      receive: (frame) => inner.receive(later ? frame.replace("\n|init|battle", "") : frame),
      build: () => inner.build(),
    };
  };
}

/** An adapter that skips `stripServerLines`: the room's lines go straight into a channel. */
function noStrip(options: A2ChannelOptions): Adapter {
  const lines: string[] = [];
  return {
    receive(frame) {
      const split = splitFrame(frame);
      if (split.room !== options.room) return false;
      lines.push(...split.lines);
      return true;
    },
    build: () => ({
      channel: playerChannel({ schema_version: 1, data_class: options.data_class, player: options.player, lines }),
      dropped: [],
      resent: 0,
    }),
  };
}

/**
 * A mutated line is noticed as a channel line difference, or, in a seat that reconnected, already
 * when the rejoin's replay no longer continues the earlier connection (`continueChannel` refuses).
 */
const noticed = (fixture: HarnessFixture, seat: Seat, at: number, differences: readonly string[]): boolean =>
  differences.some(
    (d) =>
      d.startsWith(`${who(fixture, seat)} channel line ${at + 1} differs`) ||
      (fixture.frames[seat].length > 1 && d.startsWith(`${seat}: the A2 adapter refused the frames`)),
  );

const stripBypassNoticed = (differences: readonly string[]): boolean =>
  differences.some((d) => d.includes("channel line") || d.includes("refused"));

const reconnecting = files.filter((path) => reconnected(readFixture(path)).length > 0);
const plain = files.filter((path) => !reconnecting.includes(path));

describe("A1/A2 equality (ADR-005 gate 3)", () => {
  it("finds the committed fixtures, plain ones and ones with a real reconnect", () => {
    expect(plain.length).toBeGreaterThanOrEqual(2);
    expect(reconnecting.length).toBeGreaterThanOrEqual(2);
  });

  it("has a reconnect with |sentchoice| and one without, each a re-sent move request after its own |init|battle", () => {
    const evidence = reconnecting.flatMap((path) => {
      const fixture = readFixture(path);
      return reconnected(fixture).map((seat) => {
        const connections = fixture.frames[seat];
        expect(connections).toHaveLength(2);
        expect(connections[1]?.[0]).toContain("\n|init|battle");
        const found = reconnectEvidence(connections);
        expect(found, `${path} ${seat}`).toBeDefined();
        expect(found?.phase).not.toBe("wait");
        return found?.sentchoice;
      });
    });
    expect([...new Set(evidence)].sort()).toEqual([false, true]);
  });

  describe.each(files)("%s", (path) => {
    it("is a valid harness fixture", async () => {
      await verifyHarnessFixture(engine, readFixture(path));
    });

    it("starts from raw server frames: the stored channels are what the adapter makes of each connection", () => {
      const fixture = readFixture(path);
      for (const seat of SEATS) {
        expect(fixture.channels[seat]).toHaveLength(fixture.frames[seat].length);
        fixture.frames[seat].forEach((frames, c) => {
          const { channel, requests } = channelFromFrames(frames, fixture.room, seat);
          expect(channel).toEqual(fixture.channels[seat][c]);
          expect(requests).toBeGreaterThan(0);
        });
      }
    });

    it("plays on A1 to the same per-player channels, request hashes and full observations as A2", async () => {
      const fixture = readFixture(path);
      const result = await compareA1A2(engine, snapshot, fixture);
      expect(result.differences).toEqual([]);
      for (const seat of SEATS) {
        const channels = fixture.channels[seat];
        const connections = channels.length;
        expect(result.connections[seat]).toBe(connections);
        // Not vacuous: something was compared, one observation per non-wait request.
        expect(result.requests[seat]).toBeGreaterThan(0);
        expect(result.decisions[seat]).toBeGreaterThan(0);
        expect(result.decisions[seat]).toBeLessThanOrEqual(result.requests[seat]);
        const requests = channels.flatMap((c) => c.lines.filter((l) => l.startsWith("|request|")));
        const waits = requests.filter((l) => (JSON.parse(l.slice(9)) as { wait?: true }).wait === true);
        // A request that a later connection received again is one request, and one decision.
        expect(result.requests[seat]).toBe(requests.length - result.resentAcross[seat]);
        expect(result.resentAcross[seat]).toBe(connections - 1);
        expect(result.decisions[seat]).toBe(result.requests[seat] - waits.length);
        // One observation per non-wait request a connection received, every one built and equal to
        // A1's full observation, including those of a later connection (its joined channel).
        expect(result.observations[seat]).toBe(requests.length - waits.length);
        // The last connection's joined channel is the whole battle: one continuous channel.
        expect(result.continued[seat]).toHaveLength(connections);
        expect(result.continued[seat].at(-1)?.lines.filter((l) => l.startsWith("|request|"))).toHaveLength(result.requests[seat]);
        // The adapter had work to do: several frames, and server-only lines
        // (|j|, |inactive|, |title|, |t:|, ...) that it left out of the channel.
        expect(result.frames[seat]).toBeGreaterThan(10);
        expect(result.framedLines[seat]).toBeGreaterThan(result.channelLines[seat] + result.resent[seat]);
        const raw = fixture.frames[seat].flat().join("\n");
        for (const kind of ["|j|", "|inactive|", "|t:|0", "|title|"]) expect(raw).toContain(kind);
        expect(result.channelLines[seat]).toBe(channels.reduce((sum, c) => sum + c.lines.length, 0));
      }
    });

    it("still agrees when the server re-sends a request and sends server lines between frames", async () => {
      const fixture = clone(path);
      for (const seat of SEATS) {
        resendAfter(fixture, seat, 0);
        const frames = fixture.frames[seat][0] as string[];
        frames.splice(2, 0, `>${fixture.room}\n|raw|<div class="infobox">synthetic</div>`);
      }
      const result = await compareA1A2(engine, snapshot, fixture);
      expect(result.differences).toEqual([]);
      for (const seat of SEATS) expect(result.resent[seat]).toBe(1);
    });

    describe("negative controls", () => {
      it("rejects a frame whose line differs", async () => {
        const mutated = clone(path);
        const at = (mutated.channels.p1[0] as { lines: string[] }).lines.findIndex((l) => l.startsWith("|move|"));
        expect(at).toBeGreaterThan(0);
        editLine(mutated, "p1", (l) => l.startsWith("|move|"), (l) => `${l}x`);
        const { differences } = await compareA1A2(engine, snapshot, mutated);
        expect(noticed(mutated, "p1", at, differences)).toBe(true);
      });

      it("rejects a frame with one line dropped", async () => {
        const mutated = clone(path);
        const at = (mutated.channels.p2[0] as { lines: string[] }).lines.findIndex((l) => l.startsWith("|-damage|"));
        expect(at).toBeGreaterThan(0);
        editLine(mutated, "p2", (l) => l.startsWith("|-damage|"), () => "");
        const { differences } = await compareA1A2(engine, snapshot, mutated);
        expect(noticed(mutated, "p2", at, differences)).toBe(true);
      });

      it("rejects two frames swapped", async () => {
        const mutated = clone(path);
        const frames = mutated.frames.p1[0] as string[];
        // A frame of battle lines and the request frame after it, in the other order.
        const at = frames.findIndex((frame, i) => frame.includes("\n|move|") && (frames[i + 1] ?? "").includes("|request|"));
        expect(at).toBeGreaterThan(0);
        [frames[at], frames[at + 1]] = [frames[at + 1] as string, frames[at] as string];
        const { differences } = await compareA1A2(engine, snapshot, mutated);
        expect(differences.length).toBeGreaterThan(0);
      });

      it("rejects one differing request body (hash)", async () => {
        const mutated = clone(path);
        // The last move request of the first connection (the whole battle on a plain fixture).
        editLine(
          mutated,
          "p2",
          (l) => l.startsWith("|request|") && !l.includes('"wait"') && !l.includes("forceSwitch"),
          (line) => {
            const request = JSON.parse(line.slice(9)) as { side: { pokemon: { condition: string }[] } };
            (request.side.pokemon[0] as { condition: string }).condition = "1/1";
            return `|request|${JSON.stringify(request)}`;
          },
          true,
        );
        const { differences } = await compareA1A2(engine, snapshot, mutated);
        expect(differences.some((d) => d.includes("request_hash"))).toBe(true);
        expect(differences.some((d) => d.includes("observations differ"))).toBe(true);
      });

      it("fails when A2Channel does not drop re-sent requests", async () => {
        const fixture = clone(path);
        // Sanity: the real adapter passes the very same frames.
        expect((await compareA1A2(engine, snapshot, fixture)).differences).toEqual([]);
        // The broken adapter makes each request unique, so none is a re-send: with no re-send
        // in the frames it still agrees, which is why the next frames carry one.
        for (const seat of SEATS) resendAfter(fixture, seat, 0);
        expect((await compareA1A2(engine, snapshot, fixture)).differences).toEqual([]);
        const broken = await compareA1A2(engine, snapshot, fixture, noDedupe);
        expect(broken.differences.length).toBeGreaterThan(0);
        // The extra request line is one more channel line than A1's channel has.
        expect(stripBypassNoticed(broken.differences)).toBe(true);
        // Either the channel has the extra line, or (in a seat that reconnected) the join refuses the repeated rqid.
        expect(broken.differences.some((d) => d.includes("channel line") || d.includes("refused the frames"))).toBe(true);
      });

      it("fails when the stripServerLines call is bypassed", async () => {
        const fixture = clone(path);
        const real = await compareA1A2(engine, snapshot, fixture);
        expect(real.differences).toEqual([]);
        const broken = await compareA1A2(engine, snapshot, fixture, noStrip);
        expect(broken.differences.length).toBeGreaterThan(0);
        // The unfiltered channel is not even a valid one, or has lines A1 never wrote.
        expect(stripBypassNoticed(broken.differences)).toBe(true);
      });

      it("rejects one differing decision in the input log", async () => {
        const mutated = clone(path);
        const lines = mutated.input_log.lines;
        const at = lines.indexOf(">p1 team 1, 2, 3, 4");
        expect(at).toBeGreaterThan(0);
        lines[at] = ">p1 team 2, 1, 3, 4";
        const { differences } = await compareA1A2(engine, snapshot, mutated);
        expect(differences.length).toBeGreaterThan(0);
        expect(differences.some((d) => d.includes("channel line") || d.includes("A1 could not replay"))).toBe(true);
      });

      it("rejects an input log whose >player line differs from the players A1 played", async () => {
        const mutated = clone(path);
        mutated.input_log.lines[1] = (mutated.input_log.lines[1] as string).replace('"name":"', '"name":"x');
        const { differences } = await compareA1A2(engine, snapshot, mutated);
        expect(differences).toContain("A1 was not started with the input log's >start and >player lines");
      });

      it("rejects an input log whose >start format differs", async () => {
        const mutated = clone(path);
        mutated.input_log.lines[0] = (mutated.input_log.lines[0] as string).replace('"formatid":"', '"formatid":"x');
        const { differences } = await compareA1A2(engine, snapshot, mutated);
        expect(differences).toContain("A1 was not started with the input log's >start and >player lines");
      });

      it("rejects a dropped decision", async () => {
        const mutated = clone(path);
        const lines = mutated.input_log.lines;
        lines.splice(lines.lastIndexOf(lines.filter((l) => l.startsWith(">p1 "))[1] as string), 1);
        const { differences } = await compareA1A2(engine, snapshot, mutated);
        expect(differences.length).toBeGreaterThan(0);
        expect(differences.some((d) => d.includes("choices in the input log for") || d.includes("A1 could not replay"))).toBe(true);
      });
    });
  });
});

describe.each(reconnecting)("a real reconnect: %s", (path) => {
  const seatOf = (fixture: HarnessFixture): Seat => reconnected(fixture)[0] as Seat;

  it("joins the rejoin channel to the connection before it: A1's one continuous channel, built into full observations", async () => {
    const fixture = readFixture(path);
    const seat = seatOf(fixture);
    const [before, after] = fixture.channels[seat] as [HarnessFixture["channels"]["p1"][0], HarnessFixture["channels"]["p1"][0]];
    const result = await compareA1A2(engine, snapshot, fixture);
    expect(result.differences).toEqual([]);
    const requestLines = (lines: readonly string[]) => lines.filter((l) => l.startsWith("|request|"));
    // The replayed log has no |request| line: the rejoin channel alone has no Team Preview request ...
    expect(after.lines.filter((l) => !l.startsWith("|request|")).length).toBeGreaterThan(before.lines.filter((l) => !l.startsWith("|request|")).length);
    expect(requestLines(after.lines).some((l) => JSON.parse(l.slice(9)).teamPreview === true)).toBe(false);
    const lastBefore = requestLines(before.lines).at(-1);
    expect(requestLines(after.lines)[0]).toBe(lastBefore);
    // ... and the joined channel has every request once, the re-sent one not twice.
    const joined = result.continued[seat].at(-1) as PlayerChannel;
    expect(requestLines(joined.lines).length).toBe(requestLines(before.lines).length + requestLines(after.lines).length - 1);
    expect(joined.lines.filter((l) => !l.startsWith("|request|"))).toEqual(after.lines.filter((l) => !l.startsWith("|request|")));
    // Without the join, no observation can be built from the rejoin channel.
    const raw = playerChannel(after);
    const team = { schema_version: 1 as const, data_class: "synthetic" as const, team: [] as never };
    await expect(observe({ channel: raw, team, manifest: fixture.manifest as never })).rejects.toThrow(/continueChannel/);
  });

  it("fails for an adapter that does not restart at the rejoin", async () => {
    const fixture = readFixture(path);
    const seat = seatOf(fixture);
    const { differences } = await compareA1A2(engine, snapshot, fixture, brokenRejoin());
    expect(differences.some((d) => d.startsWith(`${seat} connection 2 channel line`) || d.startsWith(`${seat}: the A2 adapter refused the frames`))).toBe(true);
  });

  it("fails for a no-dedupe adapter when the server also re-sends the request within the rejoin connection", async () => {
    const fixture = clone(path);
    const seat = seatOf(fixture);
    resendAfter(fixture, seat, 1);
    const real = await compareA1A2(engine, snapshot, fixture);
    expect(real.differences).toEqual([]);
    expect(real.resent[seat]).toBe(1);
    const broken = await compareA1A2(engine, snapshot, fixture, noDedupe);
    expect(
      broken.differences.some((d) => d.startsWith(`${seat} connection 2 channel line`) || d.startsWith(`${seat}: the A2 adapter refused the frames`)),
    ).toBe(true);
  });

  it("rejects a rejoin whose replayed log lost a line, and one whose re-sent request is another", async () => {
    const lost = clone(path);
    const seat = seatOf(lost);
    editLine(lost, seat, (l) => l.startsWith("|-damage|"), () => "", false, 1);
    const first = await compareA1A2(engine, snapshot, lost);
    expect(first.differences.some((d) => d.startsWith(`${seat}: the A2 adapter refused the frames`))).toBe(true);

    const other = clone(path);
    editLine(
      other,
      seat,
      (l) => l.startsWith("|request|"),
      (line) => {
        const request = JSON.parse(line.slice(9)) as { side: { pokemon: { condition: string }[] } };
        (request.side.pokemon[0] as { condition: string }).condition = "1/1";
        return `|request|${JSON.stringify(request)}`;
      },
      false,
      1,
    );
    // Not the same request any more (the hash differs), so it is a new one with the earlier request's rqid:
    // the join refuses it (rqid must rise), so the adapter refuses the frames.
    const second = await compareA1A2(engine, snapshot, other);
    expect(second.differences.some((d) => d.startsWith(`${seat}: the A2 adapter refused the frames`))).toBe(true);
  });

  it("rejects a second connection that does not start at an |init|battle", async () => {
    const fixture = clone(path);
    const seat = seatOf(fixture);
    const frames = fixture.frames[seat][1] as string[];
    frames[0] = (frames[0] as string).replace("\n|init|battle", "");
    const { differences } = await compareA1A2(engine, snapshot, fixture);
    expect(differences.some((d) => d.startsWith(`${seat}: the A2 adapter refused the frames`))).toBe(true);
  });

  it("rejects a rejoin that goes back to an older request, or reuses an rqid", async () => {
    const fixture = clone(path);
    const seat = seatOf(fixture);
    // The untouched copy is accepted, so the refusal below is the change's doing.
    await expect(verifyHarnessFixture(engine, clone(path))).resolves.toBeUndefined();
    editLine(fixture, seat, (l) => l.startsWith("|request|"), (l) => l.replace(/"rqid":\d+/, '"rqid":1'), false, 1);
    const { differences } = await compareA1A2(engine, snapshot, fixture);
    expect(differences.some((d) => d.includes("does not follow") || d.includes("refused the frames"))).toBe(true);
    await expect(verifyHarnessFixture(engine, fixture)).rejects.toThrow(/connection 2 starts with a request older than/);
  });

  it("rejects a |sentchoice| that was removed, one that was added, and one whose choice is not the one played", async () => {
    const pristine = readFixture(path);
    const seat = seatOf(pristine);
    const evidence = reconnectEvidence(pristine.frames[seat]);
    expect(evidence).toBeDefined();
    const edit = (change: (line: string) => string | undefined) => {
      const fixture = clone(path);
      const frames = fixture.frames[seat][1] as string[];
      const i = frames.findIndex((frame) => frame.includes("|request|"));
      const lines = (frames[i] as string).split("\n").flatMap((line) => (line.startsWith("|sentchoice|") ? [change(line)] : [line]));
      frames[i] = lines.filter((line) => line !== undefined).join("\n");
      return fixture;
    };
    if (evidence?.sentchoice === true) {
      const removed = await compareA1A2(engine, snapshot, edit(() => undefined));
      expect(removed.differences).toEqual([`${seat}: |sentchoice| is not there, the provenance says the cut came right after it sent its choice for`]);
      // Another legal choice: A1 plays it in place of the logged one and the battle goes another way.
      const changed = await compareA1A2(engine, snapshot, edit(() => "|sentchoice|move 2 2, move 2 1"));
      expect(changed.differences.length).toBeGreaterThan(0);
      expect(changed.differences.some((d) => d.includes("channel line") || d.includes("A1 could not replay"))).toBe(true);
    } else {
      const added = clone(path);
      const frames = added.frames[seat][1] as string[];
      const i = frames.findIndex((frame) => frame.includes("|request|"));
      frames[i] = `${frames[i]}\n|sentchoice|default`;
      const { differences } = await compareA1A2(engine, snapshot, added);
      expect(differences).toEqual([`${seat}: |sentchoice| is there, the provenance says the cut came when it received`]);
    }
  });
});
