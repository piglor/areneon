import { A2Channel, type A2ChannelOptions, splitFrame } from "@areneon/bot-client";
import { loadEngine, loadRulesSnapshot, playerChannel } from "@areneon/showdown-bridge";
import { channelFromFrames, type HarnessFixture, verifyHarnessFixture } from "@areneon/local-server/harness";
import { describe, expect, it } from "vitest";
import { type Adapter, compareA1A2, fixtureFiles, readFixture, SEATS } from "./a1-a2-equality.ts";

const engine = loadEngine();
const snapshot = loadRulesSnapshot(engine.commit);
const files = fixtureFiles();
const clone = (path: string) => structuredClone(readFixture(path));

type Seat = (typeof SEATS)[number];

/** Rewrites the first (or last) line of a seat's frames that `pick` selects; returns its place. */
function editLine(
  fixture: HarnessFixture,
  seat: Seat,
  pick: (line: string) => boolean,
  edit: (line: string) => string,
  last = false,
): { frame: number; line: number } {
  const frames = fixture.frames[seat];
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

const stripBypassNoticed = (differences: readonly string[]): boolean =>
  differences.some((d) => d.includes("channel line") || d.includes("refused"));

describe("A1/A2 equality (ADR-005 gate 3)", () => {
  it("finds the committed fixtures", () => {
    expect(files.length).toBeGreaterThanOrEqual(2);
  });

  describe.each(files)("%s", (path) => {
    it("is a valid harness fixture", async () => {
      await verifyHarnessFixture(engine, readFixture(path));
    });

    it("starts from raw server frames: the stored channels are what the adapter makes of them", () => {
      const fixture = readFixture(path);
      for (const seat of SEATS) {
        const { channel, requests } = channelFromFrames(fixture.frames[seat], fixture.room, seat);
        expect(channel).toEqual(fixture.channels[seat]);
        expect(requests).toBeGreaterThan(0);
      }
    });

    it("plays on A1 to the same per-player channels, request hashes and observations as A2", async () => {
      const fixture = readFixture(path);
      const result = await compareA1A2(engine, snapshot, fixture);
      expect(result.differences).toEqual([]);
      for (const seat of SEATS) {
        // Not vacuous: something was compared, one observation per non-wait request.
        expect(result.requests[seat]).toBeGreaterThan(0);
        expect(result.decisions[seat]).toBeGreaterThan(0);
        expect(result.decisions[seat]).toBeLessThanOrEqual(result.requests[seat]);
        const requests = fixture.channels[seat].lines.filter((l) => l.startsWith("|request|"));
        const waits = requests.filter((l) => (JSON.parse(l.slice(9)) as { wait?: true }).wait === true);
        expect(result.requests[seat]).toBe(requests.length);
        expect(result.decisions[seat]).toBe(requests.length - waits.length);
        // The adapter had work to do: several frames, and server-only lines
        // (|j|, |inactive|, |title|, |t:|, ...) that it left out of the channel.
        expect(result.frames[seat]).toBeGreaterThan(10);
        expect(result.framedLines[seat]).toBeGreaterThan(result.channelLines[seat] + result.resent[seat]);
        const raw = fixture.frames[seat].join("\n");
        for (const kind of ["|j|", "|inactive|", "|t:|0", "|title|"]) expect(raw).toContain(kind);
        expect(result.channelLines[seat]).toBe(fixture.channels[seat].lines.length);
      }
    });

    it("still agrees when the server re-sends a request and sends server lines between frames", async () => {
      const fixture = clone(path);
      for (const seat of SEATS) {
        const frames = fixture.frames[seat];
        const at = frames.findIndex((frame) => frame.includes("|request|") && !frame.includes('"wait":true'));
        const request = (frames[at] as string).split("\n").find((l) => l.startsWith("|request|")) as string;
        // The server's onConnect re-send: the same request in a frame of its own, plus a raw box.
        frames.splice(at + 1, 0, `>${fixture.room}\n${request}`, `>${fixture.room}\n|raw|<div class="infobox">synthetic</div>`);
      }
      const result = await compareA1A2(engine, snapshot, fixture);
      expect(result.differences).toEqual([]);
      for (const seat of SEATS) expect(result.resent[seat]).toBe(1);
    });

    describe("negative controls", () => {
      it("rejects a frame whose line differs", async () => {
        const mutated = clone(path);
        const at = mutated.channels.p1.lines.findIndex((l) => l.startsWith("|move|"));
        expect(at).toBeGreaterThan(0);
        editLine(mutated, "p1", (l) => l.startsWith("|move|"), (l) => `${l}x`);
        const { differences } = await compareA1A2(engine, snapshot, mutated);
        expect(differences.some((d) => d.startsWith(`p1 channel line ${at + 1} differs`))).toBe(true);
      });

      it("rejects a frame with one line dropped", async () => {
        const mutated = clone(path);
        const at = mutated.channels.p2.lines.findIndex((l) => l.startsWith("|damage|") || l.startsWith("|-damage|"));
        expect(at).toBeGreaterThan(0);
        editLine(mutated, "p2", (l) => l.startsWith("|-damage|"), () => "");
        const { differences } = await compareA1A2(engine, snapshot, mutated);
        expect(differences.some((d) => d.startsWith(`p2 channel line ${at + 1} differs`))).toBe(true);
      });

      it("rejects two frames swapped", async () => {
        const mutated = clone(path);
        const frames = mutated.frames.p1;
        // A frame of battle lines and the request frame after it, in the other order.
        const at = frames.findIndex((frame, i) => frame.includes("\n|move|") && (frames[i + 1] ?? "").includes("|request|"));
        expect(at).toBeGreaterThan(0);
        [frames[at], frames[at + 1]] = [frames[at + 1] as string, frames[at] as string];
        const { differences } = await compareA1A2(engine, snapshot, mutated);
        expect(differences.length).toBeGreaterThan(0);
      });

      it("rejects one differing request body (hash)", async () => {
        const mutated = clone(path);
        editLine(
          mutated,
          "p2",
          (l) => l.startsWith("|request|") && !l.includes('"wait"'),
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
        for (const seat of SEATS) {
          const frames = fixture.frames[seat];
          const at = frames.findIndex((frame) => frame.includes("|request|") && !frame.includes('"wait":true'));
          const request = (frames[at] as string).split("\n").find((l) => l.startsWith("|request|")) as string;
          frames.splice(at + 1, 0, `>${fixture.room}\n${request}`);
        }
        expect((await compareA1A2(engine, snapshot, fixture)).differences).toEqual([]);
        const broken = await compareA1A2(engine, snapshot, fixture, noDedupe);
        expect(broken.differences.length).toBeGreaterThan(0);
        // The extra request counts as one more decision than the input log answered.
        expect(broken.differences.some((d) => d.includes("decisions in the channel"))).toBe(true);
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
