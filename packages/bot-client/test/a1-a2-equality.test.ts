import { loadEngine, loadRulesSnapshot } from "@areneon/showdown-bridge";
import { verifyHarnessFixture } from "@areneon/local-server/harness";
import { beforeAll, describe, expect, it } from "vitest";
import { compareA1A2, fixtureFiles, readFixture, SEATS } from "./a1-a2-equality.ts";

const engine = loadEngine();
const snapshot = loadRulesSnapshot(engine.commit);
const files = fixtureFiles();
const clone = (path: string) => structuredClone(readFixture(path));

describe("A1/A2 equality (ADR-005 gate 3)", () => {
  it("finds the committed fixtures", () => {
    expect(files.length).toBeGreaterThanOrEqual(2);
  });

  describe.each(files)("%s", (path) => {
    it("is a valid harness fixture", async () => {
      await verifyHarnessFixture(engine, readFixture(path));
    });

    it("plays on A1 to the same per-player channels, request hashes and observations as A2", async () => {
      const result = await compareA1A2(engine, snapshot, readFixture(path));
      expect(result.differences).toEqual([]);
      for (const seat of SEATS) {
        // Not vacuous: something was compared, one observation per non-wait request.
        expect(result.requests[seat]).toBeGreaterThan(0);
        expect(result.decisions[seat]).toBeGreaterThan(0);
        expect(result.decisions[seat]).toBeLessThanOrEqual(result.requests[seat]);
      }
      const fixture = readFixture(path);
      for (const seat of SEATS) {
        const requests = fixture.channels[seat].lines.filter((l) => l.startsWith("|request|"));
        const waits = requests.filter((l) => (JSON.parse(l.slice(9)) as { wait?: true }).wait === true);
        expect(result.requests[seat]).toBe(requests.length);
        expect(result.decisions[seat]).toBe(requests.length - waits.length);
      }
    });

    describe("negative controls", () => {
      let fixture: ReturnType<typeof clone>;
      beforeAll(() => {
        fixture = clone(path);
      });

      it("rejects one differing channel line", async () => {
        const mutated = structuredClone(fixture);
        const lines = mutated.channels.p1.lines;
        const at = lines.findIndex((l) => l.startsWith("|move|"));
        expect(at).toBeGreaterThan(0);
        lines[at] = `${lines[at]}x`;
        const { differences } = await compareA1A2(engine, snapshot, mutated);
        expect(differences.some((d) => d.startsWith(`p1 channel line ${at + 1} differs`))).toBe(true);
      });

      it("rejects one differing request body (hash)", async () => {
        const mutated = structuredClone(fixture);
        const lines = mutated.channels.p2.lines;
        const at = lines.findLastIndex((l) => l.startsWith("|request|") && !l.includes('"wait"'));
        const request = JSON.parse((lines[at] as string).slice(9)) as { side: { pokemon: { condition: string }[] } };
        (request.side.pokemon[0] as { condition: string }).condition = "1/1";
        lines[at] = `|request|${JSON.stringify(request)}`;
        const { differences } = await compareA1A2(engine, snapshot, mutated);
        expect(differences.some((d) => d.includes("request_hash"))).toBe(true);
        expect(differences.some((d) => d.includes("observations differ"))).toBe(true);
      });

      it("rejects one differing decision in the input log", async () => {
        const mutated = structuredClone(fixture);
        const lines = mutated.input_log.lines;
        const at = lines.indexOf(">p1 team 1, 2, 3, 4");
        expect(at).toBeGreaterThan(0);
        lines[at] = ">p1 team 2, 1, 3, 4";
        const { differences } = await compareA1A2(engine, snapshot, mutated);
        expect(differences.length).toBeGreaterThan(0);
        expect(differences.some((d) => d.includes("channel line") || d.includes("A1 could not replay"))).toBe(true);
      });

      it("rejects an input log whose >player line differs from the players A1 played", async () => {
        const mutated = structuredClone(fixture);
        mutated.input_log.lines[1] = (mutated.input_log.lines[1] as string).replace('"name":"', '"name":"x');
        const { differences } = await compareA1A2(engine, snapshot, mutated);
        expect(differences).toContain("A1 was not started with the input log's >start and >player lines");
      });

      it("rejects a dropped decision", async () => {
        const mutated = structuredClone(fixture);
        const lines = mutated.input_log.lines;
        lines.splice(lines.lastIndexOf(lines.filter((l) => l.startsWith(">p1 "))[1] as string), 1);
        const { differences } = await compareA1A2(engine, snapshot, mutated);
        expect(differences.length).toBeGreaterThan(0);
        expect(differences.some((d) => d.includes("choices in the input log for") || d.includes("A1 could not replay"))).toBe(true);
      });
    });
  });
});
