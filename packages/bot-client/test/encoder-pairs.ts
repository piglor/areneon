// The A1 and A2 observations of the ADR-005 equality fixtures with real candidates, for the Python encoder test.
import { decodeRequest, type Engine, enumerateCandidates, type RulesSnapshot } from "@areneon/showdown-bridge";
import { compareA1A2, fixtureFiles, type ObservationPair, readFixture } from "./a1-a2-equality.ts";

// Observation logs of one decision are prefixes of a longer log: keep each stream once and cut it.
class Streams {
  readonly lists: string[][] = [];
  add(log: readonly string[]): { stream: number; length: number } {
    const at = this.lists.findIndex((list) => log.every((line, i) => list[i] === line) || list.every((line, i) => log[i] === line));
    if (at < 0) {
      this.lists.push([...log]);
      return { stream: this.lists.length - 1, length: log.length };
    }
    if (log.length > (this.lists[at] as string[]).length) this.lists[at] = [...log];
    return { stream: at, length: log.length };
  }
}
const identityOf = (o: ObservationPair["a1"]) => o.identity;

export async function encoderPairs(engine: Engine, snapshot: RulesSnapshot): Promise<Record<string, unknown>> {
  const byFormat: Record<string, { a1: Streams; a2: Streams; teams: unknown[]; decisions: unknown[] }> = {};
  for (const file of fixtureFiles()) {
    const [format, name] = file.split("/") as [string, string];
    const fixture = readFixture(file);
    const pairs: ObservationPair[] = [];
    const result = await compareA1A2(engine, snapshot, fixture, undefined, (pair) => pairs.push(pair));
    if (result.differences.length > 0) throw new Error(`${file}: ${result.differences.join("; ")}`);
    const target = (byFormat[format] ??= { a1: new Streams(), a2: new Streams(), teams: [], decisions: [] });
    for (const pair of pairs) {
      const line = pair.a1.log.at(-1) as string;
      const candidates = enumerateCandidates(decodeRequest(line.slice("|request|".length))).candidates;
      let team = target.teams.findIndex((t) => JSON.stringify(t) === JSON.stringify(pair.team));
      if (team < 0) team = target.teams.push(pair.team) - 1;
      target.decisions.push({
        fixture: name,
        seat: pair.seat,
        request_seq: pair.request_seq,
        team,
        kind: pair.a1.kind,
        a1: { ...target.a1.add(pair.a1.log), identity: identityOf(pair.a1) },
        a2: { ...target.a2.add(pair.a2.log), identity: identityOf(pair.a2), decision_context: (pair.a2 as { decision_context?: unknown }).decision_context ?? null },
        candidates: candidates.map((c) => [c.choice, c.pending]),
      });
    }
  }
  const documents: Record<string, unknown> = {};
  for (const [format, data] of Object.entries(byFormat)) {
    documents[format] = {
      schema_version: 1,
      data_class: "synthetic",
      provenance:
        "synthetic: the A1 replay (recordedChannel) and A2 (recorded server frames) observations of the ADR-005 equality fixtures with enumerateCandidates, captured by packages/bot-client/scripts/capture-encoder-pairs.ts. An observation's log is streams[stream][0:length].",
      streams: { a1: data.a1.lists, a2: data.a2.lists },
      teams: data.teams,
      decisions: data.decisions,
    };
  }
  return documents;
}
