// Re-run one played A1 battle on the pinned engine (ADR-017 section 4a, ticket #926):
//   node packages/showdown-bridge/scripts/resimulate.ts < REQUEST.json
// The request is one JSON object on stdin: {format_id, seed, max_turns, teams: {p1, p2}, choices}.
// It prints one JSON line: the result with both players' channels and the omniscient log (both
// sides' hidden information: local use only, never commit it). The caller decides whether the result
// matches what was played; this script writes no file. SHOWDOWN_DIR points at another built checkout.
import { loadEngine } from "../src/engine.ts";
import { readResimulationInput, resimulate } from "../src/resimulate.ts";

try {
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) chunks.push(chunk as Buffer);
  const input = readResimulationInput(JSON.parse(Buffer.concat(chunks).toString("utf8")));
  const result = await resimulate(loadEngine(process.env.SHOWDOWN_DIR || undefined), input);
  console.log(JSON.stringify(result));
} catch (error) {
  console.error(`resimulate: ${(error as Error).message}`);
  process.exitCode = 1;
}
