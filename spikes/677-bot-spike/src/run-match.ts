// SPIKE (#677): throwaway. Proves runtime A2 end to end without a human:
// starts the pinned server on localhost, connects two heuristic bot clients,
// one challenges the other to a private M-C Bo1, both decline OTS, the VGC
// Timer is on, and every decision is timed against it.
//
//   node spikes/677-bot-spike/src/run-match.ts [outDir]
import { join } from "node:path";
import { type DecisionRecord, runBot } from "./bot.ts";
import { writeEvidence } from "./evidence.ts";
import { DEFAULT_PORT, HOST, startServer, stopServer } from "./server.ts";
import { syntheticTeam } from "./team.ts";

const outDir = process.argv[2] ?? join(import.meta.dirname, "..", "evidence");
const url = `ws://${HOST}:${DEFAULT_PORT}/showdown/websocket`;
const decisions: DecisionRecord[] = [];
const playerView: string[] = [];

const server = await startServer(DEFAULT_PORT);
try {
  const started = performance.now();
  const host = runBot({
    url,
    name: "AreneonBotA",
    team: syntheticTeam(),
    onDecision: (d) => decisions.push(d),
    // Only bot A's own player view is kept: player-perspective, never omniscient.
    onLine: (_bot, room, line) => {
      if (room.startsWith("battle-")) playerView.push(line);
    },
  });
  await new Promise((r) => setTimeout(r, 500)); // let A log in first
  const guest = runBot({
    url,
    name: "AreneonBotB",
    team: syntheticTeam(),
    challenge: "AreneonBotA",
    onDecision: (d) => decisions.push(d),
  });
  const [a, b] = await Promise.all([host, guest]);
  const result = { ...a, timerOn: a.timerOn && b.timerOn };
  const wall = (performance.now() - started) / 1000;
  const summary = writeEvidence(outDir, "bot-vs-bot", result, decisions, playerView, wall);
  process.stdout.write(JSON.stringify(summary, null, 2) + "\n");
} finally {
  await stopServer(server);
}
