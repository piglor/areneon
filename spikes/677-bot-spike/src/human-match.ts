// SPIKE (#677): throwaway. Starts the local server and one heuristic bot
// (an integration fixture, not AI) that waits for a human's private
// challenge in [Gen 9 Champions] VGC 2026 Reg M-C, plays one Bo1 with OTS
// declined, writes the timing evidence and stops the server.
//
//   node spikes/677-bot-spike/src/human-match.ts
//
// The human's team is theirs: the bot only ever sees what the battle
// reveals. That revealed information is kept out of Git: the player-view log
// goes to the gitignored private-data/ directory.
import { join } from "node:path";
import { type DecisionRecord, runBot } from "./bot.ts";
import { writeEvidence } from "./evidence.ts";
import { DEFAULT_PORT, HOST, startServer, stopServer } from "./server.ts";
import { syntheticTeam } from "./team.ts";

const BOT_NAME = "AreneonBot";
const spikeDir = join(import.meta.dirname, "..");
const decisions: DecisionRecord[] = [];
const playerView: string[] = [];

const server = await startServer(DEFAULT_PORT);
for (const signal of ["SIGINT", "SIGTERM"] as const) {
  process.on(signal, () => void stopServer(server).then(() => process.exit(130)));
}
try {
  process.stderr.write(
    `\nReady. Open http://localhost:${DEFAULT_PORT} in a browser, pick a name, then\n` +
      `challenge ${BOT_NAME} to [Gen 9 Champions] VGC 2026 Reg M-C.\n\n`,
  );
  const started = performance.now();
  const result = await runBot({
    url: `ws://${HOST}:${DEFAULT_PORT}/showdown/websocket`,
    name: BOT_NAME,
    team: syntheticTeam(),
    onDecision: (d) => {
      decisions.push(d);
      process.stderr.write(`${JSON.stringify(d)}\n`);
    },
    onLine: (_bot, room, line) => {
      if (room.startsWith("battle-")) playerView.push(line);
    },
  });
  const wall = (performance.now() - started) / 1000;
  const summary = writeEvidence(join(spikeDir, "evidence"), "human-vs-bot", result, decisions, [], wall);
  writeEvidence(
    join(spikeDir, "../../private-data/677-human-match"),
    "human-vs-bot",
    result,
    decisions,
    playerView,
    wall,
  );
  process.stdout.write(JSON.stringify(summary, null, 2) + "\n");
} finally {
  await stopServer(server);
}
