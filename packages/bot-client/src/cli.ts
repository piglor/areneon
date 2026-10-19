import { parseArgs } from "node:util";
import type { LocalServer, LaunchOptions, RuntimeCopy, RuntimeCopyOptions } from "@areneon/local-server";
import { type Engine, importPaste, type RulesSnapshot } from "@areneon/showdown-bridge";
import type { A2BattleRecord, A2BotOptions, A2SessionResult, SessionLog } from "./session.ts";

/** What the CLI runs on; `scripts/a2-bot.ts` wires the real ones. */
export interface CliDeps {
  engine: Engine;
  snapshot: RulesSnapshot;
  readFile(path: string): string;
  createRuntimeCopy(options: RuntimeCopyOptions): RuntimeCopy;
  launch(options: LaunchOptions): Promise<LocalServer>;
  runBot(options: A2BotOptions): Promise<A2SessionResult>;
  /** Registers `handler` for SIGINT and SIGTERM; returns the unregister function. */
  onSignal(handler: (signal: string) => void): () => void;
  /** Stores one battle record (the bot's own channel and decisions) under `dir`. */
  writeRecord(dir: string, record: A2BattleRecord): void;
  /** Where records go without `--out`: gitignored `private-data/`. */
  defaultOutDir: string;
  stdout(line: string): void;
  stderr(line: string): void;
}

export const USAGE = `usage: a2-bot --challenger NAME --team PASTE [--data-class private|synthetic]
              [--format ID] [--port N] [--runtime-root DIR] [--out DIR]

Starts the pinned Showdown server on 127.0.0.1 and the A2 sparring bot on it.
The bot accepts a Bo1 challenge from NAME only. Ctrl-C stops the bot and the server.`;

const DATA_CLASSES: readonly string[] = ["private", "synthetic"];

/** JSON for the operator's log: an Error is written as its message. */
const json = (value: unknown): string =>
  JSON.stringify(value, (_key, v: unknown) => (v instanceof Error ? `${v.name}: ${v.message}` : v));

/**
 * The A2 sparring session (ADR-005 §4): launch the pinned server from a
 * runtime copy, run `runA2Bot` on it until a signal, then stop the server.
 * SIGINT and SIGTERM stop the bot and then the server, so Ctrl-C leaves no
 * server running (the launcher's own exit hook fires only on a normal exit).
 * Exit codes: 0 stopped, 1 failed, 2 usage.
 */
export async function runCli(argv: string[], deps: CliDeps): Promise<number> {
  let values;
  try {
    ({ values } = parseArgs({
      args: argv,
      options: {
        challenger: { type: "string" },
        team: { type: "string" },
        "data-class": { type: "string", default: "private" },
        format: { type: "string" },
        port: { type: "string", default: "8000" },
        "runtime-root": { type: "string" },
        out: { type: "string" },
      },
      strict: true,
    }));
  } catch (error) {
    deps.stderr(`${(error as Error).message}\n${USAGE}`);
    return 2;
  }
  const { challenger, team: teamPath, "data-class": dataClass, port: portText } = values;
  const formats = Object.keys(deps.snapshot.formats);
  const format = values.format ?? (formats.length === 1 ? formats[0] : undefined);
  const port = Number(portText);
  const problem =
    challenger === undefined || teamPath === undefined
      ? "--challenger and --team are required"
      : !DATA_CLASSES.includes(dataClass)
        ? `--data-class must be private or synthetic, not ${dataClass}`
        : !Number.isInteger(port) || port < 1 || port > 65535
          ? `--port must be an integer from 1 to 65535, not ${portText}`
          : format === undefined
            ? `--format is required: the rules snapshot has ${formats.length} formats`
            : undefined;
  if (problem !== undefined) {
    deps.stderr(`${problem}\n${USAGE}`);
    return 2;
  }

  const controller = new AbortController();
  const off = deps.onSignal((signal) => {
    deps.stderr(`${signal}: stopping the bot and the server`);
    controller.abort();
  });
  let server: LocalServer | undefined;
  try {
    const team = importPaste(deps.engine, deps.readFile(teamPath as string));
    const root = values["runtime-root"];
    const copy = deps.createRuntimeCopy(root === undefined ? {} : { runtimeRoot: root });
    server = await deps.launch({ dir: copy.dir, port });
    const name = `Areneon-bot-${server.session.bot_name_suffix}`;
    const url = `ws://127.0.0.1:${port}/showdown/websocket`;
    const out = values.out ?? deps.defaultOutDir;
    deps.stdout(
      json({ event: "ready", client: server.url, url, bot: name, challenger, format, server_pid: server.session.pid }),
    );
    const result = await deps.runBot({
      engine: deps.engine,
      snapshot: deps.snapshot,
      url,
      name,
      challenger: challenger as string,
      format: format as string,
      team,
      data_class: dataClass as A2BotOptions["data_class"],
      mode: "live_match",
      signal: controller.signal,
      onLog: (entry: SessionLog) => deps.stdout(json(entry)),
      onBattle: (record) => deps.writeRecord(out, record),
    });
    deps.stdout(json({ event: "stopped", battles: result.battles.length }));
    return 0;
  } catch (error) {
    deps.stderr(`a2-bot failed: ${json(error instanceof Error ? error : String(error))}`);
    return 1;
  } finally {
    off();
    await server?.stop();
  }
}
