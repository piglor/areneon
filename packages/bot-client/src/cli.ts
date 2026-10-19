import { isAbsolute, join, relative, resolve } from "node:path";
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
  /** Creates `dir` (and parents) with the permission bits `mode`. */
  makeDir(dir: string, mode: number): void;
  /** Writes `data` to the new file `path` with the permission bits `mode`. */
  writeFile(path: string, data: string, mode: number): void;
  /** The checkout: an `--out` inside it is refused unless the data class is synthetic. */
  repoRoot: string;
  /** The gitignored `private-data/` of the checkout: records of any data class may go here. */
  privateDataDir: string;
  /** Where records go without `--out`: a directory of `privateDataDir`. */
  defaultOutDir: string;
  /** Milliseconds since the epoch, for record file names. */
  now(): number;
  stdout(line: string): void;
  stderr(line: string): void;
}

export const USAGE = `usage: a2-bot --challenger NAME --team PASTE [--data-class private|synthetic]
              [--format ID] [--port N] [--runtime-root DIR] [--out DIR]

Starts the pinned Showdown server on 127.0.0.1 and the A2 sparring bot on it.
The bot accepts a Bo1 challenge from NAME only. Ctrl-C stops the bot and the server.

Each battle, finished or not, is stored as one JSON record (the bot's own
channel and decisions) in --out, default private-data/a2 (mode 0700, files
0600). An --out inside the checkout is refused unless --data-class is
synthetic. A record of a battle that ended in an Open Team Sheets failure
(ots_failure), a forfeited decline, or a fatal error may hold the human's
accepted team sheet, so it is only ever written under private-data/, whatever
--out says.`;

const DATA_CLASSES: readonly string[] = ["private", "synthetic"];
/** A Showdown battle room id; the only shape of `record.room` that may go into a file name. */
const BATTLE_ROOM = /^battle-[a-z0-9-]+$/;

const DIR_MODE = 0o700;
const FILE_MODE = 0o600;

/** Whether `path` is `dir` or below it (both resolved, no symlinks followed). */
const within = (dir: string, path: string): boolean => {
  const from = relative(resolve(dir), resolve(path));
  return from === "" || (!from.startsWith("..") && !isAbsolute(from));
};

/** JSON for the operator's log: an Error is written as its message. */
const json = (value: unknown): string =>
  JSON.stringify(value, (_key, v: unknown) => (v instanceof Error ? `${v.name}: ${v.message}` : v));

/**
 * What the operator's log shows of an entry. A dropped line is raw server
 * text, so only its kind (a leading `|` and the text up to the next `|`) is
 * shown; a line without a leading `|` shows nothing of itself.
 */
const operatorLog = (entry: SessionLog): unknown =>
  entry.event === "dropped" ? { event: "dropped", room: entry.room, kind: /^\|[^|]*/.exec(entry.line)?.[0] ?? "|?" } : entry;

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
  const out = values.out ?? deps.defaultOutDir;
  const problem =
    challenger === undefined || teamPath === undefined
      ? "--challenger and --team are required"
      : !DATA_CLASSES.includes(dataClass)
        ? `--data-class must be private or synthetic, not ${dataClass}`
        : !Number.isInteger(port) || port < 1 || port > 65535
          ? `--port must be an integer from 1 to 65535, not ${portText}`
          : format === undefined
            ? `--format is required: the rules snapshot has ${formats.length} formats`
            : within(deps.repoRoot, out) && !within(deps.privateDataDir, out) && dataClass !== "synthetic"
              ? `--out ${out} is inside the checkout: records of a ${dataClass} team go under private-data/ only`
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
  const stored = { rooms: new Set<string>(), failed: false };
  /**
   * Stores one record, once per room. The room id is checked before it goes
   * into a path. A record with an `ots_failure`, an `ots` of `forfeited` or a `fatal` may hold the human's accepted
   * team sheet (a `|showteam|` line that came before the failure was
   * noticed), so it is written under `private-data/` only, whatever `--out`
   * says. A record that cannot be stored is reported and fails the run.
   */
  const store = (record: A2BattleRecord): void => {
    if (stored.rooms.has(record.room)) return;
    stored.rooms.add(record.room);
    try {
      if (!BATTLE_ROOM.test(record.room)) throw new RangeError(`not a battle room id: ${JSON.stringify(record.room)}`);
      let dir = out;
      // A forfeited decline or a fatal end can leave `ots_failure` null (the `/forfeit` could
      // not be sent, or the run ended first) while a `|showteam|` line is already in the
      // channel, so those route like an `ots_failure`. The line itself is not removed from the
      // record: the channel is kept as received, and the routing rule is the guard.
      const sheet = record.ots_failure !== null || record.ots === "forfeited" || record.fatal !== null;
      if (sheet && !within(deps.privateDataDir, out)) {
        dir = deps.defaultOutDir;
        deps.stderr(`${record.room}: record goes to ${dir}, not ${out}: it may hold the opponent's team sheet`);
      }
      deps.makeDir(dir, DIR_MODE);
      deps.writeFile(join(dir, `${record.room}-${deps.now()}.json`), `${JSON.stringify(record, null, 2)}\n`, FILE_MODE);
    } catch (error) {
      stored.failed = true;
      deps.stderr(`record ${record.room} not stored: ${json(error instanceof Error ? error : String(error))}`);
    }
  };
  let server: LocalServer | undefined;
  try {
    const team = importPaste(deps.engine, deps.readFile(teamPath as string));
    const root = values["runtime-root"];
    const copy = deps.createRuntimeCopy(root === undefined ? {} : { runtimeRoot: root });
    server = await deps.launch({ dir: copy.dir, port });
    const name = `Areneon-bot-${server.session.bot_name_suffix}`;
    const url = `ws://127.0.0.1:${port}/showdown/websocket`;
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
      onLog: (entry: SessionLog) => deps.stdout(json(operatorLog(entry))),
      onBattle: (record) => store(record),
    });
    // Battles that did not end (a stop) or whose `onBattle` did not run are stored now.
    for (const record of result.battles) store(record);
    deps.stdout(json({ event: "stopped", battles: result.battles.length }));
    return stored.failed ? 1 : 0;
  } catch (error) {
    deps.stderr(`a2-bot failed: ${json(error instanceof Error ? error : String(error))}`);
    return 1;
  } finally {
    off();
    await server?.stop();
  }
}
