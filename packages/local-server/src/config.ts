import { createHash } from "node:crypto";
import { existsSync, mkdirSync, renameSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import { isDeepStrictEqual } from "node:util";
import { RUNTIME_MANIFEST } from "./runtime-copy.ts";

/** The only address the server binds to (ADR-005 §4). */
export const LOOPBACK_ADDRESS = "127.0.0.1";

/** Where the server reads its config, relative to its install root. */
export const SERVER_CONFIG_PATH = "config/config.js";

/**
 * Directories the server writes into at runtime. Upstream keeps them in git
 * (each holds a README), so they are not part of the runtime copy; without
 * them the server logs a crash for every file it tries to write.
 */
export const RUNTIME_STATE_DIRS = [
  "config/chat-plugins",
  "config/ladders",
  "logs/chat",
  "logs/modlog",
  "logs/repl",
  "logs/tickets",
] as const;

export interface ServerSetting {
  key: string;
  value: unknown;
  /** Why Areneon sets it; written above the setting in `config.js`. */
  why: string;
}

/**
 * Every setting Areneon's `config.js` overrides in the pinned
 * `config/config-example.js` (the server merges the two, `config.js` on
 * top). Anything not listed keeps the upstream default.
 */
export function serverSettings(port: number): ServerSetting[] {
  return [
    { key: "port", value: port, why: "The session's port; the launcher also passes it on the command line." },
    { key: "bindaddress", value: LOOPBACK_ADDRESS, why: "ADR-005 §4: loopback only (upstream default 0.0.0.0)." },
    { key: "backdoor", value: false, why: "ADR-005 §4: no admin access for upstream system operators." },
    { key: "consoleips", value: [], why: "ADR-005 §4: no dev console (>> and >>>) from any address." },
    { key: "repl", value: false, why: "ADR-005 §4: no REPL sockets under logs/repl/." },
    {
      key: "watchconfig",
      value: false,
      why: "The config is fixed for the session: an edit to this file is not reloaded, so the session manifest's config_sha256 stays true.",
    },
    { key: "forcetimer", value: true, why: "ADR-005 §3-4: the battle timer is always on." },
    {
      key: "noNetRequests",
      value: true,
      why:
        "No outbound HTTP(S): every request through Showdown's Net helper throws before connecting. " +
        "That covers the login server (replay upload `addreplay`, user-stats reports, CSS invalidation, " +
        "remote ladder), the Tor exit-list and season ladder fetches at startup, and chat-command lookups.",
    },
    { key: "autosavereplays", value: false, why: "No replay upload when a battle ends." },
    { key: "replaysdb", value: null, why: "No direct write to a replay database." },
    { key: "usepostgres", value: false, why: "No external Postgres (teams, modlog)." },
    { key: "remoteladder", value: false, why: "Ratings stay in the local ladder store, never the login server." },
    { key: "dnsbl", value: false, why: "No DNS blocklist queries about connecting addresses." },
    {
      key: "logchallenges",
      value: false,
      why: "ADR-005 §5: a challenge's input log holds both teams; it stays off for sparring.",
    },
    { key: "crashguardemail", value: null, why: "No crash e-mails." },
  ];
}

/** The `config/config.js` text for a session on `port`. */
export function renderServerConfig(port: number): string {
  const lines = [
    "'use strict';",
    "",
    "// Written by @areneon/local-server for one local session (ADR-005 §4). Do not edit:",
    "// the launcher rewrites it and checks the merged config before every start.",
    "",
  ];
  for (const { key, value, why } of serverSettings(port)) {
    lines.push(`// ${why}`, `exports.${key} = ${JSON.stringify(value)};`, "");
  }
  return lines.join("\n");
}

export class ServerConfigError extends Error {
  override name = "ServerConfigError";
}

export interface WrittenConfig {
  path: string;
  /** SHA-256 of the written `config.js` bytes. */
  sha256: string;
}

/**
 * Write `config/config.js` for `port` into the runtime copy at `dir`, and
 * create the directories the server writes into. Refuses any directory that
 * is not a runtime copy (it has no runtime manifest), so it never writes into
 * the `vendor/` checkout.
 */
export function writeServerConfig(dir: string, port: number): WrittenConfig {
  if (!existsSync(join(dir, RUNTIME_MANIFEST))) {
    throw new ServerConfigError(`${dir} is not a runtime copy (no ${RUNTIME_MANIFEST})`);
  }
  for (const state of RUNTIME_STATE_DIRS) mkdirSync(join(dir, state), { recursive: true });
  const text = renderServerConfig(port);
  const path = join(dir, SERVER_CONFIG_PATH);
  writeFileSync(`${path}.tmp`, text);
  renameSync(`${path}.tmp`, path);
  return { path, sha256: createHash("sha256").update(text).digest("hex") };
}

/**
 * The config the server at `dir` will run with: its `config-example.js`
 * defaults with `config.js` on top, merged as the pinned config loader does.
 */
export function loadServerConfig(dir: string): Record<string, unknown> {
  const require = createRequire(join(dir, "package.json"));
  const load = (relative: string): Record<string, unknown> => {
    const path = require.resolve(join(dir, relative));
    // Never cached, so the next check reads the file as it is then.
    try {
      return { ...(require(path) as Record<string, unknown>) };
    } finally {
      Reflect.deleteProperty(require.cache, path);
    }
  };
  return { ...load("config/config-example.js"), ...load(SERVER_CONFIG_PATH) };
}

/**
 * The replay-upload check (ADR-005 §4): refuse a config under which a local
 * battle could be published. Replays leave the server in two ways at the
 * pin: a direct write to `replaysdb`, or the login server's `addreplay`
 * action over Net, which `/savereplay` (the client's upload button) and
 * `autosavereplays` both trigger.
 */
export function checkNoReplayUpload(config: Record<string, unknown>): void {
  const problems: string[] = [];
  if (config["autosavereplays"]) problems.push("autosavereplays is on");
  if (config["replaysdb"]) problems.push("replaysdb is set");
  if (config["noNetRequests"] !== true) problems.push("noNetRequests is off, so /savereplay reaches the login server");
  if (problems.length > 0) {
    throw new ServerConfigError(`config would upload replays: ${problems.join("; ")}`);
  }
}

/**
 * Refuse a merged config that would upload replays or that differs from any
 * setting Areneon fixes for `port`.
 */
export function checkServerConfig(config: Record<string, unknown>, port: number): void {
  checkNoReplayUpload(config);
  const wrong = serverSettings(port)
    .filter(({ key, value }) => !isDeepStrictEqual(config[key], value))
    .map(({ key, value }) => `${key} is ${JSON.stringify(config[key])}, expected ${JSON.stringify(value)}`);
  if (wrong.length > 0) throw new ServerConfigError(`unsafe server config: ${wrong.join("; ")}`);
}
