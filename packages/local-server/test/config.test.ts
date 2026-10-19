import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { DEFAULT_SHOWDOWN_DIR } from "@areneon/showdown-bridge";
import {
  checkNoReplayUpload,
  checkServerConfig,
  createRuntimeCopy,
  loadServerConfig,
  renderServerConfig,
  RUNTIME_STATE_DIRS,
  SERVER_CONFIG_PATH,
  ServerConfigError,
  serverSettings,
  writeServerConfig,
} from "../src/index.ts";
import { fakeCopy, tempDir } from "./helpers.ts";

const SAFE = Object.fromEntries(serverSettings(8123).map(({ key, value }) => [key, value]));

describe("the server config", () => {
  it("binds to 127.0.0.1 and documents every setting it changes", () => {
    const text = renderServerConfig(8123);
    expect(text).toContain(`exports.bindaddress = "127.0.0.1";`);
    expect(text).toContain("exports.port = 8123;");
    expect(text).not.toContain("0.0.0.0\"");
    for (const { key, why } of serverSettings(8123)) {
      expect(text).toContain(`// ${why}\nexports.${key} = `);
    }
  });

  it("is written only into a runtime copy, with the server's state directories", () => {
    const copy = fakeCopy("// unused\n");
    const before = readdirSync(copy.dir).sort();
    const written = writeServerConfig(copy.dir, 8123);
    expect(written.path).toBe(join(copy.dir, SERVER_CONFIG_PATH));
    const bytes = readFileSync(written.path);
    expect(bytes.toString("utf8")).toBe(renderServerConfig(8123));
    expect(written.sha256).toBe(createHash("sha256").update(bytes).digest("hex"));
    for (const state of RUNTIME_STATE_DIRS) expect(statSync(join(copy.dir, state)).isDirectory()).toBe(true);
    expect(readdirSync(copy.dir).sort()).toEqual([...before, "logs"].sort());
    expect(readdirSync(join(copy.dir, "config")).sort()).toEqual([
      "chat-plugins",
      "config-example.js",
      "config.js",
      "ladders",
    ]);
  });

  it("is refused for the vendor checkout and any other directory that is not a runtime copy", () => {
    for (const dir of [DEFAULT_SHOWDOWN_DIR, tempDir("plain")]) {
      const had = existsSync(join(dir, "logs/chat"));
      expect(() => writeServerConfig(dir, 8123)).toThrow(ServerConfigError);
      expect(existsSync(join(dir, "logs/chat"))).toBe(had);
    }
    expect(readdirSync(join(DEFAULT_SHOWDOWN_DIR, "config"))).not.toContain("config.js.tmp");
  });

  it("is merged over config-example.js as the server merges it, and passes the check", () => {
    const copy = fakeCopy("// unused\n");
    writeServerConfig(copy.dir, 8123);
    const merged = loadServerConfig(copy.dir);
    expect(merged).toEqual(SAFE);
    expect(() => checkServerConfig(merged, 8123)).not.toThrow();
  });

  it("is checked again from disk, so a later edit is caught", () => {
    const copy = fakeCopy("// unused\n");
    const { path } = writeServerConfig(copy.dir, 8123);
    expect(() => checkServerConfig(loadServerConfig(copy.dir), 8123)).not.toThrow();
    writeFileSync(path, `${readFileSync(path, "utf8")}exports.autosavereplays = true;\n`);
    expect(() => checkServerConfig(loadServerConfig(copy.dir), 8123)).toThrow(/would upload replays: autosavereplays/);
  });
});

describe("the replay-upload check refuses a config that", () => {
  it.each([
    [{ autosavereplays: true }, "autosavereplays is on"],
    [{ autosavereplays: "private" }, "autosavereplays is on"],
    [{ replaysdb: { host: "db.example.invalid" } }, "replaysdb is set"],
    [{ noNetRequests: false }, "noNetRequests is off"],
    [{ noNetRequests: undefined }, "noNetRequests is off"],
  ])("sets %j", (change, reason) => {
    const config = { ...SAFE, ...change };
    expect(() => checkNoReplayUpload(config)).toThrow(ServerConfigError);
    expect(() => checkNoReplayUpload(config)).toThrow(reason);
    expect(() => checkServerConfig(config, 8123)).toThrow(/would upload replays/);
  });

  it("passes the safe config", () => {
    expect(() => checkNoReplayUpload(SAFE)).not.toThrow();
  });
});

describe("checkServerConfig refuses", () => {
  it.each([
    ["bindaddress", "0.0.0.0"],
    ["backdoor", true],
    ["consoleips", ["127.0.0.1"]],
    ["repl", true],
    ["watchconfig", true],
    ["forcetimer", false],
    ["usepostgres", "postgres://db.example.invalid"],
    ["dnsbl", true],
    ["logchallenges", true],
  ])("%s = %j", (key, value) => {
    expect(() => checkServerConfig({ ...SAFE, [key]: value }, 8123)).toThrow(
      new RegExp(`unsafe server config: ${key} is`),
    );
  });

  it("another port than the session's", () => {
    expect(() => checkServerConfig(SAFE, 8124)).toThrow("port is 8123, expected 8124");
  });
});

describe("the pinned config-example.js", () => {
  it(
    "under Areneon's config.js yields the locked settings",
    () => {
      const copy = createRuntimeCopy({ runtimeRoot: tempDir("real-config") });
      writeServerConfig(copy.dir, 8123);
      const merged = loadServerConfig(copy.dir);
      expect(() => checkServerConfig(merged, 8123)).not.toThrow();
      // The upstream defaults Areneon overrides really differ at the pin.
      const upstream = readFileSync(join(copy.dir, "config/config-example.js"), "utf8");
      expect(upstream).toContain("exports.bindaddress = '0.0.0.0';");
      expect(upstream).toContain("exports.backdoor = true;");
      expect(upstream).toContain("exports.repl = true;");
      expect(merged["loginserver"]).toMatch(/^http/);
    },
    120_000,
  );
});
