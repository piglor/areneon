// SPIKE (#672): throwaway test harness. Boots the pinned, unmodified Showdown
// server (vendor/pokemon-showdown, MIT) in this process, bound to 127.0.0.1
// only, and exposes a localhost-only control endpoint the test uses to start
// a battle with a fixed seed (the server has no player-facing seed option).
// Config is changed in memory only; no vendor file is edited.
"use strict";
const http = require("node:http");
const path = require("node:path");

const [, , portArg, controlArg] = process.argv;
const port = Number(portArg);
const controlPort = Number(controlArg);
const vendor = path.resolve(__dirname, "../../../vendor/pokemon-showdown");

process.chdir(vendor);
process.argv = [process.argv[0], "pokemon-showdown", "start", "--no-security", String(port)];
const ConfigLoader = require(path.join(vendor, "dist/server/config-loader.js"));
const Config = ConfigLoader.load();
Config.bindaddress = "127.0.0.1";
Config.subprocesses = 0;
Config.subprocessescache = Object.fromEntries(Object.keys(Config.subprocessescache).map((k) => [k, 0]));
Config.repl = false;
Config.ssl = null;
Config.loginserver = "http://127.0.0.1:9/"; // unreachable on purpose: never contact the public login server
Config.watchconfig = false;
Config.reportjoins = false;
require(path.join(vendor, "dist/server/index.js"));

function readBody(req) {
  return new Promise((resolve) => {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => resolve(body));
  });
}

http
  .createServer(async (req, res) => {
    try {
      if (req.method !== "POST" || req.url !== "/battle") throw new Error("not found");
      const { format, seed, players, team } = JSON.parse(await readBody(req));
      const packed = global.Teams.pack(team);
      const users = players.map((id) => global.Users.get(id));
      if (users.some((u) => !u || !u.connected)) throw new Error(`players not connected: ${players}`);
      const room = global.Rooms.createBattle({
        format,
        seed,
        rated: false,
        challengeType: "challenge",
        players: users.map((user) => ({ user, team: packed })),
      });
      res.end(JSON.stringify({ roomid: room.roomid }));
    } catch (error) {
      res.statusCode = 400;
      res.end(JSON.stringify({ error: String(error) }));
    }
  })
  .listen(controlPort, "127.0.0.1", () => console.log(`READY ${port} ${controlPort}`));
