import { readdirSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { playerChannel } from "../src/player-channel.ts";
import { serverOnlyLine, stripServerLines } from "../src/server-lines.ts";

const SOURCE = new URL("../src/server-lines.ts", import.meta.url);
const SHOWDOWN_FIXTURES = new URL("../../../fixtures/showdown/", import.meta.url);

interface GoldenChannel {
  channel: { schema_version: 1; data_class: "synthetic"; player: "p1" | "p2"; lines: string[] };
}

const goldenChannels = readdirSync(SHOWDOWN_FIXTURES).flatMap((format) =>
  ["p1", "p2"].map((player) => {
    const url = new URL(`${format}/channels/${player}.json`, SHOWDOWN_FIXTURES);
    return { name: `${format} ${player}`, ...(JSON.parse(readFileSync(url, "utf8")) as GoldenChannel) };
  }),
);

// One example of every kind the filter drops, as the server sends it.
const DROPPED: readonly string[] = [
  ">battle-gen9championsvgc2026regmc-1",
  "Plain-text notice from the server",
  "|request|null",
  "|j|Bob",
  "|J|Bob",
  "|join|Bob",
  "|l|Bob",
  "|L|Bob",
  "|leave|Bob",
  "|n|Bobby|bob",
  "|N|Bobby|bob",
  "|name|Bobby|bob",
  "|c|Bob|gg",
  "|c:|1760000000|Bob|gg",
  "|chat|Bob|gg",
  "|pm| Bob| Areneon Bot|hi",
  "|raw|<div class=\"infobox\">hello</div>",
  "|html|<div>hello</div>",
  "|uhtml|invites|<div>This battle needs more players</div>",
  "|uhtmlchange|invites|",
  "|uhtml|bestof|<h2>Game 1</h2>",
  "|notify|Areneon|Your turn",
  "|tempnotify|turn|Turn 2",
  "|tempnotifyoff|turn",
  "|popup|You were disconnected.",
  "|hidelines|delete|bob|1",
  "|unlink|bob",
  "|inactive|Battle timer is ON: inactive players will automatically lose when time's up.",
  "|inactive|Time left: 150 sec this turn | 140 sec total",
  "|inactiveoff|Battle timer is now OFF.",
  "|t:|1760000000",
  "|:|1760000000",
  "|timestamp|1760000000",
  "|init|battle",
  "|title|Bob vs. Areneon Bot",
  "|deinit",
  "|noinit|nonexistent|The room does not exist.",
  "|expire|",
  "|usercount|2",
  "|users|2, Bob, Areneon Bot",
  "|customgroups|[]",
  "|challstr|4|abc",
  '|updateuser| Areneon Bot|1|1|{"blockChallenges":false}',
  '|updatesearch|{"searching":[],"games":null}',
  '|updatechallenges|{"challengesFrom":{},"challengeTo":null}',
  "|nametaken|bob|Name taken.",
  "|formats|,1|S/V Singles",
  "|queryresponse|userdetails|{}",
  "|sentchoice|move 1 1, move 1 2",
];

// Engine lines the filter keeps although they look like server lines.
const KEPT: readonly string[] = [
  "",
  "|",
  "|error|[Unavailable choice] Can't move: Snorlax's Fire Punch is disabled",
  "|error|[Invalid choice] There's nothing to choose",
  "|message|Time's up! Going to tiebreaker...",
  "|-message|All players are inactive.",
  "|bigerror|You will auto-tie if the battle doesn't end in 10 turns (on turn 1000).",
  "|tier|[Gen 9 Champions] VGC 2026 Reg M-C",
  "|uhtml|otsrequest|<button name=\"send\" value=\"/acceptopenteamsheets\">Accept</button>",
  "|uhtmlchange|otsrequest|",
  "|clearpoke",
  "|request|{}",
];

describe("server-lines", () => {
  it("imports nothing, so any adapter can bundle it", () => {
    const source = readFileSync(SOURCE, "utf8");
    // As in canonical-json's test: `export … from` loads a module too. The
    // file's string literals never contain `from "`.
    expect(source).not.toMatch(/^\s*import\b/m);
    expect(source).not.toMatch(/\bfrom\s*["']/);
    expect(source).not.toMatch(/\brequire\s*\(|\bimport\s*\(|\bimport\.meta\b/);
  });

  it.each(DROPPED)("drops the server-only line %j", (line) => {
    expect(serverOnlyLine(line)).toBe(true);
  });

  it.each(KEPT)("keeps the engine line %j", (line) => {
    expect(serverOnlyLine(line)).toBe(false);
  });

  it.each(goldenChannels)("returns the golden A1 channel $name unchanged", ({ channel }) => {
    expect(channel.lines.length).toBeGreaterThan(0);
    expect(stripServerLines(channel.lines)).toEqual(channel.lines);
  });

  it("drops a chat line shaped like the opponent's exact HP, which would fail the channel", () => {
    const chat = "|c|Bob|p2a: lol|3/4";
    const { channel } = goldenChannels.find(({ name }) => name.endsWith("p1")) as GoldenChannel;
    const at = channel.lines.findIndex((line) => line.startsWith("|request|"));
    // Negative control: unfiltered, the chat line reads as p2's SECRET HP.
    const stream = [">battle-1", "|j|Bob", ...channel.lines.slice(0, at), chat, "|t:|1760000000", channel.lines[at] as string];
    expect(() => playerChannel({ ...channel, lines: stream })).toThrow(/only the opponent may see/);
    const lines = stripServerLines(stream);
    expect(lines).toEqual(channel.lines.slice(0, at + 1));
    expect(playerChannel({ ...channel, lines }).lines).toEqual(lines);
  });

  it("keeps the order of the lines it keeps", () => {
    expect(stripServerLines(["|j|Bob", "|turn|1", "|c|Bob|hi", "|", "|error|[Invalid choice] x", "|t:|1"])).toEqual([
      "|turn|1",
      "|",
      "|error|[Invalid choice] x",
    ]);
  });
});
