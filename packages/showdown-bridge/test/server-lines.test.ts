import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { type ChannelFixture, GOLDEN_CHANNELS, GOLDEN_DIR } from "../src/golden.ts";
import { playerChannel } from "../src/player-channel.ts";
import { serverOnlyLine, stripServerLines } from "../src/server-lines.ts";

const SOURCE = new URL("../src/server-lines.ts", import.meta.url);

const golden = Object.keys(GOLDEN_CHANNELS).map((file) => ({
  file,
  ...(JSON.parse(readFileSync(join(GOLDEN_DIR, file), "utf8")) as ChannelFixture),
}));
const p1 = (golden.find(({ channel }) => channel.player === "p1") as ChannelFixture).channel.lines;
const at = (prefix: string): number => p1.findIndex((line) => line.startsWith(prefix));

// One example of every kind the filter drops without a report.
const SERVER: readonly string[] = [
  ">battle-gen9championsvgc2026regmc-1",
  "Plain-text notice from the server",
  "Bob rejected open team sheets.",
  "||The battle is starting",
  "|request|null",
  "|player|p2|",
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
  '|raw|<div class="infobox">hello</div>',
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
  "|tournament|create|gen9championsvgc2026regmc|elimination",
  "|battle|battle-1|Bob|Areneon Bot",
  "|sentchoice|move 1 1, move 1 2",
  "|-hint|If you want to tie earlier, consider using `/offertie`.",
  "|-message|All players are inactive.",
  "|uhtmlchange|otsrequest|<div>changed</div>",
];

// Sim lines that are kept although some look like server lines.
const SIM: readonly string[] = [
  "",
  "|",
  "|error|[Unavailable choice] Can't move: Snorlax's Fire Punch is disabled",
  "|error|[Invalid choice] There's nothing to choose",
  "|message|Time's up! Going to tiebreaker...",
  "|-message|Sleep Clause Mod activated.",
  "|bigerror|You will auto-tie if the battle doesn't end in 10 turns (on turn 1000).",
  "|-hint|Some effects can force a Pokemon to use Outrage again in a row.",
  "|-hint|If you want to tie earlier, consider using /offertie.",
  "|tier|[Gen 9 Champions] VGC 2026 Reg M-C",
  "|clearpoke",
  "|turn|2",
  "|-terastallize|p1a: Snorlax|Normal",
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

  it.each(SERVER)("drops the server line %j without a report", (line) => {
    expect(serverOnlyLine(line)).toBe(true);
    expect(stripServerLines([line])).toEqual({ lines: [], dropped: [] });
  });

  it.each(SIM)("keeps the sim line %j", (line) => {
    expect(serverOnlyLine(line)).toBe(false);
    expect(stripServerLines([line])).toEqual({ lines: [line], dropped: [] });
  });

  it("drops and reports a line of an unknown kind", () => {
    const unknown = "|modal|Something new";
    expect(serverOnlyLine(unknown)).toBe(false);
    expect(stripServerLines(["|turn|1", unknown, "|upkeep"])).toEqual({
      lines: ["|turn|1", "|upkeep"],
      dropped: [unknown],
    });
  });

  it.each(golden)("returns the golden A1 channel $file unchanged", ({ channel }) => {
    expect(channel.lines.length).toBeGreaterThan(0);
    expect(stripServerLines(channel.lines)).toEqual({ lines: channel.lines, dropped: [] });
  });

  it("drops a chat line shaped like the opponent's exact HP, which would fail the channel", () => {
    const chat = "|c|Bob|p2a: lol|3/4";
    const end = at("|request|");
    const stream = [">battle-1", "|j|Bob", ...p1.slice(0, end), chat, "|t:|1760000000", p1[end] as string];
    const channel = { schema_version: 1, data_class: "synthetic", player: "p1" } as const;
    // Negative control: unfiltered, the chat line reads as p2's SECRET HP.
    expect(() => playerChannel({ ...channel, lines: stream })).toThrow(/only the opponent may see/);
    const { lines } = stripServerLines(stream);
    expect(lines).toEqual(p1.slice(0, end + 1));
    expect(playerChannel({ ...channel, lines }).lines).toEqual(lines);
  });

  it("keeps one Open Team Sheets prompt and the sim's own clear when the bot declines", () => {
    const prompt = at("|uhtml|otsrequest|");
    expect(at("|start")).toBeGreaterThan(prompt);
    // The server clears the prompt for the declining player during Team
    // Preview, and says so in a plain-text line; the sim clears it at |start.
    const stream = [
      ...p1.slice(0, prompt + 1),
      "|uhtmlchange|otsrequest|",
      "p1 rejected open team sheets.",
      ...p1.slice(prompt + 1),
    ];
    expect(stream.filter((line) => line === "|uhtmlchange|otsrequest|")).toHaveLength(2);
    expect(stripServerLines(stream)).toEqual({ lines: p1, dropped: [] });
    // A prompt after |start is not the sim's.
    expect(stripServerLines(["|start", p1[prompt] as string]).lines).toEqual(["|start"]);
  });

  it("keeps only the sim's |player| line per slot", () => {
    const gametype = at("|gametype|");
    const start = at("|start");
    // addPlayer's lines come before the sim's |gametype|; a leave and a
    // rejoin during Team Preview, and a rejoin after |start, follow.
    const stream = [
      "|init|battle",
      "|title|p1 vs. p2",
      "|player|p1|p1|1|",
      "|player|p2|p2|2|",
      ...p1.slice(gametype, start),
      "|player|p2|",
      "|player|p2|p2|2|",
      ...p1.slice(start, start + 1),
      "|player|p1|p1|1|",
      ...p1.slice(start + 1),
    ];
    expect(gametype).toBe(0);
    expect(stripServerLines(stream)).toEqual({ lines: p1, dropped: [] });
  });

  it("drops the server's forfeit message for a player the sim named, and keeps other messages", () => {
    const end = p1.length - 1;
    expect(p1[end]).toMatch(/^\|win\|/);
    const forfeits = [
      "|-message|p2 forfeited.",
      "|-message|p1 was made to forfeit.",
      "|-message|p2 lost due to inactivity.",
      "|-message|p1 lost by being banned.",
      "|-message|p2 lost by being offline too long.",
      "|-message|p1 was disqualified from the tournament.",
    ];
    expect(stripServerLines([...p1.slice(0, end), ...forfeits, p1[end] as string])).toEqual({ lines: p1, dropped: [] });
    // Another name, or a message before the sim named anyone, is not matched.
    const other = "|-message|Bob forfeited.";
    expect(stripServerLines([...p1, other]).lines).toEqual([...p1, other]);
    expect(stripServerLines(["|-message|p2 forfeited."]).lines).toEqual(["|-message|p2 forfeited."]);
  });
});
