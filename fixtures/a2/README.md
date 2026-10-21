# A2 harness fixtures

Synthetic bot-vs-bot battles recorded on the pinned local Showdown server (ADR-005 §5), with the server's own input log. CC0 (`fixtures/LICENSE`). Every file is `data_class: synthetic` with `rights: cc0-synthetic`: no real team, no real player, no chat.

They are the input of the A1/A2 equality test: the A1 runner plays the same input log and its observations must equal the A2 ones after removing `decision_context` and every `rqid` (ADR-008 §4). The A2 side starts from the raw server frames each bot received (`frames`): the test feeds them through `ProtocolReader` and `A2Channel` (with `stripServerLines`) itself, so a regression in the adapter changes the result and fails the test even though no fixture changes.

`bot-vs-bot-1` and `-2` are plain battles. `bot-vs-bot-3` and `-4` contain a real reconnect: the recorder closed one bot's WebSocket mid-battle, the bot reconnected through its normal path, and the server rejoined it and re-sent the request (with `|sentchoice|` in `-3`, where the bot had already answered; without it in `-4`, where the bot had not). What the equality test can and cannot compare for such a seat is under "Reconnects" below.

## What a fixture holds

`gen9championsvgc2026regmc/bot-vs-bot-<n>.json`. `schema_version` 3 stores a seat's `frames` and `channels` per connection (a list with one entry per connection the bot opened to the battle room). `bot-vs-bot-1` and `-2` are still version 2, which has exactly one connection per seat and stores `frames` and `channels` flat; `upgradeFixture` reads a version 2 fixture as the version 3 one with a single connection (content untouched), `verifyHarnessFixture` and the equality test do so, and any other version (1 stored only `channels`) is refused. A new recording is always version 3. The fields below are version 3:

| Field | Meaning |
|---|---|
| `manifest` | the version manifest: pin (`showdown_commit`), format, rules hashes |
| `seed` | the PRNG seed the server drew for the battle (the `>start` line) |
| `players` | per seat: the user `name`, the `avatar` the server put on its `|player|` line, and the team file under `fixtures/teams/<format>/` |
| `teams` | per seat: the team as the bot sent it (nicknames included) |
| `served_teams` | per seat: the team as the server used it, unpacked from the input log's own `>player` line. The server may clear nicknames (its nickname filter decides, so it varies between environments), which changes `|switch|` names. Play these on A1, not `teams` |
| `room` | the battle room both bots joined (the `>ROOMID` header of every frame) |
| `frames` | per seat, a list of connections, each a list of the room's frames exactly as that bot's WebSocket delivered them, before any parsing: each string is one frame (the `>ROOMID` header line, then its lines, `\n`-separated), in arrival order, so frame boundaries, line order and the server-only lines (`|title|`, `|j|`, `|inactive|`, `|raw|`, the `uhtml` boxes, `|t:|`) and any re-sent `|request|` are all there. Each connection's list starts at its own `|init|battle`. The lists are in the order the bot opened its connections; a connection that never received the room (only a login, or a login the server refused while the old connection was still held) has no list. Only that seat's own stream: the server sends each client only what it may see, so there is no `|showteam|`, no `|split|` and no request of the other seat. Frames of other rooms and the global room are not kept. One edit only: a `|t:|<epoch>` line (the server's wall clock) is stored as `|t:|0`; nothing reads that number and `stripServerLines` drops the line whatever it says |
| `channels` | per seat: one `PlayerChannel` per connection, in the same order. **Derived** from `frames` (what the session's `ProtocolReader` + `A2Channel` build for that connection, requests with their `rqid`; the session makes a new pair for every connection) and checked against them by `verifyHarnessFixture`; it is not a second source of truth |
| `input_log` | the server's input log, unmodified. **Omniscient**: both teams and every choice. An evaluation artifact, never an actor's input |
| `winner`, `turns` | the result, from the server's log |

Nothing is normalised. A1 must use the recorded `name` and `avatar` to produce the same `|player|` lines the channels carry (an empty avatar, as the golden battle has, would differ).

The step-by-step A1 replay (`runBattle` with `replayActor`) is #764. `verifyHarnessFixture` (`packages/local-server/src/harness.ts`) already checks, with no server, that a fixture is synthetic with matching rights, names this commit, that its input log agrees with `seed`, `players` and `served_teams` (which differs from `teams` at most in nicknames), that each channel is that player's own (no opponent secret line, no `|showteam|`, only its own requests), that each seat's `frames` are well formed, connection by connection (all headed with `room`, each starting `|init|battle`, no `|showteam|`, `|split|` or clock number, only the seat's own requests), that there is one stored channel per connection and that replaying each connection through a new `ProtocolReader` and `A2Channel` gives exactly its stored channel, that the requests go on across connections (a later connection's first request is the previous one's last sent again, same `rqid` and `request_hash`, or a newer one; never an older one, never the same `rqid` for another request), and that the pinned sim run on the input log gives the fixture's `winner` and `turns` and the channels' `|player|` lines and `|win|` line. The other channel lines are not compared with the replay: that is the equality test (#764).

## Provenance

- Server: the pinned, unmodified Showdown (`vendor/pokemon-showdown`, commit in `manifest.showdown_commit`), launched by `launchHarness`. Its config differs from the sparring one in one setting, `logchallenges: true`, so the server writes a finished challenge battle's input log under `logs/`. The sparring config (`serverSettings`, used by `launch`) never turns it on; a test asserts both.
- Players: two `runA2Bot` clients (`packages/bot-client`), both on the shared actor rule (first non-pending candidate), open team sheets declined. The sparring bot waits; the challenger sends the Bo1 challenge once the sparring bot is logged in. User names are fixed in `packages/local-server/src/recordings.ts`.
- Teams: `fixtures/teams/gen9championsvgc2026regmc/legal-synthetic.txt` and `legal-opponent.txt` (synthetic, CC0), once each way round.

## The seed

Replaying the committed input log is deterministic: `seed` plus the choices in `input_log` give the same battle on the pinned sim every time (`replayInputLog`; tested).

Re-recording yields a different battle. Fields that change: `seed`, `input_log`, `frames`, `channels`, `winner`, `turns` and the avatars in `players`. Fields that stay fixed: the user names, `room` (the room counter of a fresh server), `teams` and the team files, `manifest` (the pin) and `format_id`.

This is a server limit that the ticket wording did not anticipate: a challenge takes no seed (the server draws one when the battle starts) and `/importinputlog` needs staff rights (ADR-005 §5).

## How the frames are captured

`playBotVsBot` gives each `runA2Bot` a `connect` that wraps the real WebSocket in `tapTransport` (`packages/local-server/src/harness.ts`): a pure observer that copies every frame into a list before the session parses it and then delivers it unchanged, so the bot's behaviour is the one of a sparring run and `session.ts` is untouched. Each bot's connections are kept apart; `roomFrames` keeps the battle room's frames of every connection that received the room, one list per connection, and redacts the clock. A plain recording has one connection per bot and no re-sent request; the equality test injects one into a copy of the frames to exercise the adapter's re-send handling. The two reconnect recordings hold the real thing, below.

## Reconnects

How a recording gets one. `RECORDINGS` gives a recording an optional `drop` plan (`DropPlan` in `harness.ts`): which bot, at which of its own requests (its n-th `|request|` line on that connection, `wait` included, Team Preview is 1), and `before_choice` or `after_choice`. `playBotVsBot` hands the bot's first connection to `ConnectionDrop` (a transport wrapper like the tap, so `session.ts` and `runA2Bot` are untouched). At the plan's point it closes the WebSocket and from then on delivers nothing and sends nothing, so the bot sees an ordinary close and reconnects through its normal path (`reconnect`: a first attempt at once). `before_choice` cuts when the request arrives, before the bot can answer, so the server holds no choice and re-sends the request alone. `after_choice` cuts right after the bot's `/choose` for that request went out, so the server holds the choice and re-sends the request with `|sentchoice|`; the other bot's choice for the same request is held back until the cut bot has its request again, otherwise the turn could resolve before the cut and the server would have a new request, not the old one, to send. `recordHarnessBattle` checks the result (`reconnectEvidence`): the second connection's first request must be the first connection's last one sent again (same `rqid` and `request_hash`), not a `wait`, with `|sentchoice|` exactly when planned; otherwise the recording is refused (`ReconnectNotPlannedError`) and the recorder plays it again, up to 4 attempts, each on a fresh server (the held choice also gives up waiting after 10 s if the cut bot never comes back). The `|sentchoice|` must hold a choice the re-sent request offers (the bot's own notation, `move 1 1, move 1`; the server's input log writes the same choice differently, `move moonblast +1, move earthquake`, so the two are not compared as text). The equality test plays the held choice on A1 in place of the logged one for that decision, and requires `|sentchoice|` to be there or not as the fixture's provenance says. If a recording is still refused after the attempts, record it again: the battle, and so the request the plan lands on, differs every time, and the plan's request numbers are early enough to be reached in nearly every battle. The recorder can redo one file: `--only bot-vs-bot-3.json`.

What the server sends a rejoining connection (as recorded): `|init|battle`, the room log from the start (no `|request|` and no `|error|` line in it: the server sends those to the player alone and never logs them), the lines of the leave and the rejoin, and then, in a frame of its own, the stored `|request|` (with `|sentchoice|` if a choice is held). The session starts a new `ProtocolReader` and a new `A2Channel` for the new connection (`Connection` in `session.ts`; `A2BattleRecord.channels` has one channel per connection), so `request_seq` restarts at 1 and the first request of the new channel is the re-sent one. The A1 channel never restarts: it holds every `|request|` line of the battle.

### What the equality test compares

Per seat, per connection, with a new `ProtocolReader` and adapter for each (`compareA1A2` in `packages/bot-client/test/a1-a2-equality.ts`; the same file's header states the rule):

1. **Connection 1** is a prefix of A1's channel and is compared exactly as a plain seat is: channel lines (only a request's `rqid` is left out), `request_hash` and `request_seq`, and for every decision the observation, with `decision_context` and `rqid` removed, nothing else.
2. **A later connection** is joined to the connections before it with `continueChannel` (`player-channel.ts`; the rule of #813: a rejoin channel carries the anchor of the connection it continues). The rejoin channel alone is A1's channel without its `|request|` lines (the replayed room log holds none) and its `request_seq` counts from 1; the join puts the earlier channel's requests back at the places they had and drops the re-sent request (same `rqid` and body) once. The joined channel must equal A1's one continuous channel line for line (a cut connection's joined channel is a prefix), and at every decision the observation from it equals A1's FULL observation (log, `request_seq` and team included; only `decision_context` and `rqid` removed).
3. **Across connections** a request is identified by `rqid` + `request_hash` (ADR-008 §4): a later connection's first request with the previous connection's last `rqid` and hash is that request sent again, one decision (`resentAcross`), compared on both connections; the `rqid`s of a seat's decisions rise; the choices of the input log, assigned to the non-wait requests in order, are the same the bot made, and A1 replaying them reaches the same channels to the last line, `|win|` included.

Decision (#813): a battle with a reconnect is one continuous channel, in the Episode and in the observations (ADR-008 §3: one channel per player and battle); the per-connection channels are an input of the join, not what is stored. `buildObservation` is unchanged and still checks the team against the Team Preview request of the channel it is given; a bare rejoin channel is refused with a message that names `continueChannel`. The `|poke|` lines of the replay are no anchor: they carry only species and whether an item is held, not the sets. The join needs the earlier connection's channel of the same process; a rejoin with no earlier channel still builds no observation (fail closed). Not here: `session.ts` and `a2Episode` still keep one channel per connection and `a2Episode` refuses a battle with several; wiring them to `continueChannel` is a follow-up outside this slice.

Negative controls (`a1-a2-equality.test.ts`): an adapter that does not start a new channel at the rejoin (one channel per seat; the later `|init|battle` removed) fails with channel differences on connection 2 (the join accepts the old history as the earlier channel, but the result is not A1's); an adapter that does not drop re-sent requests fails when the rejoin connection also carries the same request twice (a copy injected into the real rejoin connection; the real rejoin alone does not exercise it, because a re-send after an `|init|` is already a new channel's first request); a rejoin whose replayed log lost a line (the join refuses it), whose re-sent request differs or whose `rqid` goes back, or a second connection with no `|init|battle`, are all refused.

## Re-recording

Only after a deliberate Showdown pin bump or a change to the bot's choices; review the diff.

```bash
make showdown
node packages/local-server/scripts/record-a2.ts [--only bot-vs-bot-3.json]
```

It starts the server on `127.0.0.1:8000` (`--port N` to move it), plays the battles of `RECORDINGS` (or the one named by `--only`) and stops the server. CI does not run it; it runs `verifyHarnessFixture` on the committed files, so a pin bump that leaves them stale fails the build.
