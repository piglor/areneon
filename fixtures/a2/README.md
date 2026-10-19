# A2 harness fixtures

Synthetic bot-vs-bot battles recorded on the pinned local Showdown server (ADR-005 §5), with the server's own input log. CC0 (`fixtures/LICENSE`). Every file is `data_class: synthetic` with `rights: cc0-synthetic`: no real team, no real player, no chat.

They are the input of the A1/A2 equality test: the A1 runner plays the same input log and its observations must equal the A2 ones after removing `decision_context` and every `rqid` (ADR-008 §4). The A2 side starts from the raw server frames each bot received (`frames`): the test feeds them through `ProtocolReader` and `A2Channel` (with `stripServerLines`) itself, so a regression in the adapter changes the result and fails the test even though no fixture changes.

## What a fixture holds

`gen9championsvgc2026regmc/bot-vs-bot-<n>.json` (`schema_version` 2; version 1 stored only `channels`, not `room` or `frames`):

| Field | Meaning |
|---|---|
| `manifest` | the version manifest: pin (`showdown_commit`), format, rules hashes |
| `seed` | the PRNG seed the server drew for the battle (the `>start` line) |
| `players` | per seat: the user `name`, the `avatar` the server put on its `|player|` line, and the team file under `fixtures/teams/<format>/` |
| `teams` | per seat: the team as the bot sent it (nicknames included) |
| `served_teams` | per seat: the team as the server used it, unpacked from the input log's own `>player` line. The server may clear nicknames (its nickname filter decides, so it varies between environments), which changes `|switch|` names. Play these on A1, not `teams` |
| `room` | the battle room both bots joined (the `>ROOMID` header of every frame) |
| `frames` | per seat: the room's frames exactly as that bot's WebSocket delivered them, before any parsing: each string is one frame (the `>ROOMID` header line, then its lines, `\n`-separated), in arrival order, so frame boundaries, line order and the server-only lines (`|title|`, `|j|`, `|inactive|`, `|raw|`, the `uhtml` boxes, `|t:|`) and any re-sent `|request|` are all there. Only that seat's own stream: the server sends each client only what it may see, so there is no `|showteam|`, no `|split|` and no request of the other seat. Frames of other rooms and the global room are not kept. One edit only: a `|t:|<epoch>` line (the server's wall clock) is stored as `|t:|0`; nothing reads that number and `stripServerLines` drops the line whatever it says |
| `channels` | per seat: its own `PlayerChannel`. **Derived** from `frames` (what the session's `ProtocolReader` + `A2Channel` build, requests with their `rqid`) and checked against them by `verifyHarnessFixture`; it is not a second source of truth |
| `input_log` | the server's input log, unmodified. **Omniscient**: both teams and every choice. An evaluation artifact, never an actor's input |
| `winner`, `turns` | the result, from the server's log |

Nothing is normalised. A1 must use the recorded `name` and `avatar` to produce the same `|player|` lines the channels carry (an empty avatar, as the golden battle has, would differ).

The step-by-step A1 replay (`runBattle` with `replayActor`) is #764. `verifyHarnessFixture` (`packages/local-server/src/harness.ts`) already checks, with no server, that a fixture is synthetic with matching rights, names this commit, that its input log agrees with `seed`, `players` and `served_teams` (which differs from `teams` at most in nicknames), that each channel is that player's own (no opponent secret line, no `|showteam|`, only its own requests), that each seat's `frames` are well formed (all headed with `room`, the first starting `|init|battle`, no `|showteam|`, `|split|` or clock number, only the seat's own requests) and that replaying them through `ProtocolReader` and `A2Channel` gives exactly the stored `channels`, and that the pinned sim run on the input log gives the fixture's `winner` and `turns` and the channels' `|player|` lines and `|win|` line. The other channel lines are not compared with the replay: that is the equality test (#764).

## Provenance

- Server: the pinned, unmodified Showdown (`vendor/pokemon-showdown`, commit in `manifest.showdown_commit`), launched by `launchHarness`. Its config differs from the sparring one in one setting, `logchallenges: true`, so the server writes a finished challenge battle's input log under `logs/`. The sparring config (`serverSettings`, used by `launch`) never turns it on; a test asserts both.
- Players: two `runA2Bot` clients (`packages/bot-client`), both on the shared actor rule (first non-pending candidate), open team sheets declined. The sparring bot waits; the challenger sends the Bo1 challenge once the sparring bot is logged in. User names are fixed in `packages/local-server/src/recordings.ts`.
- Teams: `fixtures/teams/gen9championsvgc2026regmc/legal-synthetic.txt` and `legal-opponent.txt` (synthetic, CC0), once each way round.

## The seed

Replaying the committed input log is deterministic: `seed` plus the choices in `input_log` give the same battle on the pinned sim every time (`replayInputLog`; tested).

Re-recording yields a different battle. Fields that change: `seed`, `input_log`, `frames`, `channels`, `winner`, `turns` and the avatars in `players`. Fields that stay fixed: the user names, `room` (the room counter of a fresh server), `teams` and the team files, `manifest` (the pin) and `format_id`.

This is a server limit that the ticket wording did not anticipate: a challenge takes no seed (the server draws one when the battle starts) and `/importinputlog` needs staff rights (ADR-005 §5).

## How the frames are captured

`playBotVsBot` gives each `runA2Bot` a `connect` that wraps the real WebSocket in `tapTransport` (`packages/local-server/src/harness.ts`): a pure observer that copies every frame into a list before the session parses it and then delivers it unchanged, so the bot's behaviour is the one of a sparring run and `session.ts` is untouched. Each bot's connections are kept apart; `roomFrames` keeps the battle room's frames (one connection per player is expected) and redacts the clock. The natural recordings hold no re-sent request (the server re-sends on a reconnect or rejoin, which these runs do not do); the equality test injects one into a copy of the frames to exercise the adapter's re-send handling.

## Re-recording

Only after a deliberate Showdown pin bump or a change to the bot's choices; review the diff.

```bash
make showdown
node packages/local-server/scripts/record-a2.ts
```

It starts the server on `127.0.0.1:8000` (`--port N` to move it), plays the battles of `RECORDINGS` and stops the server. CI does not run it; it runs `verifyHarnessFixture` on the committed files, so a pin bump that leaves them stale fails the build.
