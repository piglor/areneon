# A2 harness fixtures

Synthetic bot-vs-bot battles recorded on the pinned local Showdown server (ADR-005 §5), with the server's own input log. CC0 (`fixtures/LICENSE`). Every file is `data_class: synthetic` with `rights: cc0-synthetic`: no real team, no real player, no chat.

They are the input of the A1/A2 equality test: the A1 runner plays the same input log and its observations must equal the A2 ones after removing `decision_context` and every `rqid` (ADR-008 §4).

## What a fixture holds

`gen9championsvgc2026regmc/bot-vs-bot-<n>.json`:

| Field | Meaning |
|---|---|
| `manifest` | the version manifest: pin (`showdown_commit`), format, rules hashes |
| `seed` | the PRNG seed the server drew for the battle (the `>start` line) |
| `players` | per seat: the user `name`, the `avatar` the server put on its `|player|` line, and the team file under `fixtures/teams/<format>/` |
| `teams` | per seat: the team as the bot sent it (nicknames included) |
| `served_teams` | per seat: the team as the server used it, unpacked from the input log's own `>player` line. The server may clear nicknames (its nickname filter decides, so it varies between environments), which changes `|switch|` names. Play these on A1, not `teams` |
| `channels` | per seat: its own `PlayerChannel` as its bot client received it (server-only lines already dropped by `A2Channel`, requests with their `rqid`) |
| `input_log` | the server's input log, unmodified. **Omniscient**: both teams and every choice. An evaluation artifact, never an actor's input |
| `winner`, `turns` | the result, from the server's log |

Nothing is normalised. A1 must use the recorded `name` and `avatar` to produce the same `|player|` lines the channels carry (an empty avatar, as the golden battle has, would differ).

The step-by-step A1 replay (`runBattle` with `replayActor`) is #764. `verifyHarnessFixture` (`packages/local-server/src/harness.ts`) already checks, with no server, that a fixture is synthetic with matching rights, names this commit, that its input log agrees with `seed`, `players` and `served_teams` (which differs from `teams` at most in nicknames), that each channel is that player's own (no opponent secret line, no `|showteam|`, only its own requests), and that the pinned sim run on the input log gives the fixture's `winner` and `turns` and the channels' `|player|` lines and `|win|` line. The other channel lines are not compared with the replay (that is #764).

## Provenance

- Server: the pinned, unmodified Showdown (`vendor/pokemon-showdown`, commit in `manifest.showdown_commit`), launched by `launchHarness`. Its config differs from the sparring one in one setting, `logchallenges: true`, so the server writes a finished challenge battle's input log under `logs/`. The sparring config (`serverSettings`, used by `launch`) never turns it on; a test asserts both.
- Players: two `runA2Bot` clients (`packages/bot-client`), both on the shared actor rule (first non-pending candidate), open team sheets declined. The sparring bot waits; the challenger sends the Bo1 challenge once the sparring bot is logged in. User names are fixed in `packages/local-server/src/recordings.ts`.
- Teams: `fixtures/teams/gen9championsvgc2026regmc/legal-synthetic.txt` and `legal-opponent.txt` (synthetic, CC0), once each way round.

## The seed

Replaying the committed input log is deterministic: `seed` plus the choices in `input_log` give the same battle on the pinned sim every time (`replayInputLog`; tested).

Re-recording yields a different battle. Fields that change: `seed`, `input_log`, `channels`, `winner`, `turns` and the avatars in `players`. Fields that stay fixed: the user names, `teams` and the team files, `manifest` (the pin) and `format_id`.

This is a server limit that the ticket wording did not anticipate: a challenge takes no seed (the server draws one when the battle starts) and `/importinputlog` needs staff rights (ADR-005 §5).

## Re-recording

Only after a deliberate Showdown pin bump or a change to the bot's choices; review the diff.

```bash
make showdown
node packages/local-server/scripts/record-a2.ts
```

It starts the server on `127.0.0.1:8000` (`--port N` to move it), plays the battles of `RECORDINGS` and stops the server. CI does not run it; it runs `verifyHarnessFixture` on the committed files, so a pin bump that leaves them stale fails the build.
