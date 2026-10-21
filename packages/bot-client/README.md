# Human sparring guide

This guide is for one person who wants to play a Doubles battle against the Areneon bot on their own machine (runtime A2, ADR-005). Everything runs on `127.0.0.1`. **The bot never connects to any public Showdown server**: `runA2Bot` refuses every URL that is not a loopback address (`assertLocal`), and the launcher binds the server to `127.0.0.1` only.

The format is `[Gen 9 Champions] VGC 2026 Reg M-C` (id `gen9championsvgc2026regmc`), Bo1, Closed Team Sheet.

## 1. Start the server and the bot

One command starts the pinned Showdown server and the bot together. From the repository root, after `make setup` (it builds the pinned Showdown):

```bash
node packages/bot-client/scripts/a2-bot.ts --challenger <YourName> --team <bot-team.txt>
```

| Option | Meaning | Default |
|---|---|---|
| `--challenger NAME` | Required. The only user whose challenge the bot accepts. | none |
| `--team PASTE` | Required. The bot's team, a PokePaste file in the engine's canonical form. It must be valid for the format, or the run stops before play. | none |
| `--format ID` | The format id. Optional while the rules snapshot holds one format, as it does now. | the only format |
| `--port N` | The server's port, 1 to 65535. | `8000` |
| `--runtime-root DIR` | Where the verified runtime copy of the server is made. | `<repo>/.local-server/runtime` |
| `--out DIR` | Where battle records go (see section 7). | `private-data/a2` |

The program has no `--help`: an unknown option, or a missing required one, prints the usage text and exits with code 2. Other exit codes: 0 stopped, 1 failed (a failed start, a fatal session error, or a record that could not be stored).

To try the setup without a real team, use the synthetic fixture:

```bash
node packages/bot-client/scripts/a2-bot.ts --challenger <YourName> \
  --team fixtures/teams/gen9championsvgc2026regmc/legal-synthetic.txt
```

When the server is up, the program prints one JSON line:

```json
{"event":"ready","client":"http://127.0.0.1:8000","url":"ws://127.0.0.1:8000/showdown/websocket","bot":"Areneon-bot-78be5a","challenger":"<YourName>","format":"gen9championsvgc2026regmc","server_pid":12345}
```

Take the client address and the bot's name from it. The bot's name is `Areneon-bot-` plus six random hex characters, new for each start. The lines after it (`connected`, `logged_in`, one per battle event) are the operator log. They show what the bot did, not what the server sent: a line the bot's channel did not keep appears as its kind only.

## 2. Connect

Point a Showdown client at the local server. There are two ways to get one (ADR-005 §4):

- **The official web client through `localhost`.** Open `http://localhost:<port>` (the port from the `ready` line). The server's page redirects to `localhost.psim.us`, and the client served there connects to your local server.
- **A self-built, unmodified client.** Build the upstream Showdown client yourself, outside this repo (it is AGPL: Areneon does not copy or ship it, and CI does not run it), then open it with `testclient-new.html?~~localhost:<port>`. This removes the dependency on Smogon's hosting.

The battle traffic stays on `127.0.0.1`. With the official client the page itself is fetched from Smogon's static hosting, so Smogon sees your IP address; the self-built client avoids that. The local server itself makes no outbound requests (its `noNetRequests` setting is on).

The server runs without login security, so any unregistered name works: pick a throwaway name, which you also give as `--challenger`. Do not use a name or password you use elsewhere.

The server sets `forcetimer`, `autosavereplays` off and `logchallenges` off, so there is no replay upload and no challenge log.

### The proxy-lock popup

The local server marks every `127.0.0.1` connection as a proxy and locks it. You will probably see a popup saying your IP is locked due to being a proxy, and a lock symbol next to your name; the bot gets the same popup, and its operator log shows it as a `server_message` line. This is the pinned server's behaviour for loopback connections. Challenges, accepting, the Open Team Sheets decline and choices still work. This is checked with the bot against itself on the launched server (`packages/bot-client/test/session.local-server.test.ts`) and noted in the session-loop ticket (#757); it is not checked with a human client (see "Not verified" below). The earlier human-versus-bot match (section 6) predates this runtime and did not record the popup. Close the popup and carry on.

## 3. Challenge the bot

1. Challenge the bot by its name (from the `ready` line).
2. Choose exactly `[Gen 9 Champions] VGC 2026 Reg M-C`, with your own team.
3. **Untick Best-of-3 before you send the challenge.** The client defaults this format to Best-of-3. The bot does not play Bo3. From the server code (not tested with a Bo3 challenge or a human client), a Bo3 challenge reaches the bot under a format id with a custom rule appended, which is not an id in the rules snapshot, so the bot rejects it. The `(Bo3)` format is refused the same way.
4. The name you challenge from must be the one in `--challenger` (compared by Showdown user id: case, spaces and punctuation do not matter). A challenge from any other user is rejected.

The bot rejects, and says why in the operator log (`challenge_rejected`), any challenge that is not a plain battle challenge, comes from another user, or whose format id is not the plain `gen9championsvgc2026regmc` id. It plays one battle at a time: a challenge made during a battle is rejected as `busy`.

## 4. Open team sheets

Closed Team Sheet is the point of this format: Areneon plays it with the sheets declined (ADR-003, ADR-005 §2). The bot sends `/rejectopenteamsheets` on its first Team Preview request and never `/acceptopenteamsheets`. It sends its preview choice only after the server has confirmed its rejection. **You should decline too** (do not click Accept on the sheet prompt).

- If you decline, or do not answer: neither side's sheet is shown.
- If you accept: the bot's rejection still stands and the server does not show the sheets, because it shows them only when every player accepted. The battle plays on without sheets. This follows the server code; it has not been tried with a human client.
- If the bot cannot confirm its own rejection, or a team sheet is shown anyway, the bot forfeits the battle and records why (`ots_failure`: `not_confirmed`, `refused` or `showteam`).

## 5. The timer

The server runs with `forcetimer` on and the format's VGC Timer, and the bot also sends `/timer on` when it joins a battle room. You cannot switch it off for the bot: the timer stays on while the bot asks for it, even if you stop yours.

The bot reads its own timer state after every request and gives each decision a time budget (60 s at most for Team Preview, 20 s at most for a battle decision, with a 3 s safety margin; `A2_BUDGET_CONFIG` in `src/deadline.ts`). Its policy today is the shared first-legal-choice rule, so it answers at once. A request with no timer state (for example one re-sent after a reconnect) gets a zero budget and is answered at once; a request whose snapshot does not come within a second is answered by the deadline watchdog (see section 9 for the Open Team Sheets decline).

## 6. Windows client with the server in WSL2

The server listens on `127.0.0.1` inside WSL2 and the launcher cannot bind another address. WSL2 normally forwards `localhost` from Windows into the distribution, so `http://127.0.0.1:<port>` in a Windows client may just work. ADR-005 records that WSL2 in NAT mode works through localhost forwarding; that is spike evidence from the owner's human-versus-bot match (spike #677, with the spike bot, the official client, the timer on and Open Team Sheets declined, gate 1 Met 2026-10-18). It predates this runtime and its launcher, and has not been repeated with the current CLI and a Windows client. What was checked here: from inside WSL2, `curl http://127.0.0.1:<port>/` answers 200 and the port is listening on `127.0.0.1` only. If a Windows client cannot reach it, run the client inside WSL2 too. Do not work around it by exposing the server on another interface: that is outside this setup and unsupported.

## 7. What the bot records

Each battle, finished or not, is stored as one JSON record: the bot's own player channel (what it was allowed to see) and its decisions with their deadlines. Records go to `private-data/a2` (directory mode 0700, files 0600, named `<battle-room>-<epoch ms>.json`). `private-data/` is gitignored.

**A sparring record is private.** It holds your play and your challenger name, and, after an `ots_failure`, possibly your accepted team sheet. Never commit one or paste it into an issue. The program refuses an `--out` inside the checkout (the class of a record is derived from team hashes, never given; the human's team is unknown, so every sparring record is private), and a record with an `ots_failure`, a forfeited decline (`ots: forfeited`) or a fatal end (`fatal` set) is only ever written under `private-data/`, whatever `--out` says. The record itself is not stripped of a `|showteam|` line: the channel is kept as received, and this routing is the guard.

A battle that finished cleanly (the Open Team Sheets decline confirmed, no defect, no server auto-choose, a timer snapshot for every decision) is also stored as the bot's own **Episode** (ADR-008, ADR-009 slice f) as `<battle-room>-<epoch ms>.episode.json`, with a `sparring` collection record and the session id. A sparring Episode is always `private` / `owner-private`, so it goes to `private-data/a2` whatever `--out` says. A battle that spanned a reconnect is one Episode like any other: the session joins each connection's channel to the ones before it (`continueChannel`, ADR-008 §3-5), so the bot observes, decides and numbers its requests on the battle's one continuous channel, and `a2Episode` stores that channel. A rejoin that does not continue the earlier channel (it does not replay the earlier room log, a re-sent request has another `rqid`, a request is for the other seat) ends the run as a fatal error, and the record is not an Episode. A decision first made on a rejoin whose timer snapshot went to the old connection has no deadline record to carry (see the reconnect notes below), so such a battle is stored as a record only (section 9, "Unanswered request at a reconnect"). A battle that cannot be an Episode is reported on stderr (`no Episode: <reason>`) and skipped; its record is stored all the same. `a2Episode` (`src/a2-episode.ts`) is the writer: the bot never sees the opponent's full team, so an Episode of a bot-versus-bot league or evaluation battle is `private` unless the harness passes the opponent's stamped team; the opponent's class is recorded in its collection record and the opponent's team itself never is. The server removes nicknames, so the Episode's team carries the names the channel shows, so a stored synthetic A2 Episode reads back only if the reader's `known_synthetic` holds the hash of the team under those names (when they differ from the paste's).

## 8. Stopping, and leftovers

Press Ctrl-C (or send SIGTERM) in the terminal running `a2-bot`. It stops the bot, then the server's whole process group, stores any open battle's record and prints `{"event":"stopped","battles":N}`. Stopping mid-battle closes the bot's connection without a forfeit: the battle is left to the server's timer, and its unfinished record is stored.

The server also stops itself two hours after it started.

If a run was killed hard (for example with SIGKILL) and left a server behind:

1. A later start refuses with `runtime copy ... is in use by pid N`, naming the lock file `<runtime-root>/<pinned commit>.lock`. Check pid N.
2. The `in use by pid N` message names the first live process of the lock, which may be the launcher or the supervisor, not the server. The server's pid is `server_pid` in the `ready` line, or `pid` in `<runtime-root>/<pinned commit>/areneon-session.json`. Stop its group with `kill -- -<that pid>`, not N.
3. A lock whose processes no longer exist is stale and is replaced by the next start. If a start reports another process taking over the stale lock, wait; if it names a `.break` file and nothing is running, remove that file. Remove the lock file by hand only if no launcher, supervisor or server of yours is running.
4. A port already held on `127.0.0.1` makes the start fail; pick another with `--port`.

## 9. Known limits

- One battle at a time; one challenger per run.
- Reconnect: the server replays the room log to the new connection and re-sends the stored request, and the session joins the new connection's channel to the earlier ones (a rejoin that is not this battle's continuation, or is for the other seat, fails the run closed). If the bot's connection to the server closes, it reconnects at once, then every 200 ms, up to 5 times in a row without a login in between, and the server puts it back in its battle. If it never gets back, the run fails with `no connection after 5 attempts`. The server does not put a user back into a battle that ended while it was away, so after every login the bot sends `/join` for each battle it still holds as unfinished (a no-op for a room it is in already): a battle that ended meanwhile is replayed with its `|win|` or `|tie|` and no request, the session records the result from that replay, keeps the joined channel (the earlier one continued by the replay, so the end is in it), leaves the room and counts the battle as ended. If the server refuses that join (`|noinit|`, the room is gone), the battle cannot be finished and the run fails closed. The server is expected to outlive the bot's connection; it is not restarted.
- Rejoin during Team Preview. If the connection drops after the Team Preview request but before its timer snapshot, the server does not send the snapshot again, so the re-sent request has no timer state and a decision budget of 0. The Open Team Sheets decline does not take that budget: it waits for the bot's own confirmation line for `declineWaitMs` (default 5 s) from the moment it is sent, here the receipt of the re-sent request, and the battle goes on once the line arrives. The same window applies when the watchdog answers a preview whose snapshot never came. The preview choice is still held until the confirmation line, and if the line does not come within the window the bot forfeits (`ots_failure: not_confirmed`). With a snapshot nothing changes: the window is the preview `policy_budget_ms` from the request's receipt. The 5 s is not read from the timer, which is unknown without a snapshot: it is one server tick, and the turn timer keeps running during it.
- Unanswered request at a reconnect. If the connection drops while a request is still unanswered, the rejoin re-sends it but the server does not replay its `|inactive|` timer line, so the decision is first made on the rejoin with no timer snapshot (a budget of 0, answered at once). The battle plays on and its record is kept, but the decision is marked `rejoin_without_snapshot`, the session logs an `incident` event (`rejoin_without_timer_snapshot`) and no Episode is written: `a2Episode` refuses it with an `A2EpisodeError` whose `code` is `rejoin_without_timer_snapshot` (the CLI prints `no Episode: ...`). This is deliberate (ADR-008 Errata E1b): a deadline record cannot be invented, the earlier connection's snapshot would be stale, and the Episode contract is not changed for it. The battle is for sparring and analysis only, never training data. A drop after the bot has chosen, or while the opponent is choosing, is not affected: the Episode is written as for any battle.
- A fatal error (a refused choice it cannot classify, an enumerator defect, no connection after 5 attempts) forfeits the open battles, hands each unfinished battle to `onBattle` and then ends the run with exit code 1. Every such battle is stored by the usual rules (private-data only for an `ots_failure`, modes 0700/0600, room-name check). Its record has `fatal` set to the error's class name (never its message). A record can be incomplete: before the bot's first request it has no `player` and no channel, and if the channel of the connection that was open cannot be built, `fatal.channel_complete` is false and `channels` lacks that last connection.
- The bot's only play policy is the shared first-legal-choice rule. It is a sparring partner for the pipeline, not a strong player.

### Earlier evidence (spike, predates this runtime)

ADR-005 gate 1 (Met, 2026-10-18) and its evidence S11 record the owner's human-versus-bot match in the official client against the spike bot (#677): a Bo1 Reg M-C battle with the timer on and Open Team Sheets declined, 5 turns, the human won, 8 bot decisions with no fallback. That bot was not this runtime, and the proxy-lock popup was not recorded then.

### Not verified

Everything above the client was run: the launch, the `ready` line, the proxy-lock popup on the bot's own connection, Ctrl-C stopping the server, and (in the tests) a full Bo1 Reg M-C battle with the OTS decline. No human client was used. The client-side details (what the challenge dialog and the sheet prompt look like, the popup on a human connection, Windows reaching WSL2 with the current CLI, a Bo3 challenge being refused) are from the server code, the ADR and ticket notes, not from a real client of this runtime.
