# Spike #677: how a human reaches the bot (A2 vs D, own client vs E)

> **Spike code, lighter gates.** Throwaway. It answers one question for
> ADR-005 (#697) and is promoted into `packages/` or deleted afterwards.
> The bot plays a **legal heuristic: an integration fixture, not AI.**

**Question.** How does a human actually challenge the Areneon bot? (A2) a
local Showdown server plus a normal client pointed at it, or (D) a bot
account on the public server that accepts private challenges only. Also:
our own TS bot client, or poke-env (E)?

**Answer (short).** A2 works end to end on the pinned server with a
TS client of about 240 lines that reuses `@areneon/showdown-bridge`. The upstream
public client can be pointed at the local server without copying any client
code. D has no published written policy that permits or forbids it and
cannot run the pinned engine, so it is not recommended. E is viable but
duplicates the bridge in Python and has no Champions-specific data. The owner
then played a full human match against the bot in the official client
(evidence below).

## What was built

| File | Role |
|---|---|
| `src/server.ts` | Starts the pinned, **unmodified** server on `127.0.0.1:8000` with `--no-security`. Writes only upstream's gitignored `config/config.js`: localhost bind, `backdoor=false`, no dev console, no REPL, `forcetimer=true`. |
| `src/bot.ts` | Bot client over Node's built-in `WebSocket`. Refuses any non-local host. Logs in with `/trn`, sets the team with `/utm`, accepts only an **exact** `gen9championsvgc2026regmc` challenge (custom rules such as Bo3 or a removed timer are rejected), turns the timer on, sends `/rejectopenteamsheets` before its preview choice, answers every `|request|` with `/choose …|rqid`, and on `[Invalid choice]` falls back to `/choose default` and logs it. Requests are decoded by the bridge's `decodeRequest` (same `request_hash` as runtime A1). |
| `src/heuristic.ts` | First plausible choice from the bot's own request: bring the first four, first non-disabled non-ally move with foe slot 1 as target, first healthy bench Pokémon on a forced switch. The engine validates; Areneon decides nothing about legality. |
| `src/run-match.ts` | Bot vs bot: server + two bot clients, private challenge, writes `evidence/bot-vs-bot.*`, stops the server. |
| `src/team.ts`, `src/evidence.ts` | Loads the synthetic team; writes the summary, decisions JSONL and player-view log. |
| `src/human-match.ts` | Server + one bot waiting for a human challenge. Writes timing evidence to `evidence/human-vs-bot.*` and the bot's player-view log to gitignored `private-data/677-human-match/`. |

Teams: both bots use the repo's synthetic CC0 golden team
(`fixtures/showdown/gen9championsvgc2026regmc/synthetic-team.json`). No real
team is used or committed.

## Evidence: bot vs bot on the local server (A2)

`node spikes/677-bot-spike/src/run-match.ts` (Showdown `7244fe8`, ruleset hash
`4dd1f039…`, run 2026-10-18, WSL2, Node 24.11):

| Check | Result |
|---|---|
| Websocket login without a login-server assertion | Both bots named on connect (`/trn Name,0,`) |
| Private challenge → battle | `/challenge AreneonBotA, gen9championsvgc2026regmc` → `/accept` → Bo1 room `battle-gen9championsvgc2026regmc-N` |
| OTS declined by every actor | `AreneonBotA rejected open team sheets.` and `AreneonBotB rejected open team sheets.` in the room log |
| VGC Timer on | `Battle timer is ON`; every decision has a `Time left` line |
| Battle completes | 6 turns, `|win|`, 1.16 s wall clock |
| Engine rejections | 0 fallbacks across 18 decisions (preview, move, forced switch, fainted-slot `pass`) |

Per-decision timing (`evidence/bot-vs-bot.decisions.jsonl`, request receipt to
`/choose` sent, both bots):

| | Bot decision time | Server timer at the decision |
|---|---|---|
| Team Preview | 0.16–0.47 ms | 90 s this turn, 420 s bank, 90 s grace |
| Turns | p50 0.14 ms, max 0.47 ms over 18 decisions | 55 s this turn, 420 s bank on every turn |

Four further runs: 6–7 turns, 0 fallbacks, max decision 0.66 ms. The
heuristic uses < 0.002 % of the 55 s per-turn cap. Showdown's timer ticks in
5 s steps (`TICK_TIME = 5` in `server/room-battle.ts`), so the bank never moved.
That is the expected result for a heuristic. Real deadline pressure appears
only with a learned policy (#678 owns the timer-derived deadline).

The committed player-view log is bot A's own view (own request, public log).
The omniscient stream is never recorded.

## Human path (A2): verified as far as possible without a human

- `http://127.0.0.1:8000/` serves upstream's `server/static/index.html`, which
  redirects the browser to `https://localhost.psim.us`.
- `https://localhost.psim.us/` returns 200 and loads the official client from
  `play.pokemonshowdown.com`. Smogon's resolver
  `play.pokemonshowdown.com/crossdomain.php?host=localhost.psim.us` returns
  `{"host":"localhost","https":false,"id":"localhost","port":8000}`, so that
  client opens its socket to **your** `localhost:8000`. Battle traffic stays
  local; only the client's static files and the name assertion come from
  Smogon.
- `GET /showdown/info` answers `{"websocket":true,…}` and the bots' websocket
  login works, so the socket endpoint the browser uses is live.
- Upstream also documents a fully self-hosted path: build the official client
  (AGPL-3.0) **outside this repo**, unmodified, and open
  `testclient-new.html?~~localhost:8000` (client README, "Other servers").

## Evidence: a human vs the bot (A2)

The owner played one private Bo1 on 2026-10-18 with the steps below
(`evidence/human-vs-bot.{summary.json,decisions.jsonl}`, Showdown `7244fe8`,
format rules hash `549cd2aa…`). The browser ran on Windows and the server in
WSL2 in NAT mode; WSL's localhost forwarding was enough, with no mirrored
networking and no bind beyond `127.0.0.1`.

| Check | Result |
|---|---|
| Human reaches the bot | Official client via `http://localhost:8000` → `localhost.psim.us`, unregistered throwaway name, `/challenge` accepted |
| Battle completes | 5 turns, the human won, 1291 s wall clock |
| OTS declined | `AreneonBot rejected open team sheets.`; no `showteam` in the bot's view |
| VGC Timer on | `timer_on: true`; all 8 bot decisions carry the timer state |
| Bot decisions | 8 (preview, move, forced switch, fainted-slot `pass`), 0 fallbacks, max 0.62 ms against a 55 s turn cap |

Only the bot appears in `ots_rejected_by`, and that is the server's rule, not
a gap. `/rejectopenteamsheets` clears the OTS prompt for **every** player
(`server/chat-commands/core.ts`, `rejectopenteamsheets`). Sheets are shown only
when every player accepts. Once the bot has denied, the human has no button to
press and the sheets stay closed.

### Owner: step by step for the human match

1. On `main`, run `make setup` (it builds the engine and installs the
   workspace packages; `make showdown` alone does not install them).
2. Start server and bot: `node spikes/677-bot-spike/src/human-match.ts`.
   Wait for `Ready.`
3. In a desktop browser **on the same machine**, open `http://localhost:8000`.
   It redirects to `https://localhost.psim.us`, the official client.
   (WSL2: NAT mode's localhost forwarding worked. If it does not, enable
   `networkingMode=mirrored` in `.wslconfig` or use a browser inside WSL.
   Do not bind the server to `0.0.0.0`.)
4. Click **Choose name** and pick a throwaway name that nobody has
   registered; no account or password is needed. It appears in the evidence
   summary.
5. Teambuilder → new team in **[Gen 9 Champions] VGC 2026 Reg M-C**. Your
   own team stays in your browser; it is never written to the repo.
6. In the lobby type `/challenge AreneonBot, gen9championsvgc2026regmc`. Leave
   Best-of-3 and custom rules off: the bot rejects them. It accepts at once.
7. At Team Preview, bring 4. The bot has already denied Open Team Sheets,
   which removes the prompt for you as well.
8. Play to the end. The timer is on (55 s per turn, 420 s bank).
9. The script prints the summary and stops the server. Check
   `spikes/677-bot-spike/evidence/human-vs-bot.summary.json`: `AreneonBot` in
   `ots_rejected_by`, `timer_on: true`, `fallbacks` = 0. Post it on #677.
   Commit only the two `evidence/human-vs-bot.*` files.
   `private-data/677-human-match/` holds your revealed sets and stays local.

If step 3 fails, that is the blocker to record on #677 (with the browser
console error). Then try the self-hosted client above.

## Option D: bot account on the public server

Primary sources checked 2026-10-18. No bot was connected to the public server.

- **Published rules** (`https://pokemonshowdown.com/rules`): no rule
  mentions bots or automation. Rule 4 forbids gaming the ladder
  ("intentionally losing against yourself or a friend in a ladder match").
- **Server code** (pinned `7244fe8`, `server/chat-plugins/helptickets.ts`):
  staff run a bot registration through `/helpticket` ("If you are a bot
  developer, and would like to register your bot on the ladder, please fill
  out the form below"). Staff confirm with `/setbot owner,bot`
  (`server/chat-commands/admin.ts`, needs `rangeban`). Login type `10` marks
  the user `isUserBot`. `server/ladders.ts` blocks bots from suspect-test
  queues, and ladder ratings record `isBot` (`server/room-battle.ts`).
- **Staff statements** (Smogon thread "Bots are now capable of suspect reqs,
  do we care?", `https://www.smogon.com/forums/threads/bots-are-now-capable-of-suspect-reqs-do-we-care.3787411/`,
  25–26 Aug 2026): botting suspect requirements "counts as cheating", and
  community leaders are discussing new rules against ladder bots. Nothing
  there forbids a registered bot that only accepts private challenges, but
  policy is moving.

Assessment: D is not forbidden by any published rule. It sits on an
unwritten, staff-mediated registration process that is under active
discussion. Separately, D **cannot satisfy ADR invariants**: the public server
runs upstream master, not the pinned engine (the legality authority), so the
version manifest is not reproducible. It also depends on the public login
server and service availability. D is not recommended for Wave 1. If revisited,
register the bot through `/helpticket` first and never queue on the ladder.

## Option E: poke-env vs own TS client

Checked 2026-10-18 from PyPI JSON and a shallow clone of
`github.com/hsahovic/poke-env` at `7ff6212` (2026-09-18):

- Licence MIT; latest release **0.16.1 (2026-08-23)**; Python ≥ 3.10.
- Doubles: yes (`DoubleBattle`, `DoublesEnv`). VGC Team Preview is supported.
- OTS: for any Bo1 format whose ID contains `vgc` it sends
  `/rejectopenteamsheets` by default (`accept_open_team_sheet=False`). The
  timer can be turned on with `start_timer_on_battle_start`.
- Champions: **no Champions-specific handling or data**. Static dex data is
  gen1–gen9 only, and there is no `champions` string in the source. Its
  client-side battle tracking may diverge on Champions mechanics (Stat Points,
  format mod). Unverified at runtime; it was not installed or run.
- It recommends exactly the local `--no-security` server used here.

Own TS client: proven above in about 240 lines (`bot.ts` + `heuristic.ts`), with no new dependency
(Node's built-in `WebSocket`). It reuses the bridge's `decodeRequest` and
`request_hash`, so A1 and A2 share **one** request decoder. That is what the
ADR-005 gate "A1 and A2 observations equal apart from `rqid`/timer" needs. E
would add a second, Python protocol parser and battle-state model to keep in
step with the pin.

## Recommendation for ADR-005

1. **A2** for human sparring: pinned server on localhost, human uses the
   official client via `http://localhost:8000` (or a self-built unmodified
   client). A human completed a match this way.
2. **Own TS bot client** on the bridge, not poke-env. Keep E as a reference
   for protocol edge cases only.
3. **D** rejected for Wave 1: it is outside the pinned engine, and the bot
   policy is unwritten and changing.

Promotion notes if A2 is accepted: move the client into a package with tests,
add reconnect and `|error|` handling beyond `default`, make decisions async
with a deadline from the live timer (#678), and stop writing into the
vendor `config/` (pass a config through an upstream-supported path or a
wrapper checkout).

## Run

```bash
make showdown
node spikes/677-bot-spike/src/run-match.ts      # bot vs bot, rewrites evidence/
node spikes/677-bot-spike/src/human-match.ts    # waits for a human challenge
node spikes/677-bot-spike/src/server.ts         # server only (Ctrl-C stops it)
```

Gates for this spike: `pnpm exec eslint spikes/` and
`pnpm exec tsc --noEmit -p spikes/677-bot-spike`. There are no unit tests
(spike).
