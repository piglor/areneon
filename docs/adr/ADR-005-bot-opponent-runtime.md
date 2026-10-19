# ADR-005: Bot opponent runtime

**Status:** Accepted, 2026-10-19 | **Wave:** 1 (A1, A2) | **Deciders:** core team | **Date:** 2026-10-19

ADR issue: #697 · Epic: #730 (slices #731–#735) · Evidence: #677 (human reaches bot), #678 (timer-derived deadlines) · Related: [ADR-003](ADR-003-pinned-showdown-reference-simulator.md), [ADR-006](ADR-006-legal-joint-choice-enumeration.md), [ADR-008](ADR-008-cross-language-data-contracts.md)

**Revision history:**
- **r3.2 (2026-10-18):** the r3.1 delta review approved; this revision folds in its two P2s:
  - the runtime copy's file list and working directory;
  - a shared A1/A2 actor rule;
  - when to re-send a choice;
  - the OTS bullets regrouped.
- **r3.1 (2026-10-18):** answers the r3 re-review in the #697 journal:
  - the client defaults this format to Best-of-3;
  - server-layer refusals are not defects;
  - follow-up requests inherit the deadline;
  - how the runtime copy is made;
  - smaller corrections.
- **r3 (2026-10-18):** decides from the spike evidence and the owner's human match. It answers the r2 plan-stage review in the #697 journal:
  - sources and labels;
  - A2's behaviour under ADR-006;
  - the `request_seq` rule;
  - the `--no-security` risk;
  - confirming the OTS decline;
  - the boundary diagram;
  - moving gate 3 to an Epic ticket;
  - the rollout.
- **r2 (2026-10-17):** split A1 from A2. It listed the options and left the choice to the spikes.
- **r1 (2026-10-17):** rejected at the plan-stage review in the #697 journal. It merged two runtimes and never said how a human reaches the bot.

---

## Context

The goal is a sparring opponent that a human can challenge again and again in **Bo1 with closed team sheets** (CTS). Areneon needs two runtimes:

| Runtime | Use | `rqid` / timer / OTS | Engine |
|---|---|---|---|
| **A1.** In-process sim `BattleStream` | Training, evaluation, fixtures | No / No / No | Pinned `sim/`, seeded, deterministic. `request_seq` is assigned by Areneon. |
| **A2.** Bot client on a local Showdown server | Human sparring | Yes / Yes / Yes | The pinned, unmodified `server/` on `127.0.0.1`. |

Facts about the pinned engine (`7244fe8`):
- The M-C format carries `VGC Timer` and `Open Team Sheets` [S3].
  - OTS is opt-in.
  - The server shows the sheets only when **every** player accepts.
  - A rejection clears the prompt for every player [S2].
- It also sets `bestOfDefault: true`, which the server sends to the client as a display flag [S3]. The client then ticks Best-of-3 by default; that is client behaviour, observed in the human match [S8]. A Bo1 challenge needs that box unticked.
- The VGC Timer gives 90 s at preview, a 420 s bank and at most 55 s per turn.
  - A turn timeout with bank left makes the server auto-choose `default`.
  - A timeout with the bank at 0 forfeits the game [S1][S9].
- `rqid` exists only in the server layer [S1].
- `--no-security` turns on `noguestsecurity`, and the server then accepts any untrusted name without a token [S4].
- The server reads its config only from `config/config.js` under its own install root. That path is resolved from the code's location (`__dirname`), not from the working directory [S4].
- The server sends each connection only its own player's channel and request [S5].

## Decision drivers

1. **Bo1 CTS fidelity.** The human plays the real format, with the timer on and team sheets closed. The bot never sees the human's unrevealed sets.
2. **One engine, one legality authority.** A1 and A2 use the same pinned engine (ADR-003). They also use the same request decoder and the same enumerator (ADR-006).
3. **Parity between A1 and A2.** The same seeded battle gives equal observations under ADR-008 §4.
4. **Training never depends on an external service.**
5. **No copied AGPL code.** The Showdown client is AGPL and is only ever run unmodified.

## Evidence and alternatives

| Option | What it is | Evidence | Decision |
|---|---|---|---|
| **A2.** Local pinned server, plus the official client pointed at it | The human opens `http://localhost:8000`, which redirects to `localhost.psim.us`. The client resolves that host to the local server [S7]. | Bot vs bot: 18 decisions, 0 fallbacks, timer on, OTS declined [S8]. **Human vs bot (owner, 2026-10-18):** 5 turns in the official client, the human won, timer on, OTS declined, 8 bot decisions, 0 fallbacks, at most 0.62 ms per decision [S8]. | **Chosen** for sparring |
| A2, client variant: a self-built, unmodified client | The upstream client is built **outside** this repo and opened with `testclient-new.html?~~localhost:8000` [S7]. | Documented upstream; not run | Fallback if `psim.us` is unavailable |
| **Own TS bot client** on `@areneon/showdown-bridge` | About 240 lines using Node's built-in `WebSocket`. It reuses `decodeRequest` and `request_hash` [S8]. | Proven in both matches [S8] | **Chosen** |
| poke-env (Python, MIT) | A mature client. Version 0.16.1 (2026-08-23) supports doubles and declines OTS by default [S6]. | No Champions data or handling. Never run [S6]. | **Rejected.** It would add a second protocol parser and battle-state model next to the bridge, which breaks driver 2. It is kept as a protocol reference only. |

## Decision

**A1 for training and evaluation, and A2 for sparring.** A2 is the pinned, unmodified Showdown server bound to `127.0.0.1`, plus Areneon's own TypeScript bot client in a new package on `@areneon/showdown-bridge`. The human uses the official client through `localhost:8000`, or a self-built, unmodified one.

Until a learned policy exists, the bot plays a **legal heuristic**. It is labelled as an integration fixture, not AI, and it plays only on A1 and A2.

```text
 human's client                      127.0.0.1 only                     Areneon bot client (A2)
 official client (unmodified,  <-ws-> pinned Showdown server  <-ws->  decodeRequest -> request_seq
 static files from Smogon [S7])      (unmodified, --no-security,      enumerateCandidates (judge: null)
                                      forcetimer, own runtime dir)     PlayerChannel adapter -> ObservationBuilder
                                      | per-connection channel [S5]    policy | legal heuristic
                                      v                                deadline watchdog (#678)
                              omniscient state stays in the server    evidence: own decisions only
                              (never recorded by Areneon)             player-view log -> private-data/
```

### 1. A2 under ADR-006: candidates, refusals and request identity

- **Candidates.**
  - Every choice the bot sends is a candidate from `enumerateCandidates` applied to the bot's own request. On A2 the judge is `null`: the engine judges only at submission (ADR-006).
  - The legal heuristic picks the first non-pending candidate in canonical order, as `firstLegal` does on A1.
  - The spike's hand-built target sets (`heuristic.ts`) are removed.
- **`[Unavailable choice]`.** The engine has refused a pending candidate and revealed hidden state. The updated request that follows is a **new request** (`request_seq` + 1) and the policy answers it. The bot does **not** send `default`.
- **Server-layer refusals are not defects.** The server itself answers `[Invalid choice]` in two cases, without asking the sim [S1]:
  - "There's nothing to choose": no request is open, for example after a duplicate send.
  - "Sorry, too late to make a different move": the `rqid` is stale, or every player is already waiting.

  The bot classifies refusals by its own request state. If the choice was for a request that is no longer current, or no request was open, it is dropped and logged as `stale_choice`, never as a defect.
- **A sim refusal of a current, non-pending candidate** is an `EnumeratorDefectError` incident.
  - Logged as a defect, it fails every test and harness run.
  - In a live human match only, the bot then sends the next non-pending candidate, and finally `default` with the current `rqid`, so that the human's match continues. The match is marked `defect: true` and excluded from evaluation. This is the only exception to "nothing is retried", and it applies only on A2.
- **All candidates pending.** If every candidate is pending (for example, a slot that is both `maybeDisabled` and `maybeTrapped`), the heuristic submits the first pending candidate in canonical order. An `[Unavailable choice]` then gives a new request. The gate-3 harness uses the same actor rule on A1; `firstLegal` currently aborts there instead.
- **`request_seq`.** Every `|request|` the server sends counts, **including `wait`**, exactly as on A1 (`runner.ts`). `|request|null`, which the server sends when a player leaves [S1], is not a request: it does not count and never reaches `decodeRequest`.
  - A request whose `rqid` and `request_hash` both equal the previous request's is a **re-send**, not a new request. The server re-sends the current request when a player rejoins [S1].
  - If `|sentchoice|` follows a re-send, the bot does not choose again. If it does not, and the bot had already sent a choice for that `request_seq`, it sends that choice again. If the policy is still computing, the bot carries on under the inherited deadline (§3). A duplicate is refused as "nothing to choose", which the bot drops (above).
  - `rqid` is never hashed (ADR-008 §4).

### 2. CTS: the OTS decline is confirmed, and the bot fails closed

- On its preview request the bot sends `/rejectopenteamsheets` and waits for its own `<name> rejected open team sheets.` line before it sends its preview choice.
- If that line does not arrive within the preview `policy_budget`, or if a `|showteam|` line ever arrives, the bot forfeits the match and records why. During Team Preview (`turn == 0`), with the bot a player in an OTS format and still undecided, a rejection always succeeds [S2]. A missing line therefore means something is wrong.
- Only the sim emits `|showteam|`, and only once every player has accepted [S2]. A human's voluntary `!showteam` arrives as HTML, which the adapter drops (§5), so it cannot cause a false forfeit.
- The bot accepts only an exact `gen9championsvgc2026regmc` challenge. Best-of-3 and custom rules are refused [S8].

### 3. Deadlines (#678)

- The 60 s preview and 20 s battle figures are **upper bounds**. Each decision's `submit_within` and `policy_budget` come from the live timer state [S9]:
  - `submit_within = min(hard, max(f, alloc))`
  - `policy_budget = max(0, submit_within − f)`
  - The safety margin `m` is configuration (3 s on a local server), never a constant.
- A flat 20 s per decision would run out the bank around the 21st decision. Once the bank is at or below 55 s, any turn timeout forfeits the game [S9].
- **The deadline is inherited, never restarted.**
  - The updated request that follows an `[Unavailable choice]` has `update: true`. The server neither restarts the turn timer nor sends a new time snapshot for it [S1][S9]. That request therefore keeps the original request's receipt time and `submit_within`.
  - A re-send on rejoin keeps the bot's local clock running.
- **Disconnection.** A timeout while the bot is disconnected forfeits the game; it is not an auto-choose, because the server auto-chooses only for a player it knows is active [S1]. Reconnect handling must therefore be fast.
- On an overrun the bot sends the legal heuristic choice and logs the overrun. `default` is a last resort sent by the bot. It is never the server's auto-choose, which must never fire.
- An auto-choose detector logs `server_auto_choose_detected` if the server ever chooses for the bot.

### 4. Server launcher and local security

- **Bind and exposure.** The server binds to `127.0.0.1` only, and the bot client refuses any non-local host (`assertLocal`, a tested invariant). Binding to the LAN or `0.0.0.0`, or using a tunnel, needs a new ADR.
- **Locked-down config.** The server runs with `backdoor=false`, no dev console, no REPL, and `forcetimer=true` [S8].
- **Runtime directory.**
  - The launcher never writes into `vendor/`.
  - The server resolves `config/config.js` from its install root via `__dirname` [S4]. A changed working directory or a symlink farm therefore still reads `vendor/`. The server also writes `logs/` and other runtime files into that root.
  - So the launcher makes a **real copy**:
    - it first verifies the `vendor/pokemon-showdown` checkout as `loadEngine()` does (HEAD = `PINNED_SHOWDOWN_COMMIT`, build present);
    - it copies `dist/`, `server/static/` (the `index.html` redirect the human path needs), `config/config-example.js`, `node_modules/`, `package.json` and the `pokemon-showdown` entry script into a gitignored runtime directory;
    - it starts the server with its working directory set to that runtime directory, because the static and config file servers resolve from the working directory [S4];
    - it records the commit and a SHA-256 digest of the copied `dist/` in the session manifest;
    - it writes `config/config.js` only there.
  - A copy whose digest does not match the recorded one is rebuilt.
- **Short-lived server.** The server runs only for a sparring session.
- **Challenge filter.** The bot's name carries a random suffix per session. The bot accepts a challenge only from the human name given at launch.
- **Accepted residual risk.** While the server runs, any local process or any web page open on the machine can open `ws://localhost:8000`. Under `--no-security` it can claim any unregistered name that nobody is currently using [S4]. A name in use can be taken over only while its holder is disconnected, and localhost clients all share one IP [S4]. That leaves two real windows:
  1. the bot is reconnecting, which fast reconnect narrows;
  2. someone claims the human's name before the human does.

  The challenger filter (an exact match on the sender of the challenge) blocks strangers, but it cannot stop that impersonation. Together with the short-lived, localhost-only server, this keeps the risk to a local nuisance:
  - someone could disrupt a sparring match;
  - nothing can reach other data, because the server has no backdoor and no console.

  **This risk is accepted for Wave 1.**
- **Smogon dependency.**
  - With the `psim.us` path, Smogon serves the client's static files and the `crossdomain.php` lookup, and so sees the human's IP address. The battle traffic itself stays local [S7].
  - The self-built client removes that dependency.
  - The client's replay upload is checked during promotion. If a local battle can be published, the launcher disables it.

### 5. Data, privacy and replay

- **Teams.** The bot's team is a PrivateTeamSpec with `data_class` `synthetic` or `private` (ADR-008 §5). The human's team stays on the human's side.
- **What the bot sees.** The bot sees only its own channel [S5]. The A2 PlayerChannel adapter drops the server-only lines before building an observation:
  - `|j|`, `|l|`, `|c|`, `|raw|`, `|inactive|` and `|uhtml|`/`|uhtmlchange|`;
  - plain-text notices;
  - `|t:|` and `|timestamp|`;
  - `>room` headers.

  Timer lines feed only `decision_context`.
- **Chat.** Human chat is untrusted free text and never reaches an observation.
- **Records.**
  - Committed evidence holds only the bot's own decisions and a public summary.
  - The bot's player-view log of a human match goes to gitignored `private-data/`.
  - Areneon never records the server's omniscient state on A2.
- **Replay.**
  - An A2 battle is identified by `request_seq` and `request_hash`, plus the version manifest.
  - It cannot be seeded from `/challenge`, because `seed` is reachable only from server code or `/importinputlog` [S1].
  - For parity tests, the evaluation harness takes the battle's input log from the server side, as an evaluation artifact and never as actor input, and replays it on A1.
    - It gets the log by turning on `logchallenges` (off by default) in the harness config for test runs only [S1].
    - Those logs hold **both teams**. CI uses synthetic bot-vs-bot logs only. A human match's input log is never written, or goes to `private-data/`.

## Gates

| Gate | Status | Evidence |
|---|---|---|
| 1. A human in a normal client completes one private M-C Bo1 against the heuristic bot, with OTS declined | **Met** (2026-10-18) | `spikes/677-bot-spike/evidence/human-vs-bot.summary.json` [S8]. Only the bot is listed as rejecting, because its rejection clears the prompt for both players [S2]. |
| 2. Every per-turn decision time is logged and stays under the timer | **Met for the heuristic** | 8 of 8 (human match) and 18 of 18 (bot vs bot) decisions logged with timer state, all far under the cap [S8]. The #678 watchdog is not built yet, so this gate is met trivially; enforcing the deadline is an exit criterion of the Epic. |
| 3. A1 and A2 observations of the same battle are equal under ADR-008 §4 (JCS bytes equal after removing `decision_context` and every `rqid`) | **Moved to the Epic** | It needs the ADR-008 contracts (#723, #724, #727) and an A2 adapter that does not exist yet. ADR-008 §5 activates A2 only once this ADR is Accepted, so keeping gate 3 as a precondition would be circular. It becomes the Epic ticket "A2 PlayerChannel adapter and A1/A2 equality test" below, which blocks closing the Epic. |

## Consequences and risks

| Risk | Mitigation or acceptance |
|---|---|
| `--no-security` name takeover by a local process or web page | Localhost only, short-lived server, random bot suffix, challenger filter. Residual risk **accepted** (§4). |
| `localhost.psim.us` or Smogon unavailable | Fallback: a self-built, unmodified client built outside the repo [S7]. |
| Smogon sees the human's IP on the `psim.us` path | Stated here. The self-built client avoids it. |
| A pin bump changes server behaviour (OTS, timer, request re-send) | The pin-bump PR re-runs the A2 integration test and the golden fixtures (ADR-003). |
| Enumerator defect in a live match | Logged as an incident; the match continues and is excluded from evaluation (§1). |
| A2 parity drifts from A1 | Gate 3 test in CI on recorded server input logs. |
| The learned policy is slow | #678 budgets, the watchdog, and the heuristic fallback. |

- **Compatibility.** Contracts carry `request_seq` always, plus `rqid` only when the server sent it (ADR-008).
- **Licence.**
  - The pinned server is MIT and runs unmodified.
  - The client is AGPL. It is only run unmodified (served by Smogon, or built outside the repo) and never copied.
  - The new package adds no dependency.
- **Operations.** It is one local process tree and needs no network beyond the `psim.us` static files. WSL2 in NAT mode works through localhost forwarding [S8].

## Migration and rollout

1. **Epic.** After acceptance, create "[Epic][ADR-005] Bot opponent runtime" on Wave 1, related to #697, with these subtasks:
   - (a) **Bot client package** (`packages/`, ≥ 90 % coverage):
     - candidates from the enumerator and one **shared actor rule** used on both A1 and A2: the first non-pending candidate, otherwise the first pending one (it replaces `firstLegal`'s abort when every candidate is pending, so A1 and A2 cannot drift);
     - the refusal and `request_seq` rules (§1);
     - confirmed OTS decline (§2; absorbs #709);
     - the exact-format and challenger filter;
     - `assertLocal`;
     - reconnect and `|error|` handling;
     - websocket close handling.
   - (b) **Server launcher:** the runtime directory, no `vendor/` writes, the pin check, the session lifetime, and the replay-upload check (§4). A test checks that `GET /` on the copy serves the redirect.
   - (c) **Deadlines:**
     - the #678 `submit_within` / `policy_budget` watchdog, with the margin as configuration;
     - the auto-choose detector;
     - #678 criterion 3, the `DecisionContext` fields, lands with ADR-008 slice (b) (#723).
   - (d) **A2 PlayerChannel adapter and the A1/A2 equality test (gate 3).** The adapter drops the server-only lines (§5). The test replays recorded server input logs on A1. Depends on #723, #724 and #727.
   - (e) **Human sparring guide:** the owner's steps from the spike README, moved into the package docs. It includes: **untick Best-of-3** when challenging, because the client ticks it by default for this format [S3].
2. **Spike.** `spikes/677-bot-spike/` stays until (a) and (b) land, then it is deleted. Its evidence files move to the Epic's ticket journal.
3. **No persisted data** exists beyond the spike evidence, so nothing migrates.

## Non-goals

- Choosing the learned policy and how it trains. The heuristic is an integration fixture.
- LAN or tunnel exposure of A2; Bo3.
- Choosing among legal choices, and legality itself (ADR-006, ADR-003).

## Sources

S1–S9 accessed 2026-10-18. Engine sources are at the pinned commit `7244fe8e28fa8bc4d0809ae30d183c608b62bb2b` of `github.com/smogon/pokemon-showdown` (MIT).

| Label | Source | Supports |
|---|---|---|
| [S1] | `server/room-battle.ts`: timer settings (:195-206); timeout handling (`checkActivity` :391, `checkTimeout` :439-461; auto-choose only if `knownActive` :452); `RoomBattleOptions` `seed` and `inputLog` (:473-501, :576); server-layer `[Invalid choice]` refusals (:624-636); request send, `rqid++`, and `update` requests not restarting the timer (:803-816); `logchallenges` log writer (:858); request re-send on rejoin with `\|sentchoice\|` (:954-961); `\|request\|null` on leave (:1021, :1058); `config/config-example.js:432` (`logchallenges = false`) | Seeding, the timer, refusals, re-send, input logs, `rqid` only in the server |
| [S2] | `server/chat-commands/core.ts`: `acceptopenteamsheets` / `rejectopenteamsheets` (:949-1004) | Sheets shown only if every player accepts; a rejection clears the prompt for every player |
| [S3] | `config/formats.ts:288-292` (M-C: `bestOfDefault: true`, `VGC Timer`, `Open Team Sheets`); `server/rooms.ts:1501` (`bestOfDefault` sets display flag 64: the client ticks Best-of-3 by default); `data/rulesets.ts:778-787` (VGC Timer), `:1980-2010` (OTS rules) | Format, timer and OTS rules; the Bo3 default in the client |
| [S4] | `server/config-loader.ts:34-35` (`--no-security` gives `noguestsecurity`), `:43` (`CONFIG_PATH = FS('./config/config.js')`); `lib/fs.ts:31` (`ROOT_PATH` from `__dirname`); `server/sockets.ts:341-343` and `lib/static-server.ts:108` (static and config file servers resolve from the working directory); `server/users.ts:641` (no token needed for untrusted names), `:801-806` (unregistered names merge only from the same IP while the holder is disconnected) | Name-takeover risk and the config path |
| [S5] | `server/sockets.ts:261` (`extractChannelMessages` per connection; see ADR-008) | The bot sees only its own channel |
| [S6] | poke-env on PyPI, https://pypi.org/project/poke-env/ (0.16.1, 2026-08-23), and `github.com/hsahovic/poke-env` at `7ff6212` | MIT, doubles, OTS default, no Champions data |
| [S7] | `smogon/pokemon-showdown-client` README, "Other servers"; `play.pokemonshowdown.com/crossdomain.php?host=localhost.psim.us` returns `{"host":"localhost","port":8000,…}` | How the human reaches the local server; the self-built client fallback; the Smogon dependency |
| [S8] | `spikes/677-bot-spike/README.md`, `src/` and `evidence/` (bot vs bot, human vs bot), `piglor/areneon#8` and `piglor/areneon#23` | A2 works for both bots and a human; timing; the spike's configuration |
| [S9] | `docs/research/timer-deadline-budget.md` §1.3, §1.7, §2.2-2.3 and §3 (`piglor/areneon#7`) | Deadline formulas, margin, forfeit at 0 bank, overrun policy |
