# Spike #672: read the player's own Showdown request (EXP-01 / EXP-08)

**Throwaway spike** (lighter gates, AGENTS.md). Decision evidence for
ADR-004 *Player-visible observation source* (Proposed, ADR issue #696). It is
promoted into `packages/showdown-extension/` or deleted once ADR-004 is
decided. Nothing here is the advisor.

**Question:** what is the least-privileged way for a Chromium MV3 extension
to read the player's **own** `|request|` (phase, `request_seq`,
`request_hash`, `rqid`)? Option B2 (MAIN-world WebSocket/SockJS frame
wrapper) against B1 (hook on client internals) and C (DOM parsing).

**Answer (local evidence):** B2 works. On a local pinned Showdown server it
reconstructs every own request of the golden battle, byte-identical by
`request_hash` to the headless path, over raw WebSocket and SockJS framing,
across undo and socket reconnect. It rejects spectator rooms and fails
closed on unrecognised input. The live and beta clients still need the
owner's browser ([MANUAL-TEST.md](MANUAL-TEST.md)).

## Layout

| Path | Role |
|---|---|
| `extension/manifest.json` | MV3, no permissions, content scripts on `https://play.pokemonshowdown.com/*`, `https://*.pokemonshowdown.com/*` and `http://localhost/*`, `http://127.0.0.1/*` only |
| `extension/main-world.js` | MAIN world. Subclasses `WebSocket` to add one `message` listener, unframes raw or SockJS frames, keeps only whitelisted lines (never `|challstr|`) and sends them over a private `MessageChannel` port. Never wraps `send`; nothing flows back |
| `extension/protocol.js` | ISOLATED world, pure logic: envelope schema, socket-host allowlist, line whitelist re-check, own-request tracking, `request_hash` (same canonical JSON and SHA-256 as `packages/showdown-bridge/src/request-identity.ts`) |
| `extension/isolated.js` | ISOLATED world: takes the port with a capture listener registered before page script, feeds the tracker, renders the read-only panel. No submit path |
| `harness/server.cjs` | Boots the unmodified pinned server (`vendor/pokemon-showdown`, MIT) in-process on 127.0.0.1 with in-memory config, plus a localhost control endpoint that starts a battle with a fixed seed |
| `harness/client.html` | Our own minimal stand-in client page (not the AGPL client): opens a raw or SockJS-framed WebSocket |
| `test/e2e.mjs` | Playwright/Chromium with the unpacked extension; p1 is the page, p2 a Node WebSocket client |
| `test/protocol.test.mjs` | Unit checks for the wrapper's frame filter (including that `|challstr|` is never forwarded) and `protocol.js` (hash parity with the golden fixtures, every fail-closed case) |
| `evidence/e2e-result.json` | Output of the last e2e run (identities only, synthetic team) |

Run (after `make showdown`):

```bash
cd spikes/672-extension-spike
pnpm install --ignore-workspace
node node_modules/playwright-core/cli.js install chromium
node --test test/protocol.test.mjs
node test/e2e.mjs   # starts and stops its own server and headless Chromium
```

The e2e uses Playwright's Chromium (`playwright-core` 1.56.1, Chromium
141, build 1194) in new-headless mode, which loads unpacked extensions. It never
contacts a public server; the login server is set to an unreachable address.

## Evidence per ADR-004 gate

| Gate | Local (pinned server, Chromium) | Live client | Beta client |
|---|---|---|---|
| Complete own request across reconnect and undo | **Pass** | Needs owner | Needs owner |
| Spectator rooms rejected; fail closed on an unrecognised client | **Pass** | Needs owner | Needs owner |
| EXP-08: browser and headless equal by `request_hash` on golden fixtures | **Pass** | n/a (no seed control) | n/a |
| OTS-declined preview holds no hidden set data | **Pass** | Needs owner | Needs owner |

Details (from `evidence/e2e-result.json`, identical for raw and SockJS):

- **EXP-08.** The golden battle (format `gen9championsvgc2026regmc`, seed
  `sodium,000102030405060708090a0b0c0d0e0f`, synthetic team, engine
  `default` choices, both players declining OTS) gives p1 10 requests. The
  panel's `(request_seq, phase, request_hash)` list equals the headless list
  exactly, including the committed fixtures: preview `1569598e…`, move
  `e5b62f40…`, forceSwitch `75e462b6…`. The unit test also re-hashes the three
  committed fixtures with the extension's hash (with an injected `rqid`) and
  gets the committed `request_hash`.
- **Undo.** The server sends **no** new `|request|` on undo. `rqid` stays 4,
  `request_seq` stays 2, and the reconstructed request is unchanged. The
  re-sent choice tagged with the same `rqid` is accepted: no `|error|`, and
  the turn advances to `rqid` 6 (asserted).
- **Reconnect** (close the socket, open a new one, `/trn`, `/join`). The
  server re-sends the **same** request with the **same** `rqid` (8 → 8). The
  tracker records it as `resent`, keeps `request_seq` 4 and the same hash,
  and the next request continues at 5.
- **rqid semantics.** `rqid` is one counter per battle room shared by both
  sides, incremented per request, so p1 sees 2, 4, 6, … Gaps are normal.
  A choice tagged with an old `rqid` (4, while 12 is current) gets
  `|error|[Invalid choice] Sorry, too late to make a different move; the next
  turn has already started`. The tracker rejects a lower `rqid` as stale and
  fails the room closed if the same `rqid` arrives with a different hash.
- **Spectator.** The page joins a battle between two other users. It
  receives 10 battle frames and no `|request|`; the room shows
  `rejected: spectator` with no observation. A request whose `side.name` is
  not the logged-in user is dropped (`request-not-for-current-user`).
- **Fail closed.** In the e2e a hostile page script forges envelopes on the
  window bus, injects its own port, posts `ready` to ask for a new port and
  patches `MessagePort.prototype.postMessage`: no forged room appears, no
  port is offered to the page, the spy sees none of the wrapper's
  envelopes, and the real socket still reaches `showdown`. Unit tests cover
  the tracker: malformed envelopes, extra keys, non-allowlisted socket hosts
  and non-whitelisted lines are dropped and counted; a non-Showdown frame
  marks its socket `unrecognised`; an unknown request shape, a request
  without `rqid`, a request before the Showdown handshake and a seat
  mismatch each mark the room `fail-closed` with no observation; an unknown
  wrapper version sets `client: unrecognised` and withholds every
  observation.
- **No new exposure** (asserted). The extension forwarded only
  `updateuser init player poke teampreview request turn win`; never
  `challstr`. The client page's own `message` listener saw one extension
  message (`ready`, no data) and no port, `challstr` or `|request|`.
- **OTS-declined preview.** Before turn 1, p1's non-request lines are only
  `init title j t: gametype player gen tier rule clearpoke poke teampreview
  uhtml uhtmlchange teamsize start switch`. There is no `|showteam|`.
  Opponent lines are `|poke|p2|Species, L50, G|` only. None of the
  opponent's items, abilities or moves appears in any public line. The panel
  shows `cts: no showteam seen` and only those six species strings.

## Findings for the contracts (#671)

1. **`request_hash` covers `side.name`.** The request embeds the player's
   username, so EXP-08 equality holds only when names match (the e2e names
   the users `p1`/`p2`, as the headless runner does). Decide whether the
   contract hashes the request as-is or normalises `side.name`.
2. **Browser `request_seq` is page-session scoped.** The tracker counts
   distinct `rqid`s in memory. A socket reconnect keeps it; a **page reload**
   restarts it at 1 while `rqid` and `request_hash` continue. Either persist
   per-room state (for example `chrome.storage.session` keyed by room id)
   or treat `request_hash` + `rqid` as the browser-side identity.
3. **Undo has no protocol signal.** Incoming frames do not show it; the
   outgoing `/undo` is not observed by design (B2 never wraps `send`). The
   advisor must keep the current request valid until a higher `rqid`
   arrives.

## B1 vs B2 vs C

| | B1: hook client internals | **B2: wrap WebSocket/SockJS frames** | C: parse the DOM |
|---|---|---|---|
| Built here | No (analysis only) | Yes, tested end to end | No (analysis only) |
| Data | Parsed request objects | Raw `|request|` JSON, byte-exact; `rqid` included | Rendered buttons and text; no `rqid`, no full request JSON |
| Identity gates | Possible | Met locally | Cannot compute `request_hash` or see `rqid`: fails the gates |
| Client refactor (preact beta) | Breaks: depends on undocumented internals of the AGPL client | Independent of client code; depends only on the documented protocol and transport | Breaks on markup changes |
| Licence risk | High: hooking requires studying and mirroring AGPL internals | None: written from `PROTOCOL.md` | Low |
| Forgeable by page script | Yes | Yes (MAIN-world messages) | Yes (DOM) |
| Failure modes | Silent mismatch on refactor | Non-WebSocket SockJS fallback (XHR) or a socket opened before `document_start` gives no frames: shows nothing | Lossy, partial |

## Security notes

**What the requirement can mean.** The page created the socket, so page
script already receives every frame, including `|challstr|` and the full own
`|request|`, which carries the player's own team. No MAIN-world design can
hide that from page script. The criterion "no auth tokens or team data
exposed to page script" can therefore only mean **the extension adds no new
exposure**: it puts nothing into the MAIN world that the page did not already
have, and it does not rebroadcast socket data on a channel other scripts can
read more easily. (An earlier revision of this spike broke the second part by
posting every raw frame, `|challstr|` included, on `window.postMessage`.)

How the spike meets it:

- **One direction.** Data flows MAIN to ISOLATED only. The isolated world
  sends the MAIN world just `ready` and `ack`, with no data. No token, team or
  reply ever enters the MAIN world. The wrapper never wraps `send`, so
  outgoing `/trn` assertions and `/utm` team uploads never pass through it.
- **Whitelist before anything leaves the wrapper.** It forwards
  `|updateuser|NAME` (settings JSON dropped) and, in battle rooms only,
  `init`, `player` (side and name only), `poke`, `showteam` (side id only, the
  packed team dropped), `request`, `teampreview`, `turn`, `win` and `tie`.
  `|challstr|`, `|nametaken|`, query responses, chat, PMs and every other
  line are dropped. The isolated world re-checks each line against the same
  whitelist and rejects the whole envelope otherwise.
- **What still crosses worlds, and why.** The full own `|request|` JSON,
  including the player's own moves, items, abilities and stats. It is needed
  to compute `request_hash`, and it is the input the advisor needs. It is the
  player's own data, which the page already holds. It holds no opponent
  hidden information: the server sends each player only their own side.
  The player's display name also crosses, for the own-player check.
- **Channel.** A `MessageChannel` port, not `window.postMessage`. The wrapper
  offers a port at `document_start`. The isolated world takes it in a
  capture-phase listener registered at `document_start`, before any page
  script, and stops the event from reaching page listeners. The wrapper uses
  only the first acknowledged port and never offers another. It captures
  `Reflect.apply`, `MessagePort.prototype.postMessage`, `addEventListener`
  and `window.postMessage` at `document_start`, so prototype patching later
  does not intercept the port traffic.
- **Not truly private. No MAIN-to-ISOLATED channel is.** The MAIN world shares
  one JavaScript realm with the page. The port's protection depends on our
  scripts running at `document_start` before page script. Chrome documents
  that for `document_start`, but not the order between our MAIN and ISOLATED
  scripts; the `ready`/offer handshake covers both orders. At runtime the
  wrapper still uses ordinary built-ins (`String.prototype.split`,
  `JSON.parse`, `Array` methods). A page that patches them can observe what
  the wrapper handles, which is a subset of what the page already reads
  from its own socket. The port stops passive listeners on the window bus;
  it cannot stop a page that actively tampers with its own realm. A channel
  that is private from page script needs no MAIN-world code at all, for
  example `chrome.debugger` (CDP `Network.webSocketFrameReceived`) from the
  service worker. That costs the `debugger` permission and a visible
  warning bar.
- **Forgery.** For the same reason, a hostile page script can construct a
  socket through our wrapper and dispatch synthetic frames (including
  `|updateuser|`) that B2 cannot tell from real traffic. The impact is a
  wrong display only, because there is no submit path. Forged envelopes on
  the window bus, and forged ports, are ignored.
- **Untrusted input.** The isolated world checks the exact key set, types,
  the line whitelist, a 1 MiB cap per envelope, `ws:`/`wss:` and the socket
  host allowlist (`sim3.psim.us`, `localhost`, `127.0.0.1`).
- **Panel.** The spike panel lives in the page DOM (open shadow root), so the
  page can read it. It shows only own request identity (hashes, phase,
  `rqid`) and public preview species, all of which the page already has. The
  real advisor should render in `chrome.sidePanel`.
- **No auto-submit.** Nothing in the extension writes to the socket or the
  page.

## Recommendation

Adopt **B2** as the extension observation source in ADR-004, with C only as
a non-authoritative fallback indicator and B1 rejected (fragile against the
preact client, AGPL exposure). Accepting ADR-004 should still wait for the
owner's live and beta runs in [MANUAL-TEST.md](MANUAL-TEST.md), and the
three contract findings above should be settled in #671.
