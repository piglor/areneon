# Manual test: live and beta Showdown clients (owner, ~20 min)

The automated e2e covers a local server only. These steps need your own
browser and accounts. Use a **throwaway, non-competitive team**; never your
real team. Redact usernames before posting anything.

## Setup

1. Chrome or Chromium: open `chrome://extensions`, enable **Developer mode**,
   click **Load unpacked**, pick `spikes/672-extension-spike/extension`.
2. Have a second account in a second browser profile **without** the
   extension, to be the opponent.

## Live client (`https://play.pokemonshowdown.com`)

3. Open the client and log in. A panel titled *Areneon EXP-01 probe* appears
   bottom-right. Record `client` (expect `ok`), the socket `transport`
   (`sockjs` or `raw`), its `status` (expect `showdown`), and the **socket
   host** from `sockets[].url` (expect `sim3.psim.us`). A host outside
   `sim3.psim.us`, `localhost` and `127.0.0.1` is dropped by design, so the
   panel stays empty; record the host if that happens.
4. Challenge the second account to a private **[Gen 9 Champions] VGC 2026
   Reg M-C** battle. Both click **Deny Open Team Sheets**.
5. At Team Preview, record for the battle room: `role` (`player`),
   `current.phase` (`preview`), `current.rqid`, `current.request_seq`,
   `cts` (`no showteam seen`), and `opponent_preview` (species, level and
   gender only; nothing else).
6. Turn 1: pick moves, then click **Cancel** (undo) before the opponent
   moves. Record whether `current.rqid` and `request_seq` changed (expect no
   change). Pick moves again.
7. Turn 2, before choosing, force a socket reconnect: open DevTools, Network
   tab, set **Offline** for about 10 seconds, then back to **No throttling**.
   Record the new history entry: expect `event: resent`, the same `rqid` and
   `request_seq`. **Fallback** if the client does not reconnect (Offline does
   not always drop an open WebSocket): turn the computer's network off for
   about 20 seconds (Wi-Fi off or cable out), then back on, and wait for the
   client to reconnect. Record which method worked.
8. Turn 3, before choosing: **reload the tab** (F5) and return to the battle.
   Record `current.rqid`, `request_seq` and `event`. `rqid` and
   `request_hash` should match the request before the reload. `request_seq`
   is expected to restart at 1, because the spike keeps no state across
   reloads (finding 2 in README).
9. Finish or forfeit. Then open any public battle from **Watch a battle** and
   record that the room shows `rejected: spectator` with `current: null`.

## Beta client

10. Open the beta (preact) client. Write down its exact URL. If the panel does
   not appear, record that: the manifest only matches
   `*.pokemonshowdown.com`, so a beta on another host is out of scope for this
   build. Otherwise repeat steps 3 to 9.

## Report back (journal on #672)

For each client: URL, Chrome version, socket host, `client`, `transport`,
the reconnect method used, and for steps 5 to 9 the panel values (phase, `rqid`, `request_seq`, `event`, the first 8
hex of `request_hash`, `cts`, `role`). A screenshot of the panel with your
username redacted is enough. Note anything unexpected, especially a panel
that stays empty (possible non-WebSocket transport) or `fail-closed`.

Remove the extension afterwards.
