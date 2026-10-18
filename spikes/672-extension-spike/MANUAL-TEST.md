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
   bottom-right. Record `client` (expect `ok`) and the socket `transport`
   (`sockjs` or `raw`) and `status` (expect `showdown`).
4. Challenge the second account to a private **[Gen 9 Champions] VGC 2026
   Reg M-C** battle. Both click **Deny Open Team Sheets**.
5. At Team Preview, record for the battle room: `role` (`player`),
   `current.phase` (`preview`), `current.rqid`, `current.request_seq`,
   `cts` (`no showteam seen`), and `opponent_preview` (species, level and
   gender only; nothing else).
6. Turn 1: pick moves, then click **Cancel** (undo) before the opponent
   moves. Record whether `current.rqid` and `request_seq` changed (expect no
   change). Pick moves again.
7. Turn 2, before choosing: open DevTools, Network tab, set **Offline** for
   about 10 seconds, then back to **No throttling** so the client reconnects.
   Record the new history entry: expect `event: resent`, the same `rqid` and
   `request_seq`. Also try one **page reload** and record `request_seq`
   (expected to restart at 1; finding 2 in README).
8. Finish or forfeit. Then open any public battle from **Watch a battle** and
   record that the room shows `rejected: spectator` with `current: null`.

## Beta client

9. Open the beta (preact) client. Write down its exact URL. If the panel does
   not appear, record that: the manifest only matches
   `*.pokemonshowdown.com`, so a beta on another host is out of scope for this
   build. Otherwise repeat steps 3 to 8.

## Report back (journal on #672)

For each client: URL, Chrome version, `client`, `transport`, and for steps 5
to 8 the panel values (phase, `rqid`, `request_seq`, `event`, the first 8
hex of `request_hash`, `cts`, `role`). A screenshot of the panel with your
username redacted is enough. Note anything unexpected, especially a panel
that stays empty (possible non-WebSocket transport) or `fail-closed`.

Remove the extension afterwards.
