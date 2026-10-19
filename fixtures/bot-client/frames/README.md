# Protocol reader frame fixtures

Recorded server frames for the bot client's protocol reader (`packages/bot-client/src/protocol.ts`, `src/refusal.ts`). CC0 (`fixtures/LICENSE`). Synthetic: no real team, no real player, no chat.

Re-capture after a Showdown pin bump, then review the diff:

```bash
node packages/bot-client/scripts/capture-frames.ts
```

`packages/bot-client/test/frames.test.ts` re-captures both files in CI and fails on any difference. No server is started and nothing connects anywhere.

## `battle.json`: the pinned engine's own output

The choice corpus's `maybe-trapped` case (`fixtures/showdown/gen9championsvgc2026regmc/choices/maybe-trapped.json`: its seed, its synthetic teams and its choice log) is played on the pinned in-process sim (`BattleStream` + `getPlayerStreams`). p1 replays the log, then sends:

1. a pending switch (foe Mega Gengar may have Shadow Tag): the engine answers `[Unavailable choice]` and an updated request (`update: true`);
2. `move 9, move 1 1`: the engine answers `[Invalid choice]`. The tests use this real refusal to stand in for an enumerator defect, by offering the string as a non-pending candidate; the real enumerator never offers it;
3. the shared actor rule (`actorRule`) from then on, and `default` once while it holds a `wait` request: the sim answers `[Invalid choice] Can't do anything: It's not your turn`.

p2 replays the log, then plays the actor rule. The battle runs to its end. `sent` lists every choice p1 sent and how many frames it had read before.

Each chunk of p1's stream becomes one frame, as `server/room-battle.ts` at the pinned commit sends it to p1's connection (`receive`, :769-819):

- an `update` chunk becomes one `>battle-gen9championsvgc2026regmc-1` frame (`room.add` + `room.update`);
- a `sideupdate` line (`|request|`, `|error|`) becomes its own frame (`sendRoom`);
- before a `|request|` is sent, the server's battle-wide `rqid` counter is incremented and appended to the request JSON (:803-815). The counter is shared by both players. The sim emits one request per side per round, p1 then p2 (`sim/battle.ts` `sendUpdates`), and an `update` request to one side only (`sim/side.ts` `emitChoiceError`); the capture derives p1's rqids from both players' request lists and refuses to run if p2 ever gets an `update` request, which would make them ambiguous.

Differences from a real server connection: the wall-clock `|t:|` lines are dropped (as the runner drops them, so the file is deterministic), and the server's own room lines (`|init|`, `|title|`, `|j|`, `|inactive|` timer lines) are absent.

## `server-layer.json`: frames only the server sends, written by hand

These do not come from the engine. Each is written by Areneon from the pinned server source cited in its `source` field; the requests in them are copied from `battle.json`:

- `global_updateuser`: a frame with no `>ROOMID` header, which PROTOCOL.md allows for the global room;
- `other_room`: a frame for another room;
- `request_null`, `request_empty`: the withdrawn request a leaving player gets (`onLeave`), and its empty-body form;
- `resend_with_sentchoice`, `resend_without_sentchoice`: the current request re-sent on rejoin (`onConnect`), with and without the choice the server holds;
- `nothing_to_choose`, `too_late`: the server layer's own `[Invalid choice]` refusals (`choose`), sent without asking the sim.
