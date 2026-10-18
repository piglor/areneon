# Timer-derived decision deadlines (VGC Timer)

Research for Redmine #678. Decision evidence for ADR-005 (bot opponent runtime, #697).

**Question.** How should the bot's per-decision time budget be derived from the live VGC Timer so that the bank is never exhausted and the server's timeout auto-choose never fires? ADR-005 fixes 60 s (Team Preview) and 20 s (battle) as upper bounds only.

**Primary source.** Pinned Showdown submodule `vendor/pokemon-showdown` at `7244fe8e28fa8bc4d0809ae30d183c608b62bb2b`. All paths below are relative to that checkout. Facts come from reading the code. No live server was run for this note (see [Open unknowns](#open-unknowns)).

**Executable model.** `spikes/678-timer-budget/timer_budget.py`. It contains a shadow timer that mirrors the server semantics below, plus the budget functions. It is tested by `test_timer_budget.py`, which includes a Hypothesis property: the bank is never exhausted and auto-choose never fires.

## 1. Verified facts

### 1.1 The rule and its settings

| Fact | Source |
|---|---|
| `VGC Timer` sets `Timer Starting = 420`, `Timer Grace = 90`, `Timer Add Per Turn = 0`, `Timer Max Per Turn = 55`, `Timer Max First Turn = 90`, `Timeout Auto Choose`, `DC Timer Bank`. | `data/rulesets.ts:778-787` |
| `[Gen 9 Champions] VGC 2026 Reg M-C` includes `VGC Timer` (and `Open Team Sheets`). | `config/formats.ts:288-292` |
| The rule table converts those value rules into a `timer` settings object. | `sim/dex-formats.ts:286-313` |
| The format's settings override the server defaults. `timeoutAutoChoose` defaults to `false`, and VGC Timer turns it on. `accelerate` is off whenever the format sets timer rules. | `server/room-battle.ts:195-206` |
| Every player starts with `secondsLeft = starting + grace` = **510**. | `server/room-battle.ts:209-213` |
| Time is charged in whole **5 s ticks** (`TICK_TIME = 5`). | `server/room-battle.ts:42`, `344-391` |

### 1.2 How a request is timed

`nextRequest` (`server/room-battle.ts:310-343`) runs once for each non-update `|request|`. It runs right after the request has been sent to the player (`server/room-battle.ts:815-816`).

- **Wait requests** (`isWait` set, e.g. while waiting for the opponent's forced switch) cost nothing. `nextRequest` returns early (`:312-315`), and `nextTick` skips waiting players (`:352`). A submitted choice (`isWait = true`) also stops the charge.
- **Turn cap.** `turnSecondsLeft = min(secondsLeft, isFirstRequest ? 90 : 55)` (`:327-328`). Two consequences:
  - **The first timed request** gets 90 s. If the timer is already on when the battle starts, that request is Team Preview, so **battle turn 1 gets 55 s, not 90 s**. The 90 s first-turn cap lands on a battle turn only when the timer is switched on mid-battle.
  - **The turn cap shrinks with the bank.** Once the bank falls below 55 s, the per-turn time equals the bank.
- **Bank refill.** `updateTurn` (`:270-309`) clips `secondsLeft` to `starting` (420) at every new battle turn and adds `addPerTurn = 0`. Mid-turn requests (forced switches, pivots) add `min(addPerTurn, 5) = 0` (`:280`). **The bank never refills.**
- **Grace is the preview budget.** Unused grace is discarded by the turn-1 clip to 420. Any preview time ≤ 90 s therefore leaves the battle bank at exactly **420**.
- **Tick quantization.** `nextRequest` restarts the 5 s tick timer (`:317-320`, `:341-342`). Each tick charges 5 s to every player who still has to choose (`:352-361`). A decision that reaches the server within `5k + 4.99` s costs `5k`. In particular, **a decision answered in under 5 s server-side costs nothing.** The charge is always ≤ the real elapsed time.

### 1.3 Timeout: auto-choose or loss

`checkTimeout` (`server/room-battle.ts:439-461`):

- `turnSecondsLeft ≤ 0` with bank left and the player connected → the server writes `>pN default` (`:452-454`). This is the auto-choose that Areneon must never trigger.
- `turnSecondsLeft ≤ 0` with **bank = 0** → `forfeitPlayer(... 'lost due to inactivity.')` (`:455-457`). **Exhausting the bank loses the game.** It does not auto-choose.
- Once `B ≤ 55`, the turn cap equals the bank (`T = B`), so both reach 0 on the same tick. **Any turn timeout at a bank of 55 s or less is a forfeit, not an auto-choose.**
- If both players time out with no bank, the game is a tie (`:441-446`).

### 1.4 The timer is off unless someone turns it on

- The timer starts automatically only if `Config.forcetimer` is set (default `false`, `config/config-example.js:381`) or the format is Blitz (`server/room-battle.ts:607`).
- Otherwise a player starts it with `/timer on` (`server/chat-commands/core.ts:1404`). It also starts in later Bo3 games and on server restart, neither of which applies to our Bo1.
- When the timer is started mid-battle, `start()` calls `nextRequest` for every player (`server/room-battle.ts:241`). The current request then gets the 90 s first-request cap.
- On a private local server, **the human opponent controls whether the timer runs** and can turn it on at any time.

### 1.5 What a bot client actually receives

| Message | Channel | Content | Source |
|---|---|---|---|
| `\|inactive\|Time left: T sec this turn \| B sec total[ \| G sec grace]` | **Own player only** (`sendRoom`). Sent right after the `\|request\|` it belongs to. | Turn cap `T`, bank `B` (grace excluded), grace `G` | `server/room-battle.ts:333`, ordered after `:815` |
| `\|inactive\|Battle timer is ON: ...` / `\|inactiveoff\|Battle timer is now OFF.` | Room (public) | Timer on/off | `:238`, `:258` |
| `\|inactive\|NAME has N seconds left this turn.` | Room | Sent once at request time if `T ≤ 30` | `:334-336` |
| `\|inactive\|NAME has N seconds left.` | Room | Sent on ticks where `turnSecondsLeft % 30 == 0` or `≤ 20` | `:376-380` |

So the bot gets an **authoritative snapshot of its own `T`, `B` and `G` with every timed request**. That snapshot is player-perspective (own channel). It does not carry an `rqid`, so the client must pair it with the `|request|` just before it.

What the bot must track itself:

- elapsed time since receipt, on a local monotonic clock;
- the remaining-decision estimate;
- network latency;
- whether a timer-off battle may turn timed later.

Snapshots never arrive in four cases:

- wait requests;
- `update: true` re-sent requests (`:816`);
- a timer that is off;
- a bank already at 0.

Late choices are rejected by `rqid` (`server/room-battle.ts:631-636`), so a client should always send `/choose <choice>|<rqid>`.

### 1.6 The in-process sim (A1) has no timer

The `VGC Timer` rule only sets server settings. `sim/` never enforces them. This matches `docs/showdown-champions-gaps.md` (Timer row). A1 therefore needs a **shadow timer** with the semantics above if it is to produce budgets identical to A2.

### 1.7 The ticket premise, confirmed

At a flat 20 s per decision (≈20.25 s server-side), the 21st battle decision sees `B = 20`, so `T = min(B, 55) = 20`. Its 20 s budget is no longer below the turn cap. Because `T = B` (§1.3), that timeout **forfeits the game**. The test `test_fixed_twenty_seconds_exhausts_bank` checks this.

## 2. Budget function

Each decision gets **two** wall-clock allowances, measured from local receipt of the `|request|`:

- `policy_budget`: when the policy must have answered;
- `submit_within`: when the final choice must have been sent, whether it comes from the policy or the fallback.

### 2.1 Inputs

- **Timer snapshot.** `T` (turn seconds left), `B` (bank) and `G` (grace).
  - If the server timer is on, take them from the `|inactive|Time left` line.
  - Otherwise (A1, or A2 with the timer off), take them from the shadow timer. This keeps behaviour identical if the human turns the timer on later.
- **Remaining decisions `R̂`.** An estimate from player-perspective counts (§2.4).
- **Config.**

| Parameter | Value | Why |
|---|---|---|
| `preview_cap` | 60 s | ADR-005 upper bound |
| `battle_cap` | 20 s | ADR-005 upper bound |
| `safety_margin` `m` | 3 s (local server) | See §2.3 |
| `fallback_reserve` `f` | 1 s | Time to submit an already-computed legal heuristic choice |
| `bank_reserve` `B_res` | 15 s | Three ticks of bank that planned spending never touches; absorbs unplanned stalls |
| `min_remaining_decisions` | 4 | Floor for `R̂`; bounds spending per decision to ≤ ¼ of the spendable bank |

### 2.2 Formulas

```
hard          = max(0, T − m)                       # never reach the server's turn timeout
preview:
  alloc       = preview_cap
battle:
  alloc       = min(battle_cap, max(0, B − B_res) / R̂)
submit_within = min(hard, max(f, alloc))
policy_budget = max(0, submit_within − f)           # 0 ⇒ skip policy, send fallback now
```

**Preview.** The preview cap `T` is 90 s and preview spends grace only (§1.2). The result is `submit_within = min(87, 60) = 60` s and `policy_budget = 59` s. Preview never touches the battle bank.

**Battle.** The guarantees follow from the formulas:

1. `submit_within ≤ T − m`, so the choice reaches the server before `turnSecondsLeft` hits 0, provided one-way latency plus scheduling jitter stays under `m`. **Auto-choose cannot fire.**
2. Planned spending per decision is at most `(B − B_res)/R̂ ≤ (B − B_res)/4`. Spendable bank therefore shrinks at most geometrically: after `n` decisions `B − B_res ≥ (3/4)^n (B₀ − B_res)`. **It never reaches the reserve by planned spending, whatever the battle length or the estimator error.**
3. Once the allocation drops below about 5 s, tick quantization makes decisions free (§1.2). The bank settles at a fixed point, about 35 s in the worked example, and the bot still has about 4 s per decision indefinitely.
4. `T = min(B, 55) ≥ 5` whenever `B > 0`, because `B` is a multiple of 5. With `m = 3` there are always ≥ 2 s to send the fallback.

The Hypothesis property `test_bank_never_exhausted_and_autochoose_never_fires` checks 1 and 2 under the exact server semantics. It uses any `R̂ ∈ [1, 40]`, up to 300 turns and latency up to 1.5 s.

### 2.3 Safety margin

The server deadline is `t_send + T`, where `lastTick` is set just after the request is sent (`:333-342`). Node's `setTimeout` never fires early, so the server side only adds slack.

The client measures from `t_recv = t_send + d_down`, and its choice lands at `t_submit + d_up`. The margin must therefore cover:

- `d_down + d_up` (one RTT);
- client event-loop and GC stalls;
- the time for the fallback's `/choose` to leave the process.

On a local or LAN server the RTT is milliseconds, and 3 s gives more than 100× headroom while costing only 3 s of the 55 s cap.

For a remote server (ADR-005 option D) use `m = max(3 s, 2 s + 2 × RTT_p99)`, with the RTT measured per connection. The margin is a config value, not a constant, so the A2 spike can tune it from logged evidence.

### 2.4 Estimating remaining decisions

```
R̂ = max(4, ⌈1.5 × (own_alive + opp_alive)⌉ + own_bench)
```

- **`own_alive`** and **`own_bench`** come from our own request (`side.pokemon` conditions; bench = brought but not active). Each bench Pokémon can cause at most one forced-switch request.
- **`opp_alive`** = brought count (picked team size, announced as `|teampreview|N`; 4 in Doubles) minus public `|faint|` events for the opponent side. This is public log only, never hidden information.
- **The factor 1.5** turns per remaining Pokémon is deliberately pessimistic for VGC Doubles. It overestimates `R̂`, which only makes budgets smaller. At the start it gives `R̂ = 14`, so the first battle decisions are capped by the 20 s ADR bound, not by the bank.
- **Calibration.** The estimator does not have to be accurate for safety (§2.2, guarantee 2). It affects only how evenly the bank is spent. It should be refit from Wave-level episode statistics (turns and forced switches per battle) once self-play data exists.

### 2.5 Optional refinement (not adopted)

Because charges are `5·⌊elapsed/5⌋`, `submit_within` could be rounded up to just below the next tick boundary for free extra time. This depends on a tight estimate of `d_down`, and the saving is ≤ 5 s per decision. It is left for later, after A2 latency evidence exists.

## 3. Overrun policy

1. **On receipt** of a non-wait `|request|`:
   - read the timer snapshot and compute the budget;
   - **compute the legal heuristic fallback immediately**, from the request and validated through the pinned engine, before the policy runs;
   - arm a watchdog at `submit_within`. The watchdog lives outside the policy process or thread, so a hung policy cannot block it.
2. **Policy returns a legal joint choice by `policy_budget`.** Submit it with `|rqid` (A2).
3. **Policy times out, throws, or returns a choice the engine rejects** (`|error|[Invalid choice]`). Submit the fallback before `submit_within` and log a deadline record (§4.3) with outcome `fallback_timeout`, `fallback_error` or `fallback_invalid`.
4. **`policy_budget = 0`.** Skip the policy, submit the fallback at once, and log `skipped_low_budget`.
5. **Fallback rejected.** This should be impossible because it is engine-validated. As a last resort, the client itself sends `default` before `T − m/2`; `default` is a valid sim choice (`sim/side.ts:1290-1292`). Log it as `last_resort_default` with error severity, and treat it as an incident.
   - This client-sent choice is not the server's timeout auto-choose. That one never runs, because something is always submitted before the server timeout.
6. **A newer request arrives** (different `request_seq` / `request_hash`, or `rqid`) while the policy is running. Drop the stale computation without submitting it.
7. **Detecting auto-choose.** A request is superseded with no choice sent for it, or the room shows `|inactive|<bot> has N seconds left.` with `N ≤ 20`. Either is a test failure (gate: auto-choose count = 0).

Log records hold only our own request identity, our own timer snapshot and timings. They never hold opponent hidden information.

## 4. Deadline fields for the #671 contracts

#671 is not implemented yet, so this is a **field specification only**. Names follow the snake_case JSON style of the existing request identity (`request_seq`, `request_hash`).

### 4.1 Separation rule

The ADR-005 gate requires A1 and A2 observations of the same seeded battle to be equal **apart from `rqid` and timer fields**. Timer data therefore sits in a separate optional `decision_context` object next to the observation. It is not mixed into the observation's game-state fields. Equality checks exclude `decision_context` by name.

### 4.2 `DecisionContext` (on `PreviewObservation` and `BattleObservation`)

| Field | Type | Meaning |
|---|---|---|
| `timer_source` | enum `server` \| `shadow` | Where the snapshot came from (§2.1) |
| `timer_rules` | object `{starting, grace, add_per_turn, max_per_turn, max_first_turn, tick}` (int s) | Must equal the format's VGC Timer settings recorded in the version manifest. Carried for self-contained replay. |
| `turn_seconds_left` | int s | `T` at receipt |
| `bank_seconds_left` | int s | `B` at receipt (grace excluded) |
| `grace_seconds_left` | int s | `G` at receipt (non-zero only at preview) |
| `remaining_decisions_estimate` | int ≥ 1 | `R̂` (battle only; omitted at preview) |
| `safety_margin_ms` | int | `m` |
| `submit_within_ms` | int | Choice must be sent within this, from receipt |
| `policy_budget_ms` | int ≥ 0 | Policy must answer within this; 0 = skip policy |

All durations are relative. No wall-clock timestamps go into a policy input, which keeps observations deterministic and replayable.

### 4.3 `Recommendation` additions

| Field | Type | Meaning |
|---|---|---|
| `elapsed_ms` | int | Already in the glossary: receipt to choice ready |
| `deadline_outcome` | enum `in_budget` \| `fallback_timeout` \| `fallback_error` \| `fallback_invalid` \| `skipped_low_budget` \| `last_resort_default` | §3 |
| `source_policy_id` | string | The policy that produced the submitted choice (the fallback's ID when it fell back) |

### 4.4 `Episode` per-decision record additions

| Field | Type | Meaning |
|---|---|---|
| `decision_context` | `DecisionContext` | As observed |
| `deadline_outcome` | enum | As above |
| `submitted_after_ms` | int | Receipt to send |
| `bank_seconds_after` | int \| null | `B` from the next snapshot. Null if the battle ended or no snapshot was sent. |
| `server_auto_choose_detected` | bool | Must be `false`. Evidence for the "never auto-choose" gate. |

## 5. Worked example: a 40-turn battle

Output of `uv run python spikes/678-timer-budget/timer_budget.py`, using the synthetic `long_battle_schedule(40)`:

- the battle runs 40 turns plus 2 forced switches (one own faint every 10 turns from turn 5, one opponent faint every 10 turns);
- **every decision uses its full `submit_within` (worst case)**;
- server-side latency is 0.25 s;
- charges follow the exact server tick semantics.

| # | turn | T | bank before | R̂ | submit within (s) | bank after |
|---|---|---|---|---|---|---|
| 0 | 0 (preview) | 90 | 420 | 14 | 60.00 | 420 |
| 1 | 1 | 55 | 420 | 14 | 20.00 | 400 |
| 5 | 5 | 55 | 340 | 14 | 20.00 | 320 |
| 6 | 5 (forced switch) | 55 | 320 | 12 | 20.00 | 300 |
| 10 | 9 | 55 | 240 | 12 | 18.75 | 225 |
| 15 | 14 | 55 | 165 | 10 | 15.00 | 150 |
| 20 | 18 | 55 | 105 | 8 | 11.25 | 95 |
| 25 | 23 | 55 | 65 | 6 | 8.33 | 60 |
| 28 | 26 | 50 | 50 | 6 | 5.83 | 45 |
| 30 | 28 | 40 | 40 | 6 | 4.17 | 40 |
| 33 | 31 | 40 | 40 | 5 | 5.00 | 35 |
| 42 | 40 | 35 | 35 | 5 | 4.00 | 35 |

Reading the table:

- The 20 s ADR cap binds for the first 9 battle decisions. After that the budget tapers with the bank.
- From decision 34 the bank holds at **35 s**. Each 4 s decision lands under one tick, so it costs nothing.
- The battle can continue indefinitely with no auto-choose and no forfeit.
- `R̂` underestimates this stall battle badly (5 at turn 40), and the bank still survives (§2.2, guarantee 2).
- A flat 20 s policy fails at battle decision 21 (§1.7).

## 6. Open unknowns

1. **Not observed live.** None of this has been run against a local server with the timer on. The A2 spike (#677) should confirm three things:
   - the exact `|inactive|Time left` text and its ordering after `|request|`;
   - that a 4 s answer costs 0 s;
   - the measured RTT, to set `m`.
2. **Who turns the timer on.** ADR-005 has to decide whether the A2 bot sends `/timer on` at battle start, for VGC realism and a server-authoritative snapshot, or leaves the choice to the human. The budget is safe either way because of the shadow timer. But a human enabling the timer mid-battle hands the current request a fresh 90 s cap, and that A2 state cannot be reproduced in A1.
3. **Disconnects.** In a `/challenge` battle, `dcTimerBank` is on (`server/room-battle.ts:197`). A disconnected player then loses `dcSecondsLeft` (300 s, `:211`) instead of bank (`:353-366`).
   - The format's `DC Timer Bank` rule itself only sets `dcTimer` (`sim/dex-formats.ts:290-295`). This looks like an upstream quirk, worth confirming live.
   - Reconnect handling for the bot client is out of scope here.
4. **Estimator calibration.** `turns_per_pokemon = 1.5` and `min_remaining_decisions = 4` are reasoned defaults, not measured. Refit them from episode statistics. Safety does not depend on them.
5. **Policy compute on the shared host.** The 1 s fallback reserve assumes the heuristic is precomputed. If the A2 bot and a training job share a CPU, scheduling jitter may need a larger `m`. Measure it before Wave 2 model policies run under A2.
6. **Bo3 / later games.** Not in scope (Areneon is Bo1). Note that the timer auto-starts in later Bo3 games (`server/room-battle-bestof.ts:235`).
