# Reg M-C target-ruleset brief

Research for Redmine #673 (Notion Wave 1 acceptance gate). It indexes all seven gate items; the evidence for each lives in its sibling ticket (#673 itself was narrowed to the format matrix #772 and the gap reconciliation #773). It is the single entry point to the Wave 1 evidence and the recommendation for the Wave 2 go/no-go (#683).

**Brief version 2, 2026-10-20.** Version 2 records the merged Battle Data API assessment (#679). Based on `main` at `d099048`, the pinned engine `smogon/pokemon-showdown` `7244fe8e28fa8bc4d0809ae30d183c608b62bb2b` and the rules snapshot of that commit (`rules_snapshot_hash` `223fa44e8bd8193226753329eee461a2bfde1f4b91acc8895d4abcebf8089e51`, from `node packages/showdown-bridge/scripts/rules-hash.ts`). A new version of this brief is due when any of these moves, or when an open item below lands.

**How to read.** This brief consolidates; it adds no new fact. Every fact is cited by the research document and section that holds its primary source, or by the rules record, ADR or code path on `main`. Evidence that is not on `main` yet is marked **open** with its owning ticket. Status words: **met** (evidence merged on `main`), **partly** (some evidence merged, a named part missing), **open** (no merged evidence).

## 1. The target in one table

| Item | Value | Where it is evidenced |
|---|---|---|
| Regulation | Pokémon Champions Regulation Set M-C, 2026-09-09 02:00 UTC to 2026-12-02 01:59 UTC | [regmc-format-matrix.md](regmc-format-matrix.md) R1, AV1-AV4 (official news [S2] there); `rules/regulations/M-C.json` |
| Battle style | VGC Doubles | format matrix T1 (Play! Pokémon handbook §4.1); `game_type: doubles` in `rules/formats/gen9championsvgc2026regmc.json` |
| Team | Bring 6, choose 4, the first two chosen lead | format matrix T2-T3, P1; gap record `team-preview` |
| Team sheets | Closed Team Sheet: Bo1 with Open Team Sheets declined by both players; the Bo3 sibling is excluded through the forbidden rule `forceopenteamsheets` | format matrix R6-R7, V8, D5; ADR-003; gap records `open-team-sheets`, `cts-invariant` |
| Showdown format | `[Gen 9 Champions] VGC 2026 Reg M-C`, id `gen9championsvgc2026regmc`, mod `champions` | format matrix R3-R5, R8-R9 |
| Timer | Official: 90 s preview, 45 s per turn, 7 min player time, 20 min game time. Showdown `VGC Timer`: 90 s, 55 s per turn, 420 s bank, no game clock, server-only | format matrix M1-M8; [timer-deadline-budget.md](timer-deadline-budget.md) §1 |
| Mechanics and legality authority | The pinned engine only; Areneon makes no claim of parity with the game | ADR-003; [showdown-champions-gaps.md](../showdown-champions-gaps.md) |

CTS is a project choice, not a copy of either official rule: Play! Pokémon VGC uses open team lists, and what in-game Ranked Battles show at Team Preview is unverified (format matrix D5, task U7).

## 2. Evidence index

| Area | Summary | Detailed document | Ticket | State on `main` |
|---|---|---|---|---|
| Target format matrix | Ruleset ids, availability, team rules, timer, turn phases and requests, player-visible and hidden fields under CTS; every row has a source and a verification date, or is **unverified** with a task (U1-U7) | [regmc-format-matrix.md](regmc-format-matrix.md) §1-§6, §8 | #772 | Merged, Resolved |
| Differences vs Showdown | Ten differences D1-D10, each mapped to a gap record; the ledger is regenerated from `rules/gaps/` | format matrix §7; [showdown-champions-gaps.md](../showdown-champions-gaps.md) | #773 | Merged, Resolved |
| Legal move semantics | Joint choices are enumerated from the player's own request and judged by the engine; a golden choice corpus and a brute-force oracle run in CI | ADR-006; gap record `joint-legal-choices`; `fixtures/showdown/gen9championsvgc2026regmc/choices/` | Epic #711 | Merged |
| Data sources and rights | Twelve sources (D1-D12) with licence or terms, verdict and open questions Q1-Q12; unlicensed material is excluded by default | [data-sources-rights.md](data-sources-rights.md) Summary, §6 | #774 | Merged, Resolved |
| Training data decision | ADR-009 (Training data sources), Accepted 2026-10-20: train only on local self-play and synthetic teams; every third-party source stays excluded until a revision admits it | Redmine wiki `ADR-009_Training_Data_Sources` (no copy in `docs/adr/` yet) | #728 | Decision recorded; implementation Epic on Wave 3 |
| Champions Battle Data API | Endpoints, schema, availability, terms and every field classified (base-stat, usage, legal-roster). Neither terms page is marked as governing; both apply and the stricter one is followed. Verdict: **do not use** for training data, pool U, the synthetic team generator or the rules records; **use with conditions** as an owner-run, offline cross-check, pending the owner's acceptance (OQ1), with no further calls until then. ADR-009 row 4 stands | [battle-data-api-assessment.md](battle-data-api-assessment.md) Summary, §4.4, §5, §8, §9, §11; [data-sources-rights.md](data-sources-rights.md) D11, Q5 | #679 | Merged; owner decisions OQ1-OQ5 open |
| Stat Points, nature, IVs, turn order | Engine formula compared with official screenshots (5 Pokémon, 30 values, 0 mismatches); caps 32 / 66 agree; turn order read and run on the pin, game side mostly unverified (V1-V12) | [stat-formula-and-turn-order-evidence.md](stat-formula-and-turn-order-evidence.md) §1-§5, §5.4 | #680 | Merged; unknowns V1-V12 open; its §7.1 gap-ledger proposals not applied |
| Evaluation protocol | Opponent pools, held-out teams, paired seeds, both seats, team-clustered bootstrap intervals, sealed-test rules; no numeric target | [evaluation-protocol.md](evaluation-protocol.md) §3-§7, §10, §12 | #681 | Merged, Resolved |
| Decision deadlines | Per-decision budget derived from the VGC Timer state | [timer-deadline-budget.md](timer-deadline-budget.md) §2-§4; ADR-005 | #678 | Merged |

## 3. Wave 1 acceptance gate

The seven items are quoted from the Notion Wave 1 page ("Wave 1 acceptance gate"), except item 4, which is paraphrased.

| # | Gate item | Status | Evidence | What is missing |
|---|---|---|---|---|
| 1 | "A versioned target-ruleset brief cites primary game/rules references and identifies differences against Showdown explicitly" | **Met** once this brief merges | This brief (§1, §4); format matrix §1-§7 and Sources; gap ledger | — |
| 2 | "Initial team/format and legal move semantics documented; all unknown items recorded with verification task, not silently assumed" | **Partly** | Format and team rules: format matrix §3, §5, §8 (U1-U7). Legal move semantics: ADR-006, choice corpus, gap record `joint-legal-choices`. Every gap record is `unverified` until an in-game fixture exists | Read as including the owner-team feature inventory against the pinned engine and the gap ledger (#775, not started), because of the "initial team" wording. #775 is not among #683's blockers; whether item 2 needs it is the owner's call in #683, which can waive it by a named waiver |
| 3 | "Sources and reuse rights for proposed datasets inventoried; unlicensed material excluded from initial work" | **Met** | [data-sources-rights.md](data-sources-rights.md) (Summary, §5, §6); ADR-009 Accepted (#728) | — |
| 4 | Pokémon Champions Battle Data API: endpoints, schema, availability, attribution, caching and no-mirroring restrictions assessed; base-stat, usage and legal-roster fields classified | **Met** | [battle-data-api-assessment.md](battle-data-api-assessment.md) §3 (endpoints, schema, availability), §4 (terms, attribution, caching, no mirroring; Q5 answered in §4.4), §5 (field classification), §8 (verdict) | The owner's acceptance of the cross-check use (OQ1) and the other open questions OQ2-OQ5 (§11 there); none changes the assessment |
| 5 | "Champion Stat Points/nature/IV formula and turn-order mechanics have in-game/reference evidence and known unknowns recorded" | **Met** (reference evidence, not in-game fixtures) | [stat-formula-and-turn-order-evidence.md](stat-formula-and-turn-order-evidence.md) §3 (formula vs official screenshots), §4 (caps), §5 (turn order), §5.4 (unknowns V1-V12 with verification tasks); regression test `packages/showdown-bridge/test/champions-stats.test.ts`; format matrix T12-T13 | V1-V12 remain open verification tasks. The gap-ledger edits it proposes (its §7.1) are not applied: `rules/gaps/` is unchanged and needs its own PR |
| 6 | "Success/failure metrics and held-out evaluation protocol defined without fabricated Elo or win-rate thresholds" | **Met** | [evaluation-protocol.md](evaluation-protocol.md) §6.5, §6.6, §10, §12 | Parameter values are left to ADR-010 (Proposed, #729) by design |
| 7 | "Go/no-go decision for Wave 2 and a prioritized unanswered-question list recorded" | **Partly** | Recommendation in §8 and the prioritized list in §7 of this brief | The decision itself is recorded by the owner in #683 |

## 4. Differences vs the Showdown format

Taken from the format matrix §7 and the gap records; not re-derived here. Every record is `unverified` (the ledger requires an in-game fixture for `verified`).

| # | Difference | Gap record(s) | Effect on offline self-play |
|---|---|---|---|
| D1 | Per-turn timer 55 s on Showdown vs 45 s official | `timer`, `in-game-timer` | None: A1 runs no timer (format matrix I3) |
| D2 | No 20-minute game clock on Showdown; official game-time result unknown (U6) | `timer`, `in-game-timer` | None on A1 |
| D3 | Showdown timer is server-only and off by default | `timer` | None on A1; A2 deadlines follow [timer-deadline-budget.md](timer-deadline-budget.md) |
| D4 | Showdown requires exactly 6 Pokémon; VGC allows 4-6 by event; in-game Ranked size unverified (U3) | `team-preview` | Areneon runs 6-Pokémon teams only |
| D5 | CTS target is neither the VGC rule (open lists) nor a verified copy of Ranked (U7) | `open-team-sheets`, `cts-invariant` | Defines the observation contract (ADR-008) |
| D6 | Eligible species: official whitelist vs champions mod data; not compared entry by entry (U4) | `roster-and-learnsets` | Inference: teams outside the official list could be accepted (the pool is set by the champions formats data, format matrix I2) |
| D7 | Stat Point caps 32 / 66 and all-31 IVs come from the engine; official screenshots agree on 32 / 66 ([stat-formula-and-turn-order-evidence.md](stat-formula-and-turn-order-evidence.md) §4), the refusal at 33 and IVs stay unconfirmed (V1, V2) | `stat-points`, `pokepaste` | Engine is the authority (ADR-003) |
| D8 | Simultaneous last faint: engine rule vs unpublished game rules | `simultaneous-faint` | Such games should carry the gap label; no detector yet |
| D9 | Shiny flag visible at Showdown Team Preview in the champions mod | `cts-invariant` | Whether the observation strips it is open (ADR-008 owner) |
| D10 | M-C balance changes not checked against the engine's champions overrides beyond two named examples | `move-item-ability-overrides`, `override-parity`, `pp` | Inference: results depend on the pin's overrides, which are unchecked beyond Wish and Strength Sap |

Not exercised at all on `main`: the damage formula and rounding (gap record `damage-formula`: "covered by the calculator work, not by this pin"). [stat-formula-and-turn-order-evidence.md](stat-formula-and-turn-order-evidence.md) §7.1 proposes two new records, `turn-order` and `mega-evolution-order`; they are not applied yet.

## 5. Stat Points and turn order

From [stat-formula-and-turn-order-evidence.md](stat-formula-and-turn-order-evidence.md):

- **Formula** (§2, §3): in the champions mod, HP = base + SP + 75 and other stats = floor((base + SP + 20) × nature) at level 50; IVs and level are not read. It matches 30 values of 5 Pokémon from official screenshots with no mismatch. The screenshots are promotional material, not an Areneon fixture.
- **Caps** (§4): engine validator and the game's Stat Points screen agree on 66 in total and 32 per stat; the refusal of a 33rd point is not shown (V1); IVs fixed at 31 is consistent but unstated (V2).
- **Turn order** (§5): priority, then speed, then a random shuffle of exact ties; speed recomputed before each move; Trick Room without overflow in the champions mod; full paralysis 1/8. Game side mostly unverified (V3-V12); two official notices concern Mega Evolution order (V10) and switch-in ability order (V11).

## 6. Evaluation metrics

From [evaluation-protocol.md](evaluation-protocol.md); no numeric target is set anywhere.

| Metric | Definition | Section |
|---|---|---|
| Mean score per opponent | Wins, ties counted as half, over the paired block plan, with a team-clustered bootstrap interval | §6.1, §6.5 |
| Per-pool and per-seat results | Rows grouped by opponent pool (baselines, frozen checkpoints, conditional usage-statistics pool, exploiters); p1 and p2 apart | §3, §6.5 |
| Paired differences between checkpoints | Mean block difference with the same bootstrap; Holm on pre-registered comparisons | §6.3, §6.4 |
| Engine and contract health | Harness incidents (must be 0 for a valid run), `[Unavailable choice]` refusals, privacy test status | §6.5 |
| Decision latency | Per-decision compute time against `latency_budget` | §6.5, §10 |
| Elo-style rating | Reported as a view of the head-to-head matrix; never a goal, gate criterion or selection rule | §6.6 (D17, D18) |

Every threshold-like value (`h`, `α`, `n_teams`, `B_boot` and the rest) is a named parameter whose value ADR-010 records (§10). Results generalise to the synthetic team generator's distribution only (protocol header; ADR-009). Since the protocol merged, a `random-legal` baseline (its O8) has landed on `main` (`packages/showdown-bridge/src/random-legal.ts`), and the Episode `provenance.collection` field (its O3) is open as PR #79.

## 7. Open questions and unknowns, prioritized

Priority 1 blocks the Wave 1 exit or a Wave 2 gate item; priority 2 limits what results may claim; priority 3 is tracked but blocks nothing now.

| P | Question | Source of the question | Owner |
|---|---|---|---|
| 1 | Apply the gap-ledger proposals (`stat-points`, `trick-room`, `paralysis`, `pp`, `override-parity`, new `turn-order`, `mega-evolution-order`) in their own PR | [stat-formula-and-turn-order-evidence.md](stat-formula-and-turn-order-evidence.md) §7.1 | not ticketed |
| 1 | Stat Point / nature / Speed-priority and damage contracts have **no ADR and no Epic**; the damage formula is not exercised at all | Notion Wave 2 page ("Open gap"); gap record `damage-formula` | New ADR needed (none filed) |
| 1 | Owner-team feature inventory against the pinned engine and the gap ledger (if the owner reads gate item 2 as including it) | gate item 2; Notion minimum validation principle | #775; decided in #683 |
| 1 | Reconcile Wave 1 with the Development Readiness items D0-D3; record the reconciliation in #683 | #683 acceptance criteria | #683 |
| 2 | Simultaneous Mega Evolution order (a documented game issue) and speed ties | [stat-formula-and-turn-order-evidence.md](stat-formula-and-turn-order-evidence.md) §5.4 V10, V3 | #680 follow-up |
| 2 | Per-stat refusal at 33 and fixed IVs | [stat-formula-and-turn-order-evidence.md](stat-formula-and-turn-order-evidence.md) §5.4 V1, V2; format matrix U5 | #680 follow-up |
| 2 | Official eligible list vs the champions mod data, entry by entry | format matrix U4, D6; rights Q8 (read by hand) | `roster-and-learnsets` record |
| 2 | What in-game Ranked Team Preview shows; whether the observation strips the shiny flag | format matrix U7, D9 | `cts-invariant` record; ADR-008 observation contract |
| 2 | A detector that labels simultaneous-last-faint games | protocol O7; gap record `simultaneous-faint` | not ticketed |
| 2 | Evaluation parameter values, the register, the isolated store and the team generator | protocol O1, O2, O9, O11 | ADR-010 (#729) |
| 2 | Whether neural inference is deterministic on the evaluation hardware, and the rerun tolerance (bears on Wave 2 gate item 7) | protocol O5 | ADR-010 (#729) |
| 2 | Accept the Battle Data API cross-check use (owner-run, offline, conditions in its §9) as a delta revision of ADR-009, or leave it uncalled; the other owner questions OQ2-OQ5 | [battle-data-api-assessment.md](battle-data-api-assessment.md) §8, §11 | #679 (owner) |
| 3 | Ranked season dates, VGC events in the window, team size and move count in game, timeout and game-time behaviour | format matrix U1, U2, U3, U6 | format matrix §8 tasks |
| 3 | Remaining turn-order tasks V4-V9, V11, V12 | [stat-formula-and-turn-order-evidence.md](stat-formula-and-turn-order-evidence.md) §5.4 | #680 follow-up |
| 3 | Timer behaviour never observed on a running server; estimator calibration | [timer-deadline-budget.md](timer-deadline-budget.md) §6 | ADR-005 Epic follow-ups |
| 3 | Origin of PokéAPI's `champions` data; which "Champions Calc" ADR-001 means (the verdict holds either way) | rights Q6, Q7 | rights inventory owner |
| 3 | Preview Policy evaluated alone or only in the pair; harness support for forfeits; scheduling the sealed-hash scan; reviewer independence | protocol O10, O12, O13, O14 | ADR-010 (#729) |
| 3 | Third-party sources (replays, datasets, usage statistics): excluded by ADR-009's defaults; the written request to Smogon is the owner's to send | rights Q1-Q4, Q9-Q12; ADR-009 §9 decision 7 | #728 (owner) |

## 8. Wave 2 go/no-go recommendation (#683)

**Call: conditional go.** The decision, and whether any wave is complete, is #683's call; this section recommends. Wave 2's work may continue in parallel, as it already does, but Wave 1 is not complete and Wave 2 cannot be declared started until the conditions below hold and the owner records the decision in #683.

**What is evidenced.** The target is fixed and sourced (§1), every known difference from the Showdown format has a gap record (§4), rights are inventoried and the training-data decision is accepted (ADR-009), and the evaluation protocol defines metrics without thresholds (§6). On the Wave 2 side, much of the delivery gate ("Reference battle adapter + legal player-visible action contract pass fixtures", Notion Milestones) already has merged evidence:

| Wave 2 gate item (Notion Wave 2 page) | Status in this index | Evidence on `main` |
|---|---|---|
| Seeded reference battles end to end with auditable logs | Evidence on `main` | `runner.ts`; `fixtures/showdown/gen9championsvgc2026regmc/battle.summary.json` re-captured in CI; A1/A2 equality test `packages/bot-client/test/a1-a2-equality.test.ts` |
| Action legality over move, switch, target, forced replacement and preview; invalid actions fail observably | Evidence on `main` (engine-judged, no in-game fixture) | ADR-006; choice corpus and oracle; `judgeSubmission`, runner aborts on any other engine error |
| No hidden-information path into observations, feature caches or replay-derived labels | Partly | ADR-008 privacy tests (`test/observation.test.ts`, `cts.test.ts`); feature caches and labels do not exist yet |
| Fixtures cover critical mechanics; mismatches documented and block incompatible experiments | Partly | Gap ledger documents them; nothing yet blocks an experiment on a gap: no mechanism checks an experiment's teams against the gap records, and the simultaneous-faint detector (protocol O7) does not exist |
| Stat/SP formula, nature rounding, Speed/priority/Trick Room pass independent fixtures | Open | `champions-stats.test.ts` ties the pin to the screenshot values ([stat-formula-and-turn-order-evidence.md](stat-formula-and-turn-order-evidence.md) §3); no contract, ADR or Epic |
| Damage roll / KO calculation with Champions effects validated | Open | Gap record `damage-formula` only; no contract, ADR or Epic. Its rule that no AGPL Champions Calc source is reused is covered by the rights inventory's D10 verdict "Do not use" |
| Adapter contracts and version migrations documented; repeatability class and tolerances declared | Partly | ADR-008 §3 (versioning, upcasters), version manifest (ADR-003); same commit, seed and choices reproduce `omniscient_log` (`runner.ts`, `replayActor`); no declared repeatability class or tolerance |

**What is not evidenced.** Wave 1 gate item 2 lacks the owner-team inventory if it is read to include it. The Milestones page states that exit gates "cannot be waived implicitly", so these cannot be carried into Wave 2 silently.

**Conditions before #683 can record a go:**

1. Done: the Stat Point and turn-order evidence (#680) is merged and cited above (gate item 5).
2. Done: the Battle Data API assessment (#679) is merged with the field classification (gate item 4).
3. #775 delivered, if the owner reads gate item 2 as including the owner-team inventory ("initial team"); #775 has no "blocks #683" relation in Redmine today; the owner decides that in #683 and can waive it there by a named waiver.
4. The owner records the decision and the prioritized list (§7) in #683.

**Before Wave 2 can be called complete:** an ADR and an Epic for the Stat Point / nature / Speed-priority and damage contracts, with independent fixtures (Wave 2 gate items 5 and 6); a declared repeatability class and tolerances; and a mechanism that makes a gap block an incompatible experiment. None of these exists today, so the reasoning is: the adapter and legal action contract evidence supports continuing Wave 2, while the missing calculator contracts and the open Wave 1 items rule out an unconditional go; a no-go would stop work whose evidence is already green in CI.
