# Fork-oracle fidelity report (ADR-026 slice F)

Corpora: random 40 battles / 1372 own move decisions; random-choice-lock 40 battles / 1396 own move decisions; unavailable 2 battles / 62 own move decisions; search-fidelity 77 battles / 2466 own move decisions. Decisions 5296; worlds built by the committed record 2670; refused 2626.
F2 samples 3 own candidates x 2 opponent replies per turn (a fixed stride sample of the canonical list). N_ex = 20.

## Refusals of the committed record (counted, not compared)

| code:item | decisions |
| --- | ---: |
| state_unwritten:unadmitted_id | 914 |
| state_unwritten:locked_moves | 514 |
| state_unwritten:substitute | 287 |
| state_unwritten:ability_changed | 222 |
| facts_unavailable:- | 170 |
| state_unwritten:trapping | 132 |
| state_unwritten:item_replaced | 131 |
| state_unwritten:start_trigger_restore | 70 |
| state_unwritten:slot_conditions | 52 |
| reader_flagged:- | 46 |
| state_unwritten:illusion_transform | 45 |
| v1_revealed_fact:- | 29 |
| v2_request:- | 11 |
| engine_refused:- | 2 |
| v2_request:choice_lock | 1 |

## Fidelity, X-class turns apart

| mode | compared (non-X) turns | equal | share | X-class turns | X equal | X share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F1 | 1264 | 1067 | 84.41% | 1406 | 1133 | 80.58% |
| F2 | 1264 | 840 | 66.45% | 1406 | 993 | 70.62% |

F2 pairs (one pair is one own candidate against one opponent reply): non-X 6430 of 7584 equal (84.78%); X 7008 of 8433.

## Differences by kind and by cause (non-X turns)

A turn can have several causes. `writer`: a state difference classified against an item of the record or an admitted id; `information`: the player cannot know it (the opponent's exact hit points and PP, bench PP); `unmodelled`: engine state the world lacks (a volatile no writer or admitted id covers); `engine`: another field of the pre-turn state; `unclassified`: the turn differs and no pre-turn state difference names it.

| mode | outcome | draw_order | request | choice | writer | information | unmodelled | engine | unclassified |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F1 | 183 | 0 | 14 | 0 | 0 | 197 | 5 | 41 | 0 |
| F2 | 402 | 2 | 20 | 0 | 0 | 422 | 10 | 64 | 0 |

### Unmodelled state behind differing turns

| engine id | F1 turns | F2 turns |
| --- | ---: | ---: |
| allyswitch | 1 | 16 |
| minimize | 2 | 13 |
| unburden | 2 | 10 |

### Information-limited state behind differing turns

| state | F1 turns | F2 turns |
| --- | ---: | ---: |
| bench_pp | 22 | 113 |
| opponent_hp | 163 | 986 |
| opponent_pp | 196 | 1143 |

### Other pre-turn state differences behind differing turns (engine)

| state | F1 turns | F2 turns |
| --- | ---: | ---: |
| opponent_ability | 3 | 16 |
| opponent_active_slots | 36 | 163 |
| opponent_status_state_frz | 1 | 9 |
| own_status_state_frz | 1 | 12 |

## Writers of the committed record

An item passes with at least 20 exercising turns in F1 and in F2 that are equal, and no difference classified against it. "Exercised": the item is present in a non-X turn and some Pokemon acted (a `|move|` or `|cant|` line). "Blamed": a difference is classified against the item or no state difference names it. "Missed": a difference is classified against the item in a turn where the item was absent or nothing acted (the writer wrote nothing where its field differs). The F1 and F2 columns count turns; an item passes with none blamed and none missed.

| item | class | F1 exercised | F1 equal | F1 blamed | F1 missed | F2 exercised | F2 equal | F2 blamed | F2 missed | passes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| active_turns | W | 1264 | 1067 | 0 | 0 | 1264 | 840 | 0 | 0 | yes |
| bench_status | W | 50 | 45 | 0 | 0 | 50 | 25 | 0 | 0 | yes |
| choice_lock | W | 89 | 73 | 0 | 0 | 89 | 55 | 0 | 0 | yes |
| condition_duration | W | 478 | 389 | 0 | 0 | 478 | 325 | 0 | 0 | yes |
| fainted_last_turn | W | 437 | 377 | 0 | 0 | 437 | 305 | 0 | 0 | yes |
| item_removed | W | 372 | 311 | 0 | 0 | 372 | 236 | 0 | 0 | yes |
| last_move | W | 1262 | 1065 | 0 | 0 | 1262 | 838 | 0 | 0 | yes |
| protect_counter | W | 97 | 83 | 0 | 0 | 97 | 65 | 0 | 0 | yes |
| total_fainted | W | 901 | 756 | 0 | 0 | 901 | 604 | 0 | 0 | yes |
| toxic_stage | R | 23 | 17 | 0 | 0 | 23 | 16 | 0 | 0 | no |
| turn_number | W | 1264 | 1067 | 0 | 0 | 1264 | 840 | 0 | 0 | yes |

## Candidate ids (admitted one at a time into a copy of the record)

"Exercised" for `volatile:<id>`: the Pokemon that carries it acted in the turn. For `trigger:<id>`: the single admitted id covered a scenario-start restore.

| item | class | F1 exercised | F1 equal | F1 blamed | F1 missed | F2 exercised | F2 equal | F2 blamed | F2 missed | passes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| trigger:whiteherb | W | 4 | 3 | 0 | 0 | 4 | 2 | 0 | 0 | no |
| volatile:aquaring | E | 1 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | no |
| volatile:charge | E | 3 | 3 | 0 | 0 | 5 | 2 | 0 | 0 | no |
| volatile:curse | E | 3 | 2 | 0 | 0 | 4 | 4 | 0 | 0 | no |
| volatile:disable | W | 16 | 15 | 1 | 0 | 20 | 11 | 8 | 0 | no |
| volatile:encore | W | 43 | 35 | 0 | 0 | 50 | 32 | 0 | 0 | yes |
| volatile:flashfire | E | 15 | 14 | 0 | 0 | 17 | 12 | 0 | 0 | no |
| volatile:focusenergy | E | 11 | 9 | 0 | 0 | 13 | 8 | 0 | 0 | no |
| volatile:ingrain | E | 2 | 1 | 0 | 0 | 2 | 1 | 0 | 0 | no |
| volatile:leechseed | E | 4 | 1 | 0 | 0 | 4 | 0 | 0 | 0 | no |
| volatile:smackdown | E | 2 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | no |
| volatile:taunt | W | 23 | 20 | 0 | 0 | 33 | 25 | 0 | 0 | yes |
| volatile:throatchop | W | 12 | 11 | 0 | 0 | 12 | 5 | 0 | 0 | no |
| volatile:torment | E | 7 | 7 | 0 | 0 | 12 | 10 | 0 | 0 | no |
| volatile:yawn | W | 4 | 3 | 0 | 0 | 6 | 2 | 0 | 0 | no |

## Admissions this report supports

- volatile encore (class W)
- volatile taunt (class W)

## Limits of the classification

- A difference is attributed by the tags of the pre-turn state difference (`stateDelta`), not by a counterfactual run. Fields it does not compare (`slotConditions`, `abilityState`, `itemState`, `lastMoveUsed`, trapping flags, some per-turn fields) cannot be named: a turn that differs because of one of them AND carries an information tag (the opponent's hit points or PP differ in most differing turns) is classified `information`. For that reason an item passes only on its EQUAL exercising turns (at least N_ex of them in F1 and in F2), and the turns that differ for another cause are not evidence for it.
- "Exercised" for a writer means it is present and some Pokemon acted: the writers that are written every turn (`active_turns`, `last_move`, `turn_number`) are exercised in every turn, so their rows say little beyond "no difference is classified against them".
- F2 samples a fixed stride of the canonical candidate list, not a ranked top-k; a turn is one unit however many pairs it has.
- The committed record is evaluated with `admitted` emptied, over the worlds it builds: refused decisions (the table at the top) are not compared.

