# Fork-oracle fidelity report (ADR-026 slice F)

Corpora: random 40 battles / 1372 own move decisions; random-choice-lock 40 battles / 1396 own move decisions; unavailable 2 battles / 62 own move decisions; search-fidelity 200 battles / 5528 own move decisions. Decisions 8358; worlds built by the committed record 3776; refused 4582.
F2 samples 3 own candidates x 2 opponent replies per turn (a fixed stride sample of the canonical list). N_ex = 20.

## Refusals of the committed record (counted, not compared)

| code:item | decisions |
| --- | ---: |
| state_unwritten:unadmitted_id | 1760 |
| state_unwritten:locked_moves | 747 |
| state_unwritten:ability_changed | 357 |
| state_unwritten:substitute | 351 |
| facts_unavailable:- | 282 |
| state_unwritten:start_trigger_restore | 244 |
| state_unwritten:item_replaced | 232 |
| state_unwritten:trapping | 171 |
| v1_revealed_fact:- | 144 |
| state_unwritten:illusion_transform | 131 |
| state_unwritten:slot_conditions | 86 |
| reader_flagged:- | 53 |
| v2_request:- | 14 |
| v2_request:choice_lock | 8 |
| engine_refused:- | 2 |

## Fidelity, X-class turns apart

| mode | compared (non-X) turns | equal | share | X-class turns | X equal | X share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F1 | 1820 | 1535 | 84.34% | 1956 | 1582 | 80.87% |
| F2 | 1820 | 1192 | 65.49% | 1956 | 1374 | 70.24% |

F2 pairs (one pair is one own candidate against one opponent reply): non-X 9208 of 10920 equal (84.32%); X 9750 of 11733.

## Differences by kind and by cause (non-X turns)

A turn can have several causes. `writer`: a state difference classified against an item of the record or an admitted id; `information`: the player cannot know it (the opponent's exact hit points and PP, bench PP); `unmodelled`: engine state the world lacks (a volatile no writer or admitted id covers); `engine`: another field of the pre-turn state; `unclassified`: the turn differs and no pre-turn state difference names it.

| mode | outcome | draw_order | request | choice | writer | information | unmodelled | engine | unclassified |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F1 | 270 | 0 | 15 | 0 | 0 | 284 | 5 | 50 | 0 |
| F2 | 601 | 2 | 25 | 0 | 0 | 626 | 11 | 80 | 0 |

### Unmodelled state behind differing turns

| engine id | F1 turns | F2 turns |
| --- | ---: | ---: |
| allyswitch | 1 | 16 |
| minimize | 2 | 13 |
| unburden | 2 | 11 |

### Information-limited state behind differing turns

| state | F1 turns | F2 turns |
| --- | ---: | ---: |
| bench_pp | 27 | 156 |
| opponent_hp | 243 | 1503 |
| opponent_pp | 283 | 1699 |

### Other pre-turn state differences behind differing turns (engine)

| state | F1 turns | F2 turns |
| --- | ---: | ---: |
| opponent_ability | 3 | 16 |
| opponent_active_slots | 45 | 206 |
| opponent_status_state_frz | 1 | 9 |
| own_status_state_frz | 1 | 14 |

## Writers of the committed record

An item passes with at least 20 exercising turns in F1 and in F2 that are equal, and no difference classified against it. "Exercised": the item is present in a non-X turn and some Pokemon acted (a `|move|` or `|cant|` line). "Blamed": a difference is classified against the item or no state difference names it. "Missed": a difference is classified against the item in a turn where the item was absent or nothing acted (the writer wrote nothing where its field differs). The F1 and F2 columns count turns; an item passes with none blamed and none missed.

| item | class | F1 exercised | F1 equal | F1 blamed | F1 missed | F2 exercised | F2 equal | F2 blamed | F2 missed | passes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| active_turns | W | 1820 | 1535 | 0 | 0 | 1820 | 1192 | 0 | 0 | yes |
| bench_status | W | 77 | 66 | 0 | 0 | 77 | 39 | 0 | 0 | yes |
| choice_lock | W | 90 | 74 | 0 | 0 | 90 | 55 | 0 | 0 | yes |
| condition_duration | W | 635 | 523 | 0 | 0 | 635 | 431 | 0 | 0 | yes |
| fainted_last_turn | W | 692 | 590 | 0 | 0 | 692 | 475 | 0 | 0 | yes |
| item_removed | W | 713 | 596 | 0 | 0 | 713 | 452 | 0 | 0 | yes |
| last_move | W | 1816 | 1531 | 0 | 0 | 1816 | 1189 | 0 | 0 | yes |
| protect_counter | W | 129 | 108 | 0 | 0 | 129 | 86 | 0 | 0 | yes |
| total_fainted | W | 1323 | 1108 | 0 | 0 | 1323 | 875 | 0 | 0 | yes |
| toxic_stage | W | 44 | 31 | 0 | 0 | 44 | 28 | 0 | 0 | yes |
| turn_number | W | 1820 | 1535 | 0 | 0 | 1820 | 1192 | 0 | 0 | yes |

## Candidate ids (admitted one at a time into a copy of the record)

"Exercised" for `volatile:<id>`: the Pokemon that carries it acted in the turn. For `trigger:<id>`: the single admitted id covered a scenario-start restore.

| item | class | F1 exercised | F1 equal | F1 blamed | F1 missed | F2 exercised | F2 equal | F2 blamed | F2 missed | passes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| trigger:whiteherb | W | 60 | 49 | 0 | 0 | 60 | 32 | 0 | 0 | yes |
| volatile:aquaring | E | 1 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | no |
| volatile:charge | E | 3 | 3 | 0 | 0 | 5 | 2 | 0 | 0 | no |
| volatile:curse | E | 3 | 2 | 0 | 0 | 4 | 4 | 0 | 0 | no |
| volatile:disable | W | 21 | 16 | 4 | 0 | 26 | 12 | 11 | 0 | no |
| volatile:encore | W | 51 | 41 | 0 | 0 | 60 | 37 | 0 | 0 | yes |
| volatile:flashfire | W | 32 | 28 | 0 | 0 | 34 | 24 | 0 | 0 | yes |
| volatile:focusenergy | E | 46 | 38 | 0 | 0 | 65 | 47 | 0 | 0 | yes |
| volatile:healblock | W | 2 | 2 | 0 | 0 | 2 | 0 | 2 | 0 | no |
| volatile:imprison | E | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no |
| volatile:ingrain | E | 2 | 1 | 0 | 0 | 2 | 1 | 0 | 0 | no |
| volatile:leechseed | E | 4 | 1 | 0 | 0 | 6 | 0 | 0 | 0 | no |
| volatile:smackdown | E | 2 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | no |
| volatile:taunt | W | 41 | 37 | 0 | 0 | 53 | 42 | 0 | 0 | yes |
| volatile:throatchop | W | 43 | 38 | 0 | 0 | 48 | 28 | 0 | 0 | yes |
| volatile:torment | E | 53 | 45 | 0 | 0 | 65 | 47 | 0 | 0 | yes |
| volatile:yawn | W | 33 | 30 | 0 | 0 | 40 | 24 | 0 | 0 | yes |

## Admissions this report supports

- start_trigger whiteherb (class W)
- volatile encore (class W)
- volatile flashfire (class E)
- volatile focusenergy (class E)
- volatile taunt (class W)
- volatile throatchop (class W)
- volatile torment (class E)
- volatile yawn (class W)

## Limits of the classification

- A difference is attributed by the tags of the pre-turn state difference (`stateDelta`), not by a counterfactual run. Fields it does not compare (`slotConditions`, `abilityState`, `itemState`, `lastMoveUsed`, trapping flags, some per-turn fields) cannot be named: a turn that differs because of one of them AND carries an information tag (the opponent's hit points or PP differ in most differing turns) is classified `information`. For that reason an item passes only on its EQUAL exercising turns (at least N_ex of them in F1 and in F2), and the turns that differ for another cause are not evidence for it.
- "Exercised" for a writer means it is present and some Pokemon acted: the writers that are written every turn (`active_turns`, `last_move`, `turn_number`) are exercised in every turn, so their rows say little beyond "no difference is classified against them".
- F2 samples a fixed stride of the canonical candidate list, not a ranked top-k; a turn is one unit however many pairs it has.
- The committed record is evaluated with `admitted` emptied, over the worlds it builds: refused decisions (the table at the top) are not compared.

