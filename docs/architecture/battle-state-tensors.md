# Battle-state tensors

The battle-state tensor schema: the reviewed, versioned list of fields a batched battle-state tensor holds for one format, with each field's source in the pinned engine's state, the per-seat bookkeeping of what each seat was shown, the visibility classes and projection rules, and the map from today's reducer and encoder to the fields.

Part of the architecture documentation split out of `CLAUDE.md` (index: [README](README.md)). Ticket references such as `#1463` point to Redmine (`redmine.piglor.com/projects/areneon`).

**What it is, and what it is not** (ADR-030 step 5 groundwork, ADR-028 section 5, #1463). ADR-028's direct tensor path needs "a projection of the surrogate's full state ... a reviewed, versioned list of fields", and ADR-030 step 5 needs that projection to give exactly `Encoder.encode`'s tensors with privacy by construction. This is the list, written from what exists (the engine's `Battle.toJSON()`, the world facts, the shared reader, the reducer and the encoder), as data: `config/engine-state/<format>/battle-state-spec.v1.json`, the `BattleStateSpec` contract (`packages/schema/contracts/v1/battle-state-spec.schema.json`, generated TS and pydantic types, samples in `fixtures/contracts/v1/battle-state-spec/`). **No engine is built and no tensor is produced here** (that is G4, #1464). The spec is an input to step 5's build ADR, not a commitment to the direct tensor path: ADR-028 sections 4 and 5 still require channels or a forced-draw trace before any surrogate game is training data. On a pin bump (Regulation M-D) the vocabulary sizes and any new state keys change; the drift test below shows exactly which.

**Readers.** TS `packages/showdown-bridge/src/battle-state-spec.ts` (`readBattleStateSpec(value, binding?)`, `loadBattleStateSpec(engine, formatId)`, `bindingFor`, `battleStateSpecHash`, `seatView`, `pathKeys`, `coveredKeys`, `resolvePath`, `vocabularyIds`) and Python `areneon_contracts.battle_state_spec` (`read_battle_state_spec(value, engine_data?)`, `battle_state_spec_hash`, `seat_view`, `path_keys`, `vocabulary_ids`, `state_item_ids`). Both refuse the same reader-invalid samples with the same words. `spec_hash` is SHA-256 of the record's JCS bytes (pinned in `battle-state-spec.test.ts` and `tests/contracts/test_battle_state_spec.py`); a change of any value is a new record, never an edit. The record names the engine-data version 2 table by `engine_data_hash` and the team vocabulary by `vocabulary_hash`, so the hash binds both.

## Groups and shapes

| Group | Shape | Holds |
| --- | --- | --- |
| `battle` | () | turn, started, ended, the last move, the PRNG state (opaque, hidden), the Quick Claw roll (hidden) |
| `side` | (SIDE) | side conditions (present, layers, duration), faint counters, Pokemon left, team size |
| `roster` | (SIDE, ROSTER) | species and forme, Mega state, level, gender, HP and max HP, the five stored stats, status and its counters, item and item-lost flags, ability, four move slots with PP and flags, stages, types, per-turn flags |
| `active` | (SIDE, ACTIVE) | the roster index in each slot, volatiles (present, layers, duration, counter, HP), `activeTurns`, `activeMoveActions`, last moves, the request flags, slot conditions |
| `field` | () | weather, terrain and field conditions, with their durations |
| `seat_field` | (SEAT) | the seat's own request kind; elapsed turns of the weather, terrain and field conditions as the seat counts them |
| `seat_side` | (SEAT, SIDE) | elapsed turns of each side condition |
| `seat_roster` | (SEAT, SIDE, ROSTER) | the reveal masks (species, nickname, item, ability, each move slot), the last shown HP, the shown species and forme, brought, typechange types, the reducer's recent history, the history-fact states |
| `seat_active` | (SEAT, SIDE, ACTIVE) | which volatiles the seat was shown, their derived suffix and start turn, the derived stall counter, pending slot effects |

Dimensions are data, never constants of the reader: counts come from `bindingFor` (the rule table's `maxMoveCount`, the team vocabulary's team sizes, the engine's `activePerHalf` and PRNG width, the engine's stat and boost ids) and the reader checks each against the record; a vocabulary dimension (`VOLATILE`, `SIDE_CONDITION`, `PSEUDO_WEATHER`, `SLOT_CONDITION`, `TYPE`) names engine-data v2 lists and its size is the union of their ids, read from the table. A vocabulary value is an index: 0 none, 1 unknown, then 2 + the id's position in the sorted union; the reader refuses a dtype too narrow for the table's size.

## Sources

Every field names its source. **Engine** fields name a path into `Battle.toJSON()` (roots `battle`, `side`, `pokemon`, `active` (a side's Pokemon in play: the engine serialises `side.active` as references), `field`; `[]` walks an array, `{}` an object's values) and a `read` rule (`value`, `bool`, `present`, `id`, `length`, `pokemon_ref`, `species_mega`, `details_level`, `details_shiny`, `multi_hot`, `opaque`). **Derived** fields name the module that fills them (`world_facts`, `shared_reader`, `reducer`, `reducer_history`, `engine_request`, `own_team`) and the value. A seat field is always derived: the engine's state holds nothing a seat was shown.

Engine keys at the six classified levels (battle, side, pokemon, field, move slot, effect state) that no field reads are in `omitted` with a reason (`cosmetic`, `derived`, `not_in_format`, `internal`, `hidden_input`, `reference`): for example `pokemon.set` and `ppUps` are team-building inputs held only through their effects (stats, max PP), the Terastallization, Dynamax and Z keys are not in the format, and `battle.queue` and the event machinery are the engine's transient state.

**Mega Evolution and Champions specifics.** The Mega forme is `roster.species` (the current forme) and `roster.mega` (the forme's `isMega`); the stone is the held item (`roster.item`), `roster.can_mega_evo` is the own request's flag, the pre-Evolution ability is `roster.base_ability` and the forme's is `roster.ability`. Level 50 and Stat Points enter only through `roster.level`, `roster.max_hp` and `roster.stats` (the engine's stored values); the set's Stat Points, nature and IVs are `hidden_input`. The engine's status marker `fnt` is `roster.fainted`, and the typeless marker `???` sets no type flag.

## Visibility classes and projection rules

| Class | The owner | Another seat |
| --- | --- | --- |
| `public` | exact | exact |
| `own-exact` | exact | its `projection`: `hp_percent` (the display rule) or `none` |
| `revealed-only` | where its own mask is set | where that seat's mask is set; else the unknown marker |
| `hidden` | never | never |

A projection reads a field only through its class (`seatView`): nothing can project a `hidden` field, and an `own-exact` field with projection `none` (exact stats, max HP, PP, move flags, the bench order, the Illusion target, the request flags) never reaches another seat. A `revealed-only` field names its mask (`reveal`), a 0/1 seat field of the same entity and extra dimensions: the owner's mask is set by its own request (all four moves, the item, the ability), another seat's by the lines that showed the value. Hidden fields: the PRNG state, the Quick Claw roll, the sleep counter and drawn length, the remaining turns of timed weather, terrain, field and side conditions and volatiles (some depend on the source's item, some are draws), volatile counters and HP. Each seat's view of those is its derived elapsed turns, stall counter and pending counts. `hp_percent` is the Champions display (`floor(100 * hp / max)`, never 0 for a Pokemon with hit points left; `hp-display.ts` `shownPercent` with rule `floor`); the drift test holds it equal to what the shared reader shows of every opponent Pokemon in play. A benched Pokemon's shown value is the last display, and the engine changes some silently (a switch-out heal), so the encoder's opponent HP reads `seat_roster.hp_shown`, not the projection of `roster.hp`.

## Encoder and reducer mapping

`encoder` maps every state column of the encoder's `Layout` (`mon_cat`, `mon_num`, `field_cat`, `field_num`, `cand_cat`, `cand_tok`; `#` stands for a number, `*` for an id) to the spec fields it is a function of, per half: `own` (an own token, or the own side's columns of the field token), `opponent`, `shared` (fields without a side) and `query` (the column also reads the candidate or the seat's own request). The covered layouts are the base layout, the lineage run's options (`candidate_matchup`, `recent_history`, `history_facts`) and `exact_volatiles`. `reducer` maps every `BattleState.to_dict` path (sides as `SIDE`, list entries as `[]`, id-keyed maps as `*`). Candidate features, the opponent sheet and the candidate target are `out_of_scope`: they are not battle state. The reader refuses any mapping of a hidden field and any opponent or shared mapping of a field another seat cannot see.

## World-writer state items

Every state item of `config/search/world-writers.v1.json` (`STATE_ITEM_IDS`) is `held` (the fields that hold it: the turn, faint counters, the item and ability changes, last moves, the per-turn flags, the volatiles with their durations and counters, the sleep and toxic counters, slot conditions, opponent PP, the bench order) or `not_held` with a reason: `unavailable_mechanics` (Terastallization, Z-Moves, Dynamax and Ultra Burst are not mechanics of the format).

## Drift and completeness gate

- `packages/showdown-bridge/test/battle-state-spec.test.ts` replays every committed corpus (the random-play corpus, `unavailable/`, the Illusion, Baton Pass, Fling, side-guard and revival battles and the targeted fidelity corpus; about 250 battles, 8,500 requests, both seats) with a spy on the engine's `BattleStream` and reads `toJSON()` at each request: every key at a classified level is a field's or omitted (an unknown key fails, and an omitted key no state holds fails as stale); every engine source resolves on real states (on every state for a key one below the battle, a side or the field); every vocabulary value is an id of its lists; the boost and stat ids, the move count, the active count and the PRNG width are the binding's; the display rule equals the shared reader's opponent HP and the own HP is exact; and a canary plants markers in every value another seat may not see (stats, max HP, PP, volatile durations, counters, the PRNG, the bench order) and projects the state for the seat by the rules: no marker survives, while a projection that ignores the classes leaks every one (negative control). The reader's refusals and the binding (engine-data hash, vocabulary hash, counts, dtype widths) are tested there too.
- `tests/ml/model/test_battle_state_spec.py`: every state column of the three layouts matches exactly one entry of an option the layout has (a dropped entry or a wrong option fails), no entry is stale, every reducer key at every request of the committed channels (both seats; the targeted fidelity corpus included) maps, and the reducer's own opponent-secret rule agrees with the spec: where the spec maps the opponent half to nothing over fields another seat does not see exactly (stats, `moves_own`, `pp`, `can_mega_evo`, `moves_used`), the reducer's opponent half holds nothing; doctored mappings are refused by the reader.
- `tests/contracts/test_battle_state_spec.py`: the Python reader on the samples and the committed record.

**How equality with today's encoder is checked (specified here, executed in G4).** For every request of both seats of every committed corpus: pack `toJSON()` plus the seat bookkeeping into spec tensors with a CPU reference packer, project for the seat, rebuild the reducer-state view the encoder reads, and require the state part of `Encoder.encode` bit-equal to encoding the real `PolicyQuery` built from the Showdown channel.
