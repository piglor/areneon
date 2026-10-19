# Stat Point formula and turn-order evidence

Research for Redmine #680 (Notion Wave 1 gate). It feeds the Wave 2 go/no-go (#683) and the gap ledger (`rules/gaps/`), and extends the format matrix ([regmc-format-matrix.md](regmc-format-matrix.md), tasks U5 and D7).

**Scope.** How the pinned engine turns a set's Stat Points, nature and IVs into stats for Reg M-C, whether that matches what the game displays, where the Stat Point caps come from, and how the engine orders actions within a turn. The engine is the only mechanics authority for the offline training pipeline (ADR-003); this note records how far its Champions behaviour is checked against the game, and what is still assumed.

**How to read.**

- Engine facts cite `file:line` in the pinned submodule `vendor/pokemon-showdown` at `7244fe8e28fa8bc4d0809ae30d183c608b62bb2b` [S1]. Paths are relative to that checkout.
- Game facts cite a source label `[S#]` ([Sources](#6-sources)). All sources were accessed on **2026-10-19**.
- Engine outputs come from running the built pin (`make showdown`); the script logic and its output are in [§3.2](#32-engine-run).
- Facts are what a source shows or the pinned code does. Anything derived from them is marked **Inference**.
- A behaviour with no primary game source is **unverified** and has a verification task in [§5.4](#54-unknowns-and-verification-tasks).

## 1. Decision questions

| # | Question | Answer | Status |
|---|---|---|---|
| Q1 | Does the champions mod compute the same stats the game displays, for every stat, Stat Point value and nature direction in the sample? | Yes. 5 Pokémon read from official screenshots, 30 stat values, 0 mismatches ([§3](#3-comparison-with-in-game-values)); 2 more cases whose species is not named on screen are consistent with the formula but are not independent evidence (§3.1). | Checked against official screenshots, not against a captured in-game fixture |
| Q2 | Are the caps 32 per stat and 66 in total? | Engine: yes, enforced by the validator ([§4.1](#41-engine)). Game: the Stat Points screen shows a total of `66/66` and a 32-cell gauge per stat ([§4.2](#42-game)). | Agree. That the game *refuses* a 33rd point in one stat is not shown (task V1) |
| Q3 | Are IVs fixed at 31? | Engine: the validator requires 31 in every stat, and the champions formula ignores IVs. Game: no screen shows IVs; every displayed stat equals the IV-31 value. | Consistent, not stated by any primary source (task V2) |
| Q4 | What does the engine do for turn order, and what do the Champions rules change? | Priority, then speed, then a random shuffle of exact ties; speed is recomputed before each move; Trick Room reverses speed with no overflow in the champions mod; paralysis halves speed and fully paralyses 1/8 of the time ([§5](#5-turn-order)). | Engine read and run; game side mostly unverified (tasks V3-V12) |

**What stays unverified.** Every turn-order mechanic in [§5](#5-turn-order) except two official notices (a fixed switch-in ability order bug and a known issue with simultaneous Mega Evolution); the per-stat refusal at 33; IVs; a nature that changes Sp. Def (no sample has one); and any Pokémon not in the sample. The screenshots are promotional material from The Pokémon Company, not a fixture captured by Areneon, and one publisher marks its screenshots as "from a product in development and not final" [S2].

## 2. The formula in the pinned engine

### 2.1 Code path

1. A battle builds each Pokémon's stored stats through `Battle.spreadModify` (`sim/battle.ts:2350-2356`, called from `sim/pokemon.ts:1398`), which calls `this.statModify(baseStats, set, statName)` for each stat. `statModify` is a mod hook (`sim/global-types.ts:387`), and the champions mod replaces it.
2. `data/mods/champions/scripts.ts:10-40`, `statModify`:
   - `:12-13` reads the base stat and `set.evs[statName]`. In the champions mod `evs` holds Stat Points (CONTEXT.md).
   - `:14-22` applies only when the format has `Level Clause Mod` (not Reg M-C): the mainline level formula with `Math.max(2 * evs - 1, 0)` in place of `floor(EV / 4)` (the comment says "The first stat point gives 4 EVs and the others give 8 EVs").
   - `:23-27` is the Reg M-C path: HP is `stat + evs + 75`; every other stat is `stat + evs + 20`.
   - `:28-39` applies the nature: `tr(tr(stat * 110, 16) / 100)` for the raised stat and `tr(tr(stat * 90, 16) / 100)` for the lowered one (`tr` truncates; the 16-bit wrap matters only above 595, `:30-31`).
3. The level is not read on the Reg M-C path: `+20` / `+75` are the level-50 constants. `Adjust Level = 50` in the champions `Flat Rules` (`data/mods/champions/rulesets.ts:31`) sets the battle level.
4. Base stats come from the standard Gen 9 Pokédex (`data/pokedex.ts`): the champions mod has no `pokedex.ts` of its own (`data/mods/champions/` holds `abilities`, `conditions`, `formats-data`, `items`, `learnsets`, `moves`, `rulesets`, `scripts`).

So, at level 50, with SP the Stat Points of that stat and N the nature multiplier:

| Stat | Champions mod (Reg M-C path) |
|---|---|
| HP | `base + SP + 75` |
| Atk, Def, SpA, SpD, Spe | `floor((base + SP + 20) × N)`, N ∈ {0.9, 1.0, 1.1} |

### 2.2 Stat Points, nature and IVs

| Input | Engine handling | Where |
|---|---|---|
| Stat Points | Read from the set's `evs`. Per stat at most 32 and in total at most 66, enforced by the team validator, not by the formula (§4.1). PokePaste carries them on the `EVs:` line (gap record `pokepaste`). | `data/mods/champions/scripts.ts:13`; `sim/team-validator.ts:1313-1318`, `:1357-1363` |
| Nature | Same table and the same truncating multiply as the standard formula. Neutral natures change nothing. `Serious` with 0 Stat Points is refused as a probable mistake; another neutral nature is accepted. | `data/mods/champions/scripts.ts:28-39`; `sim/team-validator.ts:1339-1343` |
| IVs | The Reg M-C formula never reads `set.ivs`. The validator refuses any IV other than 31: "this format requires all IVs to be 31". | `data/mods/champions/scripts.ts:23-27`; `sim/team-validator.ts:1165-1168` |
| Level | Not read on the Reg M-C path; the battle level is 50 by `Adjust Level = 50`. | `data/mods/champions/scripts.ts:23-27`; `data/mods/champions/rulesets.ts:31` |

### 2.3 Difference from the standard formula

The standard formula (`sim/battle.ts:2358-2374`) is `floor(floor(2·base + IV + floor(EV/4)) · level/100 + 5)` (HP: `floor(floor(2·base + IV + floor(EV/4) + 100) · level/100 + 10)`), then the nature.

**Inference (algebra, no source states it).** At level 50 and IV 31, the standard value is `floor(base + 15.5 + floor(EV/4)/2) + 5`. With EV = 8·SP − 4 for SP ≥ 1 (and EV = 0 for SP = 0), `floor(EV/4) = 2·SP − 1` and the value is exactly `base + SP + 20` (HP: `base + SP + 75`). The champions formula is therefore the standard level-50, IV-31 formula under the mapping "first Stat Point = 4 EVs, each further point = 8 EVs", the same mapping the mod's own `Level Clause Mod` branch states. What differs is the budget: 32 SP equals 252 EVs, but a 66-point total spread as 32/32/2 equals 252 + 252 + 12 = 516 EVs, above the standard 510 cap; the champions mod replaces the 510 limit with 66 (`sim/dex-formats.ts:323-330`). Secondary guides differ on the EV conversion; none is used here.

## 3. Comparison with in-game values

### 3.1 Reference cases

Each case is one Pokémon on an official screenshot of the in-game Stat Points (training) or recruit screen, which shows the species, the Stat Points per stat, the stat alignment (nature) and the six resulting stats. All are level 50 (the game has no other level in battle, [regmc-format-matrix.md](regmc-format-matrix.md) T4). Base stats are the pinned engine's. "Engine" is the output of [§3.2](#32-engine-run).

| Case | Species (base HP/Atk/Def/SpA/SpD/Spe) | Stat Points HP/Atk/Def/SpA/SpD/Spe | Nature (+ / −) | Displayed in game | Engine | Source |
|---|---|---|---|---|---|---|
| C1 | Empoleon (84/86/88/111/101/60) | 32/0/0/32/0/2 | Modest (+SpA, −Atk) | 191/95/108/179/121/82 | 191/95/108/179/121/82 | [S2] training image `p04_01.jpg`; [S3] training image (same values) |
| C2 | Whimsicott (60/67/85/77/75/116) | 32/0/32/0/0/2 | Bold (+Def, −Atk) | 167/78/150/97/95/138 | 167/78/150/97/95/138 | [S2] recruit image `p02_02.jpg`; [S3] recruit image (same values) |
| C3 | Pikachu (35/55/40/50/50/90) | 0/32/2/0/0/32 | Jolly (+Spe, −SpA) | 110/107/62/63/70/156 | 110/107/62/63/70/156 | [S4] screenshot `ENG_04` (iPhone and iPad) |
| C4 | Gardevoir (68/65/65/125/115/80) | 0/0/0/32/2/32 | Timid (+Spe, −Atk) | 143/76/85/177/137/145 | 143/76/85/177/137/145 | [S5] "train" image |
| C5 | Arcanine (90/110/80/100/80/95) | 0/2/32/32/0/0 | Relaxed (+Def, −Spe) | 165/132/145/152/100/103 | 165/132/145/152/100/103 | [S5] store screenshot `f8c54a3f91ac…` (full URL in S5) |
| C6 | not named on screen (assumed Pikachu, see note) | 26/0/19/0/14/3 | Jolly | 136/75/79/63/84/124 | 136/75/79/63/84/124 | [S4] screenshot `ENG_05` (iPhone) |
| C7 | not named on screen (assumed Pikachu, see note) | 26/0/17/0/14/3 | Jolly | 136/75/77/63/84/124 | 136/75/77/63/84/124 | [S4] screenshot `ENG_05` (iPad) |

**Result: no mismatch.** All 30 values of C1-C5 are equal. C6-C7 are consistent with the formula, not independent evidence: their species is not named on screen, so their base stats were inferred from the very numbers being compared (plus a sprite edge), which makes the match circular. Per stat, C1-C5 give five named reference values each:

| Stat | Cases | Stat Point values covered | Nature effect covered |
|---|---|---|---|
| HP | C1-C5 | 0, 32 | n/a |
| Atk | C1-C5 | 0, 2, 32 | −10 % (C1, C2, C4), neutral (C3, C5) |
| Def | C1-C5 | 0, 2, 32 | +10 % (C2, C5), neutral (C1, C3, C4) |
| SpA | C1-C5 | 0, 32 | +10 % (C1), −10 % (C3), neutral (C2, C4, C5) |
| SpD | C1-C5 | 0, 2 | neutral only |
| Spe | C1-C5 | 0, 2, 32 | +10 % (C3, C4), −10 % (C5), neutral (C1, C2) |

Notes on the cases:

- **Rounding.** C4 Atk `(65 + 0 + 20) × 0.9 = 76.5` displays 76 and C5 Spe `(95 + 0 + 20) × 0.9 = 103.5` displays 103: the game truncates the nature product, as the engine does.
- **C4 and C6/C7 are edit previews.** Their screens show pending costs (red VP tags) next to some rows, so the values are those of the Stat Points and nature being set, not yet confirmed. C6 and C7 are two captures of the same edit, one Defense point step apart (19 → 79, 17 → 77). C6/C7 do not name the species; their base stats are inferred from the numbers and a Pikachu sprite edge, so their match is circular. They show only that the Defense step (2 SP → 2 stat points) behaves as the formula says, and are not counted as reference values.
- **Builds.** [S2] shows Empoleon's moves with PP 5/10/10/10, [S3] with 8/12/12/12 (the current `pp` formula, gap record `pp`), so the two were captured on different builds; the stats are the same in both.
- **Gaps in the sample.** No sample has a nature that changes Sp. Def, and HP and SpA have no small nonzero value (task V2b). A neutral nature is only seen as the unchanged stats of a non-neutral one; no neutral-nature Pokémon is in the sample.
- **Community corroboration.** The Smogon "Champions Battle Mechanics Research" thread [S8] gives the same formula (HP = base + SP + 75; others = nature × (base + SP + 20)), credited to a contributor and taken from a game client data dump. It gives no worked examples; it is not counted as a reference case.

### 3.2 Engine run

Throwaway script (not committed), run with Node 24 against the built pin through `loadEngine()`:

1. For each case, build a set `{species, nature, evs: <Stat Points>, ivs: all 31, level: 50, moves: ["protect"]}`.
2. Start `new Battle({formatid: "gen9championsvgc2026regmc", seed: "sodium,00000000000000000000000000000000"})` (stats do not depend on the seed), `setPlayer` both sides with the packed team, and read p1's Team Preview request.
3. For each Pokémon of `request.side.pokemon`, the engine's HP is the max in `condition` (`hp/maxhp`) and the other five are `stats`. Compare to the displayed values.

Output at `7244fe8e`:

```
Empoleon Modest   base 84/86/88/111/101/60  engine 191/95/108/179/121/82  shown 191/95/108/179/121/82  MATCH
Whimsicott Bold   base 60/67/85/77/75/116   engine 167/78/150/97/95/138   shown 167/78/150/97/95/138   MATCH
Pikachu Jolly     base 35/55/40/50/50/90    engine 110/107/62/63/70/156   shown 110/107/62/63/70/156   MATCH
Gardevoir Timid   base 68/65/65/125/115/80  engine 143/76/85/177/137/145  shown 143/76/85/177/137/145  MATCH
Arcanine Relaxed  base 90/110/80/100/80/95  engine 165/132/145/152/100/103 shown 165/132/145/152/100/103 MATCH
Pikachu Jolly     (C6)                      engine 136/75/79/63/84/124    shown 136/75/79/63/84/124    MATCH
Pikachu Jolly     (C7)                      engine 136/75/77/63/84/124    shown 136/75/77/63/84/124    MATCH
```

The same assertion for C1-C5 is kept as a regression test, `packages/showdown-bridge/test/champions-stats.test.ts` (it reads the stats from the Team Preview request, the public request seam), so a pin bump that changes `statModify` fails CI.

**Speed-tie run (E3).** Same script, format `gen9championsdoublescustomgame@@@!teampreview`. Each side: a Jolly Pikachu with 32 Stat Points in Speed (156 Speed) in slot 1 and a Wobbuffet with 0 Speed Stat Points in slot 2; every Pokémon knows only Splash and uses it on turn 1. Seeds: `sodium,` followed by the run index `i` as 32 hexadecimal digits, `i` = 0 … 1999. Counted: whose Pikachu has the first `|move|` line. Result on the pin: p1 first in 987 of 2,000 (49.4 %). The exact count depends on the seed list (an independent rerun with other seeds gave 1,002 of 2,000); the claim is only that the rate is about 50 %, as the PRNG shuffle implies.

## 4. Stat Point caps

### 4.1 Engine

| Rule | Value | Where |
|---|---|---|
| Stat Points are used for any mod whose name starts with `champions` | `useStatPoints = dex.currentMod.startsWith('champions')` | `sim/team-validator.ts:1155` |
| Per stat | more than 32 is refused: "has more than 32 Stat Points in HP." | `sim/team-validator.ts:1313-1318` |
| Total | `EV Limit = Auto` resolves to 66 for `champions*` mods; more is refused: "has 67 total Stat Points, which is more than this format's limit of 66." | `sim/dex-formats.ts:323-330`, `sim/team-validator.ts:1357-1363`; `EV Limit = Auto` comes from `Obtainable` (`data/rulesets.ts:170`) |
| IVs | all six must be 31 | `sim/team-validator.ts:1165-1168` |
| Zero Stat Points | refused only with `Serious` | `sim/team-validator.ts:1339-1343` |

Messages above are from running `TeamValidator.get("gen9championsvgc2026regmc").validateTeam` on the pin (33 in HP; 32/32/3 = 67 in total; 32/32/2 = 66 is accepted). The committed team fixtures check the same verdicts (`fixtures/teams/gen9championsvgc2026regmc/manifest.json`: `illegal-stat-points-over-32`, `illegal-stat-points-over-total`, `illegal-zero-stat-points`).

### 4.2 Game

| Claim | What the primary source shows | Source |
|---|---|---|
| Total 66 | The Stat Points panel header reads `66/66` (C1, C4) and `62/66`, `60/66` while editing; the edit pop-up reads "Total Stat Points 62/66". | [S2] `p04_01.jpg`; [S5] "train" image; [S4] `ENG_05` |
| Per stat 32 | The per-stat gauge in the edit pop-up has 32 cells (19 filled, 4 dimmed for the 4 unspent points of the total, 9 empty at 19 of 32). Every maximised stat in the sample shows 32 with a full bar (C1-C5). | [S4] `ENG_05` (iPhone); [S2], [S4], [S5] |
| Stat Points replace stat training | The official site: "By using VP, you can hone your Pokémon's Attack, Defense, and other stats." It states no number. | [S2] "Train Your Pokémon" |

**Where engine and game agree or differ.** They agree on both numbers. Neither official page states the caps in text: the evidence is the game's own screens. What is not shown is the game refusing a 33rd point in one stat, or what happens when 66 is spent (the dimmed cells suggest the gauge stops at the remaining total: Inference). The format matrix's "no primary source for the caps" (U5, D7) is answered for the total and, by the gauge, for the per-stat value; the refusal itself stays task V1.

## 5. Turn order

### 5.1 What the engine does

| # | Mechanic | Engine behaviour at the pin (Reg M-C) | Where |
|---|---|---|---|
| E1 | Sort order | Actions sort by `order` (low first: switches 103, Mega Evolution 104, moves 200), then `priority` (high first), then `speed` (high first), then `subOrder`, `effectOrder`. | `sim/battle.ts:397-415` (`comparePriority` and its doc comment); `sim/battle-queue.ts:172-203` |
| E2 | Priority | A move's priority is its base priority, changed by `ModifyPriority` handlers (Prankster +1 for Status moves, Gale Wings, Triage, Grassy Glide), plus a fractional part from `FractionalPriority` (Quick Claw +0.1 on a 20 % roll, Stall −0.1, Mycelium Might). | `sim/battle.ts:2644-2651`; `sim/battle-queue.ts:249`; `data/abilities.ts:3426-3436`; `data/items.ts:5013-5021` |
| E3 | Speed tie | Actions that compare equal are shuffled with the battle PRNG (Fisher-Yates), so each order of tied actors is equally likely. A seeded run gave p1 the first move in about half of 2,000 battles (recipe and seeds in [§3.2](#32-engine-run); the exact count depends on the seeds). | `sim/battle.ts:433-463` (`speedSort`); `sim/prng.ts:145-158` |
| E4 | Speed used for ordering | `getActionSpeed()`: the Speed stat with boosts and all `ModifySpe` handlers (`getStat('spe')`), capped at 10,000. | `sim/pokemon.ts:596-639`, `:641-649`; champions override `data/mods/champions/scripts.ts:45-54` |
| E5 | Speed recomputed mid-turn | Gen 8+: before each move the engine recomputes every active Pokémon's speed and re-sorts the queue, so a speed change (e.g. Icy Wind, Electroweb, a Mega Evolution) affects actors that have not moved yet. Residual effects also re-read speed. | `sim/battle.ts:2917-2924`; `:2818` (residual) |
| E6 | Trick Room | Priority −7, 5 turns (7 with Persistent). **Champions mod change:** action speed is `-speed` under Trick Room. The standard engine uses `10000 - speed` truncated to 13 bits, so a Pokémon with Speed ≥ 1809 wraps and acts first under Trick Room; the champions mod has no such wrap (its comment: "Remove Trick Room underflow"). Run on the pin: under Trick Room, Speed 1809 → 8191 (acts first) in `gen9doublesou`, → −1809 (acts last) in Reg M-C; without Trick Room, the standard 13-bit wrap (8192 → 0) is also gone in the champions mod. | `data/moves.ts:19935-19946`; `sim/pokemon.ts:641-649`; `data/mods/champions/scripts.ts:45-54` |
| E7 | Tailwind | Doubles the side's Speed (`chainModify(2)`) for 4 turns including the turn it is used. Not overridden by the champions mod. | `data/moves.ts:18870-18898` |
| E8 | Paralysis | Speed × 0.5, applied after every other Speed modifier (`onModifySpePriority: -101`), not with Quick Feet (inherited). **Champions mod change:** full paralysis chance 1/8 (standard 1/4). Run on the pin: a Jolly 32-SP Persian, 183 Speed, paralysed: 91; at +1: 137. | `data/conditions.ts:20-45`; `data/mods/champions/conditions.ts:2-10` |
| E9 | Speed items | Choice Scarf × 1.5 (183 → 274), Iron Ball × 0.5 (183 → 91); Tailwind with Scarf: 549. Lagging Tail, Full Incense, Custap Berry, Salac Berry, Quick Powder, Macho Brace, Power Weight and Room Service are marked `Past` (not available) in the champions dex; Quick Claw is available. | `data/items.ts:995-1015`, `:3062-3078`, `:5013-5021`; `Past` set by the champions mod: `data/mods/champions/items.ts:194` (Custap Berry), `:510` (Lagging Tail), `:722` (Power Weight), `:798` (Room Service), `:826` (Salac Berry); inherited from the base data: `data/items.ts:2357-2366` (Full Incense), `:3668-3680` (Macho Brace), `:5030-5044` (Quick Powder) |
| E10 | Mega Evolution | A Mega Evolution is its own action with `order` 104, so all Mega Evolutions resolve before any move, in speed order; with E5 the Mega's new Speed applies to that turn's moves. Run on the pin: a Gengar Mega Evolving (130 → 150 Speed) moved before an opposing 130-Speed Gengar on the same turn. | `sim/battle-queue.ts:182-184`, `:212-217`; `sim/battle.ts:2917-2924` |
| E11 | Switch-in ability order | Abilities that trigger on entry run in current speed order (`eachEvent` sorts actives by `pokemon.speed`, which includes item modifiers). | `sim/battle.ts:468-472`; `sim/pokemon.ts:556-558` |
| E12 | Encore | **Champions mod change:** Encore applied before the target moves replaces its queued action with the Encored move, re-resolving its priority (the upstream tests document this as "restore the priority of the originally selected move"). | `data/mods/champions/moves.ts:309-340`; `test/sim/champions.js:676-704` |

The upstream Champions test file states its own limit: "do not treat these expectations as authoritative without verifying them against the latest version of the game" (`test/sim/champions.js:6-9`). It has no test for stats, speed ties, Trick Room, Tailwind or paralysis speed.

### 5.2 What the game is documented to do

| # | Mechanic | Primary source | Agrees with the engine? |
|---|---|---|---|
| G1 | Switch-in ability order with held-item Speed | Ver. 1.0.3 update notice fixes: "The order in which Abilities trigger is not reflecting changes to the Speed stat caused by held items." [S7] | Yes after 1.0.3: E11 uses item-modified speed. Before 1.0.3 the game differed. The community thread lists the same bug for Choice Scarf [S8]. |
| G2 | Simultaneous Mega Evolution | Known issue: "When both players' Pokémon Mega Evolve simultaneously under certain conditions, the turn order may behave incorrectly." [S6] Not in the 1.0.3 fix list [S7]. | Unknown. The engine orders Megas by speed (E10). The community thread says Mega Evolution order was random and marks it fixed on 2026-04-09 [S8] (community, not official). |
| G3 | Full paralysis 1/8 | No official text found. The community thread lists it [S8]. | Matches E8 per community only. |
| G4 | Trick Room has no high-Speed overflow | No official text found. The community thread: the ">1809 Speed overflow doesn't exist" [S8]. | Matches E6 per community only. |
| G5 | Speed ties, priority brackets, Tailwind, Choice Scarf, Iron Ball, Quick Claw, paralysis Speed, dynamic speed | No primary Champions source found. The engine's speed-tie shuffle cites a 2016 comment about the mainline games that gives no evidence [S9]. | Unverified. |

The regulation pages and the VGC handbook (format matrix S2, S4) say nothing about turn order beyond Mega Evolution once per battle.

**Server-side caveat.** The community thread [S8] warns that battle calculations run server-side and can change without a game version update, so findings should be re-tested often. Any in-game observation in this section, and every V-task result, is therefore valid only for the date it was made, not for a game version; record the date with it and re-test after each maintenance or update notice.

### 5.3 What the Champions rules change

From the champions mod at the pin and the gap ledger:

| Change versus standard Gen 9 | Engine | Gap record | Game evidence |
|---|---|---|---|
| Trick Room: no speed wrap | E6 | `trick-room` ("No speed underflow") | community [S8] |
| Paralysis: full paralysis 1/8; Speed halving unchanged | E8 | `paralysis` ("Full paralysis chance 1/8") | community [S8] for 1/8; none for Speed |
| Encore re-resolves priority | E12 | none (`override-parity` covers move overrides in general) | community [S8] |
| Fake Out cannot be selected after the first turn out | `data/mods/champions/moves.ts:349-355` | `override-parity` | community [S8] |
| Speed items removed (Lagging Tail, Full Incense, Custap and Salac Berry, …) | E9 | `roster-and-learnsets` / `move-item-ability-overrides` (items list) | not compared |
| No Terastallization, Dynamax or Z-Moves, so none of their order slots occur | `sim/battle-queue.ts:185-187` never reached | `terastallization` | format matrix T11 |

### 5.4 Unknowns and verification tasks

Each task names what to test, the expected result if the engine is right, and how to record it. Record the date and time of every test as well as the game version, and repeat the test after maintenance, since server-side battle calculations can change without a version update ([§5.2](#52-what-the-game-is-documented-to-do)). In-game evidence goes in as a sanitized, synthetic-team fixture under `fixtures/` with provenance (game version, platform, date, regulation), and a gap record cites it with `kind: fixture`; the gap ledger's rule is that only such a fixture makes a row `verified`.

| Task | Unknown | Test | Expected (engine) | Record |
|---|---|---|---|---|
| V1 | Per-stat refusal | Training screen: put 32 SP in one stat and try to add a 33rd; spend 66 and try to add one more anywhere. | The gauge stops at 32; nothing beyond 66 is accepted. | Screenshots plus a JSON note (game version, species, SP before and after) under `fixtures/ingame/stat-points/`. |
| V2 | IVs fixed at 31 | Bring a Pokémon with known low IVs from Pokémon HOME, and read its stats in Champions. | Stats equal `base + SP + 20` (HP `+ 75`), i.e. IVs ignored. | Summary-screen values and the source IVs (from the source game) in a fixture; engine assertion added to `champions-stats.test.ts`. |
| V2b | Formula coverage | Read 3 more Pokémon: one with a +SpD nature, one with −SpD, one neutral nature, with small nonzero SP in HP and SpA. | Values from §2.1. | Add to the test's case list with the screenshot as a fixture. |
| V3 | Speed ties | Two Pokémon with identical displayed Speed, both using a non-priority move, ≥ 200 turns (or battles). | First mover ≈ 50 % each; with four tied actors, each order equally likely. | CSV per trial (turn, first mover, game version) as a fixture; a binomial test result in the gap record. |
| V4 | Trick Room order and duration | Trick Room with a fast and a slow Pokémon; count the turns it lasts. If reachable, test a Speed ≥ 1809 (e.g. +6 with Tailwind and Choice Scarf). | Slowest acts first in every turn, including ≥ 1809; lasts 5 turns including the turn used. | Battle log (turn, actor order, displayed Speeds) as a fixture. |
| V5 | Tailwind | Pokémon A slower than B, A's side sets Tailwind; check order and the turn count. | A outspeeds B when `2 × A > B`; lasts 4 turns including the turn used. | As V4. |
| V6 | Paralysis Speed | Paralysed Pokémon versus one with Speed between half and full of its Speed. | The paralysed one moves second (× 0.5 after other modifiers); Quick Feet ignores the drop. | As V4. |
| V7 | Full paralysis rate | ≥ 400 paralysed move attempts. | ≈ 12.5 % fully paralysed. | CSV per attempt; binomial interval in the gap record. |
| V8 | Choice Scarf, Iron Ball, Quick Claw | Holder versus a Pokémon with Speed just above the holder's unmodified Speed; Quick Claw: ≥ 200 turns. | Scarf × 1.5, Iron Ball × 0.5; Quick Claw lets the holder act first in ≈ 20 % of turns within its priority bracket. | As V4 / V7. |
| V9 | Mid-turn speed change | Icy Wind or Electroweb by a fast Pokémon on a target that was faster than its partner's target; check whether the slowed target moves later that same turn. | The order changes the same turn (E5). | As V4. |
| V10 | Mega Evolution order | Both players Mega Evolve on the same turn, Megas with different Speeds; also a Mega that becomes faster than an opponent. | Megas resolve in speed order; the new Speed applies that turn. Compare with the known issue [S6]. | Battle log with game version; recheck after each update notice. |
| V11 | Switch-in ability order | Two entry abilities (e.g. Intimidate) on Pokémon whose order flips only with a Choice Scarf. | Item-modified speed decides (post-1.0.3, [S7]). | As V4. |
| V12 | Encore and priority | Encore a target that selected a priority move into a non-priority move, and the reverse. | The Encored move's priority applies (E12). | As V4. |

## 6. Sources

Primary sources first. All accessed on 2026-10-19. Screenshots are cited for the numbers they display; nothing from them is copied into the repository.

| Label | Title | URL | Version / date | Claim it supports |
|---|---|---|---|---|
| S1 | Pokémon Showdown, pinned submodule (MIT) | <https://github.com/smogon/pokemon-showdown/tree/7244fe8e28fa8bc4d0809ae30d183c608b62bb2b> | commit `7244fe8e`, 2026-10-16 | Every engine fact and engine output (§2, §3.2, §4.1, §5.1) |
| S2 | Pokémon Champions official site, "Pokémon" page ("Recruiting Pokémon", "Train Your Pokémon") | <https://champions.pokemon.com/en-us/pokemon/>; images `https://champions.pokemon.com/_images/pokemon/train-pokemon/en/p04_01.jpg`, `…/recruiting/en/p02_02.jpg` | undated; the page footer says "Screenshots and game footage are from a product in development and not final." | C1, C2; `66/66` total; training hones "Attack, Defense, and other stats" |
| S3 | Pokémon Champions official Japanese site, 育成について and ポケモンについて pages | <https://www.pokemonchampions.jp/ja/training/>, <https://www.pokemonchampions.jp/ja/pokemon/>; images `/assets/img/training/ja/image-training-1@2x.webp`, `/assets/img/pokemon/ja/image-scout-2@2x.webp` | undated; PP values match the current `pp` formula | C1, C2 on a second build |
| S4 | Pokémon Champions, Apple App Store listing (The Pokémon Company) | <https://apps.apple.com/us/app/pok%C3%A9mon-champions/id6741503079> | listing shows app version 1.2.2; screenshots `ENG_04_POKE_CHAMP_*` and `ENG_05_POKE_CHAMP_*` (iPhone 2868×1320, iPad 2752×2064) | C3, C6, C7; 32-cell per-stat gauge; "Total Stat Points 62/66" |
| S5 | Pokémon Champions, Nintendo eShop product page | <https://www.nintendo.com/us/store/products/pokemon-champions-switch/>; images `https://assets.nintendo.com/image/upload/Microsites/pokemon-champions/switch-1/train.png`, store screenshot <https://assets.nintendo.com/image/upload/ar_16:9,c_lpad,w_1240/b_white/f_auto/q_auto/store/software/switch/70010000004519/f8c54a3f91ac2184747e62684947e88f3c2c8b0796947fb8489a074c7168ff0e> | undated | C4, C5; `66/66` |
| S6 | Pokémon Champions news, "Regarding Current Issues" | <https://champions-news.pokemon-home.com/en/page/758.html> | news index start date 2026-04-09 | Known issue: turn order with simultaneous Mega Evolution |
| S7 | Pokémon Champions news, "Pokémon Champions Ver. 1.0.3 Update Notice" | <https://champions-news.pokemon-home.com/en/page/764.html> | news index start date 2026-04-23 | Fixed: ability trigger order ignored held-item Speed changes |
| S8 | Smogon forums, "Champions Battle Mechanics Research" (community research thread; method stated in the first post: a game client data dump, plus in-game tests) | <https://www.smogon.com/forums/threads/champions-battle-mechanics-research.3780372/> (first post: `post-10943244`) | started 2026-04-07, first post edited 2026-04-25 | Community: stat formula, full paralysis 1/8, no Trick Room overflow, Encore priority, Fake Out, Mega order bug. Corroboration only |
| S9 | smogon/pokemon-showdown issue #1157, comment 214454873 | <https://github.com/smogon/pokemon-showdown/issues/1157#issuecomment-214454873> | 2016-04-25 | The speed-tie procedure the engine cites (`sim/prng.ts:145-149`); about the mainline games, no evidence given in the thread |
| S10 | Secondary, used only to find primaries: Bulbapedia "Stat point", Serebii "Training Mechanics", ChampDex "Stat Points" | <https://bulbapedia.bulbagarden.net/wiki/Stat_point>, <https://www.serebii.net/pokemonchampions/training.shtml>, <https://champdex.com/guides/stat-points> | various | Pointed to the caps and conversions; no row relies on them (Bulbapedia returned 403 to a plain fetch) |

## 7. What this changes

### 7.1 Gap ledger (proposed edits; `rules/gaps/` is not edited here)

Editing these records moves `rules_snapshot_hash` (never `format_rules_hash`) and needs its own PR with the regenerated `docs/showdown-champions-gaps.md`. None of them can become `verified`: the ledger schema requires an in-game fixture for that, and screenshots published by The Pokémon Company are not one.

| Record | Proposed change |
|---|---|
| `stat-points` | Keep `unverified`. Replace "no official source states them (format matrix U5)" with: the formula matches 30 stat values of 5 Pokémon on official screenshots (this note, `champions-stats.test.ts`); the game's Stat Points screen shows a 66 total and a 32-cell gauge per stat; the refusal at 33 and the IVs are still unconfirmed (tasks V1, V2). |
| `trick-room` | Expand the description: action speed is `-speed`, so no Speed wraps (standard: `10000 - speed` in 13 bits, Speed ≥ 1809 acts first); priority −7, 5 turns; community research reports the same [S8]; task V4. Add `data/moves.ts` `trickroom` and `sim/pokemon.ts` `getActionSpeed` to the evidence. |
| `paralysis` | Add that Speed is still halved after all other modifiers (inherited from `data/conditions.ts` `par`), Quick Feet exempt; both unverified in game (tasks V6, V7). Add `data/conditions.ts` `par` to the evidence. |
| `pp` | Note that official screenshots on a later build show 8/12/16/20 PP for base 5/10/15/20 moves (S3, S4), matching `calculatePP`; still no fixture. |
| new `turn-order` | "Actions sort by order, priority, speed; exact ties are shuffled by the PRNG; speed is recomputed before each move (Gen 8+). No primary Champions source; Ver. 1.0.3 fixed the switch-in ability order to use held-item Speed, which the engine already does." Evidence: `sim/battle.ts` `comparePriority`, `speedSort`, `sim/prng.ts` `shuffle`. Tasks V3, V5, V8, V9, V11. |
| new `mega-evolution-order` | "Megas resolve before moves in speed order and the new Speed applies the same turn. The game lists a known issue with simultaneous Mega Evolution turn order (news 2026-04-09), not in the 1.0.3 fixes." Evidence: `sim/battle-queue.ts` `resolveAction`. Task V10. Alternatively fold into `mega-evolution`. |
| `override-parity` | Name Encore's re-resolved priority and Fake Out's first-turn-only selection as examples of turn-order overrides to check (task V12). |

### 7.2 Wave 2 go/no-go (#683)

- **Stat formula gate item: evidence exists.** The engine's champions formula reproduces every displayed value in the sample (0 mismatches over 30 values of five named Pokémon; two unnamed cases are consistent but circular). The remaining risk is coverage (no Sp. Def nature, five species), not a known difference. A regression test now ties the pin to those values.
- **Caps gate item: evidence exists, one open point.** Engine and game agree on 66 total and 32 per stat; the game's refusal of a 33rd point is shown only by the gauge size (V1).
- **Turn order: engine evidenced, game mostly unverified.** No source shows a difference from the engine; two official notices (S6, S7) concern Mega Evolution order and switch-in ability order, and the second is already the engine's behaviour. The unknowns (V3-V12) do not block offline self-play, where the pinned engine is the authority by definition (ADR-003). They limit any claim that results transfer to the game, so results should carry the new `turn-order` and `mega-evolution-order` gap labels once they exist.
- **Prioritized questions for #683:** V10 (a documented open issue in the game), V3 (speed ties decide many doubles turns), V1 and V2 (cheap to check on a training screen), then V4-V9, V11, V12.
