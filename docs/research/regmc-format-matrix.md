# Reg M-C target format matrix

Research for Redmine #772 (Notion Wave 1 gate items 1-2). Gap reconciliation against this matrix is #773.

**Target format.** Areneon trains for Pokémon Champions **Regulation Set M-C**, played as VGC **Doubles**, **bring 6, choose 4**, **Bo1 with a Closed Team Sheet (CTS)**: Showdown's opt-in Open Team Sheets are declined by both players. This note fixes what that target means, row by row, and where each fact comes from.

**How to read the tables.**

- Every row has a source label `[S#]` (see [Sources](#sources)) and the date the row was verified.
- Engine rows cite `file:line` in the pinned Showdown submodule `vendor/pokemon-showdown` at `7244fe8e28fa8bc4d0809ae30d183c608b62bb2b` [S1]. Paths are relative to that checkout.
- A row whose fact has no primary source says **unverified** and gives a verification task. All of them are collected in [§8](#8-unverified-rows-and-verification-tasks).
- Facts are what a source says or the pinned code does. Anything derived from them is in [Inference](#inference) (end of §7), never in a fact row.
- Official rows come from two kinds of primary source, which do not describe the same thing:
  - **in-game Ranked Battles** rules for M-C ([S2], [S3], [S6]);
  - **Play! Pokémon VGC tournament** rules ([S4], [S5]), which apply to every Champions regulation set.

  Where they differ, both are given.

## 1. Ruleset identifiers

| # | Item | Value | Source | Verified |
|---|---|---|---|---|
| R1 | Official regulation name | "Regulation Set M-C", announced "for Ranked Battles" | [S2] title and lead | 2026-10-19 |
| R2 | Official regulation series | Champions competitions use "designated series", each divided into regulation sets lasting one to three months, announced up to 30 days before first legal competition | [S4] §2.1.1, p. 5 | 2026-10-19 |
| R3 | Showdown format name | `[Gen 9 Champions] VGC 2026 Reg M-C` | [S1] `config/formats.ts:288` | 2026-10-19 |
| R4 | Showdown format id | `gen9championsvgc2026regmc` (the id of R3); also the `format_id` of Areneon's format rules record | [S1] `config/formats.ts:288`; `rules/formats/gen9championsvgc2026regmc.json` | 2026-10-19 |
| R5 | Mod and game type | `mod: 'champions'`, `gameType: 'doubles'` | [S1] `config/formats.ts:289-290` | 2026-10-19 |
| R6 | Format ruleset | `['Flat Rules', 'VGC Timer', 'Open Team Sheets']`, `bestOfDefault: true` | [S1] `config/formats.ts:291-292` | 2026-10-19 |
| R7 | Bo3 sibling format | `[Gen 9 Champions] VGC 2026 Reg M-C (Bo3)`: `'Force Open Team Sheets', 'Best of = 3'`. Areneon does not run it; `forceopenteamsheets` is a forbidden rule in the format rules record. | [S1] `config/formats.ts:294-299`; `rules/formats/gen9championsvgc2026regmc.json` | 2026-10-19 |
| R8 | `Flat Rules` (champions mod) | `Obtainable`, `Team Preview`, `Species Clause`, `Nickname Clause`, `Item Clause = 1`, `Adjust Level = 50`, `Picked Team Size = Auto`, `Min Team Size = 6`, `Cancel Mod`; banlist `Mythical`, `Restricted Legendary` | [S1] `data/mods/champions/rulesets.ts:28-33` | 2026-10-19 |
| R9 | Resolved rule table at the pin | `flatrules, obtainable, obtainablemoves, obtainableabilities, obtainableformes, evlimit, obtainablemisc, -unreleased, -tag:unobtainable, -nonexistent, teampreview, speciesclause, nicknameclause, itemclause, adjustlevel, pickedteamsize, minteamsize, cancelmod, -tag:mythical, -tag:restrictedlegendary, vgctimer, timerstarting, timergrace, timeraddperturn, timermaxperturn, timermaxfirstturn, timeoutautochoose, dctimerbank, openteamsheets` | [S1] `getRuleTable` (`sim/dex-formats.ts:795`) on the built pin; numeric values from `resolveNumbers` (`sim/dex-formats.ts:251-333`) | 2026-10-19 |
| R10 | Regulation label in Areneon | `"regulation": "M-C"`, game type `doubles` | `rules/formats/gen9championsvgc2026regmc.json`, `rules/regulations/M-C.json` (cites [S2]) | 2026-10-19 |

## 2. Availability

| # | Item | Value | Source | Verified |
|---|---|---|---|---|
| AV1 | Reg M-C start | Wednesday 2026-09-09 02:00 UTC | [S2] "Regulation Set M-C Duration" | 2026-10-19 |
| AV2 | Reg M-C end | Wednesday 2026-12-02 01:59 UTC | [S2] "Regulation Set M-C Duration" | 2026-10-19 |
| AV3 | Roll-out | Maintenance from 2026-09-09 02:00 to 06:00 UTC added the new Pokémon and held items for M-C, with balance changes (e.g. Wish and Strength Sap PP 12 → 8) | [S3] | 2026-10-19 |
| AV4 | Areneon record | `season: {start: "2026-09-09T02:00Z", end: "2026-12-02T01:59Z"}` matches AV1-AV2 | `rules/formats/gen9championsvgc2026regmc.json` | 2026-10-19 |
| AV5 | Ranked Battles season(s) inside M-C | **Unverified.** Secondary sites name ranked seasons shorter than the regulation window (e.g. one ending 2026-10-07). No official season page was found. | none | task U1 |
| AV6 | Use of M-C at Play! Pokémon VGC events in the window | **Unverified.** [S4] says regulation sets are announced per series (R2) but names none. Secondary sources say VGC events from 2026-09-09 to 2026-12-02 use M-C. | none | task U2 |
| AV7 | Showdown availability | Format present at the pin, with mod `champions` (not the `championsregmb` mod of Reg M-B) | [S1] `config/formats.ts:287-293`, `:300-306` | 2026-10-19 |

## 3. Team rules

| # | Item | Official | Pinned engine | Source | Verified |
|---|---|---|---|---|---|
| T1 | Battle style | VGC: Double Battle; each player sends out their first two Pokémon | `gameType: 'doubles'` | [S4] §4.1, p. 11; [S1] `config/formats.ts:290` | 2026-10-19 |
| T2 | Team size registered | VGC: "between four and six Pokémon, depending on the tournament format". Ranked Battles: **unverified** (task U3) | Exactly 6: `Min Team Size = 6`, max team size defaults to 6 | [S4] §2.1, p. 5; [S1] `data/mods/champions/rulesets.ts:31`, `sim/dex-formats.ts:265-266` | 2026-10-19 |
| T3 | Pokémon chosen per game | VGC: "selects four Pokémon from their Battle Team", order decides the two leads | `Picked Team Size = Auto` → 4 for doubles; `\|teampreview\|4`; chosen order sets leads | [S4] §4.1, p. 11; [S1] `sim/dex-formats.ts:316-322`, `sim/battle.ts:1395-1401`, `sim/side.ts:1031-1071` | 2026-10-19 |
| T4 | Level | VGC: any level allowed, "auto-leveled to Lv. 50 during battle" | `Adjust Level = 50` | [S4] §2.3, p. 6; [S1] `data/mods/champions/rulesets.ts:31` | 2026-10-19 |
| T5 | Item clause | Ranked M-C: "Duplicate held items are not allowed." VGC: each Pokémon may hold an item, no two the same | `Item Clause = 1` (at most 1 of each item) | [S2] "Held Items"; [S4] §2.2, p. 5; [S1] `data/mods/champions/rulesets.ts:31`, `data/rulesets.ts:838-868` | 2026-10-19 |
| T6 | Species clause | VGC: no two Pokémon with the same National Pokédex number | `Species Clause`: rejects two sets with the same `species.num` | [S4] §2.3, p. 5; [S1] `data/rulesets.ts:788-805` | 2026-10-19 |
| T7 | Nicknames | VGC: no duplicate nicknames, no nickname that is another species' name | `Nickname Clause` | [S4] §2.3.1, p. 6; [S1] `data/rulesets.ts:806-825` | 2026-10-19 |
| T8 | Eligible Pokémon | Ranked M-C: only the Pokémon on the linked "Eligible Pokémon" page. VGC: "See Pokémon HOME for the current detailed list of eligible and restricted Pokémon." The list itself is **unverified** (task U4) | Champions dex: species not marked `isNonstandard` / `Illegal` in the mod's formats data, plus banlist `Mythical`, `Restricted Legendary` and `Obtainable` | [S2] "Eligible Pokémon"; [S6]; [S4] §2.1.1, p. 5; [S1] `data/mods/champions/formats-data.ts`, `data/mods/champions/rulesets.ts:32`, `data/rulesets.ts:166-171` | 2026-10-19 |
| T9 | Newly allowed Megas | Mega Absol Z, Mega Salamence, Mega Garchomp Z, Mega Lucario Z, Mega Golisopod, Mega Baxcalibur | All six exist and are not nonstandard in the champions mod (`Absol-Mega-Z`, `Salamence-Mega`, `Garchomp-Mega-Z`, `Lucario-Mega-Z`, `Golisopod-Mega`, `Baxcalibur-Mega`) | [S2] "Mechanics"; [S1] `data/mods/champions/formats-data.ts:1961`, `:2023`, `:2322`, `:2335`, `:3828`, `:4720` | 2026-10-19 |
| T10 | Mega Evolution limit | "You can use Mega Evolution only one time per battle." | After one Mega, `canMegaEvo = false` for the whole side | [S2] "Mechanics"; [S1] `sim/battle-actions.ts:1895-1903` | 2026-10-19 |
| T11 | Terastallization | Not mentioned for M-C. The VGC team list does not include a Tera Type | Never offered: champions `canTerastallize` returns `null`; requests omit `teraType` | [S4] §2.4.1, p. 7; [S1] `data/mods/champions/scripts.ts:179-181`, `sim/pokemon.ts:1185-1188` | 2026-10-19 |
| T12 | Stat Points | **Unverified** (task U5): no primary source found for the caps. VGC team lists record "All stats" and "Stat alignment" | The set's `evs` hold Stat Points: at most **32 per stat**, **66 in total** (`EV Limit = Auto` → 66 for `champions*` mods); message names "Stat Points" | [S4] §2.4.1, p. 7; [S1] `sim/team-validator.ts:1155`, `:1313-1318`, `:1357-1363`, `sim/dex-formats.ts:323-330` | 2026-10-19 |
| T13 | IVs | **Unverified** (task U5) | Every IV must be 31 ("this format requires all IVs to be 31") | [S1] `sim/team-validator.ts:1165-1168` | 2026-10-19 |
| T14 | Nature | VGC team list term: "Stat alignment" | Showdown `nature`; under Open Team Sheets the champions mod also shows the nature | [S4] §2.4.1, p. 7; [S1] `sim/battle.ts:3193` | 2026-10-19 |
| T15 | Moves per Pokémon | Not stated in [S2]/[S4] (task U3) | `maxMoveCount` defaults to 4 | [S1] `sim/dex-formats.ts:269` | 2026-10-19 |

## 4. Timer

The engine side is analysed in depth in [timer-deadline-budget.md](timer-deadline-budget.md). Only the identifiers and the official values are repeated here.

| # | Item | Official (Ranked M-C [S2] and VGC [S4]) | Showdown `VGC Timer` | Source | Verified |
|---|---|---|---|---|---|
| M1 | Team Preview | 90 s | `Timer Grace = 90`, `Timer Max First Turn = 90` | [S2] "Timers"; [S4] §4.3.1, p. 12; [S1] `data/rulesets.ts:778-787` | 2026-10-19 |
| M2 | Per turn | 45 s ("Turn Time" / "Move time limit") | `Timer Max Per Turn = 55` | as M1 | 2026-10-19 |
| M3 | Player time (bank) | 7 minutes ("Player Time" / "Your Time") | `Timer Starting = 420` (7 min), `Timer Add Per Turn = 0` | as M1 | 2026-10-19 |
| M4 | Game time | 20 minutes ("Total Time" / "Game time") | None: the rule sets no total game clock | as M1 | 2026-10-19 |
| M5 | Turn timeout | Not described in [S2]/[S4] (task U6) | `Timeout Auto Choose`: `>pN default` when the player's turn time is out, bank > 0 and `player.knownActive`; otherwise forfeit "lost due to inactivity". If every player's turn time is out and auto-choose is off or every bank is 0, the battle is a tie ("All players are inactive.") | [S1] `data/rulesets.ts:785`, `server/room-battle.ts:439-458` (tie: `:441-446`) | 2026-10-19 |
| M6 | Game-time expiry | VGC: gameplay ends "until the in-game time limit has expired"; result rules not given (task U6) | Not applicable (M4) | [S4] §4.1, p. 11 | 2026-10-19 |
| M7 | Who runs the timer | VGC: Championship Series competitions "automatically enforce" the limits | Server only, off unless `Config.forcetimer` or a player turns it on; the in-process sim never runs it | [S4] §4.3.1, p. 11; [S1] `server/room-battle.ts:607`, `config/config-example.js:381`; [timer-deadline-budget.md §1.4, §1.6](timer-deadline-budget.md) | 2026-10-19 |
| M8 | Disconnects | VGC: a reconnect within two turns continues the game; longer disconnects are reviewed by staff | `DC Timer Bank` (`dcTimer` on) | [S4] §4.4.1, p. 12; [S1] `data/rulesets.ts:785`, `sim/dex-formats.ts:290-295` | 2026-10-19 |

## 5. Turn phases and requests

Showdown sends each player one `|request|` JSON per decision point, on that player's own channel only. The four request shapes are typed in [S1] `sim/side.ts:131-164` and built by `Battle.getRequests` (`sim/battle.ts:1422-1469`). Areneon validates every body against `packages/schema/contracts/v1/engine-request.schema.json` before decoding a phase (`packages/showdown-bridge/src/request.ts`).

| # | Phase | Trigger | Request contents (own side only) | Choice | Source | Verified |
|---|---|---|---|---|---|---|
| P1 | Team Preview | Battle start, `Team Preview` rule | `{teamPreview: true, maxChosenTeamSize: 4, side}`; public log first shows `\|clearpoke`, one `\|poke\|` per Pokémon of both sides, then `\|teampreview\|4` | `team 1234`-style order of 4; first two lead | [S1] `data/mods/champions/rulesets.ts:34-54`, `sim/battle.ts:1395-1401`, `:1438-1444`, `sim/side.ts:139-146`, `:1031-1071` | 2026-10-19 |
| P2 | Move selection | Each turn while the side has Pokémon left | `{active: [...], side}`; per active slot: `moves` (`move`, `id`, `pp`, `maxpp`, `target`, `disabled`), optional `trapped`, `maybeTrapped`, `maybeDisabled`, `maybeLocked`, `canMegaEvo` | One action per active slot (move + target, switch, Mega) | [S1] `sim/side.ts:95-119`, `:147-156`, `sim/battle.ts:1446-1456`, `sim/pokemon.ts:1083-1151` | 2026-10-19 |
| P3 | Forced switch | A slot must be refilled (faint, pivot move) | `{forceSwitch: boolean[], side}` for every side with a flagged slot (both sides can get one); a side with no flagged slot gets a wait request | `switch N` for each flagged slot | [S1] `sim/side.ts:131-138`, `sim/battle.ts:1427-1436` | 2026-10-19 |
| P4 | Wait | The side has nothing to decide | `{wait: true, side}` | none | [S1] `sim/side.ts:157-163`, `sim/battle.ts:1459-1466` | 2026-10-19 |
| P5 | End of battle | A side has no Pokémon left, or both | No request; `activeRequest = null`; log ends `\|win\|NAME` or `\|tie`. If both sides run out (gen > 4), the win goes to the side of the last entry processed in the faint queue (`faintData.target.side`, after any `lastFirst` reorder); a call without faint data ends in a tie | none | [S1] `sim/battle.ts:1524-1545`, `:2545-2602`, `:2610-2621` | 2026-10-19 |
| P6 | `side` block (all requests) | — | Per own Pokémon: `ident`, `details`, `condition` (exact HP), `active`, `stats` (all but HP), `moves`, `baseAbility`, `item`, `pokeball`, `ability`, `commanding`, `reviving`; never `teraType` in champions | — | [S1] `sim/pokemon.ts:1153-1190`, `sim/side.ts:357-367` | 2026-10-19 |
| P7 | Official phase list | VGC: choose four and leads, then play until all four of one side faint or time expires | — | — | [S4] §4.1, p. 11 | 2026-10-19 |
| P8 | Official end of game | A competitor wins by making the opponent's final Pokémon faint. If both sides' last Pokémon faint on the last turn, "the game software will automatically decide a winner based on a set of coded rules (such as whose Pokémon fainted first or what moves were used on the last turn, among other conditions)" | — | — | [S4] §4.4, p. 12 | 2026-10-19 |

`rqid` exists only on the server path (`sim/SIM-PROTOCOL.md:752-755` [S1]). Areneon's request identity (`request_seq`, `request_hash`) covers both paths (CONTEXT.md).

## 6. Player-visible and hidden information under CTS

This is the observation contract for training: what a policy may be given for one player, and what stays hidden. Engine split messages (`|split|<side>`) send the secret version to the owner and the shared version to everyone else ([S1] `sim/battle.ts:3081-3113`). Areneon builds observations only from one player's own channel (`packages/showdown-bridge/src/observation.ts`, `player-channel.ts`).

| # | Information | Own side | Opponent at Team Preview | Opponent during battle | Source | Verified |
|---|---|---|---|---|---|---|
| V1 | Species / forme | Full (request `details`) | `\|poke\|` details: species, `L50`, gender. Xerneas, Zacian and Zamazenta formes are masked as `-*` | Revealed by `\|switch\|` / `\|detailschange\|` when sent out or Mega Evolved | [S1] `data/mods/champions/rulesets.ts:34-41`, `sim/pokemon.ts:536-542`, `sim/SIM-PROTOCOL.md:253-290` | 2026-10-19 |
| V2 | Which 4 were brought, lead order | Own choice | Hidden (all 6 shown) | Revealed only as each one switches in | [S1] `sim/battle.ts:1395-1401`, `sim/SIM-PROTOCOL.md:253` | 2026-10-19 |
| V3 | Held item | Full (`item`) | Hidden: the `\|poke\|` item field is always empty at the pin | Revealed only by `\|-item\|`, `\|-enditem\|`, `\|-mega\|` (Mega Stone) or `[from] item:` tags on other messages (e.g. `-heal`, `-damage`). `sim/SIM-PROTOCOL.md:105` documents an `item` marker at preview that the pinned code never sends | [S1] `data/mods/champions/rulesets.ts:40`, `sim/SIM-PROTOCOL.md:99-105`, `:483-538` | 2026-10-19 |
| V4 | Ability | Full (`baseAbility`, `ability`) | Hidden | Revealed only by `\|-ability\|` or `[from] ability:` tags | [S1] `sim/pokemon.ts:1176-1180`, `sim/SIM-PROTOCOL.md:512-520` | 2026-10-19 |
| V5 | Moves | Full (`moves`, PP) | Hidden | Revealed move by move by `\|move\|` | [S1] `sim/pokemon.ts:1166-1175`, `sim/SIM-PROTOCOL.md:239` | 2026-10-19 |
| V6 | Stats, Stat Points, nature, IVs | Own `stats` in the request (no HP; HP max in `condition`) | Hidden | Stats, Stat Points and IVs are never sent; under Open Team Sheets the champions mod shows nature but nulls `evs` and `ivs` (with OTS declined, nature stays hidden) | [S1] `sim/pokemon.ts:1159-1165`, `sim/battle.ts:3182-3198` | 2026-10-19 |
| V7 | HP | Exact `hp/maxhp` | Not sent | Percentage `floor(100·hp/maxhp)` (min 1), with `y`/`r`/`g` colour hints at 20 % and 50 %; exact HP only in the secret half of a split | [S1] `sim/pokemon.ts:2060-2074`, `:544-554` | 2026-10-19 |
| V8 | Team sheet (OTS) | — | Shown only if both players accept (`/acceptopenteamsheets` → `>show-openteamsheets`), only before turn 1; one rejection ends the offer | Same | [S1] `data/rulesets.ts:1980-2001`, `server/chat-commands/core.ts:949-1002`, `sim/battle-stream.ts:209-210`, `sim/battle.ts:3182-3183` | 2026-10-19 |
| V9 | Official VGC team list | VGC uses open team lists: everything on the list except stats is given to the opponent | — | — | [S4] §2.4, p. 6 | 2026-10-19 |
| V10 | Official in-game Ranked preview | **Unverified** (task U7): what Champions shows of the opponent at in-game Team Preview | — | — | none | task U7 |
| V11 | Shininess | Own `details` | The champions preview handler does not strip `, shiny` (the base handler does) | Shown on `\|switch\|` | [S1] `data/mods/champions/rulesets.ts:39`, `data/rulesets.ts:646` | 2026-10-19 |

## 7. Showdown versus the official rules

Fact rows above; the comparison and its consequences are listed here. Gap records are cross-referenced by `gap_id` (`rules/gaps/<gap_id>.json`) and are **not** edited by this note; reconciling them is #773.

| # | Difference | Rows | Gap record |
|---|---|---|---|
| D1 | Per-turn timer 55 s on Showdown vs 45 s official. | M2 | `timer`, `in-game-timer` |
| D2 | No 20-minute game clock on Showdown; the official game-time result rule is unknown. | M4, M6 | `timer`, `in-game-timer` |
| D3 | Showdown timer is server-only and off by default; Championship Series competitions "automatically enforce" the official limits ([S4] §4.3.1). | M7 | `timer` |
| D4 | Showdown requires exactly 6 registered Pokémon; VGC allows 4 to 6 by event format. | T2 | `team-preview` |
| D5 | Team sheets: VGC tournaments use open team lists; Showdown Bo1 offers OTS opt-in, Bo3 forces it. Areneon's CTS target (Bo1, OTS declined) is therefore **neither the VGC tournament rule nor a verified copy of in-game Ranked Battles**. It is a project choice (ADR-003) that defines the hidden-information observation contract. | V8-V10, R6-R7 | `open-team-sheets`, `cts-invariant` |
| D6 | Eligible species: official list is a published whitelist; Showdown derives legality from the champions mod data plus tag bans. Not yet compared entry by entry. | T8 | `roster-and-learnsets` |
| D7 | Stat Points 32/66 and all-31 IVs come from the engine only. | T12-T13 | `stat-points`, `pokepaste` |
| D8 | Simultaneous last faint: Showdown gives the win to the side of the last entry processed in the faint queue; VGC says the game decides "based on a set of coded rules (such as whose Pokémon fainted first …)". | P5, P8 | none (candidate for #773) |
| D9 | Shiny flag visible at Showdown Team Preview in the champions mod (cosmetic). This contradicts the `cts-invariant` description (species, level and gender only). | V11 | `cts-invariant` (flag for #773) |
| D10 | Balance changes rolled out with M-C ([S3]) are not checked against the engine's champions overrides. | AV3 | `move-item-ability-overrides`, `override-parity`, `pp` |

### Inference

These follow from the facts but are not stated by any source.

1. **I1.** Showdown's 55 s turn cap is presumably 45 s plus allowance for network and animation time; no source says so.
2. **I2.** Because all banned tags in `Flat Rules` hit only battle-only formes present in the champions dex at the pin (`Meloetta-Pirouette`, `Terapagos-Terastal`, from probing the built engine), the Showdown pool is effectively defined by the champions formats data, not by the tag bans.
3. **I3.** The D2 difference can change outcomes only in games that would exceed 20 minutes; the D1 difference makes Showdown more lenient per turn. Neither affects headless (A1) self-play, which runs no timer (M7).
4. **I4.** Since V3-V6 are hidden until revealed, any policy input about opponent items, abilities, moves or stats must be built from the public log only; this is the invariant `cts-invariant` records and `test/observation.test.ts` checks.
5. **I5.** Training against Bo1 CTS is stricter on hidden information than VGC tournament play (open lists). A policy trained under CTS could later receive the opponent's sheet as extra input; the reverse is not possible without retraining.

## 8. Unverified rows and verification tasks

| Task | Rows | What to check | Where |
|---|---|---|---|
| U1 | AV5 | Ranked Battles season names and dates within M-C | Official Pokémon Champions news ([S2] site, `champions-news.pokemon-home.com/en/`), in-game Ranked Battles screen |
| U2 | AV6 | Which regulation set each Play! Pokémon VGC event between 2026-09-09 and 2026-12-02 uses | Event pages on `championships.pokemon.com` / Play! Pokémon event locator, event registration announcements |
| U3 | T2, T15 | In-game Ranked team size (6 required?) and moves per Pokémon | Pokémon Champions in-game Battle Team screen; official game manual or help pages |
| U4 | T8 | Full M-C eligible list (the [S6] page renders by script and returned no list to a plain fetch) | Render [S6] in a headless renderer or read it in-game; diff against the champions formats data at the pin (#773) |
| U5 | T12, T13 | Stat Point caps (32 per stat, 66 total) and whether IVs are fixed at 31 | Official Pokémon Champions site or in-game training screen; the summary screen of a maximised Pokémon |
| U6 | M5, M6 | In-game behaviour on turn timeout and on game-time expiry (auto-select? loss? tiebreak order?) | In-game Ranked Battles help text; Play! Pokémon Penalty Guidelines (current PDF, [S5] link) |
| U7 | V10 | What in-game Ranked Team Preview shows of the opponent (species, level, gender, item presence?) | A captured in-game Team Preview screen (sanitized fixture, per the `cts-invariant` gap record) |

## Sources

| Label | Title | URL | Publisher | Date / version | Accessed |
|---|---|---|---|---|---|
| S1 | Pokémon Showdown (pinned submodule) | <https://github.com/smogon/pokemon-showdown/tree/7244fe8e28fa8bc4d0809ae30d183c608b62bb2b> | Smogon / Pokémon Showdown contributors (MIT) | commit `7244fe8e28fa8bc4d0809ae30d183c608b62bb2b`, 2026-10-16 | 2026-10-19 |
| S2 | Regulation Set M-C (Updated on September 9) | <https://champions-news.pokemon-home.com/en/page/816.html> | The Pokémon Company (Pokémon Champions news) | updated 2026-09-09 | 2026-10-19 |
| S3 | Maintenance Completion Notice | <https://champions-news.pokemon-home.com/en/page/817.html> | The Pokémon Company (Pokémon Champions news) | 2026-09-09 | 2026-10-19 |
| S4 | Play! Pokémon Video Game Championships Tournament Handbook (English) | <https://www.pokemon.com/static-assets/content-assets/cms2/pdf/play-pokemon/rules/play-pokemon-vgc-tournament-handbook-en.pdf> | The Pokémon Company International (Play! Pokémon) | last revision 2026-09-01 | 2026-10-19 |
| S5 | Play! Pokémon Rules and Resources Updated for Q3 2026 | <https://www.pokemon.com/us/news/play-pokemon-rules-and-resources-updated-for-q3-2026> | The Pokémon Company International | 2026-09-01 | 2026-10-19 |
| S6 | Eligible Pokémon (Regulation Set M-C) | <https://web-view.app.pokemonchampions.jp/battle/pages/events/rs178713870219xeaaio/en/pokemon.html> | The Pokémon Company (linked from [S2]) | undated | 2026-10-19 (list not retrieved, see U4) |

Notes on sources:

- [S4]: a copy on `mcdn.pokemon.com` still serves the earlier revision (2026-05-21, with Scarlet and Violet sections). Cite the `www.pokemon.com` path. Its revision date comes from [S4] p. 21, "Date of current issue: September 1, 2026"; [S5] announces the same update.
- Secondary sites (Victory Road, MetaVGC, Serebii, Game8, Bulbapedia) were read only to find primary sources. No row relies on them.
