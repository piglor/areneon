# Showdown Champions gap list

<!-- Generated from rules/gaps/ by `node packages/showdown-bridge/scripts/gaps-doc.ts`. Edit the records, not this file. -->

Pin: smogon/pokemon-showdown `7244fe8e28fa8bc4d0809ae30d183c608b62bb2b`. Format: `gen9championsvgc2026regmc` (`[Gen 9 Champions] VGC 2026 Reg M-C`, mod `champions`). Tracking: Redmine #670, ADR-003, ADR-007.

Areneon makes **no claim of parity with the Pokémon Champions game**. This list records what the pinned engine does for Champions so that nothing is assumed silently. A row is *unverified* until an in-game golden fixture confirms it. Results that depend on an unverified row are labelled with it.

Engine paths are relative to `vendor/pokemon-showdown`. Review the records in `rules/gaps/` in every pin-bump PR.

## Implemented by the pinned engine, unverified in-game

| Area | What Showdown does at the pin | Where |
|---|---|---|
| CTS invariant | With Open Team Sheets declined, Team Preview shows the opponent's species, level and gender only; the sheet is sent only after `>show-openteamsheets`. Tested in-process by `cts.test.ts` (#676); the server-path decline is #709. | `data/rulesets.ts` `openteamsheets`, `sim/battle.ts` `showOpenTeamSheets`, fixture `fixtures/teams/gen9championsvgc2026regmc/legal-opponent.txt` |
| Freeze | Thaws after at most 3 turns, or 25% per turn | `data/mods/champions/conditions.ts` `frz` |
| Joint legal choices | Enumerated from the player's own request and judged by the pinned engine (ADR-006, `choices.ts`, `choice-judge.ts`). The golden choice corpus covers preview, every target type in the format's move pool, Mega, forced switches, conflicting switches, visible and hidden trapping, Imprison (`maybeDisabled`/`maybeLocked`), a fainted active with no replacement and a locked move; a brute-force oracle finds no engine-accepted action outside the enumerated set. Unverified: no in-game fixture yet. | `sim/side.ts` `choose`, `chooseMove`, `chooseSwitch`, `emitChoiceError`, fixture `fixtures/showdown/gen9championsvgc2026regmc/choices/maybe-trapped.json`, fixture `fixtures/showdown/gen9championsvgc2026regmc/choices/maybe-disabled.json`, fixture `fixtures/showdown/gen9championsvgc2026regmc/choices/double-switch.json` |
| Level | `Adjust Level = 50` (Flat Rules) | `data/mods/champions/rulesets.ts` |
| Mega Evolution | Mega forms persist after fainting | `data/mods/champions/scripts.ts` `formeChange` |
| Mega Evolution requests | The engine offers `canMegaEvo` on a Mega Stone holder and accepts Mega on one slot per choice, never both (golden choice corpus `mega`, `mega-both`). `canMegaEvoX`/`canMegaEvoY` exist only in other mods (`gen7letsgo`, `monkeyspaw`), never in Reg M-C: Charizard X and Y are separate stones. Revisit if a supported format's mod defines them. | `sim/pokemon.ts` `getMoveRequestData`, `data/mods/gen7letsgo/scripts.ts` `canMegaEvoX`, `canMegaEvoY`, fixture `fixtures/showdown/gen9championsvgc2026regmc/choices/mega.json`, fixture `fixtures/showdown/gen9championsvgc2026regmc/choices/mega-both.json` |
| Moves, items, abilities | 257 move, 253 item and 15 ability overrides of the Gen 9 data (count: `grep -c $'^\t[a-z0-9]*: {' <file>`) | `data/mods/champions/moves.ts`, `data/mods/champions/items.ts`, `data/mods/champions/abilities.ts` |
| Open Team Sheets | Opt-in in Bo1 (`Open Team Sheets`), forced in Bo3. Areneon supports Bo1 only and declines. | `data/rulesets.ts` |
| Paralysis | Full paralysis chance 1/8 | `data/mods/champions/conditions.ts` `par` |
| PokePaste | `Teams.import` reads Stat Points from the `EVs:` line and silently skips lines it does not know (an `SPs:` line leaves every Stat Point at 0). A repeated line overwrites the earlier one. Areneon's `importPaste` rejects a paste unless each Pokémon's lines all reappear in that Pokémon's re-export, so it accepts only the engine's canonical form (reordered stats, `Level: 100`, `IVs: 31`, a lowercase or misspelt species or a teambuilder header are rejected; revisit if hand-written or third-party pastes become an input path). Team legality messages are logged in `fixtures/teams/<format>/verdicts.json`. | `sim/teams.ts` `import`, `export` |
| PP | Max PP is `(pp / 5 + 1) * 4`; base PP above 20 is capped to 20 | `data/mods/champions/scripts.ts` `calculatePP`, `init` |
| Roster and learnsets | Champions-specific legality data | `data/mods/champions/formats-data.ts`, `data/mods/champions/learnsets.ts` |
| Sleep | Lasts 2 or 3 turns (1/3 chance to wake on turn 2) | `data/mods/champions/conditions.ts` `slp` |
| Stat Points | The set's `evs` field holds Stat Points. Stats at level 50 are `base + SP + 20` (HP `base + SP + 75`), then the nature applies. The validator rejects a set with 0 Stat Points under Serious (it suggests another neutral nature if that was intended). | `data/mods/champions/scripts.ts` `statModify` |
| Team Preview | Bring 6, choose 4, two leads (`maxChosenTeamSize: 4`, doubles) | fixture `fixtures/showdown/gen9championsvgc2026regmc/preview.request.json` |
| Terastallization | Not offered in Reg M-C: no request in the golden fixtures or the golden choice corpus carries `canTerastallize`. The enumerator still emits `terastallize` (one per side) when a request offers it; that path has unit tests only. Revisit when a supported format allows Terastallization. | fixture `fixtures/showdown/gen9championsvgc2026regmc/turn.request.json`, fixture `fixtures/showdown/gen9championsvgc2026regmc/choices/targets-clefable-garchomp.json` |
| Timer | Showdown `VGC Timer`: 90 s preview, 420 s bank, 55 s per turn, auto-choose on timeout. This is a server rule; the in-process sim does not run it. | `data/rulesets.ts` `vgctimer` |
| Trick Room | No speed underflow | `data/mods/champions/scripts.ts` `getActionSpeed` |

## Unknown or not yet exercised

- Damage formula and rounding. These are covered by the calculator work, not by this pin.
- Champions in-game timer values compared with Showdown's VGC Timer.
- Whether every Champions move, item and ability override matches the game. No in-game fixtures exist yet.
