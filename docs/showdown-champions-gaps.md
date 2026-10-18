# Showdown Champions gap list

Pin: smogon/pokemon-showdown `7244fe8e28fa8bc4d0809ae30d183c608b62bb2b`. Format: `gen9championsvgc2026regmc` (`[Gen 9 Champions] VGC 2026 Reg M-C`, mod `champions`). Tracking: Redmine #670, ADR-003.

Areneon makes **no claim of parity with the Pokémon Champions game**. This list records what the pinned engine does for Champions so that nothing is assumed silently. Every row stays *unverified* until an in-game golden fixture confirms it. Results that depend on an unverified row are labelled with it.

Paths are relative to `vendor/pokemon-showdown`. Update this file in every pin-bump PR.

## Implemented by the pinned engine, unverified in-game

| Area | What Showdown does at the pin | Where |
|---|---|---|
| Stat Points | The set's `evs` field holds Stat Points. Stats at level 50 are `base + SP + 20` (HP `base + SP + 75`), then the nature applies. The validator rejects a set with 0 Stat Points under Serious (it suggests another neutral nature if that was intended). | `data/mods/champions/scripts.ts` `statModify` |
| Level | `Adjust Level = 50` (Flat Rules) | `data/mods/champions/rulesets.ts` |
| PP | Max PP is `(pp / 5 + 1) * 4`; base PP above 20 is capped to 20 | `data/mods/champions/scripts.ts` `calculatePP`, `init` |
| Paralysis | Full paralysis chance 1/8 | `data/mods/champions/conditions.ts` `par` |
| Sleep | Lasts 2 or 3 turns (1/3 chance to wake on turn 2) | `data/mods/champions/conditions.ts` `slp` |
| Freeze | Thaws after at most 3 turns, or 25% per turn | `data/mods/champions/conditions.ts` `frz` |
| Trick Room | No speed underflow | `data/mods/champions/scripts.ts` `getActionSpeed` |
| Mega Evolution | Mega forms persist after fainting | `data/mods/champions/scripts.ts` `formeChange` |
| Moves, items, abilities | 257 move, 253 item and 15 ability overrides of the Gen 9 data (count: `grep -c $'^\t[a-z0-9]*: {' <file>`) | `data/mods/champions/moves.ts`, `data/mods/champions/items.ts`, `data/mods/champions/abilities.ts` |
| Roster and learnsets | Champions-specific legality data | `data/mods/champions/formats-data.ts`, `data/mods/champions/learnsets.ts` |
| Team Preview | Bring 6, choose 4, two leads (`maxChosenTeamSize: 4`, doubles) | golden `preview.request.json` |
| Terastallization | Not offered in the golden turn request (no `canTerastallize`) | golden `turn.request.json` |
| Timer | Showdown `VGC Timer`: 90 s preview, 420 s bank, 55 s per turn, auto-choose on timeout. This is a server rule; the in-process sim does not run it. | `data/rulesets.ts` `vgctimer` |
| Open Team Sheets | Opt-in in Bo1 (`Open Team Sheets`), forced in Bo3. Areneon supports Bo1 only and declines. | `data/rulesets.ts` |
| PokePaste | `Teams.import` reads Stat Points from the `EVs:` line and silently skips lines it does not know (an `SPs:` line leaves every Stat Point at 0). Areneon's `importPaste` rejects any line that does not survive a re-export. Team legality messages are logged in `fixtures/teams/<format>/verdicts.json`. | `sim/teams.ts` `import`, `export` |

## Unknown or not yet exercised

- Whether every Champions move, item and ability override matches the game. No in-game fixtures exist yet.
- Mega Evolution request shape (`canMegaEvo`) and its legality in Reg M-C. The golden team holds no Mega Stone.
- Champions in-game timer values compared with Showdown's VGC Timer.
- Damage formula and rounding. These are covered by the calculator work, not by this pin.
- The CTS invariant (an OTS-declined preview shows species and details only) is owned by #676.
- Joint legal-choice enumeration is owned by #682.
