# Targeted fidelity corpus (ADR-026 Evidence 1, slice T)

License: CC0 1.0 (see `fixtures/LICENSE`). Every team is synthetic, drawn by the
seeded team generator, never a real team. The channels are each player's own
stream (`PlayerChannelLog`); no omniscient log is stored.

| File | What it is |
| --- | --- |
| `targets.json` | The reviewed input of the capture script: per target the moves and abilities a team must hold to be drawn for it, the public channel line patterns that count as one occurrence, and the minimum number of turns. `unreachable` lists what no legal team can hold (Download: no species of the format lists it). |
| `b*.json` | One seeded A1 battle each: seed, both teams, the actor seeds and kinds (`random-legal` or `pending-first` per seat), where each team came from (`origin`), both players' channels. |
| `manifest.json` | The battles (seed, turns, winner, actors) and the team sources. |
| `summary.json` | The coverage summary: per target the turns in which it occurs, in how many battles and in which turns per battle, plus the corpus's total battle turns (the denominators of slice F). |

Teams come from two sources: the dev split of `../../teams/<format>/datasets/fixtures-v1/`
and a larger generated dev split (`search-fidelity-src`, not committed: the committed
split holds no Last Respects or Mirror Coat team). It is rebuilt with
`generate` at the generator commit named in `manifest.json`, the config
`config/teams/<format>/generator.json`, the committed `fixtures-v1/pilot.json`, `--dev 3000
--validation 50 --training 50` and these public keys (they derive from labels and protect nothing):

- generation key `775cf690c68e8efce8deb29d548fc1da69c0612fb378e876a75798411ad80388` = SHA-256 of `areneon/search-fidelity/v1/generation`
- assignment key `38198457df1773d2543f0be1cbadba1ea0e5a1fb3949e3a973023ecf4ec6a8cf` = SHA-256 of `areneon/search-fidelity/v1/assignment`

Only the teams of the kept battles are committed. Re-capture with
`node packages/showdown-bridge/scripts/capture-search-fidelity-corpus.ts --dataset DIR`;
CI replays the committed battles and never re-runs the search.
