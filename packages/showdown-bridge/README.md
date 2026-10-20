# @areneon/showdown-bridge

The only package that talks to the pinned Showdown engine. See `CLAUDE.md` for the module map.

## Owner-team feature inventory

`scripts/team-inventory.ts` lists what your own teams use (species, items, abilities, moves, Mega Evolution), runs each through the pinned engine's `importPaste` and `validateTeam`, and flags every feature that a gap record in `rules/gaps/` names as quarantined. The flags are read from the records, so a new gap record changes them without a code change. The engine stays the only legality authority.

Put one PokePaste file per team, in the engine's canonical form, as `private-data/teams/<name>.txt`. `private-data/` is gitignored; never commit a real team. Then run it yourself, so your teams never enter a chat or a log:

```bash
! node packages/showdown-bridge/scripts/team-inventory.ts
```

It prints one JSON line with the report path and counts (`teams`, `legal`, `illegal`, `import_failed`, `quarantined_features`) and nothing else. The report is `private-data/reports/team-inventory.json` (mode 0600). Open it yourself: per team it holds the validator's verdict, each Pokémon's features and Mega forme, and the quarantined features with the gap ids that name them. Re-running overwrites it.

| Option | Meaning |
|---|---|
| `--teams-dir DIR` | Read the `*.txt` files directly in `DIR` instead of `<repo>/private-data/teams`. The environment variable `ARENEON_TEAMS_DIR` does the same. Use it from a worktree to point at the main checkout's `private-data/teams`. |
| `--format ID` | The format id. Optional while the rules snapshot holds one format, as it does now. |

The report goes beside the teams' `private-data` directory (`<that private-data>/reports/`). The script refuses a teams directory whose real path has no `private-data` segment, a team file that is not a regular file inside it, and any write outside a `private-data/` directory (symlinks are resolved first). It also refuses an empty directory. An error names a file at most, never a team's content. Exit codes: 0 report written (check the counts for `illegal` and `import_failed`), 1 refused or failed, 2 bad option.
