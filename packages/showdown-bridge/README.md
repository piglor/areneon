# @areneon/showdown-bridge

The only package that talks to the pinned Showdown engine. See `CLAUDE.md` for the module map.

## Owner-team feature inventory

`scripts/team-inventory.ts` lists what your own teams use (species, items, abilities, moves, Mega Evolution), runs each through the pinned engine's `importPaste` and `validateTeam`, and flags every feature that a gap record in `rules/gaps/` names as quarantined. The flags are read from the records, so a new gap record changes them without a code change. The engine stays the only legality authority.

Put one PokePaste file per team, in the engine's canonical form, as `private-data/teams/<name>.txt`. `private-data/` is gitignored; never commit a real team. Then run it yourself, so your teams never enter a chat or a log:

```bash
node packages/showdown-bridge/scripts/team-inventory.ts
```

Run it in your **own** terminal, not inside an agent session: it refuses unless stdin, stdout and stderr are all terminals, and when `CLAUDECODE` or `AI_AGENT` is set (a `! node ...` escape in an agent session therefore exits 1). This is accident prevention, not a boundary (ADR-014 X6, R12): an agent that supplies its own pty and clears its environment can run it and read its stdout line (counts only), and a same-user agent can read your terminal's scrollback files if you keep any. It checks that `/dev/tty` opens before it reads any team.

It prints one JSON line to stdout with counts (`teams`, `legal`, `illegal`, `import_failed`, `quarantined_features`) and `gap_matches` (gap id to the number of teams with a feature that gap names) and nothing else. **It writes no file.** The detail goes to your terminal device (`/dev/tty`) and nowhere else: per team the validator's verdict, each Pokémon's features and Mega forme, and the quarantined features with the gap ids that name them. It lives in your scrollback only, so clear it when you are done; scrollback and multiplexer logs may keep it. The command refuses to run unless stdin, stdout and stderr are terminals, so it cannot be piped or captured. It no longer creates `private-data/reports/team-inventory.json`; an earlier run may have left one behind, so delete it (deleting is not erasing: file-system snapshots and backups may still hold it). Teams are read through the `PrivateTeamSource` port (`src/private-team-source.ts`); the directory reader is its only adapter in this package.

| Option | Meaning |
|---|---|
| `--teams-dir DIR` | Read the `*.txt` files directly in `DIR` instead of `<repo>/private-data/teams`. The environment variable `ARENEON_TEAMS_DIR` does the same. Use it from a worktree to point at the main checkout's `private-data/teams`. |
| `--format ID` | The format id. Optional while the rules snapshot holds one format, as it does now. |

The script refuses a teams directory whose real path has no `private-data` segment (symlinks are resolved first), a team file that is not a regular file inside it, and an empty directory. An error names a file at most, never a team's content. Exit codes: 0 inventory shown (check the counts for `illegal` and `import_failed`), 1 refused (including a non-terminal session) or failed, 2 bad option.

### What the flags mean, and what they do not

- **The per-team flags are advisory name matches.** A feature is flagged when a gap record's area, description or evidence symbols contain its name as whole words (case-insensitive). Formes also match by base species (Rotom-Wash matches a record naming Rotom). The gap records have no structured per-feature list, so this is not a complete list of unverified features: the overrides and roster records name no features at all, a status record never flags the moves that inflict it, and level and Tera type are not inventoried. A team is never clean because nothing matched.
- **Read `format_wide_gaps` as well.** The detail lists every gap record of the rules snapshot (id, status, area), flagged or not, and carries `flags_are_advisory: true`.
- **Known false positives**, from prose matching: a move or item named by a word in a record's text, for example `Trick` (from "Trick Room"), `Strength` (from "Strength Sap"), or a species named in a sentence about other mechanics (Charizard, from the Mega X/Y record). Treat a flag as "look at this record", not as "this feature is broken".
- **`private-data` is a naming convention, not a trust boundary.** The script checks that a path has a `private-data` directory in its real path; it does not check who owns it or who can read it. Keep the directory under your own account, and do not point `--teams-dir` at a shared one. Nothing is written to disk, so there is no report file to protect.
