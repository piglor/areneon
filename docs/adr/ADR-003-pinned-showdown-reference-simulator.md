# ADR-003: Pinned Showdown reference simulator and format

**Status:** Accepted (2026-10-18, project owner; decision revision 2, amended per the plan-stage review recorded in the #695 journal)  
**Date:** 2026-10-17  
**Wave:** 1  
**ADR issue:** #695 · **Implementation Epic:** #689 (Wave 1 — Foundation)  
**Implementation:** #670 (Resolved), #675, #676

## Context
Areneon trains, evaluates and spars on Pokémon Showdown Champions VGC Doubles under Closed Team Sheet. Mechanics, legality and the per-player request must come from one reproducible engine. The project will not write its own simulator.

Reviewer-verified on smogon/pokemon-showdown master @ 7244fe8:
- The format `[Gen 9 Champions] VGC 2026 Reg M-C` exists: mod `champions`, doubles, ruleset `Flat Rules, VGC Timer, Open Team Sheets`. The ID `gen9championsvgc2026regmc` is inferred.
- **Open Team Sheets is opt-in in Bo1 and forced in Bo3.**
- `@pkmn/sim` 0.10.11 has no `champions` mod.

## Options
| Option | Assessment |
|---|---|
| **A. Upstream `smogon/pokemon-showdown` pinned to an exact commit SHA** (MIT), built from source | Authoritative mechanics and validator; has the M-C format |
| B. `@pkmn/sim` | **Rejected as fallback:** no Champions mod |
| B'. npm `pokemon-showdown` 0.11.11 | Predates M-C (unverified); use only if a release contains the pinned format |
| C. Custom rules engine | Out of scope |

## Decision
**A.** Every fixture, episode and benchmark carries a manifest `{showdown_commit, format_id, regulation, ruleset_hash}`. **Regulation is a manifest parameter**: M-C is short-lived and upstream already moved M-B into its own mod. Plan the next-regulation migration as a routine pin bump.

## CTS invariant (P0)
- Areneon uses only the **Bo1** M-C format. Bo3 (forced OTS) is rejected.
- Every Areneon actor sends `/rejectopenteamsheets` on the server path. On the sim path OTS is not shared.
- A fixture test proves that an OTS-declined preview exposes species/details only: no items, moves, abilities or stat points.

## PokePaste and legality
Use Showdown's own `Teams.import` / `Teams.export` and the team validator for the pinned format. Areneon does not write its own legality rules.

## Acceptance evidence
| Gate | Evidence |
|---|---|
| The format ID exists at the pinned SHA | #670, `piglor/areneon#4` (`7244fe8`, `gen9championsvgc2026regmc`, Bo1) |
| A sanitized M-C team is accepted or rejected by the validator, with reasons logged | Open: implementation condition of #675 |
| **Determinism:** same SHA + PRNG seed + input log gives an identical sim log; server timer and chat lines excluded | #670, `piglor/areneon#4` |
| Preview (choose 4, two leads), turn and forced-switch requests captured as sanitized fixtures | #670, `piglor/areneon#4` golden fixtures |
| Gap list: unsupported or unverified Champions mechanics are quarantined | #670, `docs/showdown-champions-gaps.md` |

## Consequences
- A pin upgrade is an explicit PR that re-runs the golden fixtures.
- No claim of native Champions parity.
