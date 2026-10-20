# ADR-001: Open-source licence

**Status:** Accepted (2026-10-17, project owner)  
**Wave:** 1  
**ADR issue:** #693 · **Epic:** #687 (Wave 1 — Foundation)  
**Implementation:** #669  
**Review:** plan-stage review (conditional approve), recorded in the #693 journal

## Decision
- **Source code: Apache-2.0**, with `LICENSE`, `NOTICE` and SPDX headers.
- **Synthetic and sanitized fixtures: CC0-1.0.** Real replays and other people's teams are never committed.
- **Contributions: DCO sign-off (`Signed-off-by`), no CLA.**
- **Trademark/non-affiliation notice:** Pokémon names are trademarks of their owners; Areneon is not affiliated with Nintendo, Game Freak, The Pokémon Company or Smogon.

## Context
Upstream licences:
- `smogon/pokemon-showdown` (server/simulator) and `@smogon/calc` / `@pkmn/*`: MIT, compatible with Apache-2.0. MIT notices are kept in `NOTICE` and the vendored source.
- `pokemon-showdown-client` and Champions Calc: AGPL-3.0. They are not copied.

A code licence grants no rights to Pokémon names or artwork, replays, scraped data or trained weights. Those are governed per artifact.

## Alternatives rejected
- **MIT:** no patent grant.
- **AGPL-3.0:** restricts adoption and conflicts with the permissive goal.
- **MPL-2.0 and dual MIT/Apache:** no benefit identified over Apache-2.0.

## Consequences
- The CI licence allowlist fails on AGPL/GPL dependencies in shipped packages.
- Any future reuse of AGPL code requires a superseding ADR.

## Clarification (2026-10-20)

This is a dated clarification of the production licence allowlist mentioned under Consequences. It is not a new decision: ADR-001's status and decisions above are unchanged.

- **BSL-1.0 and Zlib** are OSI-approved permissive licences. They are added to the production allowlist by owner decision under ADR-010 Q9 (accepted 2026-10-20).
- **Packages that carry only a BSD trove classifier** (no SPDX `License-Expression`) are admitted only through a named per-package exception list. Each entry names the package, the licence verified from its source or metadata, and the date of verification. A blanket mapping from the classifier to a licence is never used. The list is reviewed like any other gate change.
- **CUDA, `nvidia-*` and `triton` packages are not allowlisted.** They stay outside the checked and shipped dependency closure (ADR-010 §5.2).

The allowlist and checker change itself is tracked separately (#834).
