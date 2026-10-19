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
