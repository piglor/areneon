# Areneon

Open-source competitive VGC Doubles AI sparring partner for Pokémon Showdown (Champions Regulation M-C, Closed Team Sheet).

**Status: Wave 1 (foundation), pre-alpha. No trained model exists yet.**

- Engineering work, ADRs and experiments: [Redmine](https://redmine.piglor.com/projects/areneon) ([ADR index](https://redmine.piglor.com/projects/areneon/wiki/ADR))
- GitHub Issues are for community contributions only; internal work lives in Redmine.
- Setup and rules: [CONTRIBUTING.md](CONTRIBUTING.md). Domain glossary: [CONTEXT.md](CONTEXT.md).
- Private competitive teams must never be committed. Use sanitized fixtures only.

## Layout

```
packages/schema/             JSON Schema contracts (source of truth)
packages/showdown-bridge/    TS: headless runner, request decoder, legal choices
packages/showdown-extension/ TS: Chromium MV3 read-only advisor
vendor/pokemon-showdown/     pinned upstream simulator (git submodule)
fixtures/                    sanitized/synthetic fixtures only (CC0)
spikes/                      experiment code, promoted or deleted
private-data/                gitignored: your private teams
```

## Licence

Code: [Apache-2.0](LICENSE). Fixtures: [CC0-1.0](fixtures/LICENSE). See [NOTICE](NOTICE).

Pokémon and related names are trademarks of Nintendo, Creatures Inc., GAME FREAK inc. and The Pokémon Company. Areneon is not affiliated with them or with Smogon / Pokémon Showdown.
