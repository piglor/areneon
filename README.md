# Areneon

Open-source competitive VGC Doubles AI sparring partner for Pokémon Showdown (Champions Regulation M-C, Closed Team Sheet: Bo1 with Open Team Sheets declined).

**Status: Wave 1 (foundation), pre-alpha. No trained model exists yet.**

- Engineering work, ADRs and experiments: [Redmine](https://redmine.piglor.com/projects/areneon) ([ADR index](https://redmine.piglor.com/projects/areneon/wiki/ADR))
- GitHub Issues are for community contributions only; internal work lives in Redmine.
- Setup and rules: [CONTRIBUTING.md](CONTRIBUTING.md). Domain glossary: [CONTEXT.md](CONTEXT.md).
- Private competitive teams must never be committed. Use sanitized fixtures only.

## Layout

Polyglot monorepo (ADR-002): TypeScript, Python and Rust. Entries marked *planned* do not exist yet.

```
packages/showdown-bridge/    TS: pinned engine loader, format registry, headless runner, request decoder
packages/schema/             planned (#671): JSON Schema contracts (source of truth)
packages/showdown-extension/ planned (#672): Chromium MV3 read-only advisor
crates/                      planned: Rust Cargo workspace (calculators, rules catalog, orchestration)
services/, ml/               planned: Python services and PyTorch training
vendor/pokemon-showdown/     pinned upstream simulator (MIT, git submodule)
fixtures/                    sanitized/synthetic fixtures only (CC0)
docs/                        contributor-facing contracts and research notes (docs/research/)
spikes/                      experiment code (lighter gates), promoted or deleted
private-data/                gitignored: your private teams
```

## Licence

Code: [Apache-2.0](LICENSE). Fixtures: [CC0-1.0](fixtures/LICENSE). See [NOTICE](NOTICE).

Pokémon and related names are trademarks of Nintendo, Creatures Inc., GAME FREAK inc. and The Pokémon Company. Areneon is not affiliated with them or with Smogon / Pokémon Showdown.
