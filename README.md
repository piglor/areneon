# Areneon

Open-source research project: an AI training pipeline for competitive VGC Doubles on the pinned Pokémon Showdown engine (Champions Regulation M-C, Closed Team Sheet: Bo1 with Open Team Sheets declined). Bots learn by automatic self-play and are judged by independent, held-out evaluation.

This repository covers offline training, evaluation and local bot sparring; it ships no tool that assists a player in live battles on public servers and no bot for public servers. It is not monetized.

**Status: Wave 1 (foundation), pre-alpha. No trained model exists yet.**

- Engineering work, ADRs and experiments: [Redmine](https://redmine.piglor.com/projects/areneon) ([ADR index](https://redmine.piglor.com/projects/areneon/wiki/ADR))
- GitHub Issues are for community contributions only; internal work lives in Redmine.
- Setup and rules: [CONTRIBUTING.md](CONTRIBUTING.md). Domain glossary: [CONTEXT.md](CONTEXT.md).
- Owner runbook for the dev rules store (ADR-007): [docs/runbooks/rules-db.md](docs/runbooks/rules-db.md).
- Private competitive teams must never be committed. Use sanitized fixtures only.

## Layout

Polyglot monorepo (ADR-002): TypeScript, Python and Rust. Entries marked *planned* do not exist yet.

```
packages/showdown-bridge/    TS: pinned engine loader, rules loader and hashing, headless runner, request decoder
packages/schema/             JSON Schema contracts (source of truth): rules records and cross-language data contracts
packages/rules-store/        TS: the rules store on PostgreSQL 18 (ADR-007): idempotent import as `importer`, and its tests
packages/rules-importer/     TS: rules importer CLI and image: attested rules Releases into the rules store (ADR-007)
packages/bot-client/         TS: bot client for a local Showdown server (A2, ADR-005): transport port, local-only URL guard
packages/local-server/       TS: pinned local Showdown server for sparring (ADR-005): verified runtime copy outside vendor/, locked-down config, launcher on 127.0.0.1
rules/                       Areneon's own rules as reviewed JSON records (ADR-007), never code constants
crates/                      planned: Rust Cargo workspace (calculators, rules catalog, orchestration)
services/, ml/               planned: Python services, PyTorch training, self-play league and evaluation
vendor/pokemon-showdown/     pinned upstream simulator (MIT, git submodule)
fixtures/                    sanitized/synthetic fixtures only (CC0)
  showdown/<format>/         golden request fixtures from the pinned engine
  teams/<format>/            PokePaste team fixtures + validator verdict log
docs/                        contributor-facing contracts and research notes (docs/research/)
private-data/                gitignored: your private teams
```

## Licence

Code: [Apache-2.0](LICENSE). Fixtures: [CC0-1.0](fixtures/LICENSE). See [NOTICE](NOTICE).

Pokémon and related names are trademarks of Nintendo, Creatures Inc., GAME FREAK inc. and The Pokémon Company. Areneon is not affiliated with them or with Smogon / Pokémon Showdown.
