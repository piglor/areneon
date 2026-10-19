# Contributing to Areneon

Thanks for your interest. Areneon is pre-alpha (Wave 1).

## Where work is tracked

- Engineering work, ADRs and experiments: Redmine, https://redmine.piglor.com/projects/areneon
- GitHub Issues: community bug reports and questions only.

## Ground rules

- **Never commit a real competitive team.** Private teams go in the gitignored `private-data/`. Public fixtures under `fixtures/` must be synthetic or sanitized and are CC0. A pre-commit hook and CI reject PokePaste-shaped files elsewhere; the only exemptions are exact paths listed in `.pokepaste-guard-allow`. A second guard (`scripts/check_data_class.py`, ADR-008) rejects committed JSON or JSONL labelled `"data_class": "private"`, and team-shaped JSON outside `fixtures/` (a set, or a Pokémon from a `|request|`) unless it is labelled `"data_class": "synthetic"` at its root. It also rejects JSON it cannot parse, so do not commit JSON with comments.
- **Do not copy AGPL code** (the Pokémon Showdown client, Champions Calc). Write independently.
- The pinned Showdown engine is the only authority on legality. Areneon code never declares a choice legal by itself.

## Setup

Requirements: Node 24 LTS, pnpm 10 through corepack (`corepack enable`: it installs exactly the `packageManager` version in `package.json` and refuses it unless its sha512 matches), [uv](https://docs.astral.sh/uv/) (installs Python 3.12), GNU Make ≥ 3.82 (macOS ships 3.81: `brew install make` and run `gmake`).

```bash
git clone --recurse-submodules https://github.com/piglor/areneon.git
cd areneon
make setup check   # build pinned Showdown, install everything, run every offline CI gate
```

`gitleaks` and `osv-scanner` run in CI. `make showdown` rebuilds the pinned engine in `vendor/pokemon-showdown` after a submodule update.

### Bumping the Showdown pin

The engine is the legality authority, so a pin bump is its own PR:

1. Move the submodule (`git -C vendor/pokemon-showdown checkout <sha>`) and set `PINNED_SHOWDOWN_COMMIT` in `packages/showdown-bridge/src/engine.ts` to the same SHA.
2. `make showdown`, then `node packages/showdown-bridge/scripts/capture-golden.ts`.
3. Review every changed fixture under `fixtures/showdown/` and update the gap ledger records in `rules/gaps/`, then regenerate [docs/showdown-champions-gaps.md](docs/showdown-champions-gaps.md) with `node packages/showdown-bridge/scripts/gaps-doc.ts` (a test fails while the doc and the records differ).
4. Move the rules channel to the new build (ADR-007): set `hash` in `rules/channels/stable.json` to the output of `node packages/showdown-bridge/scripts/rules-hash.ts` and raise `seq` by 1. The rules snapshot includes the engine commit, so `node packages/showdown-bridge/scripts/check-rules.ts` (part of `make ts`) fails until you do.

### Changing the rules

Cross-language contracts (ADR-008) are JSON Schemas in `packages/schema/contracts/v<N>/`. Edit the schema, never the generated code: run `make contracts-update` and commit the schema with the regenerated TS types (`packages/schema/generated/`) and pydantic models (`packages/schema/python/areneon_contracts/models/`). `make contracts` (in `make ts`; CI runs the TS half in the typescript job and the Python half in the python job) refuses a schema outside its profile and any drift. After a contract change, also re-capture the shared sample corpus with `node packages/showdown-bridge/scripts/capture-samples.ts`.

Areneon's own rules (supported formats, forbidden engine rules, regulation sources, the gap ledger) are JSON records in `rules/`, validated by the schemas in `packages/schema/rules/` (ADR-007). Never put them in code. Edit a record in a normal PR; `make ts` checks it against the pinned engine. To make the change live, also point `rules/channels/stable.json` at the new snapshot hash (from `scripts/rules-hash.ts`) and raise `seq` by 1, in the same PR or a later one. `rules/channels/` holds exactly one file per channel in the `channel` enum of `packages/schema/rules/v1/channel.schema.json`; adding a channel means adding it to that enum, and any other file fails the check.

## Commits and pull requests

- [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `chore:` …).
- Reference the Redmine issue: `Refs #<id>`.
- **Sign off every commit (DCO):** `git commit -s`. By signing off you certify the [Developer Certificate of Origin 1.1](https://developercertificate.org/). CI rejects commits without `Signed-off-by`.
- Keep coverage at ≥90% lines for `packages/*` and `services/*`. Code that cannot be covered should be removed, not exempted.

## Licence

By contributing you agree your contributions are licensed under Apache-2.0 (code) or CC0-1.0 (fixtures).
