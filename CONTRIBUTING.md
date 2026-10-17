# Contributing to Areneon

Thanks for your interest. Areneon is pre-alpha (Wave 1).

## Where work is tracked

- Engineering work, ADRs and experiments: Redmine, https://redmine.piglor.com/projects/areneon
- GitHub Issues: community bug reports and questions only.

## Ground rules

- **Never commit a real competitive team.** Private teams go in the gitignored `private-data/`. Public fixtures under `fixtures/` must be synthetic or sanitized and are CC0. A pre-commit hook rejects PokePaste-shaped files elsewhere.
- **Do not copy AGPL code** (the Pokémon Showdown client, Champions Calc). Write independently.
- The pinned Showdown engine is the only authority on legality. Areneon code never declares a choice legal by itself.

## Setup

Requirements: Node 24 LTS, pnpm 10, [uv](https://docs.astral.sh/uv/) (installs Python 3.12).

```bash
git clone --recurse-submodules https://github.com/piglor/areneon.git
cd areneon
uv sync
pnpm install
uv run pre-commit install
make check   # lint, types, tests for Python and TypeScript
```

## Commits and pull requests

- [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `chore:` …).
- Reference the Redmine issue: `Refs #<id>`.
- **Sign off every commit (DCO):** `git commit -s`. By signing off you certify the [Developer Certificate of Origin 1.1](https://developercertificate.org/). CI rejects commits without `Signed-off-by`.
- Keep coverage at ≥90% lines for `packages/*` and `services/*`. Code that cannot be covered should be removed, not exempted.

## Licence

By contributing you agree your contributions are licensed under Apache-2.0 (code) or CC0-1.0 (fixtures).
