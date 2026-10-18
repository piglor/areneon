SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c
.PHONY: setup showdown check python ts contracts contracts-update guards team-guard licences licences-selftest

# One-time setup for a clean clone.
setup: showdown
	uv sync
	pnpm install
	uv run pre-commit install

# Build the pinned reference engine (ADR-003). Install scripts stay off:
# the sim needs no native add-ons, and vendored packages run no code here.
showdown:
	git submodule update --init vendor/pokemon-showdown
	cd vendor/pokemon-showdown && npm ci --ignore-scripts --no-audit --no-fund && node build

# Every CI gate that works offline (gitleaks and osv-scanner run in CI).
# CI calls the same targets, so the two cannot drift.
check: python ts guards

python:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	uv run pytest

ts: showdown
	pnpm lint
	pnpm typecheck
	pnpm test
	node packages/showdown-bridge/scripts/check-rules.ts
	$(MAKE) contracts

# Contract schemas (ADR-008): every schema in packages/schema/contracts/ must
# fit its profile, and the checked-in generated TS types and pydantic models
# must match a fresh run of the pinned generators. contracts-update rewrites them.
contracts:
	node packages/schema/scripts/contracts.ts
	uv run python scripts/contract_models.py

contracts-update:
	node packages/schema/scripts/contracts.ts --write
	uv run python scripts/contract_models.py --write

guards: team-guard licences

# Team guards: PokePaste text outside fixtures/, and JSON that is or may be
# private team or game data (ADR-008 data_class).
team-guard:
	git ls-files -z | xargs -0 uv run --no-project python scripts/check_no_pokepaste.py
	git ls-files -z | xargs -0 uv run --no-project python scripts/check_data_class.py

licences:
	pnpm -r licenses list --prod --json | uv run --no-project python scripts/check_licenses.py
	uv run python scripts/python_licences.py | uv run --no-project python scripts/check_licenses.py

# Proves `make licences` fails on an injected AGPL dependency, on a scratch
# copy. CI-only: it needs network, so `make check` does not run it.
licences-selftest:
	scripts/licences-selftest.sh
