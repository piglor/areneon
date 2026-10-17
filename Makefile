SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c
.PHONY: setup check python ts guards team-guard licences

# One-time setup for a clean clone.
setup:
	uv sync
	pnpm install
	uv run pre-commit install

# Every CI gate that works offline (gitleaks and osv-scanner run in CI).
# CI calls the same targets, so the two cannot drift.
check: python ts guards

python:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	uv run pytest

ts:
	pnpm lint
	pnpm typecheck
	pnpm test

guards: team-guard licences

team-guard:
	git ls-files -z | xargs -0 uv run --no-project python scripts/check_no_pokepaste.py

licences:
	pnpm -r licenses list --prod --json | uv run --no-project python scripts/check_licenses.py
