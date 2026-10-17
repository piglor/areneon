SHELL := /bin/bash
.PHONY: check python ts guards
# Runs every CI gate that works offline (gitleaks and osv-scanner run in CI).
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

guards:
	git ls-files -z | xargs -0 uv run --no-project python scripts/check_no_pokepaste.py
	set -o pipefail; pnpm -r licenses list --prod --json | uv run --no-project python scripts/check_licenses.py
