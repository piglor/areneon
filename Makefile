.PHONY: check python ts
check: python ts

python:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	uv run pytest

ts:
	pnpm lint
	pnpm typecheck
	pnpm test
