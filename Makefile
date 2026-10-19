SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c
.PHONY: setup showdown check python ts contracts contracts-update guards team-guard licences licences-selftest db-up db-down db-test db-bootstrap-test importer-image importer-e2e

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
	uv run --locked python scripts/contract_models.py

contracts-update:
	node packages/schema/scripts/contracts.ts --write
	uv run --locked python scripts/contract_models.py --write

guards: team-guard licences

# Team guards: PokePaste text outside fixtures/, and JSON that is or may be
# private team or game data (ADR-008 data_class).
team-guard:
	git ls-files -z | xargs -0 uv run --no-project python scripts/check_no_pokepaste.py
	git ls-files -z | xargs -0 uv run --no-project python scripts/check_data_class.py

licences:
	pnpm -r licenses list --prod --json | uv run --no-project python scripts/check_licenses.py
	uv run --locked python scripts/python_licences.py | uv run --no-project python scripts/check_licenses.py

# Proves `make licences` fails on an injected AGPL dependency, on a scratch
# copy. CI-only: it needs network, so `make check` does not run it.
licences-selftest:
	scripts/licences-selftest.sh

# The rules store (ADR-007) on a local Postgres 18 container bound to
# 127.0.0.1, never the Coolify database. db-up starts it fresh, bootstraps the
# roles and runs the migrations with the pinned dbmate; db-test needs it up.
# RULES_DB_PORT and RULES_DB_COMPOSE_PROJECT move it, e.g. to run the e2e
# beside the default one; db-test and the e2e follow through RULES_DB_HOST.
RULES_DB_PORT ?= 55432
RULES_DB_COMPOSE_PROJECT ?= areneon-rules-db
RULES_DB_HOST := 127.0.0.1:$(RULES_DB_PORT)
export RULES_DB_PORT RULES_DB_COMPOSE_PROJECT RULES_DB_HOST
DB_COMPOSE := docker compose -p $(RULES_DB_COMPOSE_PROJECT) -f db/compose.yaml
DBMATE := docker run --rm --network host -v "$(CURDIR)/db:/db:ro" \
	ghcr.io/amacneil/dbmate:2.36.0@sha256:520c740c6e0ad73fde2cd1ea7e2b779aaf789d22aca8858f87a478e7094535fb \
	--no-dump-schema --migrations-dir /db/migrations
# Throwaway passwords for the local container only (on Coolify: secrets).
LOCAL_ROLE_ENV := -e RULES_MIGRATOR_PASSWORD=local-migrator -e RULES_IMPORTER_PASSWORD=local-importer \
	-e RULES_READER_PASSWORD=local-reader

db-up:
	$(DB_COMPOSE) down -v --remove-orphans
	$(DB_COMPOSE) up -d --wait
	$(DB_COMPOSE) exec -T $(LOCAL_ROLE_ENV) postgres psql -U postgres -d areneon -v ON_ERROR_STOP=1 -f - < db/bootstrap.sql
	$(DBMATE) --url "postgres://rules_migrator:local-migrator@$(RULES_DB_HOST)/areneon?sslmode=disable&search_path=rules" migrate

db-down:
	$(DB_COMPOSE) down -v --remove-orphans

db-test:
	pnpm --filter @areneon/rules-store test

# db/bootstrap.sql sends no role password to the server except in its own
# CREATE ROLE (#799). Uses its own throwaway container on a scratch Docker
# network, with no published port, so it does not need or touch db-up's.
db-bootstrap-test:
	scripts/bootstrap-secrets-test.sh

# The rules importer image (#768): `local` is the production image, `e2e` adds
# a stand-in gh replaying recorded GitHub answers. importer-e2e imports the
# recorded releases into a fresh db-up database twice (commit, then no-op);
# it leaves rows behind, so run db-up again before db-test.
IMPORTER_IMAGE := areneon-rules-importer

importer-image:
	docker build -f packages/rules-importer/Dockerfile -t $(IMPORTER_IMAGE):local .
	docker build -f packages/rules-importer/Dockerfile --target e2e -t $(IMPORTER_IMAGE):e2e .

importer-e2e: importer-image db-up
	packages/rules-importer/scripts/e2e.sh $(IMPORTER_IMAGE)
