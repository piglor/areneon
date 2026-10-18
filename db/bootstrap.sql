-- One-time setup of the rules store's roles and schema (ADR-007), run once
-- per database by a superuser, before the first `dbmate migrate`:
--
--   RULES_MIGRATOR_PASSWORD=… RULES_IMPORTER_PASSWORD=… RULES_READER_PASSWORD=… \
--     psql "$SUPERUSER_URL" -v ON_ERROR_STOP=1 -f db/bootstrap.sql
--
-- Each password comes from the environment (on Coolify: a secret per role),
-- never from the repository. A missing one stops the script (ON_ERROR_STOP).
-- `make db-up` runs it against the local container.

\getenv migrator_password RULES_MIGRATOR_PASSWORD
\getenv importer_password RULES_IMPORTER_PASSWORD
\getenv reader_password RULES_READER_PASSWORD
\if :{?migrator_password}
\else
  DO $$ BEGIN RAISE EXCEPTION 'RULES_MIGRATOR_PASSWORD is not set'; END $$;
\endif
\if :{?importer_password}
\else
  DO $$ BEGIN RAISE EXCEPTION 'RULES_IMPORTER_PASSWORD is not set'; END $$;
\endif
\if :{?reader_password}
\else
  DO $$ BEGIN RAISE EXCEPTION 'RULES_READER_PASSWORD is not set'; END $$;
\endif

BEGIN;

-- Owns the tables; nobody logs in as it.
CREATE ROLE rules_owner NOLOGIN;
-- The only role that runs dbmate: DDL after SET ROLE rules_owner.
CREATE ROLE rules_migrator LOGIN PASSWORD :'migrator_password' IN ROLE rules_owner;
-- Inserts published artifacts; never updates or deletes.
CREATE ROLE importer LOGIN PASSWORD :'importer_password';
-- Live runtimes read the active snapshot.
CREATE ROLE areneon_reader LOGIN PASSWORD :'reader_password';

CREATE SCHEMA rules AUTHORIZATION rules_owner;
GRANT USAGE ON SCHEMA rules TO importer, areneon_reader;

COMMIT;
