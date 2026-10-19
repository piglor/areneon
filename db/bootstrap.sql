-- One-time setup of the rules store's roles and schema (ADR-007), run once
-- per database by a superuser, before the first `dbmate migrate`:
--
--   RULES_MIGRATOR_PASSWORD=… RULES_IMPORTER_PASSWORD=… RULES_READER_PASSWORD=… \
--     psql "$SUPERUSER_URL" -v ON_ERROR_STOP=1 -f db/bootstrap.sql
--
-- Each password comes from the environment (on Coolify: a secret per role),
-- never from the repository. A missing or blank (empty or whitespace-only)
-- one stops the script (ON_ERROR_STOP) before any password is sent to the server.
-- `make db-up` runs it against the local container.

\set ON_ERROR_STOP on

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

-- A blank secret (empty or whitespace only) is as bad as a missing one. psql
-- checks it on the client: each backtick pipes the environment variable,
-- named only, from `printf` (a shell builtin) into `tr`, which drops
-- whitespace, and `test` (a builtin) checks something is left. No password
-- goes to the server, or into a process's arguments, except inside its own
-- CREATE ROLE below. psql would expand `:space` in the backticks if a psql
-- variable of that name existed; nothing defines one, and if it did, only the
-- whitespace-only refusal would weaken.
\set migrator_present `test -n "$(printf %s "$RULES_MIGRATOR_PASSWORD" | tr -d '[:space:]')" && echo true || echo false`
\set importer_present `test -n "$(printf %s "$RULES_IMPORTER_PASSWORD" | tr -d '[:space:]')" && echo true || echo false`
\set reader_present `test -n "$(printf %s "$RULES_READER_PASSWORD" | tr -d '[:space:]')" && echo true || echo false`
\if :migrator_present
\else
  DO $$ BEGIN RAISE EXCEPTION 'RULES_MIGRATOR_PASSWORD is empty'; END $$;
\endif
\if :importer_present
\else
  DO $$ BEGIN RAISE EXCEPTION 'RULES_IMPORTER_PASSWORD is empty'; END $$;
\endif
\if :reader_present
\else
  DO $$ BEGIN RAISE EXCEPTION 'RULES_READER_PASSWORD is empty'; END $$;
\endif

BEGIN;

-- Owns the tables; nobody logs in as it.
CREATE ROLE rules_owner NOLOGIN;
-- The only role that runs dbmate. It holds the owner's rights only after
-- SET ROLE rules_owner (INHERIT FALSE), as the migrations do.
CREATE ROLE rules_migrator LOGIN PASSWORD :'migrator_password';
GRANT rules_owner TO rules_migrator WITH INHERIT FALSE, SET TRUE;
-- Inserts published artifacts; never updates or deletes.
CREATE ROLE importer LOGIN PASSWORD :'importer_password';
-- Live runtimes read the active snapshot.
CREATE ROLE areneon_reader LOGIN PASSWORD :'reader_password';

CREATE SCHEMA rules AUTHORIZATION rules_owner;
GRANT USAGE ON SCHEMA rules TO importer, areneon_reader;
-- dbmate keeps its own version table in this schema, as rules_migrator.
GRANT USAGE, CREATE ON SCHEMA rules TO rules_migrator;

-- Only these roles connect, and nobody else gets TEMP on this database.
REVOKE ALL ON DATABASE :"DBNAME" FROM PUBLIC;
GRANT CONNECT ON DATABASE :"DBNAME" TO rules_migrator, importer, areneon_reader;

COMMIT;
