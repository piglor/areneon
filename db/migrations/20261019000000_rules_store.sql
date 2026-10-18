-- The rules store (ADR-007 "Database contract"): published rules snapshots and
-- the channel history, append-only. Run by `rules_migrator` with
-- `search_path=rules`; the DDL runs as the NOLOGIN owner.

-- migrate:up
SET LOCAL ROLE rules_owner;

CREATE TABLE rules.rules_snapshot (
  -- sha256 of the exact published bytes: checked against body, never payload.
  hash text PRIMARY KEY CONSTRAINT snapshot_hash_is_sha256_of_body CHECK (hash = encode(sha256(body), 'hex')),
  body bytea NOT NULL,
  -- A query copy parsed from body: exactly body's JSON. A body that is not
  -- UTF-8 JSON fails here. convert_from is STABLE, not IMMUTABLE, only
  -- because it can depend on the server encoding; from fixed UTF-8 bytes its
  -- result never changes, which is what a CHECK needs.
  payload jsonb NOT NULL CONSTRAINT snapshot_payload_is_body CHECK (payload = convert_from(body, 'UTF8')::jsonb),
  -- Extracted from body (through payload); NULL-safe, so a missing key fails.
  schema_version integer NOT NULL
    CONSTRAINT snapshot_schema_version_from_payload
    CHECK (schema_version IS NOT DISTINCT FROM (payload ->> 'schema_version')::integer),
  showdown_commit text NOT NULL
    CONSTRAINT snapshot_showdown_commit_from_payload
    CHECK (showdown_commit IS NOT DISTINCT FROM payload ->> 'showdown_commit'),
  source_commit text NOT NULL CONSTRAINT snapshot_source_commit_is_sha1 CHECK (source_commit ~ '^[0-9a-f]{40}$'),
  published_at timestamptz NOT NULL
);

CREATE TABLE rules.rules_release (
  id bigserial PRIMARY KEY,
  -- The closed channel enum of packages/schema/rules/v1/channel.schema.json
  -- (a test keeps the two equal).
  channel text NOT NULL CONSTRAINT release_channel_in_enum CHECK (channel IN ('stable')),
  seq integer NOT NULL CONSTRAINT release_seq_positive CHECK (seq >= 1),
  hash text NOT NULL CONSTRAINT release_snapshot_exists REFERENCES rules.rules_snapshot (hash),
  source_commit text NOT NULL CONSTRAINT release_source_commit_is_sha1 CHECK (source_commit ~ '^[0-9a-f]{40}$'),
  activated_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT release_seq_unique_per_channel UNIQUE (channel, seq),
  CONSTRAINT release_source_commit_unique_per_channel UNIQUE (channel, source_commit)
);

-- Append-only: UPDATE, DELETE and TRUNCATE are granted to nobody, and refused
-- by trigger as well. Tamper-evident, not tamper-proof: a superuser or the
-- owner can still disable triggers (accepted in ADR-007).
CREATE FUNCTION rules.refuse_change() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  RAISE EXCEPTION 'the rules store is append-only: % on % refused', TG_OP, TG_TABLE_NAME;
END
$$;

CREATE TRIGGER rules_snapshot_append_only BEFORE UPDATE OR DELETE ON rules.rules_snapshot
  FOR EACH ROW EXECUTE FUNCTION rules.refuse_change();
CREATE TRIGGER rules_snapshot_no_truncate BEFORE TRUNCATE ON rules.rules_snapshot
  FOR EACH STATEMENT EXECUTE FUNCTION rules.refuse_change();
CREATE TRIGGER rules_release_append_only BEFORE UPDATE OR DELETE ON rules.rules_release
  FOR EACH ROW EXECUTE FUNCTION rules.refuse_change();
CREATE TRIGGER rules_release_no_truncate BEFORE TRUNCATE ON rules.rules_release
  FOR EACH STATEMENT EXECUTE FUNCTION rules.refuse_change();

REVOKE ALL ON rules.rules_snapshot, rules.rules_release FROM PUBLIC;
GRANT SELECT, INSERT ON rules.rules_snapshot, rules.rules_release TO importer;
GRANT USAGE ON SEQUENCE rules.rules_release_id_seq TO importer;
GRANT SELECT ON rules.rules_snapshot, rules.rules_release TO areneon_reader;
-- dbmate records the version as rules_migrator, in this same transaction.
RESET ROLE;

-- migrate:down
SET LOCAL ROLE rules_owner;
DROP TABLE rules.rules_release;
DROP TABLE rules.rules_snapshot;
DROP FUNCTION rules.refuse_change();
RESET ROLE;
