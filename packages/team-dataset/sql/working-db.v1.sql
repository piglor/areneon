-- The Areneon team working database, duckdb_schema_version 1 (ADR-013 section 7).
--
-- A derived, local file built from a verified public snapshot: never committed,
-- never published. A change of any table or column is a new file
-- (working-db.v2.sql) with a new duckdb_schema_version, never an edit here.
--
-- The tool (src/duckdb/working-database.ts) writes the six JSON Lines files read
-- below into the script's working directory; every row has been checked by the
-- shard reader first, and every paste expanded with the pinned engine's
-- Teams.import. Stat columns hold Stat Points (the champions mod).

CREATE TABLE snapshot (
  duckdb_schema_version INTEGER NOT NULL,
  snapshot_id VARCHAR NOT NULL,
  snapshot_hash VARCHAR NOT NULL,
  logical_snapshot_hash VARCHAR NOT NULL,
  parent_snapshot_hash VARCHAR,
  vocabulary_hash VARCHAR NOT NULL,
  format_id VARCHAR NOT NULL,
  regulation VARCHAR NOT NULL,
  showdown_commit VARCHAR NOT NULL,
  writer_name VARCHAR NOT NULL,
  writer_version VARCHAR NOT NULL,
  parquet_schema_version INTEGER NOT NULL,
  teams BIGINT NOT NULL
);

CREATE TABLE shards (
  split VARCHAR NOT NULL,
  object_key VARCHAR NOT NULL PRIMARY KEY,
  file_sha256 VARCHAR NOT NULL,
  bytes BIGINT NOT NULL,
  logical_hash VARCHAR NOT NULL,
  rows BIGINT NOT NULL
);

CREATE TABLE teams (
  record_hash VARCHAR NOT NULL PRIMARY KEY,
  team_hash VARCHAR NOT NULL,
  canonical_team_hash VARCHAR NOT NULL,
  dataset_id VARCHAR NOT NULL,
  split VARCHAR NOT NULL,
  origin_index BIGINT NOT NULL,
  draw_counter BIGINT NOT NULL,
  stratum BIGINT NOT NULL,
  shard_logical_hash VARCHAR NOT NULL,
  paste VARCHAR NOT NULL
);

CREATE TABLE labels (
  record_hash VARCHAR NOT NULL,
  label_id VARCHAR NOT NULL,
  value VARCHAR NOT NULL,
  PRIMARY KEY (record_hash, label_id)
);

CREATE TABLE sets (
  record_hash VARCHAR NOT NULL,
  slot INTEGER NOT NULL,
  nickname VARCHAR NOT NULL,
  species VARCHAR NOT NULL,
  item VARCHAR NOT NULL,
  ability VARCHAR NOT NULL,
  nature VARCHAR NOT NULL,
  gender VARCHAR NOT NULL,
  level INTEGER NOT NULL,
  sp_hp INTEGER NOT NULL,
  sp_atk INTEGER NOT NULL,
  sp_def INTEGER NOT NULL,
  sp_spa INTEGER NOT NULL,
  sp_spd INTEGER NOT NULL,
  sp_spe INTEGER NOT NULL,
  PRIMARY KEY (record_hash, slot)
);

CREATE TABLE moves (
  record_hash VARCHAR NOT NULL,
  slot INTEGER NOT NULL,
  move_slot INTEGER NOT NULL,
  move VARCHAR NOT NULL,
  PRIMARY KEY (record_hash, slot, move_slot)
);

INSERT INTO snapshot BY NAME
SELECT 1 AS duckdb_schema_version, s.*
FROM read_json('snapshot.jsonl', format = 'newline_delimited', columns = {
  snapshot_id: 'VARCHAR', snapshot_hash: 'VARCHAR', logical_snapshot_hash: 'VARCHAR', parent_snapshot_hash: 'VARCHAR',
  vocabulary_hash: 'VARCHAR', format_id: 'VARCHAR', regulation: 'VARCHAR', showdown_commit: 'VARCHAR',
  writer_name: 'VARCHAR', writer_version: 'VARCHAR', parquet_schema_version: 'INTEGER', teams: 'BIGINT'
}) AS s;

INSERT INTO shards BY NAME
SELECT * FROM read_json('shards.jsonl', format = 'newline_delimited', columns = {
  split: 'VARCHAR', object_key: 'VARCHAR', file_sha256: 'VARCHAR', bytes: 'BIGINT', logical_hash: 'VARCHAR', rows: 'BIGINT'
});

INSERT INTO teams BY NAME
SELECT * FROM read_json('teams.jsonl', format = 'newline_delimited', columns = {
  record_hash: 'VARCHAR', team_hash: 'VARCHAR', canonical_team_hash: 'VARCHAR', dataset_id: 'VARCHAR', split: 'VARCHAR',
  origin_index: 'BIGINT', draw_counter: 'BIGINT', stratum: 'BIGINT', shard_logical_hash: 'VARCHAR', paste: 'VARCHAR'
});

INSERT INTO labels BY NAME
SELECT * FROM read_json('labels.jsonl', format = 'newline_delimited', columns = {
  record_hash: 'VARCHAR', label_id: 'VARCHAR', value: 'VARCHAR'
});

INSERT INTO sets BY NAME
SELECT * FROM read_json('sets.jsonl', format = 'newline_delimited', columns = {
  record_hash: 'VARCHAR', slot: 'INTEGER', nickname: 'VARCHAR', species: 'VARCHAR', item: 'VARCHAR', ability: 'VARCHAR',
  nature: 'VARCHAR', gender: 'VARCHAR', level: 'INTEGER', sp_hp: 'INTEGER', sp_atk: 'INTEGER', sp_def: 'INTEGER',
  sp_spa: 'INTEGER', sp_spd: 'INTEGER', sp_spe: 'INTEGER'
});

INSERT INTO moves BY NAME
SELECT * FROM read_json('moves.jsonl', format = 'newline_delimited', columns = {
  record_hash: 'VARCHAR', slot: 'INTEGER', move_slot: 'INTEGER', move: 'VARCHAR'
});
