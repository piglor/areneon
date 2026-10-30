# @areneon/team-dataset

The synthetic team generator (ADR-013), its dataset and snapshot formats, and the tooling around them. This README covers the DuckDB working database; the generator and the Parquet shards are described in the repository's `CLAUDE.md` and ADR-013.

## The DuckDB working database

`scripts/teams.ts duckdb` builds a local, derived DuckDB file from a **public** snapshot, so teams can be explored with SQL. Private teams are out of scope here (ADR-015 has its own workflow).

DuckDB is a tool you install, not a dependency: the package imports no DuckDB library, and the file is never committed, never in a Release, R2 or an image (`*.duckdb` is gitignored and `scripts/check_binary_files.py` refuses it). The tool is `--duckdb PATH`, else `DUCKDB_BIN`, else `duckdb` on the `PATH`; `scripts/duckdb-pin.json` names the version CI uses.

```sh
# a snapshot directory (snapshot.json + pub/v1/shards/) as `scripts/duckdb-writer.ts write` leaves it
node packages/team-dataset/scripts/teams.ts duckdb --snapshot SNAPSHOT_DIR --out teams.duckdb \
  [--dataset DATASET_DIR] [--labels labels.json] [--shards SHARDS_DIR] [--require-pinned]
```

The build verifies before it loads: every shard goes through the shard reader (file size and SHA-256, layout, record rebuild, logical hash), every record must carry the label set given (default: the committed one of the snapshot's format), and every paste is expanded with the pinned engine's `Teams.import`. `--dataset` is the dataset directory whose `manifest.json` names the merged strata; a snapshot of a dataset with merged strata needs it to verify. An existing output file is never replaced, and a failed build leaves no file.

The tables are created by the versioned script `sql/working-db.v1.sql` (a change is a new script and a new `duckdb_schema_version`, never an edit):

| table | one row per | columns |
| --- | --- | --- |
| `snapshot` | the snapshot | `duckdb_schema_version`, `snapshot_id`, `snapshot_hash`, `logical_snapshot_hash`, `parent_snapshot_hash`, `vocabulary_hash`, `format_id`, `regulation`, `showdown_commit`, `writer_name`, `writer_version`, `parquet_schema_version`, `teams` |
| `shards` | shard file | `split`, `object_key`, `file_sha256`, `bytes`, `logical_hash`, `rows` |
| `teams` | team record | `record_hash`, `team_hash`, `canonical_team_hash`, `dataset_id`, `split`, `origin_index`, `draw_counter`, `stratum`, `shard_logical_hash`, `paste` |
| `labels` | team and label | `record_hash`, `label_id`, `value` |
| `sets` | Pokemon of a team | `record_hash`, `slot`, `nickname`, `species`, `item`, `ability`, `nature`, `gender`, `level`, `sp_hp` .. `sp_spe` (Stat Points) |
| `moves` | move of a set | `record_hash`, `slot`, `move_slot`, `move` |

Built so far from a local snapshot (the committed fixtures converted with `scripts/duckdb-writer.ts`). Reading a snapshot from R2 is not exercised yet: no snapshot is published, so the snapshot has to be on disk.

### Recipes

Each SQL recipe below is run by `test/duckdb/working-database.test.ts` against a database built from the committed fixtures, with the real DuckDB tool (the test is skipped where none is installed and cannot be skipped in CI, which sets `ARENEON_REQUIRE_DUCKDB=1`). `SNAPSHOT_DIR` and `OUT_DIR` stand for your own directories.

Open the database in DuckDB's local web UI (the first run downloads the `ui` extension, so it needs network access; not run in tests):

```sh
duckdb -ui teams.duckdb
```

Teams that bring a species:

<!-- recipe: filter-species -->
```sql
SELECT t.split, count(DISTINCT t.record_hash) AS teams
FROM sets s JOIN teams t USING (record_hash)
WHERE s.species = 'Gallade'
GROUP BY t.split ORDER BY t.split;
```

Teams that hold an item:

<!-- recipe: filter-item -->
```sql
SELECT t.split, count(DISTINCT t.record_hash) AS teams
FROM sets s JOIN teams t USING (record_hash)
WHERE s.item = 'Wise Glasses'
GROUP BY t.split ORDER BY t.split;
```

Teams with a move on any Pokemon:

<!-- recipe: filter-move -->
```sql
SELECT t.split, count(DISTINCT t.record_hash) AS teams
FROM moves m JOIN teams t USING (record_hash)
WHERE m.move = 'Trick Room'
GROUP BY t.split ORDER BY t.split;
```

Teams by label value (the label ids and values are those of the label set, `config/teams/<format>/labels.json`):

<!-- recipe: filter-label -->
```sql
SELECT t.split, count(*) AS teams
FROM labels l JOIN teams t USING (record_hash)
WHERE l.label_id = 'redirection' AND l.value = 'yes'
GROUP BY t.split ORDER BY t.split;
```

Read the snapshot's Parquet shards directly, with no database (the shard files are named by their SHA-256, so use the glob):

<!-- recipe: read-parquet -->
```sql
SELECT split, count(*) AS teams
FROM read_parquet('SNAPSHOT_DIR/pub/v1/shards/*')
GROUP BY split ORDER BY split;
```

Export a selection as Parquet (pastes of the teams with Fake Out and a Mega Stone):

<!-- recipe: export-selection -->
```sql
COPY (
  SELECT t.record_hash, t.split, t.paste
  FROM teams t
  JOIN labels a ON a.record_hash = t.record_hash AND a.label_id = 'fake_out' AND a.value = 'yes'
  JOIN labels b ON b.record_hash = t.record_hash AND b.label_id = 'mega_stone' AND b.value = 'yes'
  ORDER BY t.record_hash
) TO 'OUT_DIR/selection.parquet' (FORMAT parquet);
```

Read R2 Parquet with a `TYPE r2` secret (needs network and your own R2 credentials, so it is documented, not run in tests; the bucket and key names are yours):

<!-- recipe-not-run: r2 -->
```sql
CREATE SECRET r2_read (TYPE r2, KEY_ID '...', SECRET '...', ACCOUNT_ID '...');
SELECT split, count(*) FROM read_parquet('r2://BUCKET/pub/v1/shards/*') GROUP BY split;
```

Keep credentials out of shell history and out of the repository; the working database holds no credential.
