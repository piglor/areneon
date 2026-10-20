export { type Column, columnsFor, labelColumn, PARQUET_SCHEMA_VERSION, recordOf, type Row, rowOf, type RowPlan, SCHEMA_VERSION_KEY, ShardError } from "./columns.ts";
export { type ConvertOptions, type ConvertResult, convertDataset, datasetRecordsFile, fileLines, verifySnapshot, type VerifyOptions } from "./convert.ts";
export { DEFAULT_ROW_GROUP_SIZE, type DuckdbPin, DuckdbWriter, locateDuckdb, PIN_FILE, platformKey, readPin, type Runner, type WriteOptions, type WriterIdentity } from "./duckdb.ts";
export { byStratumThenHash, cellRanges, layoutSplit, MAX_SHARDS, SHARD_ROWS, shardAssigner, shardCount, sortShard } from "./layout.ts";
export { readShard, type ShardEntry, type ShardExpectation, type ShardRead } from "./reader.ts";
export { buildSnapshotManifest, checkFixtureAgreement, datasetPlan, eachSnapshotShard, readSnapshotShards, type ShardSource, shardsInDirectory, type SnapshotInput, type SnapshotRead } from "./snapshot.ts";
export { runShardsCli, type ShardsIo } from "./command.ts";
