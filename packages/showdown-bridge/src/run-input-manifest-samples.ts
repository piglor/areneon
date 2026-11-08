// The RunInputManifest's share of the sample corpus (ADR-008 section 2, ADR-024 section 3.5).
// Placeholders that pin a shape: sizes of nothing real, hashes that are single repeated digits.

type Cases = Record<string, unknown>;

/* eslint-disable @typescript-eslint/no-explicit-any */
type Rec = Record<string, any>;
const mutated = (value: unknown, change: (copy: Rec) => void): Rec => {
  const copy = structuredClone(value) as Rec;
  change(copy);
  return copy;
};

const file = (path: string, size: number, digit: string): Rec => ({ path, size, sha256: digit.repeat(64) });

export function runInputManifestSamples(): { valid: Cases; invalid: Cases; readerInvalid: Cases } {
  const base: Rec = {
    schema_version: 1,
    data_class: "synthetic",
    run_key: "tiny-run",
    commit: "1".repeat(40),
    run_config_hash: "a".repeat(64),
    dataset_hash: "b".repeat(64),
    config_file: "run-config.json",
    dataset_dir: "dataset",
    files: [file("dataset/manifest.json", 812, "2"), file("dataset/records.dev.jsonl", 40_960, "3"), file("dataset/records.training.jsonl", 262_144, "4"), file("dataset/records.validation.jsonl", 40_960, "5"), file("run-config.json", 2_048, "6")],
  };
  const withFiles = (files: Rec[], change: (copy: Rec) => void = () => undefined): Rec => mutated(base, (copy) => ((copy.files = files), change(copy)));
  const valid: Cases = {
    "run-input-manifest/minimal": withFiles([file("dataset/manifest.json", 0, "2"), file("run-config.json", 1, "6")]),
    "run-input-manifest/tiny-run": base,
    "run-input-manifest/nested-dataset": withFiles([file("inputs/dataset/a/manifest.json", 10, "2"), file("inputs/dataset/records.dev.jsonl", 20, "3"), file("inputs/run-config.json", 30, "6")], (copy) => {
      copy.dataset_dir = "inputs/dataset";
      copy.config_file = "inputs/run-config.json";
    }),
  };
  const invalid: Cases = {
    "run-input-manifest/extra-field": mutated(base, (copy) => (copy.note = "x")),
    "run-input-manifest/extra-file-field": withFiles([file("dataset/manifest.json", 1, "2"), { ...file("run-config.json", 1, "6"), mtime: 1 }]),
    "run-input-manifest/next-version": mutated(base, (copy) => (copy.schema_version = 2)),
    "run-input-manifest/not-synthetic": mutated(base, (copy) => (copy.data_class = "public")),
    "run-input-manifest/missing-commit": mutated(base, (copy) => delete copy.commit),
    "run-input-manifest/short-commit": mutated(base, (copy) => (copy.commit = "1".repeat(39))),
    "run-input-manifest/uppercase-commit": mutated(base, (copy) => (copy.commit = "A".repeat(40))),
    "run-input-manifest/short-config-hash": mutated(base, (copy) => (copy.run_config_hash = "a".repeat(63))),
    "run-input-manifest/short-dataset-hash": mutated(base, (copy) => (copy.dataset_hash = "b".repeat(63))),
    "run-input-manifest/run-name-leading-dash": mutated(base, (copy) => (copy.run_key = "-leading")),
    "run-input-manifest/run-name-empty": mutated(base, (copy) => (copy.run_key = "")),
    "run-input-manifest/one-file": withFiles([file("run-config.json", 1, "6")]),
    "run-input-manifest/fractional-size": withFiles([file("dataset/manifest.json", 1.5, "2"), file("run-config.json", 1, "6")]),
    "run-input-manifest/negative-size": withFiles([file("dataset/manifest.json", -1, "2"), file("run-config.json", 1, "6")]),
    "run-input-manifest/oversize-file": withFiles([file("dataset/manifest.json", 268_435_457, "2"), file("run-config.json", 1, "6")]),
    "run-input-manifest/short-file-hash": withFiles([{ path: "dataset/manifest.json", size: 1, sha256: "2".repeat(63) }, file("run-config.json", 1, "6")]),
    "run-input-manifest/empty-path": withFiles([file("", 1, "2"), file("run-config.json", 1, "6")]),
    "run-input-manifest/null-files": mutated(base, (copy) => (copy.files = null)),
  };
  const readerInvalid: Cases = {
    "run-input-manifest/files-unsorted": withFiles([file("run-config.json", 1, "6"), file("dataset/manifest.json", 1, "2")]),
    "run-input-manifest/duplicate-path": withFiles([file("dataset/manifest.json", 1, "2"), file("dataset/manifest.json", 1, "3"), file("run-config.json", 1, "6")]),
    "run-input-manifest/path-parent-segment": withFiles([file("dataset/../manifest.json", 1, "2"), file("run-config.json", 1, "6")]),
    "run-input-manifest/path-absolute": withFiles([file("/dataset/manifest.json", 1, "2"), file("run-config.json", 1, "6")]),
    "run-input-manifest/path-empty-segment": withFiles([file("dataset//manifest.json", 1, "2"), file("run-config.json", 1, "6")]),
    "run-input-manifest/path-dot-segment": withFiles([file("dataset/./manifest.json", 1, "2"), file("run-config.json", 1, "6")]),
    "run-input-manifest/path-backslash": withFiles([file("dataset\\manifest.json", 1, "2"), file("run-config.json", 1, "6")]),
    "run-input-manifest/paths-differ-by-case": withFiles([file("dataset/Manifest.json", 1, "2"), file("dataset/manifest.json", 1, "3"), file("run-config.json", 1, "6")]),
    "run-input-manifest/path-space": withFiles([file("dataset/my manifest.json", 1, "2"), file("run-config.json", 1, "6")]),
    "run-input-manifest/path-trailing-slash": withFiles([file("dataset/manifest.json/", 1, "2"), file("run-config.json", 1, "6")]),
    "run-input-manifest/config-file-unsafe": mutated(base, (copy) => (copy.config_file = "../run-config.json")),
    "run-input-manifest/dataset-dir-unsafe": mutated(base, (copy) => (copy.dataset_dir = "..")),
    "run-input-manifest/config-file-unlisted": mutated(base, (copy) => (copy.config_file = "other-config.json")),
    "run-input-manifest/config-inside-dataset": mutated(base, (copy) => (copy.config_file = "dataset/manifest.json")),
    "run-input-manifest/dataset-dir-empty": mutated(base, (copy) => (copy.dataset_dir = "elsewhere")),
    "run-input-manifest/file-is-directory": withFiles([file("dataset/manifest.json", 1, "2"), file("dataset/manifest.json/part", 1, "3"), file("run-config.json", 1, "6")]),
    "run-input-manifest/dataset-dir-is-file": withFiles([file("dataset", 1, "2"), file("dataset/records.dev.jsonl", 1, "3"), file("run-config.json", 1, "6")]),
  };
  return { valid, invalid, readerInvalid };
}
