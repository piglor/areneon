"""The eval-dataset option (ADR-020 E-17.1): the progress evaluation and the scripted head-to-head.

Both measure on the validation split of a team dataset. With this option the dataset is another
verified one than the run's own: a directory and its ``datasetHash``. The Node side reads it with
``readDataset`` exactly as it reads the run's own and refuses a different hash; this module holds
what the Python commands need: the two flags (go together), the arguments passed on to Node, and
the size of the validation split read from the dataset's manifest.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

_HASH = re.compile(r"[0-9a-f]{64}")


class EvalDatasetError(ValueError):
    """A refused evaluation dataset option."""


@dataclass(frozen=True)
class EvalDataset:
    """Another verified dataset's directory and hash."""

    path: Path
    dataset_hash: str

    def node_args(self) -> tuple[str, ...]:
        return ("--eval-dataset-dir", str(self.path), "--eval-dataset-hash", self.dataset_hash)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--eval-dataset-dir",
        type=Path,
        help="measure on the validation split of another verified dataset instead of the "
        "run's own (with --eval-dataset-hash)",
    )
    parser.add_argument("--eval-dataset-hash", help="its datasetHash (with --eval-dataset-dir)")


def from_values(path: Path | None, dataset_hash: str | None) -> EvalDataset | None:
    """The option as given: both or neither; a hash is 64 lower-case hex digits."""
    if (path is None) != (dataset_hash is None):
        raise EvalDatasetError("--eval-dataset-dir and --eval-dataset-hash go together")
    if path is None or dataset_hash is None:
        return None
    if _HASH.fullmatch(dataset_hash) is None:
        raise EvalDatasetError("--eval-dataset-hash must be 64 lower-case hex digits")
    return EvalDataset(path.resolve(), dataset_hash)


def validation_teams(directory: Path) -> int:
    """The size of a dataset's validation split, read from its manifest."""
    path = directory / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        return int(next(s["teams"] for s in manifest["splits"] if s["split"] == "validation"))
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        raise EvalDatasetError(f"cannot read the validation split's size from {path}") from exc
