"""A pair that reads the opponent sheet is not evaluated yet (ADR-027 slice 5 wires the seats).

The evaluation seats do not build the block, so the pair's worker would refuse every query: the
run is refused up front, with no block invented.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from areneon_ml.learner.config import read_run_config
from areneon_ml.progress.run import ProgressError, _source

from tests.ml.progress.support import dataset_config

SET = {"feature_set": "opponent-sheet/1", "feature_set_hash": "d" * 64}


def test_a_run_that_reads_the_sheet_is_refused(tmp_path: Path) -> None:
    raw = copy.deepcopy(dataset_config().raw)
    raw["model"]["encoder"]["options"] = {"opponent_sheet": SET}
    with pytest.raises(ProgressError, match="opponent_sheet is not evaluated yet"):
        _source(tmp_path, read_run_config(raw), 0)
