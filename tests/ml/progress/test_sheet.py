"""A pair that reads the opponent sheet is evaluated like any other (ADR-027 slice 5, #1272 part 3).

The refusal of part 2 is lifted: the run reaches the ordinary checks of its generation. The seats
build the blocks on the Node side (``packages/team-dataset/test/progress-sheet.python.test.ts``).
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from areneon_ml.learner.config import read_run_config
from areneon_ml.progress.run import ProgressError, _source

from tests.ml.progress.support import dataset_config

SET = {"feature_set": "opponent-sheet/1", "feature_set_hash": "d" * 64}


def test_a_run_that_reads_the_sheet_is_no_longer_refused(tmp_path: Path) -> None:
    raw = copy.deepcopy(dataset_config().raw)
    raw["model"]["encoder"]["options"] = {"opponent_sheet": SET}
    with pytest.raises(ProgressError, match="is not committed"):
        _source(tmp_path, read_run_config(raw), 0)
