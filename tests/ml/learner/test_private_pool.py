"""A learner reads a private TrainingSet only for a run config with a private pool (ADR-015 E1)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_ml.learner.data import LearnerDataError, trajectories
from areneon_ml.model.encoder import Encoder
from areneon_ml.model.network import ROLES

from tests.ml.learner.support import POLICY, make_stream, tiny_config
from tests.ml.model.support import game


def encoders() -> dict[str, Encoder]:
    return {role: Encoder(game(), tiny_config().model.encoder) for role in ROLES}


def privatised() -> tuple[str, dict[str, Any]]:
    """The default stream with every example private, and its private manifest."""
    text, manifest = make_stream()
    lines = []
    for line in text.split("\n")[:-1]:
        raw = json.loads(line)
        raw["observation"]["data_class"] = "private"
        lines.append(jcs(raw).decode("utf-8") + "\n")
    stream = "".join(lines)
    digest = sha256_hex(stream.encode("utf-8"))
    return stream, {**manifest, "data_class": "private", "stream_sha256": digest}


def test_each_pool_kind_refuses_the_other_kinds_set() -> None:
    stream, manifest = privatised()
    with pytest.raises(LearnerDataError, match="without a private pool refuses a private"):
        trajectories(stream, manifest, encoders(), POLICY)
    text, synthetic = make_stream()
    with pytest.raises(LearnerDataError, match="private pool reads private TrainingSets only"):
        trajectories(text, synthetic, encoders(), POLICY, private_pool=True)


def test_a_private_pool_reads_a_private_set() -> None:
    stream, manifest = privatised()
    assert trajectories(stream, manifest, encoders(), POLICY, private_pool=True)
