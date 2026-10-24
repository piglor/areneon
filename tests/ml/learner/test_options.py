"""Learner state and model options (ADR-010 erratum A4 item 2)."""

from __future__ import annotations

import dataclasses
import json

import pytest
from areneon_contracts.contracts import jcs
from areneon_ml.learner.config import RunConfig
from areneon_ml.learner.state import (
    LearnerOptionsError,
    LearnerStateError,
    model_options_of,
    pack_state,
    restore_state,
    state_hash,
)
from areneon_ml.model.options import REGISTRY, OptionSpec

from tests.ml.learner.support import learner, tiny_config
from tests.ml.model.test_options import with_options


def under(config: RunConfig, **options: dict[str, int]) -> RunConfig:
    """``config`` with model options set; the run config hash is kept to isolate the check."""
    model = with_options(network=options.get("network"), encoder=options.get("encoder"))
    return dataclasses.replace(config, model=model)


@pytest.fixture(autouse=True)
def _options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        REGISTRY["network"], "test_layout", OptionSpec("test_layout", lambda v: v in (0, 1), True)
    )
    monkeypatch.setitem(
        REGISTRY["encoder"], "test_feature", OptionSpec("test_feature", lambda v: v >= 0)
    )


def test_the_sidecar_records_options_only_when_present() -> None:
    plain = json.loads(pack_state(learner()).sidecar)
    assert "model_options" not in plain
    config = under(tiny_config(), network={"test_layout": 1}, encoder={"test_feature": 2})
    both = json.loads(pack_state(learner(config)).sidecar)
    assert both["model_options"] == {"encoder": {"test_feature": 2}, "network": {"test_layout": 1}}
    assert model_options_of(tiny_config().model) == {}


def test_a_state_restores_under_its_own_options() -> None:
    config = under(tiny_config(), network={"test_layout": 1})
    packed = pack_state(learner(config))
    target = learner(config)
    restore_state(target, packed.tensors, packed.sidecar, packed.hash)
    assert pack_state(target) == packed


@pytest.mark.parametrize(
    ("made", "restored"),
    [
        ({"network": {"test_layout": 1}}, {}),
        ({}, {"network": {"test_layout": 1}}),
        ({"network": {"test_layout": 1}}, {"network": {"test_layout": 0}}),
        ({"encoder": {"test_feature": 1}}, {}),
        ({}, {"encoder": {"test_feature": 1}}),
    ],
)
def test_a_state_of_other_options_is_refused_with_a_typed_error(
    made: dict[str, dict[str, int]], restored: dict[str, dict[str, int]]
) -> None:
    packed = pack_state(learner(under(tiny_config(), **made)))
    target = learner(under(tiny_config(), **restored))
    before = pack_state(target)
    with pytest.raises(LearnerOptionsError, match="other model options"):
        restore_state(target, packed.tensors, packed.sidecar, packed.hash)
    assert pack_state(target) == before
    assert issubclass(LearnerOptionsError, LearnerStateError)


def test_a_sidecar_that_names_empty_options_is_refused() -> None:
    packed = pack_state(learner())
    sidecar = json.loads(packed.sidecar)
    sidecar["model_options"] = {}
    with pytest.raises(LearnerOptionsError):
        restore_state(learner(), packed.tensors, jcs(sidecar), state_hash(packed.tensors, sidecar))
