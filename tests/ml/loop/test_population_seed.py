"""``population-seed/1`` (ADR-021 section 5) and the driver's refusal of a population config."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from areneon_ml.learner.config import read_run_config
from areneon_ml.loop.driver import LoopError
from areneon_ml.loop.population_seed import (
    POPULATION_SEED_DERIVATION,
    POPULATION_SEED_PURPOSES,
    population_draw,
)

from tests.ml.loop.support import make_loop
from tests.ml.model.support import SAMPLES

# Computed outside this code base (Python hashlib, compact sorted JSON):
#   int.from_bytes(hashlib.sha256(json.dumps({"run_key": k, "generation": g, "member": m,
#   "purpose": p, "index": i}, sort_keys=True, separators=(",", ":")).encode()).digest()[:8], "big")
# A change to any of these numbers is a new derivation version. The TS test pins the same six.
VECTORS = [
    ("tiny-run", 0, "m0", "member-init", 0, 13019428712076808456),
    ("tiny-run", 0, "m1", "member-init", 1, 18329199380263047422),
    ("tiny-run", 3, "m2", "opponent-draw", 17, 12259119117443046183),
    ("pop-main", 12, "x0", "explore", 2, 6998671094935325674),
    ("pop-main", 7, "m3", "bootstrap", 999, 18168013126753230032),
    ("pop-main", 7, "m3", "member-learner", 0, 6024445643219269802),
]


def test_the_derivation_is_named_population_seed_1() -> None:
    assert POPULATION_SEED_DERIVATION == "population-seed/1"
    assert POPULATION_SEED_PURPOSES == (
        "member-init",
        "member-learner",
        "opponent-draw",
        "explore",
        "bootstrap",
    )


@pytest.mark.parametrize("vector", VECTORS)
def test_the_draw_reproduces_the_pinned_vector(vector: tuple[str, int, str, str, int, int]) -> None:
    *args, draw = vector
    assert population_draw(*args) == draw  # type: ignore[arg-type]


def test_every_input_changes_the_draw() -> None:
    base = ("a", 1, "m0", "explore", 0)
    variants = [
        base,
        ("b", 1, "m0", "explore", 0),
        ("a", 2, "m0", "explore", 0),
        ("a", 1, "m1", "explore", 0),
        ("a", 1, "x0", "explore", 0),
        ("a", 1, "m0", "bootstrap", 0),
        ("a", 1, "m0", "explore", 1),
    ]
    assert len({population_draw(*v) for v in variants}) == len(variants)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("a", -1, "m0", "explore", 0), "count from 0"),
        (("a", 0, "m0", "explore", -1), "count from 0"),
        (("a", 0, "0", "explore", 0), "m<k> or x<k>"),
        (("a", 0, "m01", "explore", 0), "m<k> or x<k>"),
        (("a", 0, "m100", "explore", 0), "m<k> or x<k>"),
        (("a", 0, "y0", "explore", 0), "m<k> or x<k>"),
        (("a", 0, "m0", "shuffle", 0), "purpose"),
    ],
)
def test_an_input_out_of_range_is_refused(args: tuple[object, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        population_draw(*args)  # type: ignore[arg-type]


def test_the_loop_driver_refuses_a_population_config(tmp_path: Path) -> None:
    config = read_run_config(
        json.loads((SAMPLES / "training-run-config/valid/population.json").read_text())
    )
    assert config.population is True
    with pytest.raises(LoopError, match="population: this driver runs a single lineage"):
        make_loop(tmp_path, config)
    assert list(tmp_path.iterdir()) == []
