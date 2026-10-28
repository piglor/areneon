"""The progress evaluation of a pair that reads a candidate feature set (ADR-022, #1181).

The tiny loop's stand-in cannot make a features run (its stream holds no rows), so the pair sources
are stubbed here: what is checked is the Python side's own part, that the policy ids it derives
carry the ``features`` member exactly as the training loop's do (another run's pair gets the ids
of ITS config) and that the record reader recomputes them. The end-to-end run with the real
worker is ``packages/team-dataset/test/progress-features.python.test.ts``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.learner.config import RunConfig, read_run_config
from areneon_ml.loop.records import features_hash, policy_id
from areneon_ml.progress import run as run_module
from areneon_ml.progress.records import ProgressRecordError, read_progress_record
from areneon_ml.progress.run import _Source

from tests.ml.progress.support import (
    FakeProgressPlayer,
    committed_manifest,
    dataset_config,
    make_run,
)

FEATURES = "c" * 64


def with_features(config: RunConfig) -> RunConfig:
    raw = copy.deepcopy(config.raw)
    raw["model"]["encoder"]["options"] = {
        "candidate_features": {"feature_set": "probe-damage/1", "feature_set_hash": FEATURES}
    }
    return read_run_config(raw)


def stub_source(generation: int) -> _Source:
    return _Source(
        generation,
        {},
        "0" * 64,
        f"{generation:x}" * 64,
        f"{generation + 8:x}" * 64,
        Path("team-set.json"),
        {},
    )


def test_every_pair_of_a_features_run_gets_the_training_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = with_features(dataset_config())
    monkeypatch.setattr(
        run_module, "_source", lambda _d, _c, generation, **_k: stub_source(generation)
    )
    previous = run_module._previous_pair(tmp_path, config, 1)
    anchor = run_module._anchor_pair(tmp_path, config, 0)
    ablation = run_module._ablation_spec(tmp_path, config, stub_source(1))
    before = stub_source(0)
    assert previous is not None
    assert previous.policy_id_sample == policy_id(
        before.preview, before.battle, "sample", features=FEATURES
    )
    assert anchor.policy_id_argmax == policy_id(
        before.preview, before.battle, "argmax", features=FEATURES
    )
    assert ablation.policy_id_argmax == policy_id(
        before.preview, stub_source(1).battle, "argmax", features=FEATURES
    )
    assert ablation.random_policy_id == policy_id(
        run_module.RANDOM_PREVIEW_POLICY, stub_source(1).battle, "argmax", features=FEATURES
    )
    # The same run without the option keeps today's ids.
    plain = dataset_config()
    again = run_module._previous_pair(tmp_path, plain, 1)
    assert again is not None
    assert again.policy_id_sample == policy_id(before.preview, before.battle, "sample")
    assert again.policy_id_sample != previous.policy_id_sample


def test_the_other_runs_pair_has_the_ids_of_its_own_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    mine = dataset_config()
    plain = dataset_config("plain-run", (21, 22))
    featured = with_features(dataset_config("featured-run", (23, 24)))
    monkeypatch.setattr(
        run_module, "_source", lambda _d, _c, generation, **_k: stub_source(generation)
    )
    spec = run_module.VersusSpec(other_dir, 1, 2, "enc-other")
    found = stub_source(1)
    for other, expected in ((plain, None), (featured, FEATURES)):
        monkeypatch.setattr(run_module, "_read_config", lambda _d, other=other: other)
        pair = run_module._versus_pair(tmp_path / "mine", mine, spec)
        assert pair.policy_id_sample == policy_id(
            found.preview, found.battle, "sample", features=expected
        )
        assert pair.policy_id_argmax == policy_id(
            found.preview, found.battle, "argmax", features=expected
        )


def test_the_reader_recomputes_the_features_member(tmp_path: Path) -> None:
    # A record the stand-in made for a plain run, re-labelled as a features run's: its ids lack the
    # member until they carry it, and the reader checks the candidate's and the same run's pairs.
    config = with_features(dataset_config())
    run = tmp_path / "run"
    plain = make_run(run, generations=1)
    root = tmp_path / "p"
    fake = FakeProgressPlayer(plain, committed_manifest(run, 0))
    run_module.progress_generation(run, 1, root, fake, 2, 1, (0,), 2, previous_matches=True)
    from areneon_ml.loop import layout  # noqa: PLC0415

    raw = layout.read_json(root / "g000001" / "progress.json")
    raw["run_id"], raw["run_config_hash"] = config.run_key, config.hash
    with pytest.raises(ProgressRecordError, match="policy_id is not the frozen pair"):
        read_progress_record(raw, config)
    featured = copy.deepcopy(raw)
    pair = raw["checkpoints"]
    for match in featured["matches"]:
        match["policy_id"] = policy_id(
            pair["preview"], pair["battle"], match["mode"], features=features_hash(config)
        )
        opponent = match["opponent"]
        if opponent["kind"] in ("previous", "anchor"):
            opponent["policy_id"] = policy_id(
                opponent["checkpoints"]["preview"],
                opponent["checkpoints"]["battle"],
                match["mode"],
                features=FEATURES,
            )
    assert read_progress_record(featured, config)["run_id"] == config.run_key
    # A previous or anchor pair of the same run without the member is refused too.
    broken: dict[str, Any] = copy.deepcopy(featured)
    victim = next(m for m in broken["matches"] if m["opponent"]["kind"] in ("previous", "anchor"))
    victim["opponent"]["policy_id"] = policy_id(
        victim["opponent"]["checkpoints"]["preview"],
        victim["opponent"]["checkpoints"]["battle"],
        victim["mode"],
    )
    with pytest.raises(ProgressRecordError, match="pair's policy_id"):
        read_progress_record(broken, config)
