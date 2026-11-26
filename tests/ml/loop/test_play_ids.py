"""``python -m areneon_ml.loop.play_ids``: the ids a direct run of the play script needs.

The driver computes the policy id and the encoder version itself; the throughput harness runs
the play script directly, so this module computes them from the same code, and a test holds
the answer equal to what the driver's own ``Frozen`` and ``encoder_version`` give.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from areneon_ml.learner.config import read_run_config
from areneon_ml.loop import play_ids
from areneon_ml.loop.driver import Frozen
from areneon_ml.loop.records import features_hash, policy_id
from areneon_ml.model.encoder import encoder_version
from areneon_ml.model.gamedata import GameData

from tests.ml.model.support import FORMAT_DIR, sample

PREVIEW = "1" * 64
BATTLE = "2" * 64
ENGINE_DATA = FORMAT_DIR / "engine-data.json"


def run(
    tmp_path: Path, raw: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> dict[str, str]:
    config = tmp_path / "config.json"
    config.write_text(json.dumps(raw), encoding="utf-8")
    code = play_ids.main(
        [
            *("--config", str(config)),
            *("--engine-data", str(ENGINE_DATA)),
            *("--preview", PREVIEW),
            *("--battle", BATTLE),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert out.count("\n") == 1  # one line
    found: dict[str, str] = json.loads(out)
    return found


def test_a_plain_run_gets_the_plain_policy_id_and_the_encoder_version_of_its_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = sample("training-run-config/valid/tiny.json")
    found = run(tmp_path, raw, capsys)
    config = read_run_config(raw)
    assert found["policy_id"] == policy_id(PREVIEW, BATTLE)
    assert found["encoder_version"] == encoder_version(
        GameData.from_table(json.loads(ENGINE_DATA.read_text(encoding="utf-8"))),
        config.model.encoder,
    )


def test_the_settings_that_enter_the_policy_id_are_the_drivers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = copy.deepcopy(sample("training-run-config/valid/tiny.json"))
    raw["generation"]["play"] = {"temperature_ppm": 500_000}
    raw["generation"]["inference"] = {"kind": "batched", "max_batch": 2}
    found = run(tmp_path, raw, capsys)
    config = read_run_config(raw)
    frozen = Frozen(
        0,
        {"preview": PREVIEW, "battle": BATTLE},
        "",
        None,
        (),
        features_hash(config),
        config.generation.temperature_ppm,
        config.generation.inference_max_batch,
        config.generation.inference_device,
    )
    assert found["policy_id"] == frozen.policy_id
    # the tempered, batched policy is another policy than the plain one
    assert found["policy_id"] != policy_id(PREVIEW, BATTLE)
    assert found["policy_id"] == policy_id(
        PREVIEW,
        BATTLE,
        temperature_ppm=500_000,
        inference={"kind": "batched", "max_batch": 2},
    )


def test_a_pair_that_reads_candidate_features_names_them_in_its_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = sample("training-run-config/valid/candidate-features.json")
    found = run(tmp_path, raw, capsys)
    config = read_run_config(raw)
    feature_hash = features_hash(config)
    assert feature_hash is not None
    assert found["policy_id"] == policy_id(PREVIEW, BATTLE, features=feature_hash)


def test_a_config_that_is_not_a_run_config_is_refused_with_a_message_and_no_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "config.json"
    config.write_text('{"not": "a run config"}', encoding="utf-8")
    args = ["--engine-data", str(ENGINE_DATA), "--preview", PREVIEW, "--battle", BATTLE]
    assert play_ids.main(["--config", str(config), *args]) == 2
    assert "play-ids:" in capsys.readouterr().err
    assert play_ids.main(["--config", str(tmp_path / "missing.json"), *args]) == 2
    assert "play-ids:" in capsys.readouterr().err
    config.write_text("{", encoding="utf-8")
    assert play_ids.main(["--config", str(config), *args]) == 2
    assert "play-ids:" in capsys.readouterr().err


def test_a_checkpoint_id_that_is_not_a_digest_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps(sample("training-run-config/valid/tiny.json")), encoding="utf-8")
    code = play_ids.main(
        [
            *("--config", str(config)),
            *("--engine-data", str(ENGINE_DATA)),
            *("--preview", "not-a-digest"),
            *("--battle", BATTLE),
        ]
    )
    assert code == 2
    assert "64 hexadecimal" in capsys.readouterr().err
