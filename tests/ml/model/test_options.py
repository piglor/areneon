"""Optional model options (ADR-010 erratum A4 item 2): absent keeps today's bytes and hashes."""

from __future__ import annotations

import json
import struct
from typing import Any

import pytest
from areneon_ml.checkpoint.files import (
    FORMAT,
    FORMAT_OPTIONS,
    CheckpointFileError,
    CheckpointOptionsError,
    checkpoint_bytes,
    checkpoint_id,
    load_checkpoint,
    read_header,
)
from areneon_ml.model.encoder import encoder_version
from areneon_ml.model.network import ConfigError, ModelConfig, init_network
from areneon_ml.model.options import REGISTRY, OptionError, OptionSpec, read_options

from tests.ml.model.support import TINY_ENCODER, TINY_MODEL, game

# Pinned from the commit before the options existed (computed there, not here).
TODAY_CHECKPOINT_ID = "29d3b4fee589efe310884ba8b15a545da0911cdafa6f872694e105c2f0c40455"
TODAY_ENCODER_VERSION = "enc1:cec125199e5fc0b05a2a1d694ee38064851d1a4494ac4b571b3fccd6bc3f41dd"


@pytest.fixture
def test_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two test-only options: one per scope, the network one changes the layout."""
    monkeypatch.setitem(
        REGISTRY["network"], "test_layout", OptionSpec("test_layout", lambda v: v in (0, 1), True)
    )
    monkeypatch.setitem(
        REGISTRY["encoder"], "test_feature", OptionSpec("test_feature", lambda v: v >= 0)
    )


def with_options(
    network: dict[str, int] | None = None, encoder: dict[str, int] | None = None
) -> ModelConfig:
    source: dict[str, Any] = TINY_MODEL.as_dict()
    if network is not None:
        source["options"] = network
    if encoder is not None:
        source["encoder"]["options"] = encoder
    return ModelConfig.from_dict(source)


def test_without_options_today_s_bytes_and_hashes_are_unchanged() -> None:
    assert "options" not in TINY_MODEL.as_dict()
    assert "options" not in TINY_ENCODER.as_dict()
    assert encoder_version(game(), TINY_ENCODER) == TODAY_ENCODER_VERSION
    data = checkpoint_bytes(init_network("battle", game(), TINY_MODEL, seed=1), game())
    assert checkpoint_id(data) == TODAY_CHECKPOINT_ID
    assert read_header(data)["__metadata__"]["areneon.format"] == FORMAT


def test_only_the_registered_options_exist() -> None:
    assert set(REGISTRY["encoder"]) == {
        "candidate_matchup",
        "candidate_features",
        "candidate_target",
        "recent_history",
    }
    assert REGISTRY["network"] == {}
    with pytest.raises(OptionError, match="unknown network option 'x'"):
        read_options("network", {"x": 1})
    with pytest.raises(ConfigError, match="unknown encoder option"):
        ModelConfig.from_dict(
            {**TINY_MODEL.as_dict(), "encoder": {**TINY_ENCODER.as_dict(), "options": {"x": 1}}}
        )
    with pytest.raises(OptionError, match="scope"):
        read_options("other", {})
    with pytest.raises(OptionError, match="object"):
        read_options("network", [])


def test_an_empty_options_object_is_no_option() -> None:
    config = with_options(network={}, encoder={})
    assert config == TINY_MODEL
    assert config.as_dict() == TINY_MODEL.as_dict()
    assert encoder_version(game(), config.encoder) == TODAY_ENCODER_VERSION


def test_an_option_moves_the_config_the_encoder_version_and_the_checkpoint(
    test_options: None,
) -> None:
    network = with_options(network={"test_layout": 1})
    encoder = with_options(encoder={"test_feature": 3})
    assert network.as_dict()["options"] == {"test_layout": 1}
    assert encoder.as_dict()["encoder"]["options"] == {"test_feature": 3}
    for config in (network, encoder):
        assert ModelConfig.from_dict(json.loads(json.dumps(config.as_dict()))) == config
        assert config != TINY_MODEL
    assert encoder_version(game(), encoder.encoder) != TODAY_ENCODER_VERSION
    # Network options are not part of the encoder version; encoder options are.
    assert encoder_version(game(), network.encoder) == TODAY_ENCODER_VERSION
    assert encoder_version(game(), with_options(encoder={"test_feature": 4}).encoder) != (
        encoder_version(game(), encoder.encoder)
    )
    assert (
        checkpoint_id(checkpoint_bytes(init_network("battle", game(), network, 1), game()))
        != TODAY_CHECKPOINT_ID
    )


def test_an_option_value_is_checked(test_options: None) -> None:
    for bad in (2, "1", 1.5, None):
        with pytest.raises(ConfigError, match="invalid value"):
            with_options(network={"test_layout": bad})  # type: ignore[dict-item]


def test_only_a_layout_changing_option_bumps_the_checkpoint_format(test_options: None) -> None:
    cases = {
        "plain": (with_options(encoder={"test_feature": 1}), FORMAT),
        "layout": (with_options(network={"test_layout": 0}), FORMAT_OPTIONS),
    }
    for config, expected in cases.values():
        data = checkpoint_bytes(init_network("preview", game(), config, 2), game())
        assert read_header(data)["__metadata__"]["areneon.format"] == expected
        loaded = load_checkpoint(data, game(), checkpoint_id(data), "preview", configured=config)
        assert loaded.config == config
        assert checkpoint_bytes(loaded, game()) == data


def test_a_checkpoint_of_other_options_is_refused(test_options: None) -> None:
    layout = with_options(network={"test_layout": 1})
    feature = with_options(encoder={"test_feature": 1})
    for made, configured in (
        (layout, TINY_MODEL),
        (TINY_MODEL, layout),
        (layout, with_options(network={"test_layout": 0})),
        (feature, TINY_MODEL),
        (TINY_MODEL, feature),
        (feature, with_options(encoder={"test_feature": 2})),
    ):
        data = checkpoint_bytes(init_network("battle", game(), made, 1), game())
        with pytest.raises(CheckpointOptionsError, match="other model options"):
            load_checkpoint(data, game(), checkpoint_id(data), configured=configured)
    # Without a configured model the file stands on its own, as before.
    data = checkpoint_bytes(init_network("battle", game(), layout, 1), game())
    assert load_checkpoint(data, game(), checkpoint_id(data)).config == layout
    assert issubclass(CheckpointOptionsError, CheckpointFileError)


def _with_format(config: ModelConfig, format_: str) -> tuple[bytes, str]:
    """A checkpoint of ``config`` whose header says ``format_`` (and its id)."""
    data = checkpoint_bytes(init_network("battle", game(), config, 1), game())
    header = read_header(data)
    header["__metadata__"]["areneon.format"] = format_
    (old,) = struct.unpack("<Q", data[:8])
    text = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    out = struct.pack("<Q", len(text)) + text + data[8 + old :]
    return out, checkpoint_id(out)


def test_a_checkpoint_whose_format_does_not_fit_its_options_is_refused(test_options: None) -> None:
    layout = with_options(network={"test_layout": 1})
    for config, wrong in ((TINY_MODEL, FORMAT_OPTIONS), (layout, FORMAT)):
        data, found = _with_format(config, wrong)
        with pytest.raises(CheckpointFileError, match="format does not fit"):
            load_checkpoint(data, game(), found)
    data, found = _with_format(TINY_MODEL, "3")
    with pytest.raises(CheckpointFileError, match="unsupported checkpoint format"):
        load_checkpoint(data, game(), found)
