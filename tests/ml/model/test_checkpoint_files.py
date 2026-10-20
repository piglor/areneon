"""Checkpoints as safetensors files whose id is the hash of their bytes (ADR-010 §2, §5.1)."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import struct
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import areneon_ml
import pytest
import torch
from areneon_ml.checkpoint.files import (
    CheckpointFileError,
    checkpoint_bytes,
    checkpoint_id,
    load_checkpoint,
    write_checkpoint,
)
from areneon_ml.model.batch import collate
from areneon_ml.model.gamedata import GameData
from areneon_ml.model.network import init_network
from areneon_ml.worker.checkpoints import CheckpointStore
from safetensors import deserialize

from .support import TINY_MODEL, encoder, game, sample, table

Mutate = Callable[[dict[str, Any]], None]


def make(role: str = "battle", seed: int = 1) -> bytes:
    return checkpoint_bytes(init_network(role, game(), TINY_MODEL, seed), game())


def split(data: bytes) -> tuple[dict[str, Any], bytes]:
    (size,) = struct.unpack("<Q", data[:8])
    return json.loads(data[8 : 8 + size]), data[8 + size :]


def join(header: dict[str, Any], body: bytes) -> bytes:
    text = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    return struct.pack("<Q", len(text)) + text + body


def rewrite(data: bytes, change: Mutate) -> bytes:
    header, body = split(data)
    change(header)
    return join(header, body)


def test_the_same_weights_give_the_same_bytes_and_the_id_is_their_hash() -> None:
    data = make()
    assert make() == data
    assert checkpoint_id(data) == hashlib.sha256(data).hexdigest()
    assert checkpoint_id(make(seed=2)) != checkpoint_id(data)
    assert checkpoint_id(make("preview")) != checkpoint_id(data)


def test_a_checkpoint_saves_loads_and_hashes_deterministically() -> None:
    net = init_network("battle", game(), TINY_MODEL, 3)
    data = checkpoint_bytes(net, game())
    loaded = load_checkpoint(data, game(), checkpoint_id(data), role="battle")
    assert checkpoint_bytes(loaded, game()) == data  # the round trip is byte-identical
    assert loaded.config == TINY_MODEL
    query = collate([encoder().encode(sample("policy-query/valid/battle.json"))])
    assert torch.equal(net(query).logits, loaded(query).logits)
    assert not loaded.training


def test_the_file_is_a_plain_safetensors_file_with_the_documented_layout() -> None:
    data = make()
    header, body = split(data)
    (size,) = struct.unpack("<Q", data[:8])
    assert size % 8 == 0
    meta = header["__metadata__"]
    assert set(meta) == {
        "areneon.format", "areneon.role", "areneon.model_config",
        "areneon.encoder_version", "areneon.engine_data_hash",
    }  # fmt: skip
    assert meta["areneon.engine_data_hash"] == game().engine_data_hash
    assert json.loads(meta["areneon.model_config"]) == TINY_MODEL.as_dict()
    tensors = dict(deserialize(data))  # type: ignore[no-untyped-call]
    names = sorted(name for name in header if name != "__metadata__")
    assert sorted(tensors) == names
    ends = [header[name]["data_offsets"] for name in names]
    assert ends[0][0] == 0
    assert all(a[1] == b[0] for a, b in itertools.pairwise(ends))  # name order, no gaps
    assert {t["dtype"] for t in tensors.values()} == {"F32"}
    assert sum(len(t["data"]) for t in tensors.values()) == len(body)
    assert b"pickle" not in data


def test_no_checkpoint_path_uses_pickle_or_torch_load() -> None:
    root = Path(areneon_ml.__file__).parent
    for source in root.rglob("*.py"):
        text = source.read_text(encoding="utf-8")
        assert "torch.load" not in text.replace("`torch.load`", ""), source
        assert "import pickle" not in text, source
        assert "weights_only" not in text, source


def test_a_loader_refuses_bytes_that_do_not_hash_to_the_id_it_was_given() -> None:
    data = make()
    tampered = data[:-1] + bytes([data[-1] ^ 1])
    with pytest.raises(CheckpointFileError, match="hash to another id"):
        load_checkpoint(tampered, game(), checkpoint_id(data))
    with pytest.raises(CheckpointFileError, match="hash to another id"):
        load_checkpoint(make(seed=2), game(), checkpoint_id(data))
    for bad in ("", "abc", checkpoint_id(data).upper(), checkpoint_id(data) + "\n"):
        with pytest.raises(CheckpointFileError, match="64 lowercase hex"):
            load_checkpoint(data, game(), bad)
    assert load_checkpoint(data, game()).role == "battle"  # without an id the caller vouches


def test_a_loader_refuses_another_role() -> None:
    with pytest.raises(CheckpointFileError, match="holds a battle network, not preview"):
        load_checkpoint(make("battle"), game(), role="preview")


def test_a_loader_refuses_another_engine_data_table_or_encoder_version() -> None:
    changed = copy.deepcopy(table())
    changed["items"][0]["fling_power"] += 1
    with pytest.raises(CheckpointFileError, match="another engine-data table"):
        load_checkpoint(make(), GameData.from_table(changed))
    data = rewrite(
        make(), lambda h: h["__metadata__"].update({"areneon.encoder_version": "enc1:x"})
    )
    with pytest.raises(CheckpointFileError, match="another encoder version"):
        load_checkpoint(data, game())


def _drop_tensor(header: dict[str, Any]) -> None:
    del header["blocks.0.ff.0.bias"]


def _add_tensor(header: dict[str, Any]) -> None:
    header["extra"] = {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]}


def _half_precision(header: dict[str, Any]) -> None:
    header["pending.bias"]["dtype"] = "F16"


def _reshape(header: dict[str, Any]) -> None:
    header["pending.bias"]["shape"] = [1, header["pending.bias"]["shape"][0]]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_drop_tensor, "not the network's parameters"),
        (_add_tensor, "not the network's parameters"),
        (_half_precision, "dtype or shape differs"),
        (_reshape, "dtype or shape differs"),
        (
            lambda h: h["__metadata__"].update({"areneon.format": "2"}),
            "unsupported checkpoint format",
        ),
        (
            lambda h: h["__metadata__"].update({"areneon.role": "value"}),
            "configuration is not valid",
        ),
        (
            lambda h: h["__metadata__"].update({"areneon.model_config": "{"}),
            "configuration is not valid",
        ),
        (
            lambda h: h["__metadata__"].update(
                {"areneon.model_config": json.dumps({"d_model": 1})}
            ),
            "configuration is not valid",
        ),
        (lambda h: h["__metadata__"].pop("areneon.role"), "metadata must hold exactly"),
        (lambda h: h["__metadata__"].update({"extra": "x"}), "metadata must hold exactly"),
        (lambda h: h["__metadata__"].update({"areneon.format": 1}), "must be strings"),
        (lambda h: h.pop("__metadata__"), "metadata must hold exactly"),
    ],
)
def test_a_file_that_does_not_fit_the_network_is_refused(change: Mutate, message: str) -> None:
    with pytest.raises(CheckpointFileError, match=message):
        load_checkpoint(rewrite(make(), change), game())


def test_a_damaged_container_is_refused() -> None:
    data = make()
    cases = {
        "shorter than its length prefix": data[:4],
        "header length is out of range": struct.pack("<Q", 1 << 40) + data[8:],
        "header length is out of range ": struct.pack("<Q", len(data)) + data[8:],
        "the header is not JSON": struct.pack("<Q", 4) + b"\xff\xfe\xfd\xfc" + data[12:],
        "the header is not an object": struct.pack("<Q", 8) + b"[1,2,3] " + b"",
        "repeats a key": struct.pack("<Q", 16) + b'{"a":1,"a":2}   ',
        "tensor data is not valid": data[:-4],
    }
    for message, broken in cases.items():
        with pytest.raises(CheckpointFileError, match=message.strip()):
            load_checkpoint(broken, game())


def test_a_big_endian_machine_refuses_to_write(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "byteorder", "big")
    with pytest.raises(CheckpointFileError, match="little-endian"):
        make()


def test_write_checkpoint_stores_the_file_under_its_id(tmp_path: Path) -> None:
    data = make()
    found = write_checkpoint(tmp_path, data)
    assert found == checkpoint_id(data)
    assert (tmp_path / "checkpoints" / f"{found}.safetensors").read_bytes() == data
    assert write_checkpoint(tmp_path, data) == found  # idempotent
    assert [p.name for p in (tmp_path / "checkpoints").iterdir()] == [f"{found}.safetensors"]
    assert CheckpointStore(tmp_path).load(found) == data
    loaded = load_checkpoint(CheckpointStore(tmp_path).load(found), game(), found)
    assert checkpoint_bytes(loaded, game()) == data
