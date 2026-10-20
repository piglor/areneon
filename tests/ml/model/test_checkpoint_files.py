"""Checkpoints as safetensors files whose id is the hash of their bytes (ADR-010 §2, §5.1)."""

from __future__ import annotations

import ast
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
from areneon_ml.model.network import PolicyNetwork, init_network
from areneon_ml.worker.checkpoints import CheckpointStore
from safetensors import deserialize

from .support import TINY_MODEL, encoder, game, sample, table

Mutate = Callable[[dict[str, Any]], None]


def make(role: str = "battle", seed: int = 1) -> bytes:
    return checkpoint_bytes(init_network(role, game(), TINY_MODEL, seed), game())


def load(data: bytes, game_: GameData | None = None, role: str | None = None) -> PolicyNetwork:
    """The file's own id: for tests of what the loader does after the hash gate."""
    return load_checkpoint(data, game_ or game(), checkpoint_id(data), role=role)


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


BANNED_IMPORTS = {"pickle", "cPickle", "dill", "cloudpickle", "shelve", "marshal", "numpy"}
BANNED_CALLS = {("torch", "load"), ("torch", "jit", "load")}


def violations(source: str) -> list[str]:
    """Imports of a serialisation or numpy module, and calls of torch.load / torch.jit.load."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if a.name.split(".")[0] in BANNED_IMPORTS]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.split(".")[0] in BANNED_IMPORTS:
                found.append(module)
            if module in ("torch", "torch.jit") and any(a.name == "load" for a in node.names):
                found.append(f"{module}.load")
        elif isinstance(node, ast.Attribute) and node.attr == "load":
            chain = []
            value: ast.expr = node
            while isinstance(value, ast.Attribute):
                chain.append(value.attr)
                value = value.value
            if isinstance(value, ast.Name) and (value.id, *reversed(chain)) in BANNED_CALLS:
                found.append(".".join((value.id, *reversed(chain))))
    return found


def test_the_scan_finds_what_it_looks_for() -> None:
    assert violations("import pickle") == ["pickle"]
    assert violations("from numpy import array") == ["numpy"]
    assert violations("import numpy.linalg as la") == ["numpy.linalg"]
    assert violations("import torch\nx = torch.load(p)") == ["torch.load"]
    assert violations("import torch\nx = torch.jit.load(p)") == ["torch.jit.load"]
    assert violations("from torch import load") == ["torch.load"]
    assert violations("from torch.jit import load") == ["torch.jit.load"]
    assert violations("import json\nimport torch\njson.load(f)\nnp.load(f)") == []


def test_no_module_uses_pickle_torch_load_or_numpy() -> None:
    root = Path(areneon_ml.__file__).parent
    sources = list(root.rglob("*.py"))
    assert len(sources) > 25
    for source in sources:
        assert violations(source.read_text(encoding="utf-8")) == [], source


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
    with pytest.raises(TypeError):
        load_checkpoint(data, game())  # type: ignore[call-arg]


def test_a_loader_refuses_another_role() -> None:
    with pytest.raises(CheckpointFileError, match="holds a battle network, not preview"):
        load(make("battle"), role="preview")


def test_a_loader_refuses_another_engine_data_table_or_encoder_version() -> None:
    changed = copy.deepcopy(table())
    changed["items"][0]["fling_power"] += 1
    with pytest.raises(CheckpointFileError, match="another engine-data table"):
        load(make(), GameData.from_table(changed))
    data = rewrite(
        make(), lambda h: h["__metadata__"].update({"areneon.encoder_version": "enc1:x"})
    )
    with pytest.raises(CheckpointFileError, match="another encoder version"):
        load(data)


def _drop_tensor(header: dict[str, Any]) -> None:
    del header["blocks.0.ff.0.bias"]


def _add_tensor(header: dict[str, Any]) -> None:
    header["extra"] = {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]}


def _half_precision(header: dict[str, Any]) -> None:
    header["pending.bias"]["dtype"] = "F16"


def _reshape(header: dict[str, Any]) -> None:
    header["pending.bias"]["shape"] = [1, header["pending.bias"]["shape"][0]]


def _scalar_shape(header: dict[str, Any]) -> None:
    header["pending.bias"]["shape"] = 5


def _not_an_entry(header: dict[str, Any]) -> None:
    header["pending.bias"] = 5


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_drop_tensor, "not the network's parameters"),
        (_add_tensor, "not the network's parameters"),
        (_half_precision, "dtype or shape differs"),
        (_reshape, "dtype or shape differs"),
        (_scalar_shape, "dtype or shape differs"),
        (_not_an_entry, "dtype or shape differs"),
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
        load(rewrite(make(), change))


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
            load(broken)


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


def test_a_network_off_the_host_cannot_be_written() -> None:
    # data_ptr() of a device tensor is no host address: the writer copies to the host first, and
    # a tensor that cannot be copied (here the meta device, which has no data) is refused.
    net = init_network("battle", game(), TINY_MODEL, 1).to("meta")
    with pytest.raises(CheckpointFileError, match="cannot be copied to the host"):
        checkpoint_bytes(net, game())


def test_a_crafted_size_cannot_make_the_loader_allocate() -> None:
    def huge(header: dict[str, Any]) -> None:
        config = json.loads(header["__metadata__"]["areneon.model_config"])
        config["d_model"] = 1 << 40
        header["__metadata__"]["areneon.model_config"] = json.dumps(config)

    with pytest.raises(CheckpointFileError, match="d_model must be at least 1 and at most"):
        load(rewrite(make(), huge))

    def many_buckets(header: dict[str, Any]) -> None:
        config = json.loads(header["__metadata__"]["areneon.model_config"])
        config["encoder"]["volatile_buckets"] = 1 << 40
        header["__metadata__"]["areneon.model_config"] = json.dumps(config)

    with pytest.raises(
        CheckpointFileError, match="volatile_buckets must be at least 1 and at most"
    ):
        load(rewrite(make(), many_buckets))
