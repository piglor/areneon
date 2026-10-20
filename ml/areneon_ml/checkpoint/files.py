"""Checkpoints as safetensors files (ADR-010 §2, §5, §5.1).

A checkpoint is one ``.safetensors`` file: tensors and a JSON header only, no
pickle, and ``torch.load`` is never called. Its id is the SHA-256 of the
file's bytes, so the same weights, role and configuration always give the same
file and the same id, and a loader that is told an id refuses bytes that do
not hash to it. The header's ``__metadata__`` records what is needed to
rebuild the network and to refuse a mismatch: the role, the model
configuration, the encoder version and the engine-data hash. Nothing about
training (optimiser, RNG, generation) is here: that is the learner state.

The tensors are read with the library's bounds-checked ``deserialize`` and
copied out, only after the expected names, dtypes and shapes of the network
have been checked against the header, so a file cannot make this code read a
buffer of another size.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import struct
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from areneon_contracts.contracts import jcs
from safetensors import SafetensorError, deserialize

from areneon_ml.model.encoder import encoder_version
from areneon_ml.model.gamedata import GameData
from areneon_ml.model.network import ConfigError, ModelConfig, PolicyNetwork, init_network

FORMAT = "1"
LENGTH_PREFIX = 8  # bytes of the little-endian header length
PREFIX = "areneon."
MAX_HEADER_BYTES = 1 << 20  # the JSON header of a checkpoint is a few kilobytes
_ID = re.compile(r"[0-9a-f]{64}")
_DTYPE = "F32"


class CheckpointFileError(ValueError):
    """Bytes are not a checkpoint this build can load, or are not the one asked for."""


def checkpoint_id(data: bytes) -> str:
    """The id of a checkpoint file: SHA-256 of its bytes, lowercase hex."""
    return hashlib.sha256(data).hexdigest()


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise CheckpointFileError("the header repeats a key")
    return dict(pairs)


def _header(data: bytes) -> dict[str, Any]:
    """The safetensors JSON header, with a size cap and no duplicate keys."""
    if len(data) < LENGTH_PREFIX:
        raise CheckpointFileError("not a safetensors file: shorter than its length prefix")
    (size,) = struct.unpack("<Q", data[:LENGTH_PREFIX])
    if size > MAX_HEADER_BYTES or LENGTH_PREFIX + size > len(data):
        raise CheckpointFileError("not a safetensors file: the header length is out of range")
    try:
        header = json.loads(
            data[LENGTH_PREFIX : LENGTH_PREFIX + size].decode("utf-8"),
            object_pairs_hook=_no_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise CheckpointFileError("not a safetensors file: the header is not JSON") from exc
    if not isinstance(header, dict):
        raise CheckpointFileError("not a safetensors file: the header is not an object")
    return header


def _metadata(header: Mapping[str, Any]) -> dict[str, str]:
    meta = header.get("__metadata__")
    expected = {
        f"{PREFIX}{k}"
        for k in ("format", "role", "model_config", "encoder_version", "engine_data_hash")
    }
    if not isinstance(meta, dict) or set(meta) != expected:
        raise CheckpointFileError(f"the header's metadata must hold exactly {sorted(expected)}")
    if not all(isinstance(v, str) for v in meta.values()):
        raise CheckpointFileError("the header's metadata values must be strings")
    return dict(meta)


def checkpoint_bytes(network: PolicyNetwork, game: GameData) -> bytes:
    """The safetensors file of ``network``: the same bytes for the same weights.

    The library's serializer orders the metadata by hash-map iteration, so the
    bytes of one network would differ between runs and an id would not name a
    network. The layout is written here instead, in the documented format:
    an 8-byte little-endian header length, the JSON header (sorted keys,
    padded with spaces to a multiple of 8) and the tensors' bytes in name order.
    """
    if sys.byteorder != "little":
        raise CheckpointFileError("checkpoints are little-endian; this machine is not")
    state = {}
    for name, tensor in sorted(network.state_dict().items()):
        try:
            # .cpu() first: data_ptr() of a device tensor is not a host address.
            state[name] = tensor.detach().cpu().to(torch.float32).contiguous()
        except (NotImplementedError, RuntimeError) as exc:
            raise CheckpointFileError(f"tensor {name} cannot be copied to the host") from exc
    header: dict[str, Any] = {
        "__metadata__": {
            f"{PREFIX}format": FORMAT,
            f"{PREFIX}role": network.role,
            f"{PREFIX}model_config": jcs(network.config.as_dict()).decode("utf-8"),
            f"{PREFIX}encoder_version": encoder_version(game, network.config.encoder),
            f"{PREFIX}engine_data_hash": game.engine_data_hash,
        }
    }
    chunks = []
    offset = 0
    for name, tensor in state.items():
        size = tensor.numel() * tensor.element_size()
        header[name] = {
            "dtype": _DTYPE,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + size],
        }
        chunks.append(ctypes.string_at(tensor.data_ptr(), size))
        offset += size
    text = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % LENGTH_PREFIX)
    return struct.pack("<Q", len(text)) + text + b"".join(chunks)


def write_checkpoint(run_dir: Path, data: bytes) -> str:
    """Store ``data`` as ``<run_dir>/checkpoints/<id>.safetensors`` and return the id."""
    found = checkpoint_id(data)
    directory = run_dir / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{found}.safetensors"
    partial = directory / f".{found}.partial"
    partial.write_bytes(data)
    os.replace(partial, target)
    return found


def load_checkpoint(
    data: bytes, game: GameData, expected_id: str, role: str | None = None
) -> PolicyNetwork:
    """Rebuild the network a checkpoint holds, refusing anything that does not fit.

    ``expected_id`` is required: the hash gate runs before the header is read or
    anything is allocated, and the configuration is bounded as well, so a crafted
    file cannot make this allocate without limit.

    Refused: bytes that do not hash to ``expected_id``; a file of another format or
    (when ``role`` is given) another role;
    another engine-data table or encoder version than ``game``'s; a configuration
    that is not valid; tensors whose names, dtype or shapes differ from the network's.
    """
    if not _ID.fullmatch(expected_id):
        raise CheckpointFileError("a checkpoint id is 64 lowercase hex digits")
    if checkpoint_id(data) != expected_id:
        raise CheckpointFileError(f"checkpoint {expected_id}: its bytes hash to another id")
    header = _header(data)
    meta = _metadata(header)
    if meta[f"{PREFIX}format"] != FORMAT:
        raise CheckpointFileError(f"unsupported checkpoint format {meta[f'{PREFIX}format']!r}")
    if role is not None and meta[f"{PREFIX}role"] != role:
        raise CheckpointFileError(
            f"the checkpoint holds a {meta[f'{PREFIX}role']} network, not {role}"
        )
    try:
        config = ModelConfig.from_dict(json.loads(meta[f"{PREFIX}model_config"]))
        network = init_network(meta[f"{PREFIX}role"], game, config, seed=0)
    except (json.JSONDecodeError, ConfigError, RuntimeError, MemoryError) as exc:
        raise CheckpointFileError(f"the checkpoint's configuration is not valid: {exc}") from exc
    if meta[f"{PREFIX}engine_data_hash"] != game.engine_data_hash:
        raise CheckpointFileError("the checkpoint was made for another engine-data table")
    if meta[f"{PREFIX}encoder_version"] != encoder_version(game, config.encoder):
        raise CheckpointFileError("the checkpoint was made for another encoder version")
    expected = {name: tuple(t.shape) for name, t in network.state_dict().items()}
    listed = {name: entry for name, entry in header.items() if name != "__metadata__"}
    if set(listed) != set(expected):
        raise CheckpointFileError("the checkpoint's tensors are not the network's parameters")
    for name, entry in listed.items():
        shape = entry.get("shape") if isinstance(entry, dict) else None
        if not (
            isinstance(entry, dict)
            and entry.get("dtype") == _DTYPE
            and isinstance(shape, list)
            and tuple(shape) == expected[name]
        ):
            raise CheckpointFileError(f"tensor {name}: dtype or shape differs from the network's")
    try:
        tensors = dict(deserialize(data))  # type: ignore[no-untyped-call]
    except SafetensorError as exc:
        raise CheckpointFileError(f"the tensor data is not valid: {exc}") from exc
    state = {
        name: torch.frombuffer(bytearray(tensors[name]["data"]), dtype=torch.float32)
        .reshape(shape)
        .clone()
        for name, shape in expected.items()
    }
    network.load_state_dict(state, strict=True)
    network.eval()
    return network
