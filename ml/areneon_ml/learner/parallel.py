"""Check and encode a stream file in worker processes, in order (#1069, #964).

The load phase checks every line against its contract and encodes it, one process, one core. A
stream lists an Episode's examples together, so it can be cut between Episodes into pieces that
need nothing from each other: each worker checks and encodes one piece (``StreamChecker`` with the
piece's own line numbers, ``_Collector`` for the steps) and the pieces are put back together in
file order. The result is the serial one, tensor for tensor: the encoder is a pure function of its
line and the Episode's earlier lines, which are in the same piece, and the workers run torch on
one thread.

What the serial path refuses, this refuses, with the same exception, in the same order of
precedence: the manifest, the file's digest, last newline and line count (read first, here too),
then the first line that breaks the contract, then the rules over the whole stream, then the first
line the encoder or the collector refuses. A worker that meets a contract error stops at it; one
that meets an encoder error keeps checking its piece (a later contract error outranks it) and
encodes nothing more. Workers are forked, so they share the encoders and the manifest without
copying them; a platform without ``fork`` reads the stream in one process.
"""

from __future__ import annotations

import ctypes
import dataclasses
import importlib
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from areneon_contracts.training import (
    StreamChecker,
    TrainingStreamError,
    check_stream_digest,
    stream_split_points,
)

from areneon_ml.learner.data import Collected, LearnerDataError, Step, Trajectory, _Collector
from areneon_ml.model.encoder import EncodedQuery, Encoder

#: Pieces per worker: more pieces than workers keep them all busy when Episodes differ in size.
PIECES_PER_WORKER = 4
#: The fewest workers and pieces for which a parallel read is worth its start-up.
MIN_WORKERS = 2
MIN_POINTS = 3  # a piece needs two points
#: Never cut a stream into pieces smaller than this many bytes.
MIN_PIECE_BYTES = 1 << 20


#: What one worker adds to the driver's memory, in MB, for sizing: the largest resident set of a
#: worker measured on a 1.2 GB stream was 286 MB, of which the part shared with the driver by
#: fork is not new. Deliberately above what was measured.
WORKER_MB = 400


def fit_processes(requested: int, memory_mb: int | None) -> int:
    """``requested`` workers, or fewer if ``WORKER_MB`` each does not fit in ``memory_mb``."""
    if memory_mb is None:
        return requested
    return max(1, min(requested, memory_mb // WORKER_MB))


@dataclass
class _Piece:
    """What one worker saw of its piece of the stream."""

    collected: Collected
    seen: set[str]
    classes: set[str]
    #: The first contract error: its message (the first of the stream wins).
    contract_error: str | None = None
    #: The first error of the encoder or the collector: its class (module and name) and message.
    encode_error: tuple[str, str] | None = None


# What the workers inherit by fork; set before the pool starts, cleared after.
_SHARED: dict[str, Any] = {}


def _run_piece(piece: tuple[int, int, int]) -> bytes:
    """``_check_piece`` as bytes (``pack_piece``).

    ``multiprocessing`` would send a tensor through shared memory, one file descriptor per
    storage (``torch.multiprocessing`` registers that), and thousands of small tensors make that
    slower than the encoding itself; one blob of bytes is one read.
    """
    return pack_piece(_check_piece(piece))


def _check_piece(piece: tuple[int, int, int]) -> _Piece:
    """Check and encode the lines in bytes ``[start, end)``, the first being line ``first``."""
    start, end, first = piece
    torch.set_num_threads(1)
    checker = StreamChecker(_SHARED["manifest"])
    collector = _Collector(_SHARED["encoders"])
    result = _Piece(collector.collected(), checker.seen, checker.classes)
    with Path(_SHARED["path"]).open("rb") as handle:
        handle.seek(start)
        data = handle.read(end - start)
    for number, raw_line in enumerate(data.split(b"\n")[:-1], start=first):
        try:
            text = raw_line.decode("utf-8") + "\n"
        except UnicodeDecodeError:
            result.contract_error = "the stream is not UTF-8"
            return result
        try:
            raw, _ = checker.line(number, text)
        except TrainingStreamError as error:
            result.contract_error = str(error)
            return result
        if result.encode_error is None:
            try:
                collector.add(raw)
            except Exception as error:  # re-raised by the caller, in line order
                kind = type(error)
                result.encode_error = (f"{kind.__module__}.{kind.__qualname__}", str(error))
    return result


_DTYPES = {"int64": torch.int64, "float32": torch.float32}
_ALIGN = 8


def pack_piece(piece: _Piece) -> bytes:
    """A piece as one blob: an 8-byte header length, a JSON header, the tensors' bytes.

    Neither pickle nor numpy (a guard bans both in this package): the header names every Episode,
    step and field, the tensors are raw little-endian bytes, each at an offset a multiple of 8.
    """
    if sys.byteorder != "little":
        raise LearnerDataError("the loader's workers send little-endian bytes; this machine is not")
    chunks: list[bytes] = []
    size = 0

    def tensor(value: torch.Tensor) -> dict[str, Any]:
        nonlocal size
        name = str(value.dtype).removeprefix("torch.")
        if name not in _DTYPES:
            raise LearnerDataError(f"a worker cannot send a {name} tensor")
        flat = value.detach().cpu().contiguous()
        data = ctypes.string_at(flat.data_ptr(), flat.numel() * flat.element_size())
        spec = {"t": name, "shape": list(flat.shape), "at": size, "n": len(data)}
        pad = -len(data) % _ALIGN
        chunks.append(data + bytes(pad))
        size += len(data) + pad
        return spec

    steps = []
    for (episode, role), items in piece.collected.steps.items():
        coded = []
        for step in items:
            fields = {
                f.name: tensor(v)
                if isinstance(v := getattr(step.query, f.name), torch.Tensor)
                else v
                for f in dataclasses.fields(step.query)
            }
            coded.append([step.chosen, fields])
        steps.append([episode, role, coded])
    header = json.dumps(
        {
            "order": piece.collected.order,
            "scores": piece.collected.scores,
            "steps": steps,
            "seen": sorted(piece.seen),
            "classes": sorted(piece.classes),
            "contract_error": piece.contract_error,
            "encode_error": piece.encode_error,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return len(header).to_bytes(8, "little") + header + b"".join(chunks)


def unpack_piece(blob: bytes) -> _Piece:
    """The piece ``pack_piece`` made; its tensors equal the sent ones bit for bit."""
    length = int.from_bytes(blob[:8], "little")
    header = json.loads(blob[8 : 8 + length])
    base = memoryview(bytearray(blob[8 + length :]))

    def tensor(spec: dict[str, Any]) -> torch.Tensor:
        if spec["n"] == 0:
            return torch.empty(spec["shape"], dtype=_DTYPES[spec["t"]])
        found = torch.frombuffer(
            base[spec["at"] : spec["at"] + spec["n"]], dtype=_DTYPES[spec["t"]]
        )
        return found.reshape(spec["shape"])

    steps: dict[tuple[str, str], list[Step]] = {}
    for episode, role, coded in header["steps"]:
        steps[(episode, role)] = [
            Step(
                EncodedQuery(
                    **{k: tensor(v) if isinstance(v, dict) else v for k, v in fields.items()}
                ),
                chosen,
            )
            for chosen, fields in coded
        ]
    return _Piece(
        Collected(header["order"], header["scores"], steps),
        set(header["seen"]),
        set(header["classes"]),
        header["contract_error"],
        None if header["encode_error"] is None else tuple(header["encode_error"]),
    )


def _raised(kind: str, message: str) -> Exception:
    """The exception a worker met, as its own class when that class is ours and says the same."""
    module, _, name = kind.rpartition(".")
    if module.startswith("areneon_ml.") or module == "builtins":
        try:
            found = getattr(importlib.import_module(module), name)
            if isinstance(found, type) and issubclass(found, Exception):
                error: Exception = found(message)
                if str(error) == message:
                    return error
        except Exception:  # an odd class or constructor: reported by message below
            pass
    return LearnerDataError(message)


def parallel_trajectories(
    path: Path, manifest: dict[str, Any], encoders: dict[str, Encoder], processes: int
) -> list[Trajectory] | None:
    """The trajectories of a stream file read by ``processes`` workers; None if it cannot be cut.

    None (nothing checked or read) means "read it yourself": no ``fork`` here, or too few pieces.
    """
    if "fork" not in multiprocessing.get_all_start_methods():
        return None
    workers = min(processes, os.cpu_count() or 1)
    points = stream_split_points(path, workers * PIECES_PER_WORKER)
    points = _coarsen(points)
    if workers < MIN_WORKERS or len(points) < MIN_POINTS:
        return None
    try:
        marks = check_stream_digest(path, manifest, points[:-1])
    except TrainingStreamError as exc:
        raise LearnerDataError(str(exc)) from exc
    pieces = [(points[i], points[i + 1], marks[i] + 1) for i in range(len(points) - 1)]
    _SHARED.update(manifest=manifest, encoders=encoders, path=str(path))
    try:
        context = multiprocessing.get_context("fork")
        # An executor, not a Pool: a worker the operating system kills (the out-of-memory
        # killer) raises BrokenProcessPool here, where a Pool would wait for it forever.
        with ProcessPoolExecutor(min(workers, len(pieces)), mp_context=context) as pool:
            results = [unpack_piece(data) for data in pool.map(_run_piece, pieces)]
    except BrokenProcessPool as exc:
        raise LearnerDataError("a load worker died before it finished its piece") from exc
    finally:
        _SHARED.clear()
    return _join(results, manifest)


def _coarsen(points: list[int]) -> list[int]:
    """Drop cut points that would leave a piece under ``MIN_PIECE_BYTES``."""
    kept = [points[0]]
    for point in points[1:-1]:
        if point - kept[-1] >= MIN_PIECE_BYTES and points[-1] - point >= MIN_PIECE_BYTES:
            kept.append(point)
    kept.append(points[-1])
    return kept


def _join(results: list[_Piece], manifest: dict[str, Any]) -> list[Trajectory] | None:
    """Put the pieces back together the way one pass over the whole stream would have."""
    for piece in results:
        if piece.contract_error is not None:
            raise LearnerDataError(piece.contract_error)
    whole = StreamChecker(manifest)
    for piece in results:
        whole.seen |= piece.seen
        whole.classes |= piece.classes
    try:
        whole.finish()
    except TrainingStreamError as exc:
        raise LearnerDataError(str(exc)) from exc
    joined = results[0].collected
    for piece in results[1:]:
        if not joined.absorb(piece.collected):
            return None  # an Episode in two pieces: the stream does not list its Episodes together
    for piece in results:
        if piece.encode_error is not None:
            raise _raised(*piece.encode_error)
    return joined.trajectories()
