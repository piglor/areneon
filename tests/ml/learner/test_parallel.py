"""Check and encode a stream in worker processes (#1069): the serial result, tensor for tensor."""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import random
import signal
from pathlib import Path
from typing import Any

import pytest
import torch
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_contracts.training import stream_split_points
from areneon_ml.learner import parallel
from areneon_ml.learner.data import (
    Collected,
    LearnerDataError,
    Step,
    Trajectory,
    _Collector,
    trajectories,
)
from areneon_ml.learner.parallel import _check_piece
from areneon_ml.model.encoder import Encoder
from areneon_ml.model.network import ROLES
from areneon_ml.reducer.errors import UnknownLineError

from tests.ml.learner.support import POLICY, episode_lines, stream_of, tiny_config
from tests.ml.model.support import game

EPISODES = [(i, p) for i in range(6) for p in ("p1", "p2")]


def encoders() -> dict[str, Encoder]:
    config = tiny_config()
    return {role: Encoder(game(), config.model.encoder) for role in ROLES}


def examples_of(seed: int = 0, episodes: list[tuple[int, str]] | None = None) -> list[Any]:
    picker = random.Random(seed)
    out: list[Any] = []
    for index, player in episodes or EPISODES:
        out += episode_lines(index, player, picker)
    return out


def write(tmp_path: Path, text: str, manifest: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = tmp_path / "stream.jsonl"
    path.write_text(text, encoding="utf-8")
    return path, manifest


def same(a: list[Trajectory], b: list[Trajectory]) -> None:
    assert [(t.role, t.score, t.episode, len(t.steps)) for t in a] == [
        (t.role, t.score, t.episode, len(t.steps)) for t in b
    ]
    for left, right in zip(a, b, strict=True):
        for x, y in zip(left.steps, right.steps, strict=True):
            assert x.chosen == y.chosen
            assert x.query.kind == y.query.kind
            assert x.query.n_own == y.query.n_own
            for name, tensor in x.query.tensors().items():
                other = y.query.tensors()[name]
                assert tensor.dtype == other.dtype
                assert torch.equal(tensor, other), name


@pytest.fixture(autouse=True)
def small_pieces(monkeypatch: pytest.MonkeyPatch) -> None:
    # The fixtures are small: cut them in many pieces so the workers have something to share.
    monkeypatch.setattr(parallel, "MIN_PIECE_BYTES", 1)


def test_the_workers_give_the_serial_trajectories_tensor_for_tensor(tmp_path: Path) -> None:
    text, manifest = stream_of(examples_of())
    path, manifest = write(tmp_path, text, manifest)
    serial = trajectories(path, manifest, encoders(), POLICY)
    assert len(serial) == 2 * len(EPISODES)
    for processes in (2, 3, 8):
        same(serial, trajectories(path, manifest, encoders(), POLICY, processes=processes))
    # ... and the string form is still served by one process.
    same(serial, trajectories(text, manifest, encoders(), POLICY, processes=4))


def test_the_pieces_are_cut_between_episodes_and_cover_the_file(tmp_path: Path) -> None:
    text, manifest = stream_of(examples_of())
    path, _ = write(tmp_path, text, manifest)
    points = stream_split_points(path, 8)
    assert points[0] == 0
    assert points[-1] == len(text.encode())
    assert points == sorted(set(points))
    assert len(points) > 3
    data = path.read_bytes()
    for point in points[1:-1]:
        before = data[:point].splitlines()[-1]
        after = data[point:].splitlines()[0]
        assert before.split(b'"episode_hash":"')[1][:64] != after.split(b'"episode_hash":"')[1][:64]
        assert data[point - 1 : point] == b"\n"


def test_a_stream_whose_episodes_are_not_together_is_read_by_one_process(tmp_path: Path) -> None:
    examples = examples_of(episodes=[(0, "p1"), (1, "p1"), (2, "p1")])
    # Interleave two Episodes' examples: A1 B1 A2 B2 ... (the manifest lists both once).
    by_episode: dict[str, list[Any]] = {}
    for example in examples:
        by_episode.setdefault(example["episode_hash"], []).append(example)
    groups = list(by_episode.values())
    mixed = [e for pair in zip(groups[0], groups[1], strict=False) for e in pair] + groups[2]
    mixed += groups[0][len(groups[1]) :] + groups[1][len(groups[0]) :]
    text = "".join(jcs(e).decode() + "\n" for e in mixed)
    manifest = {
        **stream_of(examples)[1],
        "examples": len(mixed),
        "stream_sha256": sha256_hex(text.encode()),
    }
    path, manifest = write(tmp_path, text, manifest)
    serial = trajectories(path, manifest, encoders(), POLICY)
    same(serial, trajectories(path, manifest, encoders(), POLICY, processes=3))


def refused(path: Path, manifest: dict[str, Any], processes: int) -> str:
    with pytest.raises(Exception) as caught:  # noqa: PT011 - the type is part of what is compared
        trajectories(path, manifest, encoders(), POLICY, processes=processes)
    return f"{type(caught.value).__name__}: {caught.value}"


def rewrite(
    tmp_path: Path, examples: list[Any], change: Any, manifest_change: Any = None
) -> tuple[Path, dict[str, Any]]:
    text, manifest = stream_of(examples)
    lines = text.splitlines()
    change(lines)
    text = "".join(f"{line}\n" for line in lines)
    manifest = {**manifest, "examples": len(lines), "stream_sha256": sha256_hex(text.encode())}
    if manifest_change:
        manifest_change(manifest)
    return write(tmp_path, text, manifest)


def damaged(line: str, edit: Any) -> str:
    value = json.loads(line)
    edit(value)
    return jcs(value).decode()


def test_every_refusal_is_the_serial_one(tmp_path: Path) -> None:
    examples = examples_of()
    cases: dict[str, Any] = {}
    # A contract error in the middle of the stream.
    cases["contract"] = lambda lines: lines.__setitem__(
        len(lines) // 2, damaged(lines[len(lines) // 2], lambda v: v.pop("choice"))
    )
    # A choice that is not among the candidates (the cross-field rule).
    cases["choice"] = lambda lines: lines.__setitem__(
        len(lines) // 3, damaged(lines[len(lines) // 3], lambda v: v.update(choice="move 9"))
    )
    # An Episode the manifest does not list.
    cases["unlisted"] = lambda lines: lines.__setitem__(
        7, damaged(lines[7], lambda v: v.update(episode_hash="f" * 64))
    )

    # An encoder refusal early and a contract error later: the contract error wins.
    def both(lines: list[str]) -> None:
        lines[2] = damaged(lines[2], lambda v: v["observation"]["log"].insert(0, "|nonsense|x"))
        lines[-3] = damaged(lines[-3], lambda v: v.pop("outcome"))

    cases["both"] = both
    # Only an encoder refusal: an unknown line type in a log.
    cases["encoder"] = lambda lines: lines.__setitem__(
        4, damaged(lines[4], lambda v: v["observation"]["log"].insert(0, "|nonsense|x"))
    )
    # Two outcomes for one Episode.
    cases["outcomes"] = lambda lines: lines.__setitem__(
        5,
        damaged(lines[5], lambda v: v.update(outcome="loss" if v["outcome"] != "loss" else "win")),
    )
    for name, change in cases.items():
        path, manifest = rewrite(tmp_path, copy.deepcopy(examples), change)
        assert refused(path, manifest, 3) == refused(path, manifest, 1), name


def test_the_file_level_refusals_come_first_and_are_the_serial_ones(tmp_path: Path) -> None:
    examples = examples_of()
    text, manifest = stream_of(examples)
    path, _ = write(tmp_path, text, manifest)
    bad_hash = {**manifest, "stream_sha256": "0" * 64}
    assert refused(path, bad_hash, 3) == refused(path, bad_hash, 1)
    bad_count = {**manifest, "examples": manifest["examples"] + 1}
    assert refused(path, bad_count, 3) == refused(path, bad_count, 1)
    path.write_text(text.rstrip("\n"))
    assert refused(path, {**manifest, "stream_sha256": sha256_hex(text.rstrip("\n").encode())}, 3)
    # Data that is not UTF-8, in the middle of a line.
    raw = bytearray(text.encode())
    raw[len(raw) // 2] = 0xFF
    path.write_bytes(bytes(raw))
    broken = {**manifest, "stream_sha256": sha256_hex(bytes(raw))}
    assert refused(path, broken, 3) == refused(path, broken, 1)
    # A stream that misses a listed Episode, and one with the wrong data class.
    gone = examples[0]["episode_hash"]
    fewer = rewrite(
        tmp_path,
        examples,
        lambda lines: lines.__setitem__(
            slice(None), [line for line in lines if gone not in line[:100_000]]
        ),
    )
    assert refused(*fewer, 3) == refused(*fewer, 1)
    private = rewrite(
        tmp_path, examples, lambda lines: None, lambda m: m.update(data_class="private")
    )
    assert refused(*private, 3) == refused(*private, 1)


def test_an_exception_a_worker_cannot_send_back_is_reported_by_its_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Odd(Exception):
        def __init__(self, a: int, b: int) -> None:  # cannot be rebuilt from its args
            super().__init__(f"odd {a} {b}")

    def broken(self: Any, example: Any) -> None:
        raise Odd(1, 2)

    monkeypatch.setattr("areneon_ml.learner.data._Collector.add", broken)
    text, manifest = stream_of(examples_of())
    path, manifest = write(tmp_path, text, manifest)
    with pytest.raises(LearnerDataError, match="odd 1 2"):
        trajectories(path, manifest, encoders(), POLICY, processes=3)


def test_a_stream_too_small_to_cut_is_read_by_one_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(parallel, "MIN_PIECE_BYTES", 1 << 40)
    text, manifest = stream_of(examples_of())
    path, manifest = write(tmp_path, text, manifest)
    assert parallel.parallel_trajectories(path, manifest, encoders(), 4) is None
    same(
        trajectories(path, manifest, encoders(), POLICY),
        trajectories(path, manifest, encoders(), POLICY, processes=4),
    )
    monkeypatch.setattr("multiprocessing.get_all_start_methods", lambda: ["spawn"])
    monkeypatch.setattr(parallel, "MIN_PIECE_BYTES", 1)
    assert parallel.parallel_trajectories(path, manifest, encoders(), 4) is None


def test_a_piece_is_checked_and_encoded_in_process(tmp_path: Path) -> None:
    text, manifest = stream_of(examples_of(episodes=[(0, "p1")]))
    path, manifest = write(tmp_path, text, manifest)
    parallel._SHARED.update(manifest=manifest, encoders=encoders(), path=str(path))
    try:
        piece = _check_piece((0, len(text.encode()), 1))
    finally:
        parallel._SHARED.clear()
    assert piece.contract_error is None
    assert piece.encode_error is None
    assert len(piece.collected.order) == 1
    assert piece.seen == set(manifest["episode_hashes"])


def test_a_piece_travels_as_one_blob_and_comes_back_bit_for_bit(tmp_path: Path) -> None:
    text, manifest = stream_of(examples_of(episodes=[(0, "p1"), (0, "p2")]))
    path, manifest = write(tmp_path, text, manifest)
    parallel._SHARED.update(manifest=manifest, encoders=encoders(), path=str(path))
    try:
        sent = parallel._check_piece((0, len(text.encode()), 1))
        blob = parallel._run_piece((0, len(text.encode()), 1))
    finally:
        parallel._SHARED.clear()
    assert isinstance(blob, bytes)
    got = parallel.unpack_piece(blob)
    assert got.seen == sent.seen
    assert got.classes == sent.classes
    assert (got.contract_error, got.encode_error) == (None, None)
    same(sent.collected.trajectories(), got.collected.trajectories())
    # Errors travel too, and an empty tensor keeps its shape.
    odd = parallel._Piece(Collected([], {}, {}), set(), set(), "line 3: broken", ("a.B", "m"))
    again = parallel.unpack_piece(parallel.pack_piece(odd))
    assert (again.contract_error, again.encode_error) == ("line 3: broken", ("a.B", "m"))


def test_a_worker_refuses_to_send_a_tensor_of_a_kind_the_loader_does_not_use() -> None:
    step = next(iter(_one_piece().collected.steps.values()))[0]
    bad = Step(dataclasses.replace(step.query, pending=step.query.pending.double()), 0)
    piece = parallel._Piece(Collected(["e"], {"e": 0.0}, {("e", "battle"): [bad]}), set(), set())
    with pytest.raises(LearnerDataError, match="float64"):
        parallel.pack_piece(piece)


def _one_piece() -> Any:
    examples = examples_of(episodes=[(0, "p1")])
    collector = _Collector(encoders())
    for example in examples:
        collector.add(json.loads(jcs(example)))
    return parallel._Piece(collector.collected(), set(), set())


def test_an_exception_comes_back_as_its_own_class_only_when_it_is_ours_and_unchanged() -> None:
    mine = parallel._raised("areneon_ml.reducer.errors.UnknownLineError", "no rule")
    assert isinstance(mine, UnknownLineError)
    assert str(mine) == "no rule"
    assert type(parallel._raised("builtins.ValueError", "k")) is ValueError
    for kind in (
        "os.error",
        "builtins.int",
        "areneon_ml.reducer.errors.Nothing",
        "a.B",
        "areneon_ml.x.Y",
    ):
        assert type(parallel._raised(kind, "m")) is LearnerDataError


def test_the_memory_cap_leaves_at_least_one_worker_and_never_more_than_asked() -> None:
    assert parallel.fit_processes(6, None) == 6
    assert parallel.fit_processes(6, 10 * parallel.WORKER_MB) == 6
    assert parallel.fit_processes(6, 2 * parallel.WORKER_MB + 1) == 2
    assert parallel.fit_processes(6, 1) == 1


def killed(piece: tuple[int, int, int]) -> bytes:
    os.kill(os.getpid(), signal.SIGKILL)
    return b""


def test_a_worker_the_system_kills_is_a_refusal_not_a_hang(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(parallel, "_run_piece", killed)
    text, manifest = stream_of(examples_of())
    path, manifest = write(tmp_path, text, manifest)
    with pytest.raises(LearnerDataError, match="died"):
        trajectories(path, manifest, encoders(), POLICY, processes=3)
