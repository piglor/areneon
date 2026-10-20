"""The policy worker's session: handshake, replies, failure rules and isolation (ADR-010 §2)."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import runpy
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import areneon_ml.worker
import pytest
from areneon_contracts.contracts import jcs
from areneon_ml.worker.__main__ import EXIT_CHECKPOINT, main
from areneon_ml.worker.checkpoints import CheckpointError, CheckpointStore
from areneon_ml.worker.protocol import QueryError, check
from areneon_ml.worker.sampler import ScoreError
from areneon_ml.worker.serve import EXIT_FRAME, EXIT_OK, EXIT_REFUSED, Worker, read_query
from areneon_ml.worker.view import encoder_input

from tests.ml.helpers import (
    ENGINE_DATA,
    ENGINE_DATA_HASH,
    MAX_LINE,
    battle_query,
    hello,
    line,
    make_run_dir,
    make_worker,
    preview_query,
    sample,
    session,
)

REPLIES = 2  # the ready message and one reply


def test_a_session_answers_each_query_with_one_of_its_candidates(tmp_path: Path) -> None:
    queries = [battle_query(query_id=1), preview_query(query_id=2), battle_query(query_id=3)]
    code, frames = session(make_worker(tmp_path), line(hello()), *(line(q) for q in queries))
    assert code == EXIT_OK
    ready, *replies = frames
    assert ready["message"]["kind"] == "ready"
    check("policy-handshake", ready)
    assert [r["query_id"] for r in replies] == [1, 2, 3]
    for query, reply in zip(queries, replies, strict=True):
        check("policy-reply", reply)
        recommendation = reply["reply"]["recommendation"]
        assert recommendation["identity"] == query["identity"]
        assert recommendation["source_policy_id"] == query["policy_id"]
        assert recommendation["choice"] in [c["choice"] for c in query["candidates"]]


def test_the_ready_message_names_the_worker_it_is(tmp_path: Path) -> None:
    _, [ready] = session(make_worker(tmp_path), line(hello()))
    message = ready["message"]
    assert message["contracts"] == hello()["message"]["contracts"]
    assert message["encoder_version"] == "stub-1"
    assert message["engine_data_hash"] == ENGINE_DATA_HASH
    assert [c["role"] for c in message["checkpoints"]] == ["battle", "preview"]


def test_a_handshake_mismatch_ends_the_session_before_any_query(tmp_path: Path) -> None:
    contracts = hello()["message"]["contracts"]
    changed = [{**contracts[0], "schema_sha256": "0" * 64}, *contracts[1:]]
    code, frames = session(
        make_worker(tmp_path), line(hello(contracts=changed)), line(battle_query())
    )
    assert code == EXIT_REFUSED
    # It still says what it speaks, so the client can name the difference.
    assert len(frames) == 1
    assert frames[0]["message"]["contracts"] == contracts


@pytest.mark.parametrize(
    "frames",
    [
        [b""],
        [b"{not json}\n"],
        [line(battle_query())],
        [line({**sample("policy-handshake/valid/ready.json")})],
        [line({"schema_version": 1, "message": {"kind": "hello"}})],
    ],
)
def test_a_bad_first_frame_ends_the_session(tmp_path: Path, frames: list[bytes]) -> None:
    code, replies = session(make_worker(tmp_path), *frames)
    assert code == EXIT_FRAME
    assert replies == []


@pytest.mark.parametrize(
    "bad",
    [
        b"{not json}\n",
        b"[]\n",
        b'{"a": 1}\n',
        b"x" * (MAX_LINE + 2) + b"\n",
    ],
)
def test_a_malformed_frame_after_the_hello_ends_the_session(tmp_path: Path, bad: bytes) -> None:
    code, frames = session(make_worker(tmp_path), line(hello()), bad, line(battle_query()))
    assert code == EXIT_FRAME
    assert len(frames) == 1  # the ready message; no query was answered


def test_a_canonical_frame_that_is_not_a_query_is_refused_and_the_session_goes_on(
    tmp_path: Path,
) -> None:
    code, frames = session(
        make_worker(tmp_path), line(hello()), line({"query_id": 1}), line(battle_query(query_id=2))
    )
    assert code == EXIT_OK
    assert [f["reply"]["kind"] for f in frames[1:]] == ["error", "recommendation"]
    assert frames[1]["reply"]["error"] == "invalid_query"


def test_a_query_without_a_usable_id_cannot_be_answered(tmp_path: Path) -> None:
    broken = {k: v for k, v in battle_query().items() if k != "query_id"}
    code, frames = session(make_worker(tmp_path), line(hello()), line(broken))
    assert code == EXIT_FRAME
    assert len(frames) == 1


def test_a_query_whose_identity_differs_from_its_observations_is_refused(tmp_path: Path) -> None:
    query = sample("policy-query/reader-invalid/identity-differs.json")
    _, [_, reply] = session(make_worker(tmp_path), line(hello()), line(query))
    assert reply["reply"]["kind"] == "error"
    assert reply["reply"]["error"] == "invalid_query"
    assert "identity differs" in reply["reply"]["detail"]


def test_a_duplicate_candidate_is_refused(tmp_path: Path) -> None:
    query = sample("policy-query/reader-invalid/duplicate-candidate.json")
    _, [_, reply] = session(make_worker(tmp_path), line(hello()), line(query))
    assert reply["reply"]["error"] == "invalid_query"


@pytest.mark.parametrize("name", ["candidate-verdict", "seed", "no-candidates", "unknown-sampling"])
def test_a_query_that_breaks_its_contract_is_refused_without_echoing_its_content(
    tmp_path: Path, name: str
) -> None:
    query = sample(f"policy-query/invalid/{name}.json")
    _, [_, reply] = session(make_worker(tmp_path), line(hello()), line(query))
    assert reply["reply"]["error"] == "invalid_query"
    assert reply["reply"]["detail"].startswith("not a policy-query")
    assert "sodium" not in reply["reply"]["detail"]


def test_a_candidate_of_an_unknown_shape_is_unsupported_not_invalid(tmp_path: Path) -> None:
    query = battle_query()
    query["candidates"] = [*query["candidates"], {"choice": "shift, pass", "pending": False}]
    _, [_, reply] = session(make_worker(tmp_path), line(hello()), line(query))
    assert reply["reply"]["kind"] == "error"
    assert reply["reply"]["error"] == "unsupported_candidate"
    check("policy-reply", reply)


def test_read_query_applies_the_reader_rules() -> None:
    assert read_query(battle_query())["query_id"] == 1
    with pytest.raises(QueryError):
        read_query(sample("policy-query/reader-invalid/identity-differs.json"))


def recommendation_of(frames: list[dict[str, Any]], index: int) -> dict[str, Any]:
    recommendation: dict[str, Any] = frames[index]["reply"]["recommendation"]
    return {k: v for k, v in recommendation.items() if k != "elapsed_ms"}


def test_a_query_answers_the_same_alone_and_after_others(tmp_path: Path) -> None:
    # Batch invariance (ADR-010 §2): evaluation scores one query per pass, so what
    # was scored before cannot change the answer.
    worker = make_worker(tmp_path)
    target = battle_query(query_id=9, sampling={"kind": "sample", "u": 2**52})
    others = [preview_query(query_id=1), battle_query(query_id=2), preview_query(query_id=3)]
    _, alone = session(worker, line(hello()), line(target))
    _, after = session(worker, line(hello()), *(line(q) for q in others), line(target))
    assert recommendation_of(alone, 1) == recommendation_of(after, 4)
    scorer = worker._scorer
    first = scorer.probabilities(encoder_input(target))
    scorer.probabilities(encoder_input(others[0]))
    assert scorer.probabilities(encoder_input(target)) == first


def test_the_draw_follows_u_and_argmax_does_not(tmp_path: Path) -> None:
    worker = make_worker(tmp_path)
    queries = [
        battle_query(query_id=i + 1, sampling={"kind": "sample", "u": u})
        for i, u in enumerate([0, 2**52, 2**53 - 1])
    ]
    queries.append(battle_query(query_id=4, sampling={"kind": "argmax"}))
    _, frames = session(worker, line(hello()), *(line(q) for q in queries))
    choices = [f["reply"]["recommendation"]["choice"] for f in frames[1:]]
    candidates = [c["choice"] for c in battle_query()["candidates"]]
    assert (
        candidates.index(choices[0]) <= candidates.index(choices[1]) <= candidates.index(choices[2])
    )
    assert choices[0] != choices[2]
    probabilities = worker._scorer.probabilities(encoder_input(queries[0]))
    assert choices[3] == candidates[probabilities.index(max(probabilities))]


def test_the_a1_and_a2_forms_get_the_same_probabilities_and_choice(tmp_path: Path) -> None:
    a1 = battle_query(query_id=1, sampling={"kind": "argmax"})
    a2 = sample("policy-query/valid/battle-a2.json")
    _, frames = session(make_worker(tmp_path), line(hello()), line(a1), line(a2))
    assert recommendation_of(frames, 1)["choice"] == recommendation_of(frames, 2)["choice"]


class BrokenScorer:
    encoder_version = "broken-1"

    def __init__(self, result: list[float]) -> None:
        self._result = result

    def probabilities(self, view: dict[str, Any]) -> list[float]:
        return self._result


def test_a_scorer_that_returns_unusable_probabilities_crashes_the_worker(tmp_path: Path) -> None:
    worker = Worker(BrokenScorer([1.0]), ENGINE_DATA_HASH, [], MAX_LINE)
    with pytest.raises(ScoreError):
        session(worker, line(hello()), line(battle_query()))


def test_checkpoints_load_only_by_id_from_the_run_directory(tmp_path: Path) -> None:
    run_dir, preview, _ = make_run_dir(tmp_path)
    store = CheckpointStore(run_dir)
    assert hashlib.sha256(store.load(preview)).hexdigest() == preview
    for bad in ["../../etc/passwd", "ABC", preview.upper(), "0" * 63, "", f"{preview}/.."]:
        with pytest.raises(CheckpointError, match="64 lowercase"):
            store.load(bad)
    with pytest.raises(CheckpointError, match="cannot be read"):
        store.load("0" * 64)
    (run_dir / "checkpoints" / f"{'1' * 64}.safetensors").write_bytes(b"other bytes")
    with pytest.raises(CheckpointError, match="hash to another id"):
        store.load("1" * 64)


def test_the_worker_package_opens_no_socket() -> None:
    banned = {
        "socket",
        "socketserver",
        "http",
        "urllib",
        "ssl",
        "asyncio",
        "requests",
        "websockets",
        "ftplib",
        "smtplib",
    }
    package = Path(areneon_ml.worker.__file__).parent
    for source in package.glob("*.py"):
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            assert not {n.split(".")[0] for n in names} & banned, source.name


def test_a_session_runs_with_sockets_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the worker opened a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    code, frames = session(make_worker(tmp_path), line(hello()), line(battle_query()))
    assert code == EXIT_OK
    assert len(frames) == REPLIES


def cli(run_dir: Path, preview: str, battle: str, *extra: str) -> list[str]:
    return [
        "--run-dir", str(run_dir), "--preview", preview, "--battle", battle,
        "--engine-data", str(ENGINE_DATA), "--max-line-bytes", str(MAX_LINE), *extra,
    ]  # fmt: skip


class Stdio:
    def __init__(self, data: bytes) -> None:
        self.buffer = io.BytesIO(data)


def test_main_serves_a_session_on_stdio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir, preview, battle = make_run_dir(tmp_path)
    out = Stdio(b"")
    monkeypatch.setattr(sys, "stdin", Stdio(line(hello()) + line(battle_query())))
    monkeypatch.setattr(sys, "stdout", out)
    assert main(cli(run_dir, preview, battle)) == EXIT_OK
    assert len(out.buffer.getvalue().splitlines()) == REPLIES


def test_main_refuses_a_missing_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir, preview, _ = make_run_dir(tmp_path)
    assert main(cli(run_dir, preview, "9" * 64)) == EXIT_CHECKPOINT
    assert "cannot be read" in capsys.readouterr().err


def test_main_refuses_a_non_positive_line_limit(tmp_path: Path) -> None:
    run_dir, preview, battle = make_run_dir(tmp_path)
    args = cli(run_dir, preview, battle)
    args[args.index("--max-line-bytes") + 1] = "0"
    with pytest.raises(SystemExit):
        main(args)


def test_module_entry_point_serves_a_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, preview, battle = make_run_dir(tmp_path)
    monkeypatch.delitem(sys.modules, "areneon_ml.worker.__main__", raising=False)
    monkeypatch.setattr(sys, "argv", ["areneon_ml.worker", *cli(run_dir, preview, battle)])
    monkeypatch.setattr(sys, "stdin", Stdio(line(hello())))
    monkeypatch.setattr(sys, "stdout", Stdio(b""))
    with pytest.raises(SystemExit) as exited:
        runpy.run_module("areneon_ml.worker", run_name="__main__")
    assert exited.value.code == EXIT_OK


def test_the_worker_process_speaks_the_protocol_over_real_pipes(tmp_path: Path) -> None:
    run_dir, preview, battle = make_run_dir(tmp_path)
    query = battle_query(sampling={"kind": "argmax"})
    done = subprocess.run(
        [sys.executable, "-I", "-m", "areneon_ml.worker", *cli(run_dir, preview, battle)],
        input=line(hello()) + line(query),
        capture_output=True,
        timeout=60,
        check=False,
        env={"PYTHONPATH": str(Path(areneon_ml.__file__).parent.parent)},
    )
    assert done.returncode == EXIT_OK, done.stderr
    assert [jcs(json.loads(x)) for x in done.stdout.splitlines()] == done.stdout.splitlines()
    assert len(done.stdout.splitlines()) == REPLIES
