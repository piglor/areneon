"""The worker's session: handshake, then one reply per query (ADR-010 §2).

Stdio only. The first frame must be the client's hello; the worker answers
with its own ``ready`` and, if the two differ in the contracts they speak,
ends the session (the client refuses it too). Then each frame is a
PolicyQuery, answered in order, one at a time: this skeleton scores one query
per pass, which is also what evaluation and sealed runs require.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast

from areneon_contracts.contracts import jcs
from areneon_contracts.opponent_sheet import sheet_problem

from areneon_ml.model.encoder import EncodingError
from areneon_ml.reducer import ReducerError
from areneon_ml.worker.candidates import UnsupportedCandidate, parse_choice
from areneon_ml.worker.features import coverage_problem, rows_problem
from areneon_ml.worker.protocol import (
    FrameError,
    QueryError,
    check,
    contract_versions,
    encode_frame,
    read_frame,
)
from areneon_ml.worker.sampler import FULL_TEMPERATURE_PPM, check_probabilities, select, to_ppm
from areneon_ml.worker.scorer import Scorer, ValueScorer
from areneon_ml.worker.view import encoder_input

#: The batching a worker states when it answers several queued queries per forward pass
#: (``generation.inference``, ADR-028 section 2 F2): ``{kind, max_batch}``; absent is batch 1.
BATCHED = "batched"

#: The smallest batched forward pass; batch 1 is stated by omission.
MIN_BATCH = 2

EXIT_OK = 0
EXIT_FRAME = 2
EXIT_REFUSED = 3


class SessionRefused(RuntimeError):
    """The client's hello does not match what this worker speaks."""


@dataclass(frozen=True)
class FeatureSupport:
    """The candidate feature set a worker states and the means to judge a query's rows (ADR-022 §3).

    ``feature_set`` is ``{feature_set, feature_set_hash}``; ``damaging`` tells a damaging move from
    a status move by its id (``features.damaging_moves`` builds it from the engine-data table).
    """

    feature_set: dict[str, str]
    damaging: Callable[[str], bool]


@dataclass(frozen=True)
class SheetSupport:
    """The opponent sheet source a worker states (ADR-027 slice 4, ADR-025 erratum E4).

    ``source`` is ``{feature_set, feature_set_hash}``: the block's ``source.hash`` must equal the
    hash. ``diagnostic`` lets the worker read an ``oracle-diagnostic`` block, which a test-only
    diagnostic run may send; without it such a block is an ``invalid_query`` (an oracle block is
    never a policy input outside the labelled diagnostic, ADR-027 section 7).
    """

    source: dict[str, str]
    diagnostic: bool = False


def _check_sheet(frame: dict[str, Any], support: SheetSupport | None) -> None:
    """A query carries ``opponent_sheet`` exactly when the worker stated a source (of it)."""
    sheet = frame.get("opponent_sheet")
    if support is None:
        if sheet is not None:
            raise QueryError("the query carries opponent_sheet, but the worker stated no source")
        return
    if sheet is None:
        raise QueryError("the worker stated a sheet source, but the query has no opponent_sheet")
    if sheet["source"]["kind"] == "oracle-diagnostic" and not support.diagnostic:
        raise QueryError("an oracle-diagnostic opponent_sheet needs the diagnostic mode")
    if sheet["source"]["hash"] != support.source["feature_set_hash"]:
        raise QueryError("the query's opponent_sheet is of another source")
    problem = sheet_problem(sheet)
    if problem is not None:
        raise QueryError(f"opponent_sheet: {problem}")


def read_query(
    frame: dict[str, Any],
    features: FeatureSupport | None = None,
    sheet_support: SheetSupport | None = None,
    value_query: bool = False,
) -> dict[str, Any]:
    """The frame as a PolicyQuery: the schema, then the rules the profile cannot say.

    A query carries ``candidate_features`` exactly when the worker stated a feature set, of that
    set, with sorted and unique rows and a row for every damaging move part of every candidate.
    A ``value`` query (ADR-026 erratum E2 decision 7) is the exception: it has no candidates, no
    rows and no sheet, and is served only by a worker that stated ``value_query``; every other
    query has at least one candidate.
    """
    check("policy-query", frame)
    if frame["sampling"]["kind"] == "value":
        return _read_value_query(frame, value_query)
    if not frame["candidates"]:
        raise QueryError("only a value query may have no candidates")
    sent = frame.get("candidate_features")
    if features is None:
        if sent is not None:
            raise QueryError("the query carries candidate_features, but the worker stated no set")
    else:
        if sent is None:
            raise QueryError("the worker stated a set, but the query has no candidate_features")
        stated = features.feature_set
        if (sent["feature_set"], sent["feature_set_hash"]) != (
            stated["feature_set"],
            stated["feature_set_hash"],
        ):
            raise QueryError("the query's candidate_features are of another set")
        problem = rows_problem(sent) or coverage_problem(frame, sent, features.damaging)
        if problem is not None:
            raise QueryError(problem)
    _check_sheet(frame, sheet_support)
    if jcs(frame["identity"]) != jcs(frame["observation"]["identity"]):
        raise QueryError("the query's identity differs from its observation's identity")
    choices = [candidate["choice"] for candidate in frame["candidates"]]
    if len(set(choices)) != len(choices):
        raise QueryError("a candidate is listed twice")
    if frame["sampling"].get("temperature_ppm") == FULL_TEMPERATURE_PPM:
        raise QueryError("temperature_ppm 1000000 is the default and must be omitted")
    return frame


def _read_value_query(frame: dict[str, Any], value_query: bool) -> dict[str, Any]:
    """A ``value`` query: no candidates, rows or sheet, and a worker that stated ``value_query``."""
    if not value_query:
        raise QueryError("the worker states no value_query: its encoder reads candidate rows")
    if frame["candidates"]:
        raise QueryError("a value query carries no candidates")
    if frame.get("candidate_features") is not None or frame.get("opponent_sheet") is not None:
        raise QueryError("a value query carries no candidate_features and no opponent_sheet")
    if jcs(frame["identity"]) != jcs(frame["observation"]["identity"]):
        raise QueryError("the query's identity differs from its observation's identity")
    return frame


class Worker:
    def __init__(  # noqa: PLR0913, PLR0917
        self,
        scorer: Scorer,
        engine_data_hash: str,
        checkpoints: list[dict[str, str]],
        max_line_bytes: int,
        features: FeatureSupport | None = None,
        sheet: SheetSupport | None = None,
        value_scorer: ValueScorer | None = None,
        inference_max_batch: int | None = None,
    ) -> None:
        if inference_max_batch is not None and inference_max_batch < MIN_BATCH:
            raise ValueError("inference_max_batch is at least 2; batch 1 is stated by omission")
        self._inference_max_batch = inference_max_batch
        self._inference: dict[str, Any] | None = (
            None
            if inference_max_batch is None
            else {"kind": BATCHED, "max_batch": inference_max_batch}
        )
        self._value_scorer = value_scorer
        self._features = features
        self._sheet = sheet
        self._scorer = scorer
        #: Where ``__main__`` leaves this session's profile, when it was asked to (diagnostics).
        self.profile_dir: Path | None = None
        self._ready: dict[str, Any] = {
            "schema_version": 1,
            "message": {
                "kind": "ready",
                "contracts": contract_versions(),
                "encoder_version": scorer.encoder_version,
                "engine_data_hash": engine_data_hash,
                "checkpoints": sorted(checkpoints, key=lambda c: c["role"]),
            },
        }
        if features is not None:
            self._ready["message"]["feature_set"] = dict(features.feature_set)
        if sheet is not None:
            self._ready["message"]["opponent_sheet"] = dict(sheet.source)
        if value_scorer is not None:
            self._ready["message"]["value_query"] = True
        if self._inference is not None:
            self._ready["message"]["inference"] = dict(self._inference)
        self._max = max_line_bytes

    def handshake(self, stdin: BinaryIO, stdout: BinaryIO) -> None:
        """Read the hello, send ready; raise ``SessionRefused`` if the contracts differ."""
        frame = read_frame(stdin, self._max)
        if frame is None:
            raise FrameError("the input ended before the hello")
        check("policy-handshake", frame)
        message = frame["message"]
        if message["kind"] != "hello":
            raise FrameError("the first frame must be a hello")
        stdout.write(encode_frame(self._ready, self._max))
        stdout.flush()
        if message["contracts"] != contract_versions():
            raise SessionRefused("the client speaks other contract versions")
        if message.get("inference") != self._inference:
            raise SessionRefused("the client expects another batching than this worker states")

    def answer(self, frame: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR0911
        """The PolicyReply of one query frame."""
        started = time.perf_counter()
        query_id = frame.get("query_id")
        try:
            query = read_query(frame, self._features, self._sheet, self._value_scorer is not None)
            for candidate in query["candidates"]:
                parse_choice(candidate["choice"])
        except QueryError as exc:
            return self._error(query_id, "invalid_query", str(exc))
        except UnsupportedCandidate as exc:
            return self._error(query_id, "unsupported_candidate", str(exc))
        if query["sampling"]["kind"] == "value":
            return self._answer_value(query_id, query)
        evaluating = query["sampling"]["kind"] == "evaluate"
        value = 0.0
        try:
            temperature_ppm = query["sampling"].get("temperature_ppm")
            if evaluating:
                probabilities, value = self._scorer.evaluate(encoder_input(query))
            elif temperature_ppm is None:
                probabilities = self._scorer.probabilities(encoder_input(query))
            else:
                probabilities = self._scorer.probabilities(
                    encoder_input(query), temperature_ppm=temperature_ppm
                )
        except UnsupportedCandidate as exc:
            return self._error(query_id, "unsupported_candidate", str(exc))
        except ReducerError as exc:
            # The channel is the player's own, but the line is not echoed back.
            return self._error(query_id, "invalid_query", f"{type(exc).__name__}: {exc.message}")
        except EncodingError as exc:
            return self._error(query_id, "invalid_query", f"EncodingError: {exc}")
        check_probabilities(probabilities, len(query["candidates"]))
        return (
            self._evaluation(query, probabilities, value)
            if evaluating
            else self._recommendation(query, probabilities, started)
        )

    def answer_batch(self, frames: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """The PolicyReplies of several query frames, in their order; the forward pass batches.

        Each frame is read and refused on its own exactly as ``answer`` does; the queries that
        need the network go through ``Scorer.score_batch`` together (a scorer without it, a
        test stand-in, is asked one at a time). Only a run whose config says
        ``generation.inference`` asks for this.
        """
        started = time.perf_counter()
        replies: list[dict[str, Any] | None] = [None] * len(frames)
        scored: list[tuple[int, dict[str, Any], bool]] = []
        for index, frame in enumerate(frames):
            query_id = frame.get("query_id")
            try:
                query = read_query(
                    frame, self._features, self._sheet, self._value_scorer is not None
                )
                for candidate in query["candidates"]:
                    parse_choice(candidate["choice"])
            except QueryError as exc:
                replies[index] = self._error(query_id, "invalid_query", str(exc))
                continue
            except UnsupportedCandidate as exc:
                replies[index] = self._error(query_id, "unsupported_candidate", str(exc))
                continue
            if query["sampling"]["kind"] == "value":
                replies[index] = self._answer_value(query_id, query)
            else:
                scored.append((index, query, query["sampling"]["kind"] == "evaluate"))
        for (index, query, evaluating), found in zip(
            scored, self._score([query for _, query, _ in scored]), strict=True
        ):
            query_id = query["query_id"]
            if isinstance(found, UnsupportedCandidate):
                replies[index] = self._error(query_id, "unsupported_candidate", str(found))
            elif isinstance(found, ReducerError):
                replies[index] = self._error(
                    query_id, "invalid_query", f"{type(found).__name__}: {found.message}"
                )
            elif isinstance(found, EncodingError):
                replies[index] = self._error(query_id, "invalid_query", f"EncodingError: {found}")
            else:
                assert isinstance(found, tuple)  # every refusal was handled above
                probabilities, value = found
                check_probabilities(probabilities, len(query["candidates"]))
                replies[index] = (
                    self._evaluation(query, probabilities, value)
                    if evaluating
                    else self._recommendation(query, probabilities, started)
                )
        return [reply for reply in replies if reply is not None]

    def _score(
        self, queries: Sequence[dict[str, Any]]
    ) -> list[tuple[list[float], float] | Exception]:
        """The probabilities and value of each query, or the refusal of that query alone."""
        items = [
            (encoder_input(query), query["sampling"].get("temperature_ppm")) for query in queries
        ]
        batch = getattr(self._scorer, "score_batch", None)
        if batch is not None:
            return cast("list[tuple[list[float], float] | Exception]", batch(items))
        found: list[tuple[list[float], float] | Exception] = []
        for query, (view, temperature_ppm) in zip(queries, items, strict=True):
            try:
                if query["sampling"]["kind"] == "evaluate":
                    found.append(self._scorer.evaluate(view))
                elif temperature_ppm is None:
                    found.append((self._scorer.probabilities(view), 0.0))
                else:
                    found.append(
                        (self._scorer.probabilities(view, temperature_ppm=temperature_ppm), 0.0)
                    )
            except (UnsupportedCandidate, ReducerError, EncodingError) as exc:
                found.append(exc)
        return found

    def _answer_value(self, query_id: object, query: dict[str, Any]) -> dict[str, Any]:
        """The EvaluationReply of a ``value`` query: the value in ppm."""
        scorer = cast(ValueScorer, self._value_scorer)  # read_query refused a value query otherwise
        try:
            value = scorer.value(encoder_input(query))
        except ReducerError as exc:
            return self._error(query_id, "invalid_query", f"{type(exc).__name__}: {exc.message}")
        except EncodingError as exc:
            return self._error(query_id, "invalid_query", f"EncodingError: {exc}")
        reply = {
            "schema_version": 1,
            "query_id": query["query_id"],
            "reply": {
                "kind": "value",
                "value": {"identity": query["identity"], "value_ppm": to_ppm(value)},
            },
        }
        check("evaluation-reply", reply)
        return reply

    @staticmethod
    def _recommendation(
        query: dict[str, Any], probabilities: list[float], started: float
    ) -> dict[str, Any]:
        """The PolicyReply holding the candidate the query's sampling picks."""
        index = select(probabilities, query["sampling"])
        elapsed = int((time.perf_counter() - started) * 1000)
        reply = {
            "schema_version": 1,
            "query_id": query["query_id"],
            "reply": {
                "kind": "recommendation",
                "recommendation": {
                    "schema_version": 1,
                    "identity": query["identity"],
                    "choice": query["candidates"][index]["choice"],
                    "source_policy_id": query["policy_id"],
                    "elapsed_ms": elapsed,
                    "deadline_outcome": "in_budget",
                },
            },
        }
        check("policy-reply", reply)
        return reply

    @staticmethod
    def _evaluation(
        query: dict[str, Any], probabilities: list[float], value: float
    ) -> dict[str, Any]:
        """The EvaluationReply of an ``evaluate`` query: prior and value in ppm."""
        reply = {
            "schema_version": 1,
            "query_id": query["query_id"],
            "reply": {
                "kind": "evaluation",
                "evaluation": {
                    "identity": query["identity"],
                    "prior_ppm": [to_ppm(p) for p in probabilities],
                    "value_ppm": to_ppm(value),
                },
            },
        }
        check("evaluation-reply", reply)
        return reply

    @staticmethod
    def _error(query_id: object, error: str, detail: str) -> dict[str, Any]:
        if not isinstance(query_id, int) or isinstance(query_id, bool) or query_id < 1:
            raise FrameError("a query without a usable query_id cannot be answered")
        return {
            "schema_version": 1,
            "query_id": query_id,
            "reply": {"kind": "error", "error": error, "detail": detail[:1024]},
        }

    def serve(self, stdin: BinaryIO, stdout: BinaryIO) -> int:
        """Run one session; the process exit code."""
        try:
            self.handshake(stdin, stdout)
            if self._inference_max_batch is not None:
                return self._serve_batched(stdin, stdout, self._inference_max_batch)
            while (frame := read_frame(stdin, self._max)) is not None:
                stdout.write(encode_frame(self.answer(frame), self._max))
                stdout.flush()
        except SessionRefused:
            return EXIT_REFUSED
        except (FrameError, QueryError):
            return EXIT_FRAME
        return EXIT_OK

    def _serve_batched(self, stdin: BinaryIO, stdout: BinaryIO, max_batch: int) -> int:
        """The batched session (ADR-028 section 2 F2): answer what is queued, up to ``max_batch``.

        A reader thread turns the input into frames; the main thread takes the next frame
        (waiting for one), then every frame already queued up to ``max_batch``, answers them in
        one pass and writes the replies in their order. What is grouped depends on timing, so a
        batched session is not reproducible; a frame error is answered after the frames that
        arrived before it, as in the serial session.
        """
        frames: queue.Queue[dict[str, Any] | BaseException | None] = queue.Queue()

        def read() -> None:
            try:
                while (frame := read_frame(stdin, self._max)) is not None:
                    frames.put(frame)
                frames.put(None)
            except BaseException as exc:
                frames.put(exc)

        threading.Thread(target=read, daemon=True).start()
        ended: BaseException | None = None
        while ended is None:
            batch: list[dict[str, Any]] = []
            item = frames.get()
            while True:
                if isinstance(item, BaseException) or item is None:
                    ended = item if item is not None else StopIteration()
                    break
                batch.append(item)
                if len(batch) == max_batch:
                    break
                try:
                    item = frames.get_nowait()
                except queue.Empty:
                    break
            if batch:
                for reply in self.answer_batch(batch):
                    stdout.write(encode_frame(reply, self._max))
                stdout.flush()
        if isinstance(ended, StopIteration):
            return EXIT_OK
        if isinstance(ended, (FrameError, QueryError)):
            return EXIT_FRAME
        raise ended
