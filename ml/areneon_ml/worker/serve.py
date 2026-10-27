"""The worker's session: handshake, then one reply per query (ADR-010 §2).

Stdio only. The first frame must be the client's hello; the worker answers
with its own ``ready`` and, if the two differ in the contracts they speak,
ends the session (the client refuses it too). Then each frame is a
PolicyQuery, answered in order, one at a time: this skeleton scores one query
per pass, which is also what evaluation and sealed runs require.
"""

from __future__ import annotations

import time
from typing import Any, BinaryIO

from areneon_contracts.contracts import jcs

from areneon_ml.model.encoder import EncodingError
from areneon_ml.reducer import ReducerError
from areneon_ml.worker.candidates import UnsupportedCandidate, parse_choice
from areneon_ml.worker.protocol import (
    FrameError,
    QueryError,
    check,
    contract_versions,
    encode_frame,
    read_frame,
)
from areneon_ml.worker.sampler import FULL_TEMPERATURE_PPM, check_probabilities, select
from areneon_ml.worker.scorer import Scorer
from areneon_ml.worker.view import encoder_input

EXIT_OK = 0
EXIT_FRAME = 2
EXIT_REFUSED = 3


class SessionRefused(RuntimeError):
    """The client's hello does not match what this worker speaks."""


def read_query(frame: dict[str, Any]) -> dict[str, Any]:
    """The frame as a PolicyQuery: the schema, then the rules the profile cannot say."""
    check("policy-query", frame)
    if jcs(frame["identity"]) != jcs(frame["observation"]["identity"]):
        raise QueryError("the query's identity differs from its observation's identity")
    choices = [candidate["choice"] for candidate in frame["candidates"]]
    if len(set(choices)) != len(choices):
        raise QueryError("a candidate is listed twice")
    if frame["sampling"].get("temperature_ppm") == FULL_TEMPERATURE_PPM:
        raise QueryError("temperature_ppm 1000000 is the default and must be omitted")
    return frame


class Worker:
    def __init__(
        self,
        scorer: Scorer,
        engine_data_hash: str,
        checkpoints: list[dict[str, str]],
        max_line_bytes: int,
    ) -> None:
        self._scorer = scorer
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

    def answer(self, frame: dict[str, Any]) -> dict[str, Any]:
        """The PolicyReply of one query frame."""
        started = time.perf_counter()
        query_id = frame.get("query_id")
        try:
            query = read_query(frame)
            for candidate in query["candidates"]:
                parse_choice(candidate["choice"])
        except QueryError as exc:
            return self._error(query_id, "invalid_query", str(exc))
        except UnsupportedCandidate as exc:
            return self._error(query_id, "unsupported_candidate", str(exc))
        try:
            temperature_ppm = query["sampling"].get("temperature_ppm")
            if temperature_ppm is None:
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
            while (frame := read_frame(stdin, self._max)) is not None:
                stdout.write(encode_frame(self.answer(frame), self._max))
                stdout.flush()
        except SessionRefused:
            return EXIT_REFUSED
        except (FrameError, QueryError):
            return EXIT_FRAME
        return EXIT_OK
