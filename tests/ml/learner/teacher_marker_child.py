"""A child process of ``test_teacher_marker.py``: opens a teacher stream the way a driver process
does, so the tests can kill it with a real signal or run two of them at once.

``python -m tests.ml.learner.teacher_marker_child open|kill DIR [N]`` reads ``DIR/stream.jsonl``
and ``DIR/training-set.json``; ``kill`` makes the N-th encoded example raise SIGKILL against the
process itself, which is the kill a full pass can meet at any point.
"""

from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

from areneon_contracts.contracts import jcs, sha256_hex
from areneon_ml.learner import imitation
from areneon_ml.learner.data import _Collector
from areneon_ml.model.encoder import Encoder
from areneon_ml.model.network import ROLES

from tests.ml.learner.support import tiny_config
from tests.ml.model.support import game


def main(argv: list[str]) -> int:
    mode, directory = argv[0], Path(argv[1])
    manifest = json.loads((directory / "training-set.json").read_text())
    encoders = {role: Encoder(game(), tiny_config().model.encoder) for role in ROLES}
    if mode == "kill":
        remaining = int(argv[2])
        real = _Collector.add

        def add(self: Any, example: dict[str, Any]) -> None:
            nonlocal remaining
            remaining -= 1
            if remaining <= 0:
                os.kill(os.getpid(), signal.SIGKILL)
            real(self, example)

        _Collector.add = add  # type: ignore[method-assign]
    opened = imitation.open_teacher(
        directory / "stream.jsonl", manifest, sha256_hex(jcs(manifest)), encoders
    )
    print(len(opened))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
