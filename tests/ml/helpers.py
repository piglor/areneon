"""Shared fixtures for the policy worker tests: corpus queries and a run directory."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

from areneon_contracts.contracts import jcs
from areneon_ml.worker.checkpoints import CheckpointStore
from areneon_ml.worker.scorer import StubScorer
from areneon_ml.worker.serve import Worker

REPO = Path(__file__).resolve().parents[2]
SAMPLES = REPO / "fixtures/contracts/v1"
ENGINE_DATA = REPO / "fixtures/showdown/gen9championsvgc2026regmc/engine-data.json"
ENGINE_DATA_HASH = json.loads(ENGINE_DATA.with_name("engine-data.summary.json").read_text())[
    "engine_data_hash"
]
MAX_LINE = 4_000_000


def sample(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((SAMPLES / name).read_text(encoding="utf-8"))
    return loaded


def battle_query(**changes: Any) -> dict[str, Any]:
    return {**sample("policy-query/valid/battle.json"), **changes}


def preview_query(**changes: Any) -> dict[str, Any]:
    return {**sample("policy-query/valid/preview.json"), **changes}


def line(value: dict[str, Any]) -> bytes:
    return jcs(value) + b"\n"


def hello(**changes: Any) -> dict[str, Any]:
    message = {**sample("policy-handshake/valid/hello.json")["message"], **changes}
    return {"schema_version": 1, "message": message}


def make_run_dir(root: Path) -> tuple[Path, str, str]:
    """A run directory holding a Preview and a Battle checkpoint; their ids."""
    ids = []
    for role in ("preview", "battle"):
        data = f"stub weights for the {role} policy".encode()
        checkpoint_id = hashlib.sha256(data).hexdigest()
        (root / "checkpoints").mkdir(exist_ok=True)
        (root / "checkpoints" / f"{checkpoint_id}.safetensors").write_bytes(data)
        ids.append(checkpoint_id)
    return root, ids[0], ids[1]


def make_worker(root: Path, max_line_bytes: int = MAX_LINE) -> Worker:
    run_dir, preview, battle = make_run_dir(root)
    store = CheckpointStore(run_dir)
    return Worker(
        scorer=StubScorer({"preview": store.load(preview), "battle": store.load(battle)}),
        engine_data_hash=ENGINE_DATA_HASH,
        checkpoints=[
            {"role": "preview", "checkpoint_id": preview},
            {"role": "battle", "checkpoint_id": battle},
        ],
        max_line_bytes=max_line_bytes,
    )


def session(worker: Worker, *frames: bytes) -> tuple[int, list[dict[str, Any]]]:
    """Run a session on the given input lines: the exit code and the frames written."""
    stdout = io.BytesIO()
    code = worker.serve(io.BytesIO(b"".join(frames)), stdout)
    return code, [json.loads(text) for text in stdout.getvalue().splitlines()]
