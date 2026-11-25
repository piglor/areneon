"""``python -m areneon_ml.worker``: serve one session on stdin and stdout.

Everything it needs comes from the command line: the run directory, the two
checkpoint ids, the engine-data table and the maximum line length (a
``TrainingRunConfig`` value, never a default here). Each checkpoint is loaded
by id (its bytes must hash to it), must hold the network of its role, and must
have been made for this engine-data table and encoder version. CPU inference is
made deterministic before the first query.

A worker of a GPU-inference training run (ADR-030 step 4, #1458) is started with
``--inference-device cuda`` from the unshipped ``ml/gpu-env`` Python: it states the
device in its handshake (so every batch-1 evaluation client refuses it), runs its
batched passes on the GPU and fails closed to its CPU path on an allocation failure;
``--inference-force-cpu`` (the rollback) starts it on that path, and
``--inference-stats-dir`` is where it leaves its device counts: a file of its own (the pid and a
random token in the name, so a reused pid cannot overwrite another worker's) written when the
session starts, rewritten at most once a second while it serves and once more when it ends, so a
worker that is killed is still counted. A failing write is reported on stderr and never changes
the exit code or hides the session's own error.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import secrets
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from areneon_ml import stopwatch
from areneon_ml.checkpoint.files import CheckpointFileError, load_checkpoint
from areneon_ml.device import GPU_DEVICE
from areneon_ml.model.gamedata import GameData
from areneon_ml.model.network import ConfigError, ModelConfig
from areneon_ml.model.options import feature_set, sheet_set
from areneon_ml.model.placement import InferencePlacement
from areneon_ml.model.runtime import DeviceError, configure_cpu_inference, configure_gpu_inference
from areneon_ml.model.scorer import NetworkScorer, ScorerError
from areneon_ml.worker.checkpoints import CheckpointError, CheckpointStore
from areneon_ml.worker.features import damaging_moves
from areneon_ml.worker.serve import MIN_BATCH, FeatureSupport, SheetSupport, Worker

EXIT_CHECKPOINT = 4
#: The worker was asked for a device this torch cannot use (ADR-030 step 4).
EXIT_DEVICE = 5


def _configured_model(run_dir: Path) -> ModelConfig | None:
    """The model section of the run's ``run-config.json``, when the run directory has one.

    A checkpoint made with other model options than the run's is refused (ADR-010 erratum A4
    item 2). A directory without the record (a hand-made one) is checked by the checkpoint alone.
    """
    path = run_dir / "run-config.json"
    if not path.is_file():
        return None
    try:
        return ModelConfig.from_dict(json.loads(path.read_text(encoding="utf-8"))["model"])
    except (OSError, ValueError, KeyError, TypeError, ConfigError) as exc:
        raise CheckpointFileError(
            f"the run's run-config.json has no usable model section: {exc}"
        ) from exc


def build(argv: list[str]) -> Worker:
    parser = argparse.ArgumentParser(prog="areneon_ml.worker")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preview", required=True, help="checkpoint id of the Preview Policy")
    parser.add_argument("--battle", required=True, help="checkpoint id of the Battle Policy")
    parser.add_argument("--engine-data", type=Path, required=True)
    parser.add_argument("--max-line-bytes", type=int, required=True)
    parser.add_argument(
        "--diagnostic-sheet",
        action="store_true",
        help="accept an oracle-diagnostic opponent_sheet (a labelled, test-only diagnostic run)",
    )
    parser.add_argument(
        "--inference-max-batch",
        type=int,
        help="answer up to this many queued queries in one padded forward pass (training "
        "self-play of a run whose config sets generation.inference, ADR-028 section 2 F2); "
        "absent is batch 1, which every evaluation and sealed path uses",
    )
    parser.add_argument(
        "--inference-device",
        choices=(GPU_DEVICE,),
        help="run the batched forward passes on this device (training self-play of a run whose "
        "config says generation.inference.device, ADR-030 step 4; needs --inference-max-batch "
        "and a torch with CUDA, the ml/gpu-env environment); stated in the handshake, so every "
        "batch-1 evaluation client refuses this worker",
    )
    parser.add_argument(
        "--inference-force-cpu",
        action="store_true",
        help="the rollback of a GPU-inference run: state the device but run every pass on the CPU "
        "path (counted as forced)",
    )
    parser.add_argument(
        "--inference-stats-dir",
        type=Path,
        help="leave this GPU-inference worker's device counts here (inference-<pid>-<token>.json: "
        "written at the start, rewritten while it serves and at the end)",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help="time the encoder and the forward pass and leave worker-<pid>.json here when the "
        "session ends (the generation profile of ADR-028 section 1; the replies are the same)",
    )
    args = parser.parse_args(argv)
    if args.max_line_bytes < 1:
        parser.error("--max-line-bytes must be positive")
    if args.inference_max_batch is not None and args.inference_max_batch < MIN_BATCH:
        parser.error("--inference-max-batch must be at least 2 (batch 1 is the default)")
    if args.inference_device is not None and args.inference_max_batch is None:
        parser.error("--inference-device needs --inference-max-batch (a GPU worker batches)")
    if args.inference_device is None and (
        args.inference_force_cpu or args.inference_stats_dir is not None
    ):
        parser.error("--inference-force-cpu and --inference-stats-dir need --inference-device")
    store = CheckpointStore(args.run_dir)
    game = GameData.from_table(json.loads(args.engine_data.read_text(encoding="utf-8")))
    placement: InferencePlacement | None = None
    if args.inference_device is None or args.inference_force_cpu:
        configure_cpu_inference()
    else:
        configure_gpu_inference()  # refuses a torch without CUDA: DeviceError
    if args.inference_device is not None:
        placement = InferencePlacement(
            torch.device("cpu" if args.inference_force_cpu else args.inference_device),
            forced_cpu=args.inference_force_cpu,
        )
    configured = _configured_model(args.run_dir)
    networks = {
        role: load_checkpoint(
            store.load(checkpoint_id), game, checkpoint_id, role=role, configured=configured
        )
        for role, checkpoint_id in (("preview", args.preview), ("battle", args.battle))
    }
    # The feature set is stated from the checkpoints' own metadata (their model config). Both
    # networks share one encoder (the scorer refuses a pair that does not), so one statement.
    scorer = NetworkScorer(game, networks, placement)  # refuses a pair with different encoders
    stated = feature_set(networks["battle"].config.encoder.options)
    sheet = sheet_set(networks["battle"].config.encoder.options)
    worker = Worker(
        scorer=scorer,
        features=(
            None if stated is None else FeatureSupport(stated.as_dict(), damaging_moves(game))
        ),
        sheet=None if sheet is None else SheetSupport(sheet.as_dict(), args.diagnostic_sheet),
        value_scorer=scorer if scorer.value_query else None,
        engine_data_hash=game.engine_data_hash,
        checkpoints=[
            {"role": "battle", "checkpoint_id": args.battle},
            {"role": "preview", "checkpoint_id": args.preview},
        ],
        max_line_bytes=args.max_line_bytes,
        inference_max_batch=args.inference_max_batch,
        inference_device=args.inference_device,
    )
    worker.profile_dir = args.profile_dir
    worker.inference_stats_dir = args.inference_stats_dir
    worker.placement_stats = scorer.placement_stats
    return worker


def write_profile(directory: Path, started: resource.struct_rusage) -> Path:
    """Leave this process's profile in ``directory/worker-<pid>.json`` (whole milliseconds).

    ``cpu_ms`` is the CPU the process spent after ``started`` (taken when the session begins), so
    importing torch and loading the checkpoints do not count as play.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    spent = (usage.ru_utime + usage.ru_stime) - (started.ru_utime + started.ru_stime)
    totals = stopwatch.snapshot()

    def at(name: str, key: str) -> int:
        return stopwatch.ms(totals.get(name, {}).get(key, 0))

    figures = {
        "cpu_ms": int(spent * 1000),
        "peak_rss_bytes": int(usage.ru_maxrss) * 1024,
        "queries": totals.get("w_fwd", {}).get("count", 0),
        "w_enc_ms": at("w_enc", "wall_ns"),
        "w_fwd_ms": at("w_fwd", "wall_ns"),
        "w_enc_cpu_ms": at("w_enc", "cpu_ns"),
        "w_fwd_cpu_ms": at("w_fwd", "cpu_ns"),
    }
    path = directory / f"worker-{os.getpid()}.json"
    partial = directory / f".worker-{os.getpid()}.partial"
    partial.write_text(json.dumps(figures, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)
    return path


def write_inference_stats(directory: Path, stats: dict[str, Any], token: str | None = None) -> Path:
    """Leave a GPU-inference worker's device counts in ``directory/inference-<pid>-<token>.json``.

    The token is random unless the caller keeps one to rewrite its own file: the pid alone could
    be reused by a later worker of the same generation, whose file would then replace this one's.
    """
    name = f"inference-{os.getpid()}-{secrets.token_hex(6) if token is None else token}"
    path = directory / f"{name}.json"
    partial = directory / f".{name}.partial"
    partial.write_text(json.dumps(stats, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)
    return path


class DeviceStatsFile:
    """One worker's counts file: written at the start, throttled while it serves, once at the end.

    ``stats`` returns the worker's current counts (None for a worker that keeps none, which then
    leaves no file). Every write is best effort: a failure is reported on stderr (once) and never
    raised, because the counts are a diagnostic and must not change a session's exit code or hide
    its error.
    """

    def __init__(
        self,
        directory: Path,
        stats: Callable[[], dict[str, Any] | None],
        clock: Callable[[], float] = time.monotonic,
        interval_s: float = 1.0,
    ) -> None:
        self._directory = directory
        self._stats = stats
        self._clock = clock
        self._interval = interval_s
        self._token = secrets.token_hex(6)
        self._last: float | None = None
        self._reported = False

    def write(self) -> None:
        found = self._stats()
        if found is None:
            return
        try:
            write_inference_stats(self._directory, found, self._token)
        except OSError as exc:
            if not self._reported:  # once: the heartbeat would repeat it every interval
                self._reported = True
                print(
                    f"areneon_ml.worker: could not leave the device counts: {exc}", file=sys.stderr
                )

    def beat(self) -> None:
        """Rewrite the file if an interval has passed since the last write (the first call does)."""
        now = self._clock()
        if self._last is None or now - self._last >= self._interval:
            self._last = now
            self.write()


def main(argv: list[str] | None = None) -> int:
    try:
        worker = build(sys.argv[1:] if argv is None else argv)
    except DeviceError as exc:
        print(f"areneon_ml.worker: {exc}", file=sys.stderr)
        return EXIT_DEVICE
    except (CheckpointError, CheckpointFileError, ScorerError) as exc:
        print(f"areneon_ml.worker: {exc}", file=sys.stderr)
        return EXIT_CHECKPOINT
    counts = (
        None
        if worker.inference_stats_dir is None
        else DeviceStatsFile(worker.inference_stats_dir, worker.placement_stats)
    )
    if counts is not None:
        worker.heartbeat = counts.beat
        counts.write()  # a worker that is killed later is still counted
    try:
        return _serve(worker)
    finally:
        if counts is not None:
            counts.write()


def _serve(worker: Worker) -> int:
    profile_dir: Path | None = worker.profile_dir
    if profile_dir is None:
        return worker.serve(sys.stdin.buffer, sys.stdout.buffer)
    stopwatch.enable()
    started = resource.getrusage(resource.RUSAGE_SELF)
    try:
        return worker.serve(sys.stdin.buffer, sys.stdout.buffer)
    finally:
        write_profile(profile_dir, started)
        stopwatch.disable()


if __name__ == "__main__":
    raise SystemExit(main())
