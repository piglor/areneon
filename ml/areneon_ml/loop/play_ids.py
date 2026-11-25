"""Print the policy id and the encoder version a direct run of the play script needs.

``python -m areneon_ml.loop.play_ids --config C --engine-data E --preview P --battle B`` prints
one JSON line ``{"encoder_version": ..., "policy_id": ...}``. The generation loop computes both
itself and hands them to ``play-generation.ts``; a tool that runs that script directly (the
throughput harness) asks here instead, so the id is computed by the code that defines it: the
same ``Frozen.policy_id`` the driver uses (play temperature, batched inference and candidate
features all enter it) and the same ``encoder_version``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from areneon_ml.learner.config import RunConfigError, read_run_config
from areneon_ml.loop.driver import Frozen
from areneon_ml.loop.records import features_hash
from areneon_ml.model.encoder import encoder_version
from areneon_ml.model.gamedata import GameData

_DIGEST = re.compile(r"[0-9a-f]{64}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="play_ids", description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="the run's TrainingRunConfig")
    parser.add_argument("--engine-data", type=Path, required=True)
    parser.add_argument("--preview", required=True, help="the Preview checkpoint id")
    parser.add_argument("--battle", required=True, help="the Battle checkpoint id")
    args = parser.parse_args(argv)
    for name, value in (("preview", args.preview), ("battle", args.battle)):
        if not _DIGEST.fullmatch(value):
            message = f"play-ids: the {name} checkpoint id is not 64 hexadecimal digits"
            print(message, file=sys.stderr)
            return 2
    try:
        config = read_run_config(json.loads(args.config.read_text(encoding="utf-8")))
        game = GameData.from_table(json.loads(args.engine_data.read_text(encoding="utf-8")))
    except (OSError, ValueError, RunConfigError) as exc:
        print(f"play-ids: {exc}", file=sys.stderr)
        return 2
    pair = Frozen(
        generation=0,
        checkpoints={"preview": args.preview, "battle": args.battle},
        learner_state_hash="",
        training_set_hash=None,
        team_hashes=(),
        features=features_hash(config),
        temperature_ppm=config.generation.temperature_ppm,
        inference_max_batch=config.generation.inference_max_batch,
        inference_device=config.generation.inference_device,
    )
    found = {
        "encoder_version": encoder_version(game, config.model.encoder),
        "policy_id": pair.policy_id,
    }
    print(json.dumps(found, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
