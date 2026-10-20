"""Deterministic CPU inference (ADR-010 §2, batch invariance).

Evaluation and sealed runs score one query per forward pass on CPU with
deterministic algorithms enabled [S13] and one intra-op thread, so float
accumulation does not depend on which other battles were waiting or on how
many cores the machine has. Training self-play may batch and may use more
threads: it does not call this.
"""

from __future__ import annotations

import torch


def configure_cpu_inference() -> None:
    """Make this process's torch deterministic for evaluation; call once at start-up."""
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
