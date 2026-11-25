"""The one name of the device a GPU-inference setting states (ADR-030 step 4, #1458).

``generation.inference.device`` of a run config, the worker's handshake, the policy-id preimage
and the run metrics all say this string, and the worker's torch device is built from it. It is
here, with no import, so the loop's identity code and the worker can share it without either
pulling in the other.
"""

from __future__ import annotations

from typing import Final

#: The device a GPU-inference worker runs its batched passes on; the CPU is stated by omission.
#: It names torch's default CUDA device (``cuda:0``): a worker does not choose another card.
GPU_DEVICE: Final = "cuda"
