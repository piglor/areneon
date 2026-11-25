# GPU training environment (ADR-010 section 5.2)

A separate, **unshipped** uv project for running the PPO learner on the local NVIDIA GPU
(owner decision Q10 = G-a). It has its own `pyproject.toml` and its own `uv.lock`, depends on
the shipped `areneon-ml` and `areneon-contracts` packages by path, and resolves `torch` from the
PyTorch CUDA 13.0 index. It is **not** a member of the root uv workspace (the root
`pyproject.toml` excludes it) and **not** in `SHIPPED`. No CUDA, `nvidia-*` or `triton` package is
in the root `uv.lock` or in any shipped closure, and no Dockerfile may refer to this directory
(`scripts/check_cuda_separation.py`, in `make licences`, enforces C-1 to C-3).

Checkpoints trained here are plain safetensors files (tensors and a JSON header, no NVIDIA
code). Everything that ships or evaluates (the policy worker, evaluation, sealed runs) loads them
with CPU `torch` from the shipped environment (C-4).

## Install and run

```bash
cd ml/gpu-env
uv sync                       # large download (CUDA runtime wheels); the venv is gitignored
.venv/bin/python -c "import torch; print(torch.cuda.get_arch_list())"   # must list sm_120 for an RTX 5070 Ti

# One run: the learner on the GPU, the A1 games on the CPU worker of the shipped environment.
.venv/bin/python -m areneon_ml.loop run \
    --config <TrainingRunConfig.json with "learner": {"device": "cuda", ...}> \
    --run-dir <run directory> \
    --engine-data ../../fixtures/showdown/gen9championsvgc2026regmc/engine-data.json \
    --generations N \
    --worker-python ../../.venv/bin/python        # CPU torch, from `uv sync` at the repository root

# Learner throughput on a recorded stream (the figure the Wave 4 sizing needs):
.venv/bin/python -m areneon_ml.learner bench --config spike-config.json \
    --engine-data ../../fixtures/showdown/gen9championsvgc2026regmc/engine-data.json \
    --stream-dir <run directory>/generations/g000001 --repeats 5
```

`spike-config.json` is the TrainingRunConfig of the sm_120 spike: a 3-layer, 128-wide network with
minibatches of 64 on the GPU. A config with `"device": "cuda"` is refused where torch has no usable
CUDA device; it never falls back to the CPU (a CPU run is a different config and a different
hash). Updating the torch version means changing it in `ml/pyproject.toml` (CPU, root lock) and in
this project (CUDA, this lock) together; `tests/scripts/test_gpu_env.py` fails if they differ.

## GPU inference for training self-play (ADR-030 step 4)

A run whose config says `generation.inference: {kind: "batched", max_batch, device: "cuda"}` starts its
play workers from this environment's Python (`--inference-worker-python .venv/bin/python` on `loop run`),
with `--worker-python` still the shipped CPU Python for the batch-1 control worker. Training self-play
only: every evaluation client refuses such a worker at the handshake. On a CUDA allocation failure (torch's
`OutOfMemoryError`, or the plain `RuntimeError` of a failed context or cuBLAS handle allocation) a worker
continues on its CPU path, and `--inference-cpu` rolls every worker back to it; see
`docs/training-run.md` section 11.

Host memory: a CUDA worker's peak resident set is 1.44 GiB at the `public-gpu16` network size (a CPU worker
0.44 GiB), so `--game-processes 6 --policy-workers 2` is about 23 GB of RAM. Start with
`--game-processes 2 --policy-workers 1` and raise the counts from the first generation's measured
`peak_rss_bytes.policy_worker`. A worker uses torch's default CUDA device (`cuda:0`).

A GPU-device test for local runs, skipped in CI because the shipped environment has no CUDA: from the
repository root, `PYTHONPATH=.:packages/schema/python:ml:<dir> ml/gpu-env/.venv/bin/python -m pytest -o
addopts="" tests/ml/test_worker_gpu.py -k cuda`, where `<dir>` holds links to the root environment's `pytest`,
`_pytest`, `pluggy`, `iniconfig`, `packaging`, `pygments` and `py.py` (this environment has no `pytest`, and
it must not be changed). It runs tiny networks on the card and compares them with the CPU scorer.

## Reproducibility

The learner enables deterministic algorithms and a fixed cuBLAS workspace, but a GPU run is **not
claimed** to be bit-reproducible, across machines, drivers or torch builds. On the author's machine
three fresh runs of one generation produced the same `learner_state_hash`; that is an observation,
not a guarantee. A CUDA run is named by its own config hash (`learner.device` is part of it).

## Licences of this environment

Not part of Areneon's shipped licence set, and not checked by `make licences`. As declared by the
installed package metadata on 2026-10-20 (torch 2.14.1+cu130):

| Package | Declared licence |
|---|---|
| `torch` | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT |
| `triton` | MIT |
| `cuda-bindings`, `cuda-pathfinder` | Apache-2.0 |
| `nvidia-nvtx` | Apache 2.0 |
| `nvidia-cublas`, `-cuda-cupti`, `-cuda-nvrtc`, `-cuda-runtime`, `-cudnn-cu13`, `-cufft`, `-cufile`, `-curand`, `-cusolver`, `-cusparse`, `-nccl-cu13`, `-nvjitlink`, `-nvshmem-cu13` | `LicenseRef-NVIDIA-Proprietary` |
| `nvidia-cusparselt-cu13` | NVIDIA Proprietary Software |
| `cuda-toolkit` | none declared |

Whether NVIDIA's terms for the CUDA runtime restrict the use or publication of checkpoints trained
with it is **Unknown** and was not checked (owner decision, ADR-010 C-R3, risk R15). Nothing here
claims that such use or publication is permitted.
