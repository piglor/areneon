"""Padding several ``EncodedQuery`` into one ``Batch`` (training may batch; ADR-010 §2).

Evaluation and sealed runs never batch: they run one query per forward pass,
so a batch of one has no padding and the same tensors as the query itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from areneon_ml.model.encoder import EncodedQuery


@dataclass(frozen=True)
class Batch:
    """Queries of one kind, padded to the longest token and candidate lists."""

    kind: str
    token_mask: torch.Tensor  # bool [B, M]: a real Pokémon row
    own_mask: torch.Tensor  # bool [B, M]: a row of the player's own Pokémon
    mon_cat: torch.Tensor  # int64 [B, M, C]
    mon_num: torch.Tensor  # float32 [B, M, F]
    field_cat: torch.Tensor  # int64 [B, K]
    field_num: torch.Tensor  # float32 [B, G]
    cand_mask: torch.Tensor  # bool [B, N]: a real candidate
    cand_cat: torch.Tensor  # int64 [B, N, S, A]
    cand_tok: torch.Tensor  # int64 [B, N, S, 2], -1 for none
    team_tok: torch.Tensor  # int64 [B, N, P], -1 for none
    pending: torch.Tensor  # float32 [B, N]

    @property
    def size(self) -> int:
        return int(self.token_mask.shape[0])


def _pad(tensor: torch.Tensor, shape: Sequence[int], value: float) -> torch.Tensor:
    """``tensor`` at the top-left corner of a ``shape`` tensor filled with ``value``."""
    out = torch.full(tuple(shape), value, dtype=tensor.dtype)
    out[tuple(slice(0, n) for n in tensor.shape)] = tensor
    return out


def collate(queries: Sequence[EncodedQuery]) -> Batch:
    """Pad and stack; the queries must all be Team Preview or all be Battle queries."""
    if not queries:
        raise ValueError("a batch needs at least one query")
    kinds = {q.kind for q in queries}
    if len(kinds) != 1:
        raise ValueError(f"a batch holds one kind of query, not {sorted(kinds)}")
    rows = max(q.mon_cat.shape[0] for q in queries)
    count = max(q.pending.shape[0] for q in queries)

    def stack(name: str, pad_to: Sequence[int], value: float) -> torch.Tensor:
        return torch.stack([_pad(getattr(q, name), pad_to, value) for q in queries])

    mon_cat_width = queries[0].mon_cat.shape[1]
    mon_num_width = queries[0].mon_num.shape[1]
    slots, parts = queries[0].cand_cat.shape[1], queries[0].cand_cat.shape[2]
    picks = queries[0].team_tok.shape[1]
    token_mask = torch.zeros(len(queries), rows, dtype=torch.bool)
    own_mask = torch.zeros(len(queries), rows, dtype=torch.bool)
    cand_mask = torch.zeros(len(queries), count, dtype=torch.bool)
    for b, q in enumerate(queries):
        token_mask[b, : q.mon_cat.shape[0]] = True
        own_mask[b, : q.n_own] = True
        cand_mask[b, : q.pending.shape[0]] = True
    return Batch(
        kind=queries[0].kind,
        token_mask=token_mask,
        own_mask=own_mask,
        mon_cat=stack("mon_cat", (rows, mon_cat_width), 0),
        mon_num=stack("mon_num", (rows, mon_num_width), 0.0),
        field_cat=torch.stack([q.field_cat for q in queries]),
        field_num=torch.stack([q.field_num for q in queries]),
        cand_mask=cand_mask,
        cand_cat=stack("cand_cat", (count, slots, parts), 0),
        cand_tok=stack("cand_tok", (count, slots, 2), -1),
        team_tok=stack("team_tok", (count, picks), -1),
        pending=stack("pending", (count,), 0.0),
    )
