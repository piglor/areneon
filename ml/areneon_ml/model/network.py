"""The Battle and Preview networks (ADR-010 §3): entity transformer, candidate head, value head.

One architecture serves both roles; they differ only in how a candidate is
embedded (per-slot parts of a joint choice for ``battle``, the ordered team
positions for ``preview``) and are built, trained and stored as separate
networks. All sizes are configuration (``ModelConfig``) and every vocabulary
size comes from the engine-data export through ``GameData``. The attention is
written out with plain matrix products and a softmax, so a forward pass uses
no fused kernel whose choice could depend on the machine.

``forward`` returns one logit per candidate and one value per query. The
candidate head scores each candidate from its parts and the tokens it names
(a cross-attention over the state), an interaction with the aggregation
token [S4]; the value head reads the aggregation token only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from areneon_ml.model.batch import Batch
from areneon_ml.model.candidates import EVENTS, SLOTS
from areneon_ml.model.encoder import (
    GENDERS,
    MAX_SLOT_LETTERS,
    REQUEST_KINDS,
    TARGET_CODES,
    EncoderConfig,
    Layout,
)
from areneon_ml.model.gamedata import GameData

ROLES = ("battle", "preview")
ACTIONS = 3  # pass, move, switch
# Vocabulary kinds that carry a numeric attribute table from the engine data.
_ATTRIBUTE_KINDS = ("species", "move", "item")


class ConfigError(ValueError):
    """A model configuration is not usable."""


def _int_field(source: Mapping[str, Any], name: str) -> int:
    value = source[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer")
    return value


def _number_field(source: Mapping[str, Any], name: str) -> float:
    value = source[name]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{name} must be a number")
    return float(value)


def encoder_config_from_dict(source: Mapping[str, Any]) -> EncoderConfig:
    """The inverse of ``EncoderConfig.as_dict``; unknown or missing keys are refused."""
    expected = set(EncoderConfig().as_dict())
    if set(source) != expected:
        raise ConfigError(f"encoder config keys must be exactly {sorted(expected)}")
    ints = ("volatile_buckets", "condition_buckets", "max_moves", "max_team", "max_picks")
    try:
        return EncoderConfig(
            **{name: _int_field(source, name) for name in ints},
            stat_scale=_number_field(source, "stat_scale"),
            turn_scale=_number_field(source, "turn_scale"),
        )
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


@dataclass(frozen=True)
class ModelConfig:
    """Network sizes: configuration, never code constants (ADR-010 §3)."""

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    d_ff: int = 256

    def __post_init__(self) -> None:
        for name in ("d_model", "n_layers", "n_heads", "d_ff"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be at least 1")
        if self.d_model % self.n_heads:
            raise ConfigError("d_model must be a multiple of n_heads")

    def as_dict(self) -> dict[str, Any]:
        return {
            "encoder": self.encoder.as_dict(),
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
        }

    @classmethod
    def from_dict(cls, source: Mapping[str, Any]) -> ModelConfig:
        keys = {"encoder", "d_model", "n_layers", "n_heads", "d_ff"}
        if set(source) != keys:
            raise ConfigError(f"model config keys must be exactly {sorted(keys)}")
        encoder = source["encoder"]
        if not isinstance(encoder, Mapping):
            raise ConfigError("encoder must be an object")
        return cls(
            encoder=encoder_config_from_dict(encoder),
            **{name: _int_field(source, name) for name in keys - {"encoder"}},
        )


@dataclass(frozen=True)
class PolicyOutput:
    """``logits`` [B, N] (padding at the dtype minimum) and ``value`` [B]."""

    logits: torch.Tensor
    value: torch.Tensor


class Attention(nn.Module):
    """Multi-head attention with an explicit key mask: matmul, softmax, matmul."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.heads = n_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        return x.reshape(b, n, self.heads, d // self.heads).transpose(1, 2)

    def forward(
        self, query: torch.Tensor, memory: torch.Tensor, key_mask: torch.Tensor
    ) -> torch.Tensor:
        q, k, v = (
            self._split(self.query(query)),
            self._split(self.key(memory)),
            self._split(self.value(memory)),
        )
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        scores = scores.masked_fill(~key_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        mixed = torch.softmax(scores, dim=-1) @ v
        b, _, n, _ = mixed.shape
        out: torch.Tensor = self.out(mixed.transpose(1, 2).reshape(b, n, -1))
        return out


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d = config.d_model
        self.norm_attention = nn.LayerNorm(d)
        self.attention = Attention(d, config.n_heads)
        self.norm_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, config.d_ff), nn.GELU(), nn.Linear(config.d_ff, d))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.norm_attention(x)
        x = x + self.attention(h, h, mask)
        out: torch.Tensor = x + self.ff(self.norm_ff(x))
        return out


def _gather(tokens: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Rows of ``tokens`` [B, M, d] at ``index`` [B, ...]; index -1 gives a zero vector."""
    b = tokens.shape[0]
    padded = torch.cat([torch.zeros_like(tokens[:, :1]), tokens], dim=1)
    rows = torch.arange(b).reshape(b, *([1] * (index.dim() - 1)))
    return padded[rows, index + 1]


class PolicyNetwork(nn.Module):
    """One role's network. Build with ``GameData`` (vocabulary sizes and attribute tables)."""

    def __init__(self, role: str, game: GameData, config: ModelConfig) -> None:
        super().__init__()
        if role not in ROLES:
            raise ConfigError(f"role must be one of {ROLES}")
        self.role = role
        self.config = config
        self.layout = Layout.of(config.encoder)
        d = config.d_model
        enc = config.encoder
        sizes = {
            "species": len(game.species),
            "move": len(game.moves),
            "item": len(game.items),
            "ability": len(game.abilities),
            "status": len(game.statuses),
            "weather": len(game.weathers),
            "terrain": len(game.terrains),
            "gender": len(GENDERS),
            "slot": MAX_SLOT_LETTERS + 1,
            "request": len(REQUEST_KINDS) + 1,
            "action": ACTIONS,
            "move_slot": enc.max_moves + 1,
            "target": TARGET_CODES,
            "event": len(EVENTS) + 1,
        }
        self.embeddings = nn.ModuleDict(
            {kind: nn.Embedding(size, d) for kind, size in sizes.items()}
        )
        tables = {
            "species": game.species_attributes,
            "move": game.move_attributes,
            "item": game.item_attributes,
        }
        self.attribute_projections = nn.ModuleDict(
            {kind: nn.Linear(tables[kind].shape[1], d) for kind in _ATTRIBUTE_KINDS}
        )
        for kind in _ATTRIBUTE_KINDS:  # derived from the engine data: not part of a checkpoint
            self.register_buffer(f"attributes_{kind}", tables[kind], persistent=False)
        self.mon_cat = nn.Linear(len(self.layout.mon_cat) * d, d)
        self.mon_num = nn.Linear(len(self.layout.mon_num), d)
        self.side = nn.Embedding(2, d)
        self.field_cat = nn.Linear(len(self.layout.field_cat) * d, d)
        self.field_num = nn.Linear(len(self.layout.field_num), d)
        self.aggregate = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(d)
        if role == "battle":
            self.slot_parts = nn.ModuleList(nn.Linear(7 * d, d) for _ in range(SLOTS))
            self.candidate = nn.Linear(SLOTS * d, d)
        else:
            self.place = nn.Embedding(enc.max_picks, d)
            self.candidate = nn.Linear(enc.max_picks * 2 * d, d)
        self.pending = nn.Linear(1, d)
        self.norm_candidate = nn.LayerNorm(d)
        self.norm_memory = nn.LayerNorm(d)
        self.cross = Attention(d, config.n_heads)
        self.score = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.value = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    # -- embeddings ----------------------------------------------------------

    def embed(self, kind: str, index: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.embeddings[kind](index)
        if kind in _ATTRIBUTE_KINDS:
            table: torch.Tensor = getattr(self, f"attributes_{kind}")
            out = out + self.attribute_projections[kind](table[index])
        return out

    def _columns(self, columns: tuple[tuple[str, str], ...], values: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.embed(kind, values[..., i]) for i, (_, kind) in enumerate(columns)], -1
        )

    # -- candidates ----------------------------------------------------------

    def _battle_candidates(self, batch: Batch, mons: torch.Tensor) -> torch.Tensor:
        slots = []
        for s in range(SLOTS):
            cat, tok = batch.cand_cat[:, :, s], batch.cand_tok[:, :, s]
            parts = [
                self.embed("action", cat[..., 0]),
                self.embed("move", cat[..., 1]),
                self.embed("move_slot", cat[..., 2]),
                self.embed("target", cat[..., 3]),
                self.embed("event", cat[..., 4]),
                _gather(mons, tok[..., 0]),
                _gather(mons, tok[..., 1]),
            ]
            slots.append(self.slot_parts[s](torch.cat(parts, -1)))
        out: torch.Tensor = self.candidate(torch.cat(slots, -1))
        return out

    def _preview_candidates(self, batch: Batch, mons: torch.Tensor) -> torch.Tensor:
        picked = _gather(mons, batch.team_tok)  # [B, N, P, d]
        places = self.place.weight.expand_as(picked)
        out: torch.Tensor = self.candidate(torch.cat([places, picked], -1).flatten(2))
        return out

    # -- forward -------------------------------------------------------------

    def forward(self, batch: Batch) -> PolicyOutput:
        if batch.kind != self.role:
            raise ValueError(f"a {self.role} network cannot score a {batch.kind} batch")
        b = batch.size
        d = self.config.d_model
        side = self.side((~batch.own_mask).long())
        mons = self.mon_cat(self._columns(self.layout.mon_cat, batch.mon_cat))
        mons = mons + self.mon_num(batch.mon_num) + side
        field_token = self.field_cat(self._columns(self.layout.field_cat, batch.field_cat))
        field_token = field_token + self.field_num(batch.field_num)
        aggregate = self.aggregate.expand(b, d)
        x = torch.cat([aggregate[:, None], field_token[:, None], mons], dim=1)
        mask = torch.cat([torch.ones(b, 2, dtype=torch.bool), batch.token_mask], dim=1)
        for block in self.blocks:
            x = block(x, mask)
        x = self.final_norm(x)
        state, mons = x[:, 0], x[:, 2:]
        build = self._battle_candidates if self.role == "battle" else self._preview_candidates
        c = build(batch, mons) + self.pending(batch.pending[..., None])
        c = c + self.cross(self.norm_candidate(c), self.norm_memory(x), mask)
        agg = state[:, None].expand_as(c)
        logits = self.score(torch.cat([c, agg, c * agg], -1)).squeeze(-1)
        logits = logits.masked_fill(~batch.cand_mask, torch.finfo(logits.dtype).min)
        return PolicyOutput(logits=logits, value=self.value(state).squeeze(-1))


def init_network(role: str, game: GameData, config: ModelConfig, seed: int) -> PolicyNetwork:
    """A freshly initialised network; the same seed gives the same weights.

    The global RNG is left as it was.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return PolicyNetwork(role, game, config)
