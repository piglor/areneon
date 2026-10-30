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
from dataclasses import dataclass, fields
from typing import Any, cast

import torch
from torch import nn

from areneon_ml.model.batch import Batch
from areneon_ml.model.candidates import EVENTS, SLOTS
from areneon_ml.model.encoder import (
    BASE_CAND_COLUMNS,
    BASE_CAND_TOKENS,
    GENDERS,
    MAX_SLOT_LETTERS,
    REQUEST_KINDS,
    TARGET_CODES,
    EncoderConfig,
    Layout,
)
from areneon_ml.model.features import (
    KIND_CATEGORIES,
    PROBE_KIND_CATEGORIES,
    SPEED_CATEGORIES,
)
from areneon_ml.model.gamedata import RESERVED, GameData
from areneon_ml.model.options import (
    CANDIDATE_MATCHUP,
    FACTORED_PREVIEW,
    POLICY_OUTPUT_SCALE_PPM,
    SEPARATE_VALUE,
    SEPARATE_VALUE_DETACHED,
    SEPARATE_VALUE_NETWORK,
    SHARED_TRUNK,
    OptionError,
    Options,
    is_set,
    layout_changing,
    option_value,
    read_options,
    with_options,
)

ROLES = ("battle", "preview")
ACTIONS = 3  # pass, move, switch
PPM = 1_000_000
LEAD_PAIR = 2  # the picks at places 1 and 2 are the leads (factored_preview)
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
    expected = {f.name for f in fields(EncoderConfig)} - {"options"}
    if set(source) - {"options"} != expected:
        raise ConfigError(f"encoder config keys must be exactly {sorted(expected)} (and options)")
    try:
        options = read_options("encoder", source["options"]) if "options" in source else ()
    except OptionError as exc:
        raise ConfigError(str(exc)) from exc
    ints = ("volatile_buckets", "condition_buckets", "max_moves", "max_team", "max_picks")
    try:
        return EncoderConfig(
            **{name: _int_field(source, name) for name in ints},
            stat_scale=_number_field(source, "stat_scale"),
            turn_scale=_number_field(source, "turn_scale"),
            options=options,
        )
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


# Outer admissibility bounds (ADR-010 §5: code holds bounds, never the values). They keep a crafted
# configuration from allocating without limit; they are far above any size the slice trains.
MAX_DIMENSION = 4096
MAX_LAYERS = 64
MAX_HEADS = 64
MAX_FEED_FORWARD = 16384
MAX_SEED = 2**63 - 1


@dataclass(frozen=True)
class ModelConfig:
    """Network sizes: configuration, never code constants (ADR-010 §3).

    They come from the ``model`` section of the run's ``TrainingRunConfig``; there are no defaults.
    """

    encoder: EncoderConfig
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    options: Options = ()  # optional network options (A4 item 2); empty keeps today's bytes

    def __post_init__(self) -> None:
        bounds = {
            "d_model": MAX_DIMENSION,
            "n_layers": MAX_LAYERS,
            "n_heads": MAX_HEADS,
            "d_ff": MAX_FEED_FORWARD,
        }
        for name, bound in bounds.items():
            if not 1 <= getattr(self, name) <= bound:
                raise ConfigError(f"{name} must be at least 1 and at most {bound}")
        if self.d_model % self.n_heads:
            raise ConfigError("d_model must be a multiple of n_heads")

    def as_dict(self) -> dict[str, Any]:
        return with_options(
            {
                "encoder": self.encoder.as_dict(),
                "d_model": self.d_model,
                "n_layers": self.n_layers,
                "n_heads": self.n_heads,
                "d_ff": self.d_ff,
            },
            self.options,
        )

    @property
    def layout_changing(self) -> bool:
        """Whether an option changes the checkpoint tensor layout (its format is bumped)."""
        return layout_changing("network", self.options) or layout_changing(
            "encoder", self.encoder.options
        )

    @classmethod
    def from_dict(cls, source: Mapping[str, Any]) -> ModelConfig:
        keys = {"encoder", "d_model", "n_layers", "n_heads", "d_ff"}
        if set(source) - {"options"} != keys:
            raise ConfigError(f"model config keys must be exactly {sorted(keys)} (and options)")
        try:
            options = read_options("network", source["options"]) if "options" in source else ()
        except OptionError as exc:
            raise ConfigError(str(exc)) from exc
        encoder = source["encoder"]
        if not isinstance(encoder, Mapping):
            raise ConfigError("encoder must be an object")
        return cls(
            encoder=encoder_config_from_dict(encoder),
            options=options,
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


def _pool(
    rows: torch.Tensor, index: torch.Tensor, keep: torch.Tensor, size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The masked sum and max of ``rows`` [B, R, d] per token ``index`` [B, R] in 0..size-1.

    ``keep`` [B, R] drops a row (padding, or the other Mega flag). A token with no row gets
    zeros; the second result [B, size] says which tokens have any. Returns [B, size, 2d].
    """
    if rows.shape[1] == 0:  # no row at all in the batch: nothing to pool (amax needs one)
        none = torch.zeros(rows.shape[0], size, 1, dtype=torch.bool, device=rows.device)
        return torch.zeros(rows.shape[0], size, 2 * rows.shape[2], device=rows.device), none
    tokens = torch.arange(size, device=rows.device)
    mask = (index[:, None, :] == tokens[None, :, None]) & keep[:, None, :]  # [B, size, R]
    held = mask[..., None]
    total = (rows[:, None] * held).sum(2)
    peak = rows[:, None].masked_fill(~held, torch.finfo(rows.dtype).min).amax(2)
    some = mask.any(2)[..., None]
    peak = torch.where(some, peak, torch.zeros_like(peak))
    return torch.cat([total, peak], -1), some


def _gather(tokens: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Rows of ``tokens`` [B, M, d] at ``index`` [B, ...]; index -1 gives a zero vector."""
    b = tokens.shape[0]
    padded = torch.cat([torch.zeros_like(tokens[:, :1]), tokens], dim=1)
    rows = torch.arange(b, device=tokens.device).reshape(b, *([1] * (index.dim() - 1)))
    return padded[rows, index + 1]


class PolicyNetwork(nn.Module):
    """One role's network. Build with ``GameData`` (vocabulary sizes and attribute tables)."""

    def __init__(  # noqa: PLR0912, PLR0915
        self, role: str, game: GameData, config: ModelConfig, *, value_only: bool = False
    ) -> None:
        """``value_only`` (internal) builds just the trunk and the value head: the separate value
        network of ``separate_value``, which shares no layer with the policy network."""
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
        extra = len(self.layout.cand_cat) - BASE_CAND_COLUMNS  # option columns of a candidate
        if role == "battle" and is_set(enc.options, CANDIDATE_MATCHUP):
            # Cell categories: the reserved indices plus one per code of the export's chart.
            sizes["matchup"] = RESERVED + game.chart_codes
            sizes["same_type"] = RESERVED + 2
        numbers = len(self.layout.cand_num) if role == "battle" else 0
        targeted = role == "battle" and len(self.layout.cand_tok) > BASE_CAND_TOKENS
        if numbers:
            sizes["feature_kind"] = KIND_CATEGORIES
        if self.layout.thr_num:  # probe-damage/2: every token has speed columns, the Battle rest
            sizes["speed_relation"] = SPEED_CATEGORIES
            if role == "battle":
                sizes["probe_kind"] = PROBE_KIND_CATEGORIES
        self.targeted = targeted
        # factored_preview: only the Preview network is factored; the Battle network ignores it.
        self.factored = (
            not value_only and role == "preview" and is_set(config.options, FACTORED_PREVIEW)
        )
        if self.factored and enc.max_picks < LEAD_PAIR:
            raise ConfigError("factored_preview needs max_picks of at least 2 (the lead pair)")
        probing = role == "battle" and bool(self.layout.thr_num)  # probe-damage/2 rows
        if value_only:  # the trunk reads the token and field columns (and the probe rows' kinds)
            used = {kind for _, kind in self.layout.mon_cat + self.layout.field_cat}
            used |= {"move", "probe_kind"} if probing else set()
            sizes = {kind: size for kind, size in sizes.items() if kind in used}
        self.embeddings = nn.ModuleDict(
            {kind: nn.Embedding(size, d) for kind, size in sizes.items()}
        )
        tables = {
            "species": game.species_attributes,
            "move": game.move_attributes,
            "item": game.item_attributes,
        }
        kinds = [kind for kind in _ATTRIBUTE_KINDS if kind in sizes]
        self.attribute_projections = nn.ModuleDict(
            {kind: nn.Linear(tables[kind].shape[1], d) for kind in kinds}
        )
        for kind in kinds:  # derived from the engine data: not part of a checkpoint
            self.register_buffer(f"attributes_{kind}", tables[kind], persistent=False)
        self.mon_cat = nn.Linear(len(self.layout.mon_cat) * d, d)
        self.mon_num = nn.Linear(len(self.layout.mon_num), d)
        self.side = nn.Embedding(2, d)
        self.field_cat = nn.Linear(len(self.layout.field_cat) * d, d)
        self.field_num = nn.Linear(len(self.layout.field_num), d)
        self.aggregate = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(d)
        self.probing = probing
        self.value_net: PolicyNetwork | None = None
        if value_only:
            if probing:
                self._probe_layers(d)
            self.value = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
            return
        if role == "battle":
            # The damage rows' numbers are one more d-wide part of a slot (candidate_features).
            self.cand_features = nn.Linear(numbers, d) if numbers else None
            width = (
                7
                + extra
                + (1 if numbers else 0)
                + (1 if self.probing else 0)
                + (1 if targeted else 0)
            )
            self.slot_parts = nn.ModuleList(nn.Linear(width * d, d) for _ in range(SLOTS))
            self.candidate = nn.Linear(SLOTS * d, d)
            if self.probing:
                self._probe_layers(d)
        else:
            self.place = nn.Embedding(enc.max_picks, d)
            if self.factored:
                # u(place, token) and v(lead pair): the whole Preview score, no cross-attention.
                self.place_score = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
                self.lead_score = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
            else:
                self.candidate = nn.Linear(enc.max_picks * 2 * d, d)
        if not self.factored:
            self.pending = nn.Linear(1, d)
            self.norm_candidate = nn.LayerNorm(d)
            self.norm_memory = nn.LayerNorm(d)
            self.cross = Attention(d, config.n_heads)
            self.score = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
        # separate_value: "network" builds a second trunk and value head that share no layer with
        # this one; "detached" keeps the head here but reads the trunk's output without gradient.
        self.value_mode = option_value(config.options, SEPARATE_VALUE)
        if self.value_mode == SEPARATE_VALUE_NETWORK:
            self.value_net = PolicyNetwork(role, game, config, value_only=True)
        else:
            self.value = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        scale = option_value(config.options, POLICY_OUTPUT_SCALE_PPM)
        if isinstance(scale, int):  # the scorer's last layer starts at scale / 10^6 of its init
            heads = [self.place_score, self.lead_score] if self.factored else [self.score]
            with torch.no_grad():
                for head in heads:
                    cast(nn.Linear, head[-1]).weight.mul_(scale / PPM)

    # -- shared trunk --------------------------------------------------------

    def trunk_kinds(self) -> tuple[str, ...]:
        """The vocabulary kinds the token and field columns read (their embeddings are trunk)."""
        used = {kind for _, kind in self.layout.mon_cat + self.layout.field_cat}
        return tuple(sorted(used))

    def trunk_keys(self) -> tuple[str, ...]:
        """The ``state_dict`` keys of the trunk ``shared_trunk`` shares between the two roles:
        the embeddings (and attribute projections) of the kinds the token and field columns
        read, the token, side and field layers, the aggregation token, the transformer blocks and
        the final norm. The candidate parts, the probe layers, the heads and a separate value
        network (``value_net.*``) are role-specific and never in it."""
        kinds = self.trunk_kinds()
        prefixes = [f"embeddings.{kind}." for kind in kinds]
        prefixes += [f"attribute_projections.{kind}." for kind in kinds]
        prefixes += ["mon_cat.", "mon_num.", "side.", "field_cat.", "field_num.", "blocks."]
        prefixes.append("final_norm.")
        return tuple(
            key
            for key in self.state_dict()
            if key == "aggregate" or key.startswith(tuple(prefixes))
        )

    def _probe_layers(self, d: int) -> None:
        self.threat_num = nn.Linear(len(self.layout.thr_num), d)
        self.threat_row = nn.Linear(5 * d, d)
        self.threat_pool = nn.Linear(2 * d, d)
        self.foe_num = nn.Linear(len(self.layout.foe_num), d)
        self.foe_row = nn.Linear(3 * d, d)
        self.foe_pool = nn.Linear(2 * d, d)

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

    def _probe_summaries(
        self, batch: Batch, base: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The ``probe-damage/2`` rows pooled per token: [B, M, d] to add to a token, and [B, M, d]
        per own token for its Mega variant (used by a Mega candidate's slot part).

        A threat row is the attacking foe's token (``base``, the token before any summary), its
        move, its kind and its numbers, with the kind of the foe move's priority row; the rows into
        one Pokémon are pooled by masked sum and max. A foe's priority rows are pooled onto the foe.
        """
        if (
            batch.thr_idx is None
            or batch.thr_cat is None
            or batch.thr_num is None
            or batch.foe_idx is None
            or batch.foe_cat is None
            or batch.foe_num is None
        ):
            raise ValueError("this network reads probe-damage/2 rows; the batch has none")
        if batch.thr_cat.shape[-1] != len(self.layout.thr_cat) or batch.foe_cat.shape[-1] != len(
            self.layout.foe_cat
        ):
            raise ValueError("the batch's probe rows are not this network's layout")
        size = base.shape[1]
        threat = torch.cat(
            [
                _gather(base, batch.thr_idx[..., 1]),
                self.embed("move", batch.thr_cat[..., 0]),
                self.embed("probe_kind", batch.thr_cat[..., 1]),
                self.embed("probe_kind", batch.thr_cat[..., 2]),
                self.threat_num(batch.thr_num),
            ],
            -1,
        )
        threat = torch.nn.functional.gelu(self.threat_row(threat))
        real = batch.thr_idx[..., 0] >= 0
        mega = batch.thr_idx[..., 2] == 1
        plain, some = _pool(threat, batch.thr_idx[..., 0], real & ~mega, size)
        flipped, some_mega = _pool(threat, batch.thr_idx[..., 0], real & mega, size)
        foe = torch.cat(
            [
                self.embed("move", batch.foe_cat[..., 0]),
                self.embed("probe_kind", batch.foe_cat[..., 1]),
                self.foe_num(batch.foe_num),
            ],
            -1,
        )
        foe = torch.nn.functional.gelu(self.foe_row(foe))
        pooled, some_foe = _pool(foe, batch.foe_idx[..., 0], batch.foe_idx[..., 0] >= 0, size)
        zero = torch.zeros_like(base)
        added = torch.where(some, self.threat_pool(plain), zero) + torch.where(
            some_foe, self.foe_pool(pooled), zero
        )
        return added, torch.where(some_mega, self.threat_pool(flipped), zero)

    def _battle_candidates(
        self, batch: Batch, mons: torch.Tensor, mega: torch.Tensor | None = None
    ) -> torch.Tensor:
        if batch.cand_tok.shape[-1] != len(self.layout.cand_tok):
            raise ValueError("the batch's candidate tokens are not this network's layout")
        if batch.cand_cat.shape[-1] != len(self.layout.cand_cat):
            raise ValueError("the batch's candidate columns are not this network's layout")
        slots = []
        for s in range(SLOTS):
            cat, tok = batch.cand_cat[:, :, s], batch.cand_tok[:, :, s]
            extra = [
                self.embed(kind, cat[..., BASE_CAND_COLUMNS + i])
                for i, (_, kind) in enumerate(self.layout.cand_cat[BASE_CAND_COLUMNS:])
            ]
            parts = [
                self.embed("action", cat[..., 0]),
                self.embed("move", cat[..., 1]),
                self.embed("move_slot", cat[..., 2]),
                self.embed("target", cat[..., 3]),
                self.embed("event", cat[..., 4]),
                _gather(mons, tok[..., 0]),
                _gather(mons, tok[..., 1]),
                *extra,
            ]
            if self.targeted:
                parts.append(_gather(mons, tok[..., BASE_CAND_TOKENS]))
            if self.cand_features is not None:
                if batch.cand_num is None:
                    raise ValueError("this network reads candidate features; the batch has none")
                parts.append(self.cand_features(batch.cand_num[:, :, s]))
            if mega is not None:  # a Mega part sees how the Evolution changes what comes in
                evolves = cat[..., 4] == 1 + EVENTS.index("mega")
                parts.append(_gather(mega, tok[..., 0]) * evolves[..., None])
            slots.append(self.slot_parts[s](torch.cat(parts, -1)))
        out: torch.Tensor = self.candidate(torch.cat(slots, -1))
        return out

    def _preview_candidates(
        self, batch: Batch, mons: torch.Tensor, mega: torch.Tensor | None = None
    ) -> torch.Tensor:
        picked = _gather(mons, batch.team_tok)  # [B, N, P, d]
        places = self.place.weight.expand_as(picked)
        out: torch.Tensor = self.candidate(torch.cat([places, picked], -1).flatten(2))
        return out

    def _factored_logits(self, batch: Batch, mons: torch.Tensor) -> torch.Tensor:
        """sum over places p of u(place p, token p) + v(tokens at places 1 and 2); [B, N]."""
        picked = _gather(mons, batch.team_tok)  # [B, N, P, d]
        places = self.place.weight.expand_as(picked)
        places_total = self.place_score(torch.cat([places, picked], -1)).squeeze(-1).sum(-1)
        lead = self.lead_score(torch.cat([picked[:, :, 0], picked[:, :, 1]], -1)).squeeze(-1)
        total: torch.Tensor = places_total + lead
        return total

    # -- forward -------------------------------------------------------------

    def _trunk(
        self, batch: Batch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """The shared transformer: (x, mask, state, mons, mega), x [B, 2 + M, d]."""
        if batch.kind != self.role:
            raise ValueError(f"a {self.role} network cannot score a {batch.kind} batch")
        b = batch.size
        d = self.config.d_model
        side = self.side((~batch.own_mask).long())
        mons = self.mon_cat(self._columns(self.layout.mon_cat, batch.mon_cat))
        mons = mons + self.mon_num(batch.mon_num) + side
        mega = None
        if self.probing:
            added, mega = self._probe_summaries(batch, mons)
            mons = mons + added
        field_token = self.field_cat(self._columns(self.layout.field_cat, batch.field_cat))
        field_token = field_token + self.field_num(batch.field_num)
        aggregate = self.aggregate.expand(b, d)
        x = torch.cat([aggregate[:, None], field_token[:, None], mons], dim=1)
        fixed = torch.ones(b, 2, dtype=torch.bool, device=batch.token_mask.device)
        mask = torch.cat([fixed, batch.token_mask], dim=1)
        for block in self.blocks:
            x = block(x, mask)
        x = self.final_norm(x)
        return x, mask, x[:, 0], x[:, 2:], mega

    def _value(self, batch: Batch, state: torch.Tensor) -> torch.Tensor:
        if self.value_net is not None:  # separate_value "network": its own trunk, same inputs
            return self.value_net.value_of(batch)
        if self.value_mode == SEPARATE_VALUE_DETACHED:  # no value gradient reaches the trunk
            state = state.detach()
        value: torch.Tensor = self.value(state).squeeze(-1)
        return value

    def value_of(self, batch: Batch) -> torch.Tensor:
        """The value head alone, for the separate value network (``value_only``)."""
        value: torch.Tensor = self.value(self._trunk(batch)[2]).squeeze(-1)
        return value

    def forward(self, batch: Batch) -> PolicyOutput:
        x, mask, state, mons, mega = self._trunk(batch)
        value = self._value(batch, state)
        if self.factored:
            logits = self._factored_logits(batch, mons)
            logits = logits.masked_fill(~batch.cand_mask, torch.finfo(logits.dtype).min)
            return PolicyOutput(logits=logits, value=value)
        build = self._battle_candidates if self.role == "battle" else self._preview_candidates
        c = build(batch, mons, mega) + self.pending(batch.pending[..., None])
        c = c + self.cross(self.norm_candidate(c), self.norm_memory(x), mask)
        agg = state[:, None].expand_as(c)
        logits = self.score(torch.cat([c, agg, c * agg], -1)).squeeze(-1)
        logits = logits.masked_fill(~batch.cand_mask, torch.finfo(logits.dtype).min)
        return PolicyOutput(logits=logits, value=value)


class TrunkError(ValueError):
    """The Preview and Battle networks of a ``shared_trunk`` pair do not share a trunk."""


def shared_trunk_of(config: ModelConfig) -> bool:
    return is_set(config.options, SHARED_TRUNK)


def tie_trunk(shared: PolicyNetwork, other: PolicyNetwork) -> None:
    """Make ``other`` use ``shared``'s trunk modules (the very same parameters), so one optimiser
    step on either network moves both. ``other``'s own freshly initialised trunk is dropped."""
    if shared.config != other.config:
        raise TrunkError("a shared trunk needs the same model config on both networks")
    kinds = shared.trunk_kinds()
    for kind in kinds:
        other.embeddings[kind] = shared.embeddings[kind]
        if kind in shared.attribute_projections:
            other.attribute_projections[kind] = shared.attribute_projections[kind]
    for name in ("mon_cat", "mon_num", "side", "field_cat", "field_num", "blocks", "final_norm"):
        setattr(other, name, getattr(shared, name))
    other.aggregate = shared.aggregate


def check_shared_trunk(networks: Mapping[str, PolicyNetwork]) -> None:
    """Refuse a pair that does not agree on ``shared_trunk``, or whose trunk tensors differ.

    A pair made without the option is not looked at. Called wherever a pair is loaded (the worker's
    scorer) or restored (the learner state)."""
    flags = {role: shared_trunk_of(net.config) for role, net in networks.items()}
    if len(set(flags.values())) > 1:
        raise TrunkError("only one of the two networks was made with shared_trunk")
    if not all(flags.values()):
        return
    battle, preview = networks["battle"], networks["preview"]
    battle_state, preview_state = battle.state_dict(), preview.state_dict()
    for key in battle.trunk_keys():
        if not torch.equal(battle_state[key], preview_state[key]):
            raise TrunkError("the shared trunk tensors of the two networks differ")


def init_network(role: str, game: GameData, config: ModelConfig, seed: int) -> PolicyNetwork:
    """A freshly initialised network; the same seed gives the same weights on one machine.

    Same machine and same software only: no claim is made across instruction sets
    or torch builds. The global RNG is left as it was.
    """
    if not 0 <= seed <= MAX_SEED:
        raise ConfigError(f"seed must be between 0 and {MAX_SEED}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return PolicyNetwork(role, game, config)
