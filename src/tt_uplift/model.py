"""Two-tower S-learner uplift model with device-uplift distillation.

Minimal, single-machine port of the production model.  Drops DCN-V2, sequence
transformers, bucket/ID embeddings, and the distributed machinery — keeps the
core causal architecture that the paper's arguments rely on:

* a **device tower** and a **content tower** (MLPs) producing embeddings ``d, c``;
* an **S-learner outcome head** over ``[d, c, d*c, d*t', c*t', t]`` with explicit
  treatment interactions (heterogeneous effects);
* **session uplift** via counterfactual contrast ``y(t_high) - y(t_low)``;
* a **device uplift head** ``g(d)`` distilled from session uplift — the only part
  needed at serving time;
* optional **DoubleML** treatment-residualization head (Frisch-Waugh-Lovell).

Also provides two published NN baselines that share the tower scaffolding:

* :class:`TARNet` — shared representation + per-arm outcome heads;
* :class:`DragonNet` — TARNet + a propensity head.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn


def _mlp(input_dim: int, hidden: List[int], out_dim: int, dropout: float, bn: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        if bn:
            layers.append(nn.BatchNorm1d(h))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class CategoricalEmbedding(nn.Module):
    """Concatenated learnable embeddings for a set of categorical features."""

    def __init__(self, cardinalities: Dict[str, int], embed_dim: int = 8):
        super().__init__()
        self.feature_names = list(cardinalities.keys())
        self.embeddings = nn.ModuleDict(
            {name: nn.Embedding(cardinalities[name] + 1, embed_dim, padding_idx=0) for name in self.feature_names}
        )
        self.output_dim = len(self.feature_names) * embed_dim

    def forward(self, cat_indices: torch.Tensor) -> torch.Tensor:
        if cat_indices.shape[1] == 0:
            return torch.zeros(cat_indices.shape[0], 0, device=cat_indices.device)
        idx = cat_indices.long()
        return torch.cat([self.embeddings[name](idx[:, i]) for i, name in enumerate(self.feature_names)], dim=-1)


@dataclass
class TwoTowerConfig:
    """Hyper-parameters for :class:`TwoTowerUpliftModel`."""

    device_numeric_dim: int = 0
    content_numeric_dim: int = 0
    treatment_dim: int = 1

    device_cat_cardinalities: Dict[str, int] = field(default_factory=dict)
    content_cat_cardinalities: Dict[str, int] = field(default_factory=dict)

    cat_embed_dim: int = 8
    tower_embed_dim: int = 32

    device_hidden: List[int] = field(default_factory=lambda: [64, 32])
    content_hidden: List[int] = field(default_factory=lambda: [32, 16])
    outcome_hidden: List[int] = field(default_factory=lambda: [64, 32])
    device_uplift_hidden: List[int] = field(default_factory=lambda: [16])

    dropout: float = 0.1
    use_batchnorm: bool = True

    # DoubleML (Frisch-Waugh-Lovell partialling-out)
    use_double_ml: bool = False
    treatment_pred_hidden: List[int] = field(default_factory=lambda: [32])


class TwoTowerUpliftModel(nn.Module):
    """Two-tower S-learner uplift model (see module docstring)."""

    def __init__(self, cfg: TwoTowerConfig):
        super().__init__()
        self.cfg = cfg

        self.device_cat_embed = CategoricalEmbedding(cfg.device_cat_cardinalities, cfg.cat_embed_dim)
        self.content_cat_embed = CategoricalEmbedding(cfg.content_cat_cardinalities, cfg.cat_embed_dim)

        device_input_dim = cfg.device_numeric_dim + self.device_cat_embed.output_dim
        content_input_dim = cfg.content_numeric_dim + self.content_cat_embed.output_dim

        self.device_tower = _mlp(device_input_dim, cfg.device_hidden, cfg.tower_embed_dim, cfg.dropout, cfg.use_batchnorm)
        self.content_tower = _mlp(content_input_dim, cfg.content_hidden, cfg.tower_embed_dim, cfg.dropout, cfg.use_batchnorm)

        self.treatment_proj = nn.Sequential(nn.Linear(cfg.treatment_dim, cfg.tower_embed_dim), nn.ReLU())

        outcome_input_dim = cfg.tower_embed_dim * 3 + cfg.tower_embed_dim * 2 + cfg.treatment_dim
        self.outcome_head = _mlp(outcome_input_dim, cfg.outcome_hidden, 1, cfg.dropout, cfg.use_batchnorm)

        # Device uplift head: simple, no BN/Dropout.
        uplift_layers: list[nn.Module] = []
        prev = cfg.tower_embed_dim
        for h in cfg.device_uplift_hidden:
            uplift_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        uplift_layers.append(nn.Linear(prev, 1))
        self.device_uplift_head = nn.Sequential(*uplift_layers)

        if cfg.use_double_ml:
            self.treatment_pred_head = _mlp(
                cfg.tower_embed_dim * 3, cfg.treatment_pred_hidden, cfg.treatment_dim, cfg.dropout, cfg.use_batchnorm
            )
        else:
            self.treatment_pred_head = None

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
                if m.padding_idx is not None:
                    nn.init.zeros_(m.weight[m.padding_idx])

    def _encode_device(self, numeric: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        return self.device_tower(torch.cat([numeric, self.device_cat_embed(cat)], dim=1))

    def _encode_content(self, numeric: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        return self.content_tower(torch.cat([numeric, self.content_cat_embed(cat)], dim=1))

    def forward(
        self,
        device_numeric: torch.Tensor,
        device_cat: torch.Tensor,
        content_numeric: torch.Tensor,
        content_cat: torch.Tensor,
        treatment: torch.Tensor,
        t_high: Optional[Sequence[float]] = None,
        t_low: Optional[Sequence[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        d = self._encode_device(device_numeric, device_cat)
        c = self._encode_content(content_numeric, content_cat)
        interaction = d * c

        # DoubleML residualization: remove confounder-driven treatment allocation.
        t_hat = None
        if self.treatment_pred_head is not None:
            t_hat = self.treatment_pred_head(torch.cat([d, c, interaction], dim=1))
            t_residual = treatment - t_hat.detach()
        else:
            t_residual = treatment

        t_proj = self.treatment_proj(t_residual)
        combined = torch.cat([d, c, interaction, d * t_proj, c * t_proj, t_residual], dim=1)
        y_hat = self.outcome_head(combined)

        # S-learner counterfactual with continuous treatment.
        with torch.no_grad():
            t_hi = (
                torch.tensor(t_high, device=treatment.device, dtype=treatment.dtype).unsqueeze(0).expand_as(treatment)
                if t_high is not None
                else torch.ones_like(treatment)
            )
            t_lo = (
                torch.tensor(t_low, device=treatment.device, dtype=treatment.dtype).unsqueeze(0).expand_as(treatment)
                if t_low is not None
                else -torch.ones_like(treatment)
            )
        if t_hat is not None:
            t_hi_res = t_hi - t_hat.detach()
            t_lo_res = t_lo - t_hat.detach()
        else:
            t_hi_res, t_lo_res = t_hi, t_lo

        y_hi = self.outcome_head(
            torch.cat([d, c, interaction, d * self.treatment_proj(t_hi_res), c * self.treatment_proj(t_hi_res), t_hi_res], dim=1)
        )
        y_lo = self.outcome_head(
            torch.cat([d, c, interaction, d * self.treatment_proj(t_lo_res), c * self.treatment_proj(t_lo_res), t_lo_res], dim=1)
        )
        session_uplift = y_hi - y_lo

        device_uplift = self.device_uplift_head(d)

        out = {
            "y_hat": y_hat,
            "device_uplift": device_uplift,
            "device_embed": d,
            "session_uplift": session_uplift,
        }
        if t_hat is not None:
            out["treatment_pred"] = t_hat
        return out

    @torch.no_grad()
    def predict_device_uplift(self, device_numeric: torch.Tensor, device_cat: torch.Tensor) -> torch.Tensor:
        """Serving path: device-level uplift from device features only."""
        return self.device_uplift_head(self._encode_device(device_numeric, device_cat))


# ---------------------------------------------------------------------------
# Baselines: TARNet and DragonNet (shared representation, per-arm heads)
# ---------------------------------------------------------------------------


@dataclass
class BaselineConfig:
    """Config for :class:`TARNet` / :class:`DragonNet` baselines."""

    numeric_dim: int = 0
    cat_cardinalities: Dict[str, int] = field(default_factory=dict)
    cat_embed_dim: int = 8
    repr_dim: int = 32
    repr_hidden: List[int] = field(default_factory=lambda: [64, 32])
    head_hidden: List[int] = field(default_factory=lambda: [32])
    dropout: float = 0.1
    use_batchnorm: bool = True


class TARNet(nn.Module):
    """Treatment-Agnostic Representation Network (Shalit et al., 2017).

    A shared representation ``phi(x)`` feeds two outcome heads, one per treatment
    arm.  Uplift = ``head_1(phi) - head_0(phi)``.  Here ``x`` is the concatenation
    of device + content features (no towers — a single representation).
    """

    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg = cfg
        self.cat_embed = CategoricalEmbedding(cfg.cat_cardinalities, cfg.cat_embed_dim)
        in_dim = cfg.numeric_dim + self.cat_embed.output_dim
        self.repr = _mlp(in_dim, cfg.repr_hidden, cfg.repr_dim, cfg.dropout, cfg.use_batchnorm)
        self.head0 = _mlp(cfg.repr_dim, cfg.head_hidden, 1, cfg.dropout, cfg.use_batchnorm)
        self.head1 = _mlp(cfg.repr_dim, cfg.head_hidden, 1, cfg.dropout, cfg.use_batchnorm)

    def _phi(self, numeric: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        return self.repr(torch.cat([numeric, self.cat_embed(cat)], dim=1))

    def forward(self, numeric: torch.Tensor, cat: torch.Tensor, treatment_bin: torch.Tensor) -> Dict[str, torch.Tensor]:
        phi = self._phi(numeric, cat)
        y0, y1 = self.head0(phi), self.head1(phi)
        t = treatment_bin.view(-1, 1)
        y_hat = t * y1 + (1 - t) * y0
        return {"y_hat": y_hat, "y0": y0, "y1": y1, "uplift": y1 - y0, "repr": phi}


class DragonNet(TARNet):
    """DragonNet (Shi et al., 2019): TARNet + a propensity head on the shared rep."""

    def __init__(self, cfg: BaselineConfig):
        super().__init__(cfg)
        self.propensity_head = nn.Sequential(nn.Linear(cfg.repr_dim, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, numeric: torch.Tensor, cat: torch.Tensor, treatment_bin: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = super().forward(numeric, cat, treatment_bin)
        out["propensity_logit"] = self.propensity_head(out["repr"])
        return out
