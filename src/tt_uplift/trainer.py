"""Compact single-machine trainer for the two-tower uplift model and baselines.

Implements the combined objective from the paper::

    L = outcome_MSE + alpha * distillation_MSE + beta * ranking_loss (+ DoubleML)

where distillation trains the device uplift head to reproduce the (stop-gradient)
session-level uplift.  No distributed/AMP/checkpoint machinery — just enough to
train the small synthetic models deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .features import Tensors
from .losses import pairwise_ranking_loss
from .model import DragonNet, TARNet, TwoTowerUpliftModel


@dataclass
class TrainConfig:
    """Training hyper-parameters."""

    epochs: int = 20
    batch_size: int = 2048
    lr: float = 1e-3
    weight_decay: float = 1e-5
    alpha_distill: float = 1.0  # distillation weight (session -> device)
    beta_ranking: float = 0.0  # ranking loss weight (0 = off)
    ranking_num_pairs: int = 16
    ranking_margin: float = 0.1
    alpha_treatment_pred: float = 1.0  # DoubleML treatment-prediction weight
    t_high: float = 1.0
    t_low: float = -1.0
    seed: int = 0
    device: str = "cpu"


def set_seed(seed: int) -> None:
    """Set numpy + torch seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def _iter_batches(n: int, batch_size: int, rng: np.random.Generator):
    idx = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield idx[start : start + batch_size]


def train_two_tower(model: TwoTowerUpliftModel, data: Tensors, cfg: TrainConfig) -> TwoTowerUpliftModel:
    """Train the two-tower uplift model with outcome + distillation (+ optional) losses.

    Returns the trained model (in-place, also returned for convenience).
    """
    set_seed(cfg.seed)
    dev = torch.device(cfg.device)
    model.to(dev).train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed)
    n = data.outcome.shape[0]
    t_high, t_low = [cfg.t_high], [cfg.t_low]

    for _ in range(cfg.epochs):
        for bidx in _iter_batches(n, cfg.batch_size, rng):
            b = torch.as_tensor(bidx, dtype=torch.long)
            out = model(
                data.device_numeric[b].to(dev),
                data.device_cat[b].to(dev),
                data.content_numeric[b].to(dev),
                data.content_cat[b].to(dev),
                data.treatment[b].to(dev),
                t_high=t_high,
                t_low=t_low,
            )
            y = data.outcome[b].to(dev)
            loss = F.mse_loss(out["y_hat"], y)

            # Distillation: device uplift head -> stop-grad session uplift.
            target = out["session_uplift"].detach()
            loss = loss + cfg.alpha_distill * F.mse_loss(out["device_uplift"], target)

            if cfg.beta_ranking > 0:
                loss = loss + cfg.beta_ranking * pairwise_ranking_loss(
                    out["device_uplift"], target, margin=cfg.ranking_margin, num_pairs=cfg.ranking_num_pairs
                )

            if "treatment_pred" in out:
                loss = loss + cfg.alpha_treatment_pred * F.mse_loss(out["treatment_pred"], data.treatment[b].to(dev))

            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    return model


def train_baseline(model: TARNet | DragonNet, data: Tensors, cfg: TrainConfig, is_dragonnet: bool = False) -> TARNet:
    """Train a TARNet/DragonNet baseline (factual outcome loss + optional propensity).

    ``x`` is the concatenation of device + content features; the treatment used is
    the binary indicator.
    """
    set_seed(cfg.seed)
    dev = torch.device(cfg.device)
    model.to(dev).train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed)
    n = data.outcome.shape[0]

    numeric = torch.cat([data.device_numeric, data.content_numeric], dim=1)
    cat = torch.cat([data.device_cat, data.content_cat], dim=1)

    for _ in range(cfg.epochs):
        for bidx in _iter_batches(n, cfg.batch_size, rng):
            b = torch.as_tensor(bidx, dtype=torch.long)
            out = model(numeric[b].to(dev), cat[b].to(dev), data.treatment_binary[b].to(dev))
            loss = F.mse_loss(out["y_hat"], data.outcome[b].to(dev))
            if is_dragonnet and "propensity_logit" in out:
                loss = loss + 0.1 * F.binary_cross_entropy_with_logits(
                    out["propensity_logit"], data.treatment_binary[b].to(dev)
                )
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    return model


@torch.no_grad()
def predict_session_uplift_two_tower(model: TwoTowerUpliftModel, data: Tensors, cfg: TrainConfig) -> np.ndarray:
    """Session-level uplift from the outcome-head counterfactual contrast."""
    dev = torch.device(cfg.device)
    out = model(
        data.device_numeric.to(dev),
        data.device_cat.to(dev),
        data.content_numeric.to(dev),
        data.content_cat.to(dev),
        data.treatment.to(dev),
        t_high=[cfg.t_high],
        t_low=[cfg.t_low],
    )
    return out["session_uplift"].cpu().numpy().ravel()


@torch.no_grad()
def predict_device_uplift_two_tower(model: TwoTowerUpliftModel, data: Tensors) -> np.ndarray:
    """Device-level uplift from the distilled head (serving path)."""
    return model.predict_device_uplift(data.device_numeric, data.device_cat).cpu().numpy().ravel()


@torch.no_grad()
def predict_uplift_baseline(model: TARNet, data: Tensors) -> np.ndarray:
    """Session-level uplift from a TARNet/DragonNet baseline (head1 - head0)."""
    numeric = torch.cat([data.device_numeric, data.content_numeric], dim=1)
    cat = torch.cat([data.device_cat, data.content_cat], dim=1)
    out = model(numeric, cat, data.treatment_binary)
    return out["uplift"].cpu().numpy().ravel()
