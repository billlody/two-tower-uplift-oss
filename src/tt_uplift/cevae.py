"""CEVAE baseline: Causal Effect Variational Autoencoder (Louizos et al., 2017).

A compact, single-machine port of CEVAE for the offline baseline comparison.
CEVAE posits a **latent confounder** ``z`` that generates the observed features
``x``, the treatment ``t``, and the outcome ``y``.  Estimating the treatment
effect then reduces to intervening on ``t`` in the learned generative model::

    p(z)                      prior over the latent confounder
    p(x | z)                  feature decoder
    p(t | z)                  treatment (propensity) decoder
    p(y | z, t)               per-arm outcome decoders

Inference network (encoder) approximates the posterior ``q(z | x, t, y)``.  Two
auxiliary networks ``q(t | x)`` and ``q(y | x, t)`` provide the ``t, y`` needed
to encode ``z`` at *test* time, when the true outcome is unavailable (matching
the original paper's Fig. 1 auxiliary distributions).

Outcome is continuous here (the stratified-normalized engagement), so
``p(y | z, t)`` and ``p(x | z)`` are Gaussian with fixed unit variance (their
negative log-likelihoods reduce to MSE); ``p(t | z)`` and ``q(t | x)`` are
Bernoulli (BCE).  Uplift for a sample is the do-operator contrast::

    tau(x) = E[y | z, do(t = 1)] - E[y | z, do(t = 0)],  z ~ q(z | x, t_hat, y_hat)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch
import torch.nn as nn

from .model import CategoricalEmbedding, _mlp


@dataclass
class CEVAEConfig:
    """Hyper-parameters for :class:`CEVAE`."""

    numeric_dim: int = 0
    cat_cardinalities: Dict[str, int] = field(default_factory=dict)
    cat_embed_dim: int = 8
    latent_dim: int = 16
    hidden: List[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.0
    use_batchnorm: bool = False


class CEVAE(nn.Module):
    """Causal Effect VAE with a latent confounder (see module docstring)."""

    def __init__(self, cfg: CEVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.cat_embed = CategoricalEmbedding(cfg.cat_cardinalities, cfg.cat_embed_dim)
        x_dim = cfg.numeric_dim + self.cat_embed.output_dim
        self.x_dim = x_dim
        z, h, do, bn = cfg.latent_dim, cfg.hidden, cfg.dropout, cfg.use_batchnorm

        # Inference network q(z | x, t, y): input is [x, t, y].
        self.enc = _mlp(x_dim + 2, h, 2 * z, do, bn)

        # Auxiliary posteriors used at test time (no y observed).
        self.aux_t = _mlp(x_dim, h, 1, do, bn)  # q(t | x)
        self.aux_y0 = _mlp(x_dim, h, 1, do, bn)  # q(y | x, t=0)
        self.aux_y1 = _mlp(x_dim, h, 1, do, bn)  # q(y | x, t=1)

        # Generative decoders.
        self.dec_x = _mlp(z, h, x_dim, do, bn)  # p(x | z)
        self.dec_t = _mlp(z, h, 1, do, bn)  # p(t | z)
        self.dec_y0 = _mlp(z, h, 1, do, bn)  # p(y | z, t=0)
        self.dec_y1 = _mlp(z, h, 1, do, bn)  # p(y | z, t=1)

    def _x(self, numeric: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        return torch.cat([numeric, self.cat_embed(cat)], dim=1)

    @staticmethod
    def _reparam(mu: torch.Tensor, logvar: torch.Tensor, sample: bool) -> torch.Tensor:
        if not sample:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def encode(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.enc(torch.cat([x, t, y], dim=1)).chunk(2, dim=1)
        return mu, logvar

    def decode_y(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dec_y0(z), self.dec_y1(z)

    def forward(
        self,
        numeric: torch.Tensor,
        cat: torch.Tensor,
        treatment_bin: torch.Tensor,
        outcome: torch.Tensor,
        sample: bool = True,
    ) -> Dict[str, torch.Tensor]:
        x = self._x(numeric, cat)
        t = treatment_bin.view(-1, 1)
        mu, logvar = self.encode(x, t, outcome.view(-1, 1))
        z = self._reparam(mu, logvar, sample)

        y0, y1 = self.decode_y(z)
        return {
            "x": x,
            "z_mu": mu,
            "z_logvar": logvar,
            "x_rec": self.dec_x(z),
            "t_logit": self.dec_t(z),
            "y0": y0,
            "y1": y1,
            "y_rec": t * y1 + (1 - t) * y0,
            # Auxiliary predictions (trained toward observed t, y).
            "aux_t_logit": self.aux_t(x),
            "aux_y0": self.aux_y0(x),
            "aux_y1": self.aux_y1(x),
        }

    @torch.no_grad()
    def predict_uplift(self, numeric: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        """Do-operator uplift (probability contrast) via auxiliary nets to encode z.

        The outcome is the binary engagement label, so decoder heads emit logits
        and uplift is ``P(y=1 | do(t=1)) - P(y=1 | do(t=0))``.
        """
        x = self._x(numeric, cat)
        t_hat = torch.sigmoid(self.aux_t(x))
        # Auxiliary y-heads emit logits; feed the expected probability to the encoder.
        y_hat = t_hat * torch.sigmoid(self.aux_y1(x)) + (1 - t_hat) * torch.sigmoid(self.aux_y0(x))
        mu, _ = self.encode(x, t_hat, y_hat)  # posterior mean as point estimate
        y0, y1 = self.decode_y(mu)
        # Logit contrast, consistent with the two-tower S-learner uplift.
        return (y1 - y0).view(-1)
