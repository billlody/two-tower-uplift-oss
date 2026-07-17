"""Ranking loss for uplift-score ordering (port of the production pairwise loss).

Encourages correct ordering of device uplift scores, closing the gap between
pointwise MSE distillation and ranking-based evaluation (AUCC/Qini).
"""

from __future__ import annotations

import torch


def pairwise_ranking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    margin: float = 0.1,
    num_pairs: int = 16,
) -> torch.Tensor:
    """Batch-level logistic pairwise ranking loss.

    For each item ``i`` sample ``num_pairs`` partners ``j`` from the batch and
    penalize mis-ordered pairs relative to the (detached) target ordering.

    Parameters
    ----------
    pred : Tensor[B] or Tensor[B, 1]
        Predicted uplift scores.
    target : Tensor[B] or Tensor[B, 1]
        Target scores (detached), same shape as ``pred``.
    margin : float
        Soft margin added to the score difference.
    num_pairs : int
        Random partners sampled per item.

    Returns
    -------
    Tensor
        Scalar loss.
    """
    pred = pred.view(-1)
    target = target.view(-1)
    b = pred.shape[0]
    if b < 2:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    j_idx = torch.randint(0, b, (b, num_pairs), device=pred.device)
    pred_diff = pred.unsqueeze(1) - pred[j_idx]
    target_sign = torch.sign(target.unsqueeze(1) - target[j_idx])
    loss = torch.log1p(torch.exp(-target_sign * (pred_diff - margin * target_sign)))
    return loss.mean()
