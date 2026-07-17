"""Uplift evaluation: AUCC / normalized Qini (pure-NumPy port of production).

The AUCC (Area Under the Cumulative gain Curve, a.k.a. Qini coefficient) ranks
samples by predicted uplift, walks down the ranking, and integrates the treated-
minus-control gain.  Normalized Qini rescales against oracle and random baselines.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def compute_aucc(
    uplift_score: np.ndarray,
    outcome: np.ndarray,
    treatment: np.ndarray,
    normalize: bool = True,
    num_buckets: int = 1000,
) -> float:
    """Area Under the Cumulative gain Curve (Qini coefficient).

    Parameters
    ----------
    uplift_score : ndarray[N]
        Predicted uplift (higher = larger expected positive effect).
    outcome : ndarray[N]
        Observed outcome.
    treatment : ndarray[N]
        Binary treatment indicator (1 = treated, 0 = control).
    normalize : bool
        If True, divide the integrated area by ``N**2``.
    num_buckets : int
        Number of equal-size buckets along the ranking.

    Returns
    -------
    float
        AUCC (``nan`` if either arm is empty).
    """
    uplift_score = np.asarray(uplift_score, dtype=np.float64).ravel()
    outcome = np.asarray(outcome, dtype=np.float64).ravel()
    treatment = np.asarray(treatment, dtype=np.float64).ravel()

    finite = np.isfinite(uplift_score) & np.isfinite(outcome)
    uplift_score, outcome, treatment = uplift_score[finite], outcome[finite], treatment[finite]

    n_treat = int(np.sum(treatment == 1))
    n_ctrl = int(np.sum(treatment == 0))
    if n_treat == 0 or n_ctrl == 0:
        return float("nan")

    total_n = len(uplift_score)
    num_buckets = min(num_buckets, total_n)

    desc = np.argsort(-uplift_score)
    bucket_ids = np.empty(total_n, dtype=np.int64)
    bucket_ids[desc] = np.arange(total_n) * num_buckets // total_n

    treat_mask = treatment == 1
    ctrl_mask = treatment == 0

    b_treat_y = np.bincount(bucket_ids[treat_mask], weights=outcome[treat_mask], minlength=num_buckets)
    b_ctrl_y = np.bincount(bucket_ids[ctrl_mask], weights=outcome[ctrl_mask], minlength=num_buckets)
    b_treat_n = np.bincount(bucket_ids[treat_mask], minlength=num_buckets).astype(np.float64)
    b_ctrl_n = np.bincount(bucket_ids[ctrl_mask], minlength=num_buckets).astype(np.float64)
    b_width = np.bincount(bucket_ids, minlength=num_buckets).astype(np.float64)

    cum_treat_y = np.cumsum(b_treat_y)
    cum_ctrl_y = np.cumsum(b_ctrl_y)
    cum_treat_n = np.cumsum(b_treat_n)
    cum_ctrl_n = np.cumsum(b_ctrl_n)

    mean_ctrl_prefix = np.divide(cum_ctrl_y, cum_ctrl_n, out=np.zeros_like(cum_ctrl_y), where=cum_ctrl_n > 0)
    gain = cum_treat_y - cum_treat_n * mean_ctrl_prefix

    prev_gain = np.concatenate([[0.0], gain[:-1]])
    area = float(np.sum(((gain + prev_gain) / 2.0) * b_width))

    total_f = float(total_n)
    return area / (total_f * total_f) if normalize else area


def normalized_qini(
    uplift_score: np.ndarray,
    outcome: np.ndarray,
    treatment: np.ndarray,
    num_buckets: int = 1000,
    seed: int = 0,
) -> float:
    """Normalized Qini: ``(AUCC_model - AUCC_random) / (AUCC_oracle - AUCC_random)``.

    The oracle ranks by ``(2*t - 1) * outcome`` (perfect treated-high/control-low
    ordering); random ranks by uniform noise.

    Returns
    -------
    float
        Normalized Qini (``nan`` if denominator degenerate).
    """
    model = compute_aucc(uplift_score, outcome, treatment, num_buckets=num_buckets)
    oracle_score = (2.0 * np.asarray(treatment, dtype=np.float64).ravel() - 1.0) * np.asarray(
        outcome, dtype=np.float64
    ).ravel()
    oracle = compute_aucc(oracle_score, outcome, treatment, num_buckets=num_buckets)
    rng = np.random.default_rng(seed)
    rand = compute_aucc(rng.random(len(outcome)), outcome, treatment, num_buckets=num_buckets)

    denom = oracle - rand
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return float("nan")
    return float((model - rand) / denom)


def uplift_curve(
    uplift_score: np.ndarray,
    outcome: np.ndarray,
    treatment: np.ndarray,
    num_buckets: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(frac_targeted, normalized_gain)`` arrays for plotting."""
    uplift_score = np.asarray(uplift_score, dtype=np.float64).ravel()
    outcome = np.asarray(outcome, dtype=np.float64).ravel()
    treatment = np.asarray(treatment, dtype=np.float64).ravel()
    total_n = len(uplift_score)
    num_buckets = min(num_buckets, total_n)

    desc = np.argsort(-uplift_score)
    bucket_ids = np.empty(total_n, dtype=np.int64)
    bucket_ids[desc] = np.arange(total_n) * num_buckets // total_n
    treat_mask = treatment == 1
    ctrl_mask = treatment == 0

    b_treat_y = np.bincount(bucket_ids[treat_mask], weights=outcome[treat_mask], minlength=num_buckets)
    b_ctrl_y = np.bincount(bucket_ids[ctrl_mask], weights=outcome[ctrl_mask], minlength=num_buckets)
    b_treat_n = np.bincount(bucket_ids[treat_mask], minlength=num_buckets).astype(np.float64)
    b_ctrl_n = np.bincount(bucket_ids[ctrl_mask], minlength=num_buckets).astype(np.float64)
    b_width = np.bincount(bucket_ids, minlength=num_buckets).astype(np.float64)

    cum_treat_y = np.cumsum(b_treat_y)
    cum_ctrl_y = np.cumsum(b_ctrl_y)
    cum_treat_n = np.cumsum(b_treat_n)
    cum_ctrl_n = np.cumsum(b_ctrl_n)
    mean_ctrl_prefix = np.divide(cum_ctrl_y, cum_ctrl_n, out=np.zeros_like(cum_ctrl_y), where=cum_ctrl_n > 0)
    gain = cum_treat_y - cum_treat_n * mean_ctrl_prefix

    total_f = float(total_n)
    frac = np.concatenate([[0.0], np.cumsum(b_width) / total_f])
    norm_gain = np.concatenate([[0.0], gain / total_f])
    return frac, norm_gain
