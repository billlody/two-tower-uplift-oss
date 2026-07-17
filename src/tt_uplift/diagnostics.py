"""Causal balance diagnostics used by the confounding demo.

Mirrors the Databricks production-data notebook, but on the synthetic frame:
treatment-outcome correlation (should flip sign, not vanish), standardized mean
difference of the confounder (should shrink toward < 0.1), and the high-vs-low
ATE contrast (should flip sign).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def standardized_mean_difference(
    df: pd.DataFrame, treatment_col: str, confounder_col: str, split: str = "global", stratum_col: str = "stratum"
) -> float:
    """|SMD| of a confounder between high/low treatment groups.

    Parameters
    ----------
    split : {"global", "within"}
        ``global`` splits at the overall treatment median; ``within`` splits at a
        per-stratum median and returns the size-weighted pooled |SMD|.
    """
    if split == "global":
        med = df[treatment_col].median()
        hi = df[df[treatment_col] >= med][confounder_col]
        lo = df[df[treatment_col] < med][confounder_col]
        return _abs_smd(hi.to_numpy(), lo.to_numpy())

    num, den = 0.0, 0.0
    for _, g in df.groupby(stratum_col):
        med = g[treatment_col].median()
        hi = g[g[treatment_col] >= med][confounder_col].to_numpy()
        lo = g[g[treatment_col] < med][confounder_col].to_numpy()
        if len(hi) == 0 or len(lo) == 0:
            continue
        smd = _abs_smd(hi, lo)
        n = len(hi) + len(lo)
        num += smd * n
        den += n
    return num / den if den > 0 else float("nan")


def _abs_smd(a: np.ndarray, b: np.ndarray) -> float:
    va, vb = np.var(a), np.var(b)
    pooled = np.sqrt((va + vb) / 2.0) or 1e-8
    return float(abs(a.mean() - b.mean()) / pooled)


def within_stratum_corr(df: pd.DataFrame, treatment_col: str, outcome_col: str, stratum_col: str = "stratum") -> float:
    """Size-weighted mean within-stratum Pearson correlation(treatment, outcome)."""
    num, den = 0.0, 0.0
    for _, g in df.groupby(stratum_col):
        if len(g) < 3:
            continue
        t, y = g[treatment_col].to_numpy(), g[outcome_col].to_numpy()
        if t.std() < 1e-8 or y.std() < 1e-8:
            continue
        r = np.corrcoef(t, y)[0, 1]
        n = len(g)
        num += r * n
        den += n
    return num / den if den > 0 else float("nan")


def ate_contrast(
    df: pd.DataFrame, treatment_col: str, outcome_col: str, split: str = "global", stratum_col: str = "stratum"
) -> float:
    """Mean-outcome difference (high - low treatment).

    ``global`` uses the overall median split; ``within`` uses per-stratum median
    splits and returns the size-weighted average contrast.
    """
    if split == "global":
        med = df[treatment_col].median()
        hi = df[df[treatment_col] >= med][outcome_col].mean()
        lo = df[df[treatment_col] < med][outcome_col].mean()
        return float(hi - lo)

    num, den = 0.0, 0.0
    for _, g in df.groupby(stratum_col):
        med = g[treatment_col].median()
        hi = g[g[treatment_col] >= med][outcome_col]
        lo = g[g[treatment_col] < med][outcome_col]
        if len(hi) == 0 or len(lo) == 0:
            continue
        n = len(g)
        num += (hi.mean() - lo.mean()) * n
        den += n
    return num / den if den > 0 else float("nan")


def balance_report(
    df: pd.DataFrame,
    treatment_col: str,
    raw_outcome_col: str,
    norm_outcome_col: str,
    confounder_col: str,
    stratum_col: str = "stratum",
) -> Dict[str, float]:
    """Compute the full raw-vs-stratified diagnostic bundle used by the demo/table."""
    return {
        "corr_raw": float(np.corrcoef(df[treatment_col], df[raw_outcome_col])[0, 1]),
        "corr_within": within_stratum_corr(df, treatment_col, norm_outcome_col, stratum_col),
        "smd_global": standardized_mean_difference(df, treatment_col, confounder_col, "global", stratum_col),
        "smd_within": standardized_mean_difference(df, treatment_col, confounder_col, "within", stratum_col),
        "ate_raw": ate_contrast(df, treatment_col, raw_outcome_col, "global", stratum_col),
        "ate_norm": ate_contrast(df, treatment_col, norm_outcome_col, "within", stratum_col),
    }
