"""Heuristic session-to-device aggregation baselines.

These are the "standard remedy" the paper argues against: score sessions, then
average the scores to the device level with a hand-tuned weighting.  Demos
compare these against the end-to-end distilled device head.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def aggregate_to_device(
    device_ids: np.ndarray,
    session_scores: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Aggregate per-session uplift scores to per-device scores by (weighted) mean.

    Parameters
    ----------
    device_ids : ndarray[N]
        Device id per session.
    session_scores : ndarray[N]
        Session-level uplift score per session.
    weights : ndarray[N], optional
        Non-negative weights (e.g. recency or duration).  ``None`` => simple mean.

    Returns
    -------
    DataFrame
        Columns ``device_id`` and ``device_score``.
    """
    df = pd.DataFrame({"device_id": device_ids, "score": session_scores})
    if weights is None:
        agg = df.groupby("device_id")["score"].mean()
    else:
        df["w"] = np.clip(weights, 0.0, None)
        df["ws"] = df["w"] * df["score"]
        grp = df.groupby("device_id")
        agg = grp["ws"].sum() / grp["w"].sum().replace(0.0, np.nan)
        agg = agg.fillna(grp["score"].mean())
    return agg.rename("device_score").reset_index()


def broadcast_device_scores(sessions: pd.DataFrame, device_scores: pd.DataFrame) -> np.ndarray:
    """Map per-device scores back onto sessions (for device-level session eval).

    Returns an array aligned with ``sessions`` rows.
    """
    merged = sessions[["device_id"]].merge(device_scores, on="device_id", how="left")
    return merged["device_score"].to_numpy(dtype=float)
