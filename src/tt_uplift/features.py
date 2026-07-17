"""Feature preparation: DataFrame -> tensors, label encoders, treatment binarization.

Turns the synthetic sessions frame into the tensors the models consume.  Kept
deliberately simple: numeric features are z-scored globally, categoricals are
integer-encoded (index 0 reserved for unknown), and the continuous treatment is
z-scored with a binary indicator derived at (optionally per-content) median.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from .dgp import (
    CONTENT_CATEGORICAL,
    CONTENT_NUMERIC,
    DEVICE_CATEGORICAL,
    DEVICE_NUMERIC,
    RAW_OUTCOME_COL,
    TREATMENT_COL,
)


@dataclass
class Encoders:
    """Fitted preprocessing state (fit on train, applied to any split)."""

    numeric_mean: Dict[str, float] = field(default_factory=dict)
    numeric_std: Dict[str, float] = field(default_factory=dict)
    cat_maps: Dict[str, Dict[str, int]] = field(default_factory=dict)
    treatment_mean: float = 0.0
    treatment_std: float = 1.0
    treatment_median: float = 0.0


@dataclass
class Tensors:
    """Model-ready tensors for one split."""

    device_numeric: torch.Tensor
    device_cat: torch.Tensor
    content_numeric: torch.Tensor
    content_cat: torch.Tensor
    treatment: torch.Tensor  # z-scored, shape [N, 1]
    treatment_binary: torch.Tensor  # 0/1, shape [N, 1]
    outcome: torch.Tensor  # label column, shape [N, 1]
    device_id: torch.Tensor  # shape [N]


def fit_encoders(df: pd.DataFrame, treatment_binarize_by: str = "global") -> Encoders:
    """Fit numeric/categorical encoders and treatment stats on a training split.

    Parameters
    ----------
    df : DataFrame
    treatment_binarize_by : {"global", "content_id"}
        ``global`` splits treatment at the global median; ``content_id`` uses a
        per-content median (controls for content-level ad-load differences).

    Returns
    -------
    Encoders
    """
    enc = Encoders()
    for col in DEVICE_NUMERIC + CONTENT_NUMERIC:
        enc.numeric_mean[col] = float(df[col].mean())
        enc.numeric_std[col] = float(df[col].std() or 1.0)
    for col in DEVICE_CATEGORICAL + CONTENT_CATEGORICAL:
        cats = sorted(df[col].astype(str).unique())
        enc.cat_maps[col] = {v: i + 1 for i, v in enumerate(cats)}  # 0 = unknown
    enc.treatment_mean = float(df[TREATMENT_COL].mean())
    enc.treatment_std = float(df[TREATMENT_COL].std() or 1.0)
    enc.treatment_median = float(df[TREATMENT_COL].median())
    enc._binarize_by = treatment_binarize_by  # type: ignore[attr-defined]
    if treatment_binarize_by == "content_id":
        enc._content_median = df.groupby("content_id")[TREATMENT_COL].median().to_dict()  # type: ignore[attr-defined]
    return enc


def transform(df: pd.DataFrame, enc: Encoders, label_col: str = RAW_OUTCOME_COL) -> Tensors:
    """Apply fitted encoders to a split and return model-ready tensors.

    Parameters
    ----------
    df : DataFrame
    enc : Encoders
        Fitted via :func:`fit_encoders`.
    label_col : str
        Which outcome column to use as the training label (raw or normalized).

    Returns
    -------
    Tensors
    """

    def num_block(cols: List[str]) -> torch.Tensor:
        arr = np.stack(
            [((df[c].to_numpy(dtype=float) - enc.numeric_mean[c]) / enc.numeric_std[c]) for c in cols], axis=1
        )
        return torch.tensor(arr, dtype=torch.float32)

    def cat_block(cols: List[str]) -> torch.Tensor:
        if not cols:
            return torch.zeros((len(df), 0), dtype=torch.long)
        arr = np.stack(
            [df[c].astype(str).map(lambda v, cc=c: enc.cat_maps[cc].get(v, 0)).to_numpy(dtype=np.int64) for c in cols],
            axis=1,
        )
        return torch.tensor(arr, dtype=torch.long)

    treatment_raw = df[TREATMENT_COL].to_numpy(dtype=float)
    treatment_z = (treatment_raw - enc.treatment_mean) / enc.treatment_std

    binarize_by = getattr(enc, "_binarize_by", "global")
    if binarize_by == "content_id":
        cmed = getattr(enc, "_content_median", {})
        thresholds = df["content_id"].map(lambda k: cmed.get(k, enc.treatment_median)).to_numpy(dtype=float)
        treat_bin = (treatment_raw >= thresholds).astype(np.float32)
    else:
        treat_bin = (treatment_raw >= enc.treatment_median).astype(np.float32)

    return Tensors(
        device_numeric=num_block(DEVICE_NUMERIC),
        device_cat=cat_block(DEVICE_CATEGORICAL),
        content_numeric=num_block(CONTENT_NUMERIC),
        content_cat=cat_block(CONTENT_CATEGORICAL),
        treatment=torch.tensor(treatment_z, dtype=torch.float32).unsqueeze(1),
        treatment_binary=torch.tensor(treat_bin, dtype=torch.float32).unsqueeze(1),
        outcome=torch.tensor(df[label_col].to_numpy(dtype=float), dtype=torch.float32).unsqueeze(1),
        device_id=torch.tensor(df["device_id"].to_numpy(dtype=np.int64), dtype=torch.long),
    )


def cardinalities(enc: Encoders, cols: List[str]) -> Dict[str, int]:
    """Return ``{col: num_categories}`` for the given categorical columns."""
    return {c: len(enc.cat_maps[c]) for c in cols}


def device_cardinalities(enc: Encoders) -> Dict[str, int]:
    return cardinalities(enc, DEVICE_CATEGORICAL)


def content_cardinalities(enc: Encoders) -> Dict[str, int]:
    return cardinalities(enc, CONTENT_CATEGORICAL)
