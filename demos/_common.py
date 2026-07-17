"""Shared helpers for the demo scripts (data prep, splits, model builders)."""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd

from tt_uplift import (
    BaselineConfig,
    CEVAE,
    CEVAEConfig,
    DGPConfig,
    DragonNet,
    SyntheticData,
    TARNet,
    TwoTowerConfig,
    TwoTowerUpliftModel,
    fit_encoders,
    generate,
    stratified_zscore,
    transform,
)
from tt_uplift.dgp import CONTENT_NUMERIC, DEVICE_NUMERIC, RAW_OUTCOME_COL
from tt_uplift.features import content_cardinalities, device_cardinalities

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NORM_OUTCOME_COL = "norm_watch_percent"


def make_data(config: DGPConfig | None = None) -> SyntheticData:
    """Generate the unified world and attach the stratified-normalized outcome."""
    data = generate(config)
    data.sessions = stratified_zscore(data.sessions, RAW_OUTCOME_COL, out_col=NORM_OUTCOME_COL)
    return data


def device_train_test_split(
    df: pd.DataFrame, test_frac: float = 0.3, seed: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by device so no device leaks across train/test."""
    rng = np.random.default_rng(seed)
    devices = df["device_id"].unique()
    test_devices = set(rng.choice(devices, size=int(len(devices) * test_frac), replace=False))
    is_test = df["device_id"].isin(test_devices)
    return df[~is_test].copy(), df[is_test].copy()


def build_two_tower(enc, use_double_ml: bool = False, seed: int = 0) -> TwoTowerUpliftModel:
    """Construct a TwoTowerUpliftModel sized to the encoders."""
    import torch

    torch.manual_seed(seed)
    cfg = TwoTowerConfig(
        device_numeric_dim=len(DEVICE_NUMERIC),
        content_numeric_dim=len(CONTENT_NUMERIC),
        device_cat_cardinalities=device_cardinalities(enc),
        content_cat_cardinalities=content_cardinalities(enc),
        use_double_ml=use_double_ml,
    )
    return TwoTowerUpliftModel(cfg)


def prep(df: pd.DataFrame, label_col: str = NORM_OUTCOME_COL):
    """Fit encoders on ``df`` and return ``(encoders, tensors)``."""
    enc = fit_encoders(df)
    return enc, transform(df, enc, label_col=label_col)


def _baseline_cardinalities(enc):
    """Concatenated device+content categorical cardinalities (single-input baselines)."""
    return {**device_cardinalities(enc), **content_cardinalities(enc)}


def _baseline_numeric_dim() -> int:
    return len(DEVICE_NUMERIC) + len(CONTENT_NUMERIC)


def build_tarnet(enc, seed: int = 0) -> TARNet:
    """Construct a TARNet baseline sized to the encoders."""
    import torch

    torch.manual_seed(seed)
    cfg = BaselineConfig(numeric_dim=_baseline_numeric_dim(), cat_cardinalities=_baseline_cardinalities(enc))
    return TARNet(cfg)


def build_dragonnet(enc, seed: int = 0) -> DragonNet:
    """Construct a DragonNet baseline sized to the encoders."""
    import torch

    torch.manual_seed(seed)
    cfg = BaselineConfig(numeric_dim=_baseline_numeric_dim(), cat_cardinalities=_baseline_cardinalities(enc))
    return DragonNet(cfg)


def build_cevae(enc, seed: int = 0) -> CEVAE:
    """Construct a CEVAE baseline sized to the encoders."""
    import torch

    torch.manual_seed(seed)
    cfg = CEVAEConfig(numeric_dim=_baseline_numeric_dim(), cat_cardinalities=_baseline_cardinalities(enc))
    return CEVAE(cfg)


def banner(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)
