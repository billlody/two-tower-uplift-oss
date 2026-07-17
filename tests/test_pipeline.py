"""End-to-end smoke tests: metrics behave, model recovers device ground truth."""

import numpy as np

from tt_uplift import (
    DGPConfig,
    TrainConfig,
    compute_aucc,
    fit_encoders,
    generate,
    normalized_qini,
    predict_device_uplift_two_tower,
    stratified_zscore,
    train_two_tower,
    transform,
)
from tt_uplift.dgp import CONTENT_NUMERIC, DEVICE_NUMERIC
from tt_uplift.features import content_cardinalities, device_cardinalities
from tt_uplift.model import TwoTowerConfig, TwoTowerUpliftModel


def test_aucc_oracle_beats_random():
    rng = np.random.default_rng(0)
    n = 5000
    t = rng.integers(0, 2, n).astype(float)
    tau = rng.normal(0, 1, n)
    y = tau * t + rng.normal(0, 0.5, n)
    oracle = (2 * t - 1) * y
    rand = rng.random(n)
    assert compute_aucc(oracle, y, t) > compute_aucc(rand, y, t)


def test_normalized_qini_range_sane():
    rng = np.random.default_rng(1)
    n = 4000
    t = rng.integers(0, 2, n).astype(float)
    y = rng.normal(0, 1, n)
    nq = normalized_qini(rng.random(n), y, t)
    assert -1.0 < nq < 1.0


def test_two_tower_recovers_device_ground_truth():
    data = generate(DGPConfig(n_devices=1500, seed=7))
    df = stratified_zscore(data.sessions, "watch_percent", out_col="norm_watch_percent")
    enc = fit_encoders(df)
    tens = transform(df, enc, label_col="norm_watch_percent")
    cfg = TwoTowerConfig(
        device_numeric_dim=len(DEVICE_NUMERIC),
        content_numeric_dim=len(CONTENT_NUMERIC),
        device_cat_cardinalities=device_cardinalities(enc),
        content_cat_cardinalities=content_cardinalities(enc),
    )
    model = TwoTowerUpliftModel(cfg)
    train_two_tower(model, tens, TrainConfig(epochs=6, seed=0))
    dev_up = predict_device_uplift_two_tower(model, tens)
    truth = data.device_truth.set_index("device_id")["tau_d_uplift"]
    gt = df["device_id"].map(truth).to_numpy()
    r = np.corrcoef(dev_up, gt)[0, 1]
    assert r > 0.3, f"device head should track ground-truth tau_d, got r={r:.3f}"
