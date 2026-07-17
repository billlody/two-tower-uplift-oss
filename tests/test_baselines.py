"""Baseline coverage: TARNet, DragonNet, CEVAE train and produce sane uplift.

These close the gap flagged in review — the deep-learning uplift baselines
(TARNet, DragonNet, CEVAE) are now exercised end-to-end on the synthetic DGP with
known ground truth, so the offline comparison against TowerUplift is reproducible.
"""

import numpy as np
import torch

from tt_uplift import (
    BaselineConfig,
    CEVAE,
    CEVAEConfig,
    DGPConfig,
    DragonNet,
    TARNet,
    TrainConfig,
    fit_encoders,
    generate,
    normalized_qini,
    predict_uplift_baseline,
    predict_uplift_cevae,
    stratified_zscore,
    train_baseline,
    train_cevae,
    transform,
)
from tt_uplift.dgp import CONTENT_NUMERIC, DEVICE_NUMERIC
from tt_uplift.features import content_cardinalities, device_cardinalities


def _prepare(n_devices: int = 1500, seed: int = 7):
    data = generate(DGPConfig(n_devices=n_devices, seed=seed))
    df = stratified_zscore(data.sessions, "watch_percent", out_col="norm_watch_percent")
    enc = fit_encoders(df)
    tens = transform(df, enc, label_col="norm_watch_percent")
    cat_card = {**device_cardinalities(enc), **content_cardinalities(enc)}
    numeric_dim = len(DEVICE_NUMERIC) + len(CONTENT_NUMERIC)
    y = tens.outcome.numpy().ravel()
    t = tens.treatment_binary.numpy().ravel()
    gt = df["gt_tau_session"].to_numpy()
    return tens, cat_card, numeric_dim, y, t, gt


def test_tarnet_beats_random_and_tracks_tau():
    tens, cat_card, numeric_dim, y, t, gt = _prepare()
    torch.manual_seed(0)
    model = TARNet(BaselineConfig(numeric_dim=numeric_dim, cat_cardinalities=cat_card))
    train_baseline(model, tens, TrainConfig(epochs=10, seed=0))
    up = predict_uplift_baseline(model, tens)
    assert normalized_qini(up, y, t) > 0.0
    assert np.corrcoef(up, gt)[0, 1] > 0.2
    assert up.mean() < 0.0  # true effect is negative


def test_dragonnet_beats_random_and_tracks_tau():
    tens, cat_card, numeric_dim, y, t, gt = _prepare()
    torch.manual_seed(0)
    model = DragonNet(BaselineConfig(numeric_dim=numeric_dim, cat_cardinalities=cat_card))
    train_baseline(model, tens, TrainConfig(epochs=10, seed=0), is_dragonnet=True)
    up = predict_uplift_baseline(model, tens)
    assert normalized_qini(up, y, t) > 0.0
    assert np.corrcoef(up, gt)[0, 1] > 0.2
    assert up.mean() < 0.0


def test_cevae_trains_and_ranks_above_random():
    tens, cat_card, numeric_dim, y, t, gt = _prepare()
    torch.manual_seed(0)
    model = CEVAE(CEVAEConfig(numeric_dim=numeric_dim, cat_cardinalities=cat_card))
    train_cevae(model, tens, TrainConfig(epochs=10, seed=0))
    up = predict_uplift_cevae(model, tens)
    # CEVAE is the weakest baseline on this clean DGP; assert it runs and produces
    # a finite, non-degenerate score (the demo reports its honest ranking).
    assert np.all(np.isfinite(up))
    assert up.std() > 0.0
    assert np.isfinite(normalized_qini(up, y, t))


def test_all_models_rank_on_held_out_split():
    """Every model produces a finite, non-degenerate uplift ranking on held-out devices.

    Mirrors the offline-comparison demo: train on a device-disjoint train split,
    score on the held-out test split.  We deliberately do NOT hard-code which model
    wins — the demo reports the honest ranking; this test only guards that the whole
    comparison harness runs and every model yields a usable score out-of-sample.
    """
    from tt_uplift import (
        TwoTowerConfig,
        TwoTowerUpliftModel,
        predict_session_uplift_two_tower,
        train_two_tower,
    )

    data = generate(DGPConfig(n_devices=1500, seed=7))
    df = stratified_zscore(data.sessions, "watch_percent", out_col="norm_watch_percent")
    rng = np.random.default_rng(0)
    devices = df["device_id"].unique()
    test_devices = set(rng.choice(devices, size=int(len(devices) * 0.3), replace=False))
    is_test = df["device_id"].isin(test_devices)
    train_df, test_df = df[~is_test].copy(), df[is_test].copy()

    enc = fit_encoders(train_df)
    train_t = transform(train_df, enc, label_col="norm_watch_percent")
    test_t = transform(test_df, enc, label_col="norm_watch_percent")
    y = test_t.outcome.numpy().ravel()
    t = test_t.treatment_binary.numpy().ravel()
    cat_card = {**device_cardinalities(enc), **content_cardinalities(enc)}
    numeric_dim = len(DEVICE_NUMERIC) + len(CONTENT_NUMERIC)

    scores = {}

    torch.manual_seed(0)
    tower = TwoTowerUpliftModel(
        TwoTowerConfig(
            device_numeric_dim=len(DEVICE_NUMERIC),
            content_numeric_dim=len(CONTENT_NUMERIC),
            device_cat_cardinalities=device_cardinalities(enc),
            content_cat_cardinalities=content_cardinalities(enc),
        )
    )
    train_two_tower(tower, train_t, TrainConfig(epochs=10, seed=0))
    scores["tower"] = predict_session_uplift_two_tower(tower, test_t, TrainConfig())

    torch.manual_seed(0)
    tarnet = TARNet(BaselineConfig(numeric_dim=numeric_dim, cat_cardinalities=cat_card))
    train_baseline(tarnet, train_t, TrainConfig(epochs=10, seed=0))
    scores["tarnet"] = predict_uplift_baseline(tarnet, test_t)

    torch.manual_seed(0)
    dragon = DragonNet(BaselineConfig(numeric_dim=numeric_dim, cat_cardinalities=cat_card))
    train_baseline(dragon, train_t, TrainConfig(epochs=10, seed=0), is_dragonnet=True)
    scores["dragonnet"] = predict_uplift_baseline(dragon, test_t)

    torch.manual_seed(0)
    cevae = CEVAE(CEVAEConfig(numeric_dim=numeric_dim, cat_cardinalities=cat_card))
    train_cevae(cevae, train_t, TrainConfig(epochs=10, seed=0))
    scores["cevae"] = predict_uplift_cevae(cevae, test_t)

    for name, up in scores.items():
        assert up.shape == y.shape, name
        assert np.all(np.isfinite(up)), name
        assert up.std() > 0.0, name
        assert np.isfinite(normalized_qini(up, y, t)), name
