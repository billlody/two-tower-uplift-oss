"""Ground-truth sign checks for the unified DGP + stratified normalization."""

import numpy as np

from tt_uplift import generate, stratified_zscore
from tt_uplift.diagnostics import balance_report


def _report(seed: int = 42):
    data = generate()
    df = stratified_zscore(data.sessions, "view_time", out_col="norm_view_time")
    return balance_report(df, "ad_load", "view_time", "norm_view_time", "duration")


def test_raw_correlation_is_spuriously_positive():
    assert _report()["corr_raw"] > 0.05


def test_within_stratum_effect_is_negative_not_zero():
    corr = _report()["corr_within"]
    assert corr < -0.05, "true effect should persist as negative, not vanish to zero"


def test_confounder_balance_improves_below_threshold():
    rep = _report()
    assert rep["smd_global"] > 0.1
    assert rep["smd_within"] < 0.1


def test_ate_sign_flips_after_normalization():
    rep = _report()
    assert rep["ate_raw"] > 0
    assert rep["ate_norm"] < 0


def test_ground_truth_columns_present():
    data = generate()
    for col in ["gt_theta", "gt_tau_session", "gt_session_uplift", "stratum"]:
        assert col in data.sessions.columns
    assert {"device_id", "theta", "tau_d_uplift"}.issubset(data.device_truth.columns)
