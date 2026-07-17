"""Demo 2 — Naive device-level causal estimation fails.

Reproduces the paper's "more ads => more engagement" paradox: estimating the
treatment effect directly at the device level (without the session-level
decomposition) recovers the WRONG (positive) sign, because each device watches a
different content mix, so duration confounding cannot be stratified away at the
device grain. The session-level, within-stratum estimate recovers the correct
(negative) sign.

Outputs a comparison table to stdout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tt_uplift.dgp import RAW_OUTCOME_COL, TREATMENT_COL

from _common import NORM_OUTCOME_COL, banner, make_data


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(x, y, 1)[0])


def main() -> None:
    banner("DEMO 2: Naive device-level estimation recovers the WRONG sign")
    data = make_data()
    df = data.sessions

    # (a) Naive device-level: aggregate treatment & raw outcome per device, regress.
    dev = df.groupby("device_id").agg(
        ad_load=(TREATMENT_COL, "mean"),
        view_time=(RAW_OUTCOME_COL, "mean"),
    )
    slope_device_naive = _slope(dev["ad_load"].to_numpy(), dev["view_time"].to_numpy())

    # (b) Device-level on raw sessions (pooled, no stratification).
    slope_session_raw = _slope(df[TREATMENT_COL].to_numpy(), df[RAW_OUTCOME_COL].to_numpy())

    # (c) Session-level within-stratum (the paper's approach): normalized outcome.
    g = df.groupby("stratum")[TREATMENT_COL]
    t_within = ((df[TREATMENT_COL] - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0)
    slope_within = _slope(t_within.to_numpy(), df[NORM_OUTCOME_COL].to_numpy())

    true_sign = data.meta["true_effect_sign"]
    table = pd.DataFrame(
        [
            ["Device-level naive (per-device means)", f"{slope_device_naive:+.4f}", "positive (WRONG)"],
            ["Session-level pooled, raw outcome", f"{slope_session_raw:+.4f}", "positive (WRONG)"],
            ["Session-level within-stratum (ours)", f"{slope_within:+.4f}", "negative (CORRECT)"],
        ],
        columns=["Estimator", "Est. treatment->outcome slope", "Sign"],
    )
    print(table.to_string(index=False))
    print(f"\nTrue causal effect sign in the DGP: {true_sign.upper()}")

    assert slope_device_naive > 0, "device-level naive should show the wrong positive sign"
    assert slope_within < 0, "within-stratum should recover the correct negative sign"
    print("\n[OK] Device-level naive estimation fails; session-level decomposition recovers the truth.")
    print("Root cause: per-device content mixes differ, so duration confounding cannot")
    print("be stratified/matched at the device grain without a session-level intermediate.")


if __name__ == "__main__":
    main()
