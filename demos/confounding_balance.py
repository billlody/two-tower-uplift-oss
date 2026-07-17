"""Demo 1 — Confounding & balance after stratified normalization.

Reproduces the paper's balance argument on the synthetic world with KNOWN ground
truth. Shows that stratified normalization:

* keeps the treatment->outcome *effect* (it flips from spuriously POSITIVE to the
  true NEGATIVE sign — it does NOT vanish), and
* balances the *confounder* (|SMD| of duration drops below the 0.1 threshold).

Outputs a table to stdout and an FWL-style before/after figure to
``outputs/confounding_balance.png``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tt_uplift.diagnostics import balance_report
from tt_uplift.dgp import RAW_OUTCOME_COL, TREATMENT_COL

from _common import NORM_OUTCOME_COL, OUTPUT_DIR, banner, make_data


def _binned(x: np.ndarray, y: np.ndarray, nbins: int = 20):
    qs = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
    idx = np.clip(np.digitize(x, qs[1:-1]), 0, len(qs) - 2)
    cx, cy = [], []
    for b in range(len(qs) - 1):
        m = idx == b
        if m.sum():
            cx.append(x[m].mean())
            cy.append(y[m].mean())
    return np.array(cx), np.array(cy)


def main() -> None:
    banner("DEMO 1: Confounding & balance after stratified normalization")
    data = make_data()
    df = data.sessions

    rep = balance_report(df, TREATMENT_COL, RAW_OUTCOME_COL, NORM_OUTCOME_COL, "duration")

    table = pd.DataFrame(
        [
            ["corr(treatment, outcome)", f"{rep['corr_raw']:+.4f}", f"{rep['corr_within']:+.4f}", "flip + -> - (effect persists, NOT zero)"],
            ["|SMD| of duration (confounder)", f"{rep['smd_global']:.4f}", f"{rep['smd_within']:.4f}", "shrink < 0.1 (balance)"],
            ["ATE (high - low)", f"{rep['ate_raw']:+.4f}", f"{rep['ate_norm']:+.4f}", "flip + -> - (expected sign)"],
        ],
        columns=["Diagnostic", "Raw / Overall", "Within-stratum / Normalized", "Expected"],
    )
    print(table.to_string(index=False))
    print(f"\nGround-truth causal effect sign in the DGP: {data.meta['true_effect_sign'].upper()}")

    # Assertions against ground truth (the demo VERIFIES, it does not assert prose).
    assert rep["corr_raw"] > 0, "raw correlation should be spuriously positive"
    assert rep["corr_within"] < 0, "within-stratum correlation should be negative"
    assert rep["smd_global"] > 0.1, "confounder should be imbalanced overall"
    assert rep["smd_within"] < 0.1, "confounder should be balanced within strata"
    assert rep["ate_raw"] > 0 and rep["ate_norm"] < 0, "ATE sign should flip"
    print("\n[OK] All balance/effect signs match the known ground truth.")

    # ---- FWL-style figure: raw vs within-stratum-normalized -----------------
    # Within-stratum treatment z-score for the residual view.
    g = df.groupby("stratum")[TREATMENT_COL]
    t_within = ((df[TREATMENT_COL] - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0).to_numpy()
    t_global = ((df[TREATMENT_COL] - df[TREATMENT_COL].mean()) / df[TREATMENT_COL].std()).to_numpy()
    y_global = ((df[RAW_OUTCOME_COL] - df[RAW_OUTCOME_COL].mean()) / df[RAW_OUTCOME_COL].std()).to_numpy()
    y_norm = df[NORM_OUTCOME_COL].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bx, by = _binned(t_global, y_global)
    s_raw = np.polyfit(t_global, y_global, 1)[0]
    axes[0].plot(bx, by, "o-", color="#c0392b")
    axes[0].set_title(f"RAW: spurious positive association (slope={s_raw:+.3f})")
    axes[0].set_xlabel("treatment (global z)")
    axes[0].set_ylabel("outcome (global z)")
    axes[0].axhline(0, color="k", lw=0.6, alpha=0.5)
    axes[0].grid(alpha=0.3)

    bx2, by2 = _binned(t_within, y_norm)
    s_win = np.polyfit(t_within, y_norm, 1)[0]
    axes[1].plot(bx2, by2, "o-", color="#27ae60")
    axes[1].set_title(f"WITHIN-STRATUM: true negative effect (slope={s_win:+.3f})")
    axes[1].set_xlabel("treatment (within-stratum z)")
    axes[1].set_ylabel("normalized outcome")
    axes[1].axhline(0, color="k", lw=0.6, alpha=0.5)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Stratified normalization flips the spurious positive to the expected negative effect")
    fig.tight_layout()
    out = f"{OUTPUT_DIR}/confounding_balance.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
