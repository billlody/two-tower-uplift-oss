"""Demo 4 — Session vs device uplift Pareto frontier via the distillation weight.

Reproduces the paper's Pareto figure. Sweeps the distillation weight ``alpha`` and
plots session-level NQini (needs content features) against device-level NQini
(the distilled, serving-time score). Low alpha favors session ranking; high alpha
favors the device head. The trade-off is smooth, so alpha is a deployment knob.

Outputs ``outputs/alpha_pareto.png`` and a table to stdout.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tt_uplift import (
    TrainConfig,
    normalized_qini,
    predict_device_uplift_two_tower,
    predict_session_uplift_two_tower,
    train_two_tower,
    transform,
)

from _common import (
    NORM_OUTCOME_COL,
    OUTPUT_DIR,
    banner,
    build_two_tower,
    device_train_test_split,
    make_data,
    prep,
)

ALPHAS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]


def main() -> None:
    banner("DEMO 4: Session vs device uplift Pareto frontier over distillation weight alpha")
    data = make_data()
    train_df, test_df = device_train_test_split(data.sessions, test_frac=0.3, seed=0)
    enc, train_t = prep(train_df, label_col=NORM_OUTCOME_COL)
    test_t = transform(test_df, enc, label_col=NORM_OUTCOME_COL)

    y = test_t.outcome.numpy().ravel()
    t = test_t.treatment_binary.numpy().ravel()

    rows = []
    for a in ALPHAS:
        model = build_two_tower(enc, seed=0)
        train_two_tower(model, train_t, TrainConfig(epochs=10, alpha_distill=a, seed=0))
        sess = predict_session_uplift_two_tower(model, test_t, TrainConfig())
        dev = predict_device_uplift_two_tower(model, test_t)
        sn = normalized_qini(sess, y, t)
        dn = normalized_qini(dev, y, t)
        rows.append({"alpha": a, "session_nqini": sn, "device_nqini": dn})
        print(f"  alpha={a:>4}: session NQini={sn:+.4f}  device NQini={dn:+.4f}")

    tbl = pd.DataFrame(rows)
    print("\n" + tbl.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(tbl["session_nqini"], tbl["device_nqini"], "o-", color="#2c3e50")
    for _, r in tbl.iterrows():
        ax.annotate(f"a={r['alpha']}", (r["session_nqini"], r["device_nqini"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Session-level NQini (needs content features)")
    ax.set_ylabel("Device-level NQini (distilled, serving-time)")
    ax.set_title("Distillation weight alpha traces the session<->device Pareto frontier")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = f"{OUTPUT_DIR}/alpha_pareto.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\n[saved] {out}")
    # Sanity: device NQini should be non-decreasing-ish in alpha (trend, not strict).
    if tbl["device_nqini"].iloc[-1] >= tbl["device_nqini"].iloc[0]:
        print("[OK] Higher alpha trends toward stronger device-level ranking, as expected.")


if __name__ == "__main__":
    main()
