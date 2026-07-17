"""Demo 3 — Distillation beats heuristic session->device aggregation.

Reproduces the paper's ``agg_vs_distill`` table. Trains the two-tower S-learner,
then compares two ways to obtain a DEVICE-level uplift score:

* heuristic aggregation of the session-level scores (simple / recency / duration
  weighted means), and
* the end-to-end distilled device head.

Each device score is evaluated with normalized Qini against the (held-out)
session outcomes, and its correlation with the KNOWN device-level ground-truth
uplift ``tau_d``.

Outputs a table to stdout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tt_uplift import (
    TrainConfig,
    aggregate_to_device,
    broadcast_device_scores,
    normalized_qini,
    predict_device_uplift_two_tower,
    predict_session_uplift_two_tower,
    train_two_tower,
)

from _common import (
    NORM_OUTCOME_COL,
    banner,
    build_two_tower,
    device_train_test_split,
    make_data,
    prep,
)


def _device_nqini(device_ids, session_scores, weights, test_df, y, t) -> float:
    dev_scores = aggregate_to_device(device_ids, session_scores, weights=weights)
    broadcast = broadcast_device_scores(test_df, dev_scores)
    return normalized_qini(broadcast, y, t)


def _corr_with_truth(device_scores_df, truth) -> float:
    merged = device_scores_df.merge(truth, on="device_id", how="inner")
    return float(np.corrcoef(merged["device_score"], merged["tau_d_uplift"])[0, 1])


def main() -> None:
    banner("DEMO 3: Uplift distillation vs heuristic session->device aggregation")
    data = make_data()
    train_df, test_df = device_train_test_split(data.sessions, test_frac=0.3, seed=0)

    enc, train_t = prep(train_df, label_col=NORM_OUTCOME_COL)
    _, test_t = prep(test_df, label_col=NORM_OUTCOME_COL)
    # Use the train-fitted encoders for the test tensors (avoid leakage).
    from tt_uplift import transform as _transform

    test_t = _transform(test_df, enc, label_col=NORM_OUTCOME_COL)

    model = build_two_tower(enc, seed=0)
    train_two_tower(model, train_t, TrainConfig(epochs=10, alpha_distill=1.0, seed=0))

    # Session-level scores on TEST.
    sess_scores = predict_session_uplift_two_tower(model, test_t, TrainConfig())
    y = test_t.outcome.numpy().ravel()
    t = test_t.treatment_binary.numpy().ravel()
    device_ids = test_df["device_id"].to_numpy()

    session_nqini = normalized_qini(sess_scores, y, t)

    # Aggregation variants (weights defined on test sessions).
    recency = np.linspace(0.5, 1.5, len(test_df))  # stand-in recency weight
    duration_w = test_df["duration"].to_numpy()
    rows = []
    for name, w in [
        ("aggregate: simple mean", None),
        ("aggregate: recency-weighted", recency),
        ("aggregate: duration-weighted", duration_w),
    ]:
        nq = _device_nqini(device_ids, sess_scores, w, test_df, y, t)
        rows.append([name, f"{nq:+.4f}"])

    # Distilled device head.
    dev_up = predict_device_uplift_two_tower(model, test_t)
    dist_nqini = normalized_qini(dev_up, y, t)
    rows.append(["distill: device head (ours)", f"{dist_nqini:+.4f}"])

    print(f"Session-level NQini (reference, needs content features): {session_nqini:+.4f}\n")
    print(pd.DataFrame(rows, columns=["Device-level method", "Device NQini"]).to_string(index=False))

    # Correlation with KNOWN device ground truth.
    truth = data.device_truth[["device_id", "tau_d_uplift"]]
    dev_df = pd.DataFrame({"device_id": test_df["device_id"].to_numpy(), "device_score": dev_up})
    agg_df = aggregate_to_device(device_ids, sess_scores, weights=None)
    print("\nCorrelation with known device-level ground-truth uplift (tau_d):")
    print(f"  distilled device head : {_corr_with_truth(dev_df.drop_duplicates('device_id'), truth):+.3f}")
    print(f"  simple aggregation    : {_corr_with_truth(agg_df, truth):+.3f}")

    best_agg = max(_device_nqini(device_ids, sess_scores, w, test_df, y, t) for w in [None, recency, duration_w])
    print(f"\n[result] distilled device NQini ({dist_nqini:+.4f}) vs best aggregation ({best_agg:+.4f})")
    print(
        "\nNOTE (honest reproduction): both methods emit ONE score per device, so this is a\n"
        "fair comparison. Which one wins is regime-dependent:\n"
        "  * When the session teacher is strong and each device has many sessions (this clean\n"
        "    synthetic default), aggregation denoises well and is a strong baseline.\n"
        "  * In the paper's production regime aggregation collapses (0.34 -> 0.09) because the\n"
        "    teacher is a separate, weaker model, aggregation uses hand-tuned recency weights\n"
        "    over FEW recent sessions, and there is a temporal train/serve split. The distilled\n"
        "    head, trained end-to-end for the device target, degrades more gracefully there.\n"
        "This demo reproduces the comparison METHODOLOGY and the signal-vs-simplicity trade-off;\n"
        "it does not hard-code which side wins. See demos/README for how to sweep the regime."
    )


if __name__ == "__main__":
    main()
