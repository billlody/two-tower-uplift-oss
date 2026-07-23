# Databricks notebook source
# MAGIC %md
# MAGIC # Two-Tower Uplift — Full Test + Model Qini Comparison
# MAGIC
# MAGIC Runs the OSS artifact's test suite, then trains all five uplift models on a
# MAGIC **large** synthetic DGP (10× the demo default) and compares their normalized
# MAGIC Qini (NQini) with **bootstrap 95% confidence intervals**.
# MAGIC
# MAGIC Models: TowerUplift (session + deployed device head), TARNet, DragonNet, CEVAE.
# MAGIC All train on the binary within-stratum label with BCE (matches production).

# COMMAND ----------

# MAGIC %md ## 1. Install the repo (editable) + deps

# COMMAND ----------

# Path to the repo checkout in the Databricks workspace (a Git folder / Repo).
# Set this to wherever you synced https://github.com/billlody/two-tower-uplift-oss.
REPO_PATH = "/Workspace/Users/yanzheng@tubitv.com/two-tower-uplift-oss"

# torch/pandas/etc. are already on the ML runtime; install the package itself.
%pip install -e {REPO_PATH} --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

REPO_PATH = "/Workspace/Users/yanzheng@tubitv.com/two-tower-uplift-oss"

import sys
if f"{REPO_PATH}/src" not in sys.path:
    sys.path.insert(0, f"{REPO_PATH}/src")

import numpy as np
import torch

import tt_uplift
print("tt_uplift loaded from:", tt_uplift.__file__)
print("torch:", torch.__version__)

# COMMAND ----------

# MAGIC %md ## 2. Run the test suite (pytest)

# COMMAND ----------

import subprocess

result = subprocess.run(
    ["python", "-m", "pytest", "-q"],
    cwd=REPO_PATH,
    capture_output=True,
    text=True,
)
print(result.stdout[-4000:])
print(result.stderr[-2000:])
assert result.returncode == 0, "pytest failed — see output above"
print("\n[OK] All tests passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Generate a LARGE synthetic world (10× the demo default)
# MAGIC
# MAGIC Demo default is 4,000 devices (~48k sessions). Here we use **40,000 devices**
# MAGIC (~480k sessions) so a single run's NQini is stable and the bootstrap CIs are tight.

# COMMAND ----------

from tt_uplift import (
    DGPConfig,
    generate,
    stratified_binary_label,
    fit_encoders,
    transform,
    TrainConfig,
    TwoTowerConfig,
    TwoTowerUpliftModel,
    BaselineConfig,
    TARNet,
    DragonNet,
    CEVAE,
    CEVAEConfig,
    train_two_tower,
    train_baseline,
    train_cevae,
    predict_session_uplift_two_tower,
    predict_device_uplift_two_tower,
    predict_uplift_baseline,
    predict_uplift_cevae,
    normalized_qini,
)
from tt_uplift.dgp import CONTENT_NUMERIC, DEVICE_NUMERIC
from tt_uplift.features import content_cardinalities, device_cardinalities

N_DEVICES = 40_000          # 10x the demo default of 4,000
DGP_SEED = 42
EPOCHS = 10
TEST_FRAC = 0.3
LABEL = "label_view_time"

data = generate(DGPConfig(n_devices=N_DEVICES, seed=DGP_SEED))
df = stratified_binary_label(data.sessions, "view_time", out_col=LABEL)
print(f"Total sessions: {len(df):,} | devices: {df['device_id'].nunique():,} | "
      f"positive-label rate: {df[LABEL].mean():.3f}")

# COMMAND ----------

# MAGIC %md ### Device-disjoint train/test split

# COMMAND ----------

rng = np.random.default_rng(0)
devices = df["device_id"].unique()
test_devices = set(rng.choice(devices, size=int(len(devices) * TEST_FRAC), replace=False))
is_test = df["device_id"].isin(test_devices)
train_df, test_df = df[~is_test].copy(), df[is_test].copy()

enc = fit_encoders(train_df)
train_t = transform(train_df, enc, label_col=LABEL)
test_t = transform(test_df, enc, label_col=LABEL)

y = test_t.outcome.numpy().ravel()
t = test_t.treatment_binary.numpy().ravel()
gt_tau = test_df["gt_tau_session"].to_numpy()
cat_card = {**device_cardinalities(enc), **content_cardinalities(enc)}
numeric_dim = len(DEVICE_NUMERIC) + len(CONTENT_NUMERIC)
print(f"train sessions: {len(train_df):,} | test sessions (Qini computed here): {len(test_df):,}")

# COMMAND ----------

# MAGIC %md ## 4. Train all five models (same split, same budget)

# COMMAND ----------

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
train_two_tower(tower, train_t, TrainConfig(epochs=EPOCHS, seed=0))
scores["TowerUplift (session)"] = predict_session_uplift_two_tower(tower, test_t, TrainConfig())
scores["TowerUplift (device, deployed)"] = predict_device_uplift_two_tower(tower, test_t)
print("TowerUplift trained.")

torch.manual_seed(0)
tarnet = TARNet(BaselineConfig(numeric_dim=numeric_dim, cat_cardinalities=cat_card))
train_baseline(tarnet, train_t, TrainConfig(epochs=EPOCHS, seed=0))
scores["TARNet"] = predict_uplift_baseline(tarnet, test_t)
print("TARNet trained.")

torch.manual_seed(0)
dragon = DragonNet(BaselineConfig(numeric_dim=numeric_dim, cat_cardinalities=cat_card))
train_baseline(dragon, train_t, TrainConfig(epochs=EPOCHS, seed=0), is_dragonnet=True)
scores["DragonNet"] = predict_uplift_baseline(dragon, test_t)
print("DragonNet trained.")

torch.manual_seed(0)
cevae = CEVAE(CEVAEConfig(numeric_dim=numeric_dim, cat_cardinalities=cat_card))
train_cevae(cevae, train_t, TrainConfig(epochs=EPOCHS, seed=0))
scores["CEVAE"] = predict_uplift_cevae(cevae, test_t)
print("CEVAE trained.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. NQini with bootstrap 95% CI
# MAGIC
# MAGIC Resample the test set with replacement `B` times; report the 2.5/97.5
# MAGIC percentiles of NQini per model. Also report correlation with the known
# MAGIC per-session effect `gt_tau_session`.

# COMMAND ----------

import pandas as pd

B = 500
boot_rng = np.random.default_rng(123)
n = len(y)

rows = []
for name, s in scores.items():
    point = normalized_qini(s, y, t)
    corr_tau = float(np.corrcoef(s, gt_tau)[0, 1])
    boots = np.empty(B)
    for b in range(B):
        idx = boot_rng.integers(0, n, n)
        boots[b] = normalized_qini(s[idx], y[idx], t[idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    rows.append(
        {
            "model": name,
            "level": "device" if "device" in name else "session",
            "nqini": point,
            "ci_lo": lo,
            "ci_hi": hi,
            "corr_tau": corr_tau,
            "mean_uplift": float(s.mean()),
        }
    )

results = pd.DataFrame(rows).sort_values("nqini", ascending=False).reset_index(drop=True)
disp = results.assign(
    NQini=lambda d: d["nqini"].map(lambda v: f"{v:+.4f}"),
    **{"95% CI": lambda d: d.apply(lambda r: f"[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]", axis=1)},
    **{"corr(tau)": lambda d: d["corr_tau"].map(lambda v: f"{v:+.3f}")},
    **{"mean uplift": lambda d: d["mean_uplift"].map(lambda v: f"{v:+.3f}")},
)[["model", "level", "NQini", "95% CI", "corr(tau)", "mean uplift"]]
print(f"Bootstrap NQini (n_test={n:,}, B={B}, {N_DEVICES:,} devices, seed={DGP_SEED})\n")
print(disp.to_string(index=False))
displayHTML(disp.to_html(index=False))

# COMMAND ----------

# MAGIC %md ## 6. Plot: NQini with bootstrap CI error bars

# COMMAND ----------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plot_df = results.copy()
labels = [f"{m}\n({lv})" for m, lv in zip(plot_df["model"], plot_df["level"])]
colors = ["#2c3e50" if m.startswith("TowerUplift") else "#7f8c8d" for m in plot_df["model"]]
yerr = np.vstack([
    plot_df["nqini"] - plot_df["ci_lo"],
    plot_df["ci_hi"] - plot_df["nqini"],
])

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(range(len(plot_df)), plot_df["nqini"], yerr=yerr, capsize=4, color=colors)
ax.axhline(0.0, color="black", lw=0.8)
ax.set_xticks(range(len(plot_df)))
ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Normalized Qini (bootstrap 95% CI)")
ax.set_title(f"Uplift NQini comparison — {N_DEVICES:,}-device synthetic DGP (binary label, BCE)")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Notes
# MAGIC
# MAGIC - All models train on the **binary within-stratum label** (`1[y >= stratum mean]`)
# MAGIC   with **BCE** outcome loss and the **logit-contrast** uplift, matching the
# MAGIC   production `norm3_tvt_sec_label` classifier.
# MAGIC - Non-overlapping bootstrap CIs indicate a statistically reliable ranking at
# MAGIC   this sample size (unlike the small demo default, where seed noise dominates).
# MAGIC - The device-deployed head is a device-only `O(1)` policy scored on the
# MAGIC   session-level metric, so its NQini is expected to be lower than the
# MAGIC   content-aware session rankers.
