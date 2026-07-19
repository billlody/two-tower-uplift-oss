# Demos

Each demo reads from the **same** unified DGP (`tt_uplift.generate`) and verifies
against known ground truth. Two demos reproduce the paper's claim cleanly on the
default synthetic world; two run correctly but are **regime-dependent** — we say
so plainly rather than tuning the data to a desired outcome.

| Demo | Reproduces paper claim on default DGP? | Notes |
|------|----------------------------------------|-------|
| `confounding_balance.py` | **Yes, cleanly** | Raw corr +, within-stratum corr −, `|SMD|` 1.9→0.06, ATE sign flips. This is the balance/normalization argument (and matches the production Databricks notebook). |
| `device_level_fails.py` | **Yes, cleanly** | Naive device-level slope is + (wrong); within-stratum is − (correct). |
| `distillation_vs_aggregation.py` | **Yes** | The distilled device head reaches session NQini `~0.22` / device NQini `~0.10` (device/session ratio `~0.46`, matching the production "device Qini ≈ half session Qini" regime; see docs/DGP.md "Calibration to the production Qini ratio") and correlates `~0.83` with the device-level ground truth `τ_d`. Both methods emit one score/device, so it is a fair comparison; heuristic aggregation of a strong same-model teacher stays competitive on this clean DGP. The paper's larger 0.34→0.09 aggregation collapse comes from a *separate weaker teacher*, *hand-tuned recency weights over few recent sessions*, and a *temporal train/serve split* — production handicaps this minimal artifact does not simulate. |
| `alpha_pareto.py` | **Yes** | Sweeps the distillation weight α and plots the session-vs-device NQini frontier. The smooth trade-off is visible and higher α trends toward stronger device-level ranking. |
| `baseline_comparison.py` | **Yes (offline)** | Trains TowerUplift, TARNet, DragonNet, and CEVAE on the same device-disjoint split, then ranks them by held-out normalized Qini and correlation with the known per-session effect `gt_tau_session`. Directly answers the reviewer's "compare against deep uplift baselines (TARNet/DragonNet/CEVAE)" note **offline**, where the effect is observable. TowerUplift's session ranker leads the deep baselines here; CEVAE is weakest on this clean, low-latent-confounding DGP (it is designed for strong unobserved confounding). We do not tune the DGP to force any ordering. |

## Sweeping the regime

To explore where distillation overtakes aggregation, increase the per-session
teacher noise and shrink sessions-per-device (the production regime), e.g. edit a
demo to call:

```python
from tt_uplift import DGPConfig
DGPConfig(sessions_per_device_mean=2, outcome_noise=3.0, tau_content=1.0, feature_signal=1.2)
```

Aggregation over few noisy sessions becomes high-variance, narrowing or reversing
the gap. We deliberately do **not** ship a DGP tuned to force this — the honest
default is a clean world where aggregation is competitive.

## Why honesty here matters

This is a reproducibility artifact for a paper. Silently tuning synthetic data
until the method "wins" would undermine exactly the credibility the artifact is
meant to provide. The balance/normalization result — the paper's causal
foundation and the reviewer's actual question — reproduces cleanly and
independently in both this repo and the production-data notebook.
