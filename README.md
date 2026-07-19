# Two-Tower Session-to-Device Uplift Distillation

A minimal, single-machine **reproducibility artifact** for the KDD ADS paper on
learning **deployment-level (device) policies from session-level counterfactual
supervision**. It reproduces the paper's core arguments on a **synthetic dataset
with a known causal ground truth**, using a stripped-down port of the production
two-tower uplift model (no Spark, no distributed training, no proprietary data).

> **Why synthetic?** The paper's claims are causal. On real logs the ground-truth
> treatment effect is unobservable, so a reviewer cannot check whether the method
> *recovers* it. Here the data-generating process (DGP) has a **known** negative
> causal effect and a **known** duration confounder, so every demo *verifies*
> recovery against ground truth instead of asserting it.

## The one idea

Supervision is available per **session** (one user, one title, one context), but
the policy must be deployed per **device**. The standard remedy — train a session
model and heuristically average its scores to the device level — loses signal and
complicates serving. This repo learns a **device representation** end-to-end and
**distills** the session-level uplift into a device-only head, so serving is an
`O(1)` forward pass over device features.

## Install

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
```

## Reproduce the paper's arguments

Each demo reads from the **same** unified DGP (`tt_uplift.generate`) and writes a
table/plot into `outputs/`.

| Demo | Command | Paper element | Reproduces cleanly? |
|------|---------|---------------|---------------------|
| Confounding & balance | `python demos/confounding_balance.py` | §Offline Eval, balance table + FWL figure | **Yes.** Raw `corr(ad_load, view_time)` is spuriously **positive**; within-stratum it flips **negative** (effect persists, not zero); confounder `|SMD|` shrinks from ~1.9 to <0.1. Matches the production Databricks notebook. |
| Device-level fails | `python demos/device_level_fails.py` | §Offline Eval, "more ads = more engagement" paradox | **Yes.** Naive device-level regression recovers the **wrong (positive)** effect sign because per-device content sets differ; session decomposition fixes it. |
| Distillation vs aggregation | `python demos/distillation_vs_aggregation.py` | Table `agg_vs_distill` | **Yes.** The distilled device head reaches session NQini `~0.22` / device NQini `~0.10` — a device/session ratio of `~0.46`, matching the production regime where device Qini is about half session Qini. Its correlation with the device-level ground truth `τ_d` is `~0.83`. Heuristic aggregation of a strong same-model teacher stays competitive on this clean DGP; the paper's larger 0.34→0.09 aggregation collapse depends on production handicaps (separate weaker teacher, hand-tuned recency weights, temporal split) this minimal artifact does not simulate. See [`demos/README.md`](demos/README.md). |
| α Pareto | `python demos/alpha_pareto.py` | Figure `pareto` | **Yes.** Sweeps α and plots the session-vs-device NQini frontier; the smooth trade-off is visible and higher α trends toward stronger device-level ranking. |
| Baseline comparison | `python demos/baseline_comparison.py` | §Offline Eval, response to "compare against deep uplift baselines" | **Yes (offline).** Trains TowerUplift, **TARNet**, **DragonNet**, and **CEVAE** on the same device-disjoint split and ranks them by held-out normalized Qini + correlation with the known effect. TowerUplift's session ranker leads the deep baselines on the synthetic world; reported without tuning the DGP to force it. |

> **A note on integrity.** This artifact does **not** tune the synthetic DGP until
> the method "wins." The paper's causal foundation — the balance/normalization
> argument, which is also the reviewer question that motivated this repo —
> reproduces cleanly here and in the production-data notebook. The two
> ranking-comparison demos are reported for what they are: correct methodology
> whose outcome depends on the data regime.

Run everything:

```bash
for d in confounding_balance device_level_fails distillation_vs_aggregation alpha_pareto baseline_comparison; do
  python demos/$d.py
done
```

## What was ported (and dropped) from production

**Kept (faithful core):** two-tower S-learner with Hadamard treatment
interactions (`model.py`), device-uplift **distillation** head, optional
**DoubleML** treatment residualization, **TARNet** + **DragonNet** + **CEVAE** NN
uplift baselines (`model.py`, `cevae.py`), pairwise **ranking loss**
(`losses.py`), **AUCC / normalized Qini** (`evaluation.py`), stratified z-score
normalization (`dgp.stratified_zscore`), heuristic aggregation baselines
(`aggregation.py`), and balance diagnostics (`diagnostics.py`).

**Dropped (production-only, not needed for the arguments):** DCN-V2 cross layers,
sequence transformers over watch/ad history, bucket & high-cardinality ID
embeddings, Spark→TFRecord pipelines, multi-GPU/distributed training, and the
Databricks serving path.

## Package layout

```
src/tt_uplift/
├── dgp.py            # unified synthetic world + stratified normalization
├── features.py       # DataFrame -> tensors, encoders, treatment binarization
├── model.py          # TwoTowerUpliftModel (+ DoubleML), TARNet, DragonNet
├── cevae.py          # CEVAE baseline (latent-confounder VAE, Louizos et al. 2017)
├── losses.py         # pairwise ranking loss
├── trainer.py        # single-machine training loop + prediction helpers
├── aggregation.py    # heuristic session->device aggregation baselines
├── evaluation.py     # AUCC / normalized Qini / uplift curve
└── diagnostics.py    # balance report (corr, SMD, ATE contrast)
demos/                # five scripts: four paper arguments + baseline comparison
tests/                # ground-truth sign checks + metric sanity + baseline coverage
docs/DGP.md           # the data-generating process and its known ground truth
```

## Ground truth

See [`docs/DGP.md`](docs/DGP.md) for the generative equations and the exact signs
each demo asserts.

## License

MIT — see [LICENSE](LICENSE).
