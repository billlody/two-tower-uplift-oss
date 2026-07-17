# The Data-Generating Process (Ground Truth)

Every demo draws from a single call to `tt_uplift.generate()`, so all four
arguments in the paper come from **one** synthetic world with a **known** causal
structure. This document states the generative equations and the exact signs the
demos verify.

## Entities

| Entity | Latent | Observed |
|--------|--------|----------|
| Device `d` | tolerance `θ_d ~ N(0,1)` (unobserved) | 5 numeric + `device_type`, all noisy correlates of `θ_d` |
| Content `c` | latent duration `L_c ~ N(0,1)` | `duration` (minutes), `release_year`, `genre`, `content_type` |
| Session `s` | — | device features ⊕ content features ⊕ `ad_load` ⊕ `view_time` |

Device features load on `θ_d` with per-feature weights plus Gaussian noise, so
the model must **recover** tolerance from noisy signals (as in production, where
true tolerance is never observed).

## Content taste (drives device-level confounding)

Each device has a taste center coupled to `θ_d`; a session's content is sampled
so that content near the device's taste is more likely. Consequently devices
watch **different content mixes**, which is why duration confounding cannot be
stratified away at the device grain (Demo 2).

## Treatment (ad load), confounded on purpose

Ad load is planned at the **content-tier = stratum** level
(`content_type × genre × duration-decile`):

```
ad_load_s = duration_to_adload · mean_latent_duration(stratum(s)) + noise
```

Within a stratum the realized ad load is driven by market/time-of-day noise,
roughly independent of the residual duration — so conditioning on the stratum
(stratified normalization) balances the confounder.

## Outcome, with a KNOWN causal effect

```
τ(θ_d)      = tau_base + tau_theta · θ_d          # heterogeneous, tau_base < 0
view_time_s = beta_duration · L_c                 # confounder term (large, +)
            + τ(θ_d) · ad_load_s                  # TRUE causal effect (negative)
            + noise
```

Because `beta_duration · L_c` dominates and correlates with `ad_load`, the **raw**
`corr(ad_load, view_time)` is spuriously **positive** — the "more ads = more
engagement" paradox. Stratified z-scoring within `content_type × genre ×
duration-decile` removes the `L_c` term and exposes the true **negative** effect.

## Ground-truth artifacts (for verification, not assertion)

Returned on the sessions frame (prefixed `gt_`) and in `device_truth`:

- `gt_theta` — latent device tolerance per session
- `gt_tau_session` — true per-session treatment effect `τ(θ_d)`
- `gt_session_uplift` — counterfactual `y(t_high) − y(t_low)`
- `device_truth.tau_d_uplift` — device-level true uplift `τ_d = E_c[τ] · (t_high − t_low)`
- `stratum` — the normalization stratum id

## Signs each demo asserts

| Quantity | Raw / naive | Correct (within-stratum / model) |
|----------|-------------|----------------------------------|
| `corr(treatment, outcome)` | **positive** (confounded) | **negative** (effect persists, not zero) |
| `|SMD|` of duration | `> 0.1` (imbalanced) | `< 0.1` (balanced) |
| ATE (high − low) | **positive** | **negative** |
| device-level slope (Demo 2) | **positive** (wrong) | **negative** (within-stratum) |
| device head vs `τ_d` (Demo 3) | — | `corr > 0.3` |

> **Scope.** These diagnostics address *observed* content-mix confounding only.
> The synthetic world has no unmeasured confounders by construction; the paper
> validates the deployed policy via an online A/B test precisely because real
> logs do.

## Synthetic vs production: the demo is deliberately cleaner

This synthetic world is a **teaching instrument**, not a replica of production. In
the DGP the *only* confounder is content duration, and it acts through a stratum-
level channel that stratified normalization can fully absorb — so Demo 1 shows a
clean sign **flip** (raw `corr +`, within-stratum `corr −`) and the confounder
`|SMD|` drops below `0.1`.

Production data does **not** behave this cleanly. Measured on the *actual training
label* `norm3_tvt_sec_label` (binary; z-score of `tvt_sec` within
content-type × series-ratio × autoplay-ratio device-pattern strata, thresholded at
0) on ~305M sessions (`train_2026-07-01`), stratified normalization **barely
reduces** the treatment→label confounding:

| Diagnostic | Raw / overall | Within-stratum |
|------------|---------------|----------------|
| `corr(ad load, norm3_tvt_sec_label)` | `+0.268` | `+0.236` |
| logistic slope on label (OR per SD) | — | `+0.611` (OR `1.84`) |
| `|SMD|` of `video_duration` | `0.410` | `0.222` |

The within-stratum association stays **strongly positive** — a session with +1 SD
more ad load is `1.84×` more likely to be labeled high-engagement, the same sign as
confounding and opposite the true (negative) ad effect. The device-pattern strata
do **not** neutralize it (`+0.268 → +0.236`).

A **de-confounding ladder** localizes the residual to one channel — the outcome
metric, not the strata:

| Rung (cumulative) | Within-stratum corr(ad load, label) |
|-------------------|-------------------------------------|
| within-stratum, view-time label | `+0.236` |
| + drop bottom-20% ad load | `+0.178` |
| + **completion (`watch_percent`) label** | `+0.060` |

Refining duration resolution (`+0.177 → 0.174` across 10→50 deciles) and
residualizing on device ad-history (`+0.162`) leave it essentially unchanged,
ruling out under-resolved duration and device confounding. Switching the outcome to
a per-content completion metric (`watch_percent`) — not mechanically inflated by
longer watching — collapses it ~66%. The residual is therefore a **reverse-causality
artifact of the view-time outcome** (longer watching mechanically triggers more
midroll ad breaks: engagement → ad exposure), which no content- or device-side
stratum can remove. The paper reports this honestly and leans on the online A/B test
to validate the *policy* rather than the offline point estimates.

**Why not simply switch to the completion label?** Because a less-confounded label
scores *lower* offline AUCC, and that is expected, not disqualifying. AUCC is computed
against the observed (confounded) outcome, so it partially rewards fitting the
confounding-driven treated−control gap; it is valid only for comparing models that
share a label (confounding held fixed) and is **not comparable across labels** of
differing confounding. A completion label's lower AUCC reflects *less confounding in
the evaluation outcome*, not a worse causal target. The production label is retained
because (i) label choice cannot be made from offline AUCC, and (ii) the view-time
label aligns with the platform's view-time/revenue objectives, which the online
experiment validates. The completion result is thus evidence that offline magnitudes
are confounded — not a label recommendation.

**Why keep the synthetic world clean?** To isolate one mechanism at a time for
pedagogy: the synthetic DGP's only confounder is content duration, acting through a
stratum-level channel normalization can fully absorb — so Demo 1 shows a clean sign
flip that production does **not**. Set `DGPConfig(tau_content=..., outcome_noise=...)`
higher, or add an endogenous ad-exposure term (view-time → ad count), to approach the
messier production regime. The `stratified_normalization_diagnostics` Databricks
notebook reproduces the production numbers above (the de-confounding ladder on the
binary `norm3_tvt_sec_label`).
