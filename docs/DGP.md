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
| Session `s` | — | device features ⊕ content features ⊕ `ad_load` ⊕ `watch_percent` |

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

Latent engagement in minutes carries the confounder and the true effect:

```
τ(θ_d)      = tau_base + tau_theta · θ_d          # heterogeneous, tau_base < 0
view_time_s = beta_duration · L_c                 # confounder term (large, +)
            + τ(θ_d) · ad_load_s                  # TRUE causal effect (negative)
            + noise
```

The **observed** outcome is **watch percent** — engagement normalized by content
length:

```
watch_percent_s = view_time_s / content_minutes_s
```

Dividing by duration puts the confounder in the denominator as well as the
`beta_duration · L_c` numerator, so it does **not** factor out cleanly under
content-stratified normalization the way raw `view_time` did. The **raw**
`corr(ad_load, watch_percent)` is still spuriously **positive** (weaker than for
`view_time`: ~`+0.16` vs ~`+0.30`), and stratified z-scoring within
`content_type × genre × duration-decile` still **flips** it to the true
**negative** effect — but the flip is less dramatic, which is intentional: it
brings the synthetic world closer to the messier production regime (see below).

## Ground-truth artifacts (for verification, not assertion)

Returned on the sessions frame (prefixed `gt_`) and in `device_truth`:

- `gt_theta` — latent device tolerance per session
- `gt_tau_session` — true per-session treatment effect `τ(θ_d)`
- `gt_session_uplift` — counterfactual `y(t_high) − y(t_low)` = `τ(θ_d) · (t_high − t_low) / content_minutes`
- `device_truth.tau_d_uplift` — device-level true uplift, the per-device **mean** of `gt_session_uplift` (with the watch-percent outcome the `1/content_minutes` factor no longer collapses to a closed form, so `τ_d` is defined empirically over each device's realized content mix)
- `stratum` — the normalization stratum id

## Signs each demo asserts

| Quantity | Raw / naive | Correct (within-stratum / model) |
|----------|-------------|----------------------------------|
| `corr(treatment, outcome)` | **positive** (confounded, ~`+0.16`) | **negative** (~`−0.30`; effect persists, not zero) |
| `|SMD|` of duration | `> 0.1` (imbalanced, ~`1.9`) | `< 0.1` (balanced, ~`0.06`) |
| ATE (high − low) | **positive** | **negative** |
| device-level slope (Demo 2) | **positive** (wrong) | **negative** (within-stratum) |
| device head vs `τ_d` (Demo 3) | — | `corr > 0.3` |

(Numbers are for the `watch_percent` outcome on the default `DGPConfig`; signs, not
magnitudes, are what the demos assert.)

> **Scope.** These diagnostics address *observed* content-mix confounding only.
> The synthetic world has no unmeasured confounders by construction; the paper
> validates the deployed policy via an online A/B test precisely because real
> logs do.

## Synthetic vs production: the demo is deliberately cleaner

This synthetic world is a **teaching instrument**, not a replica of production. In
the DGP the *only* confounder is content duration. With the `watch_percent`
outcome it enters through both the numerator and the denominator, so stratified
normalization **attenuates** rather than fully absorbs it — Demo 1 still shows a
sign **flip** (raw `corr +0.16`, within-stratum `corr −0.30`) and the confounder
`|SMD|` still drops below `0.1`, but the raw correlation is weaker than under the
raw-`view_time` outcome, moving the synthetic world a step toward the production
regime.

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
stratum can remove. The paper reports this honestly, leans on the online A/B test to
validate the *policy* rather than the offline point estimates, and flags a
completion-based label as a less-confounded alternative.

**Why keep the synthetic world (relatively) clean?** To isolate one mechanism at a
time for pedagogy: the synthetic DGP's only confounder is content duration. Note the
consistency with the production ladder above — switching production to a
`watch_percent` completion label collapsed most of the residual confounding, and the
synthetic DGP now uses the same completion outcome, which is why its raw correlation
is modest (`+0.16`) rather than large. Demo 1 still flips the sign, whereas
production's *view-time* label does not. Set `DGPConfig(tau_content=...,
outcome_noise=...)` higher, or add an endogenous ad-exposure term (view-time → ad
count), to approach the messier production regime. The `stratified_normalization_diagnostics` Databricks
notebook reproduces the production numbers above (the de-confounding ladder on the
binary `norm3_tvt_sec_label`).
