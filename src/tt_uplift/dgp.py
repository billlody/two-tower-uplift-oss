"""Unified synthetic data-generating process (DGP) for the uplift artifact.

One generative world of *devices*, *content*, and *sessions* with **known ground
truth**.  Every demo in this repository draws from a single call to
:func:`generate`, so all four arguments in the paper are reproduced from the same
data rather than from four hand-tuned datasets.

The causal story (matches the paper's ad-load setting)
------------------------------------------------------
* Each **device** ``d`` has a latent **tolerance** ``theta_d ~ N(0, 1)`` — how much
  ad load it absorbs before disengaging.  It is *never observed directly*; the model
  must recover it from noisy device features.
* Each **content** item ``c`` has a **duration** ``L_c`` (the confounder), a genre,
  and a type (movie/series).
* A device's sessions draw content from *its own* taste distribution, so devices
  that like long content systematically watch long content.  This per-device content
  mixing is what makes naive device-level causal estimation fail.
* **Treatment** (ad load) is confounded on purpose: longer content carries more/denser
  ad breaks, ``ad_load = g(L_c) + noise``.
* **Outcome** (engagement) has a KNOWN causal effect::

      view_time = beta * L_c            # confounder's large effect on raw minutes
                + tau(theta_d) * ad_load # TRUE causal effect, tau < 0 (heterogeneous)
                + noise

  Because ``beta * L_c`` dominates and correlates with ``ad_load``, the *raw*
  correlation ``corr(ad_load, view_time)`` is spuriously **positive** — the
  "more ads => more engagement" paradox.  Stratified normalization within
  duration x genre x type removes ``beta * L_c`` and exposes the true negative effect.

Ground truth returned for verification (not assertion)
------------------------------------------------------
``theta_d``, ``tau(theta_d)`` per session, device-level true uplift ``tau_d``,
stratum assignment, and the counterfactual outcomes under high/low ad load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

# Small, legible feature set (the "very naive, simple features" requirement).
DEVICE_NUMERIC: List[str] = [
    "avg_session_len",
    "sessions_per_day",
    "hist_ad_exposure",
    "completion_rate",
    "days_active",
]
DEVICE_CATEGORICAL: List[str] = ["device_type"]  # TV / mobile / web
CONTENT_NUMERIC: List[str] = ["duration", "release_year"]
CONTENT_CATEGORICAL: List[str] = ["genre", "content_type"]
TREATMENT_COL: str = "ad_load"
RAW_OUTCOME_COL: str = "view_time"

_DEVICE_TYPES = ["tv", "mobile", "web"]
_GENRES = ["action", "comedy", "drama", "doc", "kids"]
_CONTENT_TYPES = ["movie", "series"]


@dataclass(frozen=True)
class DGPConfig:
    """Parameters of the synthetic world.

    All effects have documented signs so demos can assert against them.
    """

    n_devices: int = 4000
    sessions_per_device_mean: float = 12.0
    n_content: int = 800

    # Confounder strength: duration -> raw view time.  Large & positive so it
    # dominates the raw correlation and creates the spurious positive association.
    beta_duration: float = 1.2

    # Duration -> ad load (confounding channel).  Positive: longer content, more ads.
    duration_to_adload: float = 0.9
    adload_noise: float = 0.5

    # TRUE causal effect of ad load on (normalized) engagement.  NEGATIVE.
    # tau_session = tau_base + tau_theta * theta        (device-level component)
    #             + tau_content * content_effect        (session-level component)
    # The device-level target tau_d marginalizes the content component to zero, so
    # a device policy only needs the device component.  The session component is
    # what makes heuristic aggregation of FEW noisy session scores high-variance,
    # while the distilled device head (device features only) recovers tau_d by
    # pooling across all devices.
    tau_base: float = -0.60
    tau_theta: float = 0.35  # device heterogeneity: more tolerant -> less negative
    tau_content: float = 0.55  # session-level effect heterogeneity (mean zero over content)

    # How strongly latent tolerance shows up in (noisy) observed device features.
    feature_signal: float = 0.8
    feature_noise: float = 0.6

    outcome_noise: float = 0.7
    duration_deciles: int = 10
    seed: int = 42

    # Content-taste coupling: correlation between a device's tolerance and the
    # mean duration of content it watches (drives device-level confounding).
    taste_duration_coupling: float = 1.0


@dataclass
class SyntheticData:
    """Container for the generated sessions plus ground-truth arrays.

    Attributes
    ----------
    sessions : pandas.DataFrame
        One row per session with observable features, treatment, raw outcome,
        stratum id, device id, and (for convenience) the ground-truth columns
        prefixed with ``gt_``.
    device_truth : pandas.DataFrame
        One row per device: latent ``theta``, device-level true uplift ``tau_d``.
    config : DGPConfig
    """

    sessions: pd.DataFrame
    device_truth: pd.DataFrame
    config: DGPConfig
    meta: Dict = field(default_factory=dict)


def _standardize(x: np.ndarray) -> np.ndarray:
    mu = x.mean()
    sd = x.std()
    return (x - mu) / (sd if sd > 1e-8 else 1.0)


def generate(config: DGPConfig | None = None) -> SyntheticData:
    """Generate the unified synthetic dataset.

    Parameters
    ----------
    config : DGPConfig, optional
        World parameters.  Defaults to :class:`DGPConfig`.

    Returns
    -------
    SyntheticData
    """
    cfg = config or DGPConfig()
    rng = np.random.default_rng(cfg.seed)

    # ---- Devices -----------------------------------------------------------
    n_d = cfg.n_devices
    theta = rng.normal(0.0, 1.0, size=n_d)  # latent tolerance (unobserved)
    device_type_idx = rng.integers(0, len(_DEVICE_TYPES), size=n_d)

    # Observed device features: noisy correlates of theta.
    # Each feature loads on theta with a different sign/scale, plus noise.
    loadings = np.array([0.9, 0.4, 1.1, 0.7, 0.3])
    dev_numeric = np.zeros((n_d, len(DEVICE_NUMERIC)))
    for j, load in enumerate(loadings):
        dev_numeric[:, j] = (
            cfg.feature_signal * load * theta
            + cfg.feature_noise * rng.normal(0.0, 1.0, size=n_d)
        )
    # hist_ad_exposure (col 2) should read higher for tolerant devices -> keep sign.
    # completion_rate (col 3) mapped to (0,1) via logistic for realism.
    dev_numeric[:, 3] = 1.0 / (1.0 + np.exp(-dev_numeric[:, 3]))
    dev_numeric[:, 4] = np.abs(dev_numeric[:, 4]) * 100 + 10  # days_active positive

    # Each device's content-taste center (mean preferred duration), coupled to theta.
    taste_center = cfg.taste_duration_coupling * theta + rng.normal(0.0, 0.5, size=n_d)

    # ---- Content catalog ---------------------------------------------------
    n_c = cfg.n_content
    # Duration is standardized latent; also kept as a positive "minutes" scale.
    content_dur_latent = rng.normal(0.0, 1.0, size=n_c)
    content_minutes = np.clip(30 + 25 * content_dur_latent, 3, 240)
    content_genre = rng.integers(0, len(_GENRES), size=n_c)
    content_type = rng.integers(0, len(_CONTENT_TYPES), size=n_c)
    release_year = rng.integers(1990, 2027, size=n_c)

    # ---- Sessions ----------------------------------------------------------
    n_sessions_per = rng.poisson(cfg.sessions_per_device_mean, size=n_d)
    n_sessions_per = np.clip(n_sessions_per, 2, None)  # need >=2 for within-device var
    total = int(n_sessions_per.sum())

    dev_ids = np.repeat(np.arange(n_d), n_sessions_per)

    # For each session, pick content whose latent duration is near the device's taste
    # center — this induces per-device content sets (device-level confounding).
    # Sample by scoring all content against taste and softmax-sampling.
    chosen_content = np.empty(total, dtype=np.int64)
    ptr = 0
    for d in range(n_d):
        k = n_sessions_per[d]
        # Preference weight: closeness of content latent duration to device taste.
        logits = -0.5 * (content_dur_latent - taste_center[d]) ** 2
        p = np.exp(logits - logits.max())
        p /= p.sum()
        chosen_content[ptr : ptr + k] = rng.choice(n_c, size=k, p=p)
        ptr += k

    sess_dur_latent = content_dur_latent[chosen_content]
    sess_theta = theta[dev_ids]
    sess_minutes = content_minutes[chosen_content]
    sess_genre = content_genre[chosen_content]
    sess_ctype = content_type[chosen_content]

    # ---- Strata: content_type x genre x duration-decile (built BEFORE treatment
    #      because ad load is planned per content tier = stratum) ----------------
    dur_bin = pd.qcut(sess_minutes, q=cfg.duration_deciles, labels=False, duplicates="drop")
    stratum = np.char.add(
        np.char.add(
            np.char.add(np.array(_CONTENT_TYPES)[sess_ctype].astype(str), "__"),
            np.char.add(np.array(_GENRES)[sess_genre].astype(str), "__"),
        ),
        dur_bin.astype(str),
    )

    # Treatment (ad load), confounded by duration at the *content-tier* level.
    # Ad load / fill rate is planned per stratum (content tier), not per exact
    # minute; within a stratum the realized ad load is driven by market /
    # time-of-day noise, roughly independent of the residual duration.  This is
    # exactly what lets stratified normalization (which conditions on the stratum)
    # balance the confounder: the confounding lives at the stratum granularity.
    stratum_mean_latent = pd.Series(sess_dur_latent).groupby(stratum).transform("mean").to_numpy()
    ad_load = (
        cfg.duration_to_adload * stratum_mean_latent
        + cfg.adload_noise * rng.normal(0.0, 1.0, size=total)
    )

    # Heterogeneous TRUE causal effect (negative).
    #   device component:  tau_base + tau_theta * theta_d   (what a device policy needs)
    #   session component: tau_content * eps_content        (mean ~0 over content)
    # eps_content is a per-content random modifier drawn once for the catalog; it
    # averages toward zero across a device's content mix, so the device-level
    # target tau_d is unchanged, but individual sessions carry extra variation.
    content_effect = rng.normal(0.0, 1.0, size=n_c)
    sess_content_effect = content_effect[chosen_content]
    tau_session = (
        cfg.tau_base
        + cfg.tau_theta * sess_theta
        + cfg.tau_content * sess_content_effect
    )

    # Raw outcome = confounder term + causal term + noise.
    noise = cfg.outcome_noise * rng.normal(0.0, 1.0, size=total)
    view_time = (
        cfg.beta_duration * sess_dur_latent
        + tau_session * ad_load
        + noise
    )

    # Counterfactual outcomes under fixed high/low ad load (ground truth for
    # session/device uplift).  Use +1 / -1 std as high/low, matching the model's
    # default t_high / t_low.
    t_high, t_low = 1.0, -1.0
    y_high = cfg.beta_duration * sess_dur_latent + tau_session * t_high + noise
    y_low = cfg.beta_duration * sess_dur_latent + tau_session * t_low + noise
    gt_session_uplift = y_high - y_low  # = tau_session * (t_high - t_low)

    # ---- Assemble sessions frame ------------------------------------------
    df = pd.DataFrame(
        {
            "device_id": dev_ids,
            "content_id": chosen_content,
            # Device features (broadcast to sessions)
            "avg_session_len": dev_numeric[dev_ids, 0],
            "sessions_per_day": dev_numeric[dev_ids, 1],
            "hist_ad_exposure": dev_numeric[dev_ids, 2],
            "completion_rate": dev_numeric[dev_ids, 3],
            "days_active": dev_numeric[dev_ids, 4],
            "device_type": np.array(_DEVICE_TYPES)[device_type_idx[dev_ids]],
            # Content features
            "duration": content_minutes[chosen_content],
            "release_year": release_year[chosen_content].astype(float),
            "genre": np.array(_GENRES)[content_genre[chosen_content]],
            "content_type": np.array(_CONTENT_TYPES)[content_type[chosen_content]],
            # Treatment & outcome
            TREATMENT_COL: ad_load,
            RAW_OUTCOME_COL: view_time,
            # Ground truth (prefixed gt_)
            "gt_theta": sess_theta,
            "gt_tau_session": tau_session,
            "gt_session_uplift": gt_session_uplift,
            "gt_duration_latent": sess_dur_latent,
            "stratum": stratum,
        }
    )

    # ---- Device-level ground-truth uplift: tau_d = E_c[tau] per device -----
    # The session-level content component (tau_content * eps_content) has mean ~0
    # over the content distribution, so it marginalizes out: the device-level
    # target is tau_d = (tau_base + tau_theta * theta_d) * (t_high - t_low).
    # This is exactly what a device-only policy should recover.
    tau_d = (cfg.tau_base + cfg.tau_theta * theta) * (t_high - t_low)
    device_truth = pd.DataFrame(
        {
            "device_id": np.arange(n_d),
            "theta": theta,
            "tau_d_uplift": tau_d,
        }
    )

    meta = {
        "t_high": t_high,
        "t_low": t_low,
        "n_sessions": total,
        "true_effect_sign": "negative",
        "raw_corr_sign": "positive (confounded)",
    }
    return SyntheticData(sessions=df, device_truth=device_truth, config=cfg, meta=meta)


def stratified_zscore(
    df: pd.DataFrame,
    value_col: str,
    stratum_col: str = "stratum",
    out_col: str | None = None,
) -> pd.DataFrame:
    """Add a within-stratum z-scored column (paper Eq. 2 / production ``norm2``).

    Parameters
    ----------
    df : DataFrame
    value_col : str
        Column to normalize (e.g. the raw outcome).
    stratum_col : str
        Stratum id column.
    out_col : str, optional
        Output column name; defaults to ``f"norm_{value_col}"``.

    Returns
    -------
    DataFrame
        Copy of ``df`` with the normalized column added.
    """
    out = out_col or f"norm_{value_col}"
    grp = df.groupby(stratum_col)[value_col]
    mean = grp.transform("mean")
    std = grp.transform("std").replace(0.0, np.nan)
    res = df.copy()
    res[out] = ((df[value_col] - mean) / std).fillna(0.0)
    return res
