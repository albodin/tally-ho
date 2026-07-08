"""Ballistic descent-rate model.

Terminal velocity under fixed drag is ``v_t(rho) = B · rho^(-1/2)`` where ``B``
is the payload's ballistic constant. We estimate ``B`` by fitting observed
``(v_obs, rho)`` descent samples:

* **Exclude the post-burst transient** (chute deployment / tumbling): drop the
  first ~45 s / ~1 km of descent.
* **Reject outliers** (single glitch frames) via a median/MAD gate.
* **Weight recent samples** more (exponential half-life).
* **Clamp ``B``** to physically sane bounds.

When too few samples exist, fall back to the single-point shortcut
``v(alt) = v_obs(now) · sqrt(rho(now)/rho)`` - algebraically the same model
seeded from one point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DescentConfig
from .models import DescentSample


@dataclass(slots=True)
class DescentModel:
    b: float                 # ballistic constant
    residual_mps: float      # RMS fit residual, m/s (0.0 for shortcut)
    n_points: int            # samples used (1 for shortcut)
    clamped: bool = False    # whether B hit a clamp bound

    def v_t(self, rho: float) -> float:
        """Terminal descent speed at density ``rho`` (m/s, positive down)."""
        if rho <= 0.0:
            return 0.0
        return self.b * rho ** -0.5


def shortcut_descent(v_obs: float, rho_now: float, cfg: DescentConfig) -> DescentModel:
    """Single-point model seeded from the current observed descent."""
    b = v_obs * rho_now ** 0.5
    clamped = not (cfg.b_min <= b <= cfg.b_max)
    b = min(max(b, cfg.b_min), cfg.b_max)
    return DescentModel(b=b, residual_mps=0.0, n_points=1, clamped=clamped)


def fit_descent(
    samples: list[DescentSample],
    cfg: DescentConfig,
    burst_t: float | None = None,
    burst_alt: float | None = None,
) -> DescentModel | None:
    """Robust, recency-weighted fit of ``B``.

    ``burst_t``/``burst_alt`` anchor the transient exclusion on the *actual*
    burst when known. Without them the anchor falls back to the oldest sample
    present - fine live, but after a daemon restart mid-descent (persisted
    samples reloaded, or none at all) the fallback would re-discard another
    ~45 s / ~1 km of perfectly good post-restart data as "transient".

    Returns None if, after excluding the transient, there are too few usable
    samples - the caller should fall back to :func:`shortcut_descent`.
    """
    if not samples:
        return None
    kept = _exclude_transient(samples, cfg, burst_t=burst_t, burst_alt=burst_alt)
    if len(kept) < cfg.min_fit_points:
        return None

    rho = np.array([s.rho for s in kept], dtype=float)
    v = np.array([s.v_obs for s in kept], dtype=float)
    t = np.array([s.t for s in kept], dtype=float)
    valid = (rho > 0) & (v > 0)
    rho, v, t = rho[valid], v[valid], t[valid]
    if len(rho) < cfg.min_fit_points:
        return None

    b_i = v * np.sqrt(rho)            # per-point ballistic constant

    # Regime change (plan Phase 1): the descent sometimes changes character
    # mid-fall - balloon remnants detach, or the chute finally inflates around
    # 5-10 km where the air thickens. When the recent per-point constants step
    # away from the older ones, reset: keep only post-change samples, so the
    # fit converges on the new chute instead of averaging two regimes. This
    # runs BEFORE outlier rejection: while the new regime is still the minority
    # of samples, the MAD gate would reject exactly those fresh points.
    start = _regime_change_start(b_i, cfg)
    if start > 0:
        b_i, t, rho, v = b_i[start:], t[start:], rho[start:], v[start:]

    # MAD outlier rejection (glitch frames).
    med = np.median(b_i)
    mad = np.median(np.abs(b_i - med)) or 1e-9
    inliers = np.abs(b_i - med) <= 3.5 * 1.4826 * mad
    b_i, t, rho, v = b_i[inliers], t[inliers], rho[inliers], v[inliers]
    if len(b_i) < cfg.min_fit_points:
        return None

    # Recency weighting (exponential half-life on age relative to newest sample).
    age = t.max() - t
    w = 0.5 ** (age / max(cfg.recency_halflife_s, 1e-6))
    b = float(np.sum(w * b_i) / np.sum(w))

    clamped = not (cfg.b_min <= b <= cfg.b_max)
    b = min(max(b, cfg.b_min), cfg.b_max)

    # Residual of the (clamped) model against the kept samples.
    pred_v = b * rho ** -0.5
    resid = float(np.sqrt(np.average((v - pred_v) ** 2, weights=w)))
    return DescentModel(b=b, residual_mps=resid, n_points=int(len(b_i)), clamped=clamped)


def _regime_change_start(b_i: np.ndarray, cfg: DescentConfig) -> int:
    """Index of the first sample of the *current* regime, or 0 when the
    series is consistent. Recent window vs the rest, then walk back from the
    end to find where the new regime began (plan Phase 1 failure mode)."""
    k = max(cfg.min_fit_points, cfg.regime_recent_points)
    if cfg.regime_change_rel <= 0 or len(b_i) < 2 * k:
        return 0
    recent = float(np.median(b_i[-k:]))
    older = float(np.median(b_i[:-k]))
    if older <= 0 or abs(recent - older) / older <= cfg.regime_change_rel:
        return 0
    tol = cfg.regime_change_rel / 2.0
    start = len(b_i) - 1
    while start > 0 and abs(float(b_i[start - 1]) - recent) / recent <= tol:
        start -= 1
    # never reset to fewer points than the detection window itself
    return min(start, len(b_i) - k)


def _exclude_transient(
    samples: list[DescentSample],
    cfg: DescentConfig,
    burst_t: float | None = None,
    burst_alt: float | None = None,
) -> list[DescentSample]:
    """Drop the post-burst transient: first ``transient_seconds`` and first
    ``transient_drop_m`` of descent, measured from the burst
    anchors when given, else from the samples themselves."""
    t0 = burst_t if burst_t is not None else min(s.t for s in samples)
    alt_top = burst_alt if burst_alt is not None else max(s.alt for s in samples)
    return [
        s for s in samples
        if (s.t - t0) >= cfg.transient_seconds and (alt_top - s.alt) >= cfg.transient_drop_m
    ]
