"""Tests for the ballistic descent fit."""

import pytest

from windfall.atmosphere import isa_density
from windfall.config import DescentConfig
from windfall.descent import DescentModel, fit_descent, shortcut_descent
from windfall.tracker import DescentSample


def _samples(b_true=5.5, n=120, alt_top=30000.0, rate=5.0, dt=2.0):
    """Synthetic descent samples consistent with v_t = b·rho^-1/2."""
    out = []
    alt = alt_top
    t = 0.0
    while alt > 300 and len(out) < n:
        rho = isa_density(alt)
        v = b_true * rho ** -0.5
        out.append(DescentSample(t=t, alt=alt, v_obs=v, rho=rho))
        alt -= v * dt
        t += dt
    return out


def test_recovers_b():
    s = _samples(b_true=5.5)
    m = fit_descent(s, DescentConfig())
    assert m is not None
    assert m.b == pytest.approx(5.5, abs=0.1)
    assert m.residual_mps < 0.5
    assert not m.clamped


def test_v_t_relation():
    m = DescentModel(b=5.5, residual_mps=0.0, n_points=10)
    # at sea-level density ~1.225 → ~5 m/s
    assert m.v_t(1.225) == pytest.approx(5.5 / 1.225 ** 0.5, rel=1e-9)
    # thinner air → faster
    assert m.v_t(0.05) > m.v_t(1.0)


def test_transient_excluded():
    s = _samples(b_true=5.5)
    # corrupt the first 5 samples (post-burst tumbling) with huge speeds
    for i in range(5):
        s[i].v_obs *= 5
    m = fit_descent(s, DescentConfig())
    # transient excluded → still recovers true B
    assert m.b == pytest.approx(5.5, abs=0.3)


def test_glitch_frame_rejected():
    s = _samples(b_true=5.5)
    s[60].v_obs = 999.0   # single glitch deep in descent
    m = fit_descent(s, DescentConfig())
    assert m.b == pytest.approx(5.5, abs=0.3)


def test_clamp():
    s = _samples(b_true=5.5)
    cfg = DescentConfig(b_max=4.0)   # force clamp
    m = fit_descent(s, cfg)
    assert m.b == 4.0
    assert m.clamped


def test_too_few_points_returns_none():
    s = _samples()[:2]
    assert fit_descent(s, DescentConfig()) is None


def test_shortcut():
    rho = isa_density(5000)
    v = 5.5 * rho ** -0.5
    m = shortcut_descent(v, rho, DescentConfig())
    assert m.b == pytest.approx(5.5, abs=0.01)
    assert m.n_points == 1


def test_regime_change_resets_to_new_chute():
    """Chute character changes mid-fall: the fit must jump to
    the new regime, not average the two - and the old MAD gate must not reject
    the still-minority fresh samples (the cut runs first)."""
    from windfall.config import DescentConfig
    from windfall.descent import fit_descent
    from windfall.models import DescentSample

    def series():
        out = [DescentSample(t=0.0, alt=12_000.0, v_obs=8.0, rho=1.0)]  # transient
        for i in range(50):
            b = 8.0 if i < 35 else 5.5          # remnants detach at sample 35
            noise = 0.03 if i % 2 else -0.03    # keep MAD non-degenerate
            out.append(DescentSample(t=45.0 + 5.0 * i, alt=11_000.0 - 50.0 * i,
                                     v_obs=b + noise, rho=1.0))
        return out

    cfg = DescentConfig()
    model = fit_descent(series(), cfg)
    assert model is not None and not model.clamped
    assert model.b == pytest.approx(5.5, abs=0.1)

    # with detection disabled, the majority old regime wins (the MAD gate
    # rejects the fresh minority) - exactly the failure mode being fixed
    cfg_off = DescentConfig(regime_change_rel=0.0)
    model_off = fit_descent(series(), cfg_off)
    assert model_off.b == pytest.approx(8.0, abs=0.2)


def test_regime_detector_quiet_on_consistent_series():
    import numpy as np
    from windfall.config import DescentConfig
    from windfall.descent import _regime_change_start

    b_i = np.full(60, 6.0) + np.linspace(-0.05, 0.05, 60)
    assert _regime_change_start(b_i, DescentConfig()) == 0
    # too short to judge → no reset
    assert _regime_change_start(b_i[:10], DescentConfig()) == 0


def test_burst_anchors_prevent_double_transient_exclusion():
    """After a restart with no surviving samples, fresh post-restart points must
    not be re-discarded as 'post-burst transient': the anchors say burst
    happened long ago and kilometres higher (restart hardening)."""
    cfg = DescentConfig()
    # 10 samples over 10 s, deep into the descent (15 km; burst was at 30 km)
    s = []
    for i in range(10):
        alt = 15000.0 - 8.0 * i
        rho = isa_density(alt)
        s.append(DescentSample(t=600.0 + i, alt=alt,
                               v_obs=5.5 * rho ** -0.5, rho=rho))
    # without anchors the whole batch looks like a fresh transient -> no fit
    assert fit_descent(s, cfg) is None
    # with anchors the very same samples fit immediately
    m = fit_descent(s, cfg, burst_t=0.0, burst_alt=30000.0)
    assert m is not None
    assert m.n_points == 10
    assert m.b == pytest.approx(5.5, rel=0.02)
