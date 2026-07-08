"""Tests for density models."""

import pytest

from windfall.atmosphere import RHO0, isa_density, measured_density


def test_isa_sea_level():
    assert isa_density(0) == pytest.approx(1.225, abs=0.005)


def test_isa_11km():
    # Standard atmosphere density at 11 km ~ 0.3639 kg/m^3
    assert isa_density(11_000) == pytest.approx(0.3639, abs=0.01)


def test_isa_20km():
    # ~0.0889 kg/m^3 at 20 km
    assert isa_density(20_000) == pytest.approx(0.0889, abs=0.005)


def test_isa_monotonic_decreasing():
    prev = isa_density(0)
    for alt in range(1000, 35000, 1000):
        cur = isa_density(alt)
        assert cur < prev
        prev = cur


def test_measured_density_matches_isa_at_sea_level():
    # ISA sea level: 1013.25 hPa, 15 C
    rho = measured_density(1013.25, 15.0)
    assert rho == pytest.approx(1.225, abs=0.005)


def test_measured_density_rejects_nonphysical():
    assert measured_density(0, 15) is None
    assert measured_density(1000, -300) is None  # T below 0 K
    assert measured_density(None, 15) is None
