"""Instrument uncertainty models for the ferromagnetism experiment."""
from __future__ import annotations

import numpy as np

from taulab.stats import resolution_sigma

__all__ = [
    "ruler",
    "caliper",
    "reading_lsd",
    "column_lsd",
    "digital_multimeter_resistance",
    "oscilloscope_dual_cursor",
]


def ruler(resolution: float = 1e-3) -> float:
    """σ for a ruler reading (rectangular window / √12; default 1 mm)."""
    return resolution_sigma(resolution)


def caliper(resolution: float = 0.05e-3) -> float:
    """σ for a Vernier caliper reading (default 0.05 mm)."""
    return resolution_sigma(resolution)


def reading_lsd(v: float) -> float:
    """Smallest decimal place in ``v``'s printed form; 0 for NaN/±∞/0."""
    f = float(v)
    if not np.isfinite(f) or f == 0.0:
        return 0.0
    s = format(abs(f), ".15g")
    if "e" in s:
        mant, exp_str = s.split("e")
        decimals = len(mant.split(".")[1]) if "." in mant else 0
        return 10.0 ** (int(exp_str) - decimals)
    if "." in s:
        return 10.0 ** -len(s.split(".")[1])
    return 10.0 ** (len(s) - len(s.rstrip("0")))


def column_lsd(values) -> float:
    """Smallest positive LSD across ``values``; 0 if none."""
    positives = np.array([reading_lsd(float(v)) for v in values])
    positives = positives[positives > 0]
    return float(positives.min()) if positives.size else 0.0


def _rectangular_bound_sigma(bound):
    return np.asarray(bound, dtype=float) / np.sqrt(3.0)


# Keysight 34401A, 1-year (manual p. 216): full-scale range → (%rdg, %range).
_DMM_R_RANGES: dict[float, tuple[float, float]] = {
    100:       (0.010e-2, 0.004e-2),
    1_000:     (0.010e-2, 0.001e-2),
    10_000:    (0.010e-2, 0.001e-2),
    100_000:   (0.010e-2, 0.001e-2),
    1_000_000: (0.010e-2, 0.001e-2),
}


def digital_multimeter_resistance(R: float, *, include_lsd: bool = True) -> float:
    """Keysight 34401A 1-year σ on the smallest range that fits ``R``.

    The manual's ``±(%rdg + %range)`` accuracy is a Type-B bound, so it
    is converted to a standard uncertainty with a rectangular model.
    Disable ``include_lsd`` when the logged value is already rounded and
    the display quantisation would be double-counted.
    """
    rng = next(r for r in sorted(_DMM_R_RANGES) if abs(R) <= r)
    pct_rdg, pct_rng = _DMM_R_RANGES[rng]
    sigma = float(_rectangular_bound_sigma(pct_rdg * abs(R) + pct_rng * rng))
    if include_lsd:
        sigma = float(np.hypot(sigma, reading_lsd(R) / np.sqrt(12)))
    return float(sigma)


def oscilloscope_dual_cursor(V, *, gain_frac: float = 0.024,
                             floor: float = 5e-3, lsd: float | None = None):
    """Keysight DSO7012A dual-cursor ΔV standard uncertainty.

    The datasheet's ``±(gain·|V| + floor)`` cursor accuracy is treated as
    a Type-B rectangular bound and combined with display quantisation.
    """
    V_arr = np.asarray(V, dtype=float)
    base = _rectangular_bound_sigma(gain_frac * np.abs(V_arr) + float(floor))
    if lsd is not None and lsd > 0 and np.isfinite(lsd):
        base = np.sqrt(base ** 2 + (lsd / np.sqrt(12)) ** 2)
    return base if V_arr.ndim else float(base)
