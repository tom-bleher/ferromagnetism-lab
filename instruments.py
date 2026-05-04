"""Instrument uncertainty models for the ferromagnetism experiment."""
from __future__ import annotations

import numpy as np

from taulab.stats import resolution_sigma

__all__ = [
    "ruler",
    "caliper",
    "reading_resolution",
    "column_resolution",
    "digital_multimeter_resistance",
    "oscilloscope_dual_cursor",
]


def ruler(resolution: float = 1e-3) -> float:
    """σ for a ruler reading (rectangular window / √12; default 1 mm)."""
    return resolution_sigma(resolution)


def caliper(resolution: float = 0.05e-3) -> float:
    """σ for a Vernier caliper reading (default 0.05 mm)."""
    return resolution_sigma(resolution)


def reading_resolution(v: float) -> float:
    """Smallest display resolution in ``v``'s printed form; 0 for NaN/±∞/0."""
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


def column_resolution(values) -> float:
    """Smallest positive display resolution across ``values``; 0 if none."""
    positives = np.array([reading_resolution(float(v)) for v in values])
    positives = positives[positives > 0]
    return float(positives.min()) if positives.size else 0.0


def _manual_spec_uncertainty(bound):
    return np.asarray(bound, dtype=float)


# Keysight 34401A, 1-year (manual p. 216): full-scale range → (%rdg, %range).
_DMM_R_RANGES: dict[float, tuple[float, float]] = {
    100:       (0.010e-2, 0.004e-2),
    1_000:     (0.010e-2, 0.001e-2),
    10_000:    (0.010e-2, 0.001e-2),
    100_000:   (0.010e-2, 0.001e-2),
    1_000_000: (0.010e-2, 0.001e-2),
}


def digital_multimeter_resistance(R: float, *, include_resolution: bool = True) -> float:
    """Keysight 34401A 1-year σ on the smallest range that fits ``R``.

    The manual's ``±(%rdg + %range)`` accuracy is used directly as the
    instrument uncertainty, following the course-guide convention.
    Disable ``include_resolution`` when the logged value is already rounded
    and the display resolution would be double-counted.
    """
    rng = next(r for r in sorted(_DMM_R_RANGES) if abs(R) <= r)
    pct_rdg, pct_rng = _DMM_R_RANGES[rng]
    sigma = float(_manual_spec_uncertainty(pct_rdg * abs(R) + pct_rng * rng))
    if include_resolution:
        sigma = float(np.hypot(sigma, reading_resolution(R) / np.sqrt(12)))
    return float(sigma)


def oscilloscope_dual_cursor(V, *, gain_frac: float = 0.024,
                             resolution: float | None = None):
    """Keysight DSO7012A dual-cursor ΔV standard uncertainty.

    The datasheet's dual-cursor ``±(gain·full-scale)`` accuracy is used
    directly as the instrument uncertainty and combined with display
    resolution.
    """
    V_arr = np.asarray(V, dtype=float)
    base = _manual_spec_uncertainty(gain_frac * np.abs(V_arr))
    if resolution is not None and resolution > 0 and np.isfinite(resolution):
        base = np.sqrt(base ** 2 + (resolution / np.sqrt(12)) ** 2)
    return base if V_arr.ndim else float(base)
