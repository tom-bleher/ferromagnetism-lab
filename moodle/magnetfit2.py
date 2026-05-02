"""Python translation of magnetfit2.m."""
import numpy as np
from scipy.optimize import minimize_scalar

KB = 1.38e-23
MU = 0.928e-23

def magnetfit(I, Ierr, T, Tc, H, Isat):
    """Return (chi2, Msol) for the mean-field magnetization fit.

    Msol is normalized by the saturation value.
    """
    I = np.asarray(I, dtype=float)
    Ierr = np.asarray(Ierr, dtype=float)
    T = np.asarray(T, dtype=float)

    Msol = np.zeros_like(T)
    for k, Tk in enumerate(T):
        res = minimize_scalar(
            lambda m: abs(m - np.tanh(MU * H / (KB * Tk) + (Tc / Tk) * m)),
            bounds=(0, 1), method='bounded',
        )
        Msol[k] = res.x

    chi2 = np.sum((I / Isat - Msol) ** 2 / (Ierr / Isat) ** 2)
    return chi2, Msol
