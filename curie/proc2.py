import marimo

__generated_with = "0.23.5"
app = marimo.App(
    width="medium",
    app_title="Curie temperature analysis (proc3)",
)


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(SIGMA_T_RES_K, SIGMA_Y_V, mo):
    mo.md(
        r"""
    # Curie temperature analysis (clean notebook)

    This notebook rebuilds the Curie analysis from scratch with a clean, reproducible pipeline.

    **Data Cleaning & Processing**
    1. Setup and data preparation (including units and uncertainties)
    2. **Sanitization & Selection**:
        - **Series C Instability**: We detect the start-up transient in Series C by finding the global maximum of magnetization (smoothed via moving average). Data preceding this peak is discarded to ensure fits start from a stabilized state.
        - **Outlier Filtering (Hampel Filter)**: Singular anomalies are identified via a rolling 7-point median. Points deviating from the median by more than $4.8 \times \text{MAD}$ (Median Absolute Deviation) are removed.
        - **Regime Selection**: Fits are gated by physical regimes (Mean-field near transition, Curie-Weiss in the tail) to maintain model validity.
    3. **Magnetization Proxy Extraction**: Three methods are compared, including an **Adaptive Tail Fit** ($M_3$) which identifies the linear saturation region by growing the fit window as long as residuals stay within a $2.2\sigma$ tolerance.
    4. Part A: extraction of magnetization proxies (M1, M2, M3) from loop geometries.
    5. **Rough Tc Estimation**: Numerical second-derivatives are used to find the transition inflection point.
    6. Part B: Refined Curie temperature extraction using restricted temperature regimes.

    **Notation**
    - \(T\): temperature in Kelvin
    - \(H\): external field proxy (oscilloscope X channel, volts)
    - \(B\): total field proxy (oscilloscope Y channel, volts)
    - \(M\): magnetization proxy extracted from hysteresis geometry

    **Calculated constants used in the notebook**
    - Channel digitization uncertainty: $\sigma_X = \sigma_Y = {SIGMA_Y_V:.6e}$ V
    - Temperature readout resolution term: $\sigma_{T,\mathrm{res}} = {SIGMA_T_RES_K:.6e}$ K
    - The drift term is computed pointwise from the local ramp rate, so it has no single global value.
    """.replace("{SIGMA_Y_V:.6e}", f"{SIGMA_Y_V:.6e}").replace(
            "{SIGMA_T_RES_K:.6e}", f"{SIGMA_T_RES_K:.6e}"
        )
    )
    return


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.optimize import curve_fit
    from scipy.stats import chi2 as chi2_dist

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.axisbelow": True,
        }
    )

    ROOT = Path(__file__).resolve().parent
    FIG_DIR = ROOT.parent / "report" / "media" / "curie"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    COLORS = {
        "m1": "#1b9e77",
        "m2": "#d95f02",
        "m3": "#7570b3",
        "fit": "#e7298a",
        "data": "#1f77b4",
    }

    def save_figure(fig, stem):
        fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight", dpi=600)

    return COLORS, ROOT, chi2_dist, curve_fit, np, pd, plt, save_figure


@app.cell
def _(ROOT, np):
    DATA_DIR = ROOT / "data"
    SERIES_ORDER = ("series A", "series B", "series C")
    SERIES_FILES = {
        name: sorted((DATA_DIR / name).glob("CurieData_*"))[0] for name in SERIES_ORDER
    }

    SIGMA_Y_V = 0.4e-3 / np.sqrt(12.0)
    SIGMA_T_RES_K = 1e-3 / np.sqrt(12.0)

    def sigma_thermometer_k(T_K):
        T_C = np.asarray(T_K, dtype=float) - 273.15
        abs_T_C = np.abs(T_C)
        return np.where(T_C >= -60.0, 0.001 * abs_T_C + 1.0, 0.001 * abs_T_C + 2.0)

    def temperature_uncertainty_k(T_K, time_s):
        sigma_abs = sigma_thermometer_k(T_K)
        dT_dt = np.gradient(T_K, time_s)
        dt = float(np.median(np.diff(time_s))) if len(time_s) > 1 else 0.0
        sigma_drift = np.abs(dT_dt) * dt / np.sqrt(12.0)
        return np.sqrt(sigma_abs**2 + SIGMA_T_RES_K**2 + sigma_drift**2)

    def normalize_with_uncertainty(values, sigma_values):
        values = np.asarray(values, dtype=float)
        sigma_values = np.asarray(sigma_values, dtype=float)
        valid = np.isfinite(values)
        if not np.any(valid):
            return values * np.nan, sigma_values * np.nan
        vmin = np.nanmin(values[valid])
        vmax = np.nanmax(values[valid])
        scale = vmax - vmin
        if np.isclose(scale, 0.0):
            return np.zeros_like(values), np.full_like(values, np.nan)
        return (values - vmin) / scale, sigma_values / scale

    return (
        SERIES_FILES,
        SERIES_ORDER,
        SIGMA_T_RES_K,
        SIGMA_Y_V,
        normalize_with_uncertainty,
        temperature_uncertainty_k,
    )


@app.cell(hide_code=True)
def _(SIGMA_Y_V, mo):
    _content = r"""
    ## Setup details and preprocessing

    **Data preparation done here**
    - Loaded all three measurement series (`series A/B/C`) from `curie/data`.
    - Converted temperature from Celsius to Kelvin using \(T_K = T_C + 273.15\).
    - Kept the native oscilloscope channels as field proxies:
      - X channel \(\propto H\)
      - Y channel \(\propto B\)
    - **Automated Cleaning Applied**:
      - **Series C instability**: We detect the start-up transient by locating the global maximum of magnetization (smoothed via 5-point average). Data preceding this peak is discarded to exclude non-equilibrium heating.
      - **Outlier Removal (Hampel Filter)**: Point anomalies are identified via a rolling 7-point median. If $|M_i - \text{median}| > 4.8 \times \text{MAD}$, the point is discarded as a non-physical artifact.
    - Built per-loop uncertainty arrays:
      - \(\sigma_T\): thermometer spec + temperature-resolution + finite-time drift, combined as
            $$
                \sigma_T^2 = \sigma_{\text{abs}}^2 + \sigma_{\text{res}}^2 + \sigma_{\text{drift}}^2
            $$
            where the drift term is defined by the local temperature ramp rate,
            $$
                \sigma_{\text{drift}} = \left|\frac{dT}{dt}\right|\frac{\Delta t}{\sqrt{12}}
            $$
            with $\Delta t$ taken as the median sample spacing.
      - \(\sigma_X, \sigma_Y\): digitization uncertainty from the ADC step size, modeled as a uniform distribution,
        $$
          \sigma_{X,Y} = \frac{\Delta_{X,Y}}{\sqrt{12}}
        $$
        where \(\Delta_{X,Y}\) is the channel quantization step.

    **Uncertainty scope**
    - Random/statistical uncertainties are propagated through extraction and fits.
    - Hardware calibration systematics (component tolerances) are not provided in the guide;
      therefore they are not included in the numeric propagation below.

    **Calculated constants used in the notebook**
    - $\sigma_X = \sigma_Y = {SIGMA_Y_V:.6e}$ V
    - The temperature-resolution term used in the square-root sum is $\sigma_{T,\mathrm{res}} = 2.886751\times10^{-4}$ K
    - The series-wide cleanup thresholds are still method-specific, so one method's outlier does not force the others to drop the same point.
    """.replace("{SIGMA_Y_V:.6e}", f"{SIGMA_Y_V:.6e}")
    mo.md(_content)
    return


@app.cell
def _(SIGMA_Y_V, curve_fit, np):
    def sorted_channel_columns(df, prefix):
        cols = [c for c in df.columns if c.startswith(prefix)]
        return sorted(cols, key=lambda c: int(c[len(prefix):]))

    def _y_at_zero(x, y, sigma_y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        cross = np.where(np.signbit(x[:-1]) != np.signbit(x[1:]))[0]
        if len(cross) > 0:
            i = int(cross[np.argmin(np.abs(x[cross]))])
            x0, x1 = x[i], x[i + 1]
            y0, y1 = y[i], y[i + 1]
            t = -x0 / (x1 - x0)
            y_interp = (1.0 - t) * y0 + t * y1
            sigma_interp = np.sqrt((1.0 - t) ** 2 + t**2) * sigma_y
            return y_interp, sigma_interp

        i0 = int(np.argmin(np.abs(x)))
        return float(y[i0]), float(sigma_y)

    def _weighted_line_fit(x, y, sigma_y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.full_like(y, 1.0 / sigma_y, dtype=float)
        p, cov = np.polyfit(x, y, deg=1, w=w, cov=True)
        slope = float(p[0])
        intercept = float(p[1])
        sigma_slope = float(np.sqrt(cov[0, 0]))
        sigma_intercept = float(np.sqrt(cov[1, 1]))
        return slope, intercept, sigma_slope, sigma_intercept

    def _tail_linearity_rms(x, y):
        if len(x) < 3:
            return np.inf
        slope, intercept, _, _ = _weighted_line_fit(x, y, SIGMA_Y_V)
        residuals = (y - (slope * x + intercept)) / SIGMA_Y_V
        return float(np.sqrt(np.mean(residuals**2)))

    def _adaptive_tail_indices(
        x,
        y,
        side,
        min_points=5,
        max_fraction=0.35,
        residual_sigma_max=2.2,
        residual_growth=0.25,
    ):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(x)
        if n <= min_points:
            return np.arange(n, dtype=int)

        max_points = max(min_points, int(np.ceil(max_fraction * n)))
        max_points = min(max_points, n)

        order = np.argsort(x)
        if side == "pos":
            ordered = order[::-1]
        else:
            ordered = order

        selected = list(ordered[:min_points])
        x_sel = np.asarray(x[selected], dtype=float)
        y_sel = np.asarray(y[selected], dtype=float)
        curr_rms = _tail_linearity_rms(x_sel, y_sel)

        for cand in ordered[min_points:max_points]:
            trial = selected + [int(cand)]
            x_try = np.asarray(x[trial], dtype=float)
            y_try = np.asarray(y[trial], dtype=float)
            trial_rms = _tail_linearity_rms(x_try, y_try)
            allowed = max(residual_sigma_max, curr_rms * (1.0 + residual_growth))
            if trial_rms <= allowed:
                selected = trial
                curr_rms = trial_rms
            else:
                break

        return np.asarray(selected, dtype=int)

    def extract_loop_methods(x_pos, y_pos, x_neg, y_neg, edge_fraction=0.05):
        n = len(x_pos)
        n_edge = max(5, int(np.ceil(edge_fraction * n)))

        # Method 1: value at H=0 (loop opening)
        y0_pos, sy0_pos = _y_at_zero(x_pos, y_pos, SIGMA_Y_V)
        y0_neg, sy0_neg = _y_at_zero(x_neg, y_neg, SIGMA_Y_V)
        m1 = 0.5 * (y0_pos - y0_neg)
        s1 = 0.5 * np.sqrt(sy0_pos**2 + sy0_neg**2)

        # Method 2: saturation points from extreme-field windows
        idx_pos_edge = np.argsort(x_pos)[-n_edge:]
        idx_neg_edge = np.argsort(x_neg)[:n_edge]
        y_pos_sat = np.asarray(y_pos[idx_pos_edge], dtype=float)
        y_neg_sat = np.asarray(y_neg[idx_neg_edge], dtype=float)
        mean_pos_sat = float(np.mean(y_pos_sat))
        mean_neg_sat = float(np.mean(y_neg_sat))
        s_pos_sat = np.sqrt(
            np.var(y_pos_sat, ddof=1) / len(y_pos_sat) + SIGMA_Y_V**2 / len(y_pos_sat)
        )
        s_neg_sat = np.sqrt(
            np.var(y_neg_sat, ddof=1) / len(y_neg_sat) + SIGMA_Y_V**2 / len(y_neg_sat)
        )
        m2 = 0.5 * (mean_pos_sat - mean_neg_sat)
        s2 = 0.5 * np.sqrt(s_pos_sat**2 + s_neg_sat**2)

        # Method 3: adaptive tail fit from extreme fields toward the center.
        idx_pos_tail = _adaptive_tail_indices(x_pos, y_pos, side="pos")
        idx_neg_tail = _adaptive_tail_indices(x_neg, y_neg, side="neg")
        x_pos_tail = np.asarray(x_pos[idx_pos_tail], dtype=float)
        y_pos_tail = np.asarray(y_pos[idx_pos_tail], dtype=float)
        x_neg_tail = np.asarray(x_neg[idx_neg_tail], dtype=float)
        y_neg_tail = np.asarray(y_neg[idx_neg_tail], dtype=float)

        _, b_pos, _, sb_pos = _weighted_line_fit(x_pos_tail, y_pos_tail, SIGMA_Y_V)
        _, b_neg, _, sb_neg = _weighted_line_fit(x_neg_tail, y_neg_tail, SIGMA_Y_V)
        m3 = 0.5 * (b_pos - b_neg)
        s3 = 0.5 * np.sqrt(sb_pos**2 + sb_neg**2)

        return {
            "m1": float(m1),
            "s1": float(s1),
            "m2": float(m2),
            "s2": float(s2),
            "m3": float(m3),
            "s3": float(s3),
            "n_tail_pos": int(len(idx_pos_tail)),
            "n_tail_neg": int(len(idx_neg_tail)),
            "n_edge": int(n_edge),
        }

    def extract_chi_near_zero(x_pos, y_pos, x_neg, y_neg, window_fraction=0.20):
        h_max = max(np.max(np.abs(x_pos)), np.max(np.abs(x_neg)))
        h_cut = window_fraction * h_max
        x_all = np.concatenate(
            [x_pos[np.abs(x_pos) <= h_cut], x_neg[np.abs(x_neg) <= h_cut]]
        )
        y_all = np.concatenate(
            [y_pos[np.abs(x_pos) <= h_cut], y_neg[np.abs(x_neg) <= h_cut]]
        )
        if len(x_all) < 8:
            return np.nan, np.nan

        # y = a*x + b, where a is susceptibility proxy
        def line(x, a, b):
            return a * x + b

        sigma = np.full_like(y_all, SIGMA_Y_V, dtype=float)
        popt, pcov = curve_fit(line, x_all, y_all, sigma=sigma, absolute_sigma=True)
        a = float(popt[0])
        sa = float(np.sqrt(pcov[0, 0]))
        return a, sa

    def transition_temp(Tvals, Mvals):
        Tvals = np.asarray(Tvals, dtype=float)
        Mvals = np.asarray(Mvals, dtype=float)
        mask = np.isfinite(Tvals) & np.isfinite(Mvals)
        if not np.any(mask):
            return np.nan
        idx = int(np.nanargmin(np.abs(Mvals[mask] - 0.5)))
        return float(Tvals[mask][idx])

    def estimate_rough_tc(T, M):
        """Estimate Tc using the peak of the second derivative of M(T)."""
        T = np.asarray(T, dtype=float)
        M = np.asarray(M, dtype=float)
        mask = np.isfinite(T) & np.isfinite(M)
        T = T[mask]
        M = M[mask]
        if len(T) < 5:
            return np.nan
        # Smooth to avoid noise in derivatives
        from scipy.signal import savgol_filter

        window_length = min(9, len(M) if len(M) % 2 == 1 else len(M) - 1)
        if window_length < 5:
            return np.nan
        polyorder = min(3, window_length - 1)
        M_smooth = savgol_filter(M, window_length=window_length, polyorder=polyorder)
        # Find where the curve drops fastest (inflection point)
        d2M = np.gradient(np.gradient(M_smooth, T), T)
        # The peak of the second derivative happens as it curves into the tail
        idx = np.argmax(np.abs(d2M))
        return float(T[idx])

    def fit_mean_field(
        T_K, sigma_T, M_norm, sigma_M, Tc_seed=215.0, T_rough=None, cut_factor=0.75
    ):
        _T = np.asarray(T_K, dtype=float)
        _sT = np.asarray(sigma_T, dtype=float)
        M = np.asarray(M_norm, dtype=float)
        sM = np.asarray(sigma_M, dtype=float)

        # Physical Bounds:
        # 1. M < 0.9: Exclude low-T saturation where MF law fails.
        # 2. T >= cut_factor * Tc_rough: Exclude the deep low-T region below the transition window.
        mask = np.isfinite(_T) & np.isfinite(M) & np.isfinite(sM) & (M < 0.975)

        Tc_ref = T_rough if T_rough is not None else Tc_seed
        T_cut = cut_factor * Tc_ref
        mask &= _T >= T_cut

        if T_rough is not None:
            # Keep the upper guard near the rough transition to avoid finite-field tails.
            mask &= _T < T_rough - 1.5
        else:
            mask &= M > 0.05

        n_strict = int(np.sum(mask))
        if n_strict < 8:
            return {
                "ok": False,
                "reason": f"insufficient points after cutoff ({n_strict})",
                "n_strict": n_strict,
            }

        Tf = _T[mask]
        sTf = _sT[mask]
        Mf = M[mask]
        sMf = np.maximum(sM[mask], 1e-4)

        def model(Tloc, Tc, A):
            return A * np.sqrt(np.clip(Tc - Tloc, 0.0, None))

        p0 = [
            T_rough if T_rough is not None else Tc_seed,
            max(0.05, np.max(Mf) / np.sqrt(max(Tc_seed - np.min(Tf), 1.0))),
        ]
        popt, pcov = curve_fit(
            model,
            Tf,
            Mf,
            p0=p0,
            sigma=sMf,
            absolute_sigma=True,
            bounds=([150.0, 0.0], [350.0, 10.0]),
            maxfev=20000,
        )
        Tc, A = map(float, popt)
        sTc = float(np.sqrt(pcov[0, 0]))
        pred = model(Tf, Tc, A)
        res = (Mf - pred) / sMf
        dof = max(len(Tf) - 2, 1)
        chi2_red = float(np.sum(res**2) / dof)
        return {
            "ok": True,
            "Tc": Tc,
            "sigma_Tc": sTc,
            "A": A,
            "T_fit": Tf,
            "sigma_T_fit": sTf,
            "y_fit": Mf,
            "sigma_fit": sMf,
            "y_model": pred,
            "residuals": res,
            "chi2_red": chi2_red,
            "n": int(len(Tf)),
            "n_strict": n_strict,
        }

    def fit_curie_weiss(
        T_K, sigma_T, chi, sigma_chi, Tc_seed=215.0, T_rough=None, cut_factor=1.15
    ):
        _T = np.asarray(T_K, dtype=float)
        _sT = np.asarray(sigma_T, dtype=float)
        chi = np.asarray(chi, dtype=float)
        schi = np.asarray(sigma_chi, dtype=float)

        # Mask points where magnetization is too small (approaching the noise floor)
        # to avoid the 1/M singularity.
        mask = np.isfinite(_T) & np.isfinite(chi) & np.isfinite(schi) & (chi > 0.02)
        if np.sum(mask) < 10:
            return {"ok": False, "reason": "insufficient chi points"}

        inv_chi = 1.0 / chi
        sigma_inv = schi / np.maximum(chi, 1e-12) ** 2

        # CW is only valid in paramagnetic regime (far away from transition)
        cut = cut_factor * (T_rough if T_rough is not None else Tc_seed)

        fit_mask = mask & (_T >= cut)
        if np.sum(fit_mask) < 6:
            return {"ok": False, "reason": "insufficient high-T points"}

        Tf = _T[fit_mask]
        sTf = _sT[fit_mask]
        Yf = inv_chi[fit_mask]
        sYf = np.maximum(sigma_inv[fit_mask], 1e-6)

        def model(Tloc, Tc, C):
            return (Tloc - Tc) / C

        popt, pcov = curve_fit(
            model,
            Tf,
            Yf,
            p0=[T_rough if T_rough is not None else Tc_seed, 1.0],
            sigma=sYf,
            absolute_sigma=True,
            bounds=([150.0, 1e-6], [350.0, 1e6]),
            maxfev=20000,
        )
        Tc, C = map(float, popt)
        sTc = float(np.sqrt(pcov[0, 0]))
        pred = model(Tf, Tc, C)
        res = (Yf - pred) / sYf
        dof = max(len(Tf) - 2, 1)
        chi2_red = float(np.sum(res**2) / dof)
        return {
            "ok": True,
            "Tc": Tc,
            "sigma_Tc": sTc,
            "C": C,
            "T_fit": Tf,
            "sigma_T_fit": sTf,
            "y_fit": Yf,
            "sigma_fit": sYf,
            "y_model": pred,
            "residuals": res,
            "chi2_red": chi2_red,
            "n": int(len(Tf)),
        }

    return (
        estimate_rough_tc,
        extract_chi_near_zero,
        extract_loop_methods,
        fit_curie_weiss,
        fit_mean_field,
        sorted_channel_columns,
        transition_temp,
    )


@app.cell
def _(
    SERIES_FILES,
    SERIES_ORDER,
    extract_chi_near_zero,
    extract_loop_methods,
    normalize_with_uncertainty,
    np,
    pd,
    sorted_channel_columns,
    temperature_uncertainty_k,
):
    series_data = {}

    for _series_name in SERIES_ORDER:
        data_file = SERIES_FILES[_series_name]
        df = pd.read_csv(data_file, sep="\t")

        T_C = df["Temperature (C)"].to_numpy(dtype=float)
        T_K = T_C + 273.15
        time_s = df["Time (sec)"].to_numpy(dtype=float)
        sigma_T = temperature_uncertainty_k(T_K, time_s)

        x_pos_cols = sorted_channel_columns(df, "X_pos")
        y_pos_cols = sorted_channel_columns(df, "Y_pos")
        x_neg_cols = sorted_channel_columns(df, "X_neg")
        y_neg_cols = sorted_channel_columns(df, "Y_neg")

        X_pos = df[x_pos_cols].to_numpy(dtype=float)
        Y_pos = df[y_pos_cols].to_numpy(dtype=float)
        X_neg = df[x_neg_cols].to_numpy(dtype=float)
        Y_neg = df[y_neg_cols].to_numpy(dtype=float)

        n_loops = len(df)
        m1 = np.full(n_loops, np.nan)
        s1 = np.full(n_loops, np.nan)
        m2 = np.full(n_loops, np.nan)
        s2 = np.full(n_loops, np.nan)
        m3 = np.full(n_loops, np.nan)
        s3 = np.full(n_loops, np.nan)
        chi0 = np.full(n_loops, np.nan)
        schi0 = np.full(n_loops, np.nan)
        tail_points = np.full(n_loops, np.nan)

        for i in range(n_loops):
            out = extract_loop_methods(X_pos[i], Y_pos[i], X_neg[i], Y_neg[i])
            m1[i], s1[i] = out["m1"], out["s1"]
            m2[i], s2[i] = out["m2"], out["s2"]
            m3[i], s3[i] = out["m3"], out["s3"]
            tail_points[i] = 0.5 * (out["n_tail_pos"] + out["n_tail_neg"])

            chi0[i], schi0[i] = extract_chi_near_zero(
                X_pos[i], Y_pos[i], X_neg[i], Y_neg[i]
            )

        # --- DATA CLEANING (Automated Cutoffs & Outliers) ---
        # 1. Automated Cutoff: Series C is unstable before reaching its magnetic peak.
        # A and B receive a minor initial cutoff (first 4 points) to skip start-up transient.
        # # if _series_name == "series C":
        # #     # Detect the peak on a smoothed version of M3 (the cleanest proxy)
        # #     m3_filled = np.nan_to_num(m3, nan=np.nanmedian(m3))
        # #     m3_smooth = np.convolve(m3_filled, np.ones(5) / 5, mode="same")
        # #     start_idx = int(np.argmax(m3_smooth))
        # # else:
        # #     start_idx = min(4, n_loops)
        start_idx = 0

        # 2. Outlier Detection: Identify singular spikes using a rolling median + MAD (Hampel-style)
        # Apply the same logic to all three method series (M1, M2, M3) and combine flags.
        def _rolling_mad_outlier(values, window=7, thresh=4.8):
            v = np.asarray(values, dtype=float)
            ser = pd.Series(v)
            med = (
                ser.rolling(window=window, center=True)
                .median()
                .ffill()
                .bfill()
                .to_numpy()
            )
            resid = np.abs(v - med)
            mad = np.nanmedian(resid)
            return resid > (thresh * max(mad, 1e-6))

        is_outlier_m1 = _rolling_mad_outlier(m1)
        is_outlier_m2 = _rolling_mad_outlier(m2)
        is_outlier_m3 = _rolling_mad_outlier(m3)

        # Keep the shared loop axis intact; only blank out the method values that fail
        # their own Hampel-style test so one proxy cannot suppress the others.
        valid_points = np.arange(n_loops) >= start_idx

        T_C, T_K = T_C[valid_points], T_K[valid_points]
        time_s, sigma_T = time_s[valid_points], sigma_T[valid_points]
        X_pos, Y_pos = X_pos[valid_points], Y_pos[valid_points]
        X_neg, Y_neg = X_neg[valid_points], Y_neg[valid_points]
        m1, s1 = m1[valid_points], s1[valid_points]
        m2, s2 = m2[valid_points], s2[valid_points]
        m3, s3 = m3[valid_points], s3[valid_points]
        chi0, schi0 = chi0[valid_points], schi0[valid_points]
        tail_points = tail_points[valid_points]

        is_outlier_m1 = is_outlier_m1[valid_points]
        is_outlier_m2 = is_outlier_m2[valid_points]
        is_outlier_m3 = is_outlier_m3[valid_points]

        m1[is_outlier_m1] = np.nan
        s1[is_outlier_m1] = np.nan
        m2[is_outlier_m2] = np.nan
        s2[is_outlier_m2] = np.nan
        m3[is_outlier_m3] = np.nan
        s3[is_outlier_m3] = np.nan

        # Normalize the cleaned data
        m1_norm, s1_norm = normalize_with_uncertainty(m1, s1)
        m2_norm, s2_norm = normalize_with_uncertainty(m2, s2)
        m3_norm, s3_norm = normalize_with_uncertainty(m3, s3)

        series_data[_series_name] = {
            "file_name": data_file.name,
            "T_C": T_C,
            "T_K": T_K,
            "time_s": time_s,
            "sigma_T_K": sigma_T,
            "X_pos": X_pos,
            "Y_pos": Y_pos,
            "X_neg": X_neg,
            "Y_neg": Y_neg,
            "method1": m1,
            "method1_sigma": s1,
            "method2": m2,
            "method2_sigma": s2,
            "method3": m3,
            "method3_sigma": s3,
            "method1_norm": m1_norm,
            "method1_sigma_norm": s1_norm,
            "method2_norm": m2_norm,
            "method2_sigma_norm": s2_norm,
            "method3_norm": m3_norm,
            "method3_sigma_norm": s3_norm,
            "chi0": chi0,
            "chi0_sigma": schi0,
            "tail_points": tail_points,
        }
    return (series_data,)


@app.cell
def _(SERIES_ORDER, mo, np, series_data):
    """Create temperature range sliders for filtering each series"""
    # Use mo.ui.dictionary to create a reactive group of sliders.
    # This allows marimo to notify downstream cells whenever any slider changes.
    sliders = mo.ui.dictionary(
        {
            name: mo.ui.range_slider(
                value=(
                    float(np.min(series_data[name]["T_K"])),
                    float(np.max(series_data[name]["T_K"])),
                ),
                start=float(np.min(series_data[name]["T_K"])),
                stop=float(np.max(series_data[name]["T_K"])),
                step=0.5,
                label=f"{name}: [K]",
            )
            for name in SERIES_ORDER
        }
    )

    # Construct a display layout. Returning this object as the cell's result
    # ensures it is rendered as interactive HTML in the notebook.
    slider_display = mo.vstack(
        [mo.vstack([mo.md(f"**{name}**"), sliders[name]]) for name in SERIES_ORDER]
    )
    slider_display  # type: ignore
    return (sliders,)


@app.cell
def _(SERIES_ORDER, np, series_data, sliders):
    """Apply temperature range filters to the data"""
    filtered_series_data = {}

    for series_name in SERIES_ORDER:
        original_data = series_data[series_name]
        # Access the current values through the sliders.value property.
        # Accessing .value is what establishes the reactive dependency.
        T_min, T_max = sliders.value[series_name]

        # Create mask for temperature range
        mask = (original_data["T_K"] >= T_min) & (original_data["T_K"] <= T_max)

        # Apply mask to all arrays in the data dict
        filtered_series_data[series_name] = {}
        for key, val in original_data.items():
            if isinstance(val, np.ndarray):
                filtered_series_data[series_name][key] = val[mask]
            else:
                # Keep non-array values as-is
                filtered_series_data[series_name][key] = val
    return (filtered_series_data,)


@app.cell
def _(
    SERIES_ORDER,
    estimate_rough_tc,
    filtered_series_data,
    mo,
    np,
    pd,
    transition_temp,
):
    prep_rows = []
    part_a_rows = []

    for name in SERIES_ORDER:
        _d = filtered_series_data[name]
        _T = _d["T_K"]

        # Calculate rough Tc for this series based on derivative of M3
        t_rough = estimate_rough_tc(_T, _d["method3_norm"])
        filtered_series_data[name]["Tc_rough"] = t_rough

        prep_rows.append(
            {
                "Series": name,
                "Loops": len(_T),
                "T range [K]": (
                    f"{np.min(_T):.1f} to {np.max(_T):.1f}" if len(_T) > 0 else "N/A"
                ),
                "median sigma T [K]": f"{np.median(_d['sigma_T_K']):.2f}",
                "tail points (M3)": int(np.nanmedian(_d["tail_points"])),
                "Rough Tc [K]": f"{t_rough:.1f}" if np.isfinite(t_rough) else "N/A",
            }
        )

        # Calculate summary row reactively based on slider range
        part_a_rows.extend(
            [
                {
                    "Series": name,
                    "Method": "M1 @ H=0",
                    "T(M=0.5) [K]": transition_temp(_T, _d["method1_norm"]),
                    "Range(raw)": (
                        float(np.nanmax(_d["method1"]) - np.nanmin(_d["method1"]))
                        if len(_T) > 0
                        else 0
                    ),
                },
                {
                    "Series": name,
                    "Method": "M2 saturation edges",
                    "T(M=0.5) [K]": transition_temp(_T, _d["method2_norm"]),
                    "Range(raw)": (
                        float(np.nanmax(_d["method2"]) - np.nanmin(_d["method2"]))
                        if len(_T) > 0
                        else 0
                    ),
                },
                {
                    "Series": name,
                    "Method": "M3 tail extrapolation",
                    "T(M=0.5) [K]": transition_temp(_T, _d["method3_norm"]),
                    "Range(raw)": (
                        float(np.nanmax(_d["method3"]) - np.nanmin(_d["method3"]))
                        if len(_T) > 0
                        else 0
                    ),
                },
            ]
        )

    prep_df = pd.DataFrame(prep_rows)
    part_a_summary_df = pd.DataFrame(part_a_rows)

    mo.md(rf"""
    **Setup summary**

    {prep_df.to_markdown(index=False)}

    **Part A short table (normalized-curve midpoint and raw dynamic range)**

    {part_a_summary_df.to_markdown(index=False, floatfmt=".3f")}
    """)
    return


@app.cell(hide_code=True)
def _(SIGMA_Y_V, mo):
    _content = r"""
    ## Part A — Magnetization curves from three extraction methods

    For each measurement series (A, B, C), one graph is shown with all three normalized \(M(T)\) curves:
    1. **Method 1**: loop intersection at \(H=0\)
    2. **Method 2**: saturation-edge values (tips of the loop)
    3. **Method 3**: extrapolation of saturation tails to \(H=0\) using an adaptive linear fit.

     **Error propagation used in Part A**
     The three proxies are all functions of the measured loop points, so we propagate the pointwise voltage uncertainty through the extraction rule:
    1. **Method 1** uses an interpolated zero-field value. For a linear interpolation fraction \(t\),
        $$
            M_1 = (1-t) y_0 + t y_1, \qquad
            \sigma_{M_1} = \sqrt{(1-t)^2 + t^2}\,\sigma_Y
        $$
    2. **Method 2** averages the extreme-field window. For an average over \(n\) points,
        $$
            M_2 = \frac{1}{2}(\bar y_+ - \bar y_-), \qquad
            \sigma_{M_2} = \frac{1}{2}\sqrt{\sigma_+^2 + \sigma_-^2}
        $$
         with each window uncertainty estimated from the sample spread and digitization term.
    3. **Method 3** fits a line to the tail region and extrapolates to zero field. If the tail fit is \(y = a x + b\), the magnetization proxy is set by the intercepts,
        $$
            M_3 = \frac{1}{2}(b_+ - b_-), \qquad
            \sigma_{M_3} = \frac{1}{2}\sqrt{\sigma_{b_+}^2 + \sigma_{b_-}^2}
        $$
    where each \(\sigma_b\) comes from the weighted line fit covariance.

    The same pointwise channel uncertainty is used throughout Part A, so the actual digitization scale in the code is
    $$
        \sigma_X = \sigma_Y = {SIGMA_Y_V:.6e}\ \mathrm{{V}}
    $$
    and any additional numerical spread in the extracted proxies comes from the measured loop points themselves, not from a hidden extra noise model.

    **Extraction Logic for Method 3**:
    The linear "tail" of the saturation region is identified by an algorithm that starts at the extreme field tips and grows the fitting window inward. It stops when the RMS of the residuals increases by more than $25\%$ or exceeds $2.2\sigma$. This ensures the extrapolated zero-field magnetization proxy is derived from a strictly linear saturation regime.

    **Calculated values used in the code**
    - Tail fit starts from `min_points = 5` per side.
    - Tail window growth is capped at `max_fraction = 0.35` of the loop.
    - The adaptive stop rule uses `residual_sigma_max = 2.2` and `residual_growth = 0.25`.
    - The shared loop-point uncertainty is still $\sigma_Y = {SIGMA_Y_V:.6e}$ V.
    """.replace("{SIGMA_Y_V:.6e}", f"{SIGMA_Y_V:.6e}")
    mo.md(_content)
    return


@app.cell
def _(COLORS, SERIES_ORDER, filtered_series_data, np, plt, save_figure):
    def _():
        fig = plt.figure(figsize=(12, 6.8))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], wspace=0.28, hspace=0.28)
        axes = [
            fig.add_subplot(gs[:, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 1]),
        ]
        for _i, _series_name in enumerate(SERIES_ORDER):
            _d = filtered_series_data[_series_name]
            _T = _d["T_K"]
            ax = axes[_i]
            mask1 = (
                np.isfinite(_T)
                & np.isfinite(_d["method1_norm"])
                & np.isfinite(_d["method1_sigma_norm"])
                & np.isfinite(_d["sigma_T_K"])
            )
            ax.errorbar(
                _T[mask1],
                _d["method1_norm"][mask1],
                xerr=_d["sigma_T_K"][mask1],
                yerr=_d["method1_sigma_norm"][mask1],
                fmt="o-",
                ms=3.5,
                lw=1.3,
                color=COLORS["m1"],
                alpha=0.9,
                label="Method 1: H=0 intersection",
            )
            mask2 = (
                np.isfinite(_T)
                & np.isfinite(_d["method2_norm"])
                & np.isfinite(_d["method2_sigma_norm"])
                & np.isfinite(_d["sigma_T_K"])
            )
            ax.errorbar(
                _T[mask2],
                _d["method2_norm"][mask2],
                xerr=_d["sigma_T_K"][mask2],
                yerr=_d["method2_sigma_norm"][mask2],
                fmt="s-",
                ms=3.2,
                lw=1.3,
                color=COLORS["m2"],
                alpha=0.9,
                label="Method 2: saturation edges",
            )
            mask3 = (
                np.isfinite(_T)
                & np.isfinite(_d["method3_norm"])
                & np.isfinite(_d["method3_sigma_norm"])
                & np.isfinite(_d["sigma_T_K"])
            )
            ax.errorbar(
                _T[mask3],
                _d["method3_norm"][mask3],
                xerr=_d["sigma_T_K"][mask3],
                yerr=_d["method3_sigma_norm"][mask3],
                fmt="^-",
                ms=3.5,
                lw=1.3,
                color=COLORS["m3"],
                alpha=0.9,
                label="Method 3: tail extrapolation",
            )
            ax.set_title(f"{_series_name}: normalized magnetization vs temperature")
            ax.set_xlabel("Temperature [K]")
            ax.set_ylabel("Normalized magnetization")
            ax.set_ylim(-0.05, 1.05)
            ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        return fig

    fig = _()
    save_figure(fig, "normalized_magnetization_curves")
    fig  # type: ignore
    return


@app.cell(hide_code=True)
def _(SIGMA_T_RES_K, mo):
    mo.md(r"""
    ## Part B — Curie temperature extraction

    Two fitting methods are applied to each series:
    1. **Mean-field (mean-field near transition):** \(M(T)=A\sqrt{T_C-T}\)
    2. **Curie-Weiss:** \(1/\chi(T)=(T-T_C)/C\)

    **Error propagation used in Part B**
    The fit inputs are the normalized proxies from Part A, so their uncertainties carry forward directly.
    For the mean-field fit we linearize the model by squaring it for display:
    $$
    M^2(T) = A^2 (T_C - T)
    $$
    and propagate the uncertainty with the usual derivative rule,
    $$
    \sigma_{M^2} = \left|\frac{d(M^2)}{dM}\right|\sigma_M = 2|M|\,\sigma_M
    $$
    so the plotted residuals are measured in the same squared space as the fit.

    For Curie-Weiss, the plotted quantity is the reciprocal proxy, so
    $$
    \frac{1}{\chi} = M^{-1}, \qquad
    \sigma_{1/\chi} = \left|\frac{d(M^{-1})}{dM}\right|\sigma_M = \frac{\sigma_M}{M^2}
    $$
    with a small numerical floor used in the code to avoid division by zero when \(M\) is very small.

    The temperature uncertainty used by the fit preprocessing is still the quadrature sum
    $$
        \sigma_T^2 = \sigma_{\text{abs}}^2 + \sigma_{\text{res}}^2 + \sigma_{\text{drift}}^2
    $$
    with the fixed resolution contribution
    $$
        \sigma_{T,\mathrm{res}} = {SIGMA_T_RES_K:.6e}\ \mathrm{{K}}.
    $$
    For the fit regions themselves, the actual code thresholds are `mf_cut_factor = 0.75` and `cw_cut_factor = 1.15`.

    **Rough $T_c$ Estimate**
    Before performing non-linear fits, we calculate a model-independent "Rough $T_c$" to automate regime selection. The process involves:
    1. **Smoothing**: Magnetization data ($M_3$) is smoothed using a Savitzky-Golay filter (window length 9, polynomial order 3) to suppress numerical noise.
    2. **Differentiation**: We compute the numerical second derivative $\frac{d^2M}{dT^2}$ using central differences.
    3. **Localization**: We identify $T_{c,\text{rough}}$ as the temperature where $|\frac{d^2M}{dT^2}|$ is maximized.

    Physically, this locates the "knee" of the transition—the point of maximum curvature where the ferromagnetic drop-off enters the paramagnetic tail. This estimate is used to set the upper bound for Mean-field fits and the lower bound for Curie-Weiss fits.

    **Regime Selection (Data Cutting)**
    The physical models are only valid in specific temperature regimes. To ensure robust fits, points outside these regimes are automatically masked:
    - **Mean-field**: Only uses points above a lower cutoff (default \(0.75\,T_{c,\text{rough}}\)) and below the finite-field upper guard to avoid deep low-T saturation and smearing near \(T_c\).
    - **Curie-Weiss**: Only uses points far away from the transition (\(T > \text{factor} \times T_{c,\text{rough}}\)) to ensure the system is purely in the paramagnetic state. This factor is controlled via an interactive slider and defaults to 1.15.

    The plots below show the full slider-filtered range in light gray for context.
    Fits are now performed on all three extraction methods (M1, M2, M3) to allow for consistency cross-checks.

    **Calculated values used in the notebook**
    - Mean-field lower cutoff factor: `0.75`
    - Curie-Weiss lower cutoff factor: `1.15`
    - Temperature-resolution floor: $\sigma_{T,\mathrm{res}} = {SIGMA_T_RES_K:.6e}$ K
    - The mean-field and Curie-Weiss point selections are data-dependent after these fixed thresholds are applied, so the fitted point counts are reported separately in the results table.
    """.replace("{SIGMA_T_RES_K:.6e}", f"{SIGMA_T_RES_K:.6e}"))
    return


@app.cell
def _(mo):
    mf_cut_factor = mo.ui.slider(
        start=0.5,
        stop=1.0,
        step=0.01,
        value=0.76,
        label="Mean-field Cut Factor (T >= factor * Tc_rough)",
    )
    mf_cut_factor  # type: ignore
    return (mf_cut_factor,)


@app.cell
def _(mo):
    cw_cut_factor = mo.ui.slider(
        start=1.0,
        stop=2.0,
        step=0.01,
        value=1.15,
        label="Curie-Weiss Cut Factor (T > factor * Tc_rough)",
    )
    cw_cut_factor  # type: ignore
    return (cw_cut_factor,)


@app.cell
def _(
    SERIES_ORDER,
    cw_cut_factor,
    filtered_series_data,
    fit_curie_weiss,
    fit_mean_field,
    mf_cut_factor,
):
    # fit_results[model][series][proxy_method]
    fit_results = {
        "mean_field": {s: {} for s in SERIES_ORDER},
        "curie_weiss": {s: {} for s in SERIES_ORDER},
    }
    proxy_methods = ["method1", "method2", "method3"]

    for _series_name in SERIES_ORDER:
        _d = filtered_series_data[_series_name]
        _T = _d["T_K"]
        _tr = _d["Tc_rough"]

        for pm in proxy_methods:
            M = _d[f"{pm}_norm"]
            sM = _d[f"{pm}_sigma_norm"]

            # Mean-field fit
            ff = fit_mean_field(
                _T,
                _d["sigma_T_K"],
                M,
                sM,
                Tc_seed=215.0,
                T_rough=_tr,
                cut_factor=mf_cut_factor.value,
            )
            fit_results["mean_field"][_series_name][pm] = ff

            # Curie-Weiss fit (using M proxy as proxy for chi in paramagnetic regime)
            cw = fit_curie_weiss(
                _T,
                _d["sigma_T_K"],
                M,
                sM,
                Tc_seed=ff["Tc"] if ff["ok"] else 215.0,
                T_rough=_tr,
                cut_factor=cw_cut_factor.value,
            )
            fit_results["curie_weiss"][_series_name][pm] = cw
    return (fit_results,)


@app.cell
def _(SERIES_ORDER, chi2_dist, fit_results, np, pd):
    proxy_keys = ["method1", "method2", "method3"]
    proxy_labels = {
        "method1": "M1 (Inter.)",
        "method2": "M2 (Sat.)",
        "method3": "M3 (Tail)",
    }

    fit_table_rows = []
    for model_key, model_label in [
        ("mean_field", "Mean-field"),
        ("curie_weiss", "Curie-Weiss"),
    ]:
        for series in SERIES_ORDER:
            row = {"Series": series, "Model": model_label}
            vals, sigs = [], []
            n_list = []

            for pk in proxy_keys:
                res = fit_results[model_key][series][pk]
                if res["ok"]:
                    row[proxy_labels[pk]] = f"{res['Tc']:.2f} ± {res['sigma_Tc']:.2f}"
                    n_list.append(str(res["n"]))
                    vals.append(res["Tc"])
                    sigs.append(res["sigma_Tc"])
                else:
                    row[proxy_labels[pk]] = "Fail"
                    n_list.append("0")

            row["N (M1,M2,M3)"] = ", ".join(n_list)
            if vals:
                w = 1.0 / (np.array(sigs, dtype=float) ** 2)
                w_mean = np.sum(w * vals) / np.sum(w)
                w_sigma = np.sqrt(1.0 / np.sum(w))
                row["Weighted Avg Tc [K]"] = f"{w_mean:.2f} ± {w_sigma:.2f}"
            else:
                row["Weighted Avg Tc [K]"] = "N/A"
            fit_table_rows.append(row)

    summary_stats = []
    for method_key, method_label in [
        ("mean_field", "Mean-field"),
        ("curie_weiss", "Curie-Weiss"),
    ]:

        def get_stats(series_list):
            vals, sigs, chi2s, pvals, ns = [], [], [], [], []
            for series in series_list:
                for pm in proxy_keys:
                    res = fit_results[method_key][series][pm]
                    if res["ok"] and np.isfinite(res["sigma_Tc"]) and res["sigma_Tc"] > 0:
                        vals.append(res["Tc"])
                        sigs.append(res["sigma_Tc"])
                        chi2s.append(res["chi2_red"])
                        dof = max(int(res["n"]) - 2, 1)
                        pvals.append(float(chi2_dist.sf(res["chi2_red"] * dof, dof)))
                        ns.append(res["n"])

            if not vals:
                return [np.nan] * 5 + [0, 0]

            vals, sigs = np.array(vals), np.array(sigs)
            w = 1.0 / (sigs**2)
            w_mean = np.sum(w * vals) / np.sum(w)
            w_sigma = np.sqrt(1.0 / np.sum(w))
            std_dev = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            return [
                w_mean,
                w_sigma,
                std_dev,
                np.median(chi2s),
                np.mean(chi2s),
                np.median(pvals),
                np.mean(pvals),
                len(vals),
                sum(ns),
            ]

        all_stats = get_stats(SERIES_ORDER)
        pref_stats = get_stats(["series A", "series B"])

        summary_stats.append(
            {
                "Method": method_label,
                "Scope": "All (A,B,C)",
                "Weighted Tc [K]": all_stats[0],
                "Weighted sigma [K]": all_stats[1],
                "Successful Fits": int(all_stats[7]),
                "Total Points": int(all_stats[8]),
                "Median chi2/dof": all_stats[3],
                "Mean chi2/dof": all_stats[4],
                "Median P-value": all_stats[5],
                "Mean P-value": all_stats[6],
            }
        )
        summary_stats.append(
            {
                "Method": method_label,
                "Scope": "Preferred (A,B)",
                "Weighted Tc [K]": pref_stats[0],
                "Weighted sigma [K]": pref_stats[1],
                "Successful Fits": int(pref_stats[7]),
                "Total Points": int(pref_stats[8]),
                "Median chi2/dof": pref_stats[3],
                "Mean chi2/dof": pref_stats[4],
                "Median P-value": pref_stats[5],
                "Mean P-value": pref_stats[6],
            }
        )

    impact_rows = []
    for model_key, model_label in [
        ("mean_field", "Mean-field"),
        ("curie_weiss", "Curie-Weiss"),
    ]:
        for scope_label, series_list in [
            ("All (A,B,C)", SERIES_ORDER),
            ("Preferred (A,B)", ["series A", "series B"]),
        ]:
            i_row = {"Model": model_label, "Scope": scope_label}
            for pk in proxy_keys:
                v, s, ns = [], [], []
                for series in series_list:
                    res = fit_results[model_key][series][pk]
                    if res["ok"] and np.isfinite(res["sigma_Tc"]) and res["sigma_Tc"] > 0:
                        v.append(res["Tc"])
                        s.append(res["sigma_Tc"])
                        ns.append(res["n"])
                if v:
                    w = 1.0 / (np.array(s, dtype=float) ** 2)
                    wm = np.sum(w * v) / np.sum(w)
                    ws = np.sqrt(1.0 / np.sum(w))
                    i_row[proxy_labels[pk]] = f"{wm:.2f} ± {ws:.2f}"
                    i_row[f"N({pk[-1]})"] = sum(ns)
                else:
                    i_row[proxy_labels[pk]] = "N/A"
                    i_row[f"N({pk[-1]})"] = 0
            impact_rows.append(i_row)

    df_summary = pd.DataFrame(summary_stats)
    df_fit_table = pd.DataFrame(fit_table_rows)
    df_method_impact = pd.DataFrame(impact_rows)
    return df_fit_table, df_method_impact, df_summary


@app.cell
def _(df_fit_table, df_method_impact, df_summary, mo):
    mo.md(rf"""
    ### Curie Temperature Results Table
    This table shows $T_c$ estimates for each model and series, computed across all three magnetization extraction methods (M1, M2, M3).
    The **N (M1,M2,M3)** column indicates the number of points used in each respective proxy fit.

    {df_fit_table.to_markdown(index=False)}

    ### Overall Method comparison and Series C impact
    "Successful Fits" counts successful models across all proxies and series; "Total Points" is the aggregate number of data points fitted.
    The summary table now includes both median and mean chi-square diagnostics, plus matching p-values computed from the same fit ensemble.

    {df_summary.to_markdown(index=False, floatfmt=".3f")}

    ### Method Consistency (Comparison of Part A proxies)
    This table compares the weighted average $T_c$ (across measurements) for each extraction method, allowing us to evaluate the impact of the magnetization proxy choice on the final $T_c$.

    {df_method_impact.to_markdown(index=False)}
    """)
    return


@app.cell
def _(
    COLORS,
    SERIES_ORDER,
    filtered_series_data,
    fit_results,
    np,
    plt,
    save_figure,
    sliders,
):
    def _():
        fig, axs = plt.subplots(
            2, 3, figsize=(14, 7), sharex="col", gridspec_kw={"height_ratios": [3, 1]}
        )
        proxy_methods = ["method1", "method2", "method3"]
        proxy_colors = {
            "method1": COLORS["m1"],
            "method2": COLORS["m2"],
            "method3": COLORS["m3"],
        }

        for j, _series_name in enumerate(SERIES_ORDER):
            ax_top = axs[0, j]
            ax_bot = axs[1, j]
            _d = filtered_series_data[_series_name]

            # Sync X-axis with sliders
            ax_top.set_xlim(sliders.value[_series_name])
            _tr = _d["Tc_rough"]
            ax_top.axvline(
                _tr, color="black", linestyle="--", alpha=0.4, label="Rough Tc"
            )
            ax_top.set_title(f"{_series_name} | Mean-field Fit")

            for pm in proxy_methods:
                res = fit_results["mean_field"][_series_name][pm]
                color = proxy_colors[pm]

                if res["ok"]:
                    Tf, yf, sf, _ = (
                        res["T_fit"],
                        res["y_fit"],
                        res["sigma_fit"],
                        res["residuals"],
                    )
                    Tc, A = res["Tc"], res["A"]

                    # Plot full data in background but show M^2 (normalized)
                    mask_full = np.isfinite(_d["T_K"]) & np.isfinite(_d[f"{pm}_norm"])
                    ax_top.plot(
                        _d["T_K"][mask_full],
                        _d[f"{pm}_norm"][mask_full] ** 2,
                        "o",
                        ms=2,
                        color=color,
                        alpha=0.15,
                    )

                    # Prepare squared fit-values and propagate uncertainties: sigma(M^2)=2*|M|*sigma_M
                    yf_sq = yf**2
                    sf_sq = np.maximum(1e-12, 2.0 * np.abs(yf) * sf)

                    # Plot fit data in squared space
                    ax_top.errorbar(
                        Tf, yf_sq, yerr=sf_sq, fmt="o", ms=3, color=color, alpha=0.6
                    )

                    # Plot model in squared space: (A*sqrt(Tc - T))^2 = A^2 * (Tc - T)
                    xline = np.linspace(np.min(Tf), np.max(Tf), 100)
                    yline = (A**2) * np.clip(Tc - xline, 0.0, None)
                    ax_top.plot(
                        xline,
                        yline,
                        "-",
                        color=color,
                        lw=1.8,
                        label=f"{pm}: Tc={Tc:.1f}K",
                    )

                    # Residuals in squared space
                    ymodel_at_Tf = (A**2) * np.clip(Tc - Tf, 0.0, None)
                    r_sq = (yf_sq - ymodel_at_Tf) / sf_sq
                    ax_bot.plot(Tf, r_sq, "o", ms=2.5, color=color, alpha=0.7)

            ax_top.set_ylabel("M^2 (normalized)")
            ax_top.legend(loc="lower left", fontsize=7)
            ax_bot.axhline(0.0, color="k", lw=0.8)
            ax_bot.set_xlabel("Temperature [K]")
            ax_bot.set_ylabel("res/sigma")

        fig.tight_layout()
        save_figure(fig, "mean_field_fits")
        return fig

    _()
    return


@app.cell
def _(
    COLORS,
    SERIES_ORDER,
    filtered_series_data,
    fit_results,
    np,
    plt,
    save_figure,
    sliders,
):
    def _():
        fig, axs = plt.subplots(
            2, 3, figsize=(14, 7), sharex="col", gridspec_kw={"height_ratios": [3, 1]}
        )
        proxy_methods = ["method1", "method2", "method3"]
        proxy_colors = {
            "method1": COLORS["m1"],
            "method2": COLORS["m2"],
            "method3": COLORS["m3"],
        }

        for j, _series_name in enumerate(SERIES_ORDER):
            ax_top = axs[0, j]
            ax_bot = axs[1, j]
            _d = filtered_series_data[_series_name]

            # Sync X-axis with sliders
            ax_top.set_xlim(sliders.value[_series_name])
            _tr = _d["Tc_rough"]
            ax_top.axvline(
                _tr, color="black", linestyle="--", alpha=0.4, label="Rough Tc"
            )
            ax_top.set_title(f"{_series_name} | Curie-Weiss Fit")
            y_max_fit = 0.0

            for pm in proxy_methods:
                res = fit_results["curie_weiss"][_series_name][pm]
                color = proxy_colors[pm]

                if res["ok"]:
                    Tf, sTf, yf, sf, r = (
                        res["T_fit"],
                        res["sigma_T_fit"],
                        res["y_fit"],
                        res["sigma_fit"],
                        res["residuals"],
                    )
                    Tc, C = res["Tc"], res["C"]
                    y_max_fit = max(y_max_fit, np.max(yf))

                    # Plot full inverse data in background
                    mask_full = np.isfinite(_d["T_K"]) & np.isfinite(_d[f"{pm}_norm"])
                    inv_full = 1.0 / np.maximum(_d[f"{pm}_norm"][mask_full], 1e-12)
                    ax_top.plot(
                        _d["T_K"][mask_full], inv_full, "o", ms=2, color=color, alpha=0.1
                    )

                    # Plot fit data
                    ax_top.errorbar(
                        Tf, yf, xerr=sTf, yerr=sf, fmt="o", ms=3, color=color, alpha=0.6
                    )

                    # Plot model
                    xline = np.linspace(np.min(Tf), np.max(Tf), 100)
                    yline = (xline - Tc) / C
                    ax_top.plot(
                        xline,
                        yline,
                        "-",
                        color=color,
                        lw=1.8,
                        label=f"{pm}: Tc={Tc:.1f}K",
                    )

                    # Residuals
                    ax_bot.plot(Tf, r, "o", ms=2.5, color=color, alpha=0.7)

            # Limit y-axis to fitted range to prevent 1/0 singularity from flattening the plot
            if y_max_fit > 0:
                ax_top.set_ylim(-0.05 * y_max_fit, 1.5 * y_max_fit)

            ax_top.set_ylabel("1/M (normalized)")
            ax_top.legend(loc="best", fontsize=7)
            ax_bot.axhline(0.0, color="k", lw=0.8)
            ax_bot.set_xlabel("Temperature [K]")
            ax_bot.set_ylabel("res/sigma")

        fig.tight_layout()
        save_figure(fig, "curie_weiss_fits")
        return fig

    _()
    return


@app.cell
def _(df_summary, mo):
    row_ff = df_summary.loc[df_summary["Method"] == "Mean-field"].iloc[0]
    row_cw = df_summary.loc[df_summary["Method"] == "Curie-Weiss"].iloc[0]

    ff_good = (
        row_ff["Median chi2/dof"] < row_cw["Median chi2/dof"]
        if (
            row_ff["Median chi2/dof"] == row_ff["Median chi2/dof"]
            and row_cw["Median chi2/dof"] == row_cw["Median chi2/dof"]
        )
        else True
    )
    best = "Mean-field" if ff_good else "Curie-Weiss"
    best_row = row_ff if ff_good else row_cw

    mo.md(rf"""
    ## Bottom line

    - Preferred method for final report: **{best}**
    - Recommended Curie temperature estimate: **Tc = {best_row["Weighted Tc [K]"]:.2f} ± {best_row["Weighted sigma [K]"]:.2f} K**
    - Median chi2/dof: **{best_row["Median chi2/dof"]:.3f}**
    - Median p-value: **{best_row["Median P-value"]:.3f}**

    This selection is based on the better residual behavior (lower median reduced chi-squared) and consistency across the three series.
    """)
    return


if __name__ == "__main__":
    app.run()
