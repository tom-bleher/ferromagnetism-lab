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
def _(mo):
    mo.md(r"""
    # Curie temperature analysis (clean notebook)

    This notebook rebuilds the Curie analysis from scratch with a clean, reproducible pipeline.

    **Data Cleaning & Processing**
    1. Setup and data preparation (including units and uncertainties)
    2. **Handling Instabilities**: Measurement C is automatically clipped to the region after its magnetization peak to remove initial stabilization instability.
    3. **Outlier Removal**: Singular anomalies are detected via a robust rolling median-deviation filter and removed from all series.
    4. Part A: extraction of magnetization proxies (M1, M2, M3) from loop geometries.
    5. Part B: Curie temperature extraction using Far-field and Curie-Weiss fits.

    **Notation**
    - \(T\): temperature in Kelvin
    - \(H\): external field proxy (oscilloscope X channel, volts)
    - \(B\): total field proxy (oscilloscope Y channel, volts)
    - \(M\): magnetization proxy extracted from hysteresis geometry
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.optimize import curve_fit

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.axisbelow": True,
        }
    )

    COLORS = {
        "m1": "#1b9e77",
        "m2": "#d95f02",
        "m3": "#7570b3",
        "fit": "#e7298a",
        "data": "#1f77b4",
    }
    return COLORS, Path, curve_fit, np, pd, plt


@app.cell
def _(Path, np):
    ROOT = Path(__file__).resolve().parent
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
        SIGMA_Y_V,
        normalize_with_uncertainty,
        temperature_uncertainty_k,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup details and preprocessing

    **Data preparation done here**
    - Loaded all three measurement series (`series A/B/C`) from `curie/data`.
    - Converted temperature from Celsius to Kelvin using \(T_K = T_C + 273.15\).
    - Kept the native oscilloscope channels as field proxies:
      - X channel \(\propto H\)
      - Y channel \(\propto B\)
    - **Automated Cleaning Applied**:
      - **Series C instability**: The measurement is clipped to the region *after* the initial magnetization peak to skip stabilization noise.
      - **Outliers**: Singular anomalies are identified using a 7-point rolling median threshold (Hampel filter) and removed from the extraction.
    - Built per-loop uncertainty arrays:
      - \(\sigma_T\): thermometer spec + temperature-resolution + finite-time drift
      - \(\sigma_X, \sigma_Y\): digitization uncertainty (uniform quantization model)

    **Uncertainty scope**
    - Random/statistical uncertainties are propagated through extraction and fits.
    - Hardware calibration systematics (component tolerances) are not provided in the guide;
      therefore they are not included in the numeric propagation below.
    """)
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

    def extract_loop_methods(x_pos, y_pos, x_neg, y_neg, edge_fraction=0.12):
        n = len(x_pos)
        n_edge = max(6, int(np.ceil(edge_fraction * n)))

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

    def _transition_temp(Tvals, Mvals):
        if len(Tvals) == 0:
            return np.nan
        idx = int(np.nanargmin(np.abs(Mvals - 0.5)))
        return float(Tvals[idx])

    def fit_far_field(T_K, M_norm, sigma_M, Tc_seed=215.0):
        _T = np.asarray(T_K, dtype=float)
        M = np.asarray(M_norm, dtype=float)
        sM = np.asarray(sigma_M, dtype=float)

        # Masking for physical validity: mean-field law is valid near transition
        # but fails at low-T saturation or near the finite-field 'tail' at Tc.
        mask = np.isfinite(_T) & np.isfinite(M) & np.isfinite(sM) & (M > 0.05) & (M < 0.9)
        if np.sum(mask) < 8:
            mask = np.isfinite(_T) & np.isfinite(M) & np.isfinite(sM) & (M > 0.02)
        if np.sum(mask) < 6:
            return {"ok": False, "reason": "insufficient points"}

        Tf = _T[mask]
        Mf = M[mask]
        sMf = np.maximum(sM[mask], 1e-4)

        def model(Tloc, Tc, A):
            return A * np.sqrt(np.clip(Tc - Tloc, 0.0, None))

        p0 = [Tc_seed, max(0.05, np.max(Mf) / np.sqrt(max(Tc_seed - np.min(Tf), 1.0)))]
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
            "y_fit": Mf,
            "sigma_fit": sMf,
            "y_model": pred,
            "residuals": res,
            "chi2_red": chi2_red,
            "n": int(len(Tf)),
        }

    def fit_curie_weiss(T_K, chi, sigma_chi, Tc_seed=215.0):
        _T = np.asarray(T_K, dtype=float)
        chi = np.asarray(chi, dtype=float)
        schi = np.asarray(sigma_chi, dtype=float)
        mask = np.isfinite(_T) & np.isfinite(chi) & np.isfinite(schi) & (chi > 0.0)
        if np.sum(mask) < 10:
            return {"ok": False, "reason": "insufficient chi points"}

        inv_chi = 1.0 / chi
        sigma_inv = schi / np.maximum(chi, 1e-12) ** 2
        top_start = np.quantile(_T[mask], 0.65)
        cut = max(float(Tc_seed + 2.0), float(top_start))
        fit_mask = mask & (_T >= cut)
        if np.sum(fit_mask) < 6:
            fit_mask = mask & (_T >= np.quantile(_T[mask], 0.60))
        if np.sum(fit_mask) < 6:
            return {"ok": False, "reason": "insufficient high-T points"}

        Tf = _T[fit_mask]
        Yf = inv_chi[fit_mask]
        sYf = np.maximum(sigma_inv[fit_mask], 1e-6)

        def model(Tloc, Tc, C):
            return (Tloc - Tc) / C

        popt, pcov = curve_fit(
            model,
            Tf,
            Yf,
            p0=[Tc_seed, 1.0],
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
            "y_fit": Yf,
            "sigma_fit": sYf,
            "y_model": pred,
            "residuals": res,
            "chi2_red": chi2_red,
            "n": int(len(Tf)),
        }

    return (
        extract_chi_near_zero,
        extract_loop_methods,
        fit_curie_weiss,
        fit_far_field,
        sorted_channel_columns,
        _transition_temp,
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
        if _series_name == "series C":
            # Detect the peak on a smoothed version of M3 (the cleanest proxy)
            m3_filled = np.nan_to_num(m3, nan=np.nanmedian(m3))
            m3_smooth = np.convolve(m3_filled, np.ones(5) / 5, mode="same")
            start_idx = int(np.argmax(m3_smooth))
        else:
            start_idx = min(4, n_loops)

        # 2. Outlier Detection: Identify singular spikes using a rolling median
        # Basically, take a rolling windows of 7 points, compute the median, and flag points that deviate
        # from that median by more than ~4.5 times the median absolute deviation as outliers.
        m3_ser = pd.Series(m3)
        m3_med = m3_ser.rolling(window=7, center=True).median().ffill().bfill().to_numpy()
        m3_resid = np.abs(m3 - m3_med)
        m3_mad = np.nanmedian(m3_resid)
        # Threshold for 'singular points that are anomalies' (approx 4.5 sigma equivalent)
        is_outlier = m3_resid > (4.8 * max(m3_mad, 1e-6))

        # Apply Cleaning Mask to all local variables before saving to dictionary
        valid_points = (np.arange(n_loops) >= start_idx) & (~is_outlier)

        T_C, T_K = T_C[valid_points], T_K[valid_points]
        time_s, sigma_T = time_s[valid_points], sigma_T[valid_points]
        X_pos, Y_pos = X_pos[valid_points], Y_pos[valid_points]
        X_neg, Y_neg = X_neg[valid_points], Y_neg[valid_points]
        m1, s1 = m1[valid_points], s1[valid_points]
        m2, s2 = m2[valid_points], s2[valid_points]
        m3, s3 = m3[valid_points], s3[valid_points]
        chi0, schi0 = chi0[valid_points], schi0[valid_points]
        tail_points = tail_points[valid_points]

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

    return series_data


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
def _(SERIES_ORDER, filtered_series_data, mo, np, pd, _transition_temp):
    prep_rows = []
    part_a_rows = []

    for name in SERIES_ORDER:
        _d = filtered_series_data[name]
        _T = _d["T_K"]

        prep_rows.append(
            {
                "Series": name,
                "Loops": len(_T),
                "T range [K]": f"{np.min(_T):.1f} to {np.max(_T):.1f}" if len(_T) > 0 else "N/A",
                "median σT [K]": f"{np.median(_d['sigma_T_K']):.2f}",
                "tail points (M3)": int(np.nanmedian(_d["tail_points"])),
            }
        )

        # Calculate summary row reactively based on slider range
        part_a_rows.extend([
            {
                "Series": name,
                "Method": "M1 @ H=0",
                "T(M=0.5) [K]": _transition_temp(_T, _d["method1_norm"]),
                "Range(raw)": float(np.nanmax(_d["method1"]) - np.nanmin(_d["method1"])) if len(_T) > 0 else 0,
            },
            {
                "Series": name,
                "Method": "M2 saturation edges",
                "T(M=0.5) [K]": _transition_temp(_T, _d["method2_norm"]),
                "Range(raw)": float(np.nanmax(_d["method2"]) - np.nanmin(_d["method2"])) if len(_T) > 0 else 0,
            },
            {
                "Series": name,
                "Method": "M3 tail extrapolation",
                "T(M=0.5) [K]": _transition_temp(_T, _d["method3_norm"]),
                "Range(raw)": float(np.nanmax(_d["method3"]) - np.nanmin(_d["method3"])) if len(_T) > 0 else 0,
            }
        ])

    prep_df = pd.DataFrame(prep_rows)
    part_a_summary_df = pd.DataFrame(part_a_rows)

    mo.md(f"""
    **Setup summary**

    {prep_df.to_markdown(index=False)}

    **Part A short table (normalized-curve midpoint and raw dynamic range)**

    {part_a_summary_df.to_markdown(index=False, floatfmt=".3f")}
    """)
    return part_a_summary_df, prep_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part A — Magnetization curves from three extraction methods

    For each measurement series (A, B, C), one graph is shown with all three normalized \(M(T)\) curves:
    1. **Method 1**: loop intersection at \(H=0\)
    2. **Method 2**: saturation-edge values
    3. **Method 3**: adaptive linear fit to saturation tails and extrapolation to \(H=0\)
    """)
    return


@app.cell
def _(COLORS, SERIES_ORDER, filtered_series_data, plt):
    def _():
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))
        for _i, _series_name in enumerate(SERIES_ORDER):
            _d = filtered_series_data[_series_name]
            _T = _d["T_K"]
            ax = axs[_i]
            ax.errorbar(
                _T,
                _d["method1_norm"],
                yerr=_d["method1_sigma_norm"],
                fmt="o-",
                ms=3.5,
                lw=1.3,
                color=COLORS["m1"],
                alpha=0.9,
                label="Method 1: H=0 intersection",
            )
            ax.errorbar(
                _T,
                _d["method2_norm"],
                yerr=_d["method2_sigma_norm"],
                fmt="s-",
                ms=3.2,
                lw=1.3,
                color=COLORS["m2"],
                alpha=0.9,
                label="Method 2: saturation edges",
            )
            ax.errorbar(
                _T,
                _d["method3_norm"],
                yerr=_d["method3_sigma_norm"],
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
    fig  # type: ignore
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part B — Curie temperature extraction

    Two fitting methods are applied to each series:
    1. **Far-field (mean-field near transition):** \(M(T)=A\sqrt{T_C-T}\)
    2. **Curie–Weiss:** \(1/\chi(T)=(T-T_C)/C\)

    The requested third option (second-derivative divergence) is skipped here because the finite-point numerical second derivative is too unstable for these datasets and gives non-robust \(T_C\) estimates.

    **Regime Selection (Data Cutting)**
    The physical models are only valid in specific temperature regimes. To ensure robust fits, points outside these regimes are automatically masked:
    - **Far-field**: Only uses points where \(0.05 < M < 0.9\) to avoid saturation at low-T and finite-field smearing near \(T_c\).
    - **Curie–Weiss**: Only uses points in the paramagnetic regime (above \(T_c + \text{buffer}\)).

    The plots below show the full slider-filtered range in light gray for context.
    """)
    return


@app.cell
def _(SERIES_ORDER, filtered_series_data, fit_curie_weiss, fit_far_field, pd):
    fit_results = {"far_field": {}, "curie_weiss": {}}
    fit_rows = []

    for _series_name in SERIES_ORDER:
        _d = filtered_series_data[_series_name]
        _T = _d["T_K"]
        M = _d["method3_norm"]
        sM = _d["method3_sigma_norm"]

        ff = fit_far_field(_T, M, sM, Tc_seed=215.0)
        fit_results["far_field"][_series_name] = ff
        if ff["ok"]:
            fit_rows.append(
                {
                    "Series": _series_name,
                    "Fit method": "Far-field",
                    "Tc [K]": ff["Tc"],
                    "sigma Tc [K]": ff["sigma_Tc"],
                    "chi2/dof": ff["chi2_red"],
                    "Nfit": ff["n"],
                }
            )
        else:
            fit_rows.append(
                {
                    "Series": _series_name,
                    "Fit method": "Far-field",
                    "Tc [K]": float("nan"),
                    "sigma Tc [K]": float("nan"),
                    "chi2/dof": float("nan"),
                    "Nfit": 0,
                }
            )

        cw = fit_curie_weiss(
            _T, _d["chi0"], _d["chi0_sigma"], Tc_seed=ff["Tc"] if ff["ok"] else 215.0
        )
        fit_results["curie_weiss"][_series_name] = cw
        if cw["ok"]:
            fit_rows.append(
                {
                    "Series": _series_name,
                    "Fit method": "Curie-Weiss",
                    "Tc [K]": cw["Tc"],
                    "sigma Tc [K]": cw["sigma_Tc"],
                    "chi2/dof": cw["chi2_red"],
                    "Nfit": cw["n"],
                }
            )
        else:
            fit_rows.append(
                {
                    "Series": _series_name,
                    "Fit method": "Curie-Weiss",
                    "Tc [K]": float("nan"),
                    "sigma Tc [K]": float("nan"),
                    "chi2/dof": float("nan"),
                    "Nfit": 0,
                }
            )

    fit_table_df = pd.DataFrame(fit_rows)
    return fit_results, fit_table_df


@app.cell
def _(SERIES_ORDER, fit_results, np, pd):
    def _():
        comparison_rows = []
        for method_key, method_label in [
            ("far_field", "Far-field"),
            ("curie_weiss", "Curie-Weiss"),
        ]:
            vals = []
            sigs = []
            chi2s = []
            for _series_name in SERIES_ORDER:
                res = fit_results[method_key][_series_name]
                if res["ok"] and np.isfinite(res["sigma_Tc"]) and res["sigma_Tc"] > 0:
                    vals.append(res["Tc"])
                    sigs.append(res["sigma_Tc"])
                    chi2s.append(res["chi2_red"])

            if len(vals) > 0:
                vals = np.asarray(vals, dtype=float)
                sigs = np.asarray(sigs, dtype=float)
                w = 1.0 / (sigs**2)
                Tc_wmean = float(np.sum(w * vals) / np.sum(w))
                Tc_wsigma = float(np.sqrt(1.0 / np.sum(w)))
                Tc_std = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
                chi2_med = float(np.median(chi2s))
                n_good = len(vals)
            else:
                Tc_wmean = np.nan
                Tc_wsigma = np.nan
                Tc_std = np.nan
                chi2_med = np.nan
                n_good = 0
            comparison_rows.append(
                {
                    "Method": method_label,
                    "Series used": n_good,
                    "Weighted Tc [K]": Tc_wmean,
                    "Weighted sigma [K]": Tc_wsigma,
                    "Inter-series std [K]": Tc_std,
                    "Median chi2/dof": chi2_med,
                }
            )

        return comparison_rows

    comparison_rows = _()
    comparison_df = pd.DataFrame(comparison_rows)
    return (comparison_df,)


@app.cell
def _(comparison_df, fit_table_df, mo):
    mo.md(f"""
    **Fit results per series**

    {fit_table_df.to_markdown(index=False, floatfmt=".3f")}

    **Method comparison**

    {comparison_df.to_markdown(index=False, floatfmt=".3f")}
    """)
    return


@app.cell
def _(COLORS, SERIES_ORDER, fit_results, filtered_series_data, sliders, np, plt):
    def _():
        fig, axs = plt.subplots(
            2, 3, figsize=(13.0, 6.2), sharex="col", gridspec_kw={"height_ratios": [3, 1]}
        )
        for j, _series_name in enumerate(SERIES_ORDER):
            res = fit_results["far_field"][_series_name]
            ax_top = axs[0, j]
            ax_bot = axs[1, j]

            # Sync X-axis with sliders
            ax_top.set_xlim(sliders.value[_series_name])

            if res["ok"]:
                _d = filtered_series_data[_series_name]
                ax_top.plot(_d["T_K"], _d["method3_norm"], "o", ms=2, color="lightgray", alpha=0.4, label="excluded from fit")

                Tf = res["T_fit"]
                yf = res["y_fit"]
                sf = res["sigma_fit"]
                r = res["residuals"]

                xline = np.linspace(np.min(Tf), np.max(Tf), 200)
                Tc = res["Tc"]
                A = res["A"]
                yline = A * np.sqrt(np.clip(Tc - xline, 0.0, None))

                ax_top.errorbar(
                    Tf, yf, yerr=sf, fmt="o", ms=3, color=COLORS["data"], label="data"
                )
                ax_top.plot(
                    xline,
                    yline,
                    "-",
                    color=COLORS["fit"],
                    lw=1.5,
                    label=f"fit Tc={Tc:.2f} K",
                )
                ax_top.set_title(f"{_series_name} | Far-field")
                ax_top.set_ylabel("M (normalized)")
                ax_top.legend(loc="best", fontsize=8)

                ax_bot.axhline(0.0, color="k", lw=1.0)
                ax_bot.plot(Tf, r, "o", ms=3, color=COLORS["fit"])
                ax_bot.set_xlabel("Temperature [K]")
                ax_bot.set_ylabel("res/σ")
            else:
                ax_top.set_title(f"{_series_name} | Far-field (fit failed)")
                ax_bot.set_xlabel("Temperature [K]")
                ax_bot.set_ylabel("res/σ")
                ax_top.text(
                    0.5,
                    0.5,
                    "No stable fit",
                    ha="center",
                    va="center",
                    transform=ax_top.transAxes,
                )
        fig.tight_layout()
        return fig

    _()
    return


@app.cell
def _(COLORS, SERIES_ORDER, fit_results, filtered_series_data, sliders, np, plt):
    def _():
        fig, axs = plt.subplots(
            2, 3, figsize=(13.0, 6.2), sharex="col", gridspec_kw={"height_ratios": [3, 1]}
        )
        for j, _series_name in enumerate(SERIES_ORDER):
            res = fit_results["curie_weiss"][_series_name]
            ax_top = axs[0, j]
            ax_bot = axs[1, j]

            # Sync X-axis with sliders
            ax_top.set_xlim(sliders.value[_series_name])

            if res["ok"]:
                _d = filtered_series_data[_series_name]
                _inv_chi_full = 1.0 / np.maximum(_d["chi0"], 1e-12)
                ax_top.plot(_d["T_K"], _inv_chi_full, "o", ms=2, color="lightgray", alpha=0.4, label="excluded from fit")

                Tf = res["T_fit"]
                yf = res["y_fit"]
                sf = res["sigma_fit"]
                r = res["residuals"]

                xline = np.linspace(float(min(Tf)), float(max(Tf)), 200)
                Tc = res["Tc"]
                C = res["C"]
                yline = (xline - Tc) / C

                ax_top.errorbar(
                    Tf, yf, yerr=sf, fmt="o", ms=3, color=COLORS["data"], label="data"
                )
                ax_top.plot(
                    xline,
                    yline,
                    "-",
                    color=COLORS["fit"],
                    lw=1.5,
                    label=f"fit Tc={Tc:.2f} K",
                )
                ax_top.set_title(f"{_series_name} | Curie-Weiss")
                ax_top.set_ylabel("1/χ (arb. units)")
                ax_top.legend(loc="best", fontsize=8)

                ax_bot.axhline(0.0, color="k", lw=1.0)
                ax_bot.plot(Tf, r, "o", ms=3, color=COLORS["fit"])
                ax_bot.set_xlabel("Temperature [K]")
                ax_bot.set_ylabel("res/σ")
            else:
                ax_top.set_title(f"{_series_name} | Curie-Weiss (fit failed)")
                ax_bot.set_xlabel("Temperature [K]")
                ax_bot.set_ylabel("res/σ")
                ax_top.text(
                    0.5,
                    0.5,
                    "No stable fit",
                    ha="center",
                    va="center",
                    transform=ax_top.transAxes,
                )
        fig.tight_layout()
        return fig

    _()
    return


@app.cell
def _(comparison_df, mo):
    row_ff = comparison_df.loc[comparison_df["Method"] == "Far-field"].iloc[0]
    row_cw = comparison_df.loc[comparison_df["Method"] == "Curie-Weiss"].iloc[0]

    ff_good = (
        row_ff["Median chi2/dof"] < row_cw["Median chi2/dof"]
        if (
            row_ff["Median chi2/dof"] == row_ff["Median chi2/dof"]
            and row_cw["Median chi2/dof"] == row_cw["Median chi2/dof"]
        )
        else True
    )
    best = "Far-field" if ff_good else "Curie-Weiss"
    best_row = row_ff if ff_good else row_cw

    mo.md(f"""
    ## Bottom line

    - Preferred method for final report: **{best}**
    - Recommended Curie temperature estimate: **Tc = {best_row["Weighted Tc [K]"]:.2f} ± {best_row["Weighted sigma [K]"]:.2f} K**

    This selection is based on the better residual behavior (lower median reduced chi-squared) and consistency across the three series.
    """)
    return


if __name__ == "__main__":
    app.run()
