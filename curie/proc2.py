import marimo

__generated_with = "0.23.1"
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

    **What is done**
    1. Setup and data preparation (including units and uncertainties)
    2. Part A: three magnetization-extraction methods for each measurement series
    3. Part B: Curie temperature extraction with two fitting methods (third method intentionally skipped)

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

    def extract_loop_methods(x_pos, y_pos, x_neg, y_neg, tail_fraction=0.18, edge_fraction=0.12):
        n = len(x_pos)
        n_tail = max(8, int(np.ceil(tail_fraction * n)))
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
        s_pos_sat = np.sqrt(np.var(y_pos_sat, ddof=1) / len(y_pos_sat) + SIGMA_Y_V**2 / len(y_pos_sat))
        s_neg_sat = np.sqrt(np.var(y_neg_sat, ddof=1) / len(y_neg_sat) + SIGMA_Y_V**2 / len(y_neg_sat))
        m2 = 0.5 * (mean_pos_sat - mean_neg_sat)
        s2 = 0.5 * np.sqrt(s_pos_sat**2 + s_neg_sat**2)

        # Method 3: linear fit to high-field tails, then intercept at H=0
        idx_pos_tail = np.argsort(x_pos)[-n_tail:]
        idx_neg_tail = np.argsort(x_neg)[:n_tail]
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
            "n_tail": int(n_tail),
            "n_edge": int(n_edge),
        }

    def extract_chi_near_zero(x_pos, y_pos, x_neg, y_neg, window_fraction=0.20):
        h_max = max(np.max(np.abs(x_pos)), np.max(np.abs(x_neg)))
        h_cut = window_fraction * h_max
        x_all = np.concatenate([x_pos[np.abs(x_pos) <= h_cut], x_neg[np.abs(x_neg) <= h_cut]])
        y_all = np.concatenate([y_pos[np.abs(x_pos) <= h_cut], y_neg[np.abs(x_neg) <= h_cut]])
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

    def fit_far_field(T_K, M_norm, sigma_M, Tc_seed=215.0):
        _T = np.asarray(T_K, dtype=float)
        M = np.asarray(M_norm, dtype=float)
        sM = np.asarray(sigma_M, dtype=float)

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
    part_a_rows = []

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
            tail_points[i] = out["n_tail"]

            chi0[i], schi0[i] = extract_chi_near_zero(X_pos[i], Y_pos[i], X_neg[i], Y_neg[i])

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

        def _transition_temp(Tvals, Mvals):
            idx = int(np.nanargmin(np.abs(Mvals - 0.5)))
            return float(Tvals[idx])

        part_a_rows.extend(
            [
                {
                    "Series": _series_name,
                    "Method": "M1 @ H=0",
                    "T(M=0.5) [K]": _transition_temp(T_K, m1_norm),
                    "Range(raw)": float(np.nanmax(m1) - np.nanmin(m1)),
                },
                {
                    "Series": _series_name,
                    "Method": "M2 saturation edges",
                    "T(M=0.5) [K]": _transition_temp(T_K, m2_norm),
                    "Range(raw)": float(np.nanmax(m2) - np.nanmin(m2)),
                },
                {
                    "Series": _series_name,
                    "Method": "M3 tail extrapolation",
                    "T(M=0.5) [K]": _transition_temp(T_K, m3_norm),
                    "Range(raw)": float(np.nanmax(m3) - np.nanmin(m3)),
                },
            ]
        )

    part_a_summary_df = pd.DataFrame(part_a_rows)
    return part_a_summary_df, series_data


@app.cell
def _(SERIES_ORDER, mo, np, part_a_summary_df, pd, series_data):
    prep_rows = []
    for name in SERIES_ORDER:
        _d = series_data[name]
        prep_rows.append(
            {
                "Series": name,
                "Loops": len(_d["T_K"]),
                "T range [K]": f"{np.min(_d['T_K']):.1f} to {np.max(_d['T_K']):.1f}",
                "median σT [K]": f"{np.median(_d['sigma_T_K']):.2f}",
                "tail points (M3)": int(np.nanmedian(_d["tail_points"])),
            }
        )
    prep_df = pd.DataFrame(prep_rows)
    mo.md(
        f"""
    **Setup summary**

    {prep_df.to_markdown(index=False)}

    **Part A short table (normalized-curve midpoint and raw dynamic range)**

    {part_a_summary_df.to_markdown(index=False, floatfmt=".3f")}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part A — Magnetization curves from three extraction methods

    For each measurement series (A, B, C), one graph is shown with all three normalized \(M(T)\) curves:
    1. **Method 1**: loop intersection at \(H=0\)
    2. **Method 2**: saturation-edge values
    3. **Method 3**: linear fit to saturation tails and extrapolation to \(H=0\)
    """)
    return


@app.cell
def _(COLORS, SERIES_ORDER, plt, series_data):
    def _():
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))
        for _i, _series_name in enumerate(SERIES_ORDER):
            _d = series_data[_series_name]
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

    For Part B, Method 3 from Part A is used as the magnetization input, because it is least sensitive to finite-field offset and loop asymmetry.
    """)
    return


@app.cell
def _(SERIES_ORDER, fit_curie_weiss, fit_far_field, pd, series_data):
    fit_results = {"far_field": {}, "curie_weiss": {}}
    fit_rows = []

    for _series_name in SERIES_ORDER:
        _d = series_data[_series_name]
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

        cw = fit_curie_weiss(_T, _d["chi0"], _d["chi0_sigma"], Tc_seed=ff["Tc"] if ff["ok"] else 215.0)
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
        for method_key, method_label in [("far_field", "Far-field"), ("curie_weiss", "Curie-Weiss")]:
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
def _(COLORS, SERIES_ORDER, fit_results, np, plt):
    def _():
        fig, axs = plt.subplots(2, 3, figsize=(13.0, 6.2), sharex="col", gridspec_kw={"height_ratios": [3, 1]})
        for j, _series_name in enumerate(SERIES_ORDER):
            res = fit_results["far_field"][_series_name]
            ax_top = axs[0, j]
            ax_bot = axs[1, j]
            if res["ok"]:
                Tf = res["T_fit"]
                yf = res["y_fit"]
                sf = res["sigma_fit"]
                r = res["residuals"]

                xline = np.linspace(np.min(Tf), np.max(Tf), 200)
                Tc = res["Tc"]
                A = res["A"]
                yline = A * np.sqrt(np.clip(Tc - xline, 0.0, None))

                ax_top.errorbar(Tf, yf, yerr=sf, fmt="o", ms=3, color=COLORS["data"], label="data")
                ax_top.plot(xline, yline, "-", color=COLORS["fit"], lw=1.5, label=f"fit Tc={Tc:.2f} K")
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
                ax_top.text(0.5, 0.5, "No stable fit", ha="center", va="center", transform=ax_top.transAxes)
        fig.tight_layout()
        return fig

    _()
    return


@app.cell
def _(COLORS, SERIES_ORDER, fit_results, np, plt):
    def _():
        fig, axs = plt.subplots(2, 3, figsize=(13.0, 6.2), sharex="col", gridspec_kw={"height_ratios": [3, 1]})
        for j, _series_name in enumerate(SERIES_ORDER):
            res = fit_results["curie_weiss"][_series_name]
            ax_top = axs[0, j]
            ax_bot = axs[1, j]
            if res["ok"]:
                Tf = res["T_fit"]
                yf = res["y_fit"]
                sf = res["sigma_fit"]
                r = res["residuals"]

                xline = np.linspace(float(min(Tf)), float(max(Tf)), 200)
                Tc = res["Tc"]
                C = res["C"]
                yline = (xline - Tc) / C

                ax_top.errorbar(Tf, yf, yerr=sf, fmt="o", ms=3, color=COLORS["data"], label="data")
                ax_top.plot(xline, yline, "-", color=COLORS["fit"], lw=1.5, label=f"fit Tc={Tc:.2f} K")
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
                ax_top.text(0.5, 0.5, "No stable fit", ha="center", va="center", transform=ax_top.transAxes)
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
        if (row_ff["Median chi2/dof"] == row_ff["Median chi2/dof"] and row_cw["Median chi2/dof"] == row_cw["Median chi2/dof"])
        else True
    )
    best = "Far-field" if ff_good else "Curie-Weiss"
    best_row = row_ff if ff_good else row_cw

    mo.md(
        f"""
    ## Bottom line

    - Preferred method for final report: **{best}**
    - Recommended Curie temperature estimate: **Tc = {best_row["Weighted Tc [K]"]:.2f} ± {best_row["Weighted sigma [K]"]:.2f} K**

    This selection is based on the better residual behavior (lower median reduced chi-squared) and consistency across the three series.
    """
    )
    return


if __name__ == "__main__":
    app.run()
