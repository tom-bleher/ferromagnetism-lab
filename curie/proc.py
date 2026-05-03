import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium", app_title="Curie temperature sketch")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Curie-temperature processing notebook

    First-pass processing for the measured Curie data file
    `data/first/CurieData_06_05_23-03_05_2026.txt`.

    Scope:

    - calibrate LabVIEW voltages with the experiment relations from the guides;
    - subtract a high-temperature linear magnetic background;
    - implement three methods to extract a per-loop order parameter:
        1. **remanence** $M_r(T)$ at $H=0$,
        2. **fixed-field branch split** $M_\mathrm{sep}(T;H_*)$,
        3. **high-field saturation extrapolation** $M_0(T)$ from the
           reversible asymptotic regime of each branch;
    - chain Method 3 into a weighted mean-field fit
      $M_0^2(T)\propto T_c-T$ to extract $T_c\pm\sigma_{T_c}$.

    The calibration relations used are

    $$
    H = \frac{N_1 V_x}{L R_x}, \qquad
    B = \frac{R C V_y}{N_2 A}, \qquad
    M = \frac{B}{\mu_0} - H.
    $$
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from scipy.signal import savgol_filter

    plt.rcParams.update({
        "figure.figsize": (7.2, 4.5),
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "font.size": 9,
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "cm",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "legend.fontsize": 8,
        "legend.framealpha": 0.95,
        "legend.handlelength": 2.6,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })

    ROOT = Path(__file__).resolve().parent
    DATA_FILE = ROOT / "data" / "first" / "CurieData_06_05_23-03_05_2026.txt"
    DATA_XLSX = ROOT.parent / "ferromagnetism" / "data" / "data.xlsx"
    FIG_DIR = ROOT.parent.parent / "report" / "media"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    MU0 = 1.25663706127e-6
    FIELD_READY_FRACTION = 0.98
    TARGET_LOOP_TEMPERATURES_K = [180, 195, 210, 225, 240, 260, 278]

    def save_figure(fig, stem):
        fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(FIG_DIR / f"{stem}.png", bbox_inches="tight", dpi=600)

    return (
        DATA_FILE,
        DATA_XLSX,
        FIELD_READY_FRACTION,
        FIG_DIR,
        Line2D,
        MU0,
        Normalize,
        TARGET_LOOP_TEMPERATURES_K,
        np,
        pd,
        plt,
        save_figure,
        savgol_filter,
    )


@app.cell
def _(DATA_FILE, DATA_XLSX, mo, pd):
    data = pd.read_csv(DATA_FILE, sep="\t")
    apparatus = pd.read_excel(DATA_XLSX, sheet_name="apparatus").iloc[0]
    temperature_K = data["Temperature (C)"] + 273.15

    X_POS = [c for c in data.columns if c.startswith("X_pos")]
    Y_POS = [c for c in data.columns if c.startswith("Y_pos")]
    X_NEG = [c for c in data.columns if c.startswith("X_neg")]
    Y_NEG = [c for c in data.columns if c.startswith("Y_neg")]

    N = int(apparatus["N"])
    L = float(apparatus["L (m)"])
    Rx = float(apparatus["Rx (Ω)"])
    Ry = float(apparatus["Ry (Ω)"])
    C = float(apparatus["C (F)"])
    A = float(apparatus["A (m²)"])

    H_PER_X = N / (L * Rx)
    B_PER_Y = Ry * C / (N * A)

    mo.md(f"""
    **Loaded data**

    - rows: `{len(data)}` loops
    - samples per branch: `{len(X_POS)}`
    - temperature range: `{temperature_K.min():.3f}` to `{temperature_K.max():.3f}` K
    - calibration constants: `H/Vx = {H_PER_X:.6g} A m^-1 V^-1`, `B/Vy = {B_PER_Y:.6g} T V^-1`

    The calibration uses the existing apparatus constants from `../ferromagnetism/data/data.xlsx`; update `Rx`, `R`, `C`, `L`, or `A` there if the Curie run used different values.
    """)
    return B_PER_Y, H_PER_X, X_NEG, X_POS, Y_NEG, Y_POS, data


@app.cell
def _(
    B_PER_Y,
    H_PER_X,
    MU0,
    X_NEG,
    X_POS,
    Y_NEG,
    Y_POS,
    data,
    np,
    savgol_filter,
):
    TEMPERATURE_C = data["Temperature (C)"].to_numpy(float)
    TEMPERATURE_K = TEMPERATURE_C + 273.15
    TIME_S = data["Time (sec)"].to_numpy(float)

    def branches_for_row(row):
        H_pos = H_PER_X * row[X_POS].to_numpy(float)
        H_neg = H_PER_X * row[X_NEG].to_numpy(float)
        B_pos = B_PER_Y * row[Y_POS].to_numpy(float)
        B_neg = B_PER_Y * row[Y_NEG].to_numpy(float)
        M_pos = B_pos / MU0 - H_pos
        M_neg = B_neg / MU0 - H_neg
        return H_pos, M_pos, H_neg, M_neg

    def interpolate_sorted(x, y, x0):
        idx = np.argsort(x)
        return np.interp(x0, x[idx], y[idx])

    def normalize_01(y):
        y = np.asarray(y, dtype=float)
        shifted = y - float(np.min(y))
        scale = float(np.max(shifted))
        if scale == 0.0:
            return np.zeros_like(y)
        return shifted / scale

    def smooth(y, max_window=21):
        y = np.asarray(y, dtype=float)
        window = min(max_window, len(y) if len(y) % 2 == 1 else len(y) - 1)
        if window < 5:
            return y
        return savgol_filter(y, window_length=window, polyorder=2)

    def transition_diagnostics(temperature_k, signal_norm):
        y = smooth(signal_norm)
        dydt = np.gradient(y, temperature_k)
        steepest_i = int(np.argmin(dydt))
        half_crossing_k = np.nan
        for i in range(len(temperature_k) - 1):
            y0, y1 = y[i], y[i + 1]
            if (y0 - 0.5) * (y1 - 0.5) <= 0 and y0 != y1:
                half_crossing_k = temperature_k[i] + (0.5 - y0) * (temperature_k[i + 1] - temperature_k[i]) / (y1 - y0)
                break
        return {
            "half_height_K": half_crossing_k,
            "steepest_slope_K": float(temperature_k[steepest_i]),
            "steepest_slope_value": float(dydt[steepest_i]),
        }

    def _tail_order(H_arr, tail):
        if tail == "pos":
            return np.argsort(-H_arr)
        if tail == "neg":
            return np.argsort(H_arr)
        raise ValueError(f"tail must be 'pos' or 'neg', got {tail!r}")

    def _linear_intercept(h, m):
        coeffs, cov = np.polyfit(h, m, 1, cov=True)
        return float(coeffs[1]), float(np.sqrt(cov[1, 1]))

    def sat_intercept(H, M, tail, n_min=5, n_max_frac=0.5, intercept_tol_sigma=2.0):
        # On the saturated tail, M(H) ~= M0 + chi_bg * H. Grow the window
        # from n_min outward (ranked by tail-side |H|) and freeze it once
        # the intercept shifts more than intercept_tol_sigma * sigma_prev
        # — i.e. the linear regime has ended.
        H_arr = np.asarray(H, dtype=float)
        M_arr = np.asarray(M, dtype=float)
        order = _tail_order(H_arr, tail)
        H_sorted, M_sorted = H_arr[order], M_arr[order]
        n_total = len(H_sorted)
        n_max = min(n_total, max(n_min + 1, int(n_max_frac * n_total)))

        intercept, sigma = _linear_intercept(H_sorted[:n_min], M_sorted[:n_min])
        n_used = n_min
        for n in range(n_min + 1, n_max + 1):
            try:
                intercept_new, sigma_new = _linear_intercept(H_sorted[:n], M_sorted[:n])
            except (np.linalg.LinAlgError, ValueError):
                break
            if abs(intercept_new - intercept) > intercept_tol_sigma * max(sigma, 1e-30):
                break
            intercept, sigma, n_used = intercept_new, sigma_new, n
        return intercept, sigma, n_used

    def sat_intercept_fixed(H, M, tail, n=5):
        # Instructor's strict n-point fit, kept alongside the adaptive
        # estimator as a transparent baseline.
        H_arr = np.asarray(H, dtype=float)
        M_arr = np.asarray(M, dtype=float)
        order = _tail_order(H_arr, tail)
        return _linear_intercept(H_arr[order][:n], M_arr[order][:n])

    return (
        TEMPERATURE_C,
        TEMPERATURE_K,
        TIME_S,
        branches_for_row,
        interpolate_sorted,
        normalize_01,
        sat_intercept,
        sat_intercept_fixed,
        smooth,
        transition_diagnostics,
    )


@app.cell
def _(TEMPERATURE_K, branches_for_row, data, np):
    # Above the transition the remaining response is treated as the linear
    # field-induced/background part and subtracted before extracting M(T).
    high_temperature_mask = TEMPERATURE_K >= 273.15
    if high_temperature_mask.sum() < 5:
        high_temperature_mask = TEMPERATURE_K >= np.quantile(TEMPERATURE_K, 0.9)

    _H_background, _M_background = [], []
    for _, _row in data.loc[high_temperature_mask].iterrows():
        _H_pos, _M_pos, _H_neg, _M_neg = branches_for_row(_row)
        _H_background.extend([_H_pos, _H_neg])
        _M_background.extend([_M_pos, _M_neg])

    _H_background = np.concatenate(_H_background)
    _M_background = np.concatenate(_M_background)
    background_slope, background_intercept = np.polyfit(_H_background, _M_background, 1)

    def remove_background(H, M):
        return M - (background_slope * H + background_intercept)

    return (
        background_intercept,
        background_slope,
        high_temperature_mask,
        remove_background,
    )


@app.cell(hide_code=True)
def _(background_intercept, background_slope, high_temperature_mask, mo):
    mo.md(rf"""
    **Background removal**

    The guide notes that above $T_c$ the sample still has field-induced magnetization. For this sketch, the high-temperature rows (`T >= 273.15 K`, `{int(high_temperature_mask.sum())}` loops) are fit to

    $$
    M_\mathrm{{bg}}(H)=aH+b.
    $$

    Fit used here: `a = {background_slope:.6g}`, `b = {background_intercept:.6g} A/m`.

    The extracted $M(T)$ proxies below use $M_\mathrm{{corr}} = M - M_\mathrm{{bg}}$ before normalization.
    """)
    return


@app.cell
def _(
    FIELD_READY_FRACTION,
    TEMPERATURE_C,
    TEMPERATURE_K,
    TIME_S,
    branches_for_row,
    data,
    interpolate_sorted,
    normalize_01,
    np,
    pd,
    remove_background,
    sat_intercept,
    sat_intercept_fixed,
    transition_diagnostics,
):
    branch_hmax = []
    for _, _row in data.iterrows():
        _H_pos, _, _H_neg, _ = branches_for_row(_row)
        branch_hmax.append(min(float(_H_pos.max()), float(-_H_neg.min())))
    branch_hmax = np.asarray(branch_hmax)

    stable_hmax = float(np.median(branch_hmax[len(branch_hmax) // 2:]))
    ready_threshold = FIELD_READY_FRACTION * stable_hmax
    _ready_indices = np.flatnonzero(branch_hmax >= ready_threshold)
    field_ready_start_index = int(_ready_indices[0])
    field_ready_temperature_K = float(TEMPERATURE_K[field_ready_start_index])

    valid_indices = np.arange(field_ready_start_index, len(data))
    common_hmax = float(branch_hmax[valid_indices].min())
    H_STAR = 0.50 * common_hmax

    records = []
    for _i in valid_indices:
        _row = data.iloc[_i]
        _H_pos, _M_pos, _H_neg, _M_neg = branches_for_row(_row)
        _M_pos_corr = remove_background(_H_pos, _M_pos)
        _M_neg_corr = remove_background(_H_neg, _M_neg)

        _M_pos_0 = interpolate_sorted(_H_pos, _M_pos_corr, 0.0)
        _M_neg_0 = interpolate_sorted(_H_neg, _M_neg_corr, 0.0)
        _M_pos_star = interpolate_sorted(_H_pos, _M_pos_corr, H_STAR)
        _M_neg_star = interpolate_sorted(_H_neg, _M_neg_corr, -H_STAR)

        # Method 3: extrapolate each saturated tail to H=0. Each branch
        # contributes two estimates (one per signed tail); the four signed
        # intercepts are flipped to all-positive and averaged to give M0.
        # Negative-tail intercepts pick up a sign flip because the linear
        # extrapolation crosses H=0 from the opposite side.
        _tails = (
            (_H_pos, _M_pos, "pos", +1.0),
            (_H_pos, _M_pos, "neg", -1.0),
            (_H_neg, _M_neg, "pos", +1.0),
            (_H_neg, _M_neg, "neg", -1.0),
        )
        _intercepts, _sigmas, _ns, _intercepts_5pt = [], [], [], []
        for _H_branch, _M_branch, _tail, _sign in _tails:
            _b, _s, _n = sat_intercept(_H_branch, _M_branch, tail=_tail)
            _bf, _ = sat_intercept_fixed(_H_branch, _M_branch, tail=_tail)
            _intercepts.append(_sign * _b)
            _sigmas.append(_s)
            _ns.append(_n)
            _intercepts_5pt.append(_sign * _bf)

        _M0 = float(np.mean(_intercepts))
        _sigma_M0 = float(np.sqrt(np.sum(np.square(_sigmas))) / len(_sigmas))
        _M0_5pt = float(np.mean(_intercepts_5pt))

        records.append({
            "time_s": TIME_S[_i],
            "temperature_C": TEMPERATURE_C[_i],
            "temperature_K": TEMPERATURE_K[_i],
            "branch_hmax_A_per_m": branch_hmax[_i],
            "remanence_A_per_m": 0.5 * abs(_M_pos_0 - _M_neg_0),
            "fixed_field_A_per_m": 0.5 * abs(_M_pos_star - _M_neg_star),
            "M0_A_per_m": _M0,
            "sigma_M0_A_per_m": _sigma_M0,
            "M0_5pt_A_per_m": _M0_5pt,
            "M0_n_used_avg": float(np.mean(_ns)),
        })

    summary = pd.DataFrame.from_records(records)
    summary["remanence_norm"] = normalize_01(summary["remanence_A_per_m"])
    summary["fixed_field_norm"] = normalize_01(summary["fixed_field_A_per_m"])
    summary["M0_norm"] = normalize_01(np.abs(summary["M0_A_per_m"]))

    _T_used = summary["temperature_K"].to_numpy()
    diagnostics = pd.DataFrame([
        {"method": "1. remanence", **transition_diagnostics(_T_used, summary["remanence_norm"].to_numpy())},
        {"method": f"2. fixed field, H*={H_STAR:.2f} A/m", **transition_diagnostics(_T_used, summary["fixed_field_norm"].to_numpy())},
        {"method": "3. M_0 saturation extrapolation", **transition_diagnostics(_T_used, summary["M0_norm"].to_numpy())},
    ])
    return H_STAR, common_hmax, diagnostics, field_ready_temperature_K, summary


@app.cell(hide_code=True)
def _(
    H_STAR,
    common_hmax,
    diagnostics,
    field_ready_temperature_K,
    mo,
    summary,
):
    def table_md(df, columns):
        rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for _, _row in df.iterrows():
            rows.append("| " + " | ".join(str(_row[c]) for c in columns) + " |")
        return "\n".join(rows)

    diag = diagnostics.copy()
    diag["half_height_K"] = diag["half_height_K"].map(lambda x: f"{x:.2f}")
    diag["steepest_slope_K"] = diag["steepest_slope_K"].map(lambda x: f"{x:.2f}")
    diag["steepest_slope_value"] = diag["steepest_slope_value"].map(lambda x: f"{x:.4g}")

    mo.md(rf"""
    ## Extracted methods

    **Method 1: remanence**

    $$
    M_r(T)=\frac12\left|M_+(T,0)-M_-(T,0)\right|.
    $$

    **Method 2: fixed field**

    The LabVIEW table stores positive and negative half-loops separately, so this sketch uses a fixed magnitude $|H_*|$ and compares $+H_*$ on the positive branch with $-H_*$ on the negative branch:

    $$
    M_\mathrm{{sep}}(T;H_*)=\frac12\left|M_+(T,+H_*)-M_-(T,-H_*)\right|.
    $$

    Rows before `T = {field_ready_temperature_K:.3f} K` are excluded because the measured branch field amplitude had not yet reached the stable drive-field plateau. This removes the early under-magnetized garbage data before normalization.

    Here $H_* = {H_STAR:.2f}\,\mathrm{{A\,m^{{-1}}}}$, chosen as half of the common branch-amplitude range $H_\max={common_hmax:.2f}\,\mathrm{{A\,m^{{-1}}}}$ so every retained loop can be interpolated at the same field.

    **Method 3: high-field saturation extrapolation**

    On the saturated tails of each branch the loop is locally linear,
    $M(H)\approx M_0(T)+\chi_\mathrm{{bg}}(T)\,H$. A linear fit of $M$ vs
    $H$ on those tails gives a per-temperature intercept $M_0(T)$ — the
    spontaneous magnetization, distinct from the remanence (memory at
    $H=0$) and from any loop-area diagnostic.

    The fit window is grown adaptively from $n_\mathrm{{min}}=5$ points
    (the lab guide's prescription) outward; each candidate fit is
    accepted only if its intercept lies within $2\sigma$ of the previous
    fit's intercept, otherwise the window is frozen there. Both branches
    and both signed tails contribute four estimates per loop; their
    sign-flipped values are averaged to give $M_0$, with uncertainty
    propagated in quadrature. The instructor's strict 5-point fit is
    retained alongside as `M0_5pt_A_per_m` for cross-checking.

    Methods 1 and 2 are normalized to $[0,1]$ for half-height and
    steepest-slope diagnostics:

    $$
    y_\mathrm{{norm}}=\frac{{y-\min(y)}}{{\max\!\left(y-\min(y)\right)}}.
    $$

    Method 3 is reported in absolute units $\mathrm{{A\,m^{{-1}}}}$ and
    additionally normalized for the comparison plot. The proper $T_c$
    estimate uses the absolute $M_0(T)$ in a weighted $M_0^2(T)$ fit
    below.

    **Quick-look transition diagnostics** (half-height crossing and
    steepest-slope temperature on the smoothed normalized curve; no
    uncertainties — the proper $T_c$ comes from the $M_0^2(T)$ fit
    below):

    {table_md(diag, ["method", "half_height_K", "steepest_slope_K", "steepest_slope_value"])}

    Normalized ranges: remanence `{summary['remanence_norm'].min():.3f}`--`{summary['remanence_norm'].max():.3f}`, fixed-field `{summary['fixed_field_norm'].min():.3f}`--`{summary['fixed_field_norm'].max():.3f}`, $M_0$ `{summary['M0_norm'].min():.3f}`--`{summary['M0_norm'].max():.3f}`. Mean adaptive window size: `{summary['M0_n_used_avg'].mean():.1f}` points.
    """)
    return


@app.cell
def _(np, summary):
    # Mean-field square-root scaling near Tc:  M0(T)^2 ~= m * (Tc - T)
    # Weighted linear fit on rows where M0 is well above its 1-sigma noise.
    T_all = summary["temperature_K"].to_numpy()
    M0_all = summary["M0_A_per_m"].to_numpy()
    sM0_all = summary["sigma_M0_A_per_m"].to_numpy()
    M0_sq_all = M0_all ** 2
    sM0_sq_all = 2.0 * np.abs(M0_all) * sM0_all

    # Require both (a) a positive intercept (still in the ordered phase) and
    # (b) the intercept resolved well above its noise floor. Negative M0
    # values above Tc are unphysical artifacts of fitting noise to a tail
    # that is no longer linear at finite spontaneous M, and including them
    # flattens the M0^2(T) line toward zero slope.
    snr = np.abs(M0_all) / np.maximum(sM0_all, 1e-30)
    fit_mask = (M0_all > 0) & (snr > 5.0)
    if fit_mask.sum() < 5:
        # Fallback: coldest half of the retained rows.
        order_T = np.argsort(T_all)
        keep = order_T[: max(5, len(T_all) // 2)]
        fit_mask = np.zeros_like(T_all, dtype=bool)
        fit_mask[keep] = True

    T_fit = T_all[fit_mask]
    Msq_fit = M0_sq_all[fit_mask]
    sMsq_fit = sM0_sq_all[fit_mask]
    weights = 1.0 / np.maximum(sMsq_fit, 1e-30)

    coeffs_msq, cov_msq = np.polyfit(T_fit, Msq_fit, 1, w=weights, cov=True)
    msq_slope = float(coeffs_msq[0])
    msq_intercept = float(coeffs_msq[1])
    msq_sigma_slope = float(np.sqrt(cov_msq[0, 0]))
    msq_sigma_intercept = float(np.sqrt(cov_msq[1, 1]))
    msq_cov_si = float(cov_msq[0, 1])

    Tc_K = -msq_intercept / msq_slope
    var_Tc = (
        (msq_sigma_intercept / msq_slope) ** 2
        + (msq_intercept * msq_sigma_slope / msq_slope ** 2) ** 2
        - 2.0 * msq_intercept * msq_cov_si / msq_slope ** 3
    )
    sigma_Tc_K = float(np.sqrt(max(0.0, var_Tc)))
    Tc_C = Tc_K - 273.15
    return (
        M0_sq_all,
        Tc_C,
        Tc_K,
        T_all,
        fit_mask,
        msq_intercept,
        msq_slope,
        sM0_sq_all,
        sigma_Tc_K,
    )


@app.cell(hide_code=True)
def _(Tc_C, Tc_K, fit_mask, mo, sigma_Tc_K, summary):
    mo.md(rf"""
    ## Method 3 result: $T_c$ from $M_0^2(T)$

    Weighted linear fit of $M_0^2$ vs $T$ over `{int(fit_mask.sum())}`
    points where $M_0/\sigma_{{M_0}}>5$, giving

    $$
    T_c \;=\; {Tc_K:.2f}\;\pm\;{sigma_Tc_K:.2f}\;\mathrm{{K}}
    \;=\; {Tc_C:.2f}\;\pm\;{sigma_Tc_K:.2f}\;^\circ\mathrm{{C}}.
    $$

    Adaptive vs 5-point baseline (mean over retained loops):
    $\langle M_0^\text{{adapt}}\rangle = {summary['M0_A_per_m'].mean():.3g}$,
    $\langle M_0^\text{{5pt}}\rangle = {summary['M0_5pt_A_per_m'].mean():.3g}\;\mathrm{{A\,m^{{-1}}}}$.
    """)
    return


@app.cell
def _(
    M0_sq_all,
    T_all,
    Tc_C,
    Tc_K,
    fit_mask,
    msq_intercept,
    msq_slope,
    np,
    plt,
    sM0_sq_all,
    save_figure,
    sigma_Tc_K,
):
    fig_msq, ax_msq = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)

    scale = 1e6  # (A/m)^2 -> (kA/m)^2

    excl = ~fit_mask
    if excl.any():
        ax_msq.errorbar(
            T_all[excl],
            M0_sq_all[excl] / scale,
            yerr=sM0_sq_all[excl] / scale,
            fmt="o",
            color="0.65",
            markersize=2.6,
            elinewidth=0.6,
            alpha=0.55,
            label="excluded (low SNR)",
        )
    ax_msq.errorbar(
        T_all[fit_mask],
        M0_sq_all[fit_mask] / scale,
        yerr=sM0_sq_all[fit_mask] / scale,
        fmt="o",
        color="C0",
        markersize=3.4,
        elinewidth=0.7,
        alpha=0.85,
        label="fit window ($M_0/\\sigma_{M_0}>5$)",
    )

    T_line = np.linspace(float(T_all[fit_mask].min()), Tc_K + 5.0, 200)
    ax_msq.plot(
        T_line,
        (msq_slope * T_line + msq_intercept) / scale,
        "-",
        color="C3",
        linewidth=2.0,
        label=rf"linear fit, $T_c={Tc_K:.1f}\pm{sigma_Tc_K:.1f}\,\mathrm{{K}}$",
    )
    ax_msq.axhline(0, color="0.4", linewidth=0.6, linestyle="--")
    ax_msq.axvline(Tc_K, color="C3", linewidth=0.8, linestyle=":")
    ymin, ymax = ax_msq.get_ylim()
    ax_msq.fill_betweenx(
        [ymin, ymax],
        Tc_K - sigma_Tc_K,
        Tc_K + sigma_Tc_K,
        color="C3",
        alpha=0.10,
    )
    ax_msq.set_ylim(ymin, ymax)

    ax_msq.set_xlabel(r"$T$ (K)")
    ax_msq.set_ylabel(r"$M_0^2$ (kA$^2\,$m$^{-2}$)")
    ax_msq.set_title(rf"Method 3: $M_0^2(T)$ extrapolation, $T_c\approx{Tc_C:.1f}\,^\circ$C")
    ax_msq.minorticks_on()
    ax_msq.grid(True, which="major", alpha=0.25)
    ax_msq.grid(True, which="minor", alpha=0.10)
    ax_msq.legend(loc="upper right")

    save_figure(fig_msq, "curie_method3_M0sq")
    fig_msq
    return


@app.cell
def _(
    Normalize,
    TARGET_LOOP_TEMPERATURES_K,
    TEMPERATURE_K,
    branches_for_row,
    data,
    field_ready_temperature_K,
    np,
    plt,
    remove_background,
    save_figure,
):
    valid_loop_indices = np.flatnonzero(TEMPERATURE_K >= field_ready_temperature_K)
    selected_indices = []
    for target in TARGET_LOOP_TEMPERATURES_K:
        _nearest = valid_loop_indices[np.argmin(np.abs(TEMPERATURE_K[valid_loop_indices] - target))]
        selected_indices.append(int(_nearest))
    selected_indices = sorted(set(selected_indices), key=lambda i: TEMPERATURE_K[i])

    loop_temperatures = TEMPERATURE_K[selected_indices]
    cmap = plt.colormaps["viridis"]
    norm = Normalize(vmin=float(loop_temperatures.min()), vmax=float(loop_temperatures.max()))

    fig_loops, ax_loops = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    max_abs_h = 0.0
    for _i in selected_indices:
        _row = data.iloc[_i]
        _H_pos, _M_pos, _H_neg, _M_neg = branches_for_row(_row)
        _M_pos_corr = remove_background(_H_pos, _M_pos) / 1e3
        _M_neg_corr = remove_background(_H_neg, _M_neg) / 1e3
        _color = cmap(norm(TEMPERATURE_K[_i]))

        ax_loops.plot(_H_pos, _M_pos_corr, color=_color, linewidth=1.8)
        ax_loops.plot(_H_neg, _M_neg_corr, color=_color, linewidth=1.8)
        _loop_hmax = float(np.max(np.abs(np.concatenate([_H_pos, _H_neg]))))
        max_abs_h = max(max_abs_h, _loop_hmax)

    ax_loops.axhline(0, color="0.25", linewidth=0.8)
    ax_loops.axvline(0, color="0.25", linewidth=0.8)
    ax_loops.set_xlim(-1.05 * max_abs_h, 1.05 * max_abs_h)
    ax_loops.set_xlabel(r"$H$ (A m$^{-1}$)")
    ax_loops.set_ylabel(r"$M_\mathrm{corr}$ (kA m$^{-1}$)")
    ax_loops.set_title("Representative hysteresis half-loops")
    ax_loops.minorticks_on()
    ax_loops.grid(True, which="major", alpha=0.25)
    ax_loops.grid(True, which="minor", alpha=0.10)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    colorbar = fig_loops.colorbar(sm, ax=ax_loops, pad=0.02)
    colorbar.set_label(r"$T$ (K)")

    save_figure(fig_loops, "curie_selected_loops")
    fig_loops
    return


@app.cell
def _(Line2D, diagnostics, np, plt, save_figure, smooth, summary):
    fig_methods, ax_methods = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)

    temperature = summary["temperature_K"].to_numpy()
    series = [
        ("remanence_norm", "Remanence", "C0", "o"),
        ("fixed_field_norm", "Fixed field", "C3", "s"),
        ("M0_norm", r"$M_0$ extrap.", "C2", "^"),
    ]

    legend_handles = []
    for _column, _label, _color, _marker in series:
        _values = summary[_column].to_numpy()
        ax_methods.plot(
            temperature,
            _values,
            linestyle="None",
            marker=_marker,
            markersize=3.0,
            markeredgewidth=0.0,
            color=_color,
            alpha=0.28,
        )
        ax_methods.plot(temperature, smooth(_values), color=_color, linewidth=2.1)
        legend_handles.append(Line2D([0], [0], color=_color, linewidth=2.1, label=_label))

    for (_, _row), (_, _, _color, _) in zip(diagnostics.iterrows(), series):
        ax_methods.axvline(_row["half_height_K"], color=_color, linestyle="--", linewidth=1.0, alpha=0.65)

    steepest_temperatures = np.unique(np.round(diagnostics["steepest_slope_K"].to_numpy(float)[: len(series)], 2))
    for _i, _steepest_temperature in enumerate(steepest_temperatures):
        ax_methods.axvline(
            _steepest_temperature,
            color="0.15",
            linestyle=":",
            linewidth=1.2,
            alpha=0.75,
            label="steepest slope" if _i == 0 else None,
        )

    ax_methods.axhline(0.5, color="0.45", linestyle="-", linewidth=0.8, alpha=0.35)
    ax_methods.set_xlabel(r"$T$ (K)")
    ax_methods.set_ylabel("normalized signal")
    ax_methods.set_ylim(-0.03, 1.03)
    ax_methods.set_title("Normalized Curie-transition proxies")
    ax_methods.minorticks_on()
    ax_methods.grid(True, which="major", alpha=0.25)
    ax_methods.grid(True, which="minor", alpha=0.10)

    legend_handles.extend([
        Line2D([0], [0], color="0.45", linestyle="--", linewidth=1.0, label="half-height $T_c$"),
        Line2D([0], [0], color="0.15", linestyle=":", linewidth=1.2, label="steepest-slope $T_c$"),
    ])
    ax_methods.legend(handles=legend_handles, loc="upper right")

    save_figure(fig_methods, "curie_method123_normalized")
    fig_methods
    return


@app.cell
def _(FIG_DIR, summary):
    summary_path = FIG_DIR / "curie_method123_summary.csv"
    summary.to_csv(summary_path, index=False)
    summary_path
    return


if __name__ == "__main__":
    app.run()
