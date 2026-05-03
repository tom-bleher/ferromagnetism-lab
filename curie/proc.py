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
    - implement methods 1 and 2 from slide 8 of the Curie presentation, plus a loop-area proxy;
    - normalize each extracted order-parameter proxy to the range $0$--$1$;
    - leave uncertainty propagation for a later pass.

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

    return (
        TEMPERATURE_C,
        TEMPERATURE_K,
        TIME_S,
        branches_for_row,
        interpolate_sorted,
        normalize_01,
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
    H_grid = np.linspace(0.0, common_hmax, 400)
    for _i in valid_indices:
        _row = data.iloc[_i]
        _H_pos, _M_pos, _H_neg, _M_neg = branches_for_row(_row)
        _M_pos_corr = remove_background(_H_pos, _M_pos)
        _M_neg_corr = remove_background(_H_neg, _M_neg)

        _M_pos_0 = interpolate_sorted(_H_pos, _M_pos_corr, 0.0)
        _M_neg_0 = interpolate_sorted(_H_neg, _M_neg_corr, 0.0)
        _M_pos_star = interpolate_sorted(_H_pos, _M_pos_corr, H_STAR)
        _M_neg_star = interpolate_sorted(_H_neg, _M_neg_corr, -H_STAR)
        _M_pos_grid = interpolate_sorted(_H_pos, _M_pos_corr, H_grid)
        _M_neg_grid = interpolate_sorted(_H_neg, _M_neg_corr, -H_grid)

        records.append({
            "time_s": TIME_S[_i],
            "temperature_C": TEMPERATURE_C[_i],
            "temperature_K": TEMPERATURE_K[_i],
            "branch_hmax_A_per_m": branch_hmax[_i],
            "remanence_A_per_m": 0.5 * abs(_M_pos_0 - _M_neg_0),
            "fixed_field_A_per_m": 0.5 * abs(_M_pos_star - _M_neg_star),
            "loop_area_proxy": float(np.trapezoid(np.abs(_M_pos_grid - _M_neg_grid), H_grid)),
        })

    summary = pd.DataFrame.from_records(records)
    summary["remanence_norm"] = normalize_01(summary["remanence_A_per_m"])
    summary["fixed_field_norm"] = normalize_01(summary["fixed_field_A_per_m"])
    summary["loop_area_norm"] = normalize_01(summary["loop_area_proxy"])

    _T_used = summary["temperature_K"].to_numpy()
    diagnostics = pd.DataFrame([
        {"method": "1. remanence", **transition_diagnostics(_T_used, summary["remanence_norm"].to_numpy())},
        {"method": f"2. fixed field, H*={H_STAR:.2f} A/m", **transition_diagnostics(_T_used, summary["fixed_field_norm"].to_numpy())},
        {"method": "3. loop-area proxy", **transition_diagnostics(_T_used, summary["loop_area_norm"].to_numpy())},
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

    **Method 3: loop-area proxy**

    The data file stores positive and negative half-branches. As a practical area diagnostic, this notebook integrates the branch separation over the common field-amplitude range:

    $$
    A_\mathrm{{loop}}^*(T)=\int_0^{{H_\max}}\left|M_+(T,+H)-M_-(T,-H)\right|\,dH.
    $$

    This is a hysteresis/order proxy rather than a literal closed-contour area, but it uses the full branch shape instead of a single interpolation point.

    Each method is then normalized by first offsetting the retained data to make the minimum zero, and then dividing by the shifted maximum:

    $$
    y_\mathrm{{norm}}=\frac{{y-\min(y)}}{{\max\!\left(y-\min(y)\right)}}.
    $$

    No clipping is applied.

    **Rough transition diagnostics, no uncertainties yet**

    {table_md(diag, ["method", "half_height_K", "steepest_slope_K", "steepest_slope_value"])}

    Normalized ranges: remanence `{summary['remanence_norm'].min():.3f}`--`{summary['remanence_norm'].max():.3f}`, fixed-field `{summary['fixed_field_norm'].min():.3f}`--`{summary['fixed_field_norm'].max():.3f}`, loop-area `{summary['loop_area_norm'].min():.3f}`--`{summary['loop_area_norm'].max():.3f}`.
    """)
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
    save_figure(fig_methods, "curie_method12_normalized")
    fig_methods
    return


@app.cell
def _(FIG_DIR, summary):
    summary_path = FIG_DIR / "curie_method12_summary.csv"
    summary.to_csv(summary_path, index=False)
    summary.to_csv(FIG_DIR / "curie_method123_summary.csv", index=False)
    summary_path
    return


if __name__ == "__main__":
    app.run()
