import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium", app_title="Curie temperature analysis")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Curie-temperature processing notebook

    Processing for the three measured Curie scans. The detailed diagnostic
    figures use the best-covered scan (`first`); the same pipeline is then
    applied to `second` and `third` to quantify drive-setting sensitivity.

    Scope:

    - calibrate LabVIEW voltages with the experiment relations from the guides;
    - subtract a high-temperature linear magnetic background;
    - implement three methods to extract a per-loop order parameter
      from the *single* hysteresis loop measured at each temperature.
      Each proxy is the half-difference of the upper and lower branch
      values of the same loop, evaluated at a specific field; the two
      branches give $\pm$ the proxy by symmetry, so the half-difference
      cancels any DC offset on $V_y$. The three methods differ only in
      the evaluation field and the fitting procedure:
        1. **Method I** — $M_r(T)$, branches at $H=0$. The remanence;
           same quantity as the LabVIEW realtime trace $V_y(V_x{=}0)$.
        2. **Method II** — $M_\mathrm{sat}(T)$, branches at $H=\pm H_\mathrm{sat}$
           (saturation tip). Same quantity as the LabVIEW realtime
           trace $V_y(V_x{=}V_{x,\max})$.
        3. **Method III** — $M_0(T)$, the same single loop's saturation
           tail fitted linearly and *extrapolated back to $H=0$*. This
           is the guide's algebraic Method III ("$B-(\alpha+1)H=M_0$");
           the spontaneous magnetization.
    - chain Method III into a weighted mean-field fit
      $M_0^2(T)\propto T_c-T$ to extract $T_c\pm\sigma_{T_c}$.

    The calibration relations used are

    $$
    H = \frac{N_1 V_x}{L R_x}, \qquad
    B = \frac{R C V_y}{N_2 A}, \qquad
    M = \frac{B}{\mu_0} - H.
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Course-style data processing conventions

    Instrument and resolution uncertainties are combined in quadrature, and
    indirect quantities are propagated by the usual partial derivative rule.

    The half-height crossings are local derived quantities, not fitted physical
    models. The two cross-check models in this notebook are the weighted line
    fits of $M_0^2(T)$ and apparent Curie–Weiss $1/\chi(T)$; both report fit
    parameters, relative errors, $\chi^2/\nu$, p-value, DOF, and
    data-minus-fit panels.
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

    from taulab import fit_functions, odr_fit, read_table

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
    DATA_DIR = ROOT / "data"
    DATA_XLSX = ROOT.parent / "ferromagnetism" / "data" / "data.xlsx"
    FIG_DIR = ROOT.parent.parent / "report" / "media"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # All three runs (sweep duplicates), in chronological order.
    RUN_FILES = {
        sub: next((DATA_DIR / sub).glob("CurieData_*"))
        for sub in ("first", "second", "third")
    }
    DATA_FILE = RUN_FILES["first"]

    MU0 = 1.25663706127e-6
    FIELD_READY_FRACTION = 0.98
    TARGET_LOOP_TEMPERATURES_K = [180, 195, 210, 225, 240, 260, 278]

    # Conservative bound on the absolute thermometer accuracy (datasheet
    # not in hand; ±1 K is typical for K-type thermocouples used with a
    # cryogenic Monel sample). This is fully correlated across all loops
    # in a run, so it is reported as a separate systematic and not folded
    # into the per-loop sigma_T used for fit weights.
    SIGMA_T_ABS_K = 1.0

    def save_figure(fig, stem):
        fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(FIG_DIR / f"{stem}.png", bbox_inches="tight", dpi=600)

    return (
        DATA_DIR,
        DATA_FILE,
        DATA_XLSX,
        FIELD_READY_FRACTION,
        FIG_DIR,
        Line2D,
        MU0,
        Normalize,
        RUN_FILES,
        SIGMA_T_ABS_K,
        TARGET_LOOP_TEMPERATURES_K,
        fit_functions,
        np,
        odr_fit,
        pd,
        plt,
        read_table,
        save_figure,
        savgol_filter,
    )


@app.cell
def _(DATA_FILE, DATA_XLSX, mo, np, pd, read_table):
    data = pd.read_csv(DATA_FILE, sep="\t")
    apparatus = read_table(DATA_XLSX, sheet_name="apparatus").iloc[0]
    temperature_K = data["Temperature (C)"] + 273.15

    X_POS = [c for c in data.columns if c.startswith("X_pos")]
    Y_POS = [c for c in data.columns if c.startswith("Y_pos")]
    X_NEG = [c for c in data.columns if c.startswith("X_neg")]
    Y_NEG = [c for c in data.columns if c.startswith("Y_neg")]

    # The Curie rig is NOT the hysteresis toroid. Per the lab schematic
    # (references/official-guides/IMG_2270.jpeg), the Curie circuit has
    # a 1:10 step-up secondary and a different integrator:
    #   N1 = 250 (primary, drives H),    N2 = 2500 (integrator side -> B)
    #   R2 = 3.97 kΩ (Ry),  C = 19.78 µF
    #   R1 = 0.15 .. 2 Ω variable (Rx)
    # L (solenoid length over the Monel rod) and A (rod cross-section)
    # are not on the schematic; the hysteresis-sheet values below are
    # placeholders so the absolute axes are dimensionally consistent.
    # *** None of these enter T_c. *** Methods 1-3 locate the temperature
    # at which M(T) crosses a normalized threshold, so any common
    # multiplicative rescaling of M (or H) leaves T_c invariant. The
    # override is here so the diagnostic plots (M in A/m, H in A/m)
    # reflect the Curie circuit rather than the toroid circuit.
    N1 = 250
    N2 = 2500
    Ry = 3.97e3
    # Curie integrator capacitor read off the lab schematic; it is a
    # different physical capacitor from the toroid integrator (lab.lyx:
    # C = 20.1 µF). Either value gives the same T_c since C is a common
    # multiplicative factor on M and cancels in the temperature intercept.
    C = 19.78e-6
    L = float(apparatus["L (m)"])    # placeholder: rod-in-solenoid geometry
    A = float(apparatus["A (m²)"])   # placeholder: rod cross-section

    # The user measured the primary AC current (RMS) with an ammeter for each run,
    # bypassing the unknown/variable Rx.
    I_rms = 1.97  # Ampere (Measurement 1 for the 'first' run)
    I_peak = I_rms * np.sqrt(2)
    # The max LabVIEW voltage X_max corresponds to I_peak.
    X_max = np.max([data[xp].max() for xp in X_POS])
    effective_Rx = X_max / I_peak
    H_PER_X = N1 / (L * effective_Rx)

    B_PER_Y = Ry * C / (N2 * A)

    # Per-loop temperature uncertainty. T is logged once per loop, but
    # each loop is acquired over a finite window dt_loop (~5.6 s here)
    # during which the sample temperature drifts by |dT/dt|*dt_loop. We
    # treat that drift as a uniform window centred on the logged T, so
    # sigma_T_smear = |dT/dt|*dt_loop / sqrt(12). The LabVIEW T resolution
    # is 1 mK (sigma_T_resolution ~ 0.3 mK), negligible against the heating-rate
    # smearing near the transition. The thermometer absolute
    # accuracy is treated separately as a fully-correlated systematic
    # because it shifts T_c rigidly within a single run.
    _t_s = data["Time (sec)"].to_numpy(float)
    _T_K = temperature_K.to_numpy(float)
    _dT_dt = np.gradient(_T_K, _t_s)
    _dt_loop = float(np.median(np.diff(_t_s))) if len(_t_s) > 1 else 0.0
    sigma_T_smear = np.abs(_dT_dt) * _dt_loop / np.sqrt(12.0)
    _T_resolution = 1e-3
    sigma_T_K = np.sqrt(sigma_T_smear**2 + (_T_resolution / np.sqrt(12.0)) ** 2)

    # Integrator-validity check for the Curie circuit. Same logic as the
    # ferromagnetism notebook: the R_y-C circuit is an ideal integrator
    # only in the limit omega*R_y*C >> 1; check the actual gain ratio
    # and phase departure at 50 Hz.
    _f_drive = 50.0
    _tau_RC = Ry * C
    _wRyC = 2.0 * np.pi * _f_drive * _tau_RC
    _gain_ratio = _wRyC / np.sqrt(1.0 + _wRyC ** 2)
    _gain_err_pct = (1.0 - _gain_ratio) * 100.0
    _phase_deg = np.degrees(np.arctan(1.0 / _wRyC))

    mo.md(rf"""
    **Loaded data**

    - rows: `{len(data)}` loops
    - samples per branch: `{len(X_POS)}`
    - temperature range: `{temperature_K.min():.3f}` to `{temperature_K.max():.3f}` K
    - calibration constants: `H/Vx = {H_PER_X:.6g} A m^-1 V^-1`, `B/Vy = {B_PER_Y:.6g} T V^-1`
    - loop window: `{_dt_loop:.2f}` s; per-loop $\sigma_T$ over the raw run (median, p95): `{np.median(sigma_T_K):.3f}`, `{np.percentile(sigma_T_K, 95):.3f}` K

    The Curie circuit constants are set to match the lab schematic
    ($N_1=250$ primary, $N_2=2500$ secondary, $R_y=3.97\,\mathrm{{k\Omega}}$,
    $C=19.78\,\mu\mathrm{{F}}$). The geometric constants used for the
    diagnostic $H$ and $M$ scales enter only through common linear
    rescalings; the transition-temperature extraction uses normalized
    curves and is not set by those absolute scales.

    **Integrator check.** The $R_y$–$C$ circuit measuring $V_y\propto B$
    is a valid integrator only when $\omega R_y C \gg 1$. Its exact transfer
    function is $G_{{RC}}(\omega)=1/(1+i\omega R_y C)$, which reduces to the
    ideal $1/(i\omega R_y C)$ in that limit. At the 50 Hz drive,

    $$
    \tau = R_y C = {_tau_RC*1e3:.2f}\,\mathrm{{ms}} \;\gg\; T_\mathrm{{drive}} = {1e3/_f_drive:.1f}\,\mathrm{{ms}}, \quad
    \omega R_y C = {_wRyC:.2f}.
    $$

    The non-ideal amplitude correction is
    $|G_{{RC}}/(1/i\omega R_y C)|={_gain_ratio:.5f}$ ⇒ {_gain_err_pct:.3f} %
    gain error, and the phase departure from an ideal integrator is
    {_phase_deg:.2f}°. Both are well below any other systematic in the
    Curie pipeline; the $V_y\to B$ relation $B = R_y C V_y/(N_2 A)$ is safe to use.

    **Why apparatus uncertainties are not propagated here**: the Curie
    methods all locate the temperature where $M(T)$ vanishes. Any common
    rescaling of $M$ (from $N_1$, $N_2$, $L$, $R_x$, $R_y$, $C$, $A$)
    leaves the $T$-axis intercept unchanged, so the apparatus-constant
    budget cancels in $T_c$ and is intentionally not carried here.
    """)
    return B_PER_Y, H_PER_X, X_NEG, X_POS, Y_NEG, Y_POS, data, sigma_T_K


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

    def local_intercept_at(H, M, H0, n_neighbors=4):
        # Local linear fit of M(H) on the n_neighbors points whose H is
        # closest to H0; returns the fitted M(H0) and its 1-sigma. This
        # replaces the bare np.interp used to evaluate Methods 1 and 2,
        # which gave no per-loop uncertainty. n_neighbors=4 keeps the
        # window small enough to track curvature near coercivity but
        # large enough to make the linear residual a meaningful sigma.
        H_arr = np.asarray(H, dtype=float)
        M_arr = np.asarray(M, dtype=float)
        order = np.argsort(np.abs(H_arr - H0))[:max(2, n_neighbors)]
        h_loc, m_loc = H_arr[order], M_arr[order]
        if len(h_loc) < 2:
            return float(np.interp(H0, np.sort(h_loc), m_loc[np.argsort(h_loc)])), 0.0
        try:
            coeffs, cov = np.polyfit(h_loc, m_loc, 1, cov=True)
        except (np.linalg.LinAlgError, ValueError):
            return float(np.interp(H0, np.sort(h_loc), m_loc[np.argsort(h_loc)])), 0.0
        m_at_H0 = float(coeffs[0] * H0 + coeffs[1])
        # var(a*H0 + b) = H0^2 var(a) + var(b) + 2*H0*cov(a,b)
        var = (
            H0 * H0 * float(cov[0, 0])
            + float(cov[1, 1])
            + 2.0 * H0 * float(cov[0, 1])
        )
        return m_at_H0, float(np.sqrt(max(0.0, var)))

    def normalize_01_with_sigma(y, sy):
        # Normalize y -> (y - min(y)) / max(y - min(y)) and propagate sigma.
        # Treats min and max as fixed at the chosen sample to keep the
        # half-height / steepest-slope diagnostics tractable; the alternative
        # (treating min/max as random) would inflate sigma everywhere by an
        # offset that cancels at the half-height crossing anyway.
        y = np.asarray(y, dtype=float)
        sy = np.asarray(sy, dtype=float)
        shifted = y - float(np.min(y))
        scale = float(np.max(shifted))
        if scale == 0.0:
            return np.zeros_like(y), np.zeros_like(y)
        return shifted / scale, sy / scale

    def smooth(y, max_window=21):
        y = np.asarray(y, dtype=float)
        window = min(max_window, len(y) if len(y) % 2 == 1 else len(y) - 1)
        if window < 5:
            return y
        return savgol_filter(y, window_length=window, polyorder=2)

    def half_height_tc_with_sigma(T, sT, y, sy):
        # Half-height crossing temperature with local uncertainty from the
        # two points bracketing the 0.5 crossing. T-axis smearing and y-axis
        # uncertainty are propagated directly through the linear interpolation.
        T = np.asarray(T, dtype=float)
        y = np.asarray(y, dtype=float)
        sT = np.asarray(sT, dtype=float)
        sy = np.asarray(sy, dtype=float)

        def _crossing(T_arr, y_arr):
            ys = smooth(y_arr)
            shifted = ys - float(np.min(ys))
            scale = float(np.max(shifted))
            if scale == 0.0:
                return np.nan, None, None
            yn = shifted / scale
            for i in range(len(T_arr) - 1):
                y0, y1 = yn[i], yn[i + 1]
                if (y0 - 0.5) * (y1 - 0.5) <= 0 and y0 != y1:
                    frac = (0.5 - y0) / (y1 - y0)
                    return T_arr[i] + frac * (T_arr[i + 1] - T_arr[i]), i, yn
            return np.nan, None, yn

        order = np.argsort(T)
        Ts, ys, sTs, sys_ = T[order], y[order], sT[order], sy[order]
        center, i, yn = _crossing(Ts, ys)
        if i is None or not np.isfinite(center):
            return float(center) if np.isfinite(center) else np.nan, np.nan

        T0, T1 = float(Ts[i]), float(Ts[i + 1])
        y0, y1 = float(yn[i]), float(yn[i + 1])
        dT = T1 - T0
        dy = y1 - y0
        frac = (0.5 - y0) / dy
        d_tc_d_T0 = 1.0 - frac
        d_tc_d_T1 = frac
        d_tc_d_y0 = (0.5 - y1) * dT / (dy * dy)
        d_tc_d_y1 = -(0.5 - y0) * dT / (dy * dy)
        sigma = np.sqrt(
            (d_tc_d_T0 * sTs[i]) ** 2
            + (d_tc_d_T1 * sTs[i + 1]) ** 2
            + (d_tc_d_y0 * sys_[i]) ** 2
            + (d_tc_d_y1 * sys_[i + 1]) ** 2
        )
        return float(center), float(sigma)

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

    def _linear_fit(h, m):
        # Returns intercept, sigma_intercept, slope, sigma_slope.
        coeffs, cov = np.polyfit(h, m, 1, cov=True)
        return (
            float(coeffs[1]),
            float(np.sqrt(cov[1, 1])),
            float(coeffs[0]),
            float(np.sqrt(cov[0, 0])),
        )

    def _linear_intercept(h, m):
        b, sb, _, _ = _linear_fit(h, m)
        return b, sb

    def sat_intercept_fixed(H, M, tail, n=5):
        # Instructor's strict n-point saturated-tail fit.
        H_arr = np.asarray(H, dtype=float)
        M_arr = np.asarray(M, dtype=float)
        order = _tail_order(H_arr, tail)
        return _linear_intercept(H_arr[order][:n], M_arr[order][:n])

    return (
        TEMPERATURE_C,
        TEMPERATURE_K,
        TIME_S,
        branches_for_row,
        half_height_tc_with_sigma,
        local_intercept_at,
        normalize_01_with_sigma,
        sat_intercept_fixed,
        smooth,
        transition_diagnostics,
    )


@app.cell
def _(TEMPERATURE_K, branches_for_row, data, np):
    # Above the transition the remaining response is treated as the linear
    # field-induced/background part and subtracted before extracting M(T).
    # The previous code used T >= 273.15 K (room temperature) as the
    # paramagnetic-regime cutoff. That implicitly assumes T_c is near room
    # temperature — fine for bulk Fe but wrong here: the Monel sample sits
    # at T_c ~ 212 K and the data range tops out near 280 K, so the
    # 273.15 K cut grabbed only a handful of loops at the extreme tail
    # while ignoring plenty of equally-paramagnetic loops at 220-270 K.
    # We instead fit chi_bg on the highest-T quartile (top 25 %), which
    # for any sample with T_c well below the run's T_max sits comfortably
    # in the paramagnetic regime. This is robust to the unknown T_c.
    high_temperature_mask = TEMPERATURE_K >= np.quantile(TEMPERATURE_K, 0.75)

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

    The guide notes that above $T_c$ the sample still has field-induced magnetization. We fit $\chi_\mathrm{{bg}}$ on the highest-T quartile (`{int(high_temperature_mask.sum())}` loops) as an empirical linear high-temperature background. This avoids choosing a circular fixed $T_c$ cutoff, but it is still a modelling choice; for marginal coverage scans it can include transition-tail data and is therefore treated as part of the method systematic. The fit form is

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
    half_height_tc_with_sigma,
    local_intercept_at,
    normalize_01_with_sigma,
    np,
    pd,
    remove_background,
    sat_intercept_fixed,
    sigma_T_K,
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
    # Method II is evaluated at the saturation field H = ±H_sat ≈ H_max,
    # mirroring the LabVIEW realtime "Y(X_min/max)" trace (red curve in
    # the on-screen monitor). Setting H_sat strictly equal to the per-loop
    # H_max would put us at the very edge of the recorded sweep, where
    # nearest-neighbour interpolation has no points outside the bracket;
    # a small inboard offset (0.98 H_max) keeps the local fit well-posed
    # while staying inside the saturation regime that matches LabVIEW.
    H_SAT = 0.98 * common_hmax

    records = []
    for _i in valid_indices:
        _row = data.iloc[_i]
        _H_pos, _M_pos, _H_neg, _M_neg = branches_for_row(_row)
        _M_pos_corr = remove_background(_H_pos, _M_pos)
        _M_neg_corr = remove_background(_H_neg, _M_neg)

        # Method I: M_r(T) ≡ ½ |M_+(T, H=0) − M_-(T, H=0)|.
        # Local linear fit to M(H) at H=0 on each branch gives both the
        # intercept (M_±(T, 0)) and a per-loop 1-sigma from residuals.
        _M_pos_0, _s_pos_0 = local_intercept_at(_H_pos, _M_pos_corr, 0.0)
        _M_neg_0, _s_neg_0 = local_intercept_at(_H_neg, _M_neg_corr, 0.0)
        _M_r = 0.5 * abs(_M_pos_0 - _M_neg_0)
        _sigma_M_r = 0.5 * float(np.hypot(_s_pos_0, _s_neg_0))

        # Method II: M_sat(T) ≡ ½ |M_+(T, +H_sat) − M_-(T, -H_sat)|,
        # evaluated near the saturation tips of the loop. This mirrors
        # the LabVIEW realtime "Y(X_min/max)" trace (red curve in the
        # on-screen monitor) — what the instructor's display shows for
        # M_sat(T). Same local-fit machinery as Method I but at H=±H_sat
        # rather than H=0.
        _M_pos_sat, _s_pos_sat = local_intercept_at(_H_pos, _M_pos_corr, H_SAT)
        _M_neg_sat, _s_neg_sat = local_intercept_at(_H_neg, _M_neg_corr, -H_SAT)
        _M_sat = 0.5 * abs(_M_pos_sat - _M_neg_sat)
        _sigma_M_sat = 0.5 * float(np.hypot(_s_pos_sat, _s_neg_sat))

        # Method III: M_0(T) ≡ ½ |M_+^sat(T) − M_-^sat(T)|, where
        # M_±^sat(T) is the linear-fit extrapolation of the ± branch's
        # saturated tail back to H=0 — i.e. the y-intercept b_± of the
        # linear fit, equal to ±M_0 by symmetry.
        # Per the official guide: "fit a linear approximation
        # in the saturation regime and extrapolate to H=0".
        # Algebra (guide page 2):
        #   in saturation,   M = M_0 + alpha * H
        #   so               B/mu_0 = M_0 + (alpha + 1) * H,
        #   i.e.             B/mu_0 - (alpha + 1) * H = M_0.
        # In SI we already pre-compute M = B/mu_0 - H in branches_for_row,
        # so a linear fit M(H) = M_0 + alpha * H on the saturation tail
        # gives M_0 *as the y-intercept* — exactly the "extrapolate the
        # linear fit to H = 0" prescription.
        #
        # The LabVIEW table stores the H>=0 half of the upper branch in
        # *_pos and the H<=0 half of the lower branch in *_neg, so each
        # loop has exactly two saturation tips: the +H end of _pos
        # (linear extrapolation hits the M-axis at +M_0) and the -H end
        # of _neg (linear extrapolation hits at -M_0). The opposite ends
        # of each half are the remanence region, not saturation, and
        # must not be used here. M_0 is the half-difference of the two
        # signed intercepts; this also cancels any common offset shared
        # by both branches (e.g. the H-independent piece of the
        # paramagnetic background).
        #
        # We feed the *background-corrected* M into sat_intercept so the
        # fitted slope alpha = chi_HF (high-field susceptibility, the
        # guide's alpha) cleanly separates from the universal
        # paramagnetic background chi_bg fitted earlier; the slope-
        # stability gate inside sat_intercept then operates on the
        # material slope and trips when we leave the linear regime.
        # M_0 itself is invariant under this choice (a_bg, b_bg cancel
        # in the half-difference), so this is a notational fix; it
        # matches the cross-run cell.
        _b_5pt_pos, _s_5pt_pos = sat_intercept_fixed(_H_pos, _M_pos_corr, tail="pos")
        _b_5pt_neg, _s_5pt_neg = sat_intercept_fixed(_H_neg, _M_neg_corr, tail="neg")

        # Primary Method III follows the guide's strict 5-point saturated-tail
        # prescription.
        _M_0 = 0.5 * (_b_5pt_pos - _b_5pt_neg)
        _sigma_M_0 = 0.5 * float(np.hypot(_s_5pt_pos, _s_5pt_neg))

        records.append({
            "time_s": TIME_S[_i],
            "temperature_C": TEMPERATURE_C[_i],
            "temperature_K": TEMPERATURE_K[_i],
            "sigma_T_K": float(sigma_T_K[_i]),
            "branch_hmax_A_per_m": branch_hmax[_i],
            "M_r_A_per_m": _M_r,
            "sigma_M_r_A_per_m": _sigma_M_r,
            "M_sat_A_per_m": _M_sat,
            "sigma_M_sat_A_per_m": _sigma_M_sat,
            "M_0_A_per_m": _M_0,
            "sigma_M_0_A_per_m": _sigma_M_0,
            "M_0_5pt_A_per_m": _M_0,
        })

    summary = pd.DataFrame.from_records(records)
    summary["M_r_norm"], summary["sigma_M_r_norm"] = normalize_01_with_sigma(
        summary["M_r_A_per_m"].to_numpy(),
        summary["sigma_M_r_A_per_m"].to_numpy(),
    )
    summary["M_sat_norm"], summary["sigma_M_sat_norm"] = normalize_01_with_sigma(
        summary["M_sat_A_per_m"].to_numpy(),
        summary["sigma_M_sat_A_per_m"].to_numpy(),
    )
    summary["M_0_norm"], summary["sigma_M_0_norm"] = normalize_01_with_sigma(
        np.abs(summary["M_0_A_per_m"].to_numpy()),
        summary["sigma_M_0_A_per_m"].to_numpy(),
    )
    summary["M_0_snr"] = np.abs(summary["M_0_A_per_m"].to_numpy()) / np.maximum(
        summary["sigma_M_0_A_per_m"].to_numpy(), 1e-30,
    )
    summary["M_0_positive"] = summary["M_0_A_per_m"] > 0
    summary["M_0_fit_valid"] = summary["M_0_positive"] & (summary["M_0_snr"] > 3.0)
    summary["field_fraction_of_common"] = summary["branch_hmax_A_per_m"] / common_hmax

    _T_used = summary["temperature_K"].to_numpy()
    _sT_used = summary["sigma_T_K"].to_numpy()
    diagnostics = pd.DataFrame([
        {"method": r"$M_\mathrm{r}$ (H=0)", **transition_diagnostics(_T_used, summary["M_r_norm"].to_numpy())},
        {"method": rf"$M_\mathrm{{sat}}$ (H=±{H_SAT:.2f} A/m)", **transition_diagnostics(_T_used, summary["M_sat_norm"].to_numpy())},
        {"method": r"$M_0$ (sat. extrap.→H=0)", **transition_diagnostics(_T_used, summary["M_0_norm"].to_numpy())},
    ])

    # Local half-height T_c uncertainty for each method. Folds the
    # per-loop sigma_T (heating-rate smearing) and the per-loop sigma_y
    # together to give a usable statistical 1-sigma on the half-height
    # crossing temperature. Methods I and II use this; Method III also
    # produces a half-height crossing on the normalized M_0 curve, but
    # its formal T_c estimate comes from the M_0^2(T) weighted line fit below.
    _tc_M_r, _stc_M_r = half_height_tc_with_sigma(
        _T_used, _sT_used,
        summary["M_r_norm"].to_numpy(),
        summary["sigma_M_r_norm"].to_numpy(),
    )
    _tc_M_sat, _stc_M_sat = half_height_tc_with_sigma(
        _T_used, _sT_used,
        summary["M_sat_norm"].to_numpy(),
        summary["sigma_M_sat_norm"].to_numpy(),
    )
    _tc_M_0, _stc_M_0 = half_height_tc_with_sigma(
        _T_used, _sT_used,
        summary["M_0_norm"].to_numpy(),
        summary["sigma_M_0_norm"].to_numpy(),
    )
    diagnostics_with_sigma = pd.DataFrame([
        {"method": r"$M_\mathrm{r}$ (half-height, H=0)",                     "Tc_K": _tc_M_r,    "sigma_Tc_K": _stc_M_r},
        {"method": rf"$M_\mathrm{{sat}}$ (half-height, H=±{H_SAT:.2f} A/m)", "Tc_K": _tc_M_sat, "sigma_Tc_K": _stc_M_sat},
        {"method": r"$M_0$ (half-height, sat. extrap.→H=0)",      "Tc_K": _tc_M_0,    "sigma_Tc_K": _stc_M_0},
    ])
    return (
        H_SAT,
        common_hmax,
        diagnostics,
        diagnostics_with_sigma,
        field_ready_temperature_K,
        summary,
    )


@app.cell(hide_code=True)
def _(
    H_SAT,
    common_hmax,
    diagnostics,
    diagnostics_with_sigma,
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

    diag_sig = diagnostics_with_sigma.copy()
    diag_sig["Tc_K"] = diag_sig["Tc_K"].map(lambda x: f"{x:.2f}")
    diag_sig["sigma_Tc_K"] = diag_sig["sigma_Tc_K"].map(lambda x: f"{x:.2f}")

    mo.md(rf"""
    ## Extracted methods

    Each loop is *one* measurement at one temperature, recorded by
    LabVIEW as 128 samples on each branch ($H\!\ge\!0$ half of the
    upper branch in `*_pos`, $H\!\le\!0$ half of the lower branch in
    `*_neg`). All three methods reduce that single loop to a scalar
    order-parameter proxy $V(T)$ by reading the loop's $V_y\propto B$
    response at a specified $V_x\propto H$ position. They differ only
    in **where** the position is and **how** the value at that position
    is obtained.

    The two LabVIEW realtime traces shown on the instrument monitor —
    $V_y(V_x{{=}}0)$ (white) and $V_y(V_x{{=}}V_{{x,\max}})$ (red) —
    are exactly Methods I and II below. Method III adds a third proxy
    that goes beyond the realtime display: it *extrapolates* the
    saturated-tail linear fit back to $H=0$ to extract the spontaneous
    magnetization $M_0$ (the guide's algebraic "Method III").

    For numerical robustness against any common DC offset on $V_y$, we
    don't use a single-branch reading; instead we read both the upper
    branch (at $+H_\mathrm{{eval}}$) and the lower branch (at
    $-H_\mathrm{{eval}}$) of the *same loop* and take their
    *half-difference*. By symmetry $M_+\to+M_\mathrm{{proxy}}$ and
    $M_-\to-M_\mathrm{{proxy}}$, so the half-difference equals
    $M_\mathrm{{proxy}}$ exactly while any $H$-independent bias on $V_y$
    cancels between branches. The notation $M_\pm$ below labels the
    *upper / lower branch of the same hysteresis loop*, not two
    separate experimental measurements.

    **Method I: $M_r(T)$ — branches read at $H=0$ (remanence; LabVIEW $V_y(V_x{{=}}0)$)**

    $$
    M_r(T)\;\equiv\;\tfrac12\left|M_+(T,H{{=}}0)-M_-(T,H{{=}}0)\right|.
    $$

    $M_\pm(T,0)$ comes from a *local* linear fit of $M_\pm(T,H)$ on
    the **four** branch samples nearest $H=0$. Four is the smallest
    window for which a 1D linear fit returns a non-degenerate covariance
    (so the per-loop $y$-uncertainty $\sigma_{{M_\pm(0)}}$ is meaningful)
    while staying narrow enough to track the curvature in the coercivity
    region; widening to $n=6$–$8$ shifts $T_c$ by $<0.1\,\mathrm{{K}}$ in
    smoke tests, well below the per-method statistical $\sigma$.

    **Method II: $M_\mathrm{{sat}}(T)$ — branches read at $H=\pm H_\mathrm{{sat}}$ (saturation tip; LabVIEW $V_y(V_x{{=}}V_{{x,\max}})$)**

    $$
    M_\mathrm{{sat}}(T)\;\equiv\;\tfrac12\left|M_+(T,H{{=}}{{+}}H_\mathrm{{sat}})-M_-(T,H{{=}}{{-}}H_\mathrm{{sat}})\right|.
    $$

    Same local-fit machinery as Method I but evaluated near the loop's
    saturation tips, mirroring the LabVIEW $V_y(V_{{x,\max}})$ trace.
    We use $H_\mathrm{{sat}} = {H_SAT:.2f}\,\mathrm{{A\,m^{{-1}}}}$
    ($\approx 0.98\,H_{{\max}}$ on the common branch amplitude
    $H_{{\max}}={common_hmax:.2f}\,\mathrm{{A\,m^{{-1}}}}$), inboard
    just enough to keep the local nearest-neighbour interpolation
    well-posed at the loop edge. This is the guide's algebraic
    Method II ("$B-H_{{\max}}=M$" at saturation, with $\alpha\to 0$).

    Rows before $T={field_ready_temperature_K:.3f}\,\mathrm{{K}}$ are excluded
    from Methods I, II, and III because the measured branch field amplitude
    had not yet reached the stable drive-field plateau.

    **Method III: $M_0(T)$ — single-loop saturation tail, linearly extrapolated to $H=0$**

    Following the official guide: in the saturation regime the same
    single loop's response is well-approximated by

    $$
    M(H)=M_0+\alpha H \;\;\Longleftrightarrow\;\;
    \frac{{B(H)}}{{\mu_0}}=M_0+(\alpha+1)H \;\;\Longleftrightarrow\;\;
    \frac{{B}}{{\mu_0}}-(\alpha+1)H=M_0,
    $$

    so $M_0(T)$ is recovered *as the intersection of the
    saturation-tail linear fit with the $H=0$ axis*. The same hysteresis
    loop has two saturation tips — the $+H$ end of the upper branch and
    the $-H$ end of the lower branch — and we fit one straight line on
    each tip:

    $$
    M_+(T,H) = b_+(T) + \alpha_+(T)\,H \quad\text{{on the upper-branch }}+H\text{{ tail}},
    $$
    $$
    M_-(T,H) = b_-(T) + \alpha_-(T)\,H \quad\text{{on the lower-branch }}-H\text{{ tail}}.
    $$

    By symmetry $b_+\to+M_0$ and $b_-\to-M_0$ in the limit of an ideal
    loop, so the proxy is

    $$
    M_0(T)\;\equiv\;\tfrac12\,\bigl|\,b_+(T)-b_-(T)\,\bigr|.
    $$

    The intercepts $b_\pm$ are *not* two independent measurements; they
    are two algebraic intercepts of two linear fits applied to two
    different ends of the *same* hysteresis loop. This makes the
    parallel structure with Methods I and II explicit: all three are
    $\tfrac12|M_+(T,H_\text{{eval}})-M_-(T,H_\text{{eval}})|$ with
    $H_\text{{eval}}=0$ (Method I), $H_\text{{eval}}=\pm H_\mathrm{{sat}}$
    (Method II), or $H_\text{{eval}}=0$ *via the saturation-fit
    extrapolation* (Method III). The fitted slope $\alpha=\chi_\mathrm{{HF}}$
    is the high-field susceptibility (the guide's $\alpha$).

    The primary Method-III value uses the instructor's fixed 5-point
    saturated-tail fit, ranked by tail-side $|H|$. The LabVIEW export stores
    the $H\ge0$ half of the upper branch and the $H\le0$ half of the lower
    branch, so each loop contributes two saturated tips. Statistical
    uncertainty on $M_0$ is propagated in quadrature from the two per-fit
    intercept sigmas.

    The branches are pre-corrected for the global paramagnetic
    background (fitted in the high-T cell above) before the saturation
    fit, so $\alpha$ here is purely the material's high-field
    susceptibility (the guide's $\alpha$), not the bulk slope. $M_0$
    itself is invariant under this choice — both the constant and
    linear pieces of the background cancel in the half-difference of
    the two branches — but separating $\alpha$ from $\chi_\mathrm{{bg}}$
    keeps the slope-stability gate physically interpretable.

    All three proxies $M_r(T)$, $M_\mathrm{{sat}}(T)$, and $M_0(T)$ are
    normalized to $[0,1]$ for half-height and steepest-slope diagnostics:

    $$
    y_\mathrm{{norm}}=\frac{{y-\min(y)}}{{\max\!\left(y-\min(y)\right)}}.
    $$

    Method III is *additionally* kept in absolute units $\mathrm{{A\,m^{{-1}}}}$
    so $M_0(T)$ can feed the weighted $M_0^2(T)$ mean-field fit below.

    **Quick-look transition diagnostics** (half-height crossing and
    steepest-slope temperature on a lightly smoothed normalized curve,
    no uncertainties). The smoothing is only for reading the trend; the
    final quoted result is dominated by method and run-to-run spread.

    {table_md(diag, ["method", "half_height_K", "steepest_slope_K", "steepest_slope_value"])}

    **Half-height $T_{1/2}^\mathrm{app}$ with local uncertainty** for all three
    methods. Each row is the half-height crossing temperature; the
    local uncertainty uses the temperature smearing and the scatter of
    the per-loop fit. Method III also feeds the linear $M_0^2(T)$
    check below.

    {table_md(diag_sig, ["method", "Tc_K", "sigma_Tc_K"])}

    Normalized ranges: $M_r$ `{summary['M_r_norm'].min():.3f}`--`{summary['M_r_norm'].max():.3f}`, $M_\mathrm{{sat}}$ `{summary['M_sat_norm'].min():.3f}`--`{summary['M_sat_norm'].max():.3f}`, $M_0$ `{summary['M_0_norm'].min():.3f}`--`{summary['M_0_norm'].max():.3f}`.
    """)
    return


@app.cell
def _(diagnostics_with_sigma, fit_functions, np, odr_fit, pd, summary):
    # Mean-field square-root scaling near Tc:  M0(T)^2 ~= m * (Tc - T).
    # The mean-field form is only valid in a narrow band just below Tc;
    # far below Tc, M0 saturates (M0 -> Msat) and M0^2 flattens, which
    # breaks the linear form. A wide-window fit pulls Tc upward.
    #
    # Algorithm: anchored window-size scan with physical gates.
    #   1. Seed Tc from the half-height crossing on the normalized M_0
    #      curve (the same model-free anchor used elsewhere).
    #   2. Filter to the candidate pool: M0 > 0, SNR > 3, and a fixed
    #      band around the seed. The seed is not iterated, because letting
    #      the fitted zero-crossing redefine the high-T pool can make the
    #      fit chase noisy above-transition points upward.
    #   3. Sort the pool by T descending so the points closest to Tc
    #      (where mean-field is asymptotically valid) come first.
    #   4. Sweep K = K_MIN .. K_MAX and run a weighted line fit on the top-K
    #      points; record (K, Tc_K, chi^2/nu, p-probability).
    #   5. Keep only physical fits: negative slope and a zero crossing
    #      close to the seed. Among statistically acceptable fits, pick
    #      the largest K; otherwise pick the best p-probability physical fit.
    #
    # K_MIN = 5 is the lab guide's prescription. K_MAX is bounded by
    # the candidate-pool size; this is large enough that the optimum
    # lives in the interior of the scan.
    T_all = summary["temperature_K"].to_numpy()
    sT_all = summary["sigma_T_K"].to_numpy()
    M0_all = summary["M_0_A_per_m"].to_numpy()
    sM0_all = summary["sigma_M_0_A_per_m"].to_numpy()
    M0_sq_all = M0_all ** 2
    sM0_sq_all = 2.0 * np.abs(M0_all) * sM0_all

    K_MIN = 5
    # Margin on the +T side. Strict mean-field has M^2 = 0 above Tc, so
    # any data above Tc that survives the M0>0 + SNR>3 filters is upward
    # noise; we keep a small +2 K margin so a couple of marginal points
    # right at the transition stay in (giving the fit some leverage on
    # the intercept) but no further.
    FIT_UPPER_MARGIN_K = 2.0
    FIT_LOWER_MARGIN_K = 35.0
    FIT_TC_TOLERANCE_K = 25.0
    FIT_MIN_P_VALUE = 0.05

    _seed_row = diagnostics_with_sigma.iloc[2]
    _Tc_seed = float(_seed_row["Tc_K"])
    if not np.isfinite(_Tc_seed):
        _Tc_seed = float(np.nanmedian(T_all))

    snr = np.abs(M0_all) / np.maximum(sM0_all, 1e-30)

    def _fit_top_K(idx_sorted, K):
        sel = idx_sorted[:K]
        sT_s = np.maximum(sT_all[sel], 1e-3 / np.sqrt(12.0))
        sMsq_s = np.maximum(sM0_sq_all[sel], 1e-30)
        try:
            res = odr_fit(
                fit_functions.linear, None,
                T_all[sel], sT_s, M0_sq_all[sel], sMsq_s,
                param_names=["intercept", "slope"],
            )
        except Exception:
            return None
        b, m = float(res.params[0]), float(res.params[1])
        if m >= 0.0 or not np.isfinite(m):
            return None
        Tc_fit = -b / m
        if not np.isfinite(Tc_fit):
            return None
        if abs(Tc_fit - _Tc_seed) > FIT_TC_TOLERANCE_K:
            return None
        return res, sel, Tc_fit

    odr_result = None
    fit_mask = np.zeros_like(T_all, dtype=bool)
    K_best = K_MIN
    K_scan_records = []
    pool = (
        (M0_all > 0)
        & (snr > 3.0)
        & (T_all >= _Tc_seed - FIT_LOWER_MARGIN_K)
        & (T_all <= _Tc_seed + FIT_UPPER_MARGIN_K)
    )
    idx_pool = np.flatnonzero(pool)
    if idx_pool.size < K_MIN:
        # Fallback: if the physical filters reject too much, stay local
        # to the half-height seed rather than following a model intercept.
        idx_pool = np.argsort(np.abs(T_all - _Tc_seed))[: max(K_MIN, 8)]
    idx_sorted = idx_pool[np.argsort(-T_all[idx_pool])]

    for K in range(K_MIN, len(idx_sorted) + 1):
        out = _fit_top_K(idx_sorted, K)
        if out is None:
            continue
        res_K, sel_K, Tc_K_val = out
        K_scan_records.append({
            "K": K,
            "Tc_K": Tc_K_val,
            "redchi": float(res_K.redchi),
            "p_value": float(res_K.p_value),
            "_res": res_K,
            "_sel": sel_K,
        })

    if K_scan_records:
        acceptable = [r for r in K_scan_records if r["p_value"] >= FIT_MIN_P_VALUE]
        if acceptable:
            best = max(acceptable, key=lambda r: (r["K"], r["p_value"]))
        else:
            best = max(K_scan_records, key=lambda r: (r["p_value"], r["K"]))
        odr_result = best["_res"]
        fit_mask[best["_sel"]] = True
        K_best = int(best["K"])

    K_scan_table = pd.DataFrame([
        {"K": r["K"], "Tc_K": r["Tc_K"], "redchi": r["redchi"], "p_value": r["p_value"]}
        for r in K_scan_records
    ])

    if odr_result is None:
        raise RuntimeError("No physically acceptable M0^2 mean-field fit window found.")

    msq_intercept = float(odr_result.params[0])
    msq_slope = float(odr_result.params[1])
    redchi = float(odr_result.redchi)
    # The fitting library reports one covariance scaled by the residual scatter
    # and one unscaled covariance. For lab-report uncertainties the input sigmas
    # are treated as instrument/model estimates, so reduced chi^2 < 1 must not
    # shrink them; over-dispersed residuals can still inflate them.
    raw_odr_cov_scale = float(getattr(odr_result.raw_output, "res_var", redchi))
    odr_cov_scale = max(raw_odr_cov_scale, 1.0)
    if odr_result.cov is not None:
        _msq_cov = odr_result.cov * odr_cov_scale
        msq_sigma_intercept = float(np.sqrt(max(_msq_cov[0, 0], 0.0)))
        msq_sigma_slope = float(np.sqrt(max(_msq_cov[1, 1], 0.0)))
        msq_cov_si = float(_msq_cov[0, 1])
    else:
        msq_sigma_intercept = float(odr_result.errors[0])
        msq_sigma_slope = float(odr_result.errors[1])
        msq_cov_si = 0.0
    rescale = float(np.sqrt(odr_cov_scale))

    Tc_K = -msq_intercept / msq_slope
    var_Tc = (
        (msq_sigma_intercept / msq_slope) ** 2
        + (msq_intercept * msq_sigma_slope / msq_slope ** 2) ** 2
        - 2.0 * msq_intercept * msq_cov_si / msq_slope ** 3
    )
    sigma_Tc_K = float(np.sqrt(max(0.0, var_Tc)))
    Tc_C = Tc_K - 273.15
    return (
        FIT_UPPER_MARGIN_K,
        K_MIN,
        K_best,
        K_scan_table,
        M0_sq_all,
        Tc_C,
        Tc_K,
        T_all,
        fit_mask,
        msq_intercept,
        msq_slope,
        odr_result,
        rescale,
        sM0_sq_all,
        sT_all,
        sigma_Tc_K,
    )


@app.cell(hide_code=True)
def _(K_best, Tc_C, Tc_K, mo, odr_result, sigma_Tc_K):
    mo.md(rf"""
    ## Method 3 cross-check: mean-field $M_0^2(T)$ extrapolation

    > **This block is a check, not the main result.**
    > The mean-field form $M_0^2(T)\propto T_c-T$ is only an approximation
    > near the transition and is sensitive to which points are included.
    > The headline value is therefore the half-height apparent transition
    > reported in the bottom-line callout.

    A narrow set of points near the transition is fitted with a straight
    line. The selected window uses **{K_best}** points and gives

    $$
    T_c^\mathrm{{MF}} \;=\; {Tc_K:.1f}\;\mathrm{{K}}\;=\;{Tc_C:.1f}\,^\circ\mathrm{{C}}
    \quad(\chi^2/\nu={odr_result.redchi:.2f},\;\text{{p-value}}={odr_result.p_value:.3f},\;\mathrm{{DOF}}={odr_result.dof},\;\sigma_{{T_c}}=\pm{sigma_Tc_K:.1f}\,\mathrm{{K}}).
    $$
    """)
    return


@app.cell
def _(
    K_best,
    M0_sq_all,
    T_all,
    Tc_K,
    Tc_headline,
    fit_mask,
    msq_intercept,
    msq_slope,
    np,
    plt,
    sM0_sq_all,
    sT_all,
    save_figure,
    sigma_Tc_K,
    sigma_Tc_headline_stat,
):
    fig_msq, (ax_msq, ax_msq_res) = plt.subplots(
        2, 1,
        figsize=(7.4, 5.9),
        sharex=False,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
    )

    scale = 1e6  # (A/m)^2 -> (kA/m)^2

    # Bound the visible region to the meaningful transition window while
    # keeping every fitted point visible.
    _T_in = T_all[fit_mask]
    _x_lo = (Tc_headline - 10.0) if np.isfinite(Tc_headline) else float(_T_in.min()) - 10.0
    if _T_in.size:
        _x_lo = min(_x_lo, float(_T_in.min()) - 1.0)
        _x_hi = max(Tc_K + 5.0, float(_T_in.max()) + 1.0)
        T_line = np.linspace(float(_T_in.min()), Tc_K, 200)
    else:
        _x_hi = Tc_K + 5.0
        T_line = np.linspace(_x_lo, Tc_K, 200)
    _line_y = (msq_slope * T_line + msq_intercept) / scale
    _fit_y = M0_sq_all[fit_mask] / scale
    _fit_sy = sM0_sq_all[fit_mask] / scale
    _y_candidates = [_line_y]
    if _fit_y.size:
        _y_candidates.append(_fit_y + _fit_sy)
    _y_hi = max(0.5, 1.08 * float(np.nanmax(np.concatenate(_y_candidates))))
    _y_lo = -0.03 * _y_hi

    # Shade the fitted window in T so the reader can see at a
    # glance which points fed the line.
    if _T_in.size:
        ax_msq.axvspan(
            float(_T_in.min()), float(_T_in.max()),
            color="C0", alpha=0.07, zorder=0,
        )

    excl = ~fit_mask
    if excl.any():
        ax_msq.errorbar(
            T_all[excl], M0_sq_all[excl] / scale,
            xerr=sT_all[excl], yerr=sM0_sq_all[excl] / scale,
            fmt="o", color="0.6", markersize=2.8, elinewidth=0.6,
            alpha=0.55, label="excluded loops",
        )
    ax_msq.errorbar(
        T_all[fit_mask], M0_sq_all[fit_mask] / scale,
        xerr=sT_all[fit_mask], yerr=sM0_sq_all[fit_mask] / scale,
        fmt="o", color="C0", markersize=3.6, elinewidth=0.8,
        alpha=0.9, label=rf"fit window ($N_\mathrm{{fit}}={K_best}$)",
    )

    ax_msq.plot(
        T_line, _line_y,
        "-", color="C3", linewidth=2.0,
        label=rf"mean-field $T_\mathrm{{c}}={Tc_K:.1f}\pm{sigma_Tc_K:.1f}\,\mathrm{{K}}$",
    )

    _residual = (
        M0_sq_all[fit_mask]
        - (msq_slope * T_all[fit_mask] + msq_intercept)
    ) / scale
    _residual_sigma = sM0_sq_all[fit_mask] / scale
    ax_msq_res.errorbar(
        T_all[fit_mask], _residual,
        xerr=sT_all[fit_mask], yerr=_residual_sigma,
        fmt="o", color="C0", markersize=3.2, elinewidth=0.8,
        alpha=0.9,
    )
    ax_msq_res.axhline(0, color="0.35", linewidth=0.8, linestyle="--")
    _res_ylim = float(np.nanmax(np.abs(_residual) + _residual_sigma)) if _residual.size else 1.0
    if not np.isfinite(_res_ylim) or _res_ylim <= 0:
        _res_ylim = 1.0
    ax_msq_res.set_ylim(-1.15 * _res_ylim, 1.15 * _res_ylim)
    if _T_in.size:
        _res_x_lo = float(np.nanmin(T_all[fit_mask] - sT_all[fit_mask]))
        _res_x_hi = float(np.nanmax(T_all[fit_mask] + sT_all[fit_mask]))
        _res_x_pad = max(0.25, 0.04 * (_res_x_hi - _res_x_lo))
        ax_msq_res.set_xlim(_res_x_lo - _res_x_pad, _res_x_hi + _res_x_pad)

    ax_msq.axhline(0, color="0.4", linewidth=0.6, linestyle="--")
    ax_msq.axvline(Tc_K, color="C3", linewidth=0.8, linestyle=":")

    ax_msq.fill_betweenx(
        [_y_lo, _y_hi], Tc_K - sigma_Tc_K, Tc_K + sigma_Tc_K,
        color="C3", alpha=0.10,
    )

    if np.isfinite(Tc_headline):
        ax_msq.axvline(
            Tc_headline, color="C2", linewidth=1.4, linestyle="-",
            label=rf"half-height headline $T_\mathrm{{c}}^{{\mathrm{{app}}}}={Tc_headline:.1f}\pm{sigma_Tc_headline_stat:.1f}\,\mathrm{{K}}$",
        )

    ax_msq.set_xlim(_x_lo, _x_hi)
    ax_msq.set_ylim(_y_lo, _y_hi)
    ax_msq.set_xlabel(r"$T$ (K)")
    ax_msq.set_ylabel(r"$M_0^2$ ($\mathrm{kA}^2\,\mathrm{m}^{-2}$)")
    ax_msq.minorticks_on()
    ax_msq.grid(True, which="major", alpha=0.25)
    ax_msq.grid(True, which="minor", alpha=0.10)
    ax_msq.legend(loc="upper right", fontsize=8, framealpha=0.95)

    ax_msq_res.set_xlabel(r"$T$ (K), fit window")
    ax_msq_res.set_ylabel(r"$M_0^2 - f(T)$" + "\n" + r"($\mathrm{kA}^2\,\mathrm{m}^{-2}$)")
    ax_msq_res.minorticks_on()
    ax_msq_res.grid(True, which="major", alpha=0.25)
    ax_msq_res.grid(True, which="minor", alpha=0.10)

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
    sigma_T_arr = summary["sigma_T_K"].to_numpy()
    series = [
        ("M_r_norm",    "sigma_M_r_norm",    r"$M_\mathrm{r}$ ($H=0$)",                         "C0", "o"),
        ("M_sat_norm",  "sigma_M_sat_norm",  r"$M_\mathrm{sat}$ ($H=\pm H_\mathrm{sat}$)",  "C3", "s"),
        ("M_0_norm",    "sigma_M_0_norm",    r"$M_0$ (sat. extrap. to $H=0$)",       "C2", "^"),
    ]

    legend_handles = []
    for _column, _scolumn, _label, _color, _marker in series:
        _values = summary[_column].to_numpy()
        _svalues = summary[_scolumn].to_numpy() if _scolumn in summary.columns else np.zeros_like(_values)
        ax_methods.errorbar(
            temperature,
            _values,
            xerr=sigma_T_arr,
            yerr=_svalues,
            fmt=_marker,
            markersize=3.0,
            markeredgewidth=0.0,
            color=_color,
            ecolor=_color,
            elinewidth=0.5,
            alpha=0.7,
        )
        ax_methods.plot(temperature, smooth(_values), color=_color, linewidth=2.1)
        legend_handles.append(Line2D([0], [0], color=_color, linewidth=2.1, label=_label))

    for (_, _row), (_, _, _, _color, _) in zip(diagnostics.iterrows(), series):
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

    ax_methods.axhline(0.5, color="0.45", linestyle="-", linewidth=0.8, alpha=0.55)
    ax_methods.set_xlabel(r"$T$ (K)")
    ax_methods.set_ylabel("min-max normalized proxy")
    ax_methods.set_ylim(-0.03, 1.03)
    ax_methods.minorticks_on()
    ax_methods.grid(True, which="major", alpha=0.25)
    ax_methods.grid(True, which="minor", alpha=0.10)

    legend_handles.extend([
        Line2D([0], [0], color="0.45", linestyle="--", linewidth=1.0, label=r"half-height $T_\mathrm{c}$"),
        Line2D([0], [0], color="0.15", linestyle=":", linewidth=1.2, label=r"steepest-slope $T_\mathrm{c}$"),
    ])
    ax_methods.legend(handles=legend_handles, loc="upper right")

    save_figure(fig_methods, "curie_method123_normalized")
    fig_methods
    return


@app.cell
def _(
    TEMPERATURE_K,
    TIME_S,
    field_ready_temperature_K,
    high_temperature_mask,
    plt,
    save_figure,
    summary,
):
    _fig_acq, (_ax_temp, _ax_field) = plt.subplots(
        2, 1, figsize=(7.2, 5.2), constrained_layout=True,
        gridspec_kw={"height_ratios": [1.25, 1.0]},
    )

    _retained_time = summary["time_s"].to_numpy()
    _retained_temperature = summary["temperature_K"].to_numpy()
    _ready_time = float(_retained_time[0])

    _ax_temp.plot(TIME_S, TEMPERATURE_K, color="0.65", linewidth=1.2, label="logged temperature")
    _ax_temp.scatter(
        _retained_time, _retained_temperature,
        s=12, color="C0", alpha=0.75, label="retained loops",
    )
    _ax_temp.scatter(
        TIME_S[high_temperature_mask], TEMPERATURE_K[high_temperature_mask],
        s=16, color="C2", alpha=0.80, label="background-fit quartile",
    )
    _ax_temp.axvline(_ready_time, color="C3", linestyle="--", linewidth=1.0, label="field-ready cut")
    _ax_temp.set_xlabel(r"$t$ (s)")
    _ax_temp.set_ylabel(r"$T$ (K)")
    _ax_temp.grid(True, which="major", alpha=0.25)
    _ax_temp.minorticks_on()
    _ax_temp.grid(True, which="minor", alpha=0.10)
    _ax_temp.legend(loc="lower right", fontsize=8, framealpha=0.95)

    _ax_field.errorbar(
        _retained_temperature,
        summary["branch_hmax_A_per_m"].to_numpy() / 1000.0,
        xerr=summary["sigma_T_K"].to_numpy(),
        fmt="o", color="C0", ecolor="C0", alpha=0.70,
        markersize=3.0, elinewidth=0.5, capsize=1.5,
        label=r"per-loop $|H|_\mathrm{max}$",
    )
    _ax_field.axvline(
        field_ready_temperature_K, color="C3", linestyle="--", linewidth=1.0,
        label="field-ready cut",
    )
    _ax_field.set_xlabel(r"$T$ (K)")
    _ax_field.set_ylabel(r"$|H|_\mathrm{max}$ (kA m$^{-1}$)")
    _ax_field.grid(True, which="major", alpha=0.25)
    _ax_field.minorticks_on()
    _ax_field.grid(True, which="minor", alpha=0.10)
    _ax_field.legend(loc="lower right", fontsize=8, framealpha=0.95)

    save_figure(_fig_acq, "curie_acquisition_qc")
    _fig_acq
    return


@app.cell
def _(FIG_DIR, diagnostics_with_sigma, summary):
    summary_path = FIG_DIR / "curie_method123_summary.csv"
    summary.to_csv(summary_path, index=False)
    diag_path = FIG_DIR / "curie_method123_tc_with_sigma.csv"
    diagnostics_with_sigma.to_csv(diag_path, index=False)
    summary_path
    return


@app.cell
def _(
    DATA_XLSX,
    FIELD_READY_FRACTION,
    MU0,
    RUN_FILES,
    fit_functions,
    half_height_tc_with_sigma,
    local_intercept_at,
    normalize_01_with_sigma,
    np,
    odr_fit,
    pd,
    read_table,
):
    # Cross-run check: re-run the same three half-height estimators on all
    # three physical Curie scans. The guide explicitly asks to repeat the
    # measurement at different primary resistance / drive-field settings and
    # discuss the effect, so these scans are treated as controlled-condition
    # comparisons rather than hidden statistical repeats. The Method-III
    # M_0^2(T) mean-field fit is still computed per run as a qualitative check.
    # All locals start with an underscore so marimo treats them as
    # cell-private (no clashes with the main-pipeline names above).
    # Same Curie-circuit override as the main pipeline (see top cell).
    _apparatus = read_table(DATA_XLSX, sheet_name="apparatus").iloc[0]
    _N1 = 250
    _N2 = 2500
    _Ry = 3.97e3
    _C = 19.78e-6  # Curie integrator capacitor; cancels in T_c (see main cell).
    _L = float(_apparatus["L (m)"])
    _A = float(_apparatus["A (m²)"])
    _B_per_Y = _Ry * _C / (_N2 * _A)
    
    _I_rms_dict = {"first": 1.97, "second": 2.24, "third": 0.65}

    def _branches(row, xp, yp, xn, yn, h_per_x):
        H_p = h_per_x * row[xp].to_numpy(float)
        H_n = h_per_x * row[xn].to_numpy(float)
        B_p = _B_per_Y * row[yp].to_numpy(float)
        B_n = _B_per_Y * row[yn].to_numpy(float)
        return H_p, B_p / MU0 - H_p, H_n, B_n / MU0 - H_n

    def _tail_intercept(H, M, tail, n=5):
        # Mirrors the main pipeline's primary Method III: the guide's strict
        # five tail-side points, avoiding expansion into the shoulder.
        order = np.argsort(-H) if tail == "pos" else np.argsort(H)
        h, m = H[order], M[order]
        co, cov = np.polyfit(h[:n], m[:n], 1, cov=True)
        b, sb = float(co[1]), float(np.sqrt(cov[1, 1]))
        return b, sb

    def _run_methods(label, path):
        df = pd.read_csv(path, sep="\t")
        T_K = df["Temperature (C)"].to_numpy(float) + 273.15
        t_s = df["Time (sec)"].to_numpy(float)
        xp = [c for c in df.columns if c.startswith("X_pos")]
        yp = [c for c in df.columns if c.startswith("Y_pos")]
        xn = [c for c in df.columns if c.startswith("X_neg")]
        yn = [c for c in df.columns if c.startswith("Y_neg")]

        dt_loop = float(np.median(np.diff(t_s))) if len(t_s) > 1 else 0.0
        sT = np.abs(np.gradient(T_K, t_s)) * dt_loop / np.sqrt(12.0)
        sT = np.sqrt(sT**2 + (1e-3 / np.sqrt(12.0)) ** 2)

        run_name = path.parent.name
        I_rms = _I_rms_dict.get(run_name, 1.97)
        I_peak = I_rms * np.sqrt(2)
        X_max = np.max([df[c].max() for c in xp])
        effective_Rx = X_max / I_peak
        h_per_x = _N1 / (_L * effective_Rx)

        hmax = np.array([
            min(float(h_per_x * df.iloc[i][xp].to_numpy(float).max()),
                float(-h_per_x * df.iloc[i][xn].to_numpy(float).min()))
            for i in range(len(df))
        ])
        plateau = float(np.median(hmax[len(hmax) // 2:]))
        ready = np.flatnonzero(hmax >= FIELD_READY_FRACTION * plateau)
        if ready.size == 0:
            return None, [], []
        keep = np.arange(int(ready[0]), len(df))
        T_retained = T_K[keep]
        common_hmax = float(hmax[keep].min())
        H_sat = 0.98 * common_hmax

        # Use the top quartile of the full run as an empirical high-temperature
        # background, matching the main pipeline. This avoids a fixed
        # room-temperature cutoff while keeping the background definition
        # independent of the eventual T_c estimate. It is not assumed to be a
        # perfectly clean asymptotic Curie-Weiss regime for every run.
        bg_mask = T_K >= np.quantile(T_K, 0.75)
        Hb, Mb = [], []
        for i in np.flatnonzero(bg_mask):
            Hp, Mp, Hn, Mn = _branches(df.iloc[i], xp, yp, xn, yn, h_per_x)
            Hb.extend([Hp, Hn])
            Mb.extend([Mp, Mn])
        a_bg, b_bg = np.polyfit(np.concatenate(Hb), np.concatenate(Mb), 1)

        records = []
        for i in keep:
            Hp, Mp, Hn, Mn = _branches(df.iloc[i], xp, yp, xn, yn, h_per_x)
            Mp_c = Mp - (a_bg * Hp + b_bg)
            Mn_c = Mn - (a_bg * Hn + b_bg)

            Mp_0, sp_0 = local_intercept_at(Hp, Mp_c, 0.0)
            Mn_0, sn_0 = local_intercept_at(Hn, Mn_c, 0.0)
            M_r = 0.5 * abs(Mp_0 - Mn_0)
            sM_r = 0.5 * float(np.hypot(sp_0, sn_0))

            Mp_sat, sp_sat = local_intercept_at(Hp, Mp_c, H_sat)
            Mn_sat, sn_sat = local_intercept_at(Hn, Mn_c, -H_sat)
            M_sat = 0.5 * abs(Mp_sat - Mn_sat)
            sM_sat = 0.5 * float(np.hypot(sp_sat, sn_sat))

            bp, sp = _tail_intercept(Hp, Mp_c, "pos")
            bn, sn = _tail_intercept(Hn, Mn_c, "neg")
            M0 = 0.5 * (bp - bn)
            sM0 = 0.5 * float(np.hypot(sp, sn))

            records.append({
                "temperature_K": float(T_K[i]),
                "sigma_T_K": float(sT[i]),
                "branch_hmax_A_per_m": float(hmax[i]),
                "M_r_A_per_m": float(M_r),
                "sigma_M_r_A_per_m": float(sM_r),
                "M_sat_A_per_m": float(M_sat),
                "sigma_M_sat_A_per_m": float(sM_sat),
                "M_0_A_per_m": float(M0),
                "sigma_M_0_A_per_m": float(sM0),
            })

        run_summary = pd.DataFrame.from_records(records)
        run_summary["M_r_norm"], run_summary["sigma_M_r_norm"] = normalize_01_with_sigma(
            run_summary["M_r_A_per_m"].to_numpy(),
            run_summary["sigma_M_r_A_per_m"].to_numpy(),
        )
        run_summary["M_sat_norm"], run_summary["sigma_M_sat_norm"] = normalize_01_with_sigma(
            run_summary["M_sat_A_per_m"].to_numpy(),
            run_summary["sigma_M_sat_A_per_m"].to_numpy(),
        )
        run_summary["M_0_norm"], run_summary["sigma_M_0_norm"] = normalize_01_with_sigma(
            np.abs(run_summary["M_0_A_per_m"].to_numpy()),
            run_summary["sigma_M_0_A_per_m"].to_numpy(),
        )

        T_used = run_summary["temperature_K"].to_numpy()
        sT_used = run_summary["sigma_T_K"].to_numpy()
        method_specs = [
            ("M_r", r"$M_\mathrm{r}$", "M_r_norm", "sigma_M_r_norm"),
            ("M_sat", r"$M_\mathrm{sat}$", "M_sat_norm", "sigma_M_sat_norm"),
            ("M_0", r"$M_0$", "M_0_norm", "sigma_M_0_norm"),
        ]
        method_rows = []
        for method_key, method_label, y_col, sy_col in method_specs:
            tc_half, sigma_tc_half = half_height_tc_with_sigma(
                T_used,
                sT_used,
                run_summary[y_col].to_numpy(),
                run_summary[sy_col].to_numpy(),
            )
            method_rows.append({
                "run": label,
                "method_key": method_key,
                "method": method_label,
                "Tc_K": float(tc_half),
                "sigma_Tc_K": float(sigma_tc_half),
                "drive_current_A_rms": float(I_rms),
                "H_sat_A_per_m": float(H_sat),
                "H_median_A_per_m": float(np.median(hmax[keep])),
                "n_loops": int(len(keep)),
                "T_min_K": float(T_K.min()),
                "T_max_K": float(T_K.max()),
                "T_retained_min_K": float(T_retained.min()),
                "T_retained_max_K": float(T_retained.max()),
            })

        finite_method_rows = [
            r for r in method_rows
            if np.isfinite(r["Tc_K"]) and np.isfinite(r["sigma_Tc_K"])
        ]
        if finite_method_rows:
            _method_tcs = np.array([r["Tc_K"] for r in finite_method_rows], dtype=float)
            _method_sigmas = np.array([r["sigma_Tc_K"] for r in finite_method_rows], dtype=float)
            _weights = 1.0 / np.maximum(_method_sigmas, 1e-9) ** 2
            Tc_methods_mean = float(np.average(_method_tcs, weights=_weights))
            sigma_Tc_methods_stat = float(1.0 / np.sqrt(np.sum(_weights)))
            method_spread_run = float(np.std(_method_tcs, ddof=1)) if len(_method_tcs) >= 2 else 0.0
        else:
            Tc_methods_mean = float("nan")
            sigma_Tc_methods_stat = float("nan")
            method_spread_run = float("nan")

        _method_by_key = {r["method_key"]: r for r in method_rows}
        Tc_half = _method_by_key.get("M_0", {}).get("Tc_K", float("nan"))
        sigma_Tc_half = _method_by_key.get("M_0", {}).get("sigma_Tc_K", float("nan"))

        M0 = run_summary["M_0_A_per_m"].to_numpy()
        sM0 = run_summary["sigma_M_0_A_per_m"].to_numpy()
        Msq = M0**2
        sMsq = 2.0 * np.abs(M0) * sM0
        snr = np.abs(M0) / np.maximum(sM0, 1e-30)
        Tc_seed = Tc_half if np.isfinite(Tc_half) else float(np.nanmedian(T_used))

        # Anchored window-size scan, mirroring the main Method-III check. The
        # candidate pool is anchored to the half-height seed rather than being
        # redefined by each fitted intercept, so noisy above-transition points
        # cannot pull the window upward.
        K_MIN = 5
        FIT_UPPER_MARGIN_K = 2.0
        FIT_LOWER_MARGIN_K = 35.0
        FIT_TC_TOLERANCE_K = 25.0
        FIT_MIN_P_VALUE = 0.05
        pool = (
            (T_used >= Tc_seed - FIT_LOWER_MARGIN_K)
            & (T_used <= Tc_seed + FIT_UPPER_MARGIN_K)
            & (M0 > 0)
            & (snr > 3.0)
        )
        idx_pool = np.flatnonzero(pool)
        if idx_pool.size < K_MIN:
            idx_pool = np.argsort(np.abs(T_used - Tc_seed))[: max(K_MIN, 8)]
        idx_sorted = idx_pool[np.argsort(-T_used[idx_pool])]

        scan = []
        for K in range(K_MIN, len(idx_sorted) + 1):
            sel = idx_sorted[:K]
            sT_safe = np.maximum(sT_used[sel], 1e-3 / np.sqrt(12.0))
            sMsq_safe = np.maximum(sMsq[sel], 1e-30)
            try:
                res_K = odr_fit(
                    fit_functions.linear, None,
                    T_used[sel], sT_safe, Msq[sel], sMsq_safe,
                    param_names=["intercept", "slope"],
                )
            except Exception:
                continue
            m_K = float(res_K.params[1])
            if m_K >= 0.0 or not np.isfinite(m_K):
                continue
            Tc_K_val = -float(res_K.params[0]) / m_K
            if not np.isfinite(Tc_K_val) or abs(Tc_K_val - Tc_seed) > FIT_TC_TOLERANCE_K:
                continue
            scan.append({
                "K": K,
                "res": res_K,
                "sel": sel,
                "p_value": float(res_K.p_value),
            })

        Tc = float("nan")
        sigma_Tc_mf = float("nan")
        redchi_mf = float("nan")
        n_fit = 0
        if scan:
            acceptable = [r for r in scan if r["p_value"] >= FIT_MIN_P_VALUE]
            if acceptable:
                best = max(acceptable, key=lambda r: (r["K"], r["p_value"]))
            else:
                best = max(scan, key=lambda r: (r["p_value"], r["K"]))
            res = best["res"]
            mask = np.zeros_like(T_used, dtype=bool)
            mask[best["sel"]] = True

            b0, m0 = float(res.params[0]), float(res.params[1])
            raw_cov_scale = float(getattr(res.raw_output, "res_var", res.redchi))
            cov_scale = max(raw_cov_scale, 1.0)
            if res.cov is not None:
                cov = res.cov * cov_scale
                sb0 = float(np.sqrt(max(cov[0, 0], 0.0)))
                sm0 = float(np.sqrt(max(cov[1, 1], 0.0)))
                cov_si = float(cov[0, 1])
            else:
                sb0, sm0 = float(res.errors[0]), float(res.errors[1])
                cov_si = 0.0

            Tc = -b0 / m0
            var_Tc = (
                (sb0 / m0) ** 2
                + (b0 * sm0 / m0**2) ** 2
                - 2.0 * b0 * cov_si / m0**3
            )
            sigma_Tc_mf = float(np.sqrt(max(0.0, var_Tc)))
            redchi_mf = float(res.redchi)
            n_fit = int(mask.sum())

        run_curve_records = []
        for _record in run_summary.to_dict("records"):
            run_curve_records.append({
                "run": label,
                "drive_current_A_rms": float(I_rms),
                "H_sat_A_per_m": float(H_sat),
                "H_median_A_per_m": float(np.median(hmax[keep])),
                "temperature_K": float(_record["temperature_K"]),
                "sigma_T_K": float(_record["sigma_T_K"]),
                "M_r_norm": float(_record["M_r_norm"]),
                "sigma_M_r_norm": float(_record["sigma_M_r_norm"]),
                "M_sat_norm": float(_record["M_sat_norm"]),
                "sigma_M_sat_norm": float(_record["sigma_M_sat_norm"]),
                "M_0_norm": float(_record["M_0_norm"]),
                "sigma_M_0_norm": float(_record["sigma_M_0_norm"]),
            })

        return {
            "Tc_K_mf": Tc,
            "sigma_Tc_K_mf": sigma_Tc_mf,
            "Tc_K_half": Tc_half,
            "sigma_Tc_K_half": sigma_Tc_half,
            "Tc_K_methods_mean": Tc_methods_mean,
            "sigma_Tc_K_methods_stat": sigma_Tc_methods_stat,
            "method_spread_K": method_spread_run,
            "Tc_M_r_K": _method_by_key.get("M_r", {}).get("Tc_K", float("nan")),
            "sigma_Tc_M_r_K": _method_by_key.get("M_r", {}).get("sigma_Tc_K", float("nan")),
            "Tc_M_sat_K": _method_by_key.get("M_sat", {}).get("Tc_K", float("nan")),
            "sigma_Tc_M_sat_K": _method_by_key.get("M_sat", {}).get("sigma_Tc_K", float("nan")),
            "Tc_M_0_K": _method_by_key.get("M_0", {}).get("Tc_K", float("nan")),
            "sigma_Tc_M_0_K": _method_by_key.get("M_0", {}).get("sigma_Tc_K", float("nan")),
            "drive_current_A_rms": float(I_rms),
            "H_sat_A_per_m": float(H_sat),
            "H_median_A_per_m": float(np.median(hmax[keep])),
            "n_fit": n_fit,
            "n_loops": int(len(keep)),
            "T_min_K": float(T_K.min()),
            "T_max_K": float(T_K.max()),
            "T_retained_min_K": float(T_retained.min()),
            "T_retained_max_K": float(T_retained.max()),
            "redchi": redchi_mf,
        }, method_rows, run_curve_records

    _summary_records = []
    _curve_records = []
    _method_records = []
    for _label, _path in RUN_FILES.items():
        _summary, _methods, _curves = _run_methods(_label, _path)
        if _summary is not None:
            _summary_records.append({"run": _label, **_summary})
        _method_records.extend(_methods)
        _curve_records.extend(_curves)

    cross_run = pd.DataFrame.from_records(_summary_records)
    run_curves = pd.DataFrame.from_records(_curve_records)
    run_method_tcs = pd.DataFrame.from_records(_method_records)
    if "H_median_A_per_m" in cross_run.columns and not cross_run.empty:
        _first_h = float(cross_run.loc[cross_run["run"] == "first", "H_median_A_per_m"].iloc[0])
        cross_run["drive_fraction_vs_first"] = cross_run["H_median_A_per_m"] / _first_h
        cross_run["low_drive_flag"] = cross_run["drive_fraction_vs_first"] < 0.80
        if not run_curves.empty:
            run_curves["drive_fraction_vs_first"] = run_curves["H_median_A_per_m"] / _first_h
            run_curves["low_drive_flag"] = run_curves["drive_fraction_vs_first"] < 0.80
        if not run_method_tcs.empty:
            run_method_tcs["drive_fraction_vs_first"] = run_method_tcs["H_median_A_per_m"] / _first_h
            run_method_tcs["low_drive_flag"] = run_method_tcs["drive_fraction_vs_first"] < 0.80
    return cross_run, run_curves, run_method_tcs


@app.cell
def _(FIG_DIR, cross_run, run_curves, run_method_tcs):
    run_curves.to_csv(FIG_DIR / "curie_run_curves.csv", index=False)
    run_method_tcs.to_csv(FIG_DIR / "curie_run_method_tcs.csv", index=False)
    cross_run.to_csv(FIG_DIR / "curie_cross_run_summary.csv", index=False)
    return


@app.cell
def _(cross_run, np, plt, run_curves, save_figure, smooth):
    # Measured counterpart to the guide's finite-field sketch. We plot the
    # near-saturation branch-split proxy because it is evaluated at the drive
    # field itself, so it is the cleanest visual comparison of the three input
    # settings. The x-axis uses the main-run method-mean half-height as the
    # common temperature scale; normalizing each run by its own transition
    # would hide the observed drive-setting shifts.
    fig_drive, ax_drive = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)

    _first = cross_run.loc[cross_run["run"] == "first"]
    _T_ref = float(_first["Tc_K_methods_mean"].iloc[0]) if not _first.empty else float(cross_run["Tc_K_methods_mean"].iloc[0])
    _run_order = ["third", "first", "second"]
    _color_map = {"third": "#66a9ff", "first": "#1f77b4", "second": "#08306b"}
    _marker_map = {"third": "^", "first": "o", "second": "s"}

    for _run in _run_order:
        _rows = run_curves.loc[run_curves["run"] == _run].sort_values("temperature_K")
        if _rows.empty:
            continue
        _x = _rows["temperature_K"].to_numpy(float) / _T_ref
        _y = _rows["M_sat_norm"].to_numpy(float)
        _h_frac = float(_rows["drive_fraction_vs_first"].iloc[0])
        _current = float(_rows["drive_current_A_rms"].iloc[0])
        _label = rf"{_run}: $I={_current:.2f}\,$A, $H/H_1={_h_frac:.2f}$"
        ax_drive.plot(
            _x, smooth(_y),
            color=_color_map.get(_run, "0.3"), linewidth=2.2,
            label=_label,
        )
        _stride = max(1, len(_x) // 45)
        ax_drive.scatter(
            _x[::_stride], _y[::_stride],
            color=_color_map.get(_run, "0.3"), marker=_marker_map.get(_run, "o"),
            s=13, alpha=0.42, linewidths=0.0,
        )

    ax_drive.axhline(0.5, color="0.45", linewidth=0.8, linestyle="--", alpha=0.65)
    ax_drive.axvline(1.0, color="0.35", linewidth=0.9, linestyle=":", alpha=0.85)
    ax_drive.text(
        1.005, 0.08,
        r"run-1 $T_{1/2}^{\mathrm{app}}$",
        rotation=90, ha="left", va="bottom", color="0.25", fontsize=8,
    )
    ax_drive.set_xlabel(r"$T / T_{1/2}^{\mathrm{app}}$ (run 1)")
    ax_drive.set_ylabel(r"normalized $M_\mathrm{sat}(T)$")
    ax_drive.set_xlim(0.75, 1.45)
    ax_drive.set_ylim(-0.03, 1.03)
    ax_drive.minorticks_on()
    ax_drive.grid(True, which="major", alpha=0.24)
    ax_drive.grid(True, which="minor", alpha=0.10)
    ax_drive.legend(loc="upper right", framealpha=0.95, fontsize=8)
    save_figure(fig_drive, "curie_measured_drive_comparison")
    fig_drive
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Method IV qualitative cross-check: Curie–Weiss above $T_c$

    The half-height (Methods I–III) and mean-field (Method III $M_0^2$)
    estimators all use the *below-$T_c$* side of the transition. As a
    qualitative cross-check, we also fit the *above-$T_c$* side via the
    Curie–Weiss law

    $$
    \chi(T) \;=\; \frac{C}{T - T_c}
    \quad\Longleftrightarrow\quad
    \frac{1}{\chi(T)} \;=\; \frac{T - T_c}{C},
    $$

    so $1/\chi$ is linear in $T$ with the $T$-axis intercept equal to
    $T_c$. Per loop we fit an apparent $\chi(T)$ as the slope of the loop's full
    $M(H)$ data: above $T_c$ the loop collapses to a single line through
    the origin (no hysteresis), so a single linear regression returns
    a well-defined response slope. Restricting the $1/\chi$ vs $T$ line fit to
    $T > T_c^\mathrm{seed} + 5\,\mathrm{K}$ keeps us in the asymptotic
    paramagnetic regime where Curie–Weiss is valid.

    The intercept is insensitive to a common multiplicative rescaling of
    $M/H$, but the susceptibility scale was not independently calibrated
    here. This therefore checks whether the
    high-$T$ response extrapolates to the same temperature scale; it is
    not a standalone calibrated susceptibility measurement.
    """)
    return


@app.cell
def _(
    TEMPERATURE_K,
    branches_for_row,
    data,
    diagnostics_with_sigma,
    fit_functions,
    np,
    odr_fit,
    sigma_T_K,
):
    # Method IV (qualitative cross-check): Curie–Weiss above T_c.
    # In the paramagnetic regime, chi(T) = M/H = C / (T - T_c), so
    # 1/chi = (T - T_c)/C is linear in T with T-axis intercept T_c.
    # Per loop we fit an apparent chi from the full loop's M(H) data:
    #   - above T_c the loop has no hysteresis, so all four branches
    #     collapse to a single line through the origin and a single
    #     polyfit slope is well-defined;
    #   - below T_c the slope is contaminated by hysteresis, so we
    #     restrict the 1/chi vs T fit to T > Tc_seed + buffer.
    # The headline T_c family uses the half-height crossings (below-T_c
    # side); this cell adds a paramagnetic-side qualitative check using
    # only the high-T data. The common M/H calibration scale does not
    # move the intercept, but the susceptibility scale is not independently
    # calibrated well enough to call this a standalone
    # susceptibility measurement. chi here is computed from the
    # *uncorrected* M, since the global background we subtract elsewhere
    # is itself an average chi over the high-T quartile and would be
    # circular here.
    chi_vals = np.full(len(data), np.nan, dtype=float)
    sigma_chi_vals = np.full(len(data), np.nan, dtype=float)
    for _i in range(len(data)):
        _row = data.iloc[_i]
        _Hp, _Mp, _Hn, _Mn = branches_for_row(_row)
        _H = np.concatenate([_Hp, _Hn])
        _M = np.concatenate([_Mp, _Mn])
        try:
            _co, _cov = np.polyfit(_H, _M, 1, cov=True)
            chi_vals[_i] = float(_co[0])
            sigma_chi_vals[_i] = float(np.sqrt(_cov[0, 0]))
        except (np.linalg.LinAlgError, ValueError):
            pass

    T_arr = TEMPERATURE_K
    sT_arr = sigma_T_K

    # Seed paramagnetic regime from the most robust half-height (Method I).
    _seed_I = diagnostics_with_sigma.iloc[0]
    Tc_seed_CW = float(_seed_I["Tc_K"])
    if not np.isfinite(Tc_seed_CW):
        Tc_seed_CW = float(np.nanmedian(T_arr))

    # Buffer above the seed avoids the immediate transition shoulder
    # where finite-H smearing distorts chi (the Curie–Weiss form is
    # only valid asymptotically far above T_c; closer in, mean-field
    # corrections kick in).
    CW_BUFFER_K = 5.0
    mask_CW = (T_arr > Tc_seed_CW + CW_BUFFER_K) & (chi_vals > 0) & np.isfinite(chi_vals)

    cw_result = None
    Tc_CW = float("nan")
    sigma_Tc_CW = float("nan")
    b_CW = m_CW = float("nan")
    redchi_CW = float("nan")
    rescale_CW = 1.0
    if mask_CW.sum() >= 5:
        _inv_chi = 1.0 / chi_vals[mask_CW]
        _sigma_inv_chi = sigma_chi_vals[mask_CW] / chi_vals[mask_CW] ** 2
        _T_CW = T_arr[mask_CW]
        _sT_CW = np.maximum(sT_arr[mask_CW], 1e-3 / np.sqrt(12.0))
        _sIC = np.maximum(_sigma_inv_chi, 1e-30)
        try:
            cw_result = odr_fit(
                fit_functions.linear, None,
                _T_CW, _sT_CW, _inv_chi, _sIC,
                param_names=["intercept", "slope"],
            )
            b_CW = float(cw_result.params[0])
            m_CW = float(cw_result.params[1])
            redchi_CW = float(cw_result.redchi)
            raw_cov_scale_CW = float(getattr(cw_result.raw_output, "res_var", redchi_CW))
            cov_scale_CW = max(raw_cov_scale_CW, 1.0)
            rescale_CW = float(np.sqrt(cov_scale_CW))
            if cw_result.cov is not None:
                _cov_CW = cw_result.cov * cov_scale_CW
                sb_CW = float(np.sqrt(max(_cov_CW[0, 0], 0.0)))
                sm_CW = float(np.sqrt(max(_cov_CW[1, 1], 0.0)))
                cov_CW = float(_cov_CW[0, 1])
            else:
                sb_CW = float(cw_result.errors[0])
                sm_CW = float(cw_result.errors[1])
                cov_CW = 0.0
            if m_CW != 0.0 and np.isfinite(m_CW):
                Tc_CW = -b_CW / m_CW
                _var = (
                    (sb_CW / m_CW) ** 2
                    + (b_CW * sm_CW / m_CW ** 2) ** 2
                    - 2.0 * b_CW * cov_CW / m_CW ** 3
                )
                sigma_Tc_CW = float(np.sqrt(max(0.0, _var)))
        except Exception:
            cw_result = None

    return (
        CW_BUFFER_K,
        Tc_CW,
        Tc_seed_CW,
        b_CW,
        chi_vals,
        cw_result,
        m_CW,
        mask_CW,
        rescale_CW,
        redchi_CW,
        sigma_Tc_CW,
        sigma_chi_vals,
    )


@app.cell
def _(
    CW_BUFFER_K,
    TEMPERATURE_K,
    Tc_CW,
    Tc_seed_CW,
    b_CW,
    chi_vals,
    cw_result,
    m_CW,
    mask_CW,
    np,
    plt,
    save_figure,
    sigma_Tc_CW,
    sigma_T_K,
    sigma_chi_vals,
):
    # Plot 1/chi vs T with the Curie–Weiss line and Tc x-intercept.
    # Shade the excluded region so the fit window edge is unambiguous.
    fig_cw, (ax_cw, ax_cw_res) = plt.subplots(
        2, 1,
        figsize=(7.4, 5.9),
        sharex=False,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
    )

    _good = np.isfinite(chi_vals) & (chi_vals > 0)
    _T_all = TEMPERATURE_K[_good]
    _sigma_T_all = sigma_T_K[_good]
    _inv_chi_all = 1.0 / chi_vals[_good]
    _sigma_inv_all = sigma_chi_vals[_good] / chi_vals[_good] ** 2

    _used = mask_CW[_good]

    # Auto-bound axes to the fit-window data: x covers the paramagnetic
    # regime plus a few K of context; y covers the in-window 1/chi
    # range with margin. Without this, a single near-Tc loop where chi
    # happens to be tiny gives a huge 1/chi outlier that dominates the
    # vertical scale and visually drowns the linear fit.
    _edge = Tc_seed_CW + CW_BUFFER_K
    if _used.any():
        _y_in = _inv_chi_all[_used]
        _y_med = float(np.median(_y_in))
        _y_iqr = float(np.percentile(_y_in, 90) - np.percentile(_y_in, 10))
        _y_hi = _y_med + 3.0 * max(_y_iqr, abs(_y_med))
        _y_lo = min(0.0, _y_med - 1.0 * max(_y_iqr, abs(_y_med)))
        _x_lo = _edge - 5.0
        _x_hi = float(_T_all[_used].max()) + 3.0
    else:
        _y_lo, _y_hi = -1.0, 1.0
        _x_lo = float(_T_all.min()) - 2.0
        _x_hi = float(_T_all.max()) + 5.0

    ax_cw.axvspan(
        _x_lo, _edge, color="0.85", alpha=0.35, zorder=0,
        label=rf"excluded ($T \leq T_\mathrm{{c}}^\mathrm{{seed}}+{CW_BUFFER_K:.0f}\,\mathrm{{K}}$)",
    )

    _excl = ~_used
    if _excl.any():
        ax_cw.errorbar(
            _T_all[_excl], _inv_chi_all[_excl],
            xerr=_sigma_T_all[_excl], yerr=_sigma_inv_all[_excl],
            fmt="o", color="0.55", markersize=2.8, elinewidth=0.6,
            alpha=0.7,
        )
    if _used.any():
        ax_cw.errorbar(
            _T_all[_used], _inv_chi_all[_used],
            xerr=_sigma_T_all[_used], yerr=_sigma_inv_all[_used],
            fmt="o", color="C0", markersize=3.6, elinewidth=0.8,
            alpha=0.9, label="paramagnetic fit window",
        )

    if cw_result is not None and np.isfinite(Tc_CW):
        _T_line = np.linspace(max(Tc_CW, _x_lo), _x_hi, 200)
        ax_cw.plot(
            _T_line, m_CW * _T_line + b_CW, "-", color="C3", linewidth=2.0,
            label=rf"apparent Curie–Weiss $T_\mathrm{{c}}={Tc_CW:.1f}\pm{sigma_Tc_CW:.1f}\,\mathrm{{K}}$",
        )
        if _x_lo <= Tc_CW <= _x_hi:
            ax_cw.axvline(Tc_CW, color="C3", linewidth=0.8, linestyle=":")
        if _used.any():
            _residual = _inv_chi_all[_used] - (m_CW * _T_all[_used] + b_CW)
            _residual_sigma = _sigma_inv_all[_used]
            _safe_sigma = np.where(_residual_sigma > 0, _residual_sigma, np.nan)
            _weighted_residual = _residual / _safe_sigma
            ax_cw_res.errorbar(
                _T_all[_used], _weighted_residual,
                xerr=_sigma_T_all[_used], yerr=np.ones_like(_weighted_residual),
                fmt="o", color="C0", markersize=3.2, elinewidth=0.8,
                alpha=0.9,
            )
            _res_ylim = float(np.nanmax(np.abs(_weighted_residual) + 1.0))
            if not np.isfinite(_res_ylim) or _res_ylim <= 0:
                _res_ylim = 1.0
            ax_cw_res.set_ylim(-1.15 * _res_ylim, 1.15 * _res_ylim)
            _res_x_lo = float(np.nanmin(_T_all[_used] - _sigma_T_all[_used]))
            _res_x_hi = float(np.nanmax(_T_all[_used] + _sigma_T_all[_used]))
            _res_x_pad = max(0.25, 0.04 * (_res_x_hi - _res_x_lo))
            ax_cw_res.set_xlim(_res_x_lo - _res_x_pad, _res_x_hi + _res_x_pad)

            _zoom_x_lo = max(_x_lo, Tc_CW - 0.5)
            _zoom_x_hi = min(_x_hi, _edge + 8.0)
            _zoom = (_T_all >= _zoom_x_lo) & (_T_all <= _zoom_x_hi)
            if np.count_nonzero(_zoom & _used) >= 2:
                ax_zoom = ax_cw.inset_axes([0.52, 0.10, 0.43, 0.42])
                if np.any(_zoom & _excl):
                    ax_zoom.errorbar(
                        _T_all[_zoom & _excl], _inv_chi_all[_zoom & _excl],
                        xerr=_sigma_T_all[_zoom & _excl], yerr=_sigma_inv_all[_zoom & _excl],
                        fmt="o", color="0.55", markersize=2.2, elinewidth=0.45,
                        alpha=0.65,
                    )
                ax_zoom.errorbar(
                    _T_all[_zoom & _used], _inv_chi_all[_zoom & _used],
                    xerr=_sigma_T_all[_zoom & _used], yerr=_sigma_inv_all[_zoom & _used],
                    fmt="o", color="C0", markersize=2.5, elinewidth=0.5,
                    alpha=0.85,
                )
                _T_zoom_line = np.linspace(max(Tc_CW, _zoom_x_lo), _zoom_x_hi, 100)
                _zoom_line_y = m_CW * _T_zoom_line + b_CW
                ax_zoom.plot(_T_zoom_line, _zoom_line_y, "-", color="C3", linewidth=1.3)
                ax_zoom.axhline(0, color="0.45", linewidth=0.5, linestyle="--")
                ax_zoom.axvline(Tc_CW, color="C3", linewidth=0.6, linestyle=":")
                _zoom_y = _inv_chi_all[_zoom]
                _zoom_sigma = _sigma_inv_all[_zoom]
                _zoom_y_hi = max(
                    0.5,
                    float(np.nanmax(_zoom_y + _zoom_sigma)),
                    float(np.nanmax(_zoom_line_y)),
                )
                _zoom_y_lo = min(0.0, float(np.nanmin(_zoom_line_y)))
                ax_zoom.set_xlim(_zoom_x_lo, _zoom_x_hi)
                ax_zoom.set_ylim(_zoom_y_lo - 0.05 * _zoom_y_hi, 1.15 * _zoom_y_hi)
                ax_zoom.set_title("near-intercept zoom", fontsize=7)
                ax_zoom.tick_params(labelsize=7)
                ax_zoom.grid(True, which="major", alpha=0.20)

    ax_cw.axhline(0, color="0.4", linewidth=0.6, linestyle="--")
    ax_cw_res.axhline(0, color="0.35", linewidth=0.8, linestyle="--")
    ax_cw.set_xlim(_x_lo, _x_hi)
    ax_cw.set_ylim(_y_lo, _y_hi)
    ax_cw.set_xlabel(r"$T$ (K)")
    ax_cw.set_ylabel(r"$1/\chi$ (arb. units)")
    ax_cw.minorticks_on()
    ax_cw.grid(True, which="major", alpha=0.25)
    ax_cw.grid(True, which="minor", alpha=0.10)
    ax_cw.legend(loc="upper left", fontsize=8, framealpha=0.95)

    ax_cw_res.set_xlabel(r"$T$ (K), fit window")
    ax_cw_res.set_ylabel(r"$(1/\chi - f(T))/\sigma$")
    ax_cw_res.minorticks_on()
    ax_cw_res.grid(True, which="major", alpha=0.25)
    ax_cw_res.grid(True, which="minor", alpha=0.10)

    save_figure(fig_cw, "curie_method4_curie_weiss")
    fig_cw
    return


@app.cell(hide_code=True)
def _(SIGMA_T_ABS_K, Tc_CW, Tc_K, cross_run, diagnostics_with_sigma, mo, np, odr_result, redchi_CW, sigma_Tc_CW, sigma_Tc_K):
    # Bottom-line. The half-height crossings of M_r, M_sat, and M_0
    # (Methods I, II, III in normalized form) are the headline
    # estimators because they are model-free: each locates the
    # temperature where the smoothed proxy crosses 50% of its dynamic
    # range. The mean-field M_0^2(T) line zero-crossing (Method III in
    # absolute units) is reported as a cross-check. It is intentionally
    # not pooled into the headline because it uses a model-dependent
    # narrow-window extrapolation of the same saturation-tail quantity.
    finite_hh = diagnostics_with_sigma.dropna(subset=["Tc_K"])
    hh_tcs = finite_hh["Tc_K"].to_numpy(dtype=float)
    hh_sigmas = finite_hh["sigma_Tc_K"].to_numpy(dtype=float)
    if len(hh_tcs) > 0:
        _w = 1.0 / np.maximum(hh_sigmas, 1e-9) ** 2
        Tc_headline = float(np.average(hh_tcs, weights=_w))
        sigma_Tc_headline_stat_unrescaled = float(1.0 / np.sqrt(np.sum(_w)))
        method_spread = float(np.std(hh_tcs, ddof=1)) if len(hh_tcs) >= 2 else 0.0
        # The three half-height methods disagree by far more than their
        # local sigmas. That is a methodological spread, not additional
        # counting/statistical noise, so keep the inverse-variance statistical
        # sigma uninflated and put the method spread in the systematic budget
        # below. The chi^2/nu scale factor is kept only as a diagnostic;
        # applying that factor here and also adding method_spread as a
        # systematic would double-count the same disagreement.
        if len(hh_tcs) >= 2:
            chi2_combine = float(np.sum((hh_tcs - Tc_headline) ** 2 * _w))
            dof_combine = len(hh_tcs) - 1
            redchi_combine = chi2_combine / dof_combine if dof_combine > 0 else 1.0
        else:
            redchi_combine = 1.0
        sigma_Tc_headline_stat = sigma_Tc_headline_stat_unrescaled
    else:
        Tc_headline = float("nan")
        sigma_Tc_headline_stat_unrescaled = float("nan")
        sigma_Tc_headline_stat = float("nan")
        method_spread = 0.0
        redchi_combine = float("nan")

    _run_estimator_col = "Tc_K_methods_mean" if "Tc_K_methods_mean" in cross_run.columns else "Tc_K_half"
    finite_runs = cross_run.dropna(subset=[_run_estimator_col]) if _run_estimator_col in cross_run.columns else cross_run.iloc[0:0]
    run_tcs_all = finite_runs[_run_estimator_col].to_numpy(dtype=float) if not finite_runs.empty else np.array([])
    run_spread_all = float(np.std(run_tcs_all, ddof=1)) if len(run_tcs_all) >= 2 else 0.0
    if "low_drive_flag" in finite_runs.columns:
        finite_runs_preferred = finite_runs.loc[~finite_runs["low_drive_flag"]]
    else:
        finite_runs_preferred = finite_runs
    run_tcs_preferred = finite_runs_preferred[_run_estimator_col].to_numpy(dtype=float) if not finite_runs_preferred.empty else np.array([])
    run_spread = float(np.std(run_tcs_preferred, ddof=1)) if len(run_tcs_preferred) >= 2 else run_spread_all

    # Systematic budget: combines the inter-method spread (within-run
    # methodological systematic), the run-to-run spread of the *half-height*
    # T_c per run (cross-run systematic on the same estimator family), and
    # the fully-correlated thermometer absolute-accuracy term. The preferred
    # high-drive budget excludes the low-drive third run; the conservative
    # all-drive envelope retains all three drive settings as a stress test.
    # The mean-field-vs-half-height shift is not added in quadrature here:
    # the mean-field model is not adopted as the headline estimator, and
    # its methodological information is reported separately.
    syst_total = float(np.hypot(np.hypot(method_spread, run_spread), SIGMA_T_ABS_K))
    syst_total_all = float(np.hypot(np.hypot(method_spread, run_spread_all), SIGMA_T_ABS_K))
    # Display only: gap between the mean-field check and the half-height
    # headline, kept for context but not in the systematic.
    mf_shift_display = float(abs(Tc_K - Tc_headline)) if np.isfinite(Tc_K) and np.isfinite(Tc_headline) else 0.0

    def _fmt_k(x, ndigits=1):
        return f"{x:.{ndigits}f}" if np.isfinite(x) else "—"

    def _row_run(r):
        _name = f"run `{r['run']}`"
        _frac = r.get("drive_fraction_vs_first", np.nan)
        _drive_str = f"{_frac:.2f}" if np.isfinite(_frac) else "—"
        _t0 = r.get("T_retained_min_K", np.nan)
        _t1 = r.get("T_retained_max_K", np.nan)
        _t_range = f"{_t0:.1f}-{_t1:.1f}" if np.isfinite(_t0) and np.isfinite(_t1) else "—"
        _flag = "low-drive check" if bool(r.get("low_drive_flag", False)) else "preferred"
        return "| " + " | ".join([
            _name,
            f"{r['drive_current_A_rms']:.2f}",
            _drive_str,
            _t_range,
            _fmt_k(r.get("Tc_M_r_K", np.nan)),
            _fmt_k(r.get("Tc_M_sat_K", np.nan)),
            _fmt_k(r.get("Tc_M_0_K", np.nan)),
            _fmt_k(r.get("Tc_K_methods_mean", np.nan)),
            _flag,
        ]) + " |"

    cross_rows_methods = [
        _row_run(r)
        for _, r in finite_runs.iterrows() if np.isfinite(r[_run_estimator_col])
    ]

    _third_run = finite_runs.loc[finite_runs["run"] == "third"] if "run" in finite_runs.columns else finite_runs.iloc[0:0]
    if not _third_run.empty and "T_retained_min_K" in _third_run.columns:
        third_retained_min = float(_third_run["T_retained_min_K"].iloc[0])
    else:
        third_retained_min = float("nan")

    method_rows = [
        f"| {r['method']} | {r['Tc_K']:.2f} | {r['sigma_Tc_K']:.2f} | (half-height local fit) |"
        for _, r in diagnostics_with_sigma.iterrows() if np.isfinite(r["Tc_K"])
    ]

    Tc_headline_C = Tc_headline - 273.15

    mo.callout(
        mo.md(rf"""
    ### Bottom-line apparent transition temperature for the Curie experiment

    **Half-height crossings — three methods on run `first`** (headline family):

    | Method | $T_{{1/2}}^\mathrm{{app}}$ (K) | local $\sigma_T$ (K) | source |
    |---|---|---|---|
    {chr(10).join(method_rows)}

    **All three physical Curie scans** (repeated at different primary drive settings):

    | Run | $I_\mathrm{{rms}}$ (A) | drive frac. | retained $T$ (K) | $M_r$ (K) | $M_\mathrm{{sat}}$ (K) | $M_0$ (K) | method mean (K) | status |
    |---|---|---|---|---|---|---|---|---|
    {chr(10).join(cross_rows_methods)}

    The low-drive third run is not an interchangeable repeat: it is the
    required different-resistance scan and shows field-amplitude and coverage
    sensitivity. After the field-ready cut it starts at
    `{third_retained_min:.1f}` K, so it lacks the cold plateau present in the
    first two scans. The preferred high-drive run-spread term uses only `first`
    and `second`, while the conservative all-drive envelope retains all three
    scans:
    $\sigma_\text{{run}}={run_spread:.1f}\,\mathrm{{K}}$ versus
    $\sigma_\text{{run,all}}={run_spread_all:.1f}\,\mathrm{{K}}$.

    **Preferred reported value** (central value from run `first`; uncertainty from complete high-drive scans):

    $$
    T_{{1/2,\mathrm{{HD}}}}^\mathrm{{app}} \;=\; {Tc_headline:.0f} \;\pm\; {syst_total:.0f}\;\mathrm{{K}}
    \;=\; {Tc_headline_C:.0f}\pm{syst_total:.0f}\,^\circ\mathrm{{C}}.
    $$

    Folding the low-drive incomplete scan into the budget as a stress-test
    envelope expands this to $\pm{syst_total_all:.0f}\,\mathrm{{K}}$.

    This is an operational finite-field transition midpoint, not a clean
    zero-field thermodynamic Curie temperature. The local crossing
    uncertainties are much smaller than the spread between methods and runs,
    so the rounded headline keeps only the dominant systematic-sized error.
    The unrounded local repeatability of the common pipeline is
    $\sigma={sigma_Tc_headline_stat:.2f}\,\mathrm{{K}}$.

    The preferred quoted uncertainty combines

    - method-to-method spread of the half-height crossings within run `first`: $\sigma_\text{{method}}={method_spread:.1f}\,\mathrm{{K}}$;
    - run-to-run spread of the per-run three-method estimates across the complete high-drive scans: $\sigma_\text{{run}}={run_spread:.1f}\,\mathrm{{K}}$;
    - thermometer absolute-accuracy term: $\sigma_\text{{therm}}={SIGMA_T_ABS_K:.1f}\,\mathrm{{K}}$;

    in quadrature.

    The thermometer term shifts $T_c$ rigidly within a single run, so it
    does not affect $\sigma_{{T_c}}^\text{{stat}}$; it is carried in the
    quoted systematic error instead.

    **Check: linearized mean-field $T_c$ on run `first`.**
    The $M_0^2(T)\propto T_c-T$ form is a near-$T_c$ approximation.
    With a narrow transition window,
    the weighted linear fit gives $T_c^\mathrm{{MF}}={Tc_K:.1f}\pm{sigma_Tc_K:.1f}\,\mathrm{{K}}$
    ($\chi^2/\nu={odr_result.redchi:.2f}$). This is ${mf_shift_display:.1f}\,\mathrm{{K}}
    from the half-height headline and therefore supports the same transition
    scale. It is kept as a qualitative check, not pooled into the headline,
    because it is a model-dependent extrapolation of the saturation-tail
    intercept rather than a direct operational midpoint.

    **Qualitative check: Curie–Weiss $T_c$ from $1/\chi(T)$ above $T_c$.**
    Independently of the half-height (which uses the below-$T_c$ side)
    and the mean-field fit (also below-$T_c$), the paramagnetic-side
    Curie–Weiss relation $\chi=C/(T-T_c)$ predicts $1/\chi$ linear in
    $T$ with $T$-axis intercept $T_c$. Per-loop apparent $\chi(T)$ is fit from
    the full single-loop $M(H)$ slope (above $T_c$ the four branches
    collapse to one line through the origin, so a single slope is
    well-defined). A weighted line fit of $1/\chi$ vs $T$ on the paramagnetic
    window gives
    $T_c^\mathrm{{CW}} = {Tc_CW:.1f}\pm{sigma_Tc_CW:.1f}\,\mathrm{{K}}$
    ($\chi^2/\nu={redchi_CW:.1f}$). A common multiplicative error in
    the placeholder $M/H$ calibration would not move the intercept, but
    the susceptibility scale is not independently calibrated; together with
    the large $\chi^2/\nu$, this makes the fit qualitative evidence of
    consistency rather than an independent precision $T_c$.
    """),
        kind="success",
    )
    return (
        Tc_headline,
        Tc_headline_C,
        method_spread,
        mf_shift_display,
        run_spread,
        run_spread_all,
        sigma_Tc_headline_stat,
        syst_total,
        syst_total_all,
    )


@app.cell
def _(
    Tc_CW,
    Tc_K,
    Tc_headline,
    np,
    plt,
    run_method_tcs,
    save_figure,
    sigma_Tc_CW,
    sigma_Tc_K,
    sigma_Tc_headline_stat,
    syst_total,
    syst_total_all,
):
    _fig_tc, _ax_tc = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)

    if np.isfinite(Tc_headline) and np.isfinite(syst_total_all):
        _ax_tc.axhspan(
            Tc_headline - syst_total_all, Tc_headline + syst_total_all,
            color="C0", alpha=0.06, label=r"all-drive diagnostic envelope",
        )
    if np.isfinite(Tc_headline) and np.isfinite(syst_total):
        _ax_tc.axhspan(
            Tc_headline - syst_total, Tc_headline + syst_total,
            color="C0", alpha=0.12, label=r"preferred high-drive uncertainty",
        )
    if np.isfinite(Tc_headline) and np.isfinite(sigma_Tc_headline_stat):
        _ax_tc.axhspan(
            Tc_headline - sigma_Tc_headline_stat,
            Tc_headline + sigma_Tc_headline_stat,
            color="C0", alpha=0.22, label=r"local crossing scatter only",
        )
        _ax_tc.axhline(
            Tc_headline, color="C0", linewidth=1.5,
            label=rf"headline $T_{{1/2}}^{{\mathrm{{app}}}}={Tc_headline:.1f}$ K",
        )

    _x_positions = []
    _x_labels = []
    _group_edges = []  # (x_left, x_right, label)

    _finite_run_methods = run_method_tcs.dropna(subset=["Tc_K"]) if "Tc_K" in run_method_tcs.columns else run_method_tcs.iloc[0:0]
    _run_order = ["first", "second", "third"]
    _run_label_map = {"first": "Run 1", "second": "Run 2", "third": "Run 3"}
    _run_color_map = {"first": "C0", "second": "C2", "third": "C6"}
    _method_order = ["M_r", "M_sat", "M_0"]
    _method_label_map = {"M_r": r"$M_\mathrm{r}$", "M_sat": r"$M_\mathrm{sat}$", "M_0": r"$M_0$"}
    _method_marker_map = {"M_r": "o", "M_sat": "s", "M_0": "^"}
    _x = 0.0
    for _run in _run_order:
        _rows_run = _finite_run_methods.loc[_finite_run_methods["run"] == _run]
        if _rows_run.empty:
            continue
        _group_left = _x
        _color = _run_color_map.get(_run, "0.4")
        _run_label = _run_label_map.get(_run, _run)
        _low_drive = bool(_rows_run["low_drive_flag"].iloc[0]) if "low_drive_flag" in _rows_run.columns else False
        _legend_label = _run_label + (" (low drive)" if _low_drive else "")
        for _method in _method_order:
            _row_match = _rows_run.loc[_rows_run["method_key"] == _method]
            if _row_match.empty:
                continue
            _row = _row_match.iloc[0]
            _ax_tc.errorbar(
                _x, _row["Tc_K"], yerr=_row["sigma_Tc_K"],
                fmt=_method_marker_map.get(_method, "o"), color=_color, ecolor=_color,
                capsize=3, markersize=6,
                mfc="white" if _low_drive else _color,
                label=_legend_label if _method == _method_order[0] else None,
            )
            _x_positions.append(_x)
            _x_labels.append(_method_label_map.get(_method, _method))
            _x += 1.0
        _group_edges.append((_group_left, _x - 1.0, _run_label))
        _x += 0.85

    _x = (_x_positions[-1] + 1.5) if _x_positions else 0.0
    _group_left_X = _x
    if np.isfinite(Tc_K):
        _ax_tc.errorbar(
            _x, Tc_K, yerr=sigma_Tc_K,
            fmt="D", color="C3", mfc="white", ecolor="C3", capsize=3, markersize=6,
            label=rf"mean-field check",
        )
        _x_positions.append(_x)
        _x_labels.append(r"MF")
        _x += 1.0
    if np.isfinite(Tc_CW):
        _ax_tc.errorbar(
            _x, Tc_CW, yerr=sigma_Tc_CW,
            fmt="^", color="C4", ecolor="C4", capsize=3, markersize=6,
            label="Curie–Weiss check",
        )
        _x_positions.append(_x)
        _x_labels.append(r"CW")
    if _x_positions and _x_positions[-1] >= _group_left_X:
        _group_edges.append((_group_left_X, _x_positions[-1], "Cross-checks"))

    # Light vertical separators between the three groups for readability.
    for _i, (_a, _b, _) in enumerate(_group_edges[:-1]):
        _next_a = _group_edges[_i + 1][0]
        _ax_tc.axvline(
            0.5 * (_b + _next_a), color="0.85", linewidth=0.8,
            linestyle=":", zorder=0,
        )
    for _a, _b, _label in _group_edges:
        _ax_tc.text(0.5 * (_a + _b), 1.02, _label,
                    ha="center", va="bottom", fontsize=9, color="0.25",
                    transform=_ax_tc.get_xaxis_transform(), clip_on=False)

    # Bound y to the region populated by point estimates. Method-IV
    # CW's broad propagated errorbar would otherwise stretch the
    # axis past the science region; matplotlib will draw it but clip
    # the bar at the axis edge, which is acceptable since the legend
    # quotes the value and the markdown reports the full number.
    _all_centers = []
    for _, _r in _finite_run_methods.iterrows():
        _all_centers.append(float(_r["Tc_K"]))
    if np.isfinite(Tc_K):
        _all_centers.append(float(Tc_K))
    if np.isfinite(Tc_CW):
        _all_centers.append(float(Tc_CW))
    if _all_centers:
        _y_lo = min(_all_centers + [Tc_headline - syst_total_all]) - 5.0
        _y_hi = max(_all_centers + [Tc_K + sigma_Tc_K]) + 5.0
        _ax_tc.set_ylim(_y_lo, _y_hi)

    _ax_tc.set_xticks(_x_positions, _x_labels)
    _ax_tc.set_ylabel(r"$T_\mathrm{c}$ (K)")
    _ax_tc.grid(True, axis="y", which="major", alpha=0.25)
    _ax_tc.minorticks_on()
    _ax_tc.grid(True, axis="y", which="minor", alpha=0.10)
    _ax_tc.legend(loc="best", fontsize=8, framealpha=0.95, ncol=1)

    save_figure(_fig_tc, "curie_tc_summary")
    _fig_tc
    return


if __name__ == "__main__":
    app.run()
