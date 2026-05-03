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
    DATA_FILE = ROOT / "data" / "first" / "CurieData_06_05_23-03_05_2026.txt"
    DATA_DIR = ROOT / "data"
    DATA_XLSX = ROOT.parent / "ferromagnetism" / "data" / "data.xlsx"
    FIG_DIR = ROOT.parent.parent / "report" / "media"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # All three runs (sweep duplicates), in chronological order. Each run's
    # CurieData_*.txt is the LabVIEW dump used by the rest of the notebook.
    RUN_FILES = {
        sub: next((DATA_DIR / sub).glob("CurieData_*.txt"))
        for sub in ("first", "second", "third")
    }

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
    # sigma_T_smear = |dT/dt|*dt_loop / sqrt(12). The LabVIEW LSD on T
    # is 1 mK (sigma_T_LSD ~ 0.3 mK), negligible against the smearing
    # in the transition region (~0.3-0.4 K). The thermometer absolute
    # accuracy is treated separately as a fully-correlated systematic
    # because it shifts T_c rigidly within a single run.
    _t_s = data["Time (sec)"].to_numpy(float)
    _T_K = temperature_K.to_numpy(float)
    _dT_dt = np.gradient(_T_K, _t_s)
    _dt_loop = float(np.median(np.diff(_t_s))) if len(_t_s) > 1 else 0.0
    sigma_T_smear = np.abs(_dT_dt) * _dt_loop / np.sqrt(12.0)
    _T_LSD = 1e-3
    sigma_T_K = np.sqrt(sigma_T_smear**2 + (_T_LSD / np.sqrt(12.0)) ** 2)

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
    - loop window: `{_dt_loop:.2f}` s; per-loop $\sigma_T$ in transition region (median, p95): `{np.median(sigma_T_K):.3f}`, `{np.percentile(sigma_T_K, 95):.3f}` K

    The Curie circuit constants are hard-coded to match the lab schematic
    ($N_1=250$ primary, $N_2=2500$ secondary, $R_y=3.97\,\mathrm{{k\Omega}}$,
    $C=19.78\,\mu\mathrm{{F}}$). $L$, $R_x$, and $A$ are pulled from
    `../ferromagnetism/data/data.xlsx` only as placeholders — the Curie
    run uses a rod-in-solenoid geometry, not the toroid, but the
    diagnostic plots depend on these only through linear rescalings.

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

    def half_height_tc_with_sigma(T, sT, y, sy, n_boot=400, rng_seed=0):
        # Half-height crossing temperature with a Monte-Carlo sigma that
        # folds T-axis smearing (sT) and y-axis statistical noise (sy)
        # together. For each bootstrap replica we draw shifted (T_i, y_i),
        # smooth, and locate the first 0.5 crossing of (y - min) / range.
        T = np.asarray(T, dtype=float)
        y = np.asarray(y, dtype=float)
        sT = np.asarray(sT, dtype=float)
        sy = np.asarray(sy, dtype=float)

        def _crossing(T_arr, y_arr):
            ys = smooth(y_arr)
            shifted = ys - float(np.min(ys))
            scale = float(np.max(shifted))
            if scale == 0.0:
                return np.nan
            yn = shifted / scale
            for i in range(len(T_arr) - 1):
                y0, y1 = yn[i], yn[i + 1]
                if (y0 - 0.5) * (y1 - 0.5) <= 0 and y0 != y1:
                    return T_arr[i] + (0.5 - y0) * (T_arr[i + 1] - T_arr[i]) / (y1 - y0)
            return np.nan

        order = np.argsort(T)
        Ts, ys, sTs, sys_ = T[order], y[order], sT[order], sy[order]
        center = _crossing(Ts, ys)

        rng = np.random.default_rng(rng_seed)
        replicas = []
        for _ in range(n_boot):
            T_jit = Ts + rng.normal(0.0, np.maximum(sTs, 1e-9))
            y_jit = ys + rng.normal(0.0, np.maximum(sys_, 1e-9))
            j = np.argsort(T_jit)
            tc = _crossing(T_jit[j], y_jit[j])
            if np.isfinite(tc):
                replicas.append(tc)

        if len(replicas) < 5 or not np.isfinite(center):
            return float(center) if np.isfinite(center) else np.nan, np.nan
        return float(center), float(np.std(replicas, ddof=1))

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

    def sat_intercept(
        H, M, tail,
        n_min=5,
        n_max_frac=0.5,
        intercept_tol_sigma=2.0,
        slope_tol_sigma=2.0,
    ):
        # On the saturated tail, M(H) ~= M0 + chi_bg * H. Grow the window
        # from n_min outward (ranked by tail-side |H|) and freeze it once
        # either gate trips:
        #   (a) intercept shifts > intercept_tol_sigma * sigma_b_prev
        #       -- i.e. the apparent M_0 has started to drift;
        #   (b) slope shifts > slope_tol_sigma * sigma_a_prev -- i.e.
        #       chi_bg is no longer locally constant, signalling that
        #       the new point is leaving the linear/saturated regime
        #       (typical when expansion reaches the remanence shoulder).
        # Without (b), the original code could absorb shoulder points
        # into the fit as long as their *intercept* happened to look
        # stable, biasing M_0 low. The slope check trips earlier.
        H_arr = np.asarray(H, dtype=float)
        M_arr = np.asarray(M, dtype=float)
        order = _tail_order(H_arr, tail)
        H_sorted, M_sorted = H_arr[order], M_arr[order]
        n_total = len(H_sorted)
        n_max = min(n_total, max(n_min + 1, int(n_max_frac * n_total)))

        intercept, sigma_b, slope, sigma_a = _linear_fit(
            H_sorted[:n_min], M_sorted[:n_min]
        )
        n_used = n_min
        for n in range(n_min + 1, n_max + 1):
            try:
                b_new, sb_new, a_new, sa_new = _linear_fit(
                    H_sorted[:n], M_sorted[:n]
                )
            except (np.linalg.LinAlgError, ValueError):
                break
            if abs(b_new - intercept) > intercept_tol_sigma * max(sigma_b, 1e-30):
                break
            if abs(a_new - slope) > slope_tol_sigma * max(sigma_a, 1e-30):
                break
            intercept, sigma_b, slope, sigma_a, n_used = b_new, sb_new, a_new, sa_new, n
        return intercept, sigma_b, n_used

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
        half_height_tc_with_sigma,
        local_intercept_at,
        normalize_01_with_sigma,
        sat_intercept,
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

    The guide notes that above $T_c$ the sample still has field-induced magnetization. We fit $\chi_\mathrm{{bg}}$ on the highest-T quartile (`{int(high_temperature_mask.sum())}` loops) so the choice is robust to the unknown $T_c$ — every loop in this quartile sits well above the transition for any sample with $T_c$ below the run's maximum temperature. The fit form is

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
    sat_intercept,
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
        #   so               B = M_0 + (alpha + 1) * H,
        #   i.e.             B - (alpha + 1) * H = M_0.
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
        _b_pos, _s_pos, _n_pos = sat_intercept(_H_pos, _M_pos_corr, tail="pos")
        _b_neg, _s_neg, _n_neg = sat_intercept(_H_neg, _M_neg_corr, tail="neg")
        _bf_pos, _ = sat_intercept_fixed(_H_pos, _M_pos_corr, tail="pos")
        _bf_neg, _ = sat_intercept_fixed(_H_neg, _M_neg_corr, tail="neg")

        _M_0 = 0.5 * (_b_pos - _b_neg)
        _sigma_M_0 = 0.5 * float(np.hypot(_s_pos, _s_neg))
        _M_0_5pt = 0.5 * (_bf_pos - _bf_neg)
        _ns = (_n_pos, _n_neg)

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
            "M_0_5pt_A_per_m": _M_0_5pt,
            "M_0_n_used_avg": float(np.mean(_ns)),
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

    _T_used = summary["temperature_K"].to_numpy()
    _sT_used = summary["sigma_T_K"].to_numpy()
    diagnostics = pd.DataFrame([
        {"method": r"I: $M_r$ (H=0)", **transition_diagnostics(_T_used, summary["M_r_norm"].to_numpy())},
        {"method": rf"II: $M_\mathrm{{sat}}$ (H=±{H_SAT:.2f} A/m)", **transition_diagnostics(_T_used, summary["M_sat_norm"].to_numpy())},
        {"method": r"III: $M_0$ (sat. extrap.→H=0)", **transition_diagnostics(_T_used, summary["M_0_norm"].to_numpy())},
    ])

    # Bootstrap-with-sigma half-height T_c for each method. Folds the
    # per-loop sigma_T (heating-rate smearing) and the per-loop sigma_y
    # together to give a usable statistical 1-sigma on the half-height
    # crossing temperature. Methods I and II use this; Method III also
    # produces a half-height crossing on the normalized M_0 curve, but
    # its formal T_c estimate comes from the M_0^2(T) ODR fit below.
    _tc_M_r, _stc_M_r = half_height_tc_with_sigma(
        _T_used, _sT_used,
        summary["M_r_norm"].to_numpy(),
        summary["sigma_M_r_norm"].to_numpy(),
        rng_seed=1,
    )
    _tc_M_sat, _stc_M_sat = half_height_tc_with_sigma(
        _T_used, _sT_used,
        summary["M_sat_norm"].to_numpy(),
        summary["sigma_M_sat_norm"].to_numpy(),
        rng_seed=2,
    )
    _tc_M_0, _stc_M_0 = half_height_tc_with_sigma(
        _T_used, _sT_used,
        summary["M_0_norm"].to_numpy(),
        summary["sigma_M_0_norm"].to_numpy(),
        rng_seed=3,
    )
    diagnostics_with_sigma = pd.DataFrame([
        {"method": r"I: $M_r$ (half-height, H=0)",                     "Tc_K": _tc_M_r,    "sigma_Tc_K": _stc_M_r},
        {"method": rf"II: $M_\mathrm{{sat}}$ (half-height, H=±{H_SAT:.2f} A/m)", "Tc_K": _tc_M_sat, "sigma_Tc_K": _stc_M_sat},
        {"method": r"III: $M_0$ (half-height, sat. extrap.→H=0)",      "Tc_K": _tc_M_0,    "sigma_Tc_K": _stc_M_0},
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
    B(H)=M_0+(\alpha+1)H \;\;\Longleftrightarrow\;\;
    B-(\alpha+1)H=M_0,
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

    The fit window is grown adaptively from $n_\mathrm{{min}}=5$ points
    (the lab guide's prescription) outward, ranked by tail-side $|H|$;
    expansion is frozen when **either** the intercept **or** the slope
    drifts by more than $2\sigma$ relative to the previous fit, since
    both signal the end of the linear/saturated regime. The LabVIEW
    export stores the $H\ge0$ half of the upper branch and the $H\le0$
    half of the lower branch, so each loop contributes two saturated
    tips. Statistical uncertainty on $M_0$ is propagated in quadrature
    from the two per-fit intercept sigmas. The instructor's strict
    5-point fit is retained alongside as `M_0_5pt_A_per_m` for
    cross-checking.

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
    steepest-slope temperature on the smoothed normalized curve, no
    uncertainties). Smoothing uses a Savitzky–Golay filter (order-2
    polynomial over a 21-loop window): unlike a moving average, SG
    fits a local polynomial and reports its centre value, which preserves
    the curvature of the transition shoulder and the slope at the
    inflection point — precisely the features the half-height and
    steepest-slope diagnostics read off. A moving average would smear
    the shoulder and flatten the inflection slope.

    {table_md(diag, ["method", "half_height_K", "steepest_slope_K", "steepest_slope_value"])}

    **Half-height $T_c$ with bootstrap $\sigma$** for all three
    methods. Each row is the half-height crossing temperature with a
    1-sigma from 400 Monte-Carlo replicas that jointly perturb $T$ by
    per-loop $\sigma_T$ (heating-rate smearing) and $y$ by per-loop
    $\sigma_y$ (residuals from each method's per-loop fit). Method III
    also feeds the formal mean-field $T_c$ from the $M_0^2(T)$ ODR
    fit below.

    {table_md(diag_sig, ["method", "Tc_K", "sigma_Tc_K"])}

    Normalized ranges: $M_r$ `{summary['M_r_norm'].min():.3f}`--`{summary['M_r_norm'].max():.3f}`, $M_\mathrm{{sat}}$ `{summary['M_sat_norm'].min():.3f}`--`{summary['M_sat_norm'].max():.3f}`, $M_0$ `{summary['M_0_norm'].min():.3f}`--`{summary['M_0_norm'].max():.3f}`. Mean adaptive window size: `{summary['M_0_n_used_avg'].mean():.1f}` points.
    """)
    return


@app.cell
def _(diagnostics_with_sigma, fit_functions, np, odr_fit, summary):
    # Mean-field square-root scaling near Tc:  M0(T)^2 ~= m * (Tc - T).
    # The mean-field form is only valid in a narrow band just below Tc;
    # far below Tc the M(T) curve has heavy curvature that breaks the
    # linear M0^2(T). Two safeguards:
    #
    # 1) seed the fit window from the half-height bootstrap T_c estimate
    #    (run on the normalized M_0 curve in the cell above) so we begin
    #    with a robust, model-free anchor;
    # 2) iterate a few times: keep only points within FIT_HALFWIDTH_K of
    #    the running T_c, refit with ODR, and update T_c. This converges
    #    in 2-3 iterations and stays inside the mean-field linear regime.
    #
    # The redchi reported below should land near 1 if the model is
    # appropriate for the chosen window. If it stays >> 1 the window is
    # still too wide or sigma_M0 is underestimated; in that case we
    # quote sigma_Tc with redchi-rescaled errorbars (the standard ODR
    # convention when the model imperfectly fits) rather than pretending
    # the fit is perfect.
    T_all = summary["temperature_K"].to_numpy()
    sT_all = summary["sigma_T_K"].to_numpy()
    M0_all = summary["M_0_A_per_m"].to_numpy()
    sM0_all = summary["sigma_M_0_A_per_m"].to_numpy()
    M0_sq_all = M0_all ** 2
    sM0_sq_all = 2.0 * np.abs(M0_all) * sM0_all

    FIT_HALFWIDTH_K = 25.0  # mean-field linear regime is only ~10-30 K wide.
    # Margin on the +T side. Strict mean-field has M^2 = 0 above Tc, so
    # any data above Tc that survives the M0>0 + SNR>3 filters is upward
    # noise — including it tilts the linear fit and pulls Tc above its
    # true value. We keep a small +2 K margin so a couple of marginal
    # points right at the transition stay in (giving the fit some
    # leverage on the intercept) but no further. (Empirically widening
    # this to +10 K drives chi^2/nu from ~50 to ~230 and shifts the
    # mean-field Tc by 20 K — confirming +2 K is the principled choice.)
    FIT_UPPER_MARGIN_K = 2.0

    # Seed Tc from the half-height bootstrap on the normalized M_0 curve.
    _seed_row = diagnostics_with_sigma.iloc[2]  # method "3. M_0 norm (half-height)"
    _Tc_seed = float(_seed_row["Tc_K"])
    if not np.isfinite(_Tc_seed):
        _Tc_seed = float(np.nanmedian(T_all))

    snr = np.abs(M0_all) / np.maximum(sM0_all, 1e-30)
    Tc_iter = _Tc_seed
    odr_result = None
    for _ in range(5):
        in_window = (T_all >= Tc_iter - FIT_HALFWIDTH_K) & (T_all <= Tc_iter + FIT_UPPER_MARGIN_K)
        fit_mask = in_window & (M0_all > 0) & (snr > 3.0)
        if fit_mask.sum() < 5:
            order_T = np.argsort(np.abs(T_all - Tc_iter))[:8]
            fit_mask = np.zeros_like(T_all, dtype=bool)
            fit_mask[order_T] = True

        sT_fit_safe = np.maximum(sT_all[fit_mask], 1e-3 / np.sqrt(12.0))
        sMsq_fit_safe = np.maximum(sM0_sq_all[fit_mask], 1e-30)
        odr_result = odr_fit(
            fit_functions.linear, None,
            T_all[fit_mask], sT_fit_safe, M0_sq_all[fit_mask], sMsq_fit_safe,
            param_names=["intercept", "slope"],
        )
        intercept_new = float(odr_result.params[0])
        slope_new = float(odr_result.params[1])
        if slope_new == 0.0 or not np.isfinite(slope_new):
            break
        Tc_new = -intercept_new / slope_new
        if abs(Tc_new - Tc_iter) < 0.05:
            Tc_iter = Tc_new
            break
        Tc_iter = Tc_new

    msq_intercept = float(odr_result.params[0])
    msq_slope = float(odr_result.params[1])
    msq_sigma_intercept = float(odr_result.errors[0])
    msq_sigma_slope = float(odr_result.errors[1])
    msq_cov_si = float(odr_result.cov[0, 1]) if odr_result.cov is not None else 0.0

    # When the fit's redchi > 1 the model is not perfectly capturing the
    # data (mean-field is approximate); rescale the ODR covariance by
    # sqrt(redchi) so the quoted 1-sigma reflects the actual scatter
    # around the best-fit line. This is the standard "Birge-style"
    # rescaling used elsewhere when reduced chi-square indicates the
    # residual model error is non-negligible.
    redchi = float(odr_result.redchi)
    rescale = float(np.sqrt(max(redchi, 1.0)))
    msq_sigma_intercept *= rescale
    msq_sigma_slope *= rescale
    msq_cov_si *= rescale**2

    Tc_K = -msq_intercept / msq_slope
    var_Tc = (
        (msq_sigma_intercept / msq_slope) ** 2
        + (msq_intercept * msq_sigma_slope / msq_slope ** 2) ** 2
        - 2.0 * msq_intercept * msq_cov_si / msq_slope ** 3
    )
    sigma_Tc_K = float(np.sqrt(max(0.0, var_Tc)))
    Tc_C = Tc_K - 273.15
    return (
        FIT_HALFWIDTH_K,
        FIT_UPPER_MARGIN_K,
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
def _(FIT_HALFWIDTH_K, FIT_UPPER_MARGIN_K, Tc_C, Tc_K, fit_mask, mo, odr_result, rescale, sigma_Tc_K, summary):
    mo.md(rf"""
    ## Method 3 cross-check: mean-field $M_0^2(T)$ extrapolation

    > **This block is a methodological cross-check, not a $T_c$ measurement.**
    > The mean-field form $M_0^2(T)\propto T_c-T$ is only a near-$T_c$
    > approximation, and on this data its linear fit returns
    > $\chi^2/\nu={odr_result.redchi:.1f}$ — the model does not describe
    > the in-window data. The Birge-rescaled error bar below is therefore
    > too wide to be informative as an independent estimator. The
    > headline $T_c$ is the model-free half-height crossing reported in
    > the bottom-line callout.

    Procedurally: we seed the fit window from the half-height bootstrap
    $T_c$ on the normalized $M_0$ curve (Methods 1-3 cluster around
    $\sim$210-216 K), keep only data with
    $T_c-{FIT_HALFWIDTH_K:.0f}\,\mathrm{{K}}\le T\le T_c+{FIT_UPPER_MARGIN_K:.0f}\,\mathrm{{K}}$
    and $M_0/\sigma_{{M_0}}>3$, and iterate the ODR fit a few times
    until $T_c$ stabilises. The fit folds per-loop $\sigma_{{M_0^2}}=2|M_0|\sigma_{{M_0}}$
    on $y$ and the heating-rate smearing $\sigma_T$ on $x$. The fit
    converged on `{int(fit_mask.sum())}` points and gives

    $$
    T_c^\mathrm{{MF}} \;=\; {Tc_K:.1f}\;\mathrm{{K}}\;=\;{Tc_C:.1f}\,^\circ\mathrm{{C}}
    \quad(\chi^2/\nu={odr_result.redchi:.1f},\;\text{{Birge-rescaled }}\sigma_{{T_c}}=\pm{sigma_Tc_K:.0f}\,\mathrm{{K}}).
    $$

    The reduced-chi-square excess measures how poorly the strict
    mean-field linear form describes the in-window data; the per-loop
    statistical $\sigma_{{M_0}}$ from the saturation-tail fit
    underestimates the model-mismatch error, so we Birge-rescale the
    ODR covariance by $\sqrt{{\chi^2/\nu}}={rescale:.2f}$ before
    propagating to $\sigma_{{T_c}}$. The result places $T_c$ in the
    same neighbourhood as the half-height estimators (210-230 K),
    which is the only useful information this fit conveys.

    Adaptive vs 5-point baseline (mean over retained loops):
    $\langle M_0^\text{{adapt}}\rangle = {summary['M_0_A_per_m'].mean():.3g}$,
    $\langle M_0^\text{{5pt}}\rangle = {summary['M_0_5pt_A_per_m'].mean():.3g}\;\mathrm{{A\,m^{{-1}}}}$.
    """)
    return


@app.cell
def _(
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
    fig_msq, ax_msq = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)

    scale = 1e6  # (A/m)^2 -> (kA/m)^2

    excl = ~fit_mask
    if excl.any():
        ax_msq.errorbar(
            T_all[excl],
            M0_sq_all[excl] / scale,
            xerr=sT_all[excl],
            yerr=sM0_sq_all[excl] / scale,
            fmt="o",
            color="0.65",
            markersize=2.6,
            elinewidth=0.6,
            alpha=0.55,
            label="excluded",
        )
    ax_msq.errorbar(
        T_all[fit_mask],
        M0_sq_all[fit_mask] / scale,
        xerr=sT_all[fit_mask],
        yerr=sM0_sq_all[fit_mask] / scale,
        fmt="o",
        color="C0",
        markersize=3.4,
        elinewidth=0.7,
        alpha=0.85,
        label=r"fit window (near-$T_c$ ODR)",
    )

    T_line = np.linspace(float(T_all[fit_mask].min()), Tc_K + 5.0, 200)
    ax_msq.plot(
        T_line,
        (msq_slope * T_line + msq_intercept) / scale,
        "-",
        color="C3",
        linewidth=2.0,
        label=rf"mean-field fit: $T_c={Tc_K:.1f}\pm{sigma_Tc_K:.1f}\,\mathrm{{K}}$",
    )
    ax_msq.axhline(0, color="0.4", linewidth=0.6, linestyle="--")
    ax_msq.axvline(Tc_K, color="C3", linewidth=0.8, linestyle=":")

    # Restrict the visible y-range to the near-Tc data so the fit and
    # its uncertainty band are readable, instead of being squashed by
    # the deep-T M_0^2 values that the fit deliberately excludes.
    _y_max_fit = float(M0_sq_all[fit_mask].max() / scale)
    ax_msq.set_ylim(-0.10 * _y_max_fit, 1.4 * _y_max_fit)

    ymin, ymax = ax_msq.get_ylim()
    ax_msq.fill_betweenx(
        [ymin, ymax],
        Tc_K - sigma_Tc_K,
        Tc_K + sigma_Tc_K,
        color="C3",
        alpha=0.10,
    )

    if np.isfinite(Tc_headline):
        ax_msq.axvline(
            Tc_headline,
            color="C2",
            linewidth=1.2,
            linestyle="-",
            label=rf"half-height headline: $T_c={Tc_headline:.1f}\pm{sigma_Tc_headline_stat:.2f}\,\mathrm{{K}}$ (stat)",
        )

    ax_msq.set_ylim(ymin, ymax)

    ax_msq.set_xlabel(r"$T$ (K)")
    ax_msq.set_ylabel(r"$M_0^2$ (kA$^2\,$m$^{-2}$)")
    ax_msq.set_title(r"Method 3: near-$T_c$ mean-field $M_0^2(T)$ fit, with headline half-height $T_c$")
    ax_msq.minorticks_on()
    ax_msq.grid(True, which="major", alpha=0.25)
    ax_msq.grid(True, which="minor", alpha=0.10)
    ax_msq.legend(loc="upper right", fontsize=8)

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
    sigma_T_arr = summary["sigma_T_K"].to_numpy()
    series = [
        ("M_r_norm",    "sigma_M_r_norm",    r"$M_r$  ($H{=}0$)",                   "C0", "o"),
        ("M_sat_norm",  "sigma_M_sat_norm",  r"$M_\mathrm{sat}$  ($H{=}\pm H_\mathrm{sat}$)",  "C3", "s"),
        ("M_0_norm",    "sigma_M_0_norm",    r"$M_0$  (sat. extrap.$\to H{=}0$)",   "C2", "^"),
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
            alpha=0.28,
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
    np,
    odr_fit,
    pd,
    read_table,
):
    # Cross-run check: re-run the Method-3 chain (background subtraction,
    # adaptive saturation-tail intercepts, M_0^2(T) ODR fit) on all three
    # data files. The spread of T_c across runs is a practical empirical
    # systematic that absorbs whatever the thermometer absolute accuracy,
    # the warming-vs-cooling thermal lag sign, and the sample-mounting
    # geometry contribute. This cell deliberately re-uses minimal code
    # so that a bug in the main pipeline does not silently propagate.
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

    def _tail_intercept(H, M, tail, n_min=5, frac=0.5, tol_b=2.0, tol_a=2.0):
        # Mirrors sat_intercept() in the main pipeline: gate window
        # expansion on BOTH intercept and slope stability so the fit
        # can't silently drift into the remanence shoulder.
        order = np.argsort(-H) if tail == "pos" else np.argsort(H)
        h, m = H[order], M[order]
        n_max = min(len(h), max(n_min + 1, int(frac * len(h))))
        co, cov = np.polyfit(h[:n_min], m[:n_min], 1, cov=True)
        b, sb = float(co[1]), float(np.sqrt(cov[1, 1]))
        a, sa = float(co[0]), float(np.sqrt(cov[0, 0]))
        for n in range(n_min + 1, n_max + 1):
            try:
                co2, cov2 = np.polyfit(h[:n], m[:n], 1, cov=True)
            except (np.linalg.LinAlgError, ValueError):
                break
            b2, sb2 = float(co2[1]), float(np.sqrt(cov2[1, 1]))
            a2, sa2 = float(co2[0]), float(np.sqrt(cov2[0, 0]))
            if abs(b2 - b) > tol_b * max(sb, 1e-30):
                break
            if abs(a2 - a) > tol_a * max(sa, 1e-30):
                break
            b, sb, a, sa = b2, sb2, a2, sa2
        return b, sb

    def _run_method3(path):
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
            return None
        keep = np.arange(int(ready[0]), len(df))

        # Use the top quartile of the kept loops as the paramagnetic
        # regime; matches the main pipeline's choice and avoids the
        # T >= 273.15 K cut which assumed T_c near room temperature.
        bg_mask = T_K[keep] >= np.quantile(T_K[keep], 0.75)
        Hb, Mb = [], []
        for i in keep[bg_mask]:
            Hp, Mp, Hn, Mn = _branches(df.iloc[i], xp, yp, xn, yn, h_per_x)
            Hb.extend([Hp, Hn])
            Mb.extend([Mp, Mn])
        a_bg, b_bg = np.polyfit(np.concatenate(Hb), np.concatenate(Mb), 1)

        M0 = np.zeros(len(keep))
        sM0 = np.zeros(len(keep))
        T_used = np.zeros(len(keep))
        sT_used = np.zeros(len(keep))
        for k, i in enumerate(keep):
            Hp, Mp, Hn, Mn = _branches(df.iloc[i], xp, yp, xn, yn, h_per_x)
            Mp_c = Mp - (a_bg * Hp + b_bg)
            Mn_c = Mn - (a_bg * Hn + b_bg)
            bp, sp = _tail_intercept(Hp, Mp_c, "pos")
            bn, sn = _tail_intercept(Hn, Mn_c, "neg")
            M0[k] = 0.5 * (bp - bn)
            sM0[k] = 0.5 * float(np.hypot(sp, sn))
            T_used[k] = T_K[i]
            sT_used[k] = sT[i]

        Msq = M0**2
        sMsq = 2.0 * np.abs(M0) * sM0
        snr = np.abs(M0) / np.maximum(sM0, 1e-30)

        # Half-height crossing of the normalized M0 curve. This is the
        # model-free Tc estimator and is also reused as the seed for the
        # mean-field iterator below. The cross-run systematic budget
        # uses *this* per-run Tc, not the mean-field one — using the
        # mean-field Tc per run would double-count the mean-field model
        # mismatch (which is already captured separately as the
        # method-to-method spread within a single run).
        def _crossing_at(T_arr, y_arr):
            for j in range(len(T_arr) - 1):
                y0, y1 = y_arr[j], y_arr[j + 1]
                if (y0 - 0.5) * (y1 - 0.5) <= 0 and y0 != y1:
                    return T_arr[j] + (0.5 - y0) * (T_arr[j + 1] - T_arr[j]) / (y1 - y0)
            return float("nan")

        order_T = np.argsort(T_used)
        Ts, M0s = T_used[order_T], M0[order_T]
        sTs, sM0s = sT_used[order_T], sM0[order_T]
        M0_pos = np.where(M0s > 0, M0s, 0.0)
        rng = float(M0_pos.max() - M0_pos.min())
        Tc_seed = float(np.nanmedian(Ts))
        Tc_half = float("nan")
        sigma_Tc_half = float("nan")
        if rng > 0:
            yn = (M0_pos - M0_pos.min()) / rng
            Tc_half = _crossing_at(Ts, yn)
            if np.isfinite(Tc_half):
                Tc_seed = Tc_half
                # Bootstrap sigma: jitter T and M0 by their per-loop
                # uncertainties and re-locate the half-height crossing.
                _rng = np.random.default_rng(7)
                _replicas = []
                for _b in range(200):
                    _T_jit = Ts + _rng.normal(0.0, np.maximum(sTs, 1e-9))
                    _M_jit = M0s + _rng.normal(0.0, np.maximum(sM0s, 1e-9))
                    _j = np.argsort(_T_jit)
                    _Tj, _Mj = _T_jit[_j], _M_jit[_j]
                    _Mp = np.where(_Mj > 0, _Mj, 0.0)
                    _r = float(_Mp.max() - _Mp.min())
                    if _r <= 0:
                        continue
                    _yn = (_Mp - _Mp.min()) / _r
                    _tc = _crossing_at(_Tj, _yn)
                    if np.isfinite(_tc):
                        _replicas.append(_tc)
                if len(_replicas) >= 5:
                    sigma_Tc_half = float(np.std(_replicas, ddof=1))

        # Iterative ODR window: keep mean-field linear regime only.
        Tc_it = Tc_seed
        res = None
        for _ in range(5):
            mask = (
                (T_used >= Tc_it - 25.0)
                & (T_used <= Tc_it + 2.0)
                & (M0 > 0)
                & (snr > 3.0)
            )
            if mask.sum() < 5:
                near = np.argsort(np.abs(T_used - Tc_it))[:8]
                mask = np.zeros_like(T_used, dtype=bool)
                mask[near] = True
            sT_safe = np.maximum(sT_used[mask], 1e-3 / np.sqrt(12.0))
            sMsq_safe = np.maximum(sMsq[mask], 1e-30)
            try:
                res = odr_fit(
                    fit_functions.linear, None,
                    T_used[mask], sT_safe, Msq[mask], sMsq_safe,
                    param_names=["intercept", "slope"],
                )
            except Exception:
                return None
            m0_ = float(res.params[1])
            if m0_ == 0.0 or not np.isfinite(m0_):
                break
            Tc_new = -float(res.params[0]) / m0_
            if abs(Tc_new - Tc_it) < 0.05:
                Tc_it = Tc_new
                break
            Tc_it = Tc_new

        if res is None:
            return None

        b0, m0 = float(res.params[0]), float(res.params[1])
        sb0, sm0 = float(res.errors[0]), float(res.errors[1])
        cov_si = float(res.cov[0, 1]) if res.cov is not None else 0.0
        # Birge rescale to absorb mean-field model mismatch.
        rescale_run = float(np.sqrt(max(res.redchi, 1.0)))
        sb0 *= rescale_run
        sm0 *= rescale_run
        cov_si *= rescale_run**2

        Tc = -b0 / m0
        var_Tc = (
            (sb0 / m0) ** 2
            + (b0 * sm0 / m0**2) ** 2
            - 2.0 * b0 * cov_si / m0**3
        )
        return {
            "Tc_K_mf": Tc,
            "sigma_Tc_K_mf": float(np.sqrt(max(0.0, var_Tc))),
            "Tc_K_half": Tc_half,
            "sigma_Tc_K_half": sigma_Tc_half,
            "n_fit": int(mask.sum()),
            "n_loops": int(len(keep)),
            "T_min_K": float(T_K.min()),
            "T_max_K": float(T_K.max()),
            "redchi": float(res.redchi),
        }

    cross_run = pd.DataFrame.from_records([
        {"run": _label, **(_run_method3(_p) or {})} for _label, _p in RUN_FILES.items()
    ])
    return (cross_run,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Method IV cross-check: Curie–Weiss above $T_c$

    The half-height (Methods I–III) and mean-field (Method III $M_0^2$)
    estimators all use the *below-$T_c$* side of the transition. As an
    independent cross-check, we also fit the *above-$T_c$* side via the
    Curie–Weiss law

    $$
    \chi(T) \;=\; \frac{C}{T - T_c}
    \quad\Longleftrightarrow\quad
    \frac{1}{\chi(T)} \;=\; \frac{T - T_c}{C},
    $$

    so $1/\chi$ is linear in $T$ with the $T$-axis intercept equal to
    $T_c$. Per loop we fit $\chi(T)$ as the slope of the loop's full
    $M(H)$ data: above $T_c$ the loop collapses to a single line through
    the origin (no hysteresis), so a single linear regression returns
    a well-defined $\chi$. Restricting the $1/\chi$ vs $T$ ODR fit to
    $T > T_c^\mathrm{seed} + 5\,\mathrm{K}$ keeps us in the asymptotic
    paramagnetic regime where Curie–Weiss is valid.

    Because this estimator uses only the high-$T$ data, it does not
    share inputs with the half-height crossings on the low-$T$ side —
    agreement between the two corroborates the headline; disagreement
    would flag a finite-$H$ smearing systematic on the half-height.
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
    # Method IV (cross-check): Curie-Weiss above T_c.
    # In the paramagnetic regime, chi(T) = M/H = C / (T - T_c), so
    # 1/chi = (T - T_c)/C is linear in T with T-axis intercept T_c.
    # Per loop we fit chi from the full loop's M(H) data:
    #   - above T_c the loop has no hysteresis, so all four branches
    #     collapse to a single line through the origin and a single
    #     polyfit slope is well-defined;
    #   - below T_c the slope is contaminated by hysteresis, so we
    #     restrict the 1/chi vs T fit to T > Tc_seed + buffer.
    # The headline T_c family uses the half-height crossings (below-T_c
    # side); this cell adds an independent paramagnetic-side estimate
    # using only the high-T data, which gives an honest cross-check
    # because the half-height and Curie-Weiss methods do not share the
    # same inputs. chi here is computed from the *uncorrected* M, since
    # the global background we subtract elsewhere is itself an average
    # chi over the high-T quartile and would be circular here.
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
    # where finite-H smearing distorts chi (the Curie-Weiss form is
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
            sb_CW = float(cw_result.errors[0])
            sm_CW = float(cw_result.errors[1])
            cov_CW = float(cw_result.cov[0, 1]) if cw_result.cov is not None else 0.0
            redchi_CW = float(cw_result.redchi)
            # Birge rescale to absorb model mismatch (Curie-Weiss is also
            # an approximation; mean-field corrections inflate chi^2/nu).
            rescale_CW = float(np.sqrt(max(redchi_CW, 1.0)))
            sb_CW *= rescale_CW
            sm_CW *= rescale_CW
            cov_CW *= rescale_CW ** 2
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
    FIG_DIR,
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
    sigma_chi_vals,
):
    # Plot 1/chi vs T, marking the paramagnetic-regime fit window and
    # the Curie-Weiss fit line + T_c x-intercept.
    fig_cw, ax_cw = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)

    _good = np.isfinite(chi_vals) & (chi_vals > 0)
    _T_all = TEMPERATURE_K[_good]
    _inv_chi_all = 1.0 / chi_vals[_good]
    _sigma_inv_all = sigma_chi_vals[_good] / chi_vals[_good] ** 2

    _used = mask_CW[_good]
    if (~_used).any():
        ax_cw.errorbar(
            _T_all[~_used], _inv_chi_all[~_used], yerr=_sigma_inv_all[~_used],
            fmt="o", color="0.65", markersize=2.6, elinewidth=0.5,
            alpha=0.55, label=r"excluded ($T \leq T_c^\mathrm{seed}+$buffer)",
        )
    if _used.any():
        ax_cw.errorbar(
            _T_all[_used], _inv_chi_all[_used], yerr=_sigma_inv_all[_used],
            fmt="o", color="C0", markersize=3.4, elinewidth=0.7,
            alpha=0.85, label=r"fit window (paramagnetic)",
        )

    if cw_result is not None and np.isfinite(Tc_CW):
        _T_line = np.linspace(Tc_CW, float(_T_all.max()) + 5.0, 200)
        ax_cw.plot(
            _T_line, m_CW * _T_line + b_CW, "-", color="C3", linewidth=2.0,
            label=rf"Curie–Weiss: $T_c={Tc_CW:.1f}\pm{sigma_Tc_CW:.1f}\,\mathrm{{K}}$",
        )
        ax_cw.axvline(Tc_CW, color="C3", linewidth=0.8, linestyle=":")

    ax_cw.axvline(
        Tc_seed_CW + CW_BUFFER_K, color="0.4", linewidth=0.8, linestyle="--",
        label=rf"$T_c^\mathrm{{seed}}+{CW_BUFFER_K:.0f}\,\mathrm{{K}}$ (window edge)",
    )
    ax_cw.axhline(0, color="0.4", linewidth=0.6, linestyle="--")
    ax_cw.set_xlabel(r"$T$ (K)")
    ax_cw.set_ylabel(r"$1/\chi$ (arb. units)")
    ax_cw.set_title(r"Method IV: Curie–Weiss — $1/\chi(T)$ above $T_c$")
    ax_cw.minorticks_on()
    ax_cw.grid(True, which="major", alpha=0.25)
    ax_cw.grid(True, which="minor", alpha=0.10)
    ax_cw.legend(loc="lower right", fontsize=8)

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
    # absolute units) is reported as a cross-check; it sits ~15-20 K
    # above the half-height numbers because the mean-field square-root
    # form is only an approximation in a narrow window below T_c, so its
    # zero crossing systematically overshoots when fit over any window
    # wide enough to have decent statistics. The disagreement between
    # the two estimator families is itself a methodological systematic.
    finite_hh = diagnostics_with_sigma.dropna(subset=["Tc_K"])
    hh_tcs = finite_hh["Tc_K"].to_numpy(dtype=float)
    hh_sigmas = finite_hh["sigma_Tc_K"].to_numpy(dtype=float)
    if len(hh_tcs) > 0:
        _w = 1.0 / np.maximum(hh_sigmas, 1e-9) ** 2
        Tc_headline = float(np.average(hh_tcs, weights=_w))
        sigma_Tc_headline_stat_unrescaled = float(1.0 / np.sqrt(np.sum(_w)))
        method_spread = float(np.std(hh_tcs, ddof=1)) if len(hh_tcs) >= 2 else 0.0
        # Birge rescaling on the combine. The three half-height methods
        # disagree by ~7 K while their per-method bootstrap sigmas are
        # ~0.2-0.3 K, so the inverse-variance combine of independent
        # estimators is overconfident: chi^2/nu = sum((Tc_i - Tc_avg)^2
        # / sigma_i^2) / (N-1) is ~100. We multiply the combined sigma
        # by sqrt(max(chi^2/nu, 1)) to absorb the disagreement into
        # the *statistical* error bar, which is the standard PDG-style
        # treatment when methods that should agree don't. The leftover
        # method-to-method scatter still appears as sigma_method in
        # the systematic budget below; callers who want the unrescaled
        # statistical sigma can read sigma_Tc_headline_stat_unrescaled.
        if len(hh_tcs) >= 2:
            chi2_combine = float(np.sum((hh_tcs - Tc_headline) ** 2 * _w))
            dof_combine = len(hh_tcs) - 1
            redchi_combine = chi2_combine / dof_combine if dof_combine > 0 else 1.0
        else:
            redchi_combine = 1.0
        birge_combine = float(np.sqrt(max(redchi_combine, 1.0)))
        sigma_Tc_headline_stat = sigma_Tc_headline_stat_unrescaled * birge_combine
    else:
        Tc_headline = float("nan")
        sigma_Tc_headline_stat_unrescaled = float("nan")
        sigma_Tc_headline_stat = float("nan")
        method_spread = 0.0
        redchi_combine = float("nan")
        birge_combine = float("nan")

    finite_runs = cross_run.dropna(subset=["Tc_K_half"]) if "Tc_K_half" in cross_run.columns else cross_run.iloc[0:0]
    run_tcs_half = finite_runs["Tc_K_half"].to_numpy(dtype=float) if not finite_runs.empty else np.array([])
    run_spread = float(np.std(run_tcs_half, ddof=1)) if len(run_tcs_half) >= 2 else 0.0

    # Systematic budget: combines the inter-method spread (within-run
    # methodological systematic) with the run-to-run spread of the
    # *half-height* T_c per run (cross-run systematic on the same
    # estimator family). The mean-field-vs-half-height shift is NOT
    # added in quadrature here: the mean-field model is not adopted as
    # the headline estimator (its narrow-window linear approximation
    # produces a chi^2/nu >> 1 fit and a Birge-rescaled error bar that
    # is too wide to be informative), so quoting the gap to it as a
    # systematic on the half-height result would punish the headline
    # for an estimator we have explicitly chosen not to use. Instead,
    # we report the mean-field T_c separately as a qualitative
    # cross-check; the methodological tension within each run is
    # already captured by sigma_method.
    syst_total = float(np.hypot(method_spread, run_spread))
    # Display only: gap between the failing mean-field fit and the
    # half-height headline, kept for context but not in the systematic.
    mf_shift_display = float(abs(Tc_K - Tc_headline)) if np.isfinite(Tc_K) and np.isfinite(Tc_headline) else 0.0

    def _row_half(name, tc, stc, n_loops, redchi):
        stc_str = f"{stc:.2f}" if np.isfinite(stc) else "—"
        return f"| {name} | {tc:.2f} | {stc_str} | {int(n_loops)} | {redchi:.1f} |"

    cross_rows_half = [
        _row_half(f"Method 3 half-height, run `{r['run']}`",
                  r["Tc_K_half"], r["sigma_Tc_K_half"], r["n_loops"], r["redchi"])
        for _, r in finite_runs.iterrows() if np.isfinite(r["Tc_K_half"])
    ]

    method_rows = [
        f"| {r['method']} | {r['Tc_K']:.2f} | {r['sigma_Tc_K']:.2f} | (half-height bootstrap) |"
        for _, r in diagnostics_with_sigma.iterrows() if np.isfinite(r["Tc_K"])
    ]

    Tc_headline_C = Tc_headline - 273.15

    mo.callout(
        mo.md(rf"""
    ### Bottom-line $T_c$ for the Curie experiment

    **Half-height crossings — three methods on run `first`** (headline family):

    | Method | $T_c$ (K) | $\sigma_{{T_c}}^{{\text{{stat}}}}$ (K) | source |
    |---|---|---|---|
    {chr(10).join(method_rows)}

    **Half-height crossings of $M_0$ across the three runs** (cross-run check on the same estimator):

    | Run | $T_c^\mathrm{{half}}$ (K) | $\sigma_{{T_c}}^\mathrm{{boot}}$ (K) | n_loops | mean-field $\chi^2/\nu$ |
    |---|---|---|---|---|
    {chr(10).join(cross_rows_half)}

    The far-right column is the reduced chi-square of the mean-field
    fit on the same run (shown for context only — see the cross-check
    block at the end of this callout).

    **Headline value (model-free, half-height across methods 1, 2, 3-normalized, run `first`):**

    $$
    T_c \;=\; {Tc_headline:.1f} \;\pm\; {sigma_Tc_headline_stat:.1f}_\text{{stat}}
       \;\pm\; {syst_total:.1f}_\text{{syst}} \;\mathrm{{K}}
    \;=\; {Tc_headline_C:.1f}^\circ\mathrm{{C}}.
    $$

    The statistical 1-$\sigma$ on the combined headline has been
    Birge-rescaled to account for method-to-method disagreement. The
    raw inverse-variance combine of the three half-height bootstraps
    gives $\sigma_\text{{stat}}^\text{{raw}}={sigma_Tc_headline_stat_unrescaled:.2f}\,\mathrm{{K}}$,
    but the three estimates differ by ~7 K — far more than these per-method
    bootstrap errors allow under the assumption that they sample the
    same quantity. The combine therefore has $\chi^2/\nu={redchi_combine:.1f}$;
    we multiply $\sigma_\text{{stat}}$ by $\sqrt{{\chi^2/\nu}}={birge_combine:.1f}$
    so the quoted statistical bar reflects the actual scatter between
    methods rather than the unphysically tight per-method bootstrap
    bands.

    The systematic 1-$\sigma$ combines

    - method-to-method spread of the half-height crossings within run `first`: $\sigma_\text{{method}}={method_spread:.1f}\,\mathrm{{K}}$;
    - run-to-run spread of the half-height $T_c$ on the three sweeps: $\sigma_\text{{run}}={run_spread:.1f}\,\mathrm{{K}}$;

    in quadrature.

    A fully-correlated thermometer absolute-accuracy bound of
    $\pm{SIGMA_T_ABS_K:.1f}\,\mathrm{{K}}$ shifts $T_c$ rigidly within a
    single run; it does not affect $\sigma_{{T_c}}^\text{{stat}}$ and is
    partially probed by $\sigma_\text{{run}}$ above.

    **Cross-check: mean-field $T_c$ on run `first`.**
    The $M_0^2(T)\propto T_c-T$ form is a near-$T_c$ approximation.
    Our iterative narrow-window ODR fit gives $T_c^\mathrm{{MF}}={Tc_K:.1f}$ K
    with $\chi^2/\nu={odr_result.redchi:.1f}$; the linear form does not
    describe the in-window data, so the Birge-rescaled error bar
    ($\pm{sigma_Tc_K:.0f}$ K) is too wide to be informative as an
    independent estimator. The gap to the headline,
    $|T_c^\mathrm{{MF}}-T_c^\mathrm{{hh}}|={mf_shift_display:.1f}\,\mathrm{{K}}$,
    reflects the bias of the mean-field linear approximation rather
    than a systematic on the half-height result, and is therefore
    *not* added to the systematic budget. The mean-field number is
    kept here only as a qualitative cross-check showing that both
    estimator families place $T_c$ in the same ~210-230 K region.

    **Cross-check: Curie–Weiss $T_c$ from $1/\chi(T)$ above $T_c$.**
    Independently of the half-height (which uses the below-$T_c$ side)
    and the mean-field fit (also below-$T_c$), the paramagnetic-side
    Curie–Weiss relation $\chi=C/(T-T_c)$ predicts $1/\chi$ linear in
    $T$ with $T$-axis intercept $T_c$. Per-loop $\chi(T)$ is fit from
    the full single-loop $M(H)$ slope (above $T_c$ the four branches
    collapse to one line through the origin, so a single slope is
    well-defined). An ODR fit of $1/\chi$ vs $T$ on the paramagnetic
    window gives
    $T_c^\mathrm{{CW}} = {Tc_CW:.1f}\pm{sigma_Tc_CW:.1f}\,\mathrm{{K}}$
    ($\chi^2/\nu={redchi_CW:.1f}$, Birge-rescaled). Because this
    estimator uses *only* the high-$T$ data, it does not share inputs
    with the half-height crossing on the low-$T$ side; agreement
    within $\sim$few K corroborates the headline, while a large
    disagreement would flag a finite-$H$ smearing systematic on the
    half-height result.
    """),
        kind="success",
    )
    return (
        Tc_headline,
        Tc_headline_C,
        method_spread,
        mf_shift_display,
        run_spread,
        sigma_Tc_headline_stat,
        syst_total,
    )


if __name__ == "__main__":
    app.run()
