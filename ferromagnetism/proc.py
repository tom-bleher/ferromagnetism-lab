import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium", app_title="Ferromagnetism")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ferromagnetism

    ### **Part A — Peak-envelope magnetization curve of the iron core**

    Primary AC current is swept; the scope dual-cursor readings
    $\Delta V_x,\Delta V_y$ span the **full peak-to-peak** of each axis
    (the positive to negative saturation distance on the hysteresis loop),
    so $V_\text{peak} = \Delta V/2$ and

    $$H = \frac{N}{2\,L\,R_x}\,\Delta V_x, \qquad B = \frac{R_y\,C}{2\,N\,A}\,\Delta V_y.$$

    A smooth guide curve gives peak-envelope $B(H)$ and
    $\mu_{\mathrm{r}}(H)=\frac{B}{\mu_0 H}$ curves. The full loop trajectory was not
    exported, so this is an envelope measurement rather than a first-branch
    magnetization curve and cannot give $B_r$, $H_c$, or loop area.

    ### **Part B — Vacuum permeability $\mu_0$**

    Copper plates of total thickness $L'$ are inserted with $B$ held constant.
    Ampère's law gives

    $$\frac{B}{\mu}\,L + \frac{B}{\mu_0}\,L' = N I \;\;\Longrightarrow\;\; \frac{N I}{B} = \frac{1}{\mu_0}\,L' + \frac{L}{\mu_\text{iron}},$$

    so a linear fit of $N I/B$ vs $L'$ yields $\mu_0$ from the inverse slope
    and $\mu_\text{iron}$ from the intercept.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Course-style data processing conventions

    Instrument uncertainties use the manufacturer specs combined with the
    display-resolution uncertainty. Independent uncertainty contributions are
    combined in quadrature, and indirect quantities are propagated by the usual
    partial derivative rule.

    The copper-gap measurement is the only fitted model in this notebook. For
    that fit the notebook reports the fit parameters, relative errors,
    $\chi^2/\nu$, p-value, DOF, and a data-minus-fit plot. The smooth curves in
    Part A are only guide curves through the measured envelope points; they are
    not fitted physical models.
    """)
    return


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from cycler import cycler
    from instruments import (
        caliper,
        digital_multimeter_resistance,
        oscilloscope_dual_cursor,
        ruler,
    )
    from scipy.interpolate import PchipInterpolator
    from taulab import (
        PhysicalSize,
        combine,
        fit_functions,
        nsigma,
        odr_fit,
        read_table,
    )
    from uncertainties import ufloat
    from uncertainties import unumpy as unp

    # ColorBrewer Dark2: muted, high-contrast colours suitable for print.
    BREWER = {
        "teal": "#1b9e77",
        "orange": "#d95f02",
        "purple": "#7570b3",
        "rose": "#e7298a",
        "green": "#66a61e",
        "gold": "#e6ab02",
        "brown": "#a6761d",
        "gray": "#666666",
    }

    plt.rcParams.update(
        {
            "figure.figsize": (8.0, 4.8),
            "figure.dpi": 110,
            "savefig.dpi": 600,
            "font.size": 11,
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "cm",
            "axes.prop_cycle": cycler(
                color=[
                    BREWER["teal"],
                    BREWER["orange"],
                    BREWER["purple"],
                    BREWER["green"],
                    BREWER["rose"],
                    BREWER["gold"],
                    BREWER["brown"],
                    BREWER["gray"],
                ]
            ),
            "axes.titlesize": 12,
            "axes.titlepad": 8,
            "axes.labelsize": 11,
            "axes.labelpad": 5,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "legend.fontsize": 9,
            "legend.framealpha": 0.92,
            "lines.linewidth": 1.4,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )

    _here = Path(__file__).resolve().parent
    DATA_XLSX = _here / "data" / "data.xlsx"
    FIG_DIR = _here.parent / "report" / "media"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # CODATA-2022 μ₀ (post-2019 SI); uncertainty ~10⁻¹⁶ is negligible
    # vs. our ~10⁻⁸ measurement σ, but is carried for nσ reporting.
    MU0_THEO = 1.25663706127e-6
    MU0_THEO_U = ufloat(MU0_THEO, 2.0e-16)

    def fmt(x, unit=""):
        """LaTeX $v \\pm \\sigma\\,\\mathrm{unit}$ for ufloat / PhysicalSize."""
        if hasattr(x, "value"):
            x = ufloat(x.value, x.uncertainty)
        if getattr(x, "s", None) == 0:
            core = f"{x.n:.6g}"
        else:
            core = format(x, ".2uL")
        umod = r"\,\mathrm{" + unit + "}" if unit else ""
        return f"${core}{umod}$"

    return (
        BREWER,
        DATA_XLSX,
        FIG_DIR,
        MU0_THEO,
        MU0_THEO_U,
        PchipInterpolator,
        PhysicalSize,
        caliper,
        combine,
        digital_multimeter_resistance,
        fit_functions,
        fmt,
        np,
        nsigma,
        odr_fit,
        oscilloscope_dual_cursor,
        pd,
        plt,
        read_table,
        ruler,
        ufloat,
        unp,
    )


@app.cell
def _(BREWER, FIG_DIR, pd, plt):
    _loop_path = (
        FIG_DIR.parent.parent
        / "curie"
        / "data"
        / "series A"
        / "HystLoop_m92.341000C.txt"
    )
    _loop = pd.read_csv(_loop_path, sep="\t")
    _x = _loop["X (Volt)"].to_numpy(float)
    _y = _loop["Y (Volt)"].to_numpy(float)

    _xmin, _xmax = float(_x.min()), float(_x.max())
    _ymin, _ymax = float(_y.min()), float(_y.max())
    _dx, _dy = _xmax - _xmin, _ymax - _ymin
    _xpad, _ypad = 0.10 * _dx, 0.13 * _dy
    _xleft, _xright = _xmin - _xpad, _xmax + 0.20 * _dx
    _ybottom, _ytop = _ymin - 0.20 * _dy, _ymax + _ypad

    _fig, _ax = plt.subplots(figsize=(6.35, 3.95), constrained_layout=True)
    _loop_color = BREWER["teal"]
    _cursor_color = BREWER["orange"]
    _axis_color = "0.25"

    _ax.plot(_x, _y, color=_loop_color, linewidth=2.2, solid_capstyle="round")
    _ax.axhline(0.0, color=_axis_color, linewidth=0.85)
    _ax.axvline(0.0, color=_axis_color, linewidth=0.85)
    _ax.vlines(
        [_xmin, _xmax],
        _ybottom + 0.03 * _dy,
        _ytop - 0.03 * _dy,
        color=_cursor_color,
        linestyle=(0, (5, 3)),
        linewidth=1.15,
    )
    _ax.hlines(
        [_ymin, _ymax],
        _xleft + 0.025 * _dx,
        _xright - 0.045 * _dx,
        color=_cursor_color,
        linestyle=(0, (5, 3)),
        linewidth=1.15,
    )
    _ax.scatter(
        [_xmin, _xmax],
        [_ymin, _ymax],
        s=38,
        facecolor="white",
        edgecolor=_loop_color,
        linewidth=1.25,
        zorder=4,
    )

    _arrow_y = _ymin - 0.105 * _dy
    _arrow_x = _xmax + 0.095 * _dx
    _ax.annotate(
        "",
        xy=(_xmax, _arrow_y),
        xytext=(_xmin, _arrow_y),
        arrowprops={"arrowstyle": "<->", "color": _cursor_color, "linewidth": 1.45},
    )
    _ax.annotate(
        "",
        xy=(_arrow_x, _ymax),
        xytext=(_arrow_x, _ymin),
        arrowprops={"arrowstyle": "<->", "color": _cursor_color, "linewidth": 1.45},
    )
    _ax.text(
        0.5 * (_xmin + _xmax),
        _arrow_y - 0.018 * _dy,
        r"$\Delta V_x$",
        ha="center",
        va="top",
        color=_cursor_color,
        fontsize=12.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.0},
    )
    _ax.text(
        _arrow_x + 0.022 * _dx,
        0.5 * (_ymin + _ymax),
        r"$\Delta V_y$",
        ha="left",
        va="center",
        color=_cursor_color,
        fontsize=12.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.0},
    )
    _ax.set_xlabel(r"$V_x$ (V)")
    _ax.set_ylabel(r"$V_y$ (V)", labelpad=12)
    _ax.set_xlim(_xleft, _xright)
    _ax.set_ylim(_ybottom, _ytop)
    _ax.minorticks_on()
    _ax.grid(True, which="major", alpha=0.22)
    _ax.grid(True, which="minor", alpha=0.08)

    _fig.savefig(FIG_DIR / "hysteresis_schematic.pdf", bbox_inches="tight", dpi=600)
    plt.close(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    _budget_md = r"""
    | Quantity | Uncertainty treatment |
    |---|---|
    | $L$ (Ampère loop) | two ruler readings with $\sigma_L=2\sqrt{\sigma_a^2+\sigma_b^2}$ |
    | $L'$ | caliper resolution, evaluated per measurement |
    | $R_x$ (current-sense) | HP 34401A $100\,\Omega$-range instr. bound combined with display resolution: $\sigma_{R_x}=\sqrt{a_{R_x,\mathrm{instr}}^2+\sigma_{R,q}^2}$ |
    | $R_y$ (integrator) | HP 34401A $10\,\mathrm{k\Omega}$-range instr. bound (using the manual's 20% overrange) combined with the logged-value resolution: $\sigma_{R_y}=\sqrt{a_{R_y,\mathrm{instr}}^2+\sigma_{R,q}^2}$ |
    | $C$ (integrator) | given relative uncertainty, $\sigma_C/C=10^{-4}$ |
    | $A$ (core cross-section) | indirect product $A=ab$, with $\sigma_A/A=\sqrt{(\sigma_a/a)^2+(\sigma_b/b)^2}$ |
    | $\Delta V_x,\,\Delta V_y$ (scope, dual-cursor) | instr. bound combined with $0.5\,\mathrm{mV}$ display resolution, evaluated per measured voltage |

    The HP 34401A resistance specs assume 4-wire ohms or 2-wire ohms with
    Math Null; no separate lead-resistance term is added here.

    Numerical summary for single-valued calibration constants:

    | Quantity | Value | Standard uncertainty | Relative uncertainty |
    |---|---:|---:|---:|
    | $L$ | $0.4800\,\mathrm{m}$ | $0.00082\,\mathrm{m}$ | $0.17\%$ |
    | $R_x$ | $2.999\,\Omega$ | $0.0043\,\Omega$ | $0.14\%$ |
    | $R_y$ | $11.10\,\mathrm{k\Omega}$ | $0.029\,\mathrm{k\Omega}$ | $0.26\%$ |
    | $C$ | $20.1\,\mu\mathrm{F}$ | $2.0\,\mathrm{nF}$ | $0.010\%$ |
    | $A$ | $16.0\,\mathrm{cm^2}$ | $0.16\,\mathrm{cm^2}$ | $1.0\%$ |
    """
    mo.vstack([mo.md("## Uncertainties"), mo.center(mo.md(_budget_md))])
    return


@app.cell
def _(
    DATA_XLSX,
    MU0_THEO,
    caliper,
    combine,
    digital_multimeter_resistance,
    fmt,
    mo,
    np,
    oscilloscope_dual_cursor,
    read_table,
    ruler,
    ufloat,
):
    _p = read_table(DATA_XLSX, sheet_name="apparatus").iloc[0]
    N = int(_p["N"])
    L_nom = float(_p["L (m)"])
    Rx_nom = float(_p["Rx (Ω)"])
    Ry_nom = float(_p["Ry (Ω)"])
    C_nom = float(_p["C (F)"])
    A_nom = float(_p["A (m²)"])

    # L = 2(a + b) from two 1-mm-ruler readings ⇒ σ_L = 2√(σ_a²+σ_b²).
    u_L = 2.0 * combine(ruler(), ruler())
    u_Lp = caliper()

    u_Rx = digital_multimeter_resistance(Rx_nom)
    u_Ry = digital_multimeter_resistance(Ry_nom)
    u_C = 0.0001 * C_nom

    # A = a·b with a = b = 4 cm on a 1-mm ruler ⇒ σ_A/A ≈ 1.0 %.
    _u_side_rel = ruler() / 0.04
    u_A = A_nom * combine(_u_side_rel, _u_side_rel)

    def u_V(V, resolution=0.5e-3):
        """Scope ΔV σ: resolution is 0.5 mV, so σ_res = resolution/√12."""
        V = np.asarray(V, dtype=float)
        return oscilloscope_dual_cursor(V, resolution=resolution)

    L = ufloat(L_nom, u_L, "L")
    Rx = ufloat(Rx_nom, u_Rx, "Rx")
    Ry = ufloat(Ry_nom, u_Ry, "Ry")
    C = ufloat(C_nom, u_C, "C")
    A = ufloat(A_nom, u_A, "A")

    # ΔV is peak-to-peak; V_peak = ΔV/2, and the 1/2 is absorbed into
    # the calibration constants so they apply directly to ΔV.
    H_per_Vx = N / (2 * L * Rx)
    B_per_Vy = Ry * C / (2 * N * A)

    f_drive = 50.0
    omega = 2 * np.pi * f_drive
    T_drive = 1.0 / f_drive
    tau_RC = Ry * C
    wRyC = omega * tau_RC
    _wRyC_nom = wRyC.n
    _integrator_gain_ratio = _wRyC_nom / np.sqrt(1 + _wRyC_nom**2)
    _integrator_gain_error_pct = (1 - _integrator_gain_ratio) * 100
    _integrator_phase_deg = np.degrees(np.arctan(1 / _wRyC_nom))

    mo.md(f"""
    **Apparatus constants**

    | | value |
    |---|---|
    | $N$ | ${N}$ |
    | $L$ | {fmt(L, "m")} |
    | $R_x$ | {fmt(Rx, r"\\Omega")} |
    | $R_y$ | {fmt(Ry, r"\\Omega")} |
    | $C$ | {fmt(C, "F")} |
    | $A$ | {fmt(A, "m^{{2}}")} |
    | $\\mu_0^{{\\mathrm{{CODATA}}}}$ | ${MU0_THEO:.4e}\\,\\mathrm{{T\\,m\\,A^{{-1}}}}$ |

    **Integrator check**

    The $R_y$–$C$ circuit is a valid integrator at the 50 Hz drive
    frequency. Its exact transfer function is
    $G_{{RC}}(\\omega)=1/(1+i\\omega R_yC)$, which reduces to the ideal
    integrator $1/(i\\omega R_yC)$ when $\\omega R_yC\\gg1$:

    $$\\tau = R_y C = ({tau_RC * 1e3:.2uL})\\,\\mathrm{{ms}} \\;\\gg\\; T = 1/f = {T_drive * 1e3:.1f}\\,\\mathrm{{ms}}, \\qquad \\omega\\tau = {wRyC:.2uL}.$$

    Quantitatively,
    $\\left|G_{{RC}}/(1/i\\omega R_yC)\\right|={_integrator_gain_ratio:.5f}$,
    so the non-ideal amplitude correction is only {_integrator_gain_error_pct:.3f}%
    and the phase departure from an ideal integrator is {_integrator_phase_deg:.2f}°.
    """)
    return B_per_Vy, H_per_Vx, L, N, Rx, u_Lp, u_V


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part A — Peak-envelope magnetization curve $B(H)$ and $\mu_{\mathrm{r}}(H)$

    **Scope of Part A.** The data captured here is a 13-row table of
    $(\Delta V_x, \Delta V_y)$ peak-to-peak readings, one per primary
    drive level. Each row characterizes a single Lissajous loop by its
    *peak-to-peak extent* (top-bottom and left-right excursion on the
    XY display); the loop trajectory itself was not exported. From
    these readings we can extract $B_{\text{peak}}$, $H_{\text{peak}}$,
    $\mu_{\mathrm{r}}(H)$, and the high-field $B$-value, but **not** the
    remanence $B_r=B(H=0)$, the coercivity $H_c=H$ at $B=0$, or the
    hysteresis-loop area (energy loss per cycle). Those would require
    sampling the full loop (XY-mode waveform export, as the Curie
    experiment does). This is a known scope-of-data limitation.
    """)
    return


@app.cell
def _(B_per_Vy, DATA_XLSX, H_per_Vx, MU0_THEO, read_table, u_V, unp):
    # Workbook sheet name is historical; these points are peak-to-peak envelope readings.
    pt1 = read_table(DATA_XLSX, sheet_name="virgin-curve", usecols=[0, 1])
    pt1.columns = ["dVx", "dVy"]

    Vx_u = unp.uarray(pt1["dVx"].to_numpy(), u_V(pt1["dVx"].to_numpy()))
    Vy_u = unp.uarray(pt1["dVy"].to_numpy(), u_V(pt1["dVy"].to_numpy()))

    H_u = H_per_Vx * Vx_u
    B_u = B_per_Vy * Vy_u
    mu_rel_u = B_u / H_u / MU0_THEO

    pt1["H"] = unp.nominal_values(H_u)
    pt1["sH"] = unp.std_devs(H_u)
    pt1["B"] = unp.nominal_values(B_u)
    pt1["sB"] = unp.std_devs(B_u)
    pt1["mu_rel"] = unp.nominal_values(mu_rel_u)
    pt1["s_mu_rel"] = unp.std_devs(mu_rel_u)
    return (pt1,)


@app.cell
def _(BREWER, FIG_DIR, PchipInterpolator, np, plt, pt1):
    # B(H) anchored at the origin for plotting the peak-envelope curve;
    # μ_r(H) not anchored — μ_r(0) is a finite initial permeability we
    # do not measure directly.
    H_nodes = np.concatenate([[0.0], pt1["H"].to_numpy()])
    B_nodes = np.concatenate([[0.0], pt1["B"].to_numpy()])
    H_sorted, _ui = np.unique(H_nodes, return_index=True)
    interp_B = PchipInterpolator(H_sorted, B_nodes[_ui])
    H_grid = np.linspace(0.0, pt1["H"].max() * 1.01, 500)

    H_mu = pt1["H"].to_numpy()
    mu_vals = pt1["mu_rel"].to_numpy()
    Hmu_sorted, _uj = np.unique(H_mu, return_index=True)
    interp_mu = PchipInterpolator(Hmu_sorted, mu_vals[_uj])
    Hmu_grid = np.linspace(H_mu.min(), H_mu.max(), 500)
    fig, axB = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    axMu = axB.twinx()

    C_B, C_MU = BREWER["teal"], BREWER["orange"]
    FS_LABEL, FS_TICK, FS_LEGEND = 15, 13, 11

    axB.plot(
        H_grid,
        interp_B(H_grid),
        "-",
        color=C_B,
        alpha=0.85,
        linewidth=1.8,
        label="_nolegend_",
    )
    axB.errorbar(
        pt1["H"],
        pt1["B"],
        xerr=pt1["sH"],
        yerr=pt1["sB"],
        fmt="o",
        color=C_B,
        mfc="white",
        markersize=6,
        ecolor=C_B,
        elinewidth=1.0,
        capsize=3.0,
        label=r"$B(H)$",
    )
    axB.set_xlabel(r"$H$ (A$\cdot$m$^{-1}$)", fontsize=FS_LABEL)
    axB.set_ylabel(r"$B$ (T)", color=C_B, fontsize=FS_LABEL)
    axB.tick_params(axis="x", labelsize=FS_TICK)
    axB.tick_params(axis="y", colors=C_B, labelsize=FS_TICK)
    axB.spines["left"].set_color(C_B)
    axB.minorticks_on()
    axB.grid(True, which="major", alpha=0.25)
    axB.grid(True, which="minor", alpha=0.10)

    axMu.plot(
        Hmu_grid,
        interp_mu(Hmu_grid),
        "-",
        color=C_MU,
        alpha=0.85,
        linewidth=1.8,
        label="_nolegend_",
    )
    axMu.errorbar(
        pt1["H"],
        pt1["mu_rel"],
        xerr=pt1["sH"],
        yerr=pt1["s_mu_rel"],
        fmt="s",
        color=C_MU,
        mfc="white",
        markersize=6,
        ecolor=C_MU,
        elinewidth=1.0,
        capsize=3.0,
        label=r"$\mu_{\mathrm{r}}(H)$",
    )
    axMu.set_ylabel(r"$\mu_{\mathrm{r}}=\mu/\mu_0$", color=C_MU, fontsize=FS_LABEL)
    axMu.tick_params(axis="y", colors=C_MU, labelsize=FS_TICK)
    axMu.spines["right"].set_color(C_MU)
    axMu.minorticks_on()
    axMu.grid(False)

    _handles_B, _labels_B = axB.get_legend_handles_labels()
    _handles_mu, _labels_mu = axMu.get_legend_handles_labels()
    axB.legend(
        _handles_B + _handles_mu,
        _labels_B + _labels_mu,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        framealpha=0.92,
        fontsize=FS_LEGEND,
        borderpad=0.55,
        labelspacing=0.45,
        handlelength=2.4,
    )

    _x_hi = float((pt1["H"] + pt1["sH"]).max()) * 1.02
    axB.set_xlim(0.0, _x_hi)
    _B_hi = max(
        float((pt1["B"] + pt1["sB"]).max()),
        float(np.nanmax(interp_B(H_grid))),
    )
    axB.set_ylim(0.0, _B_hi * 1.08)
    _mu_lo = float((pt1["mu_rel"] - pt1["s_mu_rel"]).min())
    _mu_hi = float((pt1["mu_rel"] + pt1["s_mu_rel"]).max())
    _mu_pad = 0.04 * (_mu_hi - _mu_lo)
    axMu.set_ylim(max(0.0, _mu_lo - _mu_pad), _mu_hi + _mu_pad)

    fig.savefig(FIG_DIR / "fig_BH_mu.pdf", bbox_inches="tight", dpi=600)
    fig  # type: ignore
    return


@app.cell
def _(fmt, mo, pt1, ufloat):
    _i = pt1["mu_rel"].idxmax()
    mu_peak = ufloat(pt1.loc[_i, "mu_rel"], pt1.loc[_i, "s_mu_rel"])
    H_peak = ufloat(pt1.loc[_i, "H"], pt1.loc[_i, "sH"])
    B_peak = ufloat(pt1.loc[_i, "B"], pt1.loc[_i, "sB"])
    B_max = ufloat(pt1["B"].iloc[-1], pt1["sB"].iloc[-1])
    H_max = ufloat(pt1["H"].iloc[-1], pt1["sH"].iloc[-1])

    mo.callout(
        mo.md(f"""
    **Part A Results**

    * $\\mu_{{\\mathrm{{max}}}}/\\mu_0 =$ {fmt(mu_peak)} at $H \\approx$ {fmt(H_peak, r"A\\,m^{{-1}}")}; $B(H_{{\\mathrm{{peak}}}}) =$ {fmt(B_peak, "T")}.
    * $B(H_{{\\mathrm{{max}}}}) =$ {fmt(B_max, "T")} at $H =$ {fmt(H_max, r"A\\,m^{{-1}}")} (saturation regime).
    """),
        kind="success",
    )
    return B_max, mu_peak


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part B — Permeability of vacuum

    Linear model:

    $$\frac{N I}{B} = \frac{1}{\mu_0}\,L' + \frac{L}{\mu_\text{iron}}.$$

    For each sheet $V_y$ (and therefore $B$) is held constant while $L'$ is
    varied.
    """)
    return


@app.cell
def _(
    BREWER,
    B_per_Vy,
    DATA_XLSX,
    H_per_Vx,
    L,
    MU0_THEO,
    MU0_THEO_U,
    N,
    PhysicalSize,
    Rx,
    fit_functions,
    fmt,
    mo,
    np,
    nsigma,
    odr_fit,
    pd,
    read_table,
    u_Lp,
    u_V,
    unp,
):
    def fit_gap(sheet):
        r"""Weighted fit of :math:`NI/B` vs :math:`L'`."""
        raw = read_table(DATA_XLSX, sheet_name=sheet)
        # Keep only rows whose plate count parses as a number; a blank row
        # then a text summary block follows the data.
        _plates = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
        df = raw.loc[_plates.notna()].iloc[:, [0, 1, 3, 4]].copy()
        df.columns = ["plates", "Lp_mm", "dVx", "dVy"]
        df = df.reset_index(drop=True)

        Lp = df["Lp_mm"].to_numpy() * 1e-3
        Vx, Vy = df["dVx"].to_numpy(), df["dVy"].to_numpy()
        sVx, sVy = u_V(Vx), u_V(Vy)

        Vx_u = unp.uarray(Vx, sVx)
        Vy_u = unp.uarray(Vy, sVy)
        B_u = B_per_Vy * Vy_u
        NI_B = N * Vx_u / (2 * Rx) / B_u
        y_vals = unp.nominal_values(NI_B)
        sy_vals = unp.std_devs(NI_B)

        x, sx = Lp, np.full_like(Lp, u_Lp)
        result = odr_fit(fit_functions.linear, None, x, sx, y_vals, sy_vals)

        slope = PhysicalSize.from_ufloat(result.param(1).to_ufloat())
        intercept = PhysicalSize.from_ufloat(result.param(0).to_ufloat())

        return dict(
            result=result,
            slope=slope,
            intercept=intercept,
            mu0_exp=1.0 / slope,
            mu_iron=L / intercept,
            y_fit=y_vals,
            sy_fit=sy_vals,
            x=x,
            sx=sx,
            B_mean=np.mean(unp.nominal_values(B_u)),
            H_max=H_per_Vx.nominal_value * Vx.max(),
        )

    RUNS = {
        "vacuum-permeability-a": dict(
            label="series a", marker="o", color=BREWER["teal"]
        ),
        "vacuum-permeability-b": dict(
            label="series b", marker="s", color=BREWER["orange"]
        ),
    }
    fits = {sheet: fit_gap(sheet) for sheet in RUNS}

    def _rel_pct(x):
        return rf"${100 * abs(x.uncertainty / x.value):.2g}\%$"

    _rows = [
        (r"$\langle B\rangle$", lambda f: rf"${f['B_mean']:.2f}\,\mathrm{{T}}$"),
        (
            r"$H_{\mathrm{max}}$",
            lambda f: rf"${f['H_max']:.0f}\,\mathrm{{A\cdot m^{{-1}}}}$",
        ),
        (r"slope $= 1/\mu_0$", lambda f: fmt(f["slope"], r"A\,T^{-1}\,m^{-1}")),
        (r"slope relative error", lambda f: _rel_pct(f["slope"])),
        (
            r"intercept $= L/\mu_{\mathrm{iron}}$",
            lambda f: fmt(f["intercept"], r"A\,T^{-1}"),
        ),
        (r"intercept relative error", lambda f: _rel_pct(f["intercept"])),
        (r"$\mu_0^{\mathrm{exp}}$", lambda f: fmt(f["mu0_exp"], r"T\cdot m\cdot A^{-1}")),
        (r"$\mu_0$ relative error", lambda f: _rel_pct(f["mu0_exp"])),
        (
            r"$n_\sigma$ vs CODATA",
            lambda f: rf"${nsigma(f['mu0_exp'], MU0_THEO_U):.1f}$",
        ),
        (r"$\mu_{\mathrm{iron}}/\mu_0$", lambda f: fmt(f["mu_iron"] / MU0_THEO)),
        (
            r"$\chi^2/\nu$",
            lambda f: (
                rf"${f['result'].redchi:.2f}\;({f['result'].chi2:.2f}/{f['result'].dof})$"
            ),
        ),
        (r"p-value", lambda f: rf"${f['result'].p_value:.3f}$"),
        (r"DOF", lambda f: rf"${f['result'].dof}$"),
    ]
    _run_cols = " | ".join(RUNS[s]["label"] for s in fits)
    _header = f"| Quantity | {_run_cols} |"
    _sep = "|---" * (1 + len(fits)) + "|"
    _body = [
        f"| {label} | " + " | ".join(fn(f) for f in fits.values()) + " |"
        for label, fn in _rows
    ]
    mo.center(mo.md("\n".join([_header, _sep, *_body])))
    return RUNS, fits


@app.cell
def _(FIG_DIR, RUNS, fits, np, plt):
    fig2, (ax_fit, ax_res) = plt.subplots(
        2,
        1,
        figsize=(8.4, 6.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.4, 1.0]},
    )
    _x_edges = []
    _y_edges = []
    _res_edges = []

    def _draw(sheet, f):
        style = RUNS[sheet]
        Lp_mm, sx_mm = f["x"] * 1e3, f["sx"] * 1e3
        slope_NIB = f["slope"].value
        intercept_NIB = f["intercept"].value
        yhat_NIB = slope_NIB * f["x"] + intercept_NIB
        residual_NIB = f["y_fit"] - yhat_NIB
        _x_edges.append(Lp_mm - sx_mm)
        _x_edges.append(Lp_mm + sx_mm)
        _y_edges.append(f["y_fit"] - f["sy_fit"])
        _y_edges.append(f["y_fit"] + f["sy_fit"])
        _res_edges.append(np.abs(residual_NIB) + f["sy_fit"])

        ax_fit.errorbar(
            Lp_mm,
            f["y_fit"],
            xerr=sx_mm,
            yerr=f["sy_fit"],
            fmt=style["marker"],
            color=style["color"],
            mfc="white",
            markersize=5,
            ecolor=style["color"],
            elinewidth=0.9,
            capsize=2.5,
            label=style["label"],
        )
        xs_m = np.linspace(
            max(0.0, float(np.min(f["x"] - f["sx"]))),
            float(np.max(f["x"] + f["sx"])),
            80,
        )
        _y_edges.append(slope_NIB * xs_m + intercept_NIB)
        ax_fit.plot(
            xs_m * 1e3,
            slope_NIB * xs_m + intercept_NIB,
            "-",
            color=style["color"],
            alpha=0.85,
        )

        ax_res.errorbar(
            Lp_mm,
            residual_NIB,
            xerr=sx_mm,
            yerr=f["sy_fit"],
            fmt=style["marker"],
            color=style["color"],
            mfc="white",
            markersize=5,
            ecolor=style["color"],
            elinewidth=0.9,
            capsize=2.5,
        )

    for _sheet, _f in fits.items():
        _draw(_sheet, _f)

    ax_fit.set_ylabel(r"$NI/B$ (A$\cdot$T$^{-1}$)")
    ax_fit.legend(loc="upper left")
    ax_fit.minorticks_on()
    ax_fit.grid(True, which="major", alpha=0.25)
    ax_fit.grid(True, which="minor", alpha=0.10)

    if _x_edges:
        _x_all = np.concatenate(_x_edges)
        _x_lo = float(np.nanmin(_x_all))
        _x_hi = float(np.nanmax(_x_all))
        _x_pad = max(0.01, 0.04 * (_x_hi - _x_lo))
        ax_fit.set_xlim(_x_lo - _x_pad, _x_hi + _x_pad)
    if _y_edges:
        _y_all = np.concatenate(_y_edges)
        _y_lo = float(np.nanmin(_y_all))
        _y_hi = float(np.nanmax(_y_all))
        _y_pad = max(8.0, 0.05 * (_y_hi - _y_lo))
        ax_fit.set_ylim(_y_lo - _y_pad, _y_hi + _y_pad)

    ax_res.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    if _res_edges:
        _res_lim = float(np.nanmax(np.concatenate(_res_edges)))
        if np.isfinite(_res_lim) and _res_lim > 0:
            ax_res.set_ylim(-1.12 * _res_lim, 1.12 * _res_lim)
    ax_res.set_xlabel(r"$L'$ (mm)")
    ax_res.set_ylabel(r"$NI/B - f(L')$" + "\n" + r"(A$\cdot$T$^{-1}$)")
    ax_res.minorticks_on()
    ax_res.grid(True, which="major", alpha=0.25)
    ax_res.grid(True, which="minor", alpha=0.10)

    fig2.savefig(FIG_DIR / "fig_copper_gap.pdf", bbox_inches="tight", dpi=600)
    fig2  # type: ignore
    return


@app.cell(hide_code=True)
def _(MU0_THEO, RUNS, fits, mo):
    # μ_0 scale check. The direct fit uses NI/B =
    # (N²A/(R_x R_y C))·(ΔV_x/ΔV_y), so the log-derivatives are exact:
    #
    #   d ln μ_0 / d ln N   = -2     d ln μ_0 / d ln R_x = +1
    #   d ln μ_0 / d ln A   = -1     d ln μ_0 / d ln R_y = +1
    #   d ln μ_0 / d ln C   = +1     d ln μ_0 / d ln L'  = +1
    #
    # (L' enters via the fit's x-axis: slope ∝ 1/L' in dimensional terms,
    # so any global rescaling of L' propagates with sign +1 into μ_0.)
    #
    # If μ_0^exp / μ_0^CODATA = ratio < 1, the fractional shift the model
    # has to explain in *each* quantity (acting alone) to bring the
    # measurement back to CODATA is:
    #
    #     ε_i = (1/ratio - 1) / s_i,
    #
    # where s_i is the log-derivative above. Smaller |ε| means a smaller
    # systematic in that quantity is sufficient to wipe out the bias —
    # i.e. that quantity is the most likely culprit.
    sens = {
        "N": -2.0,
        "A": -1.0,
        "R_x": +1.0,
        "R_y": +1.0,
        "C": +1.0,
        "L'": +1.0,
    }

    rows = [
        "| run | quantity | $d\\ln\\mu_0/d\\ln X$ | shift in $X$ to recover CODATA |",
        "|---|---|---:|---:|",
    ]
    for sheet, f in fits.items():
        ratio = f["mu0_exp"].value / MU0_THEO
        delta = (1.0 / ratio) - 1.0  # fractional change in μ_0 needed
        label = RUNS[sheet]["label"]
        for q, s in sens.items():
            eps = delta / s
            rows.append(f"| {label} | ${q}$ | {s:+.0f} | {eps * 100:+.1f}% |")

    table_md = "\n".join(rows)
    mo.md(rf"""
    ### $\mu_0$ scale check — which apparatus quantity could explain the bias?

    Each row shows the fractional change in one apparatus quantity that
    would, *acting alone*, restore $\mu_0^\mathrm{{exp}}$ to its CODATA
    value. Quantities listed with the smallest required shift are the
    most likely systematic culprits given the calibration uncertainty
    budget on the previous table.

    {table_md}

    Reading the table against the apparatus uncertainty budget:

    - $N=250$ is a count; no uncertainty. A 21 % miscount is implausible.
    - $A$: budgeted at 1.0 %. To explain a 30–35 % bias on its own,
      the cross-section would need to be 23 cm² instead of 16 cm² —
      far outside the budget.
    - $R_x$ and $C$: budgeted below 0.2 %. $R_y$ is budgeted at about
      0.26 %. Off-by-30 % is not credible for any of these in isolation.
      (The film capacitor $C$ is the loosest by typical-tolerance arguments,
      but its measured value is closer to its nominal than the bias requires.)
    - $L'$: budgeted at $\sim$15 µm absolute, which is 3–14 % relative
      depending on the plate count. Two effects compound: (i) the
      caliper resolution itself is appreciable on a 0.10 mm plate, and (ii)
      copper plates may stick or include burrs that bias the *effective*
      gap larger than nominal. **A coherent +30 % bias on $L'$ would
      explain the entire $\mu_0$ shortfall** — and is the only entry
      whose required shift is comparable to its plausible systematic
      range.

    Conclusion: the dominant systematic is most plausibly the effective
    copper-plate gap $L'$ (geometric thickness vs effective magnetic
    gap including mounting/contact effects), with the core cross-section
    $A$ as a secondary candidate. $R_x$, $R_y$, $C$, and $N$ remain
    individually too tightly constrained to be primary suspects.
    """)
    return


@app.cell
def _(B_max, MU0_THEO_U, RUNS, fits, fmt, mo, mu_peak, nsigma):
    _per_run = "\n".join(
        f"* *{RUNS[sheet]['label']}* — "
        f"{fmt(f['mu0_exp'], r'T\cdot m\cdot A^{-1}')} "
        f"(${nsigma(f['mu0_exp'], MU0_THEO_U):.1f}\\,\\sigma$ below CODATA)."
        for sheet, f in fits.items()
    )

    mo.callout(
        mo.md(f"""
    ### Bottomline

    **Part A** — $\\mu_{{\\mathrm{{max}}}}/\\mu_0 =$ {fmt(mu_peak)}; saturation $B =$ {fmt(B_max, "T")}
    (consistent with soft-iron saturation $\\sim 1.2$–$2.1\\,\\mathrm{{T}}$).

    **Part B** — $\\mu_0^{{\\mathrm{{exp}}}}$ from the runs:

    {_per_run}
    """),
        kind="info",
    )
    return


if __name__ == "__main__":
    app.run()
